# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import subprocess
import unittest

import paddle
import paddle.nn.functional as F
import paddlefleet_ops
from paddle.distributed import fleet
from paddlefleet_ops.utils import get_cuda_version

# from tests.unit_tests.test_utilities import Utils
import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.transformer.moe.moe_layer import MoELayer
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

if paddlefleet_ops.is_sonic_moe_available():
    from paddlefleet_ops.sonicmoe.functional import (
        clear_all_fp8_weight_caches,
    )


def get_gpu_models_via_nvidia_smi():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
        )
        models = output.decode().strip().replace("NVIDIA", "")
        return models
    except Exception as e:
        return ["Unknown"]


def judge_machine_type():
    if not paddle.is_compiled_with_cuda():
        return "No CUDA GPU"
    models = get_gpu_models_via_nvidia_smi()
    for model in models:
        name = model.upper()
        if "V" in name:
            return "V"
        elif "H" in name:
            return "H"


result = judge_machine_type()
print("你的机器类型是：", result)
version, cuda_minor = get_cuda_version()
print("CUDA version:", version)


def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


# ── Module-level fleet initialization (only once) ─────────────────────
_strategy = fleet.DistributedStrategy()
_strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": 1,
    "pp_degree": 1,
    "sharding_degree": 1,
    "sep_degree": 1,
    "cp_degree": 1,
    "ep_degree": 1,
    "moe_sharding_degree": 1,
    "order": [
        "sharding",
        "moe_sharding",
        "pp",
        "sep",
        "cp",
        "dp",
        "ep",
        "mp",
    ],
}
fleet.init(is_collective=True, strategy=_strategy)
_hcg = fleet.get_hybrid_communicate_group()
ps.initialize_model_parallel(_hcg)


@unittest.skipUnless(
    paddlefleet_ops.is_sonic_moe_available(),
    "Sonic-MoE not available (requires Python>=3.12, CUDA>=12.9, SM>=90)",
)
class TestSonicMoELayerPrecision(unittest.TestCase):
    """Precision comparison at the MoELayer level:
    baseline grouped_gemm vs BF16 sonic-moe vs FP8 sonic-moe.
    """

    def setUp(self):
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        self.seed = 46
        self.hidden_size = 2048
        self.n_routed_experts = 8
        self.acc_steps = 1

    @staticmethod
    def _small_init_method(tensor):
        """Small uniform init for precision tests (matches pre-update behavior)."""
        paddle.nn.initializer.Uniform(-0.001, 0.001)(tensor)

    def _build_transformer_config(
        self, using_sonic_moe=False, fp8=None, moe_deep_gemm=None
    ):
        kwargs = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": 4,
            "n_routed_experts": self.n_routed_experts,
            "use_cpu_initialization": False,
            "num_experts_per_tok": 2,
            "tensor_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "sequence_parallel": False,
            "bf16": True,
            "params_dtype": paddle.bfloat16,
            "moe_intermediate_size": 1024,
            "gated_linear_unit": True,
            "n_shared_experts": 0,
            "hidden_act": F.silu,
            "moe_expert_fusion": True,
            "bias_activation_fusion": True,
            "moe_token_dispatcher_type": "alltoall",
            "moe_use_fusion_node": True,
            "using_sonic_moe": using_sonic_moe,
            "fp8": fp8,
            "fp8_wgrad": True,
            "init_method": self._small_init_method,
            "output_layer_init_method": self._small_init_method,
        }
        if moe_deep_gemm is not None:
            kwargs["moe_deep_gemm"] = moe_deep_gemm
        return TransformerConfig(**kwargs)

    def _build_moe_layer(
        self, using_sonic_moe=False, fp8=None, moe_deep_gemm=None
    ):
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        transformer_config = self._build_transformer_config(
            using_sonic_moe=using_sonic_moe,
            fp8=fp8,
            moe_deep_gemm=moe_deep_gemm,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            transformer_config,
            num_experts=self.n_routed_experts,
        )

        moe_layer = MoELayer(
            transformer_config,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            self.pg_collection,
        )

        return moe_layer

    @staticmethod
    def _collect_grads(layer):
        grads = {}
        for name, param in layer.named_parameters():
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is not None:
                grads[name] = grad.detach().clone()
        return grads

    @staticmethod
    def _clear_grads(layer):
        for _, param in layer.named_parameters():
            if hasattr(param, "main_grad") and param.main_grad is not None:
                param.main_grad.zero_()
            if param.grad is not None:
                param.grad.zero_()

    def _flush_sonic_expert_layout(self, moe_layer):
        expert = getattr(moe_layer, "grouped_gemm_experts", None)
        if not hasattr(expert, "flush_to_grouped_layout"):
            return

        weight_ptrs = (expert.weight1.data_ptr(), expert.weight2.data_ptr())
        grad_ptrs = []
        for param in (expert.weight1, expert.weight2):
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            grad_ptrs.append(None if grad is None else grad.data_ptr())

        expert.flush_to_grouped_layout()

        self.assertEqual(expert.weight1.data_ptr(), weight_ptrs[0])
        self.assertEqual(expert.weight2.data_ptr(), weight_ptrs[1])
        for param, grad_ptr in zip((expert.weight1, expert.weight2), grad_ptrs):
            if grad_ptr is None:
                continue
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            self.assertEqual(grad.data_ptr(), grad_ptr)

    def _run_accumulated_forward_backward(self, moe_layer, input_data_list):
        self._clear_grads(moe_layer)
        losses = []
        output = None
        for input_data in input_data_list:
            hidden_states = input_data.detach().clone()
            hidden_states.stop_gradient = False
            with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
                output = moe_layer(hidden_states)[0]
                loss = output.sum()
            loss.backward()
            losses.append(loss.item())
        self._flush_sonic_expert_layout(moe_layer)
        return (
            losses[-1],
            output.detach().clone(),
            self._collect_grads(moe_layer),
        )

    def test_moe_layer_precision(self):
        """Test MoELayer precision: BF16 sonic-moe vs baseline, FP8 vs BF16.

        Both baseline and sonic layers are built with the same seed.
        SonicMoEExpert reuses GroupedMLPExpert initialization, so expert
        weights are initialized in the same layout and values as baseline.

        Uses BMMFunction (moe_deep_gemm=False) for the baseline to avoid
        a known bug in DeepGEMMBMMFunction that corrupts expert outputs
        beyond the first expert in batched mode.

        Checks:
          1. BF16 sonic-moe vs baseline: forward output and gradients
             should match within tight tolerance (1e-4).
          2. FP8 sonic-moe vs BF16 sonic-moe: output and gradients
             should be close (5e-3).
        """
        moe_layer_baseline = self._build_moe_layer(
            using_sonic_moe=False, moe_deep_gemm=False
        )
        moe_layer_sonic_bf16 = self._build_moe_layer(using_sonic_moe=True)
        moe_layer_sonic_fp8 = self._build_moe_layer(
            using_sonic_moe=True, fp8="e4m3"
        )

        input_data_list = []
        for step_idx in range(self.acc_steps):
            paddle.seed(self.seed + step_idx)
            data = paddle.randn(
                [2, 64, self.hidden_size], dtype=paddle.bfloat16
            )
            input_data_list.append(data)

        loss_bl, output_bl, grads_bl = self._run_accumulated_forward_backward(
            moe_layer_baseline, input_data_list
        )
        loss_bf16, output_bf16, grads_bf16 = (
            self._run_accumulated_forward_backward(
                moe_layer_sonic_bf16, input_data_list
            )
        )
        loss_fp8, output_fp8, grads_fp8 = (
            self._run_accumulated_forward_backward(
                moe_layer_sonic_fp8, input_data_list
            )
        )
        clear_all_fp8_weight_caches()

        # ── BF16 sonic-moe vs baseline ──
        bf16_tol = 5e-5
        output_diff = calc_diff(output_bf16, output_bl)
        print(f"BF16 sonic vs Baseline: output diff = {output_diff:.6e}")
        self.assertLess(
            output_diff,
            bf16_tol,
            f"BF16 sonic vs Baseline output diff too large: {output_diff:.6e}",
        )

        adiff = abs(loss_bf16 - loss_bl)
        rdiff = adiff / max(abs(loss_bl), 1e-12)
        print(
            f"BF16 sonic vs Baseline: loss rdiff = {rdiff:.6e}, "
            f"adiff = {adiff:.6e}"
        )
        self.assertTrue(
            adiff < 1e-4 or rdiff < 1e-5,
            f"BF16 sonic vs Baseline loss diff too large "
            f"(bl={loss_bl}, bf16={loss_bf16})",
        )

        # Gradient comparison. Sonic expert params keep GroupedMLP layout.
        gate_key = "gate.weight"
        if gate_key in grads_bl and gate_key in grads_bf16:
            diff = calc_diff(grads_bl[gate_key], grads_bf16[gate_key])
            print(
                f"BF16 sonic vs Baseline: grad diff = {diff:.6e} for {gate_key}"
            )
            self.assertLess(diff, bf16_tol)

        w1_key = "grouped_gemm_experts.weight1"
        if w1_key in grads_bl and w1_key in grads_bf16:
            diff = calc_diff(grads_bl[w1_key], grads_bf16[w1_key])
            print(
                f"BF16 sonic vs Baseline: grad diff = {diff:.6e} for {w1_key}"
            )
            self.assertLess(diff, bf16_tol)

        w2_key = "grouped_gemm_experts.weight2"
        if w2_key in grads_bl and w2_key in grads_bf16:
            diff = calc_diff(grads_bl[w2_key], grads_bf16[w2_key])
            print(
                f"BF16 sonic vs Baseline: grad diff = {diff:.6e} for {w2_key}"
            )
            self.assertLess(diff, bf16_tol)

        # ── FP8 sonic-moe vs BF16 sonic-moe ──
        fp8_tol = 5e-3
        adiff = abs(loss_fp8 - loss_bf16)
        rdiff = adiff / max(abs(loss_bf16), 1e-12)
        print(f"FP8 vs BF16: loss rdiff = {rdiff:.6e}, adiff = {adiff:.6e}")
        self.assertTrue(
            adiff < 1e-4 or rdiff < 1e-3,
            f"FP8 sonic-moe loss deviates too much from BF16 "
            f"(bf16={loss_bf16}, fp8={loss_fp8})",
        )

        output_diff = calc_diff(output_fp8, output_bf16)
        print(f"FP8 vs BF16: output diff = {output_diff:.6e}")
        self.assertLess(output_diff, fp8_tol, "FP8 output diff too large")

        common_fp8_grads = set(grads_bf16) & set(grads_fp8)
        self.assertTrue(
            common_fp8_grads,
            "No common FP8 grad tensors found",
        )
        for name in sorted(common_fp8_grads):
            g1 = grads_bf16[name]
            g2 = grads_fp8[name]
            diff = calc_diff(g1, g2)
            print(f"FP8 vs BF16: grad diff = {diff:.6e} for {name}")
            self.assertLess(
                diff,
                fp8_tol,
                f"FP8 grad diff too large for {name}: {diff:.6e}",
            )

        print("All precision checks passed!")


if __name__ == "__main__":
    unittest.main()
