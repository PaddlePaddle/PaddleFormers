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

import random
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
import paddlefleet_ops
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils

from paddleformers.fleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.training.global_vars import unset_global_variables
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.moe.moe_layer import MoELayer
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

if paddlefleet_ops.is_sonic_moe_available():
    from paddlefleet_ops.sonicmoe.functional import clear_all_fp8_weight_caches


class SonicMoETopk(paddle.autograd.PyLayer):
    """PyLayer wrapping SonicMoE's high-performance topk operator.

    Supports:
      - softmax_fusion: fuse softmax into the topk kernel (input is
        raw logits; output scores are softmax-normalized).
      - n_group > 1: group-limited greedy topk selection.
    """

    @staticmethod
    def forward(
        ctx,
        scores,
        k,
        n_group,
        topk_group,
        softmax_fusion,
    ):
        """
        Args:
            scores: [T, E] tensor. If softmax_fusion=True, these are raw
                logits (pre-softmax). Otherwise, already-scored gates.
            k: number of experts per token.
            n_group: number of expert groups. 1 means no grouping.
            topk_group: number of groups to select when n_group > 1.
            softmax_fusion: whether to fuse softmax into the topk kernel.

        Returns:
            top_gate: [T, k] float32 scores of selected experts.
            top_idx: [T, k] int64 indices of selected experts.
        """
        from paddlefleet_ops.sonicmoe.functional.forward import (
            _topk_fwd,
        )

        T, E = scores.shape

        if n_group > 1:
            # Group-limited greedy: select top groups first, then topk
            # within those groups.
            group_scores = scores.reshape([T, n_group, -1]).max(axis=-1)
            group_idx = paddle.topk(
                group_scores, k=topk_group, axis=-1, sorted=True
            )[1]
            group_mask = paddle.zeros_like(group_scores).put_along_axis(
                group_idx,
                paddle.to_tensor(1.0, dtype="float32"),
                axis=-1,
            )
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand([T, n_group, E // n_group])
                .reshape([T, -1])
            )
            # Apply mask: zero out non-selected groups.
            # Use a very negative value for softmax_fusion (logits)
            # so softmax gives ~0, or just zero for non-fusion.
            if softmax_fusion:
                masked_scores = scores + (1.0 - score_mask) * (-1e9)
            else:
                masked_scores = scores * score_mask
        else:
            masked_scores = scores

        topk_scores = paddle.empty([T, k], dtype=paddle.float32)
        topk_indices = paddle.empty([T, k], dtype=paddle.int32)
        # print("==== sonicmoe topk fwd ====")
        _topk_fwd(
            masked_scores,
            k,
            topk_scores,
            topk_indices,
            require_softmax_fusion=softmax_fusion,
        )

        ctx.save_for_backward(topk_scores, topk_indices)
        ctx.E = E
        ctx.K = k
        ctx.softmax_fusion = softmax_fusion
        ctx.input_dtype = scores.dtype

        return topk_scores, topk_indices.cast(paddle.int64)

    @staticmethod
    def backward(ctx, dtopk_score, _):
        from paddlefleet_ops.sonicmoe.functional.backward import (
            _softmax_topk_bwd,
        )

        # print("==== sonicmoe topk bwd ====")
        # assert 0
        topk_scores, topk_indices = ctx.saved_tensor()
        T = dtopk_score.shape[0]
        K = ctx.K
        if ctx.softmax_fusion:
            dlogits = paddle.zeros([T, ctx.E], dtype=ctx.input_dtype)
            _softmax_topk_bwd(
                dlogits, None, dtopk_score, topk_scores, topk_indices, K
            )
            return dlogits
        else:
            # No softmax fusion: gradient is simply scattered back
            dscores = paddle.zeros([T, ctx.E], dtype=dtopk_score.dtype)
            dscores = dscores.put_along_axis_(
                topk_indices.cast(paddle.int64),
                dtopk_score,
                axis=1,
            )
            return dscores


@unittest.skipUnless(
    paddlefleet_ops.is_sonic_moe_available(),
    "Sonic-MoE not available (requires Python>=3.12, CUDA>=12.9, SM>=90)",
)
class TestSonicMoEExpertParallelPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 8,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 8,
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
        initialize_fleet(strategy=strategy)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    @classmethod
    def tearDownClass(cls):
        unset_global_variables()

    def setUp(self):
        self.seed = 123
        self.hidden_size = 2048
        self.n_routed_experts = 64

        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        paddle.manual_seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        self.pg_collection = self.__class__.pg_collection

    @staticmethod
    def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        if denominator.item() == 0:
            return 0.0
        sim = 2 * (x * y).sum() / denominator
        return (1 - sim).item()

    @staticmethod
    def _small_init_method(tensor):
        """Small uniform init for precision tests (matches pre-update behavior)."""
        paddle.nn.initializer.Uniform(-0.001, 0.001)(tensor)

    def _build_transformer_config(
        self,
        using_sonic_moe=False,
        fp8=None,
        moe_deep_gemm=False,
        fp8_wgrad=True,
        expert_model_parallel_size=4,
    ):
        return TransformerConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            n_routed_experts=self.n_routed_experts,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=expert_model_parallel_size,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=1024,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            moe_expert_fusion=True,
            moe_deep_gemm=moe_deep_gemm,
            bias_activation_fusion=True,
            moe_token_dispatcher_type="deepep",
            moe_use_fusion_node=True,
            using_sonic_moe=using_sonic_moe,
            fp8=fp8,
            fp8_wgrad=fp8_wgrad,
            init_method=self._small_init_method,
            output_layer_init_method=self._small_init_method,
        )

    def _build_moe_layer(
        self,
        using_sonic_moe=False,
        fp8=None,
        moe_deep_gemm=False,
        fp8_wgrad=True,
        expert_model_parallel_size=4,
        pg_collection=None,
    ):
        transformer_config = self._build_transformer_config(
            using_sonic_moe=using_sonic_moe,
            fp8=fp8,
            moe_deep_gemm=moe_deep_gemm,
            fp8_wgrad=fp8_wgrad,
            expert_model_parallel_size=expert_model_parallel_size,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            transformer_config,
            num_experts=self.n_routed_experts,
        )
        moe_layer = MoELayer(
            transformer_config,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            pg_collection or self.pg_collection,
        )
        mix_precision_utils.MixPrecisionLayer(moe_layer, dtype="bfloat16")
        for param in moe_layer.parameters():
            if hasattr(param, "main_grad") and param.main_grad is None:
                param.main_grad = paddle.zeros_like(param, dtype=paddle.float32)
        return moe_layer

    @staticmethod
    def _expert_slice_for_rank(tensor, ep_rank, ep_size):
        chunk_size = tensor.shape[0] // ep_size
        return tensor[ep_rank * chunk_size : (ep_rank + 1) * chunk_size]

    @classmethod
    def _copy_single_card_weights_to_ep(
        cls, src_layer, dst_layer, ep_rank, ep_size
    ):
        src_params = dict(src_layer.named_parameters())
        for name, dst_param in dst_layer.named_parameters():
            src_param = src_params[name]
            if "grouped_gemm_experts.weight" in name:
                src_param = cls._expert_slice_for_rank(
                    src_param, ep_rank, ep_size
                )
            dst_param.set_value(src_param.clone())

    @classmethod
    def _gather_ep_expert_grads(cls, grads, ep_group, ep_size):
        gathered_grads = {}
        for name, grad in grads.items():
            if "grouped_gemm_experts.weight" not in name:
                gathered_grads[name] = grad
                continue
            parts = []
            dist.all_gather(parts, grad, group=ep_group)
            gathered_grads[name] = paddle.concat(parts, axis=0) / ep_size
        return gathered_grads

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

    def _run_forward_backward(self, moe_layer, input_data):
        moe_layer = paddle.amp.decorate(
            models=moe_layer,
            level="O2",
            dtype="bfloat16",
            master_grad=True,
            master_weight=True,
        )
        # mix_precision_utils.MixPrecisionLayer(moe_layer, dtype="bfloat16")
        hidden_states = input_data.detach().clone()
        hidden_states.stop_gradient = False
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            output = moe_layer(hidden_states)[0]
            loss = output.sum()
            # loss = paddle.mean(paddle.square(output.cast("float32")))
        loss.backward()
        self._flush_sonic_expert_layout(moe_layer)
        return (
            output.detach().clone(),
            loss.item(),
            hidden_states.grad.detach().clone(),
            self._collect_grads(moe_layer),
        )

    def _run_accumulated_forward_backward(self, moe_layer, input_data_list):
        self._clear_grads(moe_layer)
        outputs = []
        for input_data in input_data_list:
            hidden_states = input_data.detach().clone()
            hidden_states.stop_gradient = False
            with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
                output = moe_layer(hidden_states)[0]
                loss = paddle.mean(paddle.square(output.cast("float32")))
            loss.backward()
            outputs.append(output.detach().clone())
        self._flush_sonic_expert_layout(moe_layer)
        return outputs[-1], self._collect_grads(moe_layer)

    def _assert_loss_close(self, lhs, rhs, tol, title):
        loss_rdiff = abs(lhs - rhs) / max(abs(rhs), 1e-12)
        print(f"{title}: loss relative diff = {loss_rdiff:.6e}")
        self.assertLess(
            loss_rdiff,
            tol,
            f"{title} loss deviates too much: lhs={lhs}, rhs={rhs}",
        )

    def _assert_tensor_diff_less(self, lhs, rhs, tol, title):
        diff = self.calc_diff(lhs, rhs)
        print(f"{title}: diff = {diff:.6e}")
        self.assertLess(diff, tol, f"{title} diff too large: diff={diff:.6e}")

    def _assert_grad_diff_less(
        self,
        lhs_grads,
        rhs_grads,
        tol,
        title,
    ):
        lhs_names = set(lhs_grads)
        rhs_names = set(rhs_grads)
        self.assertEqual(
            lhs_names,
            rhs_names,
            (
                f"Gradient tensors mismatch for {title}: "
                f"lhs_only={sorted(lhs_names - rhs_names)}, "
                f"rhs_only={sorted(rhs_names - lhs_names)}"
            ),
        )
        self.assertTrue(lhs_names, f"No grad tensors found for {title}")
        for name in sorted(lhs_names):
            grad_tol = tol[name] if isinstance(tol, dict) else tol
            self._assert_tensor_diff_less(
                lhs_grads[name],
                rhs_grads[name],
                tol=grad_tol,
                title=f"{title} grad {name}",
            )

    def run_test_sonic_moe_ep_grad_accumulation(self):
        acc_steps = 1

        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_base = self._build_moe_layer(using_sonic_moe=False)
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_sonic_bf16 = self._build_moe_layer(using_sonic_moe=True)
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_sonic_fp8 = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
        )
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_sonic_fp8_dispatch = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
        )
        moe_layer_sonic_fp8_dispatch.fp8_dispatch = True

        input_data_list = []
        for step_idx in range(acc_steps):
            paddle.seed(self.seed + step_idx)
            input_data_list.append(
                paddle.randn(
                    [4, 256, self.hidden_size],
                    dtype=paddle.bfloat16,
                )
            )

        output_base, grads_base = self._run_accumulated_forward_backward(
            moe_layer_base, input_data_list
        )
        output_bf16, grads_bf16 = self._run_accumulated_forward_backward(
            moe_layer_sonic_bf16, input_data_list
        )
        moe_layer_sonic_fp8.grouped_gemm_experts.quant_weight()
        output_fp8, grads_fp8 = self._run_accumulated_forward_backward(
            moe_layer_sonic_fp8, input_data_list
        )
        moe_layer_sonic_fp8_dispatch.grouped_gemm_experts.quant_weight()
        output_fp8_dispatch, grads_fp8_dispatch = (
            self._run_accumulated_forward_backward(
                moe_layer_sonic_fp8_dispatch, input_data_list
            )
        )
        clear_all_fp8_weight_caches()

        self._assert_tensor_diff_less(
            output_bf16,
            output_base,
            tol=1e-2,
            title="Sonic-MoE BF16 vs Baseline final output",
        )
        self._assert_grad_diff_less(
            grads_bf16,
            grads_base,
            tol=1e-2,
            title="Sonic-MoE BF16 vs Baseline accumulated grad",
        )

        fp8_tol = 5e-3
        self._assert_tensor_diff_less(
            output_fp8,
            output_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8 vs BF16 final output",
        )
        self._assert_grad_diff_less(
            grads_fp8,
            grads_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8 vs BF16 accumulated grad",
        )

        self._assert_tensor_diff_less(
            output_fp8_dispatch,
            output_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8-dispatch vs BF16 final output",
        )
        self._assert_grad_diff_less(
            grads_fp8_dispatch,
            grads_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8-dispatch vs BF16 accumulated grad",
        )
        self._assert_tensor_diff_less(
            output_fp8_dispatch,
            output_fp8,
            tol=1e-8,
            title="Sonic-MoE FP8-dispatch vs FP8 final output",
        )
        self._assert_grad_diff_less(
            grads_fp8_dispatch,
            grads_fp8,
            tol=1e-8,
            title="Sonic-MoE FP8-dispatch vs FP8 accumulated grad",
        )

        print("Final output and parameter gradient precision checks passed!")

    def run_test_ep_precision(self):
        ep_size = self.pg_collection.ep.nranks
        ep_rank = dist.get_rank(self.pg_collection.ep)
        single_rank_group = dist.new_group([dist.get_rank()])
        single_pg_collection = ProcessGroupCollection(
            ep=single_rank_group,
            expt_dp=single_rank_group,
        )

        moe_layer_single = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
            expert_model_parallel_size=1,
            pg_collection=single_pg_collection,
        )
        moe_layer_ep = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
            expert_model_parallel_size=ep_size,
            pg_collection=self.pg_collection,
        )
        self._copy_single_card_weights_to_ep(
            moe_layer_single,
            moe_layer_ep,
            ep_rank,
            ep_size,
        )

        paddle.seed(self.seed + 2024)
        input_data = paddle.randn(
            [4, 256, self.hidden_size],
            dtype=paddle.bfloat16,
        )

        moe_layer_single.grouped_gemm_experts.quant_weight()
        output_single, loss_single, input_grad_single, grads_single = (
            self._run_forward_backward(moe_layer_single, input_data)
        )
        moe_layer_ep.grouped_gemm_experts.quant_weight()
        output_ep, loss_ep, input_grad_ep, grads_ep = (
            self._run_forward_backward(moe_layer_ep, input_data)
        )
        grads_ep = self._gather_ep_expert_grads(
            grads_ep,
            self.pg_collection.ep,
            ep_size,
        )
        clear_all_fp8_weight_caches()

        self._assert_loss_close(
            loss_ep,
            loss_single,
            tol=2e-2,
            title="Sonic-MoE FP8 EP vs single-card",
        )
        self._assert_tensor_diff_less(
            output_ep,
            output_single,
            tol=5e-6,
            title="Sonic-MoE FP8 EP vs single-card output",
        )
        self._assert_tensor_diff_less(
            input_grad_ep,
            input_grad_single,
            tol=5e-6,
            title="Sonic-MoE FP8 EP vs single-card input grad",
        )
        self._assert_grad_diff_less(
            grads_ep,
            grads_single,
            tol=5e-6,
            title="Sonic-MoE FP8 EP vs single-card param grad",
        )

        print("Expert-parallel FP8 precision checks passed!")

    def run_test_bf16_wgrad(self):
        acc_steps = 1

        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_sonic_bf16 = self._build_moe_layer(using_sonic_moe=True)
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_sonic_fp8 = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
            fp8_wgrad=False,
        )

        input_data_list = []
        for step_idx in range(acc_steps):
            paddle.seed(self.seed + step_idx)
            input_data_list.append(
                paddle.randn(
                    [4, 256, self.hidden_size],
                    dtype=paddle.bfloat16,
                )
            )

        output_bf16, grads_bf16 = self._run_accumulated_forward_backward(
            moe_layer_sonic_bf16, input_data_list
        )
        output_fp8, grads_fp8 = self._run_accumulated_forward_backward(
            moe_layer_sonic_fp8, input_data_list
        )

        fp8_tol = 5e-3
        self._assert_tensor_diff_less(
            output_fp8,
            output_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8 vs BF16 final output",
        )
        self._assert_grad_diff_less(
            grads_fp8,
            grads_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8 vs BF16 accumulated grad",
        )

    def test_sonic_moe_all(self):
        self.run_test_sonic_moe_ep_grad_accumulation()
        self.run_test_ep_precision()
        self.run_test_bf16_wgrad()


if __name__ == "__main__":
    unittest.main()
