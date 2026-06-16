# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import importlib.util
import os
import sys
import types
import unittest

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import paddle

from paddleformers.fleet.transformer.moe import fp8_utils
from paddleformers.fleet.transformer.moe.fp8_utils import (
    ExpertsGroupGemmContiguousNode,
    tilewise_quant,
)


class TensorBox:
    def __init__(self, tensor, with_main_grad=False):
        self.tensor = tensor
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.grad = None
        self.stop_gradient = False
        self.hook_called = False
        if with_main_grad:
            self.main_grad = None

    def cast(self, dtype):
        return self.tensor.cast(dtype)

    def _apply_backward_hook(self):
        self.hook_called = True

    def _slice(self, start, end):
        return TensorBox(self.tensor[start:end], hasattr(self, "main_grad"))


class Projection:
    def __init__(self, shape, with_main_grad=False):
        self.weight = TensorBox(
            paddle.ones(shape, dtype="float32"), with_main_grad
        )


class Expert:
    def __init__(self, with_main_grad=False):
        self.up_gate_proj = Projection([2, 4], with_main_grad)
        self.down_proj = Projection([4, 2], with_main_grad)


class GroupedWeights:
    def __init__(self, with_main_grad=False):
        self.weight1 = TensorBox(
            paddle.ones([2, 2, 4], dtype="float32"), with_main_grad
        )
        self.weight2 = TensorBox(
            paddle.ones([2, 4, 2], dtype="float32"), with_main_grad
        )


class LoRAGroupedWeights:
    def __init__(self):
        self.weight1 = paddle.ones([1, 2, 4], dtype="float32")
        self.weight2 = paddle.ones([1, 4, 2], dtype="float32")
        self.weight1_lora_A = TensorBox(
            paddle.ones([1, 2, 2], dtype="float32"), True
        )
        self.weight1_lora_B = TensorBox(
            paddle.ones([1, 2, 4], dtype="float32"), False
        )
        self.weight2_lora_A = TensorBox(
            paddle.ones([1, 4, 2], dtype="float32"), True
        )
        self.weight2_lora_B = TensorBox(
            paddle.ones([1, 2, 2], dtype="float32"), False
        )
        self.disable_lora = False
        self.merged = False
        self.scaling = 0.5

    def get_delta_weight(self, lora_a, lora_b):
        return paddle.bmm(lora_a.tensor, lora_b.tensor) * self.scaling


class CustomMap:
    def __init__(self, with_main_grad=False):
        self.experts = [Expert(with_main_grad), None, Expert(with_main_grad)]
        self.grouped_gemm_experts = GroupedWeights(with_main_grad)


class BackwardImplRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, out_grad, unzipped_probs, a2a_async_fn=None):
        del a2a_async_fn
        self.calls.append((out_grad.shape, unzipped_probs.shape))
        return out_grad, paddle.ones([out_grad.shape[0], 1], dtype="float32")


class TaskRecorder:
    def __init__(self):
        self.wait_called = False

    def wait(self):
        self.wait_called = True


class MethodCalls:
    def __init__(self):
        self.names = []

    def bwd_down_input_fp8(
        self, expert_w2, out_grad, o1, unzipped_probs, inplace_swiglu_prob=False
    ):
        del expert_w2, unzipped_probs, inplace_swiglu_prob
        self.names.append("bwd_down_input_fp8")
        do1 = paddle.ones(o1.shape, dtype="float32")
        o2_s = paddle.ones([out_grad.shape[0], 2], dtype="float32")
        probs_grad = paddle.ones([out_grad.shape[0], 1], dtype="float32")
        return do1, o2_s, probs_grad

    def bwd_gate_up_input_fp8(self, do1, expert_w1, dx=None):
        del do1, expert_w1
        self.names.append("bwd_gate_up_input_fp8")
        return dx

    def bwd_gate_up_weight(self, do1, input_x, expert_w1, clear_input=False):
        del do1, input_x, expert_w1, clear_input
        self.names.append("bwd_gate_up_weight")

    def bwd_down_weight(self, out_grad, o2_s, expert_w2):
        del out_grad, o2_s, expert_w2
        self.names.append("bwd_down_weight")

    def bf16_weight_grad(self, dy, x, weights, p2p_overlap=False):
        del dy, x, weights, p2p_overlap
        self.names.append("bf16_weight_grad")

    def reset_state(self):
        self.names.append("reset_state")


class FunctionalWithBatchedGemm:
    def __init__(self, original):
        self.original = original
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.original, name)

    def batched_gemm(
        self, x, dy, tokens_per_expert, trans_lhs=False, trans_rhs=False
    ):
        del x, dy, tokens_per_expert, trans_lhs, trans_rhs
        self.calls.append("batched_gemm")
        return paddle.ones([1, 2, 3], dtype="float32")


class FakeDeepGemm:
    def __init__(self):
        self.calls = []

    def m_grouped_bf16_gemm_nn_contiguous(self, *args, **kwargs):
        self.calls.append(("m_grouped_bf16_gemm_nn_contiguous", args, kwargs))

    def m_grouped_bf16_gemm_nt_contiguous(self, *args, **kwargs):
        self.calls.append(("m_grouped_bf16_gemm_nt_contiguous", args, kwargs))

    def m_grouped_fp8_gemm_nt_contiguous(self, *args, **kwargs):
        self.calls.append(("m_grouped_fp8_gemm_nt_contiguous", args, kwargs))

    def fp8_gemm_nt(self, *args, **kwargs):
        self.calls.append(("fp8_gemm_nt", args, kwargs))


class FakeClearTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.cleared = False

    def __array__(self, dtype=None):
        array = self.tensor.numpy()
        if dtype is not None:
            return array.astype(dtype)
        return array

    def _clear_to_zero_allocation(self):
        self.cleared = True


class TestFP8ImportAndUtilityBranchesNoMock(unittest.TestCase):
    def test_import_branches_and_swiglu_fallback(self):
        path = fp8_utils.__file__
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        old_fqo = sys.modules.get("FusedQuantOps")
        had_fqo = "FusedQuantOps" in sys.modules
        functional = paddle.nn.functional
        had_swiglu = hasattr(functional, "swiglu")
        old_swiglu = getattr(functional, "swiglu", None)
        incubate_functional = paddle.incubate.nn.functional
        had_fused_transpose = hasattr(
            incubate_functional, "fused_transpose_wlch_split_quant"
        )
        old_fused_transpose = getattr(
            incubate_functional, "fused_transpose_wlch_split_quant", None
        )
        fake_fqo = types.ModuleType("FusedQuantOps")
        # importlib.util.find_spec requires __spec__ to be set on the module,
        # otherwise it raises ValueError when the module is already in sys.modules.
        fake_fqo.__spec__ = importlib.util.spec_from_loader(
            "FusedQuantOps", loader=None
        )

        def fake_bwd(o1, do2, probs, flag):
            return o1, probs, do2, flag

        fake_fqo.fused_swiglu_probs_bwd = fake_bwd
        try:
            sys.modules["FusedQuantOps"] = fake_fqo
            if had_swiglu:
                delattr(functional, "swiglu")
            if had_fused_transpose:
                delattr(incubate_functional, "fused_transpose_wlch_split_quant")
            namespace = {
                "__name__": "fp8_utils_import_branch_exec",
                "__file__": path,
                "__package__": "paddleformers.fleet.transformer.moe",
            }
            exec(compile(source, path, "exec"), namespace)
            self.assertTrue(namespace["USE_INPLACE_SWIGLU_BWD"])
            # _fused_swiglu_probs_bwd is set when FusedQuantOps is available.
            self.assertIn("_fused_swiglu_probs_bwd", namespace)
            self.assertIsNone(namespace["fused_transpose_wlch_split_quant"])
            x = paddle.arange(8, dtype="float32").reshape([2, 4])
            self.assertEqual(namespace["swiglu"](x).shape, [2, 2])
            self.assertEqual(
                namespace["swiglu"](
                    paddle.ones([2, 2], dtype="float32"),
                    paddle.full([2, 2], 2.0, dtype="float32"),
                ).shape,
                [2, 2],
            )
        finally:
            if had_fqo:
                sys.modules["FusedQuantOps"] = old_fqo
            else:
                sys.modules.pop("FusedQuantOps", None)
            if had_swiglu:
                functional.swiglu = old_swiglu
            if had_fused_transpose:
                incubate_functional.fused_transpose_wlch_split_quant = (
                    old_fused_transpose
                )

    def test_fwd_swiglu_and_lora_weight_grad_accumulation(self):
        node = ExpertsGroupGemmContiguousNode(CustomMap())
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        y = node.fwd_swiglu(x)
        self.assertEqual(y.shape, [2, 2])
        dw = paddle.ones([1, 2, 3], dtype="float32")
        lora_a = TensorBox(paddle.ones([1, 2, 2], dtype="float32"), True)
        lora_b = TensorBox(paddle.ones([1, 2, 3], dtype="float32"), False)

        node._lora_weight_grad(dw, lora_a, lora_b, 0.5)
        node._lora_weight_grad(dw, lora_a, lora_b, 0.5)

        self.assertIsNotNone(lora_a.main_grad)
        self.assertIsNotNone(lora_b.grad)
        self.assertEqual(lora_a.main_grad.shape, [1, 2, 2])
        self.assertEqual(lora_b.grad.shape, [1, 2, 3])

    def test_empty_backward_initializes_main_grads(self):
        split_node = ExpertsGroupGemmContiguousNode(
            CustomMap(with_main_grad=True), use_fp8_mlp=False
        )
        split_node.tokens_per_expert = [0, 0]
        dx, probs_grad = split_node.backward(
            paddle.empty([0, 2], dtype="float32"),
            paddle.empty([0], dtype="float32"),
        )
        self.assertEqual(dx.shape, [0, 2])
        self.assertEqual(probs_grad.shape, [0, 1])
        self.assertIsNotNone(split_node.experts[0].down_proj.weight.main_grad)
        self.assertIsNotNone(
            split_node.experts[0].up_gate_proj.weight.main_grad
        )

        grouped_node = ExpertsGroupGemmContiguousNode(
            CustomMap(with_main_grad=True),
            use_fp8_mlp=False,
            moe_expert_fusion=True,
        )
        grouped_node.tokens_per_expert = [0, 0]
        grouped_dx, grouped_probs_grad = grouped_node.backward(
            paddle.empty([0, 2], dtype="float32"),
            paddle.empty([0], dtype="float32"),
        )
        self.assertEqual(grouped_dx.shape, [0, 2])
        self.assertEqual(grouped_probs_grad.shape, [0, 1])
        self.assertIsNotNone(
            grouped_node.grouped_gemm_experts.weight1.main_grad
        )
        self.assertIsNotNone(
            grouped_node.grouped_gemm_experts.weight2.main_grad
        )

    def test_backward_subbatch_slices_and_restores_state(self):
        node = ExpertsGroupGemmContiguousNode(
            CustomMap(),
            expert_id=0,
            moe_subbatch_token_num_after_dispatch=128,
            use_fp8_mlp=False,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )
        original_input = paddle.ones([256, 2], dtype="float32")
        original_input_fp8 = paddle.ones([256, 2], dtype="float32")
        original_input_scale = paddle.ones([256, 1], dtype="float32")
        original_o1 = paddle.ones([256, 4], dtype="float32")
        node.input = original_input
        node.input_fp8 = original_input_fp8
        node.input_scale = original_input_scale
        node.o1 = original_o1
        node.tokens_per_expert = [256]
        recorder = BackwardImplRecorder()
        node.backward_impl = recorder

        out_grad = paddle.ones([256, 2], dtype="float32")
        probs = paddle.ones([256], dtype="float32")
        dx, probs_grad = node.backward(out_grad, probs)

        self.assertIs(dx, out_grad)
        self.assertEqual(probs_grad.shape, [256, 1])
        self.assertEqual(len(recorder.calls), 2)
        self.assertIs(node.input, original_input)
        self.assertIs(node.input_fp8, original_input_fp8)
        self.assertEqual(node.tokens_per_expert, [256])

    def test_backward_impl_fp8_a2a_and_non_a2a_branches(self):
        old_flag = fp8_utils.USE_INPLACE_SWIGLU_BWD
        try:
            fp8_utils.USE_INPLACE_SWIGLU_BWD = False
            node = ExpertsGroupGemmContiguousNode(
                CustomMap(with_main_grad=True), use_fp8_mlp=True
            )
            node.o1 = paddle.ones([2, 4], dtype="float32")
            node.input = paddle.ones([2, 2], dtype="float32")
            node.input_fp8 = paddle.ones([2, 2], dtype="float32")
            node.input_scale = paddle.ones([2, 1], dtype="float32")
            node.tokens_per_expert = [2]
            node.use_bf16_gemm_weight_grad = True
            node.dw_p2p_overlap = False
            calls = MethodCalls()
            node.bwd_down_input_fp8 = calls.bwd_down_input_fp8
            node.bwd_gate_up_input_fp8 = calls.bwd_gate_up_input_fp8
            node.bwd_gate_up_weight = calls.bwd_gate_up_weight
            node.bwd_down_weight = calls.bwd_down_weight
            node.bf16_weight_grad = calls.bf16_weight_grad
            node.reset_state = calls.reset_state
            task = TaskRecorder()

            def a2a(x):
                return x, task

            out_grad = paddle.ones([2, 2], dtype="float32")
            probs = paddle.ones([2, 1], dtype="float32")
            dx, probs_grad = node.backward_impl_fp8(
                out_grad, probs, a2a_async_fn=a2a
            )
            self.assertIs(dx, out_grad)
            self.assertEqual(probs_grad.shape, [2, 1])
            self.assertTrue(task.wait_called)
            self.assertIn("bf16_weight_grad", calls.names)
            self.assertIn("reset_state", calls.names)

            fp8_utils.USE_INPLACE_SWIGLU_BWD = True
            node = ExpertsGroupGemmContiguousNode(
                CustomMap(with_main_grad=True), use_fp8_mlp=True
            )
            node.o1 = paddle.ones([2, 4], dtype="float32")
            node.input = paddle.ones([2, 2], dtype="float32")
            node.input_fp8 = paddle.ones([2, 2], dtype="float32")
            node.input_scale = paddle.ones([2, 1], dtype="float32")
            node.tokens_per_expert = [2]
            node.use_bf16_gemm_weight_grad = False
            node.dw_p2p_overlap = False
            calls = MethodCalls()
            node.bwd_down_input_fp8 = calls.bwd_down_input_fp8
            node.bwd_gate_up_input_fp8 = calls.bwd_gate_up_input_fp8
            node.bwd_gate_up_weight = calls.bwd_gate_up_weight
            node.bwd_down_weight = calls.bwd_down_weight
            node.bf16_weight_grad = calls.bf16_weight_grad
            node.reset_state = calls.reset_state
            dx, probs_grad = node.backward_impl_fp8(out_grad, probs)
            self.assertIs(dx, out_grad)
            self.assertEqual(probs_grad.shape, [2, 1])
            self.assertIn("bwd_gate_up_weight", calls.names)
            self.assertIn("bwd_down_weight", calls.names)
        finally:
            fp8_utils.USE_INPLACE_SWIGLU_BWD = old_flag

    def test_grouped_bf16_weight_grad_no_main_grad_batched_gemm_hook(self):
        old_functional = paddle.incubate.nn.functional
        wrapper = FunctionalWithBatchedGemm(old_functional)
        try:
            paddle.incubate.nn.functional = wrapper
            node = ExpertsGroupGemmContiguousNode(
                CustomMap(),
                use_fp8_mlp=False,
                moe_expert_fusion=True,
                moe_deep_gemm=False,
            )
            node.tokens_per_expert = [2]
            weight = TensorBox(paddle.ones([1, 2, 3], dtype="float32"), False)
            x = paddle.ones([2, 2], dtype="float32")
            dy = paddle.ones([2, 3], dtype="float32")
            node.bf16_weight_grad(dy, x, weight)
            self.assertEqual(wrapper.calls, ["batched_gemm"])
            self.assertIsNotNone(weight.grad)
            self.assertTrue(weight.hook_called)
        finally:
            paddle.incubate.nn.functional = old_functional

    def test_blackwell_tilewise_quant_sets_pow2_flag(self):
        old_cuda = paddle.device.cuda

        class FakeCuda:
            @staticmethod
            def get_device_capability():
                return (10, 0)

        try:
            paddle.device.cuda = FakeCuda
            x_fp8, x_scale = tilewise_quant(
                paddle.empty([0, 128], dtype="float32")
            )
        finally:
            paddle.device.cuda = old_cuda

        self.assertEqual(x_fp8.shape, [0, 128])
        self.assertEqual(x_scale.shape, [0, 1])

    def test_direct_fp8_bf16_branch_methods_with_stubs(self):
        old_fused_stack = fp8_utils.fused_stack_quant
        old_split_group = fp8_utils.split_group_gemm
        old_deep = getattr(fp8_utils, "deep_gemm", None)
        old_quant = paddle.incubate.nn.functional.fp8_quant_blockwise
        old_swiglu_quant = getattr(
            fp8_utils, "fuse_weighted_swiglu_fp8_quant", None
        )
        old_swiglu_bwd_flag = fp8_utils.USE_INPLACE_SWIGLU_BWD
        old_fused_weighted_bwd = getattr(
            fp8_utils, "fused_swiglu_weighted_bwd", None
        )
        old_fused_weighted_clamp_bwd = getattr(
            fp8_utils, "fused_swiglu_weighted_clamp_bwd", None
        )
        old_fused_backward = getattr(
            fp8_utils, "fused_swiglu_scale_backward", None
        )
        old_fused_forward = fp8_utils.fused_swiglu_scale_forward
        old_fused_swiglu_probs_bwd = getattr(
            fp8_utils, "_fused_swiglu_probs_bwd", None
        )
        fake_deep = FakeDeepGemm()
        split_calls = []
        try:
            fp8_utils.deep_gemm = fake_deep
            fp8_utils.fused_stack_quant = (
                lambda weights,
                transpose=False,
                num_expert=None,
                use_ue8m0=False: (
                    paddle.ones([num_expert or 1, 2, 4], dtype="float32"),
                    paddle.ones([num_expert or 1, 2, 1], dtype="float32"),
                )
            )

            def fake_split(*args, **kwargs):
                split_calls.append((args, kwargs))

            fp8_utils.split_group_gemm = fake_split
            paddle.incubate.nn.functional.fp8_quant_blockwise = (
                lambda x, **kwargs: (
                    x,
                    paddle.ones([x.shape[0], 1], dtype="float32"),
                )
            )
            fp8_utils.fuse_weighted_swiglu_fp8_quant = (
                lambda o1, probs, **kwargs: (
                    o1[:, :2],
                    paddle.ones([o1.shape[0], 1], dtype="float32"),
                )
            )
            fp8_utils.fused_swiglu_scale_forward = (
                lambda o1, probs: paddle.ones(
                    [o1.shape[0], max(1, o1.shape[1] // 2)], dtype=o1.dtype
                )
            )
            fp8_utils.fused_swiglu_scale_backward = lambda o1, probs, do2: (
                paddle.ones(o1.shape, dtype=o1.dtype),
                paddle.ones([o1.shape[0], 1], dtype="float32"),
            )
            fp8_utils.USE_INPLACE_SWIGLU_BWD = True
            # _fused_swiglu_probs_bwd(o1, do2, probs, flag) -> (do1, probs_grad, o2_s)
            fp8_utils._fused_swiglu_probs_bwd = lambda o1, do2, probs, flag: (
                paddle.ones(o1.shape, dtype=o1.dtype),
                paddle.ones([o1.shape[0], 1], dtype="float32"),
                paddle.ones(
                    [o1.shape[0], max(1, o1.shape[1] // 2)], dtype=o1.dtype
                ),
            )
            fp8_utils.fused_swiglu_weighted_clamp_bwd = (
                lambda o1, probs, do2, cv: (
                    paddle.ones(o1.shape, dtype=o1.dtype),
                    paddle.ones([o1.shape[0], 1], dtype="float32"),
                    paddle.ones(
                        [o1.shape[0], max(1, o1.shape[1] // 2)], dtype=o1.dtype
                    ),
                )
            )

            node = ExpertsGroupGemmContiguousNode(
                CustomMap(),
                use_fp8_mlp=True,
                moe_deep_gemm=True,
                moe_expert_fusion=True,
                use_ue8m0=True,
            )
            node.tokens_per_expert = [2]
            node.tokens_per_expert_indices = paddle.to_tensor(
                [0, 0], dtype="int32"
            )
            node.m_indices = paddle.to_tensor([0, 0], dtype="int32")
            weights1 = paddle.ones([1, 2, 4], dtype="float32")
            weights2 = paddle.ones([1, 2, 4], dtype="float32")
            gate = node.fwd_gate_up_fp8(
                paddle.ones([2, 2], dtype="float32"), weights1, 1, [2]
            )
            down = node.fwd_down_fp8(
                paddle.ones([2, 4], dtype="float32"),
                paddle.ones([2, 1], dtype="float32"),
                weights2,
                1,
            )
            do1, o2_s, probs_grad = node.bwd_down_input_fp8(
                weights2,
                paddle.ones([2, 2], dtype="float32"),
                paddle.ones([2, 4], dtype="float32"),
                paddle.ones([2, 1], dtype="float32"),
            )
            dx = node.bwd_gate_up_input_fp8(do1, weights1)
            empty_gate = node.fwd_gate_up_bf16(
                paddle.empty([0, 2], dtype="float32"), weights1
            )
            empty_down = node.fwd_down_bf16(
                paddle.empty([0, 4], dtype="float32"),
                paddle.empty([0, 1], dtype="float32"),
                weights2,
            )
            clear_box = FakeClearTensor(paddle.ones([2, 4], dtype="float32"))
            node.fwd_down_bf16(
                clear_box,
                paddle.ones([2, 1], dtype="float32"),
                weights2,
                clear_o1=True,
            )
            node.use_fp8_mlp = False
            do1_bf16, _, _ = node.bwd_down_input_bf16(
                weights2,
                paddle.ones([2, 2], dtype="float32"),
                paddle.ones([2, 4], dtype="float32"),
                paddle.ones([2, 1], dtype="float32"),
            )
            dx_bf16 = node.bwd_gate_up_input_bf16(do1_bf16, weights1)
            node.use_fp8_mlp = True
            empty_down_grad = node.bwd_down_input_bf16(
                weights2,
                paddle.empty([0, 2], dtype="float32"),
                paddle.empty([0, 4], dtype="float32"),
                paddle.empty([0, 1], dtype="float32"),
            )[0]
            empty_gate_grad = node.bwd_gate_up_input_bf16(
                paddle.empty([0, 4], dtype="float32"), weights1
            )
        finally:
            fp8_utils.fused_stack_quant = old_fused_stack
            fp8_utils.split_group_gemm = old_split_group
            if old_deep is None:
                if hasattr(fp8_utils, "deep_gemm"):
                    delattr(fp8_utils, "deep_gemm")
            else:
                fp8_utils.deep_gemm = old_deep
            paddle.incubate.nn.functional.fp8_quant_blockwise = old_quant
            if old_swiglu_quant is None:
                if hasattr(fp8_utils, "fuse_weighted_swiglu_fp8_quant"):
                    delattr(fp8_utils, "fuse_weighted_swiglu_fp8_quant")
            else:
                fp8_utils.fuse_weighted_swiglu_fp8_quant = old_swiglu_quant
            fp8_utils.USE_INPLACE_SWIGLU_BWD = old_swiglu_bwd_flag
            if old_fused_weighted_bwd is None:
                if hasattr(fp8_utils, "fused_swiglu_weighted_bwd"):
                    delattr(fp8_utils, "fused_swiglu_weighted_bwd")
            else:
                fp8_utils.fused_swiglu_weighted_bwd = old_fused_weighted_bwd
            if old_fused_weighted_clamp_bwd is None:
                if hasattr(fp8_utils, "fused_swiglu_weighted_clamp_bwd"):
                    delattr(fp8_utils, "fused_swiglu_weighted_clamp_bwd")
            else:
                fp8_utils.fused_swiglu_weighted_clamp_bwd = (
                    old_fused_weighted_clamp_bwd
                )
            if old_fused_swiglu_probs_bwd is None:
                if hasattr(fp8_utils, "_fused_swiglu_probs_bwd"):
                    delattr(fp8_utils, "_fused_swiglu_probs_bwd")
            else:
                fp8_utils._fused_swiglu_probs_bwd = old_fused_swiglu_probs_bwd
            if old_fused_backward is None:
                if hasattr(fp8_utils, "fused_swiglu_scale_backward"):
                    delattr(fp8_utils, "fused_swiglu_scale_backward")
            else:
                fp8_utils.fused_swiglu_scale_backward = old_fused_backward
            fp8_utils.fused_swiglu_scale_forward = old_fused_forward

        self.assertEqual(gate.shape, [2, 2])
        self.assertEqual(down.shape, [2, 2])
        self.assertEqual(o2_s.shape, [2, 2])
        self.assertEqual(probs_grad.shape, [2, 1])
        self.assertEqual(dx.shape, [2, 2])
        self.assertEqual(empty_gate.shape, [0, 4])
        self.assertEqual(empty_down.shape, [0, 4])
        self.assertTrue(clear_box.cleared)
        self.assertEqual(dx_bf16.shape, [2, 2])
        self.assertEqual(empty_down_grad.shape, [0, 4])
        self.assertEqual(empty_gate_grad.shape, [0, 2])
        self.assertTrue(fake_deep.calls)

    def test_weight_and_lora_backward_paths_with_stubs(self):
        old_fused_backward = getattr(
            fp8_utils, "fused_swiglu_scale_backward", None
        )
        old_fused_forward = fp8_utils.fused_swiglu_scale_forward
        old_batched = paddle.incubate.nn.functional.batched_gemm
        old_kitchen = fp8_utils.kitchen_gemm
        old_deep = getattr(fp8_utils, "deep_gemm", None)
        old_dequant = getattr(
            paddle.incubate.nn.functional, "fused_act_dequant", None
        )
        had_dequant = hasattr(
            paddle.incubate.nn.functional, "fused_act_dequant"
        )
        fake_deep = FakeDeepGemm()
        kitchen_calls = []
        try:
            fp8_utils.fused_swiglu_scale_forward = (
                lambda o1, probs: paddle.ones(
                    [o1.shape[0], max(1, o1.shape[1] // 2)], dtype=o1.dtype
                )
            )
            fp8_utils.fused_swiglu_scale_backward = lambda o1, probs, do2: (
                paddle.ones(o1.shape, dtype=o1.dtype),
                paddle.ones([o1.shape[0], 1], dtype="float32"),
            )

            def fake_batched_gemm(
                x, dy, tokens_per_expert, trans_lhs=False, trans_rhs=False
            ):
                del tokens_per_expert, trans_rhs
                if len(x.shape) == 2:
                    if trans_lhs:
                        if x.shape[1] == 4:
                            return paddle.ones(
                                [1, 4, dy.shape[1]], dtype="float32"
                            )
                        return paddle.ones(
                            [1, x.shape[1], dy.shape[1]], dtype="float32"
                        )
                    return paddle.ones([1, x.shape[1], 2], dtype="float32")
                return paddle.ones([1, 2, 4], dtype="float32")

            paddle.incubate.nn.functional.batched_gemm = fake_batched_gemm
            paddle.incubate.nn.functional.fused_act_dequant = (
                lambda x, scale: paddle.ones(x.shape, dtype="float32")
            )
            fp8_utils.kitchen_gemm = (
                lambda *args, **kwargs: kitchen_calls.append((args, kwargs))
            )
            fp8_utils.deep_gemm = fake_deep

            node = ExpertsGroupGemmContiguousNode(
                CustomMap(), use_fp8_mlp=True, moe_expert_fusion=False
            )
            node.tokens_per_expert = [2]
            node.fused_transpose_split_quant = lambda x, scale, tokens, pow2: (
                paddle.ones([2, 2, 2], dtype="float32"),
                paddle.ones([2, 2, 1], dtype="float32"),
            )
            w_main = TensorBox(paddle.ones([2, 2], dtype="float32"), True)
            w_grad = TensorBox(paddle.ones([2, 2], dtype="float32"), False)
            node.use_ue8m0 = False
            node.bwd_down_weight(
                paddle.ones([2, 2], dtype="float32"),
                paddle.ones([2, 2], dtype="float32"),
                [w_main, w_grad],
            )
            node.dequant_input = False
            node.input = paddle.ones([2, 2], dtype="float32")
            node.bwd_gate_up_weight(
                paddle.ones([2, 2], dtype="float32"),
                None,
                [w_main, w_grad],
                clear_input=True,
            )
            node.use_ue8m0 = True
            node.dequant_input = True
            node.input_fp8 = paddle.ones([2, 2], dtype="float32")
            node.input_scale = paddle.ones([2, 1], dtype="float32")
            node.bwd_gate_up_weight(
                paddle.ones([2, 2], dtype="float32"),
                None,
                [w_main],
                clear_input=True,
            )
            node.bwd_down_weight(
                paddle.ones([2, 2], dtype="float32"),
                paddle.ones([2, 2], dtype="float32"),
                [w_main],
            )
            grouped_node = ExpertsGroupGemmContiguousNode(
                CustomMap(), use_fp8_mlp=False, moe_expert_fusion=True
            )
            grouped_node.tokens_per_expert = [2]
            grouped_node.input = paddle.ones([2, 2], dtype="float32")
            grouped_node.dequant_input = False
            grouped_node.use_fp8_mlp = False
            grouped_node.moe_deep_gemm = False
            grouped_weight = TensorBox(
                paddle.ones([1, 2, 2], dtype="float32"), False
            )
            grouped_node.bf16_weight_grad(
                paddle.ones([2, 2], dtype="float32"), None, grouped_weight
            )

            lora_node = ExpertsGroupGemmContiguousNode(
                CustomMap(), use_fp8_mlp=False, moe_expert_fusion=True
            )
            lora_node.grouped_gemm_experts = LoRAGroupedWeights()
            lora_node.tokens_per_expert = [2]
            lora_node.o1 = paddle.ones([2, 4], dtype="float32")
            lora_node.input = paddle.ones([2, 2], dtype="float32")
            lora_node.bwd_down_input_bf16 = lambda w2, grad, o1, probs: (
                paddle.ones([2, 4], dtype="float32"),
                paddle.ones([2, 4], dtype="float32"),
                paddle.ones([2, 1], dtype="float32"),
            )
            lora_node.bwd_gate_up_input_bf16 = lambda do1, w1: paddle.ones(
                [2, 2], dtype="float32"
            )
            dx, probs_grad = lora_node.backward_impl_bf16(
                paddle.ones([2, 2], dtype="float32"),
                paddle.ones([2, 1], dtype="float32"),
            )
            with self.assertRaises(NotImplementedError):
                lora_node.backward_impl_bf16(
                    paddle.ones([2, 2], dtype="float32"),
                    paddle.ones([2, 1], dtype="float32"),
                    a2a_async_fn=lambda x: x,
                )
        finally:
            if old_fused_backward is None:
                if hasattr(fp8_utils, "fused_swiglu_scale_backward"):
                    delattr(fp8_utils, "fused_swiglu_scale_backward")
            else:
                fp8_utils.fused_swiglu_scale_backward = old_fused_backward
            fp8_utils.fused_swiglu_scale_forward = old_fused_forward
            paddle.incubate.nn.functional.batched_gemm = old_batched
            fp8_utils.kitchen_gemm = old_kitchen
            if old_deep is None:
                if hasattr(fp8_utils, "deep_gemm"):
                    delattr(fp8_utils, "deep_gemm")
            else:
                fp8_utils.deep_gemm = old_deep
            if had_dequant:
                paddle.incubate.nn.functional.fused_act_dequant = old_dequant
            else:
                delattr(paddle.incubate.nn.functional, "fused_act_dequant")

        self.assertIsNotNone(w_main.main_grad)
        self.assertIsNotNone(w_grad.grad)
        self.assertTrue(w_main.hook_called)
        self.assertTrue(w_grad.hook_called)
        self.assertTrue(kitchen_calls)
        self.assertEqual(dx.shape, [2, 2])
        self.assertEqual(probs_grad.shape, [2, 1])
        self.assertIsNotNone(
            lora_node.grouped_gemm_experts.weight1_lora_A.main_grad
        )
        self.assertIsNotNone(
            lora_node.grouped_gemm_experts.weight2_lora_A.main_grad
        )

    def test_backward_impl_fp8_a2a_non_bf16_weight_paths(self):
        node = ExpertsGroupGemmContiguousNode(
            CustomMap(),
            use_fp8_mlp=True,
            moe_expert_fusion=True,
            moe_deep_gemm=True,
        )
        node.grouped_gemm_experts = GroupedWeights()
        node.tokens_per_expert = [2]
        node.o1 = paddle.ones([2, 4], dtype="float32")
        node.input = paddle.ones([2, 2], dtype="float32")
        node.use_bf16_gemm_weight_grad = False
        node.dw_p2p_overlap = False
        node.bwd_down_input_fp8 = (
            lambda w2, grad, o1, probs, inplace_swiglu_prob=False: (
                paddle.ones([2, 4], dtype="float32"),
                paddle.ones([2, 2], dtype="float32"),
                paddle.ones([2, 1], dtype="float32"),
            )
        )
        calls = []
        node.bwd_gate_up_input_fp8 = lambda do1, w1, dx=None: dx
        node.bwd_gate_up_weight = lambda *args, **kwargs: calls.append(
            "gate_weight"
        )
        node.bwd_down_weight = lambda *args, **kwargs: calls.append(
            "down_weight"
        )
        task = TaskRecorder()

        def a2a(x):
            return x, task

        dx, probs_grad = node.backward_impl_fp8(
            paddle.ones([2, 2], dtype="float32"),
            paddle.ones([2, 1], dtype="float32"),
            a2a_async_fn=a2a,
        )

        self.assertEqual(dx.shape, [2, 2])
        self.assertEqual(probs_grad.shape, [2, 1])
        self.assertEqual(calls, ["gate_weight", "down_weight"])
        self.assertTrue(task.wait_called)


if __name__ == "__main__":
    unittest.main()
