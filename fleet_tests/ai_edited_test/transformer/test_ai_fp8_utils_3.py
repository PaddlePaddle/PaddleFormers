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
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import paddle

from paddleformers.fleet.transformer.moe import fp8_utils
from paddleformers.fleet.transformer.moe.fp8_utils import ExpertsGroupGemmContiguousNode


class Capability:
    def __init__(self, major):
        self.major = major

    def __call__(self):
        return (self.major, 0)


class QuantRecorder:
    def __init__(self, calls, name):
        self.calls = calls
        self.name = name

    def __call__(self, weights, use_pow2_scale, use_ue8m0, scale_transpose):
        self.calls.append(
            (
                self.name,
                len(weights),
                use_pow2_scale,
                use_ue8m0,
                scale_transpose,
            )
        )
        weight = paddle.arange(8, dtype="float32").reshape([2, 4])
        scale = paddle.arange(6, dtype="float32").reshape([2, 3])
        return weight, scale


class DeepGemmRecorder:
    def __init__(self):
        self.calls = []

    def fp8_gemm_nt(self, x_pair, w_pair, out, *args, **kwargs):
        del args, kwargs
        self.calls.append((x_pair[0].shape, w_pair[0].shape, out.shape))
        out.set_value(paddle.ones(out.shape, dtype=out.dtype))


class IncubateFunctional:
    def __init__(self, original, calls):
        self.original = original
        self.calls = calls

    def __getattr__(self, name):
        return getattr(self.original, name)

    def fp8_gemm_blockwise(self, **kwargs):
        self.calls.append(kwargs)
        out = kwargs.get("out")
        if out is not None:
            out.set_value(paddle.full(out.shape, 2.0, dtype=out.dtype))
            return out
        a = kwargs["a"]
        b = kwargs["b"]
        return paddle.full(
            [a.shape[0], b.shape[0]], 3.0, dtype=kwargs["out_dtype"]
        )


class Weight:
    def __init__(self, shape):
        self.shape = shape
        self.grad = None
        self.stop_gradient = False
        self.hook_called = False

    def _apply_backward_hook(self):
        self.hook_called = True


class Expert:
    def __init__(self):
        self.up_gate_proj = Projection([2, 4])
        self.down_proj = Projection([4, 2])


class Projection:
    def __init__(self, shape):
        self.weight = Weight(shape)


class CustomMap:
    def __init__(self):
        self.experts = [Expert(), None, Expert()]
        self.grouped_gemm_experts = GroupedWeights()


class GroupedWeights:
    def __init__(self):
        self.weight1 = Weight([2, 2, 4])
        self.weight2 = Weight([2, 4, 2])


class AsyncTask:
    def __init__(self):
        self.wait_called = False

    def wait(self):
        self.wait_called = True


class TestFP8UtilityExtraNoMock(unittest.TestCase):
    def setUp(self):
        self.old_capability = paddle.device.cuda.get_device_capability
        self.old_stack = getattr(fp8_utils, "fuse_stack_fp8_quant", None)
        self.old_stack_t = getattr(
            fp8_utils, "fuse_stack_transpose_fp8_quant", None
        )
        self.old_deep_gemm = getattr(fp8_utils, "deep_gemm", None)
        self.old_functional = paddle.incubate.nn.functional

    def tearDown(self):
        paddle.device.cuda.get_device_capability = self.old_capability
        if self.old_stack is None:
            if hasattr(fp8_utils, "fuse_stack_fp8_quant"):
                delattr(fp8_utils, "fuse_stack_fp8_quant")
        else:
            fp8_utils.fuse_stack_fp8_quant = self.old_stack
        if self.old_stack_t is None:
            if hasattr(fp8_utils, "fuse_stack_transpose_fp8_quant"):
                delattr(fp8_utils, "fuse_stack_transpose_fp8_quant")
        else:
            fp8_utils.fuse_stack_transpose_fp8_quant = self.old_stack_t
        if self.old_deep_gemm is None:
            if hasattr(fp8_utils, "deep_gemm"):
                delattr(fp8_utils, "deep_gemm")
        else:
            fp8_utils.deep_gemm = self.old_deep_gemm
        paddle.incubate.nn.functional = self.old_functional

    def test_fused_stack_quant_without_cache_records_transpose_and_ue8m0(self):
        calls = []
        paddle.device.cuda.get_device_capability = Capability(10)
        fp8_utils.fuse_stack_fp8_quant = QuantRecorder(calls, "stack")
        fp8_utils.fuse_stack_transpose_fp8_quant = QuantRecorder(
            calls, "transpose"
        )
        weights = [paddle.ones([2, 2], dtype="float32")]

        weight, scale = fp8_utils.fused_stack_quant_without_cache(
            weights, transpose=True, use_ue8m0=True
        )
        weight2, scale2 = fp8_utils.fused_stack_quant_without_cache(
            weights, transpose=False, use_ue8m0=False
        )

        self.assertEqual(weight.shape, [2, 4])
        self.assertEqual(scale.shape, [3, 2])
        self.assertEqual(weight2.shape, [2, 4])
        self.assertEqual(scale2.shape, [2, 3])
        self.assertEqual(calls[0], ("transpose", 1, True, True, True))
        self.assertEqual(calls[1], ("stack", 1, True, False, False))

    def test_tilewise_quant_empty_branch_validates_alignment(self):
        empty = paddle.empty([0, 128], dtype="float32")
        x_fp8, x_scale = fp8_utils.tilewise_quant(empty)

        self.assertEqual(x_fp8.shape, [0, 128])
        self.assertEqual(x_scale.shape, [0, 1])
        with self.assertRaises(AssertionError):
            fp8_utils.tilewise_quant(paddle.empty([0, 127], dtype="float32"))

    def test_split_group_gemm_skips_empty_experts_and_aligns_scales(self):
        recorder = DeepGemmRecorder()
        fp8_utils.deep_gemm = recorder
        x = paddle.ones([3, 2], dtype="float32")
        x_scale = paddle.ones([3, 2], dtype="float32")
        weights = paddle.ones([3, 2, 2], dtype="float32")
        scales = paddle.ones([3, 2, 2], dtype="float32")
        out = paddle.zeros([3, 2], dtype="float32")

        result = fp8_utils.split_group_gemm(
            x, x_scale, weights, scales, [1, 0, 2], out, use_ue8m0=True
        )

        self.assertIs(result, out)
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(
            out.numpy().tolist(), [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        )

    def test_kitchen_gemm_empty_and_non_empty_paths(self):
        calls = []
        paddle.incubate.nn.functional = IncubateFunctional(
            self.old_functional, calls
        )
        empty = paddle.empty([0, 2], dtype="float32")
        weight = paddle.ones([3, 2], dtype="float32")
        out = paddle.ones([0, 3], dtype="float32")

        result = fp8_utils.kitchen_gemm(
            empty, empty, weight, weight, True, False, out=out
        )
        non_empty = fp8_utils.kitchen_gemm(
            paddle.ones([2, 2], dtype="float32"),
            paddle.ones([2, 1], dtype="float32"),
            weight,
            paddle.ones([3, 1], dtype="float32"),
            False,
            True,
            rtn_dtype=paddle.float32,
        )

        self.assertEqual(result.shape, [0, 3])
        self.assertEqual(
            non_empty.numpy().tolist(), [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]
        )
        self.assertEqual(calls[0]["accumulate"], False)


class TestExpertsGroupGemmNodeExtraNoMock(unittest.TestCase):
    def test_cache_reset_and_empty_forward_paths(self):
        node = ExpertsGroupGemmContiguousNode(CustomMap(), use_fp8_mlp=False)
        tensors = [[1, 2], paddle.arange(2), "input", "fp8", "scale", "o1"]
        node.set_cached_tensors(tensors)
        self.assertEqual(node.cached_tensors(), tensors)
        node.clear_cached_tensors()
        self.assertEqual(
            node.cached_tensors(), [None, None, None, None, None, None]
        )

        node.tokens_per_expert = [0, 0]
        empty = paddle.empty([0, 2], dtype="float32")
        expert_w1 = [paddle.ones([2, 4], dtype="float32")]
        gate_up = node.fwd_gate_up_bf16(empty, expert_w1)
        down = node.fwd_down_bf16(
            paddle.empty([0, 4], dtype="float32"),
            paddle.empty([0], dtype="float32"),
            [paddle.ones([2, 2], dtype="float32")],
        )
        dx = node.bwd_gate_up_input_bf16(
            paddle.empty([0, 4], dtype="float32"),
            [paddle.ones([2, 4], dtype="float32")],
        )

        self.assertEqual(gate_up.shape, [0, 4])
        self.assertEqual(down.shape, [0, 2])
        self.assertEqual(dx.shape, [0, 2])
        node.reset_state()
        self.assertIsNone(node.tokens_per_expert)

    def test_backward_empty_initializes_split_and_grouped_weight_grads(self):
        split_node = ExpertsGroupGemmContiguousNode(
            CustomMap(), use_fp8_mlp=False
        )
        split_node.tokens_per_expert = [0, 0]
        dx, probs_grad = split_node.backward(
            paddle.empty([0, 2], dtype="float32"),
            paddle.empty([0], dtype="float32"),
        )
        self.assertEqual(dx.shape, [0, 2])
        self.assertEqual(probs_grad.shape, [0, 1])
        self.assertIsNotNone(split_node.experts[0].down_proj.weight.grad)
        self.assertIsNotNone(split_node.experts[2].up_gate_proj.weight.grad)

        grouped_node = ExpertsGroupGemmContiguousNode(
            CustomMap(), use_fp8_mlp=False, moe_expert_fusion=True
        )
        grouped_node.tokens_per_expert = [0, 0]
        task = AsyncTask()

        def a2a(value):
            return value + 1.0, task

        grouped_dx, grouped_probs_grad = grouped_node.backward(
            paddle.empty([0, 2], dtype="float32"),
            paddle.empty([0], dtype="float32"),
            a2a_async_fn=a2a,
        )

        self.assertTrue(task.wait_called)
        self.assertEqual(grouped_dx.shape, [0, 2])
        self.assertEqual(grouped_probs_grad.shape, [0, 1])
        self.assertIsNotNone(grouped_node.grouped_gemm_experts.weight1.grad)
        self.assertIsNotNone(grouped_node.grouped_gemm_experts.weight2.grad)


if __name__ == "__main__":
    unittest.main()
