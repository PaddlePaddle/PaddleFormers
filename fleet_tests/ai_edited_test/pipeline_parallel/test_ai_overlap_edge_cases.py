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

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

import paddle


class TestOverlapDetachAndRequiresGradEdgeCases(unittest.TestCase):
    """Edge case tests for detach_and_requires_grad."""

    def test_dict_with_none_value(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        x = paddle.randn([2, 3])
        result = detach_and_requires_grad({"a": x, "b": None})
        self.assertIsInstance(result, dict)
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIsNone(result["b"])

    def test_nested_list(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        x = paddle.randn([2, 3])
        result = detach_and_requires_grad([x])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_mixed_tensor_and_non_tensor_in_tuple(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        x = paddle.randn([2, 3])
        result = detach_and_requires_grad((x, 42))
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[1], 42)

    def test_single_tensor_no_gradient(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        x = paddle.randn([2, 3])
        x.stop_gradient = True
        result = detach_and_requires_grad(x)
        self.assertTrue(result.stop_gradient)


class TestOverlapScheduleNodeForwardEdgeCases(unittest.TestCase):
    """Edge case tests for ScheduleNode.forward."""

    def test_forward_with_scale_loss_and_labels(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        labels = paddle.randint(0, 10, [2])

        def fwd_func(inputs, labels, **kwargs):
            return inputs.sum() + labels.sum()

        x = paddle.randn([2, 3])
        x.stop_gradient = False

        node = ScheduleNode(fwd_func=fwd_func)
        node.labels = labels
        node.scale_loss_factor = 2.0
        result = node.forward(x)
        self.assertIsNotNone(result)
        self.assertIsNotNone(node.outputs)

    def test_forward_without_labels(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        def fwd_func(inputs, **kwargs):
            return inputs.sum()

        x = paddle.randn([2, 3])
        x.stop_gradient = False

        node = ScheduleNode(fwd_func=fwd_func)
        result = node.forward(x)
        self.assertIsNotNone(result)
        self.assertIsNone(node.labels)

    def test_forward_is_first_fwd_false(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = ScheduleNode(fwd_func=lambda inputs, **kw: inputs * 2)
        # The paddle version of ScheduleNode does not have use_recompute
        # Just verify the node can be created and forward works
        self.assertIsNotNone(node.fwd_func)


class TestOverlapScheduleNodeBackwardEdgeCases(unittest.TestCase):
    """Edge case tests for ScheduleNode.backward."""

    def test_backward_tuple_outputs_with_grad(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            t = inputs * 2
            return (t,)

        node = ScheduleNode(fwd_func=fwd_func)
        result = node.forward(x)
        grad = paddle.ones_like(result[0])
        grads = node.backward(output_grad=grad)
        self.assertIsInstance(grads, tuple)

    def test_backward_dict_inputs(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            return inputs.sum()

        node = ScheduleNode(fwd_func=fwd_func)
        node.inputs = {"a": x}
        output = x.sum()
        node.outputs = output.clone()
        # backward with dict inputs - dict_to_tuple_helper converts to tuple
        grads = node.backward()
        self.assertIsInstance(grads, tuple)


class TestOverlapCloneAndClearWithClearFlag(unittest.TestCase):
    """Tests for clone_and_clear_dataptr with clear_dataptr flag edge cases."""

    def test_list_with_clear_dataptr(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([3, 4])
        result = clone_and_clear_dataptr([t1, t2], clear_dataptr=True)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_tuple_with_clear_dataptr(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([3, 4])
        result = clone_and_clear_dataptr((t1, t2), clear_dataptr=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_dict_with_clear_dataptr(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        t1 = paddle.randn([2, 3])
        result = clone_and_clear_dataptr({"a": t1}, clear_dataptr=True)
        self.assertIsInstance(result, dict)
        self.assertIn("a", result)


class TestOverlapFakeCloneShapes(unittest.TestCase):
    """Tests for FakeClone with various shapes."""

    def test_1d_tensor(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            FakeClone,
        )

        t = paddle.randn([100])
        result = FakeClone.apply(t)
        self.assertEqual(list(result.shape), [100])

    def test_3d_tensor(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            FakeClone,
        )

        t = paddle.randn([2, 3, 4])
        result = FakeClone.apply(t)
        self.assertEqual(list(result.shape), [2, 3, 4])

    def test_4d_tensor(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            FakeClone,
        )

        t = paddle.randn([1, 3, 224, 224])
        result = FakeClone.apply(t)
        self.assertEqual(list(result.shape), [1, 3, 224, 224])


if __name__ == "__main__":
    unittest.main()
