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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest
from unittest.mock import MagicMock


class TestOverlapFakeClone(unittest.TestCase):
    """Tests for FakeClone in forward_backward_overlap_utils."""


class TestOverlapDetachAndRequiresGrad(unittest.TestCase):
    """Tests for detach_and_requires_grad."""

    def test_single_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        x = paddle.randn([2, 3])
        x.stop_gradient = False
        result = detach_and_requires_grad(x)
        self.assertFalse(result.stop_gradient)

    def test_single_tensor_stop_gradient(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        x = paddle.randn([2, 3])
        x.stop_gradient = True
        result = detach_and_requires_grad(x)
        self.assertTrue(result.stop_gradient)

    def test_tuple_of_tensors(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        x = paddle.randn([2, 3])
        x.stop_gradient = False
        y = paddle.randn([2, 3])
        y.stop_gradient = True
        result = detach_and_requires_grad((x, y))
        self.assertIsInstance(result, tuple)

    def test_list_of_tensors(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        x = paddle.randn([2, 3])
        result = detach_and_requires_grad([x])
        self.assertIsInstance(result, list)

    def test_dict_of_tensors(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        x = paddle.randn([2, 3])
        x.stop_gradient = True
        result = detach_and_requires_grad({"a": x})
        self.assertIsInstance(result, dict)
        self.assertTrue(result["a"].stop_gradient)

    def test_nested_tuple(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        x = paddle.randn([2, 3])
        x.stop_gradient = False
        y = paddle.randn([2, 3])
        result = detach_and_requires_grad(((x,), y))
        self.assertIsInstance(result, tuple)

    def test_none_input(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        result = detach_and_requires_grad((None,))
        self.assertIsNone(result[0])


class TestOverlapCloneAndClearDataptr(unittest.TestCase):
    """Tests for clone_and_clear_dataptr."""

    def test_single_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        x = paddle.randn([2, 3])
        result = clone_and_clear_dataptr(x)
        self.assertEqual(result.shape, x.shape)

    def test_tuple_of_tensors(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        x = paddle.randn([2, 3])
        y = paddle.randn([2, 3])
        result = clone_and_clear_dataptr((x, y))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_dict_of_tensors(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        x = paddle.randn([2, 3])
        result = clone_and_clear_dataptr({"a": x})
        self.assertIsInstance(result, dict)

    def test_clear_dataptr_dict(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        x = paddle.randn([2, 3])
        result = clone_and_clear_dataptr({"a": x}, clear_dataptr=True)
        self.assertIsInstance(result, dict)


class TestOverlapScheduleChunk(unittest.TestCase):
    """Tests for ScheduleChunk."""

    def test_check_nodes_valid(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleChunk,
            ScheduleNode,
        )

        node = ScheduleNode(lambda x: x)
        chunk = ScheduleChunk([node])
        # Should not raise

    def test_check_nodes_invalid(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleChunk,
        )

        with self.assertRaises(AssertionError):
            ScheduleChunk(["not_a_node"])


class TestOverlapScheduleNode(unittest.TestCase):
    """Tests for ScheduleNode in forward_backward_overlap_utils."""

    def test_init(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = ScheduleNode(fwd_func=lambda x: x, name="test_node")
        self.assertEqual(node.name, "test_node")
        self.assertIsNone(node.labels)
        self.assertIsNone(node.scale_loss_factor)

    def test_forward_simple(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        def fwd_func(inputs, **kwargs):
            return inputs * 2

        x = paddle.randn([2, 3])
        x.stop_gradient = False
        node = ScheduleNode(fwd_func=fwd_func)
        result = node.forward(x)
        self.assertIsNotNone(node.inputs)
        self.assertIsNotNone(node.outputs)

    def test_forward_with_labels(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        labels = paddle.randn([2])

        def fwd_func(inputs, labels, **kwargs):
            return inputs.sum()

        x = paddle.randn([2, 3])
        x.stop_gradient = False
        node = ScheduleNode(fwd_func=fwd_func)
        node.labels = labels
        result = node.forward(x)
        self.assertIsNotNone(result)

    def test_forward_with_scale_loss_factor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        def fwd_func(inputs, **kwargs):
            return inputs.sum()

        x = paddle.randn([2, 3])
        x.stop_gradient = False
        node = ScheduleNode(fwd_func=fwd_func)
        node.scale_loss_factor = 2.0
        result = node.forward(x)
        self.assertIsNotNone(result)

    def test_backward_no_grad_single(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            return inputs.sum()

        node = ScheduleNode(fwd_func=fwd_func)
        node.forward(x)
        grads = node.backward()
        self.assertIsInstance(grads, tuple)

    def test_backward_no_grad_tuple(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            return (inputs.sum(),)

        node = ScheduleNode(fwd_func=fwd_func)
        node.forward(x)
        grads = node.backward()
        self.assertIsInstance(grads, tuple)

    def test_backward_with_output_grad(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            return inputs * 2

        node = ScheduleNode(fwd_func=fwd_func)
        out = node.forward(x)
        grad = paddle.ones_like(out)
        grads = node.backward(output_grad=grad)
        self.assertIsInstance(grads, tuple)

    def test_backward_with_scaler(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            return inputs.sum()

        node = ScheduleNode(fwd_func=fwd_func)
        node.forward(x)
        mock_scaler = MagicMock()
        mock_scaler.scale.return_value = x.sum()
        grads = node.backward(scaler=mock_scaler)
        self.assertIsInstance(grads, tuple)

    def test_reset_states(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = ScheduleNode(fwd_func=lambda x: x)
        node.inputs = [1, 2]
        node.outputs = [3, 4]
        node.labels = "label"
        node.scale_loss_factor = 2.0
        node._reset_states()
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)
        self.assertIsNone(node.labels)
        self.assertIsNone(node.scale_loss_factor)


if __name__ == "__main__":
    unittest.main()
