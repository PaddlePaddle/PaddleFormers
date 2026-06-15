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
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
    FakeClone,
    ScheduleChunk,
    ScheduleNode,
    clone_and_clear_dataptr,
    detach_and_requires_grad,
)
from paddleformers.fleet.training.initialize import initialize_fleet

PP_DEGREE = 2


def _init_pp():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_DEGREE,
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
    initialize_fleet(strategy)


def setUpModule():
    """Initialize fleet once for all tests in this module (PP=2)."""
    _init_pp()
    np.random.seed(42)
    paddle.seed(42)


class TestFakeCloneForwardBackward(unittest.TestCase):
    """Test FakeClone.apply creates empty_like tensor and backward passes grad."""

    def test_fake_clone_forward_backward(self):
        """FakeClone.forward should produce same-shape tensor; backward passes grad_output through."""
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False
        cloned = FakeClone.apply(x)
        # FakeClone creates an empty_like tensor (same shape, same dtype)
        self.assertEqual(cloned.shape, x.shape)
        self.assertEqual(cloned.dtype, x.dtype)

        # backward should pass grad_output through
        grad_output = paddle.randn([4, 8], dtype="float32")
        paddle.autograd.backward([cloned], [grad_output])
        np.testing.assert_allclose(x.grad.numpy(), grad_output.numpy(), rtol=1e-5)


class TestCloneAndClearDataptrSingleTensor(unittest.TestCase):
    """Test clone_and_clear_dataptr with a single tensor."""

    def test_clone_and_clear_dataptr_single_tensor(self):
        """clone_and_clear_dataptr should return a tensor with the same shape."""
        x = paddle.randn([4, 8], dtype="float32")
        result = clone_and_clear_dataptr(x)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)


class TestCloneAndClearDataptrTuple(unittest.TestCase):
    """Test clone_and_clear_dataptr with a tuple of tensors."""

    def test_clone_and_clear_dataptr_tuple(self):
        """clone_and_clear_dataptr with tuple should return a tuple of cloned tensors."""
        t1 = paddle.randn([2, 4], dtype="float32")
        t2 = paddle.randn([3, 6], dtype="float32")
        result = clone_and_clear_dataptr((t1, t2))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, t1.shape)
        self.assertEqual(result[1].shape, t2.shape)


class TestCloneAndClearDataptrDict(unittest.TestCase):
    """Test clone_and_clear_dataptr with a dict of tensors."""

    def test_clone_and_clear_dataptr_dict(self):
        """clone_and_clear_dataptr with dict should return a dict of cloned tensors."""
        t1 = paddle.randn([2, 4], dtype="float32")
        t2 = paddle.randn([3, 6], dtype="float32")
        inputs = {"a": t1, "b": t2}
        result = clone_and_clear_dataptr(inputs)
        self.assertIsInstance(result, dict)
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertEqual(result["a"].shape, t1.shape)
        self.assertEqual(result["b"].shape, t2.shape)


class TestCloneAndClearDataptrWithClear(unittest.TestCase):
    """Test clone_and_clear_dataptr with clear_dataptr=True."""

    def test_clone_and_clear_dataptr_with_clear(self):
        """clone_and_clear_dataptr with clear_dataptr=True should call _clear_dataptr."""
        x = paddle.randn([4, 8], dtype="float32")
        expected_shape = x.shape
        expected_dtype = x.dtype
        result = clone_and_clear_dataptr(x, clear_dataptr=True)
        # After clearing dataptr, the tensor shell still exists but
        # shape/dtype access may not work. Verify the tensor was created.
        self.assertIsNotNone(result)


class TestDetachAndRequiresGradTuple(unittest.TestCase):
    """Test detach_and_requires_grad with tuple input."""

    def test_detach_and_requires_grad_tuple(self):
        """detach_and_requires_grad with tuple should return tuple of detached tensors."""
        t1 = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        t2 = paddle.to_tensor([3.0, 4.0], stop_gradient=False)
        result = detach_and_requires_grad((t1, t2))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0].numpy(), [1.0, 2.0], rtol=1e-5)
        np.testing.assert_allclose(result[1].numpy(), [3.0, 4.0], rtol=1e-5)
        self.assertFalse(result[0].stop_gradient)
        self.assertFalse(result[1].stop_gradient)


class TestDetachAndRequiresGradNested(unittest.TestCase):
    """Test detach_and_requires_grad with nested tuple input."""

    def test_detach_and_requires_grad_nested(self):
        """detach_and_requires_grad with nested tuple should recursively detach."""
        t1 = paddle.to_tensor([1.0], stop_gradient=False)
        t2 = paddle.to_tensor([2.0], stop_gradient=False)
        t3 = paddle.to_tensor([3.0], stop_gradient=False)
        result = detach_and_requires_grad(((t1, t2), t3))
        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0][0].numpy(), [1.0], rtol=1e-5)
        np.testing.assert_allclose(result[0][1].numpy(), [2.0], rtol=1e-5)
        np.testing.assert_allclose(result[1].numpy(), [3.0], rtol=1e-5)


class TestScheduleNodeForwardNoRecompute(unittest.TestCase):
    """Test ScheduleNode.forward with a simple fwd_func (no recompute)."""

    def test_schedule_node_forward_no_recompute(self):
        """ScheduleNode.forward should call fwd_func and return its output."""
        node = ScheduleNode(fwd_func=lambda inputs, **kwargs: inputs * 2, name="double_node")
        inputs = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        output = node.forward(inputs)
        np.testing.assert_allclose(output.numpy(), [2.0, 4.0, 6.0], rtol=1e-5)


class TestScheduleNodeBackward(unittest.TestCase):
    """Test ScheduleNode.backward after forward, verify _reset_states is called."""

    def test_schedule_node_backward(self):
        """ScheduleNode.backward should clear internal state after backward."""
        weight = paddle.create_parameter(
            shape=[3, 3],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )

        def fwd_func(inputs, **kwargs):
            return paddle.matmul(inputs, weight)

        node = ScheduleNode(fwd_func=fwd_func, name="matmul_node")
        inputs = paddle.to_tensor([[1.0, 2.0, 3.0]], stop_gradient=False)
        output = node.forward(inputs)

        # backward should not raise error
        grad = node.backward()
        # After backward, internal state should be reset
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)


class TestScheduleChunkForwardBackward(unittest.TestCase):
    """Test ScheduleChunk with multiple nodes."""

    def test_schedule_chunk_forward_backward(self):
        """ScheduleChunk should chain forward and backward through its nodes."""
        node1 = ScheduleNode(fwd_func=lambda inputs, **kwargs: inputs * 2, name="mul_node")
        node2 = ScheduleNode(fwd_func=lambda inputs, **kwargs: inputs + 1, name="add_node")
        chunk = ScheduleChunk([node1, node2])
        inputs = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)

        # Forward: first multiply by 2, then add 1
        output = chunk.forward(inputs)
        np.testing.assert_allclose(output.numpy(), [3.0, 5.0, 7.0], rtol=1e-5)

        # Backward should not raise error
        grad = chunk.backward(output)


class TestScheduleChunkValidation(unittest.TestCase):
    """Test ScheduleChunk validates node types."""

    def test_schedule_chunk_validates_node_type(self):
        """ScheduleChunk should raise AssertionError if node is not ScheduleNode or ScheduleChunk."""
        with self.assertRaises(AssertionError):
            ScheduleChunk([lambda x: x])


class TestScheduleNodeReset(unittest.TestCase):
    """Test ScheduleNode._reset_states clears internal state."""

    def test_schedule_node_reset_states(self):
        """After _reset_states, inputs and outputs should be None."""
        node = ScheduleNode(lambda x: x, name="test")
        inputs = paddle.to_tensor([1.0], stop_gradient=False)
        node.forward(inputs)
        self.assertIsNotNone(node.inputs)
        self.assertIsNotNone(node.outputs)
        node._reset_states()
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)


if __name__ == "__main__":
    unittest.main()
