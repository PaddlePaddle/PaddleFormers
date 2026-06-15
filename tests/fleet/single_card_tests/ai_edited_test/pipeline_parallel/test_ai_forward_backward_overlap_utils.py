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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest


class TestScheduleChunkInit(unittest.TestCase):
    """Tests for ScheduleChunk initialization."""

    def test_init_empty_nodes(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        chunk = ScheduleChunk([])
        self.assertEqual(len(chunk.nodes), 0)

    def test_init_with_schedule_chunk_node(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleChunk,
            ScheduleNode,
        )

        node = ScheduleNode(lambda x: x)
        chunk = ScheduleChunk([node])
        self.assertEqual(len(chunk.nodes), 1)
        self.assertIs(chunk.nodes[0], node)


class TestScheduleChunkForward(unittest.TestCase):
    """Tests for ScheduleChunk.forward."""

    def test_forward_empty_chunk(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        chunk = ScheduleChunk([])
        result = chunk.forward("input")
        self.assertEqual(result, "input")

    def test_forward_chain_empty(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        chunk = ScheduleChunk([])
        result = chunk.forward((1, 2, 3))
        self.assertEqual(result, (1, 2, 3))


class TestScheduleChunkBackward(unittest.TestCase):
    """Tests for ScheduleChunk.backward."""

    def test_backward_empty_chunk(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        chunk = ScheduleChunk([])
        result = chunk.backward("grad")
        self.assertEqual(result, "grad")

    def test_backward_empty_chunk_tuple(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        chunk = ScheduleChunk([])
        result = chunk.backward((1.0, 2.0))
        self.assertEqual(result, (1.0, 2.0))


class TestScheduleChunkCheckNodesValid(unittest.TestCase):
    """Tests for ScheduleChunk._check_nodes_valid."""

    def test_valid_schedule_node(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleChunk,
            ScheduleNode,
        )

        node = ScheduleNode(lambda x: x)
        chunk = ScheduleChunk([node])
        self.assertEqual(len(chunk.nodes), 1)

    def test_valid_schedule_chunk_nested(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        inner = ScheduleChunk([])
        outer = ScheduleChunk([inner])
        self.assertEqual(len(outer.nodes), 1)
        self.assertIsInstance(outer.nodes[0], ScheduleChunk)

    def test_invalid_node_type(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        with self.assertRaises(AssertionError):
            ScheduleChunk(["not_a_node"])

    def test_invalid_node_type_int(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        with self.assertRaises(AssertionError):
            ScheduleChunk([42])


class TestFakeCloneApply(unittest.TestCase):
    """Tests for FakeClone static methods."""

    def test_forward_returns_empty_like(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            FakeClone,
        )

        input_tensor = paddle.randn([3, 4])
        result = FakeClone.apply(input_tensor)
        self.assertEqual(result.shape, [3, 4])
        # Result is a new tensor (empty_like)
        self.assertFalse(paddle.allclose(result, input_tensor))

    def test_forward_preserves_shape(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            FakeClone,
        )

        input_tensor = paddle.randn([5, 10, 20])
        result = FakeClone.apply(input_tensor)
        self.assertEqual(list(result.shape), [5, 10, 20])


class TestDetachAndRequiresGrad(unittest.TestCase):
    """Tests for detach_and_requires_grad function."""

    def test_single_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        tensor = paddle.randn([3, 4])
        tensor.stop_gradient = False
        result = detach_and_requires_grad(tensor)
        self.assertIsInstance(result, paddle.Tensor)
        self.assertFalse(result.stop_gradient)

    def test_single_tensor_stop_gradient_true(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        tensor = paddle.randn([3, 4])
        tensor.stop_gradient = True
        result = detach_and_requires_grad(tensor)
        self.assertTrue(result.stop_gradient)

    def test_tuple_of_tensors(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 3])
        result = detach_and_requires_grad((t1, t2))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_list_of_tensors(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 3])
        result = detach_and_requires_grad([t1, t2])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_dict_input(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        data = {"a": paddle.randn([2, 3]), "b": paddle.randn([2, 3])}
        result = detach_and_requires_grad(data)
        self.assertIsInstance(result, dict)
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_nested_tuple_with_none(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        t1 = paddle.randn([2, 3])
        result = detach_and_requires_grad((t1, None))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[1])


class TestCloneAndClearDataptr(unittest.TestCase):
    """Tests for clone_and_clear_dataptr function."""

    def test_single_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        tensor = paddle.randn([3, 4])
        result = clone_and_clear_dataptr(tensor)
        self.assertIsInstance(result, paddle.Tensor)
        self.assertEqual(list(result.shape), [3, 4])

    def test_tuple_of_tensors(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 3])
        result = clone_and_clear_dataptr((t1, t2))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_tuple_with_none(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        t1 = paddle.randn([2, 3])
        result = clone_and_clear_dataptr((t1, None))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)

    def test_dict_of_tensors(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        data = {"a": paddle.randn([2, 3])}
        result = clone_and_clear_dataptr(data)
        self.assertIsInstance(result, dict)
        self.assertIn("a", result)

    def test_clear_dataptr_flag(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        tensor = paddle.randn([3, 4])
        result = clone_and_clear_dataptr(tensor, clear_dataptr=True)
        self.assertIsInstance(result, paddle.Tensor)


class TestScheduleNodeInit(unittest.TestCase):
    """Tests for ScheduleNode initialization."""

    def test_init_defaults(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        node = ScheduleNode(lambda x: x * 2, name="test_node")
        self.assertEqual(node.name, "test_node")
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)
        self.assertIsNone(node.labels)
        self.assertIsNone(node.scale_loss_factor)

    def test_init_with_name(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        node = ScheduleNode(lambda x: x, name="my_node")
        self.assertEqual(node.name, "my_node")

    def test_init_default_name(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        node = ScheduleNode(lambda x: x)
        self.assertEqual(node.name, "")


class TestScheduleNodeResetStates(unittest.TestCase):
    """Tests for ScheduleNode._reset_states."""

    def test_reset_clears_all(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        node = ScheduleNode(lambda x: x)
        node.inputs = "some_input"
        node.outputs = "some_output"
        node.labels = "some_labels"
        node.scale_loss_factor = 2.0

        node._reset_states()
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)
        self.assertIsNone(node.labels)
        self.assertIsNone(node.scale_loss_factor)

    def test_reset_idempotent(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        node = ScheduleNode(lambda x: x)
        node._reset_states()
        node._reset_states()
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)


if __name__ == "__main__":
    unittest.main()
