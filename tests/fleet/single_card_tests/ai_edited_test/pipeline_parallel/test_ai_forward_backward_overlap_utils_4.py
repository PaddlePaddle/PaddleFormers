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
from unittest.mock import MagicMock, patch

import paddle


class TestOverlapScheduleNodeRecompute(unittest.TestCase):
    """Tests for ScheduleNode with recompute in forward_backward_overlap_utils."""

    def test_forward_with_recompute(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = ScheduleNode(
            lambda inputs, **kw: inputs * 2, name="recompute_node"
        )
        node.use_recompute = True
        node.fw_rng_state = MagicMock()
        node.fwd_rng_state_tracker = {}
        node.fwd_numpy_state = MagicMock()
        node.fwd_random_state = MagicMock()
        node.fwd_custom_state = None
        node.is_fw_autocast = False
        node.amp_level = "O1"
        node.amp_dtype = "float32"
        node.amp_white_list = []
        node.amp_black_list = []

        x = paddle.randn([2, 3])
        x.stop_gradient = False

        with (
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.switch_rng_state_tracker"
            ) as mock_switch,
            patch("paddle.amp.auto_cast"),
            patch("paddle.base.dygraph.guard"),
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.framework"
            ) as mock_fw,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.custom_state_manager"
            ) as mock_csm,
        ):
            mock_csm.custom_get_state_func = MagicMock(return_value=None)
            mock_csm.custom_set_state_func = MagicMock()
            mock_switch.return_value.__enter__ = MagicMock()
            mock_switch.return_value.__exit__ = MagicMock(return_value=False)

            mock_tracer = MagicMock()
            mock_tracer._has_grad = False
            mock_fw._dygraph_tracer.return_value = mock_tracer

            result = node.forward(x)
            self.assertIsNotNone(result)

    def test_forward_without_recompute(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = ScheduleNode(lambda inputs, **kw: inputs * 2, name="normal_node")
        node.use_recompute = False

        x = paddle.randn([2, 3])
        x.stop_gradient = False
        result = node.forward(x)
        self.assertIsNotNone(result)
        self.assertIsNotNone(node.inputs)
        self.assertIsNotNone(node.outputs)


class TestOverlapScheduleNodeBackwardGrad(unittest.TestCase):
    """Tests for ScheduleNode backward with various grad configurations."""

    def test_backward_with_tuple_output_grad(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            return inputs * 2

        node = ScheduleNode(fwd_func=fwd_func)
        out = node.forward(x)

        # Create tuple output_grad
        grad = paddle.ones_like(out)
        grads = node.backward(output_grad=(grad,))
        self.assertIsInstance(grads, tuple)

    def test_backward_dict_outputs(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            return inputs.sum()

        node = ScheduleNode(fwd_func=fwd_func)
        result = node.forward(x)
        # backward with no output_grad (scalar loss)
        grads = node.backward()
        self.assertIsInstance(grads, tuple)


class TestOverlapFakeCloneBackward(unittest.TestCase):
    """Tests for FakeClone backward pass."""

    def test_fake_clone_backward(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            FakeClone,
        )

        x = paddle.randn([3, 4])
        x.stop_gradient = False
        y = FakeClone.apply(x)
        # FakeClone.backward returns grad_output directly
        grad_output = paddle.ones_like(y)
        result = FakeClone.backward(None, grad_output)
        self.assertTrue(paddle.allclose(result, grad_output))


class TestOverlapCloneAndClearDataptrEdgeCases(unittest.TestCase):
    """Edge case tests for clone_and_clear_dataptr."""

    def test_list_with_none(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        t = paddle.randn([2, 3])
        result = clone_and_clear_dataptr([t, None])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # None is filtered

    def test_dict_with_none(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        t = paddle.randn([2, 3])
        result = clone_and_clear_dataptr({"a": t, "b": None})
        self.assertIsInstance(result, dict)
        self.assertIn("a", result)
        self.assertNotIn("b", result)

    def test_single_tensor_clear_dataptr(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        t = paddle.randn([2, 3])
        result = clone_and_clear_dataptr(t, clear_dataptr=True)
        self.assertIsInstance(result, paddle.Tensor)

    def test_list_clear_dataptr(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([3, 4])
        result = clone_and_clear_dataptr([t1, t2], clear_dataptr=True)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)


class TestOverlapScheduleChunkForwardBackward(unittest.TestCase):
    """Tests for ScheduleChunk forward/backward with actual nodes."""

    def test_forward_with_single_node(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleChunk,
            ScheduleNode,
        )

        x = paddle.randn([2, 3])
        x.stop_gradient = False

        node = ScheduleNode(fwd_func=lambda inputs, **kw: inputs * 2)
        chunk = ScheduleChunk([node])
        result = chunk.forward(x)
        self.assertIsNotNone(result)

    def test_forward_with_multiple_nodes(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleChunk,
            ScheduleNode,
        )

        x = paddle.randn([2, 3])
        x.stop_gradient = False

        node1 = ScheduleNode(fwd_func=lambda inputs, **kw: inputs * 2)
        node2 = ScheduleNode(fwd_func=lambda inputs, **kw: inputs + 1)
        chunk = ScheduleChunk([node1, node2])
        result = chunk.forward(x)
        self.assertIsNotNone(result)

    def test_backward_with_multiple_nodes(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleChunk,
            ScheduleNode,
        )

        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            return inputs.sum()

        node1 = ScheduleNode(fwd_func=fwd_func)
        node2 = ScheduleNode(fwd_func=lambda inputs, **kw: inputs)
        chunk = ScheduleChunk([node2, node1])
        chunk.forward(x)
        # backward is called in reverse order
        grads = chunk.backward(output_grad=None)
        self.assertIsNotNone(grads)


if __name__ == "__main__":
    unittest.main()
