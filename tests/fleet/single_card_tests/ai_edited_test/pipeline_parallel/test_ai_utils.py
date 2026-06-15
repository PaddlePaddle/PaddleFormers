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
from unittest.mock import MagicMock, patch

import paddle


class TestNoopScheduleNode(unittest.TestCase):
    """Tests for NoopScheduleNode in pipeline_parallel/utils.py."""

    def test_forward_passthrough_tensor(self):
        from paddleformers.fleet.pipeline_parallel.utils import NoopScheduleNode

        node = NoopScheduleNode()
        t = paddle.randn([2, 3])
        result = node.forward(t)
        self.assertIs(result, t)

    def test_forward_passthrough_tuple(self):
        from paddleformers.fleet.pipeline_parallel.utils import NoopScheduleNode

        node = NoopScheduleNode()
        data = (1, 2, 3)
        result = node.forward(data)
        self.assertEqual(result, data)

    def test_backward_passthrough_tensor(self):
        from paddleformers.fleet.pipeline_parallel.utils import NoopScheduleNode

        node = NoopScheduleNode()
        t = paddle.randn([2, 3])
        result = node.backward(t)
        self.assertIs(result, t)

    def test_backward_passthrough_list(self):
        from paddleformers.fleet.pipeline_parallel.utils import NoopScheduleNode

        node = NoopScheduleNode()
        data = [1.0, 2.0, 3.0]
        result = node.backward(data)
        self.assertEqual(result, data)


class TestScheduleNodeInit(unittest.TestCase):
    """Tests for ScheduleNode initialization in pipeline_parallel/utils.py."""

    def test_init_with_backward_func(self):
        from paddleformers.fleet.pipeline_parallel.utils import ScheduleNode

        def custom_backward(outputs, output_grad):
            return output_grad

        stream = MagicMock()
        event = MagicMock()
        node = ScheduleNode(
            forward_func=lambda x: x,
            stream=stream,
            event=event,
            backward_func=custom_backward,
            name="test_node",
        )
        self.assertEqual(node.name, "test_node")
        self.assertEqual(node.backward_func, custom_backward)
        self.assertEqual(node.stream, stream)
        self.assertEqual(node.event, event)
        self.assertFalse(node.free_input)

    def test_init_free_input_assertion(self):
        from paddleformers.fleet.pipeline_parallel.utils import ScheduleNode

        stream = MagicMock()
        event = MagicMock()
        with self.assertRaises(AssertionError):
            ScheduleNode(
                forward_func=lambda x: x,
                stream=stream,
                event=event,
                free_input=True,
            )

    def test_default_backward_func(self):
        from paddleformers.fleet.pipeline_parallel.utils import ScheduleNode

        stream = MagicMock()
        event = MagicMock()
        node = ScheduleNode(
            forward_func=lambda x: x,
            stream=stream,
            event=event,
        )
        self.assertEqual(node.backward_func, node.default_backward_func)


class TestScheduleNodeDefaultBackward(unittest.TestCase):
    """Tests for ScheduleNode.default_backward_func."""

    def test_backward_no_grad_single_output(self):
        from paddleformers.fleet.pipeline_parallel.utils import ScheduleNode

        stream = MagicMock()
        event = MagicMock()

        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False

        node = ScheduleNode(
            forward_func=lambda x: x.sum(),
            stream=stream,
            event=event,
            name="test",
        )
        node.inputs = [x]
        node.output = x.sum()

        with patch.object(node, "_reset_states"):
            # Test with output_grad=None and single tensor output
            result = node.default_backward_func(node.output, None)
            node._reset_states()

    def test_backward_with_grad_single(self):
        from paddleformers.fleet.pipeline_parallel.utils import ScheduleNode

        stream = MagicMock()
        event = MagicMock()

        x = paddle.randn([2, 3], dtype="float32")
        x.stop_gradient = False

        node = ScheduleNode(
            forward_func=lambda x: x * 2,
            stream=stream,
            event=event,
        )
        node.inputs = [x]
        output = x * 2
        node.output = output

        grad = paddle.ones_like(output)
        with patch.object(node, "_reset_states"):
            result = node.default_backward_func(output, (grad,))
            node._reset_states()


class TestScheduleNodeResetStates(unittest.TestCase):
    """Tests for ScheduleNode._reset_states."""

    def test_reset_clears_inputs_outputs(self):
        from paddleformers.fleet.pipeline_parallel.utils import ScheduleNode

        stream = MagicMock()
        event = MagicMock()
        node = ScheduleNode(
            forward_func=lambda x: x,
            stream=stream,
            event=event,
        )
        node.inputs = [MagicMock()]
        node.outputs = MagicMock()
        node._reset_states()
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)


class TestAbstractSchedulePlan(unittest.TestCase):
    """Tests for AbstractSchedulePlan."""

    def test_cannot_instantiate(self):
        from paddleformers.fleet.pipeline_parallel.utils import AbstractSchedulePlan

        with self.assertRaises(TypeError):
            AbstractSchedulePlan()

    def test_subclass_must_implement_run(self):
        from paddleformers.fleet.pipeline_parallel.utils import AbstractSchedulePlan

        class IncompletePlan(AbstractSchedulePlan):
            pass

        with self.assertRaises(TypeError):
            IncompletePlan()

    def test_subclass_with_run(self):
        from paddleformers.fleet.pipeline_parallel.utils import AbstractSchedulePlan

        class ConcretePlan(AbstractSchedulePlan):
            @staticmethod
            def run(
                f_schedule_plan,
                b_schedule_plan,
                grad=None,
                pre_forward=None,
                pre_backward=None,
                post_forward=None,
                post_backward=None,
            ):
                return None

        plan = ConcretePlan()
        result = plan.run(None, None)
        self.assertIsNone(result)


class TestSetGetStreams(unittest.TestCase):
    """Tests for set_streams with default parameters."""

    def test_set_streams_default_comp(self):
        import paddleformers.fleet.pipeline_parallel.utils as utils_mod

        orig_comp = utils_mod._COMP_STREAM
        orig_comm = utils_mod._COMM_STREAM

        try:
            utils_mod._COMP_STREAM = None
            utils_mod._COMM_STREAM = None

            with (
                patch("paddle.cuda.current_stream", return_value=MagicMock()),
                patch("paddle.cuda.Stream", return_value=MagicMock()),
            ):
                utils_mod.set_streams()
                self.assertIsNotNone(utils_mod.get_comp_stream())
                self.assertIsNotNone(utils_mod.get_comm_stream())
        finally:
            utils_mod._COMP_STREAM = orig_comp
            utils_mod._COMM_STREAM = orig_comm

    def test_get_comp_stream_none(self):
        import paddleformers.fleet.pipeline_parallel.utils as utils_mod

        orig_comp = utils_mod._COMP_STREAM
        try:
            utils_mod._COMP_STREAM = None
            result = utils_mod.get_comp_stream()
            self.assertIsNone(result)
        finally:
            utils_mod._COMP_STREAM = orig_comp

    def test_get_comm_stream_none(self):
        import paddleformers.fleet.pipeline_parallel.utils as utils_mod

        orig_comm = utils_mod._COMM_STREAM
        try:
            utils_mod._COMM_STREAM = None
            result = utils_mod.get_comm_stream()
            self.assertIsNone(result)
        finally:
            utils_mod._COMM_STREAM = orig_comm


if __name__ == "__main__":
    unittest.main()
