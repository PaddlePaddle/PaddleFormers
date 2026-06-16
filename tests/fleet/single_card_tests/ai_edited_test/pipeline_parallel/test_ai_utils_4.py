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


class TestGetPpFirstRank(unittest.TestCase):
    """Tests for get_pp_first_rank."""

    def test_first_rank(self):
        from paddleformers.fleet.pipeline_parallel.utils import (
            get_pp_first_rank,
        )

        mock_group = MagicMock()
        mock_group.ranks.return_value = [3, 5, 7, 9]
        result = get_pp_first_rank(mock_group)
        self.assertEqual(result, 3)

    def test_single_rank(self):
        from paddleformers.fleet.pipeline_parallel.utils import (
            get_pp_first_rank,
        )

        mock_group = MagicMock()
        mock_group.ranks.return_value = [0]
        result = get_pp_first_rank(mock_group)
        self.assertEqual(result, 0)


class TestGetPpLastRank(unittest.TestCase):
    """Tests for get_pp_last_rank."""

    def test_last_rank(self):
        from paddleformers.fleet.pipeline_parallel.utils import get_pp_last_rank

        mock_group = MagicMock()
        mock_group.ranks.return_value = [3, 5, 7, 9]
        result = get_pp_last_rank(mock_group)
        self.assertEqual(result, 9)


class TestGetPpNextRank(unittest.TestCase):
    """Tests for get_pp_next_rank."""

    def test_not_last_stage(self):
        from paddleformers.fleet.pipeline_parallel.utils import get_pp_next_rank

        mock_group = MagicMock()
        mock_group.ranks.return_value = [3, 5, 7, 9]
        with (
            patch(
                "paddleformers.fleet.pipeline_parallel.utils.is_pp_last_stage",
                return_value=False,
            ),
            patch(
                "paddleformers.fleet.pipeline_parallel.utils.get_pg_rank",
                return_value=1,
            ),
        ):
            result = get_pp_next_rank(mock_group)
            self.assertEqual(result, 7)

    def test_last_stage_returns_none(self):
        from paddleformers.fleet.pipeline_parallel.utils import get_pp_next_rank

        mock_group = MagicMock()
        with patch(
            "paddleformers.fleet.pipeline_parallel.utils.is_pp_last_stage",
            return_value=True,
        ):
            result = get_pp_next_rank(mock_group)
            self.assertIsNone(result)


class TestGetPpPrevRank(unittest.TestCase):
    """Tests for get_pp_prev_rank."""

    def test_not_first_stage(self):
        from paddleformers.fleet.pipeline_parallel.utils import get_pp_prev_rank

        mock_group = MagicMock()
        mock_group.ranks.return_value = [3, 5, 7, 9]
        with (
            patch(
                "paddleformers.fleet.pipeline_parallel.utils.is_pp_first_stage",
                return_value=False,
            ),
            patch(
                "paddleformers.fleet.pipeline_parallel.utils.get_pg_rank",
                return_value=2,
            ),
        ):
            result = get_pp_prev_rank(mock_group)
            # ranks[2-1] = ranks[1] = 5
            self.assertEqual(result, 5)

    def test_first_stage_returns_none(self):
        from paddleformers.fleet.pipeline_parallel.utils import get_pp_prev_rank

        mock_group = MagicMock()
        with patch(
            "paddleformers.fleet.pipeline_parallel.utils.is_pp_first_stage",
            return_value=True,
        ):
            result = get_pp_prev_rank(mock_group)
            self.assertIsNone(result)


class TestMakeViewless(unittest.TestCase):
    """Tests for make_viewless utility function."""

    def test_non_view_tensor(self):
        from paddleformers.fleet.pipeline_parallel.utils import make_viewless

        tensor = paddle.randn([2, 3])
        tensor.stop_gradient = True
        # A non-view tensor is returned as-is by make_viewless_tensor
        with patch(
            "paddleformers.fleet.pipeline_parallel.utils.make_viewless_tensor",
            return_value=tensor,
        ):
            result = make_viewless(tensor)
            self.assertIs(result, tensor)

    def test_view_tensor_keep_graph(self):
        from paddleformers.fleet.pipeline_parallel.utils import make_viewless

        tensor = paddle.randn([4, 6])
        tensor.stop_gradient = False
        # Mock the make_viewless_tensor to return a new tensor
        new_tensor = paddle.randn([4, 6])
        with patch(
            "paddleformers.fleet.pipeline_parallel.utils.make_viewless_tensor",
            return_value=new_tensor,
        ):
            result = make_viewless(tensor)
            self.assertEqual(list(result.shape), [4, 6])


class TestStreamAcquireContext(unittest.TestCase):
    """Tests for stream_acquire_context."""

    def test_context_calls_wait_and_record(self):
        from paddleformers.fleet.pipeline_parallel.utils import (
            stream_acquire_context,
        )

        mock_stream = MagicMock()
        mock_event = MagicMock()
        with stream_acquire_context(mock_stream, mock_event):
            pass
        mock_event.wait.assert_called_once_with(mock_stream)
        mock_event.record.assert_called_once_with(mock_stream)


class TestNoopScheduleNode(unittest.TestCase):
    """Tests for NoopScheduleNode."""

    def test_forward_passthrough(self):
        from paddleformers.fleet.pipeline_parallel.utils import NoopScheduleNode

        node = NoopScheduleNode()
        inputs = [1, 2, 3]
        result = node.forward(inputs)
        self.assertEqual(result, inputs)

    def test_backward_passthrough(self):
        from paddleformers.fleet.pipeline_parallel.utils import NoopScheduleNode

        node = NoopScheduleNode()
        outgrads = [0.1, 0.2]
        result = node.backward(outgrads)
        self.assertEqual(result, outgrads)


class TestSetGetStreams(unittest.TestCase):
    """Tests for set_streams, get_comp_stream, get_comm_stream."""

    def test_set_and_get_streams(self):
        import paddleformers.fleet.pipeline_parallel.utils as utils_mod
        from paddleformers.fleet.pipeline_parallel.utils import (
            get_comm_stream,
            get_comp_stream,
            set_streams,
        )

        # Save original
        orig_comp = utils_mod._COMP_STREAM
        orig_comm = utils_mod._COMM_STREAM

        try:
            # Reset to None
            utils_mod._COMP_STREAM = None
            utils_mod._COMM_STREAM = None

            comp = MagicMock()
            comm = MagicMock()
            set_streams(comp_stream=comp, comm_stream=comm)
            self.assertIs(get_comp_stream(), comp)
            self.assertIs(get_comm_stream(), comm)
        finally:
            utils_mod._COMP_STREAM = orig_comp
            utils_mod._COMM_STREAM = orig_comm

    def test_set_streams_already_set(self):
        import paddleformers.fleet.pipeline_parallel.utils as utils_mod
        from paddleformers.fleet.pipeline_parallel.utils import (
            get_comm_stream,
            get_comp_stream,
            set_streams,
        )

        orig_comp = utils_mod._COMP_STREAM
        orig_comm = utils_mod._COMM_STREAM

        try:
            comp1 = MagicMock()
            comm1 = MagicMock()
            utils_mod._COMP_STREAM = comp1
            utils_mod._COMM_STREAM = comm1

            comp2 = MagicMock()
            comm2 = MagicMock()
            set_streams(comp_stream=comp2, comm_stream=comm2)
            # Should keep original values
            self.assertIs(get_comp_stream(), comp1)
            self.assertIs(get_comm_stream(), comm1)
        finally:
            utils_mod._COMP_STREAM = orig_comp
            utils_mod._COMM_STREAM = orig_comm


if __name__ == "__main__":
    unittest.main()
