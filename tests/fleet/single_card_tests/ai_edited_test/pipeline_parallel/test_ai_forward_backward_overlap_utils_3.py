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


class TestOverlapScheduleNodeFirstForward(unittest.TestCase):
    """Tests for ScheduleNode.first_forward in forward_backward_overlap_utils."""

    def test_first_forward_sets_use_recompute(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = ScheduleNode(lambda x, **kw: x.sum(), name="test")
        self.assertFalse(node.use_recompute)

        with (
            patch("paddle.get_rng_state", return_value=MagicMock()),
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.get_rng_state_tracker"
            ) as mock_tracker,
            patch("numpy.random.get_state", return_value=MagicMock()),
            patch("random.getstate", return_value=MagicMock()),
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.custom_state_manager"
            ) as mock_csm,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.framework"
            ) as mock_fw,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.core"
            ) as mock_core,
        ):
            mock_tracker_inst = MagicMock()
            mock_tracker_inst.get_states_tracker.return_value = {}
            mock_tracker.return_value = mock_tracker_inst

            mock_csm.custom_get_state_func.return_value = None

            mock_tracer = MagicMock()
            mock_tracer._amp_level = mock_core.AmpLevel.O0
            mock_tracer._amp_dtype = "float32"
            mock_tracer._get_amp_op_list.return_value = ([], [])
            mock_fw._dygraph_tracer.return_value = mock_tracer

            import paddle

            x = paddle.randn([2, 3])
            result = node.forward(x, is_first_fwd=True)
            self.assertTrue(node.use_recompute)

    def test_first_forward_preserves_rng_state(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = ScheduleNode(lambda x, **kw: x.sum(), name="test")

        with (
            patch("paddle.get_rng_state", return_value=MagicMock()) as mock_rng,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.get_rng_state_tracker"
            ) as mock_tracker,
            patch("numpy.random.get_state", return_value=MagicMock()),
            patch("random.getstate", return_value=MagicMock()),
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.custom_state_manager"
            ) as mock_csm,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.framework"
            ) as mock_fw,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.core"
            ) as mock_core,
        ):
            mock_tracker_inst = MagicMock()
            mock_tracker_inst.get_states_tracker.return_value = {}
            mock_tracker.return_value = mock_tracker_inst

            mock_csm.custom_get_state_func.return_value = None

            mock_tracer = MagicMock()
            mock_tracer._amp_level = mock_core.AmpLevel.O0
            mock_tracer._amp_dtype = "float32"
            mock_tracer._get_amp_op_list.return_value = ([], [])
            mock_fw._dygraph_tracer.return_value = mock_tracer

            import paddle

            x = paddle.randn([2, 3])
            result = node.forward(x, is_first_fwd=True)
            mock_rng.assert_called_once()
            self.assertTrue(node.use_recompute)
            self.assertIsNotNone(node.fw_rng_state)

    def test_first_forward_amp_level_o2(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = ScheduleNode(lambda x, **kw: x.sum(), name="test")

        with (
            patch("paddle.get_rng_state", return_value=MagicMock()),
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.get_rng_state_tracker"
            ) as mock_tracker,
            patch("numpy.random.get_state", return_value=MagicMock()),
            patch("random.getstate", return_value=MagicMock()),
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.custom_state_manager"
            ) as mock_csm,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.framework"
            ) as mock_fw,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.core"
            ) as mock_core,
        ):
            mock_tracker_inst = MagicMock()
            mock_tracker_inst.get_states_tracker.return_value = {}
            mock_tracker.return_value = mock_tracker_inst

            mock_csm.custom_get_state_func.return_value = None

            mock_tracer = MagicMock()
            mock_tracer._amp_level = mock_core.AmpLevel.O2
            mock_tracer._amp_dtype = "float16"
            mock_tracer._get_amp_op_list.return_value = ([], [])
            mock_fw._dygraph_tracer.return_value = mock_tracer

            import paddle

            x = paddle.randn([2, 3])
            result = node.forward(x, is_first_fwd=True)
            self.assertEqual(node.amp_level, "O2")
            self.assertEqual(node.amp_dtype, "float16")

    def test_first_forward_amp_level_o1(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = ScheduleNode(lambda x, **kw: x.sum(), name="test")

        with (
            patch("paddle.get_rng_state", return_value=MagicMock()),
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.get_rng_state_tracker"
            ) as mock_tracker,
            patch("numpy.random.get_state", return_value=MagicMock()),
            patch("random.getstate", return_value=MagicMock()),
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.custom_state_manager"
            ) as mock_csm,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.framework"
            ) as mock_fw,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.core"
            ) as mock_core,
        ):
            mock_tracker_inst = MagicMock()
            mock_tracker_inst.get_states_tracker.return_value = {}
            mock_tracker.return_value = mock_tracker_inst

            mock_csm.custom_get_state_func.return_value = None

            mock_tracer = MagicMock()
            mock_tracer._amp_level = mock_core.AmpLevel.O1
            mock_tracer._amp_dtype = "bfloat16"
            mock_tracer._get_amp_op_list.return_value = ([], [])
            mock_fw._dygraph_tracer.return_value = mock_tracer

            import paddle

            x = paddle.randn([2, 3])
            result = node.forward(x, is_first_fwd=True)
            self.assertEqual(node.amp_level, "O1")
            self.assertEqual(node.amp_dtype, "bfloat16")

    def test_first_forward_unsupported_amp_level(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = ScheduleNode(lambda x, **kw: x.sum(), name="test")

        with (
            patch("paddle.get_rng_state", return_value=MagicMock()),
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.get_rng_state_tracker"
            ) as mock_tracker,
            patch("numpy.random.get_state", return_value=MagicMock()),
            patch("random.getstate", return_value=MagicMock()),
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.custom_state_manager"
            ) as mock_csm,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.framework"
            ) as mock_fw,
            patch(
                "paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils.core"
            ) as mock_core,
        ):
            mock_tracker_inst = MagicMock()
            mock_tracker_inst.get_states_tracker.return_value = {}
            mock_tracker.return_value = mock_tracker_inst

            mock_csm.custom_get_state_func.return_value = None

            mock_tracer = MagicMock()
            mock_tracer._amp_level = 999  # Invalid level
            mock_tracer._amp_dtype = "float32"
            mock_tracer._get_amp_op_list.return_value = ([], [])
            mock_fw._dygraph_tracer.return_value = mock_tracer

            import paddle

            x = paddle.randn([2, 3])
            with self.assertRaises(ValueError):
                node.forward(x, is_first_fwd=True)


if __name__ == "__main__":
    unittest.main()
