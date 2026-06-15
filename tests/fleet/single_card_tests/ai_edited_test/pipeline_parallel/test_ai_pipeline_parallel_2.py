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


class TestGetAlignModeScale(unittest.TestCase):
    """Tests for _get_align_mode_scale function."""

    @patch("paddle.distributed.fleet.get_hybrid_communicate_group")
    def test_align_mode_scale(self, mock_get_hcg):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            _get_align_mode_scale,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_data_parallel_world_size.return_value = 2
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_get_hcg.return_value = mock_hcg

        result = _get_align_mode_scale()
        self.assertEqual(result, 2)

    @patch("paddle.distributed.fleet.get_hybrid_communicate_group")
    def test_align_mode_scale_zero(self, mock_get_hcg):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            _get_align_mode_scale,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_data_parallel_world_size.return_value = 0
        mock_hcg.get_sharding_parallel_world_size.return_value = 0
        mock_get_hcg.return_value = mock_hcg

        result = _get_align_mode_scale()
        self.assertEqual(result, 1)


class TestPipelineParallelMicroStepLocations(unittest.TestCase):
    """Tests for PipelineParallelMicroStepLocations enum."""

    def test_enum_values(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelMicroStepLocations,
        )

        self.assertEqual(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN.value,
            "forward_begin",
        )
        self.assertEqual(PipelineParallelMicroStepLocations.FORWARD_END.value, "forward_end")
        self.assertEqual(
            PipelineParallelMicroStepLocations.BACKWARD_BEGIN.value,
            "backward_begin",
        )
        self.assertEqual(
            PipelineParallelMicroStepLocations.BACKWARD_END.value,
            "backward_end",
        )

    def test_enum_members(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelMicroStepLocations,
        )

        members = list(PipelineParallelMicroStepLocations)
        self.assertEqual(len(members), 4)


class TestPipelineParallelMicroStepCallback(unittest.TestCase):
    """Tests for PipelineParallelMicroStepCallback."""

    def test_register_and_trigger_hook(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        callback = PipelineParallelMicroStepCallback()
        hook_calls = []

        def my_hook(**kwargs):
            hook_calls.append(kwargs)

        callback.register_hook(PipelineParallelMicroStepLocations.FORWARD_BEGIN, my_hook)
        callback.on_location(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN,
            step_id=0,
        )
        self.assertEqual(len(hook_calls), 1)
        self.assertEqual(hook_calls[0]["step_id"], 0)

    def test_register_invalid_location(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
        )

        callback = PipelineParallelMicroStepCallback()
        with self.assertRaises(AssertionError):
            callback.register_hook("invalid_location", lambda: None)

    def test_on_invalid_location(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
        )

        callback = PipelineParallelMicroStepCallback()
        with self.assertRaises(AssertionError):
            callback.on_location("invalid_location")

    def test_multiple_hooks(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        callback = PipelineParallelMicroStepCallback()
        results = []

        callback.register_hook(
            PipelineParallelMicroStepLocations.BACKWARD_END,
            lambda **kw: results.append(1),
        )
        callback.register_hook(
            PipelineParallelMicroStepLocations.BACKWARD_END,
            lambda **kw: results.append(2),
        )
        callback.on_location(PipelineParallelMicroStepLocations.BACKWARD_END)
        self.assertEqual(results, [1, 2])

    def test_hooks_per_location_separate(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        callback = PipelineParallelMicroStepCallback()
        results = []

        callback.register_hook(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN,
            lambda **kw: results.append("fwd_begin"),
        )
        callback.register_hook(
            PipelineParallelMicroStepLocations.FORWARD_END,
            lambda **kw: results.append("fwd_end"),
        )

        callback.on_location(PipelineParallelMicroStepLocations.FORWARD_BEGIN)
        self.assertEqual(results, ["fwd_begin"])

        callback.on_location(PipelineParallelMicroStepLocations.FORWARD_END)
        self.assertEqual(results, ["fwd_begin", "fwd_end"])


class TestFakeMicroDataset(unittest.TestCase):
    """Tests for FakeMicroDataset."""


class TestPipelineParallelCallbacksGlobal(unittest.TestCase):
    """Tests for the global pipeline_parallel_callbacks_ instance."""

    def test_global_callbacks_exist(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            pipeline_parallel_callbacks_,
        )

        self.assertIsNotNone(pipeline_parallel_callbacks_)

    def test_global_callbacks_has_all_locations(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelMicroStepLocations,
            pipeline_parallel_callbacks_,
        )

        for location in PipelineParallelMicroStepLocations:
            self.assertIn(location, pipeline_parallel_callbacks_.hooks)


if __name__ == "__main__":
    unittest.main()
