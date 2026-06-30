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

import sys
import unittest
from unittest.mock import MagicMock, patch

from paddleformers.utils.profiler import ProfilerOptions, add_profiler_step


class TestProfilerOptions(unittest.TestCase):
    def test_default_options(self):
        opts = ProfilerOptions("batch_range=[10,20]")
        self.assertEqual(opts["batch_range"], [10, 20])
        self.assertEqual(opts["state"], "All")
        self.assertEqual(opts["sorted_key"], "total")
        self.assertEqual(opts["tracer_option"], "Default")
        self.assertEqual(opts["profile_path"], "/tmp/profile")
        self.assertTrue(opts["exit_on_finished"])
        self.assertTrue(opts["timer_only"])
        self.assertFalse(opts["record_shapes"])

    def test_parse_batch_range(self):
        opts = ProfilerOptions("batch_range=[50,60]")
        self.assertEqual(opts["batch_range"], [50, 60])

    def test_parse_state(self):
        opts = ProfilerOptions("state=GPU")
        self.assertEqual(opts["state"], "GPU")

    def test_parse_sorted_key(self):
        opts = ProfilerOptions("sorted_key=max")
        self.assertEqual(opts["sorted_key"], "max")

    def test_parse_tracer_option(self):
        opts = ProfilerOptions("tracer_option=OpDetail")
        self.assertEqual(opts["tracer_option"], "OpDetail")

    def test_parse_profile_path(self):
        opts = ProfilerOptions("profile_path=/tmp/my_profile")
        self.assertEqual(opts["profile_path"], "/tmp/my_profile")

    def test_parse_exit_on_finished_true(self):
        opts = ProfilerOptions("exit_on_finished=true")
        self.assertTrue(opts["exit_on_finished"])

    def test_parse_exit_on_finished_false(self):
        opts = ProfilerOptions("exit_on_finished=false")
        self.assertFalse(opts["exit_on_finished"])

    def test_parse_exit_on_finished_yes(self):
        opts = ProfilerOptions("exit_on_finished=yes")
        self.assertTrue(opts["exit_on_finished"])

    def test_parse_exit_on_finished_1(self):
        opts = ProfilerOptions("exit_on_finished=1")
        self.assertTrue(opts["exit_on_finished"])

    def test_parse_timer_only(self):
        opts = ProfilerOptions("timer_only=False")
        self.assertEqual(opts["timer_only"], "False")

    def test_parse_record_shapes(self):
        opts = ProfilerOptions("record_shapes=True")
        self.assertEqual(opts["record_shapes"], "True")

    def test_invalid_batch_range_ignored(self):
        # Invalid range where start >= end should be ignored
        opts = ProfilerOptions("batch_range=[60,50]")
        # Should keep default
        self.assertEqual(opts["batch_range"], [10, 20])

    def test_negative_start_ignored(self):
        opts = ProfilerOptions("batch_range=[-1,10]")
        self.assertEqual(opts["batch_range"], [10, 20])

    def test_invalid_option_name_raises(self):
        opts = ProfilerOptions("batch_range=[10,20]")
        with self.assertRaises(ValueError):
            opts["nonexistent_key"]

    def test_multiple_options(self):
        opts = ProfilerOptions("batch_range=[50,60];state=CPU;sorted_key=ave")
        self.assertEqual(opts["batch_range"], [50, 60])
        self.assertEqual(opts["state"], "CPU")
        self.assertEqual(opts["sorted_key"], "ave")

    def test_spaces_in_options(self):
        opts = ProfilerOptions("batch_range = [50, 60] ; state = GPU")
        self.assertEqual(opts["batch_range"], [50, 60])
        self.assertEqual(opts["state"], "GPU")


class TestAddProfilerStep(unittest.TestCase):
    def test_none_options_returns_immediately(self):
        # Should return without error
        add_profiler_step(None)

    @unittest.skip("paddle.profiler.Profiler patch does not work in CI")
    def test_creates_profiler_options(self):
        import paddleformers.utils.profiler as profiler_module

        # Reset global state
        profiler_module._profiler_options = None
        profiler_module._prof = None
        profiler_module._profiler_step_id = 0

        with patch.object(profiler_module, "_prof", None):
            with patch("paddle.profiler.Profiler") as mock_profiler_cls:
                mock_prof = MagicMock()
                mock_profiler_cls.return_value = mock_prof

                # This should create the profiler
                with patch.object(sys, "exit"):
                    add_profiler_step("batch_range=[1,3];profile_path=/tmp/test_profile")

                # Check that options were created
                self.assertIsNotNone(profiler_module._profiler_options)

        # Reset global state
        profiler_module._profiler_options = None
        profiler_module._prof = None
        profiler_module._profiler_step_id = 0


if __name__ == "__main__":
    unittest.main()
