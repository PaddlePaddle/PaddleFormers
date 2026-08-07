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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import time
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import paddle


class TestTimer(unittest.TestCase):
    """Tests for the _Timer class in paddleformers.fleet.timers module."""

    def test_timer_init(self):
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test_timer")
        self.assertEqual(timer.name, "test_timer")
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)

    def test_timer_start(self):
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test_timer")
        timer.start()
        self.assertTrue(timer.started_)

    def test_timer_start_already_started_raises(self):
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test_timer")
        timer.start()
        with self.assertRaises(AssertionError):
            timer.start()

    def test_timer_stop(self):
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test_timer")
        timer.start()
        time.sleep(0.01)
        timer.stop()
        self.assertFalse(timer.started_)
        self.assertGreater(timer.elapsed_, 0.0)

    def test_timer_stop_not_started_raises(self):
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test_timer")
        with self.assertRaises(AssertionError):
            timer.stop()

    def test_timer_reset(self):
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test_timer")
        timer.start()
        time.sleep(0.01)
        timer.stop()
        self.assertGreater(timer.elapsed_, 0.0)
        timer.reset()
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)

    def test_timer_elapsed_with_reset(self):
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test_timer")
        timer.start()
        time.sleep(0.01)
        timer.stop()
        elapsed = timer.elapsed(reset=True)
        self.assertGreater(elapsed, 0.0)
        self.assertEqual(timer.elapsed_, 0.0)

    def test_timer_elapsed_without_reset(self):
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test_timer")
        timer.start()
        time.sleep(0.01)
        timer.stop()
        elapsed = timer.elapsed(reset=False)
        self.assertGreater(elapsed, 0.0)
        self.assertGreater(timer.elapsed_, 0.0)

    def test_timer_elapsed_while_running(self):
        """Test elapsed() stops the timer, returns value, then restarts."""
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test_timer")
        timer.start()
        time.sleep(0.01)
        elapsed = timer.elapsed(reset=False)
        self.assertGreater(elapsed, 0.0)
        # Timer should be running again after elapsed() call
        self.assertTrue(timer.started_)
        timer.stop()


class TestRuntimeTimer(unittest.TestCase):
    """Tests for the RuntimeTimer class in paddleformers.fleet.timers module."""

    def test_runtime_timer_init(self):
        from paddleformers.fleet.timers import RuntimeTimer

        rt = RuntimeTimer()
        self.assertIsNotNone(rt.timer)

    def test_runtime_timer_start(self):
        from paddleformers.fleet.timers import RuntimeTimer

        rt = RuntimeTimer()
        rt.start("my_operation")
        self.assertTrue(rt.timer.started_)
        self.assertEqual(rt.timer.name, "my_operation")

    def test_runtime_timer_stop(self):
        from paddleformers.fleet.timers import RuntimeTimer

        rt = RuntimeTimer()
        rt.start("my_operation")
        time.sleep(0.01)
        rt.stop()
        self.assertFalse(rt.timer.started_)

    def test_runtime_timer_log(self):
        from paddleformers.fleet.timers import RuntimeTimer

        rt = RuntimeTimer()
        rt.start("test_op")
        time.sleep(0.01)
        rt.stop()
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            rt.log()
            output = mock_out.getvalue()
        self.assertIn("[timelog]", output)
        self.assertIn("test_op", output)


class TestTimers(unittest.TestCase):
    """Tests for the Timers class in paddleformers.fleet.timers module."""

    def test_timers_init(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        self.assertEqual(timers.timers, {})

    def test_timers_call_creates_new_timer(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timer = timers("forward")
        self.assertIsNotNone(timer)
        self.assertEqual(timer.name, "forward")

    def test_timers_call_returns_existing_timer(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timer1 = timers("forward")
        timer2 = timers("forward")
        self.assertIs(timer1, timer2)

    def test_timers_call_multiple_names(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timers("forward")
        timers("backward")
        timers("optimizer")
        self.assertEqual(len(timers.timers), 3)

    def test_timers_log(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        t = timers("test_timer")
        t.start()
        time.sleep(0.01)
        t.stop()
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            timers.log(names=["test_timer"])
            output = mock_out.getvalue()
        self.assertIn("time (ms)", output)
        self.assertIn("test_timer", output)

    def test_timers_log_with_normalizer(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        t = timers("forward")
        t.start()
        time.sleep(0.01)
        t.stop()
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            timers.log(names=["forward"], normalizer=2.0)
            output = mock_out.getvalue()
        self.assertIn("forward", output)

    def test_timers_log_asserts_positive_normalizer(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timers("forward")
        with self.assertRaises(AssertionError):
            timers.log(names=["forward"], normalizer=0.0)

    def test_timers_log_asserts_negative_normalizer(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timers("forward")
        with self.assertRaises(AssertionError):
            timers.log(names=["forward"], normalizer=-1.0)

    def test_timers_info(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        t = timers("info_test")
        t.start()
        time.sleep(0.01)
        t.stop()
        result = timers.info(names=["info_test"])
        self.assertIsInstance(result, dict)
        self.assertIn("info_test", result)
        self.assertGreater(result["info_test"], 0.0)

    def test_timers_info_sorted_by_name(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timers("b_timer").start()
        timers("b_timer").stop()
        timers("a_timer").start()
        timers("a_timer").stop()
        result = timers.info(names=["b_timer", "a_timer"])
        keys = list(result.keys())
        self.assertEqual(keys, ["a_timer", "b_timer"])

    def test_timers_info_with_normalizer(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        t = timers("norm_test")
        t.start()
        time.sleep(0.01)
        t.stop()
        result = timers.info(names=["norm_test"], normalizer=5.0)
        self.assertIn("norm_test", result)

    def test_timers_info_asserts_positive_normalizer(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timers("t")
        with self.assertRaises(AssertionError):
            timers.info(names=["t"], normalizer=0.0)

    def test_timers_write_summary_writer(self):
        """Test Timers.write with a SummaryWriter mock."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        t = timers("write_test")
        t.start()
        time.sleep(0.01)
        t.stop()

        mock_writer = MagicMock()
        # Make isinstance check pass for SummaryWriter
        with patch("paddleformers.fleet.timers.SummaryWriter", type(mock_writer)):
            timers.write(names=["write_test"], writer=mock_writer, iteration=10)
        mock_writer.add_scalar.assert_called_once()

    def test_timers_write_asserts_positive_normalizer(self):
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timers("t")
        with self.assertRaises(AssertionError):
            timers.write(
                names=["t"], writer=MagicMock(), iteration=0, normalizer=0.0
            )

    def test_timers_call_with_use_event_no_cuda(self):
        """Test Timers call with use_event=True but no CUDA returns _Timer."""
        from paddleformers.fleet.timers import Timers, _GPUEventTimer, _Timer

        timers = Timers()
        timer = timers("event_test", use_event=True)
        # When CUDA is not available, _GPUEventTimer is aliased to _Timer
        # When CUDA is available but _GPUEventTimer imported successfully,
        # a _GPUEventTimer instance is created (which may not be _Timer subclass)
        if paddle.is_compiled_with_cuda() and _GPUEventTimer is not _Timer:
            # Real CUDA event timer is created
            self.assertIsNotNone(timer)
        else:
            self.assertIsInstance(timer, _Timer)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_timers_call_with_use_event_cuda(self):
        """Test Timers call with use_event=True on CUDA device."""
        from paddleformers.fleet.timers import Timers

        paddle.device.set_device("gpu:0")
        timers = Timers()
        timer = timers("gpu_event_test", use_event=True)
        self.assertIsNotNone(timer)

    def test_timers_log_multiple_timers_sorted(self):
        """Test that log sorts timers by descending time."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        t1 = timers("fast")
        t1.start()
        time.sleep(0.005)
        t1.stop()

        t2 = timers("slow")
        t2.start()
        time.sleep(0.02)
        t2.stop()

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            timers.log(names=["fast", "slow"])
            output = mock_out.getvalue()

        # slow should appear after fast (descending order)
        slow_pos = output.index("slow")
        fast_pos = output.index("fast")
        self.assertLess(slow_pos, fast_pos)
