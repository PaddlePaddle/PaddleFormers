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


# Tests for src/paddleformers.fleet/timers.py
# Additional tests for Timers class, _Timer, RuntimeTimer

import unittest
from unittest import mock


class TestTimerBasicOperations(unittest.TestCase):
    """Tests for _Timer basic operations."""

    def test_timer_creation(self):
        """Test timer creation with a name."""
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test_timer")
        self.assertEqual(timer.name, "test_timer")
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)

    def test_timer_start(self):
        """Test timer start sets started_ flag."""
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test")
        with mock.patch("paddle.device.synchronize"):
            with mock.patch("paddle.device.get_device", return_value="cpu"):
                timer.start()
        self.assertTrue(timer.started_)

    def test_timer_start_already_started_raises(self):
        """Test starting an already started timer raises."""
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test")
        timer.started_ = True
        with self.assertRaises(AssertionError):
            timer.start()

    def test_timer_stop(self):
        """Test timer stop clears started_ flag."""
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test")
        timer.started_ = True
        with mock.patch("paddle.device.synchronize"):
            with mock.patch("paddle.device.get_device", return_value="cpu"):
                timer.stop()
        self.assertFalse(timer.started_)
        self.assertGreaterEqual(timer.elapsed_, 0.0)

    def test_timer_stop_not_started_raises(self):
        """Test stopping a not-started timer raises."""
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test")
        with self.assertRaises(AssertionError):
            timer.stop()

    def test_timer_reset(self):
        """Test timer reset clears elapsed and started."""
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test")
        timer.elapsed_ = 10.0
        timer.started_ = True
        timer.reset()
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)


class TestTimerElapsed(unittest.TestCase):
    """Tests for _Timer.elapsed method."""

    def test_elapsed_resets_by_default(self):
        """Test elapsed resets elapsed time by default."""
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test")
        timer.elapsed_ = 5.0
        with mock.patch("paddle.device.get_device", return_value="cpu"):
            elapsed = timer.elapsed(reset=True)
        self.assertAlmostEqual(elapsed, 5.0)
        self.assertEqual(timer.elapsed_, 0.0)

    def test_elapsed_no_reset(self):
        """Test elapsed with reset=False preserves elapsed time."""
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test")
        timer.elapsed_ = 5.0
        with mock.patch("paddle.device.get_device", return_value="cpu"):
            elapsed = timer.elapsed(reset=False)
        self.assertAlmostEqual(elapsed, 5.0)
        self.assertAlmostEqual(timer.elapsed_, 5.0)

    def test_elapsed_while_running(self):
        """Test elapsed stops running timer, gets time, then restarts."""
        from paddleformers.fleet.timers import _Timer

        timer = _Timer("test")
        timer.started_ = True
        timer.elapsed_ = 3.0
        with mock.patch("paddle.device.get_device", return_value="cpu"):
            with mock.patch("paddle.device.synchronize"):
                elapsed = timer.elapsed(reset=True)
        # The elapsed() method restarts the timer if it was running,
        # so started_ should be True after elapsed() returns.
        self.assertTrue(timer.started_)
        # The returned elapsed time should be approximately the
        # previously accumulated time (3.0) plus any time from stop().
        self.assertGreaterEqual(elapsed, 3.0)


class TestTimersBasic(unittest.TestCase):
    """Tests for Timers class basic operations."""

    def test_timers_creation(self):
        """Test Timers creation starts with empty dict."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        self.assertEqual(len(timers.timers), 0)

    def test_timers_call_creates_timer(self):
        """Test calling Timers creates a new timer."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timer = timers("forward")
        self.assertEqual(timer.name, "forward")
        self.assertIn("forward", timers.timers)

    def test_timers_call_returns_existing(self):
        """Test calling Timers returns existing timer for same name."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        t1 = timers("forward")
        t2 = timers("forward")
        self.assertIs(t1, t2)

    def test_timers_call_use_event(self):
        """Test calling Timers with use_event=True on non-CUDA returns _Timer."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            timer = timers("test", use_event=True)
            self.assertIsInstance(timer, type(timers("test")))


class TestTimersInfo(unittest.TestCase):
    """Tests for Timers.info method."""

    def test_info_returns_dict(self):
        """Test info returns a dictionary."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timers("timer1")
        timers("timer1").elapsed_ = 1.0
        timers("timer2")
        timers("timer2").elapsed_ = 2.0

        with mock.patch("paddle.device.get_device", return_value="cpu"):
            result = timers.info(names=["timer1", "timer2"], normalizer=1.0)
        self.assertIsInstance(result, dict)
        self.assertIn("timer1", result)
        self.assertIn("timer2", result)

    def test_info_sorted_by_name(self):
        """Test info returns dict sorted by name."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timers("beta")
        timers("beta").elapsed_ = 1.0
        timers("alpha")
        timers("alpha").elapsed_ = 2.0

        with mock.patch("paddle.device.get_device", return_value="cpu"):
            result = timers.info(names=["beta", "alpha"], normalizer=1.0)
        keys = list(result.keys())
        self.assertEqual(keys, ["alpha", "beta"])

    def test_info_normalizer(self):
        """Test info applies normalizer correctly."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timers("test")
        timers("test").elapsed_ = 100.0

        with mock.patch("paddle.device.get_device", return_value="cpu"):
            result = timers.info(names=["test"], normalizer=10.0)
        # 100ms / 10 * 1000 = 10000ms
        self.assertAlmostEqual(result["test"], 10000.0)

    def test_info_normalizer_assertion(self):
        """Test info asserts normalizer > 0."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        with self.assertRaises(AssertionError):
            timers.info(names=["test"], normalizer=0.0)

    def test_info_empty_names(self):
        """Test info with empty names list."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        result = timers.info(names=[], normalizer=1.0)
        self.assertEqual(result, {})


class TestRuntimeTimer(unittest.TestCase):
    """Tests for RuntimeTimer class."""

    def test_runtime_timer_creation(self):
        """Test RuntimeTimer creation."""
        from paddleformers.fleet.timers import RuntimeTimer

        rt = RuntimeTimer()
        self.assertIsNotNone(rt.timer)

    def test_runtime_timer_start(self):
        """Test RuntimeTimer start sets name."""
        from paddleformers.fleet.timers import RuntimeTimer

        rt = RuntimeTimer()
        with mock.patch.object(rt.timer, "start"):
            rt.start("my_timer")
            self.assertEqual(rt.timer.name, "my_timer")

    def test_runtime_timer_stop(self):
        """Test RuntimeTimer stop calls timer stop."""
        from paddleformers.fleet.timers import RuntimeTimer

        rt = RuntimeTimer()
        with mock.patch.object(rt.timer, "stop"):
            rt.stop()

    def test_runtime_timer_log(self):
        """Test RuntimeTimer log prints and resets."""
        from paddleformers.fleet.timers import RuntimeTimer

        rt = RuntimeTimer()
        with mock.patch.object(rt.timer, "elapsed", return_value=1.5):
            with mock.patch("builtins.print") as mock_print:
                rt.log()
                mock_print.assert_called_once()
                output = mock_print.call_args[0][0]
                self.assertIn("1.50s", output)


class TestTimersLog(unittest.TestCase):
    """Tests for Timers.log method."""

    def test_log_prints_timer_info(self):
        """Test log prints timer information."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timers("test")
        timers("test").elapsed_ = 1.5

        with mock.patch("paddle.device.get_device", return_value="cpu"):
            with mock.patch("builtins.print") as mock_print:
                timers.log(names=["test"], normalizer=1.0)
                mock_print.assert_called_once()

    def test_log_sorted_by_time(self):
        """Test log sorts timers by time descending."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        timers("fast")
        timers("fast").elapsed_ = 0.5
        timers("slow")
        timers("slow").elapsed_ = 5.0

        with mock.patch("paddle.device.get_device", return_value="cpu"):
            with mock.patch("builtins.print") as mock_print:
                timers.log(names=["fast", "slow"], normalizer=1.0)
                output = mock_print.call_args[0][0]
                # slow should appear before fast (descending)
                slow_pos = output.find("slow")
                fast_pos = output.find("fast")
                self.assertLess(slow_pos, fast_pos)

    def test_log_normalizer_assertion(self):
        """Test log asserts normalizer > 0."""
        from paddleformers.fleet.timers import Timers

        timers = Timers()
        with self.assertRaises(AssertionError):
            timers.log(names=["test"], normalizer=-1.0)


if __name__ == "__main__":
    unittest.main()
