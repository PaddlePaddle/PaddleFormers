# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import unittest
from unittest.mock import MagicMock, patch

from paddleformers.trainer.plugins.timer import (
    RuntimeTimer,
    Timers,
    _Timer,
    disable_timers,
    get_timers,
    set_timers,
)


class TestTimerBasic(unittest.TestCase):
    """Tests for the _Timer class."""

    def test_init(self):
        timer = _Timer("test_timer")
        self.assertEqual(timer.name, "test_timer")
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)
        self.assertIsNotNone(timer.start_time)

    def test_start_and_stop(self):
        timer = _Timer("test")
        with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
            timer.start()
            self.assertTrue(timer.started_)
            timer.stop()
            self.assertFalse(timer.started_)
            self.assertGreater(timer.elapsed_, 0.0)

    def test_start_twice_raises(self):
        timer = _Timer("test")
        with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
            timer.start()
            with self.assertRaises(AssertionError):
                timer.start()

    def test_stop_without_start_raises(self):
        timer = _Timer("test")
        with self.assertRaises(AssertionError):
            timer.stop()

    def test_reset(self):
        timer = _Timer("test")
        with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
            timer.start()
            timer.stop()
            self.assertGreater(timer.elapsed_, 0.0)
            timer.reset()
            self.assertEqual(timer.elapsed_, 0.0)
            self.assertFalse(timer.started_)

    def test_elapsed_with_reset(self):
        timer = _Timer("test")
        with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
            timer.start()
            timer.stop()
            elapsed = timer.elapsed(reset=True)
            self.assertGreater(elapsed, 0.0)
            self.assertEqual(timer.elapsed_, 0.0)

    def test_elapsed_without_reset(self):
        timer = _Timer("test")
        with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
            timer.start()
            timer.stop()
            elapsed = timer.elapsed(reset=False)
            self.assertGreater(elapsed, 0.0)
            self.assertGreater(timer.elapsed_, 0.0)

    def test_elapsed_while_running(self):
        timer = _Timer("test")
        with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
            timer.start()
            elapsed = timer.elapsed(reset=True)
            self.assertGreater(elapsed, 0.0)
            # After elapsed with reset, timer should still be running
            self.assertTrue(timer.started_)

    def test_elapsed_while_running_no_reset(self):
        timer = _Timer("test")
        with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
            timer.start()
            elapsed = timer.elapsed(reset=False)
            self.assertGreater(elapsed, 0.0)
            self.assertTrue(timer.started_)
            self.assertGreater(timer.elapsed_, 0.0)

    def test_cpu_device_no_synchronize(self):
        timer = _Timer("cpu_test")
        with patch("paddle.device.get_device", return_value="cpu"):
            timer.start()
            self.assertTrue(timer.started_)
            timer.stop()
            self.assertFalse(timer.started_)


class TestRuntimeTimer(unittest.TestCase):
    """Tests for the RuntimeTimer class."""

    def test_init(self):
        rt = RuntimeTimer("runtime_test")
        self.assertIsNotNone(rt.timer)

    def test_start_and_stop(self):
        rt = RuntimeTimer("rt")
        with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
            rt.start("phase1")
            self.assertEqual(rt.timer.name, "phase1")
            rt.stop()

    def test_log(self):
        rt = RuntimeTimer("rt")
        with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
            rt.start("test_phase")
            rt.stop()
            result = rt.log()
            self.assertIn("test_phase", result)
            self.assertIn("timelog", result)
            # Timer should be reset after log
            self.assertEqual(rt.timer.elapsed_, 0.0)

    def test_log_while_running(self):
        rt = RuntimeTimer("rt")
        with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
            rt.start("running_phase")
            # Don't stop - log should handle running timer
            result = rt.log()
            self.assertIn("running_phase", result)
            self.assertFalse(rt.timer.started_)


class TestTimers(unittest.TestCase):
    """Tests for the Timers class."""

    def test_init(self):
        timers = Timers()
        self.assertEqual(timers.timers, {})

    def test_call_creates_timer(self):
        timers = Timers()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            timer = timers("test_timer")
        self.assertIsNotNone(timer)
        self.assertEqual(timer.name, "test_timer")

    def test_call_returns_existing_timer(self):
        timers = Timers()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            timer1 = timers("shared_timer")
            timer2 = timers("shared_timer")
        self.assertIs(timer1, timer2)

    def test_call_with_event_timer(self):
        timers = Timers()
        with patch("paddle.is_compiled_with_cuda", return_value=True), patch(
            "paddleformers.trainer.plugins.timer._GPUEventTimer", _Timer
        ):
            timer = timers("event_timer", use_event=True)
        self.assertIsNotNone(timer)
        self.assertEqual(timer.name, "event_timer")

    def test_call_type_mismatch_raises(self):
        """Test that requesting different timer type for existing name raises."""
        timers = Timers()
        # Create a _Timer first
        timer1 = timers("my_timer")
        self.assertIsInstance(timer1, _Timer)
        # Now try to get the same name with use_event=True when CUDA is available
        # This should raise because the existing timer is _Timer, not _GPUEventTimer
        with patch("paddle.is_compiled_with_cuda", return_value=True):
            # When _GPUEventTimer is different from _Timer (at module load time),
            # requesting a different type for the same name raises AssertionError
            import paddleformers.trainer.plugins.timer as timer_module

            original = timer_module._GPUEventTimer
            try:
                # Create a distinct class to force type mismatch
                class FakeGPUEventTimer(_Timer):
                    pass

                timer_module._GPUEventTimer = FakeGPUEventTimer
                with self.assertRaises(AssertionError):
                    timers("my_timer", use_event=True)
            finally:
                timer_module._GPUEventTimer = original

    def test_write(self):
        timers = Timers()
        mock_writer = MagicMock()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            timer = timers("write_test")
            with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
                timer.start()
                timer.stop()
            timers.write(["write_test"], mock_writer, iteration=5, normalizer=1.0)
        mock_writer.add_scalar.assert_called_once()
        args = mock_writer.add_scalar.call_args
        self.assertIn("write_test", args[0][0])

    def test_write_with_normalizer(self):
        timers = Timers()
        mock_writer = MagicMock()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            timer = timers("norm_test")
            with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
                timer.start()
                timer.stop()
            timers.write(["norm_test"], mock_writer, iteration=1, normalizer=2.0)
        mock_writer.add_scalar.assert_called_once()

    def test_write_asserts_positive_normalizer(self):
        timers = Timers()
        mock_writer = MagicMock()
        with self.assertRaises(AssertionError):
            timers.write([], mock_writer, iteration=1, normalizer=0.0)

    def test_write_asserts_negative_normalizer(self):
        timers = Timers()
        mock_writer = MagicMock()
        with self.assertRaises(AssertionError):
            timers.write([], mock_writer, iteration=1, normalizer=-1.0)

    def test_log_single_timer(self):
        timers = Timers()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            timer = timers("log_test")
            with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
                timer.start()
                timer.stop()
            result = timers.log(["log_test"], normalizer=1.0)
        self.assertIn("log_test", result)
        self.assertIn("time (ms)", result)

    def test_log_multiple_timers_sorted(self):
        timers = Timers()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            for name in ["timer_a", "timer_b", "timer_c"]:
                timer = timers(name)
                with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
                    timer.start()
                    timer.stop()
            result = timers.log(["timer_a", "timer_b", "timer_c"], normalizer=1.0)
        # Should contain all timers
        self.assertIn("timer_a", result)
        self.assertIn("timer_b", result)
        self.assertIn("timer_c", result)

    def test_log_asserts_positive_normalizer(self):
        timers = Timers()
        with self.assertRaises(AssertionError):
            timers.log([], normalizer=0.0)

    def test_info(self):
        timers = Timers()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            timer = timers("info_test")
            with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
                timer.start()
                timer.stop()
            result = timers.info(["info_test"], normalizer=1.0)
        self.assertIsInstance(result, dict)
        self.assertIn("info_test", result)

    def test_info_multiple_timers_sorted_by_name(self):
        timers = Timers()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            for name in ["charlie", "alpha", "bravo"]:
                timer = timers(name)
                with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
                    timer.start()
                    timer.stop()
            result = timers.info(["charlie", "alpha", "bravo"], normalizer=1.0)
        keys = list(result.keys())
        # Sorted by key in reverse=False means ascending
        self.assertEqual(keys, sorted(keys))

    def test_info_asserts_positive_normalizer(self):
        timers = Timers()
        with self.assertRaises(AssertionError):
            timers.info([], normalizer=-1.0)

    def test_info_with_reset(self):
        timers = Timers()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            timer = timers("reset_info")
            with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
                timer.start()
                timer.stop()
            result = timers.info(["reset_info"], normalizer=1.0, reset=True)
        self.assertIsInstance(result, dict)

    def test_write_no_reset(self):
        timers = Timers()
        mock_writer = MagicMock()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            timer = timers("no_reset")
            with patch("paddle.device.get_device", return_value="gpu:0"), patch("paddle.device.synchronize"):
                timer.start()
                timer.stop()
                timer.elapsed_
            timers.write(["no_reset"], mock_writer, iteration=1, normalizer=1.0, reset=False)
            # After write with reset=False, timer should still have time
            self.assertGreater(timer.elapsed_, 0.0)


class TestGlobalTimers(unittest.TestCase):
    """Tests for global timer functions."""

    def test_get_timers_initial_none(self):
        # Save and restore global state
        import paddleformers.trainer.plugins.timer as timer_module

        original = timer_module._GLOBAL_TIMERS
        try:
            timer_module._GLOBAL_TIMERS = None
            result = get_timers()
            self.assertIsNone(result)
        finally:
            timer_module._GLOBAL_TIMERS = original

    def test_set_timers(self):
        import paddleformers.trainer.plugins.timer as timer_module

        original = timer_module._GLOBAL_TIMERS
        try:
            set_timers()
            result = get_timers()
            self.assertIsInstance(result, Timers)
        finally:
            timer_module._GLOBAL_TIMERS = original

    def test_disable_timers(self):
        import paddleformers.trainer.plugins.timer as timer_module

        original = timer_module._GLOBAL_TIMERS
        try:
            set_timers()
            disable_timers()
            result = get_timers()
            self.assertIsNone(result)
        finally:
            timer_module._GLOBAL_TIMERS = original


class TestTimerGPUTimerFallback(unittest.TestCase):
    """Tests for _GPUEventTimer fallback to _Timer."""

    def test_gpu_event_timer_is_timer_when_none(self):
        import paddleformers.trainer.plugins.timer as timer_module

        # When _GPUEventTimer is None (import failed), it falls back to _Timer
        # This is already handled at module level
        self.assertIsNotNone(timer_module._GPUEventTimer)


if __name__ == "__main__":
    unittest.main()
