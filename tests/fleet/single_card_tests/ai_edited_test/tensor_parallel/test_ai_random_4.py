# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
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
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.tensor_parallel.random import (
    CudaRNGStatesTracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
)


class TestCudaRNGStatesTrackerInit(unittest.TestCase):
    """Tests for CudaRNGStatesTracker initialization."""

    def test_init_resets_state(self):
        """CudaRNGStatesTracker should start in reset state."""
        tracker = CudaRNGStatesTracker()
        self.assertFalse(tracker.is_initialized())
        self.assertEqual(tracker.states_, {})
        self.assertEqual(tracker.seeds_, set())

    def test_use_cudagraphable_rng_must_be_false(self):
        """CudaRNGStatesTracker should reject use_cudagraphable_rng=True."""
        with self.assertRaises(AssertionError):
            CudaRNGStatesTracker(use_cudagraphable_rng=True)


class TestCudaRNGStatesTrackerReset(unittest.TestCase):
    """Tests for CudaRNGStatesTracker.reset."""

    def test_reset_clears_state(self):
        """reset should clear all tracked states."""
        tracker = CudaRNGStatesTracker()
        tracker._is_initialized = True
        tracker.states_ = {"test": MagicMock()}
        tracker.seeds_ = {42}
        tracker.reset()
        self.assertFalse(tracker.is_initialized())
        self.assertEqual(tracker.states_, {})
        self.assertEqual(tracker.seeds_, set())


class TestCudaRNGStatesTrackerGetSetStates(unittest.TestCase):
    """Tests for CudaRNGStatesTracker get_states/set_states."""

    def test_get_states_returns_copy(self):
        """get_states should return a copy of the states dict."""
        tracker = CudaRNGStatesTracker()
        tracker.states_ = {"test": "value"}
        result = tracker.get_states()
        self.assertEqual(result, {"test": "value"})
        # Should be a new dict
        self.assertIsNot(result, tracker.states_)

    def test_set_states_marks_initialized(self):
        """set_states should mark tracker as initialized."""
        tracker = CudaRNGStatesTracker()
        tracker.set_states({"test": "value"})
        self.assertTrue(tracker.is_initialized())
        self.assertEqual(tracker.states_, {"test": "value"})


class TestCudaRNGStatesTrackerAdd(unittest.TestCase):
    """Tests for CudaRNGStatesTracker.add."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_add_sets_initialized(self):
        """add should mark tracker as initialized."""
        tracker = CudaRNGStatesTracker()
        tracker.add("test_state", 42)
        self.assertTrue(tracker.is_initialized())

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_add_tracks_seed(self):
        """add should track the seed."""
        tracker = CudaRNGStatesTracker()
        tracker.add("test_state", 42)
        self.assertIn(42, tracker.seeds_)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_add_rejects_duplicate_seed(self):
        """add should reject duplicate seeds."""
        tracker = CudaRNGStatesTracker()
        tracker.add("state1", 42)
        with self.assertRaises(ValueError):
            tracker.add("state2", 42)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_add_rejects_duplicate_name(self):
        """add should reject duplicate state names."""
        tracker = CudaRNGStatesTracker()
        tracker.add("test_state", 42)
        with self.assertRaises(ValueError):
            tracker.add("test_state", 43)


class TestCudaRNGStatesTrackerFork(unittest.TestCase):
    """Tests for CudaRNGStatesTracker.fork."""

    def test_fork_raises_for_unknown_name(self):
        """fork should raise when state name is not tracked."""
        tracker = CudaRNGStatesTracker()
        # fork is a contextmanager, exception is raised on __enter__
        with self.assertRaises(Exception), tracker.fork("unknown_state"):
            pass


class TestGetTrackerNames(unittest.TestCase):
    """Tests for tracker name getter functions."""

    def test_get_expert_parallel_rng_tracker_name(self):
        """Should return the expert parallel tracker name."""
        name = get_expert_parallel_rng_tracker_name()
        self.assertIsInstance(name, str)
        self.assertEqual(name, "expert-parallel-rng")

    def test_get_data_parallel_rng_tracker_name(self):
        """Should return the data parallel tracker name."""
        name = get_data_parallel_rng_tracker_name()
        self.assertIsInstance(name, str)
        self.assertEqual(name, "data-parallel-rng")


if __name__ == "__main__":
    unittest.main()
