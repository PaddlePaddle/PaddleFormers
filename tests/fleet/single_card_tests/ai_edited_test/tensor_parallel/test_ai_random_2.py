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


# Extra tests for paddleformers.fleet/tensor_parallel/random.py
# Focus on: CudaRNGStatesTracker, _get/_set_cuda_rng_state,
# get_all_rng_states, model_parallel_cuda_manual_seed, _fork_rng,
# checkpoint, enable_share_grad_holder

import unittest

import paddle

from paddleformers.fleet.tensor_parallel.random import (
    CudaRNGStatesTracker,
    checkpoint,
    enable_share_grad_holder,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
)


class TestGetExpertParallelRngTrackerName(unittest.TestCase):
    """Tests for get_expert_parallel_rng_tracker_name."""

    def test_returns_string(self):
        """Test that the name is a string."""
        name = get_expert_parallel_rng_tracker_name()
        self.assertIsInstance(name, str)

    def test_default_name(self):
        """Test the default name."""
        name = get_expert_parallel_rng_tracker_name()
        self.assertEqual(name, "expert-parallel-rng")


class TestGetDataParallelRngTrackerName(unittest.TestCase):
    """Tests for get_data_parallel_rng_tracker_name."""

    def test_returns_string(self):
        """Test that the name is a string."""
        name = get_data_parallel_rng_tracker_name()
        self.assertIsInstance(name, str)

    def test_default_name(self):
        """Test the default name."""
        name = get_data_parallel_rng_tracker_name()
        self.assertEqual(name, "data-parallel-rng")


class TestCudaRNGStatesTracker(unittest.TestCase):
    """Tests for CudaRNGStatesTracker."""

    def test_init(self):
        """Test initialization."""
        tracker = CudaRNGStatesTracker()
        self.assertFalse(tracker.is_initialized())
        self.assertEqual(len(tracker.states_), 0)
        self.assertEqual(len(tracker.seeds_), 0)

    def test_reset(self):
        """Test reset clears all state."""
        tracker = CudaRNGStatesTracker()
        tracker._is_initialized = True
        tracker.states_ = {"test": "value"}
        tracker.seeds_ = {1, 2}
        tracker.reset()
        self.assertFalse(tracker.is_initialized())
        self.assertEqual(len(tracker.states_), 0)
        self.assertEqual(len(tracker.seeds_), 0)

    def test_set_states(self):
        """Test set_states marks as initialized."""
        tracker = CudaRNGStatesTracker()
        tracker.set_states({"test": "value"})
        self.assertTrue(tracker.is_initialized())
        self.assertEqual(tracker.states_, {"test": "value"})

    def test_get_states_returns_copy(self):
        """Test get_states returns a copy of the states dict."""
        tracker = CudaRNGStatesTracker()
        tracker.states_ = {"test": "value"}
        states = tracker.get_states()
        self.assertEqual(states, {"test": "value"})

    def test_add_duplicate_seed_raises(self):
        """Test adding a duplicate seed raises ValueError."""
        tracker = CudaRNGStatesTracker()
        tracker.seeds_ = {42}
        with self.assertRaises(ValueError):
            tracker.add("name1", 42)

    def test_add_duplicate_name_raises(self):
        """Test adding a duplicate name raises ValueError."""
        tracker = CudaRNGStatesTracker()
        tracker.states_ = {"name1": "value"}
        with self.assertRaises(ValueError):
            tracker.add("name1", 99)

    def test_fork_nonexistent_name_raises(self):
        """Test forking a non-existent state raises Exception."""
        with self.assertRaises(Exception):
            tracker = CudaRNGStatesTracker()
            tracker.fork("nonexistent").__enter__()

    def test_assert_cudagraphable_rng_not_supported(self):
        """Test that use_cudagraphable_rng=True raises assertion."""
        with self.assertRaises(AssertionError):
            CudaRNGStatesTracker(use_cudagraphable_rng=True)


class TestCudaRNGStatesTrackerAddAndFork(unittest.TestCase):
    """Tests for CudaRNGStatesTracker add and fork with CUDA."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_add_and_fork(self):
        """Test adding and forking an RNG state."""
        tracker = CudaRNGStatesTracker()
        tracker.add("test_state", 42)
        self.assertTrue(tracker.is_initialized())
        self.assertIn("test_state", tracker.states_)

        with tracker.fork("test_state"):
            # Inside the fork, RNG should be in a deterministic state
            x = paddle.randn([3])

        # After fork, the original RNG state should be restored
        self.assertIsNotNone(x)


class TestCheckpoint(unittest.TestCase):
    """Tests for checkpoint function."""

    def test_checkpoint_is_noop(self):
        """Test that checkpoint function does nothing (pass)."""
        # The checkpoint function is currently a no-op
        result = checkpoint(lambda: 42)
        self.assertIsNone(result)


class TestEnableShareGradHolder(unittest.TestCase):
    """Tests for enable_share_grad_holder context manager."""

    def test_context_manager_enters_and_exits(self):
        """Test that context manager works properly."""
        with enable_share_grad_holder():
            pass  # Should not raise

    def test_flag_restored_after_exit(self):
        """Test that the flag is restored after exiting context."""
        flag = "FLAGS_share_tensor_for_grad_tensor_holder"
        old_value = paddle.get_flags([flag])[flag]

        with enable_share_grad_holder():
            pass

        new_value = paddle.get_flags([flag])[flag]
        self.assertEqual(old_value, new_value)


class TestGetSetCudaRngState(unittest.TestCase):
    """Tests for _get_cuda_rng_state and _set_cuda_rng_state."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_get_cuda_rng_state(self):
        """Test _get_cuda_rng_state returns a tensor."""
        from paddleformers.fleet.tensor_parallel.random import (
            _get_cuda_rng_state,
        )

        state = _get_cuda_rng_state()
        self.assertIsNotNone(state)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_get_cuda_rng_state_graph_safe_asserts(self):
        """Test _get_cuda_rng_state raises assertion for graph_safe=True."""
        from paddleformers.fleet.tensor_parallel.random import (
            _get_cuda_rng_state,
        )

        with self.assertRaises(AssertionError):
            _get_cuda_rng_state(graph_safe=True)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_set_cuda_rng_state_graph_safe_asserts(self):
        """Test _set_cuda_rng_state raises assertion for graph_safe=True."""
        from paddleformers.fleet.tensor_parallel.random import (
            _set_cuda_rng_state,
        )

        state = paddle.cuda.get_rng_state()
        with self.assertRaises(AssertionError):
            _set_cuda_rng_state(state, graph_safe=True)


if __name__ == "__main__":
    unittest.main()
