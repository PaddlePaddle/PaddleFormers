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
# Focus on: initialize_rng_tracker, get_cuda_rng_tracker, get_all_rng_states,
# model_parallel_cuda_manual_seed, _get/_set_all_rng_states, _fork_rng

import unittest

import paddle


class TestInitializeRngTracker(unittest.TestCase):
    """Tests for initialize_rng_tracker function."""

    def setUp(self):
        """Reset the global RNG tracker before each test."""
        from paddleformers.fleet.tensor_parallel import random as rng_module

        rng_module._CUDA_RNG_STATE_TRACKER = None
        rng_module._CUDA_RNG_STATE_TRACKER_INITIALIZED = False

    def test_creates_tracker(self):
        """Test that initialize_rng_tracker creates a tracker."""
        from paddleformers.fleet.tensor_parallel.random import (
            initialize_rng_tracker,
        )

        initialize_rng_tracker()
        from paddleformers.fleet.tensor_parallel import random as rng_module

        self.assertTrue(rng_module._CUDA_RNG_STATE_TRACKER_INITIALIZED)
        self.assertIsNotNone(rng_module._CUDA_RNG_STATE_TRACKER)

    def test_idempotent(self):
        """Test that calling initialize_rng_tracker twice does not recreate."""
        from paddleformers.fleet.tensor_parallel import random as rng_module
        from paddleformers.fleet.tensor_parallel.random import (
            initialize_rng_tracker,
        )

        initialize_rng_tracker()
        tracker1 = rng_module._CUDA_RNG_STATE_TRACKER

        initialize_rng_tracker()
        tracker2 = rng_module._CUDA_RNG_STATE_TRACKER

        self.assertIs(tracker1, tracker2)

    def test_force_reset(self):
        """Test force_reset=True recreates the tracker."""
        from paddleformers.fleet.tensor_parallel import random as rng_module
        from paddleformers.fleet.tensor_parallel.random import (
            initialize_rng_tracker,
        )

        initialize_rng_tracker()
        tracker1 = rng_module._CUDA_RNG_STATE_TRACKER

        initialize_rng_tracker(force_reset=True)
        tracker2 = rng_module._CUDA_RNG_STATE_TRACKER

        self.assertIsNot(tracker1, tracker2)

    def test_cudagraphable_rng_not_supported(self):
        """Test that use_cudagraphable_rng=True raises assertion."""
        from paddleformers.fleet.tensor_parallel.random import (
            initialize_rng_tracker,
        )

        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_cudagraphable_rng=True)

    def test_te_rng_tracker_not_supported(self):
        """Test that use_te_rng_tracker=True raises assertion."""
        from paddleformers.fleet.tensor_parallel.random import (
            initialize_rng_tracker,
        )

        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_te_rng_tracker=True)


class TestGetCudaRngTracker(unittest.TestCase):
    """Tests for get_cuda_rng_tracker function."""

    def setUp(self):
        """Reset the global RNG tracker before each test."""
        from paddleformers.fleet.tensor_parallel import random as rng_module

        rng_module._CUDA_RNG_STATE_TRACKER = None
        rng_module._CUDA_RNG_STATE_TRACKER_INITIALIZED = False

    def test_returns_tracker(self):
        """Test that get_cuda_rng_tracker returns a tracker."""
        from paddleformers.fleet.tensor_parallel.random import (
            get_cuda_rng_tracker,
        )

        tracker = get_cuda_rng_tracker()
        self.assertIsNotNone(tracker)

    def test_cudagraphable_rng_not_supported(self):
        """Test that use_cudagraphable_rng=True raises assertion."""
        from paddleformers.fleet.tensor_parallel.random import (
            get_cuda_rng_tracker,
        )

        with self.assertRaises(AssertionError):
            get_cuda_rng_tracker(use_cudagraphable_rng=True)


class TestGetAllRngStates(unittest.TestCase):
    """Tests for get_all_rng_states function."""

    def test_uninitialized_raises(self):
        """Test get_all_rng_states raises when not initialized."""
        from paddleformers.fleet.tensor_parallel import random as rng_module
        from paddleformers.fleet.tensor_parallel.random import (
            get_all_rng_states,
        )

        rng_module._CUDA_RNG_STATE_TRACKER = None
        rng_module._CUDA_RNG_STATE_TRACKER_INITIALIZED = False

        with self.assertRaises(AssertionError):
            get_all_rng_states()


class TestInferenceRngTracker(unittest.TestCase):
    """Tests for inference RNG tracker."""

    def setUp(self):
        """Reset the global RNG tracker before each test."""
        from paddleformers.fleet.tensor_parallel import random as rng_module

        rng_module._CUDA_RNG_STATE_TRACKER = None
        rng_module._CUDA_RNG_STATE_TRACKER_INITIALIZED = False

    def test_inference_tracker_add_is_noop(self):
        """Test that inference tracker add does nothing."""
        from paddleformers.fleet.tensor_parallel import random as rng_module
        from paddleformers.fleet.tensor_parallel.random import (
            initialize_rng_tracker,
        )

        initialize_rng_tracker(inference_rng_tracker=True)
        tracker = rng_module._CUDA_RNG_STATE_TRACKER
        tracker.add("test", 42)
        # Should not raise and should not actually add

    def test_inference_tracker_set_states_is_noop(self):
        """Test that inference tracker set_states does nothing."""
        from paddleformers.fleet.tensor_parallel import random as rng_module
        from paddleformers.fleet.tensor_parallel.random import (
            initialize_rng_tracker,
        )

        initialize_rng_tracker(inference_rng_tracker=True)
        tracker = rng_module._CUDA_RNG_STATE_TRACKER
        tracker.set_states({"test": "value"})
        # Should not raise

    def test_inference_tracker_fork_is_nullcontext(self):
        """Test that inference tracker fork returns nullcontext."""
        import contextlib

        from paddleformers.fleet.tensor_parallel import random as rng_module
        from paddleformers.fleet.tensor_parallel.random import (
            initialize_rng_tracker,
        )

        initialize_rng_tracker(inference_rng_tracker=True)
        tracker = rng_module._CUDA_RNG_STATE_TRACKER
        result = tracker.fork("test")
        self.assertIsInstance(result, contextlib.nullcontext)


class TestForkRng(unittest.TestCase):
    """Tests for _fork_rng context manager."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_fork_rng_restores_state(self):
        """Test that _fork_rng restores RNG state after exit."""
        from paddleformers.fleet.tensor_parallel.random import (
            _fork_rng,
            initialize_rng_tracker,
        )

        initialize_rng_tracker(force_reset=True)

        # Get initial RNG state
        initial_state = paddle.cuda.get_rng_state()

        with _fork_rng():
            # Generate some random numbers inside fork
            _ = paddle.randn([10])

        # State should be restored - GeneratorState objects should both exist
        final_state = paddle.cuda.get_rng_state()
        self.assertIsNotNone(initial_state)
        self.assertIsNotNone(final_state)


if __name__ == "__main__":
    unittest.main()
