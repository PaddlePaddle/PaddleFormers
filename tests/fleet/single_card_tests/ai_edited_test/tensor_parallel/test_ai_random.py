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

import paddle

from paddleformers.fleet.tensor_parallel.random import (
    _MODEL_PARALLEL_RNG_TRACKER_NAME,
    CudaRNGStatesTracker,
    _fork_rng,
    _get_cuda_rng_state,
    _set_cuda_rng_state,
    checkpoint,
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)


class TestCudaRNGStatesTrackerInit(unittest.TestCase):
    """Tests for CudaRNGStatesTracker initialization."""

    def test_init_default(self):
        """Test default initialization."""
        tracker = CudaRNGStatesTracker()
        self.assertFalse(tracker._is_initialized)
        self.assertEqual(tracker.states_, {})
        self.assertEqual(tracker.seeds_, set())
        self.assertFalse(tracker.use_cudagraphable_rng)
        self.assertFalse(tracker.is_inference_rng_tracker)

    def test_init_cudagraphable_raises(self):
        """Test that use_cudagraphable_rng=True raises."""
        with self.assertRaises(AssertionError):
            CudaRNGStatesTracker(use_cudagraphable_rng=True)

    def test_init_is_inference(self):
        """Test inference tracker flag is set."""
        tracker = CudaRNGStatesTracker(is_inference_rng_tracker=True)
        self.assertTrue(tracker.is_inference_rng_tracker)


class TestCudaRNGStatesTrackerReset(unittest.TestCase):
    """Tests for CudaRNGStatesTracker.reset method."""

    def test_reset_clears_state(self):
        """Test reset clears all state."""
        tracker = CudaRNGStatesTracker()
        tracker._is_initialized = True
        tracker.states_ = {"a": MagicMock()}
        tracker.seeds_ = {42}
        tracker.reset()
        self.assertFalse(tracker._is_initialized)
        self.assertEqual(tracker.states_, {})
        self.assertEqual(tracker.seeds_, set())


class TestCudaRNGStatesTrackerIsInitialized(unittest.TestCase):
    """Tests for CudaRNGStatesTracker.is_initialized method."""

    def test_not_initialized(self):
        """Test is_initialized returns False initially."""
        tracker = CudaRNGStatesTracker()
        self.assertFalse(tracker.is_initialized())


class TestCudaRNGStatesTrackerGetStates(unittest.TestCase):
    """Tests for CudaRNGStatesTracker.get_states method."""

    def test_get_states_empty(self):
        """Test get_states returns empty dict when no states."""
        tracker = CudaRNGStatesTracker()
        states = tracker.get_states()
        self.assertEqual(states, {})

    def test_get_states_returns_copy(self):
        """Test get_states returns a copy, not the original dict."""
        tracker = CudaRNGStatesTracker()
        tracker.states_ = {"a": MagicMock()}
        states = tracker.get_states()
        self.assertIsNot(states, tracker.states_)
        self.assertIn("a", states)


class TestCudaRNGStatesTrackerSetStates(unittest.TestCase):
    """Tests for CudaRNGStatesTracker.set_states method."""

    def test_set_states_initializes(self):
        """Test set_states sets _is_initialized to True."""
        tracker = CudaRNGStatesTracker()
        self.assertFalse(tracker._is_initialized)
        tracker.set_states({"a": MagicMock()})
        self.assertTrue(tracker._is_initialized)


class TestCudaRNGStatesTrackerAdd(unittest.TestCase):
    """Tests for CudaRNGStatesTracker.add method."""

    @patch("paddleformers.fleet.tensor_parallel.random.paddle.cuda.get_rng_state")
    @patch("paddleformers.fleet.tensor_parallel.random.paddle.cuda.manual_seed")
    @patch("paddleformers.fleet.tensor_parallel.random._set_cuda_rng_state")
    def test_add_new_state(self, mock_set_rng, mock_manual_seed, mock_get_rng):
        """Test adding a new RNG state."""
        mock_get_rng.return_value = MagicMock()
        tracker = CudaRNGStatesTracker()
        tracker.add("test", 42)
        self.assertTrue(tracker._is_initialized)
        self.assertIn("test", tracker.states_)
        self.assertIn(42, tracker.seeds_)

    @patch("paddleformers.fleet.tensor_parallel.random.paddle.cuda.get_rng_state")
    @patch("paddleformers.fleet.tensor_parallel.random.paddle.cuda.manual_seed")
    @patch("paddleformers.fleet.tensor_parallel.random._set_cuda_rng_state")
    def test_add_duplicate_seed_raises(self, mock_set_rng, mock_manual_seed, mock_get_rng):
        """Test adding a duplicate seed raises ValueError."""
        mock_get_rng.return_value = MagicMock()
        tracker = CudaRNGStatesTracker()
        tracker.add("test1", 42)
        with self.assertRaises(ValueError) as ctx:
            tracker.add("test2", 42)
        self.assertIn("seed 42 already exists", str(ctx.exception))

    @patch("paddleformers.fleet.tensor_parallel.random.paddle.cuda.get_rng_state")
    @patch("paddleformers.fleet.tensor_parallel.random.paddle.cuda.manual_seed")
    @patch("paddleformers.fleet.tensor_parallel.random._set_cuda_rng_state")
    def test_add_duplicate_name_raises(self, mock_set_rng, mock_manual_seed, mock_get_rng):
        """Test adding a duplicate name raises ValueError."""
        mock_get_rng.return_value = MagicMock()
        tracker = CudaRNGStatesTracker()
        tracker.add("test", 42)
        with self.assertRaises(ValueError) as ctx:
            tracker.add("test", 43)
        self.assertIn("cuda rng state test already exists", str(ctx.exception))


class TestCudaRNGStatesTrackerFork(unittest.TestCase):
    """Tests for CudaRNGStatesTracker.fork context manager."""

    @patch("paddleformers.fleet.tensor_parallel.random._set_cuda_rng_state")
    @patch("paddleformers.fleet.tensor_parallel.random._get_cuda_rng_state")
    @patch("paddleformers.fleet.tensor_parallel.random.paddle.get_rng_state")
    def test_fork_unknown_name_raises(self, mock_cpu_get, mock_get, mock_set):
        """Test fork raises for unknown RNG state name."""
        tracker = CudaRNGStatesTracker()
        with self.assertRaises(Exception) as ctx:  # noqa: SIM117
            with tracker.fork("unknown"):
                pass
        self.assertIn("not added", str(ctx.exception))

    @patch("paddleformers.fleet.tensor_parallel.random._set_cuda_rng_state")
    @patch("paddleformers.fleet.tensor_parallel.random._get_cuda_rng_state")
    @patch("paddleformers.fleet.tensor_parallel.random.paddle.get_rng_state")
    def test_fork_restores_state(self, mock_cpu_get, mock_get, mock_set):
        """Test fork restores original CUDA RNG state after context."""
        tracker = CudaRNGStatesTracker()
        mock_state = MagicMock()
        tracker.states_ = {"test": mock_state}
        mock_get.return_value = MagicMock()
        mock_cpu_get.return_value = MagicMock()
        with tracker.fork("test"):
            pass
        # After context, _set_cuda_rng_state should be called to restore
        self.assertTrue(mock_set.called)


class TestGetCudaRNGState(unittest.TestCase):
    """Tests for _get_cuda_rng_state function."""

    def test_graph_safe_raises(self):
        """Test that graph_safe=True raises AssertionError."""
        with self.assertRaises(AssertionError):
            _get_cuda_rng_state(graph_safe=True)

    @patch("paddleformers.fleet.tensor_parallel.random.paddle.cuda.get_rng_state")
    def test_normal_call(self, mock_get):
        """Test normal call delegates to paddle.cuda.get_rng_state."""
        mock_get.return_value = MagicMock()
        result = _get_cuda_rng_state()
        self.assertIsNotNone(result)


class TestSetCudaRNGState(unittest.TestCase):
    """Tests for _set_cuda_rng_state function."""

    def test_graph_safe_raises(self):
        """Test that graph_safe=True raises AssertionError."""
        with self.assertRaises(AssertionError):
            _set_cuda_rng_state(MagicMock(), graph_safe=True)

    @patch("paddleformers.fleet.tensor_parallel.random.paddle.cuda.set_rng_state")
    def test_normal_call(self, mock_set):
        """Test normal call delegates to paddle.cuda.set_rng_state."""
        state = MagicMock()
        _set_cuda_rng_state(state)
        mock_set.assert_called_once()


class TestInitializeRNGTracker(unittest.TestCase):
    """Tests for initialize_rng_tracker function."""

    def test_cudagraphable_raises(self):
        """Test that use_cudagraphable_rng=True raises."""
        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_cudagraphable_rng=True)

    def test_te_rng_tracker_raises(self):
        """Test that use_te_rng_tracker=True raises."""
        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_te_rng_tracker=True)

    @patch("paddleformers.fleet.tensor_parallel.random._CUDA_RNG_STATE_TRACKER", None)
    @patch(
        "paddleformers.fleet.tensor_parallel.random._CUDA_RNG_STATE_TRACKER_INITIALIZED",
        False,
    )
    def test_initializes_tracker(self):
        """Test that initialize_rng_tracker creates a tracker."""
        # Need to manipulate globals through the module
        import paddleformers.fleet.tensor_parallel.random as random_mod

        original_tracker = getattr(random_mod, "_CUDA_RNG_STATE_TRACKER", None)
        original_init = getattr(random_mod, "_CUDA_RNG_STATE_TRACKER_INITIALIZED", False)
        random_mod._CUDA_RNG_STATE_TRACKER = None
        random_mod._CUDA_RNG_STATE_TRACKER_INITIALIZED = False
        try:
            initialize_rng_tracker()
            self.assertIsNotNone(random_mod._CUDA_RNG_STATE_TRACKER)
            self.assertTrue(random_mod._CUDA_RNG_STATE_TRACKER_INITIALIZED)
        finally:
            random_mod._CUDA_RNG_STATE_TRACKER = original_tracker
            random_mod._CUDA_RNG_STATE_TRACKER_INITIALIZED = original_init


class TestGetCudaRNGTracker(unittest.TestCase):
    """Tests for get_cuda_rng_tracker function."""

    def test_cudagraphable_raises(self):
        """Test that use_cudagraphable_rng=True raises."""
        with self.assertRaises(AssertionError):
            get_cuda_rng_tracker(use_cudagraphable_rng=True)

    def test_te_rng_tracker_raises(self):
        """Test that use_te_rng_tracker=True raises."""
        with self.assertRaises(AssertionError):
            get_cuda_rng_tracker(use_te_rng_tracker=True)


class TestTrackerNames(unittest.TestCase):
    """Tests for RNG tracker name getters."""

    def test_model_parallel_name(self):
        """Test model parallel tracker name."""
        self.assertEqual(_MODEL_PARALLEL_RNG_TRACKER_NAME, "model-parallel-rng")

    def test_expert_parallel_name(self):
        """Test expert parallel tracker name getter."""
        self.assertEqual(get_expert_parallel_rng_tracker_name(), "expert-parallel-rng")

    def test_data_parallel_name(self):
        """Test data parallel tracker name getter."""
        self.assertEqual(get_data_parallel_rng_tracker_name(), "data-parallel-rng")


class TestModelParallelCudaManualSeed(unittest.TestCase):
    """Tests for model_parallel_cuda_manual_seed function."""

    def test_te_rng_tracker_raises(self):
        """Test that te_rng_tracker=True raises."""
        with self.assertRaises(AssertionError):
            model_parallel_cuda_manual_seed(42, te_rng_tracker=True)

    def test_cudagraphable_raises(self):
        """Test that use_cudagraphable_rng=True raises."""
        with self.assertRaises(AssertionError):
            model_parallel_cuda_manual_seed(42, use_cudagraphable_rng=True)

    @patch(
        "paddleformers.fleet.tensor_parallel.random.get_expert_tensor_parallel_rank",
        return_value=0,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.random.get_expert_model_parallel_rank",
        return_value=0,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.random.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch("paddleformers.fleet.tensor_parallel.random.initialize_rng_tracker")
    @patch("paddleformers.fleet.tensor_parallel.random.paddle.cuda.manual_seed")
    def test_manual_seed_calls(
        self,
        mock_manual_seed,
        mock_init,
        mock_tp_rank,
        mock_ep_rank,
        mock_etp_rank,
    ):
        """Test model_parallel_cuda_manual_seed sets seeds properly."""
        import paddleformers.fleet.tensor_parallel.random as random_mod

        original_tracker = getattr(random_mod, "_CUDA_RNG_STATE_TRACKER", None)
        mock_tracker = MagicMock()
        random_mod._CUDA_RNG_STATE_TRACKER = mock_tracker
        try:
            model_parallel_cuda_manual_seed(42)
            mock_init.assert_called()
            mock_tracker.reset.assert_called()
            # Check add is called three times for data, model, expert
            self.assertEqual(mock_tracker.add.call_count, 3)
        finally:
            random_mod._CUDA_RNG_STATE_TRACKER = original_tracker


class TestForkRNG(unittest.TestCase):
    """Tests for _fork_rng context manager."""

    @patch("paddleformers.fleet.tensor_parallel.random._set_all_rng_states")
    @patch("paddleformers.fleet.tensor_parallel.random._get_all_rng_states")
    def test_fork_rng_restores(self, mock_get_all, mock_set_all):
        """Test _fork_rng restores RNG states after context."""
        mock_get_all.return_value = (MagicMock(), MagicMock(), MagicMock())
        with _fork_rng():
            pass
        mock_set_all.assert_called_once()


class TestCheckpoint(unittest.TestCase):
    """Tests for checkpoint function."""

    def test_checkpoint_passes(self):
        """Test checkpoint function does nothing (pass)."""
        result = checkpoint(lambda x: x, paddle.randn([2, 4]))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
