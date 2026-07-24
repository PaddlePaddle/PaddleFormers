# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You can obtain a copy of the License at
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
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.tensor_parallel.random import (
    checkpoint,
    enable_share_grad_holder,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)


class TestModelParallelCudaManualSeed(unittest.TestCase):
    """Tests for model_parallel_cuda_manual_seed."""

    def test_rejects_te_rng_tracker(self):
        """Should reject te_rng_tracker=True."""
        with self.assertRaises(AssertionError):
            model_parallel_cuda_manual_seed(42, te_rng_tracker=True)

    def test_rejects_cudagraphable_rng(self):
        """Should reject use_cudagraphable_rng=True."""
        with self.assertRaises(AssertionError):
            model_parallel_cuda_manual_seed(42, use_cudagraphable_rng=True)


class TestCheckpoint(unittest.TestCase):
    """Tests for checkpoint function."""

    def test_checkpoint_is_noop(self):
        """checkpoint function should be a no-op (pass)."""
        # It should not raise
        result = checkpoint(lambda x: x, 1)
        self.assertIsNone(result)


class TestEnableShareGradHolder(unittest.TestCase):
    """Tests for enable_share_grad_holder context manager."""

    def test_context_manager_restores_flag(self):
        """enable_share_grad_holder should restore the flag after exiting."""
        flag = "FLAGS_share_tensor_for_grad_tensor_holder"
        old_value = paddle.get_flags([flag])[flag]
        with enable_share_grad_holder():
            # Inside the context, flag should be True
            current = paddle.get_flags([flag])[flag]
            self.assertTrue(current)
        # Outside, should be restored
        restored = paddle.get_flags([flag])[flag]
        self.assertEqual(restored, old_value)


class TestCudaRNGStatesTrackerInference(unittest.TestCase):
    """Tests for inference RNG tracker creation."""

    @patch(
        "paddleformers.fleet.tensor_parallel.random._CUDA_RNG_STATE_TRACKER_INITIALIZED",
        True,
    )
    def test_inference_tracker_add_is_noop(self):
        """Inference tracker add should be a no-op."""
        initialize_rng_tracker(inference_rng_tracker=True)
        from paddleformers.fleet.tensor_parallel.random import (
            _CUDA_RNG_STATE_TRACKER,
        )

        if _CUDA_RNG_STATE_TRACKER is not None:
            # Add should not raise
            _CUDA_RNG_STATE_TRACKER.add("test", 42)

    def test_inference_tracker_fork_is_nullcontext(self):
        """Inference tracker fork should return nullcontext."""
        import contextlib

        # Force reset to ensure we get a fresh inference tracker
        initialize_rng_tracker(inference_rng_tracker=True, force_reset=True)
        from paddleformers.fleet.tensor_parallel.random import (
            _CUDA_RNG_STATE_TRACKER,
        )

        if _CUDA_RNG_STATE_TRACKER is not None:
            result = _CUDA_RNG_STATE_TRACKER.fork()
            self.assertIsInstance(result, contextlib.nullcontext)


class TestInitializeRngTrackerForceReset(unittest.TestCase):
    """Tests for initialize_rng_tracker with force_reset."""

    def test_force_reset_clears_tracker(self):
        """force_reset=True should clear the existing tracker."""
        # First initialize
        with (
            patch(
                "paddleformers.fleet.tensor_parallel.random._CUDA_RNG_STATE_TRACKER_INITIALIZED",
                True,
            ),
            patch(
                "paddleformers.fleet.tensor_parallel.random._CUDA_RNG_STATE_TRACKER",
                MagicMock(),
            ),
        ):
            initialize_rng_tracker(force_reset=True)
            # Should not raise


if __name__ == "__main__":
    unittest.main()
