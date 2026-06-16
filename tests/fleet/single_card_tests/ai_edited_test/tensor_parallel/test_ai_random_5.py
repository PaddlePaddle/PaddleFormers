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
from unittest.mock import patch

import paddle

from paddleformers.fleet.tensor_parallel.random import (
    _get_cuda_rng_state,
    _set_cuda_rng_state,
    get_all_rng_states,
    initialize_rng_tracker,
)


class TestGetCudaRngState(unittest.TestCase):
    """Tests for _get_cuda_rng_state."""

    def test_rejects_graph_safe(self):
        """Should reject graph_safe=True."""
        with self.assertRaises(AssertionError):
            _get_cuda_rng_state(graph_safe=True)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_returns_rng_state(self):
        """Should return a GeneratorState when called with default args."""
        result = _get_cuda_rng_state()
        self.assertIsNotNone(result)


class TestSetCudaRngState(unittest.TestCase):
    """Tests for _set_cuda_rng_state."""

    def test_rejects_graph_safe(self):
        """Should reject graph_safe=True."""
        with self.assertRaises(AssertionError):
            _set_cuda_rng_state(None, graph_safe=True)


class TestInitializeRngTracker(unittest.TestCase):
    """Tests for initialize_rng_tracker."""

    def test_rejects_use_cudagraphable_rng(self):
        """Should reject use_cudagraphable_rng=True."""
        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_cudagraphable_rng=True)

    def test_rejects_use_te_rng_tracker(self):
        """Should reject use_te_rng_tracker=True."""
        with self.assertRaises(AssertionError):
            initialize_rng_tracker(use_te_rng_tracker=True)

    @patch(
        "paddleformers.fleet.tensor_parallel.random._CUDA_RNG_STATE_TRACKER_INITIALIZED",
        True,
    )
    def test_returns_early_if_already_initialized(self):
        """Should return early if already initialized without force_reset."""
        # This should not raise or create a new tracker
        result = initialize_rng_tracker()
        # Just checking it doesn't crash


class TestGetAllRngStates(unittest.TestCase):
    """Tests for get_all_rng_states."""

    @patch(
        "paddleformers.fleet.tensor_parallel.random._CUDA_RNG_STATE_TRACKER_INITIALIZED",
        False,
    )
    def test_raises_when_not_initialized(self):
        """Should raise when tracker not initialized."""
        with self.assertRaises(AssertionError):
            get_all_rng_states()


if __name__ == "__main__":
    unittest.main()
