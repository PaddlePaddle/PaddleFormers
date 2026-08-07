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
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.transformer.moe.moe_utils import (
    _all_gather_local_tokens,
    _log_summary,
    _log_tokens_per_expert,
    all_gather_group,
    barrier_ep,
    global_moe_balance_training_logs_enabled,
    reduce_scatter_group,
    sort_chunks_by_idxs,
)


class TestBarrierEp(unittest.TestCase):
    """Tests for barrier_ep function."""

    @patch("paddleformers.fleet.transformer.moe.moe_utils.paddle.distributed.barrier")
    def test_barrier_ep_calls_barrier(self, mock_barrier):
        """barrier_ep should call paddle.distributed.barrier with the group."""
        mock_group = MagicMock()
        barrier_ep(mock_group)
        mock_barrier.assert_called_once_with(mock_group)


class TestGlobalMoeBalanceLogsEnabled(unittest.TestCase):
    """Tests for global_moe_balance_training_logs_enabled."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs",
        return_value=None,
    )
    def test_returns_false_when_no_logs(self, mock_get):
        """Should return False when global training logs is None."""
        self.assertFalse(global_moe_balance_training_logs_enabled())

    @patch("paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs")
    def test_returns_false_when_no_attribute(self, mock_get):
        """Should return False when logs has no is_moe_balance_logs_enabled."""
        mock_logs = MagicMock(spec=[])
        mock_get.return_value = mock_logs
        self.assertFalse(global_moe_balance_training_logs_enabled())

    @patch("paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs")
    def test_returns_true_when_enabled(self, mock_get):
        """Should return True when is_moe_balance_logs_enabled returns True."""
        mock_logs = MagicMock()
        mock_logs.is_moe_balance_logs_enabled.return_value = True
        mock_get.return_value = mock_logs
        self.assertTrue(global_moe_balance_training_logs_enabled())


class TestAllGatherLocalTokens(unittest.TestCase):
    """Tests for _all_gather_local_tokens."""

    def test_returns_reshaped_tensor_when_single_process(self):
        """Should return reshaped tensor when group size is 1 or group is None."""
        tokens = paddle.to_tensor([1, 2, 3, 4], dtype="int64")
        result = _all_gather_local_tokens(tokens, None)
        self.assertEqual(list(result.shape), [1, 4])


class TestSortChunksByIdxs(unittest.TestCase):
    """Tests for sort_chunks_by_idxs."""

    def test_sort_chunks_basic(self):
        """sort_chunks_by_idxs should split and reorder tensor."""
        paddle.disable_static()
        input_tensor = paddle.randn([10, 4])
        split_sizes = paddle.to_tensor([3, 4, 3])
        sorted_idxs = paddle.to_tensor([2, 0, 1])
        output, permuted_probs = sort_chunks_by_idxs(
            input_tensor, split_sizes, sorted_idxs
        )
        self.assertEqual(output.shape[0], 10)
        self.assertIsNone(permuted_probs)


class TestLogSummary(unittest.TestCase):
    """Tests for _log_summary."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs",
        return_value=None,
    )
    def test_returns_early_when_no_logs(self, mock_get):
        """Should return early when no global training logs."""
        _log_summary("test_key", 0, paddle.randn([4]))
        # Should not raise any errors

    @patch("paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs")
    def test_logs_summary_stats(self, mock_get):
        """Should log max, min, var, median, mean stats."""
        mock_logs = MagicMock()
        mock_get.return_value = mock_logs
        _log_summary("test_key", 0, paddle.to_tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertTrue(mock_logs.update.called)
        call_kwargs = mock_logs.update.call_args[1]
        self.assertIn("test_key_layer_0_max", call_kwargs)
        self.assertIn("test_key_layer_0_min", call_kwargs)
        self.assertIn("test_key_layer_0_mean", call_kwargs)


class TestLogTokensPerExpert(unittest.TestCase):
    """Tests for _log_tokens_per_expert."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs",
        return_value=None,
    )
    def test_handles_zero_count(self, mock_get):
        """Should handle count of 0 by replacing with ones."""
        summary_data = paddle.to_tensor([1.0, 2.0])
        count = paddle.to_tensor([0])
        _log_tokens_per_expert("key", 0, summary_data, count)
        # Should not raise any errors


class TestReduceScatterGroup(unittest.TestCase):
    """Tests for reduce_scatter_group."""

    def test_returns_clone_for_single_process(self):
        """Should return a clone when parallelism is 1."""
        mock_group = MagicMock()
        mock_group.nranks = 1
        input_tensor = paddle.randn([4, 8])
        result = reduce_scatter_group(input_tensor, group=mock_group)
        self.assertEqual(result.shape, input_tensor.shape)
        self.assertTrue(paddle.allclose(result, input_tensor))


class TestAllGatherGroup(unittest.TestCase):
    """Tests for all_gather_group."""

    def test_returns_clone_for_single_process(self):
        """Should return a clone when parallelism is 1."""
        mock_group = MagicMock()
        mock_group.nranks = 1
        input_tensor = paddle.randn([4, 8])
        result = all_gather_group(input_tensor, group=mock_group)
        self.assertEqual(result.shape, input_tensor.shape)
        self.assertTrue(paddle.allclose(result, input_tensor))


if __name__ == "__main__":
    unittest.main()
