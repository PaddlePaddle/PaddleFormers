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

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.transformer.moe.moe_utils import (
    AddAuxiliaryLoss,
    FakeClone,
    _all_gather_local_tokens,
    _log_summary,
    _log_tokens_per_expert,
    detach_and_requires_grad_,
    is_tensor,
    log_moe_balance,
    log_moe_losses,
    permute,
    sort_chunks_by_idxs,
    unpermute,
)


class TestAddAuxiliaryLossDetailed(unittest.TestCase):
    """Detailed tests for AddAuxiliaryLoss."""

    def test_forward_with_grad(self):
        """Test forward with loss that has gradient."""
        x = paddle.randn([4, 8])
        loss = paddle.to_tensor(0.5)
        loss.stop_gradient = False
        out = AddAuxiliaryLoss.apply(x, loss)
        self.assertEqual(out.shape, x.shape)

    def test_forward_with_no_grad_loss(self):
        """Test forward with loss that has no gradient."""
        x = paddle.randn([4, 8])
        loss = paddle.to_tensor(0.5)
        loss.stop_gradient = True
        out = AddAuxiliaryLoss.apply(x, loss)
        self.assertEqual(out.shape, x.shape)

    def test_backward_with_required_aux_loss(self):
        """Test backward when aux loss gradient is required."""
        x = paddle.randn([4, 8])
        x.stop_gradient = False
        loss = paddle.to_tensor(0.5)
        loss.stop_gradient = False
        out = AddAuxiliaryLoss.apply(x, loss)
        out_sum = out.sum()
        out_sum.backward()
        self.assertIsNotNone(x.grad)


class TestFakeClone(unittest.TestCase):
    """Tests for FakeClone PyLayer."""

    def test_forward_contiguous(self):
        """Test FakeClone with contiguous input."""
        x = paddle.randn([4, 8])
        out = FakeClone.apply(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_non_contiguous(self):
        """Test FakeClone with non-contiguous input."""
        x = paddle.randn([4, 8])
        x_t = x.transpose([1, 0])
        out = FakeClone.apply(x_t)
        self.assertEqual(out.shape, x_t.shape)

    def test_backward_passes_grad(self):
        """Test FakeClone backward passes gradient through."""
        x = paddle.randn([4, 8])
        x.stop_gradient = False
        out = FakeClone.apply(x)
        out_sum = out.sum()
        out_sum.backward()
        self.assertIsNotNone(x.grad)


class TestDetachAndRequiresGrad(unittest.TestCase):
    """Tests for detach_and_requires_grad_."""

    def test_with_tensors(self):
        """Test detach_and_requires_grad_ with tensors."""
        x = paddle.randn([4, 8])
        x.stop_gradient = False
        y = paddle.randn([4, 8])
        y.stop_gradient = True

        ret = detach_and_requires_grad_(x, y)
        self.assertTrue(ret[0].stop_gradient is False)
        self.assertTrue(ret[1].stop_gradient is True)

    def test_with_mixed_types(self):
        """Test detach_and_requires_grad_ with mixed types."""
        x = paddle.randn([4, 8])
        x.stop_gradient = False
        val = 42

        ret = detach_and_requires_grad_(x, val)
        self.assertEqual(ret[1], 42)


class TestIsTensor(unittest.TestCase):
    """Tests for is_tensor."""

    def test_with_tensor(self):
        self.assertTrue(is_tensor(paddle.randn([4])))

    def test_with_non_tensor(self):
        self.assertFalse(is_tensor(42))
        self.assertFalse(is_tensor([1, 2, 3]))


class TestSortChunksByIdxs(unittest.TestCase):
    """Tests for sort_chunks_by_idxs."""

    def test_basic_sort(self):
        """Test basic sort_chunks_by_idxs."""
        input_tensor = paddle.randn([6, 4])
        split_sizes = paddle.to_tensor([2, 2, 2])
        sorted_idxs = paddle.to_tensor([2, 0, 1])

        output, permuted_probs = sort_chunks_by_idxs(
            input_tensor, split_sizes, sorted_idxs
        )
        self.assertEqual(output.shape[0], 6)
        self.assertIsNone(permuted_probs)


class TestAllGatherLocalTokens(unittest.TestCase):
    """Tests for _all_gather_local_tokens."""

    def test_single_rank(self):
        """Test _all_gather_local_tokens with group=None."""
        tokens = paddle.to_tensor([1, 2, 3], dtype="float32")
        result = _all_gather_local_tokens(tokens, group=None)
        self.assertEqual(result.shape, [1, 3])

    def test_with_mock_group(self):
        """Test _all_gather_local_tokens with mock group of size 1."""
        tokens = paddle.to_tensor([1, 2, 3], dtype="float32")
        mock_group = MagicMock()
        mock_group.is_member.return_value = False

        with patch(
            "paddleformers.fleet.transformer.moe.moe_utils.get_pg_size", return_value=1
        ):
            result = _all_gather_local_tokens(tokens, group=mock_group)
            self.assertEqual(result.shape[0], 1)


class TestLogMoeLosses(unittest.TestCase):
    """Tests for log_moe_losses."""

    def test_with_logs_disabled(self):
        """Test log_moe_losses when global logs are not enabled."""
        with patch(
            "paddleformers.fleet.transformer.moe.moe_utils.global_moe_balance_training_logs_enabled",
            return_value=False,
        ):
            # Should not raise
            log_moe_losses(1, aux_loss=paddle.to_tensor(0.1))

    def test_with_none_logs(self):
        """Test log_moe_losses when get_global_training_logs returns None."""
        with (
            patch(
                "paddleformers.fleet.transformer.moe.moe_utils.global_moe_balance_training_logs_enabled",
                return_value=True,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs",
                return_value=None,
            ),
        ):
            log_moe_losses(1, aux_loss=paddle.to_tensor(0.1))

    def test_with_valid_logs(self):
        """Test log_moe_losses with valid logs object."""
        mock_logs = MagicMock()
        with (
            patch(
                "paddleformers.fleet.transformer.moe.moe_utils.global_moe_balance_training_logs_enabled",
                return_value=True,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs",
                return_value=mock_logs,
            ),
        ):
            log_moe_losses(
                layer_number=1,
                aux_loss=paddle.to_tensor(0.1),
                z_loss=paddle.to_tensor(0.01),
            )
            mock_logs.update.assert_called_once()


class TestLogMoeBalance(unittest.TestCase):
    """Tests for log_moe_balance."""

    def test_with_none_tokens_per_expert(self):
        """Test log_moe_balance with None tokens_per_expert."""
        # Should not raise
        log_moe_balance(1, None, 2, None)

    def test_with_list_tokens_per_expert(self):
        """Test log_moe_balance with list tokens_per_expert."""
        mock_logs = MagicMock()
        with (
            patch(
                "paddleformers.fleet.transformer.moe.moe_utils.global_moe_balance_training_logs_enabled",
                return_value=True,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs",
                return_value=mock_logs,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.moe_utils._all_gather_local_tokens",
                return_value=paddle.to_tensor(
                    [[1, 2], [3, 4]], dtype="float32"
                ),
            ),
        ):
            log_moe_balance(
                layer_number=1,
                moe_group=None,
                num_experts_per_tok=2,
                tokens_per_expert=[1, 2],
            )


class TestLogSummary(unittest.TestCase):
    """Tests for _log_summary."""

    def test_with_empty_tensor(self):
        """Test _log_summary with empty tensor."""
        mock_logs = MagicMock()
        with patch(
            "paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs",
            return_value=mock_logs,
        ):
            data = paddle.to_tensor([], dtype="float32")
            _log_summary("test_key", 1, data)
            # Should not call update for empty tensor

    def test_with_valid_data(self):
        """Test _log_summary with valid data."""
        mock_logs = MagicMock()
        with patch(
            "paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs",
            return_value=mock_logs,
        ):
            data = paddle.to_tensor([1.0, 2.0, 3.0])
            _log_summary("test_key", 1, data)
            mock_logs.update.assert_called_once()


class TestLogTokensPerExpert(unittest.TestCase):
    """Tests for _log_tokens_per_expert."""

    def test_with_zero_count(self):
        """Test _log_tokens_per_expert with zero count."""
        mock_logs = MagicMock()
        with (
            patch(
                "paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs",
                return_value=mock_logs,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.moe_utils._log_summary",
            ) as mock_log_summary,
        ):
            summary_data = paddle.to_tensor([1.0, 2.0])
            count = paddle.to_tensor([0])
            _log_tokens_per_expert("test_key", 1, summary_data, count)
            # count is 0, should be replaced with ones_like
            self.assertEqual(mock_log_summary.call_count, 2)


class TestPermuteUnpermuteRoundTrip(unittest.TestCase):
    """Test permute + unpermute round trip with probs."""

    def test_round_trip_with_probs(self):
        """Test that permute then unpermute restores original shape."""
        tokens = paddle.randn([4, 8], dtype="float32")
        routing_map = paddle.to_tensor(
            [[1, 0], [0, 1], [1, 0], [0, 1]], dtype="float32"
        )
        probs = paddle.to_tensor(
            [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]], dtype="float32"
        )

        permuted, sorted_indices = permute(tokens, routing_map)
        self.assertEqual(permuted.shape[0], 4)

        restored = unpermute(
            permuted,
            sorted_indices,
            restore_shape=[4, 8],
            probs=probs,
            routing_map=routing_map,
        )
        self.assertEqual(restored.shape, [4, 8])


if __name__ == "__main__":
    unittest.main()
