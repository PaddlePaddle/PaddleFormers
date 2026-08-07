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

from paddleformers.fleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    MainLanguageLoss,
    MTPLanguageLoss,
    subbatch,
)


class TestSubbatchWithRecompute(unittest.TestCase):
    """Test subbatch with use_recompute=True."""

    def test_subbatch_with_recompute(self):
        """Test subbatch with recompute enabled."""

        def simple_fn(x, y):
            return x + y

        sb_fn = subbatch(
            simple_fn,
            arg_idx=[0, 1],
            axis=[0, 0],
            bs=3,
            out_idx=0,
            use_recompute=True,
        )
        x = paddle.randn([9, 5])
        y = paddle.randn([9, 5])
        result = sb_fn(x, y)
        expected = x + y
        self.assertTrue(paddle.allclose(result, expected, atol=1e-5))

    def test_subbatch_axis1(self):
        """Test subbatch on axis 1."""

        def simple_fn(x, y):
            return x + y

        sb_fn = subbatch(
            simple_fn,
            arg_idx=[0, 1],
            axis=[1, 1],
            bs=3,
            out_idx=1,
        )
        x = paddle.randn([2, 9, 5])
        y = paddle.randn([2, 9, 5])
        result = sb_fn(x, y)
        expected = x + y
        self.assertTrue(paddle.allclose(result, expected, atol=1e-5))


class TestLanguageLossExperimentalVersion(unittest.TestCase):
    """Test LanguageLoss with gpt_model_use_experimental_version=True."""

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_impl_experimental_version(
        self, mock_dist, mock_tp, mock_cp
    ):
        """Test forward_impl with experimental version line-wise loss."""
        mock_config = MagicMock()
        mock_config.parallel_output = False
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.gpt_model_use_experimental_version = True

        loss_fn = LanguageLoss(config=mock_config)
        vocab_size = 128
        seq_len = 8
        logits = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, seq_len])
        loss = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(loss)

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_impl_experimental_all_ignored(
        self, mock_dist, mock_tp, mock_cp
    ):
        """Test forward_impl with experimental version and all labels ignored."""
        mock_config = MagicMock()
        mock_config.parallel_output = False
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.gpt_model_use_experimental_version = True

        loss_fn = LanguageLoss(config=mock_config)
        vocab_size = 128
        seq_len = 8
        logits = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.full([2, seq_len], -100, dtype="int64")
        loss = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(loss)


class TestLanguageLossMD5Probe(unittest.TestCase):
    """Test LanguageLoss with LOG_LAYER_MD5 env var."""

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    @patch("paddle.distributed.get_rank", return_value=0)
    def test_forward_impl_with_md5_probe(
        self, mock_rank, mock_dist, mock_tp, mock_cp
    ):
        """Test forward_impl with LOG_LAYER_MD5=1."""
        mock_config = MagicMock()
        mock_config.parallel_output = False
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.gpt_model_use_experimental_version = False

        loss_fn = LanguageLoss(config=mock_config)
        vocab_size = 128
        seq_len = 8
        logits = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, seq_len])

        # Set env var
        os.environ["LOG_LAYER_MD5"] = "1"
        try:
            loss = loss_fn.forward_impl(logits, labels)
            self.assertIsNotNone(loss)
        finally:
            os.environ.pop("LOG_LAYER_MD5", None)


class TestLanguageLossForwardWithListLogits(unittest.TestCase):
    """Test LanguageLoss.forward with list logits (MTP mode)."""

    @patch("paddle.device.cuda.empty_cache")
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_mtp_experimental_version(
        self, mock_dist, mock_tp, mock_cp, mock_cache
    ):
        """Test MTP forward with experimental version."""
        mock_config = MagicMock()
        mock_config.parallel_output = False
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.num_nextn_predict_layers = 2
        mock_config.mtp_load_weight_only = False
        mock_config.mtp_distillation_loss = False
        mock_config.train_mtp_only = False
        mock_config.add_mtp_loss = True
        mock_config.mtp_loss_scaling_factor = 0.5
        mock_config.gpt_model_use_experimental_version = True
        mock_config.fused_linear_ce_loss_chunk = 0

        loss_fn = LanguageLoss(config=mock_config)
        vocab_size = 128
        seq_len = 8
        logits = [
            paddle.randn([2, seq_len, vocab_size]),
            paddle.randn([2, seq_len, vocab_size]),
            paddle.randn([2, seq_len, vocab_size]),
        ]
        labels = paddle.randint(0, vocab_size, [2, seq_len + 2])
        loss = loss_fn.forward(logits, labels)
        self.assertIsNotNone(loss)

    @patch("paddle.device.cuda.empty_cache")
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_mtp_train_mtp_only(
        self, mock_dist, mock_tp, mock_cp, mock_cache
    ):
        """Test MTP forward with train_mtp_only=True."""
        mock_config = MagicMock()
        mock_config.parallel_output = False
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.num_nextn_predict_layers = 2
        mock_config.mtp_load_weight_only = False
        mock_config.mtp_distillation_loss = False
        mock_config.train_mtp_only = True
        mock_config.add_mtp_loss = True
        mock_config.mtp_loss_scaling_factor = 1.0
        mock_config.gpt_model_use_experimental_version = False

        loss_fn = LanguageLoss(config=mock_config)
        vocab_size = 128
        seq_len = 8
        logits = [
            paddle.randn([2, seq_len, vocab_size]),
            paddle.randn([2, seq_len, vocab_size]),
            paddle.randn([2, seq_len, vocab_size]),
        ]
        labels = paddle.randint(0, vocab_size, [2, seq_len + 2])
        loss = loss_fn.forward(logits, labels)
        self.assertIsNotNone(loss)


class TestMainLanguageLossInit(unittest.TestCase):
    """Test MainLanguageLoss initialization."""

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_init(self, mock_dist, mock_tp):
        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss = MainLanguageLoss(config=mock_config)
        self.assertIsNotNone(loss)

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_build_schedule_node(self, mock_dist, mock_tp):
        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss = MainLanguageLoss(config=mock_config)
        node = loss.build_schedule_node()
        self.assertIsNotNone(node)


class TestMTPLanguageLossInit(unittest.TestCase):
    """Test MTPLanguageLoss initialization."""

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_init(self, mock_dist, mock_tp):
        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss = MTPLanguageLoss(config=mock_config)
        self.assertIsNotNone(loss)

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_build_schedule_node(self, mock_dist, mock_tp):
        mock_config = MagicMock()
        mock_config.parallel_output = True
        mock_config.loss_subbatch_sequence_length = 0

        loss = MTPLanguageLoss(config=mock_config)
        node = loss.build_schedule_node()
        self.assertIsNotNone(node)


class TestLanguageLossSubbatchForward(unittest.TestCase):
    """Test LanguageLoss forward_impl with subbatch."""

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_subbatch_forward_short_seq(self, mock_dist, mock_tp, mock_cp):
        """Test subbatch with sequence shorter than subbatch length."""
        mock_config = MagicMock()
        mock_config.parallel_output = False
        mock_config.loss_subbatch_sequence_length = 512
        mock_config.gpt_model_use_experimental_version = False

        loss_fn = LanguageLoss(config=mock_config)
        self.assertTrue(loss_fn.use_subbatch)
        # seq_len=8 < loss_subbatch_sequence_length=512, no subbatching
        vocab_size = 128
        logits = paddle.randn([2, 8, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, 8])
        loss = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(loss)

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_subbatch_forward_long_seq(self, mock_dist, mock_tp, mock_cp):
        """Test subbatch with sequence longer than subbatch length."""
        mock_config = MagicMock()
        mock_config.parallel_output = False
        mock_config.loss_subbatch_sequence_length = 4
        mock_config.gpt_model_use_experimental_version = False

        loss_fn = LanguageLoss(config=mock_config)
        self.assertTrue(loss_fn.use_subbatch)
        vocab_size = 128
        logits = paddle.randn([2, 16, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, 16])
        loss = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(loss)


class TestLanguageLossForwardRecompute(unittest.TestCase):
    """Test LanguageLoss._forward with recompute."""

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_with_loss_fn_recompute(self, mock_dist, mock_tp, mock_cp):
        """Test _forward with recompute_modules including loss_fn."""
        mock_config = MagicMock()
        mock_config.parallel_output = False
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.recompute_modules = ["loss_fn"]

        loss_fn = LanguageLoss(config=mock_config)
        vocab_size = 128
        logits = paddle.randn([2, 8, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, 8])
        loss = loss_fn._forward(logits, labels)
        self.assertIsNotNone(loss)

    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_context_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.models.common.language_loss.language_loss.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch("paddle.distributed.is_initialized", return_value=False)
    def test_forward_without_loss_fn_recompute(
        self, mock_dist, mock_tp, mock_cp
    ):
        """Test _forward without recompute for loss_fn."""
        mock_config = MagicMock()
        mock_config.parallel_output = False
        mock_config.loss_subbatch_sequence_length = 0
        mock_config.recompute_modules = None

        loss_fn = LanguageLoss(config=mock_config)
        vocab_size = 128
        logits = paddle.randn([2, 8, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, 8])
        loss = loss_fn._forward(logits, labels)
        self.assertIsNotNone(loss)


if __name__ == "__main__":
    unittest.main()
