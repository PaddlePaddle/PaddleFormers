# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    subbatch,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _init_fleet():
    """Initialize fleet if not already done."""
    if not ps.have_global_memory_buffer():
        seed = 46
        np.random.seed(seed)
        paddle.manual_seed(seed)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
        }
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        ps.initialize_model_parallel(hcg)


class TestSubbatch(unittest.TestCase):
    """Test the subbatch utility function."""

    def test_subbatch_no_chunking_needed(self):
        """When axis_width < bs, the original function should be called directly."""

        def simple_func(a, b):
            return a + b

        sb_fn = subbatch(
            simple_func, arg_idx=[0, 1], axis=[0, 0], bs=100, out_idx=0
        )
        a = paddle.randn([10, 5])
        b = paddle.randn([10, 5])
        result = sb_fn(a, b)
        expected = a + b
        np.testing.assert_allclose(result, expected)

    def test_subbatch_with_chunking(self):
        """Test subbatch splits input and processes in chunks."""

        def simple_func(a, b):
            return a * b

        bs = 4
        sb_fn = subbatch(
            simple_func, arg_idx=[0, 1], axis=[0, 0], bs=bs, out_idx=0
        )
        a = paddle.randn([10, 5])
        b = paddle.randn([10, 5])
        result = sb_fn(a, b)
        expected = a * b
        np.testing.assert_allclose(result, expected)

    def test_subbatch_assert_axis_width_equal(self):
        """Assert that batch sizes must be equal across subbatched args."""
        with self.assertRaises(AssertionError):

            def f(a, b):
                return a

            sb_fn = subbatch(f, arg_idx=[0, 1], axis=[0, 0], bs=10, out_idx=0)
            a = paddle.randn([10, 5])
            b = paddle.randn([20, 5])
            sb_fn(a, b)

    def test_subbatch_same_arg_idx(self):
        """Test same_arg_idx parameter where two args share the same tensor."""

        def simple_func(a, b):
            return a + b

        bs = 3
        sb_fn = subbatch(
            simple_func,
            arg_idx=[0],
            axis=[0],
            bs=bs,
            out_idx=0,
            same_arg_idx={1: 0},
        )
        a = paddle.randn([10, 5])
        result = sb_fn(a, a)
        expected = a + a
        np.testing.assert_allclose(result, expected)


class TestLanguageLoss(unittest.TestCase):
    """Test LanguageLoss class."""

    def setUp(self):
        _init_fleet()

    def _make_config(self, **overrides):
        """Helper to create a TransformerConfig with sensible defaults."""
        defaults = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "use_cpu_initialization": True,
            "parallel_output": False,
            "loss_subbatch_sequence_length": 0,
        }
        defaults.update(overrides)
        return TransformerConfig(**defaults)

    def test_language_loss_init_basic(self):
        """Test basic LanguageLoss initialization."""
        config = self._make_config()
        loss_fn = LanguageLoss(config)
        self.assertIsNotNone(loss_fn)
        self.assertFalse(loss_fn.enable_parallel_cross_entropy)
        self.assertFalse(loss_fn.use_subbatch)

    def test_language_loss_forward_impl_simple(self):
        """Test forward_impl with simple logits and labels."""
        config = self._make_config()
        loss_fn = LanguageLoss(config)
        vocab_size = 128
        seq_len = 8
        logits = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, seq_len])
        loss = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(loss)
        self.assertEqual(loss.shape, [])

    def test_language_loss_forward_impl_with_ignored_index(self):
        """Test forward_impl when all labels are -100."""
        config = self._make_config()
        loss_fn = LanguageLoss(config)
        vocab_size = 128
        seq_len = 8
        logits = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.full([2, seq_len], -100, dtype="int64")
        loss = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(loss)

    def test_language_loss_forward_simple(self):
        """Test forward with tensor logits."""
        config = self._make_config()
        loss_fn = LanguageLoss(config)
        vocab_size = 128
        seq_len = 8
        logits = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, seq_len])
        loss = loss_fn.forward(logits, labels)
        self.assertIsNotNone(loss)

    def test_language_loss_forward_recompute(self):
        """Test forward when recompute_modules includes 'loss_fn'."""
        config = self._make_config(recompute_modules=["loss_fn"])
        loss_fn = LanguageLoss(config)
        vocab_size = 128
        seq_len = 8
        logits = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, seq_len])
        loss = loss_fn.forward(logits, labels)
        self.assertIsNotNone(loss)

    def test_language_loss_use_subbatch(self):
        """Test forward_impl with subbatch enabled."""
        config = self._make_config(loss_subbatch_sequence_length=4)
        loss_fn = LanguageLoss(config)
        self.assertTrue(loss_fn.use_subbatch)
        vocab_size = 128
        seq_len = 16
        logits = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, seq_len])
        loss = loss_fn.forward_impl(logits, labels)
        self.assertIsNotNone(loss)

    def test_language_loss_mtp_list_input(self):
        """Test forward with list of logits (MTP mode)."""
        config = self._make_config(
            num_nextn_predict_layers=2,
            mtp_load_weight_only=False,
            add_mtp_loss=False,
            mtp_distillation_loss=False,
            train_mtp_only=False,
        )
        loss_fn = LanguageLoss(config)
        vocab_size = 128
        seq_len = 8
        logits_0 = paddle.randn([2, seq_len, vocab_size])
        logits_1 = paddle.randn([2, seq_len, vocab_size])
        logits_2 = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, seq_len + 2])
        loss = loss_fn.forward([logits_0, logits_1, logits_2], labels)
        self.assertIsNotNone(loss)

    def test_language_loss_mtp_train_mtp_only(self):
        """Test MTP mode with train_mtp_only=True."""
        config = self._make_config(
            num_nextn_predict_layers=1,
            mtp_load_weight_only=False,
            add_mtp_loss=False,
            mtp_distillation_loss=False,
            train_mtp_only=True,
        )
        loss_fn = LanguageLoss(config)
        vocab_size = 128
        seq_len = 8
        logits_0 = paddle.randn([2, seq_len, vocab_size])
        logits_1 = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, seq_len + 1])
        loss = loss_fn.forward([logits_0, logits_1], labels)
        self.assertIsNotNone(loss)

    def test_language_loss_mtp_distillation_loss(self):
        """Test MTP mode with distillation loss."""
        config = self._make_config(
            num_nextn_predict_layers=1,
            mtp_load_weight_only=False,
            add_mtp_loss=False,
            mtp_distillation_loss=True,
        )
        loss_fn = LanguageLoss(config)
        vocab_size = 128
        seq_len = 8
        logits_0 = paddle.randn([2, seq_len, vocab_size])
        logits_1 = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, seq_len + 1])
        loss = loss_fn.forward([logits_0, logits_1], labels)
        self.assertIsNotNone(loss)

    def test_language_loss_mtp_add_loss(self):
        """Test MTP mode with add_mtp_loss=True."""
        config = self._make_config(
            num_nextn_predict_layers=1,
            mtp_load_weight_only=False,
            add_mtp_loss=True,
            mtp_loss_scaling_factor=0.5,
            mtp_distillation_loss=False,
        )
        loss_fn = LanguageLoss(config)
        vocab_size = 128
        seq_len = 8
        logits_0 = paddle.randn([2, seq_len, vocab_size])
        logits_1 = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, seq_len + 1])
        loss = loss_fn.forward([logits_0, logits_1], labels)
        self.assertIsNotNone(loss)

    def test_language_loss_mtp_tracker(self):
        """Test that MTP loss tracker is populated correctly."""
        config = self._make_config(
            num_nextn_predict_layers=2,
            mtp_load_weight_only=False,
            add_mtp_loss=False,
            mtp_distillation_loss=False,
        )
        LanguageLoss.mtp_loss_tracker = {}
        loss_fn = LanguageLoss(config)
        vocab_size = 128
        seq_len = 8
        logits_0 = paddle.randn([2, seq_len, vocab_size])
        logits_1 = paddle.randn([2, seq_len, vocab_size])
        logits_2 = paddle.randn([2, seq_len, vocab_size])
        labels = paddle.randint(0, vocab_size, [2, seq_len + 2])
        loss_fn.forward([logits_0, logits_1, logits_2], labels)
        self.assertIn("mtp_1_loss", LanguageLoss.mtp_loss_tracker)
        self.assertIn("mtp_2_loss", LanguageLoss.mtp_loss_tracker)

    def test_build_schedule_node(self):
        """Test build_schedule_node returns a ScheduleNode."""
        config = self._make_config()
        loss_fn = LanguageLoss(config)
        node = loss_fn.build_schedule_node()
        self.assertIsNotNone(node)

    def test_language_loss_assert_mtp_preconditions(self):
        """Test that list input triggers assertion if config is wrong."""
        config = self._make_config(num_nextn_predict_layers=0)
        loss_fn = LanguageLoss(config)
        with self.assertRaises(AssertionError):
            loss_fn.forward([], paddle.randint(0, 10, [2, 8]))


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
