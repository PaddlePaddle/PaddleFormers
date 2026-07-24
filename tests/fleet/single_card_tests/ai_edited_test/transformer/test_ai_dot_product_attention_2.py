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

from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.transformer.dot_product_attention import (
    DotProductAttention,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "context_parallel_size": 1,
        "attention_dropout": 0.0,
        "attention_softmax_in_fp32": True,
        "masked_softmax_fusion": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestDotProductAttentionConstruction(unittest.TestCase):
    """Test DotProductAttention construction."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_softmax_scale_computed(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(head_dim=64, num_attention_heads=4)
        dpa = DotProductAttention(
            config,
            layer_number=1,
            attn_mask_type="padding",
            attention_type="self",
            pg_collection=mock_pg_obj,
        )
        expected = 1.0 / (64**0.5)
        self.assertAlmostEqual(dpa.softmax_scale, expected, places=5)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_custom_softmax_scale(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config()
        dpa = DotProductAttention(
            config,
            layer_number=1,
            attn_mask_type="padding",
            attention_type="self",
            softmax_scale=0.5,
            pg_collection=mock_pg_obj,
        )
        self.assertAlmostEqual(dpa.softmax_scale, 0.5)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_qk_layer_scaling(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(apply_query_key_layer_scaling=True)
        dpa = DotProductAttention(
            config,
            layer_number=2,
            attn_mask_type="padding",
            attention_type="self",
            pg_collection=mock_pg_obj,
        )
        base = 1.0 / (32**0.5)
        self.assertAlmostEqual(dpa.softmax_scale, base / 2.0, places=5)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_context_parallel_assertion(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(context_parallel_size=2)
        dpa = DotProductAttention(
            config,
            layer_number=1,
            attn_mask_type="padding",
            attention_type="self",
            pg_collection=mock_pg_obj,
        )
        self.assertIsInstance(dpa, DotProductAttention)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_pg_collection_without_tp_raises(self, mock_pg):
        mock_pg.return_value = None
        with self.assertRaises(Exception):
            DotProductAttention(
                _make_config(), 1, "padding", "self", pg_collection=None
            )


class TestDotProductAttentionSoftmaxType(unittest.TestCase):
    """Test softmax type configuration."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_vanilla_softmax(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(softmax_type="vanilla")
        dpa = DotProductAttention(
            config,
            layer_number=1,
            attn_mask_type="padding",
            attention_type="self",
            pg_collection=mock_pg_obj,
        )
        self.assertIsNone(dpa.softmax_offset)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_off_by_one_softmax(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(
            softmax_type="off-by-one", perform_initialization=False
        )
        dpa = DotProductAttention(
            config,
            layer_number=1,
            attn_mask_type="padding",
            attention_type="self",
            pg_collection=mock_pg_obj,
        )
        self.assertIsNotNone(dpa.softmax_offset)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_learnable_softmax(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(
            softmax_type="learnable", perform_initialization=False
        )
        dpa = DotProductAttention(
            config,
            layer_number=1,
            attn_mask_type="padding",
            attention_type="self",
            pg_collection=mock_pg_obj,
        )
        self.assertIsInstance(dpa.softmax_offset, paddle.nn.Parameter)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_invalid_softmax_type_raises(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(softmax_type="invalid")
        with self.assertRaises(ValueError):
            DotProductAttention(
                config,
                layer_number=1,
                attn_mask_type="padding",
                attention_type="self",
                pg_collection=mock_pg_obj,
            )


class TestDotProductAttentionNumHeads(unittest.TestCase):
    """Test head partition calculations."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_gqa_partition(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.tp.world_size = 2
        mock_pg.return_value = mock_pg_obj

        config = _make_config(
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
            tensor_model_parallel_size=2,
        )
        dpa = DotProductAttention(
            config,
            layer_number=1,
            attn_mask_type="padding",
            attention_type="self",
            pg_collection=mock_pg_obj,
        )
        self.assertEqual(dpa.num_attention_heads_per_partition, 4)
        self.assertEqual(dpa.num_query_groups_per_partition, 2)


class TestDotProductAttentionContextParallel(unittest.TestCase):
    """Tests for DotProductAttention with context parallelism (formerly CPDotProductAttention)."""

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size"
    )
    def test_forward_asserts_packed_seq(self, mock_cp_size):
        mock_cp_size.return_value = 2
        config = _make_config()
        attn = DotProductAttention(
            config,
            layer_number=1,
            attn_mask_type="padding",
            attention_type="self",
        )
        q = paddle.randn([1, 4, 2, 16])
        k = paddle.randn([1, 4, 2, 16])
        v = paddle.randn([1, 4, 2, 16])
        mask = paddle.zeros([1, 1, 4, 4], dtype="float32")
        with self.assertRaises(AssertionError):
            attn(q, k, v, mask, packed_seq_params="fake")

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size"
    )
    def test_forward_asserts_attention_bias(self, mock_cp_size):
        mock_cp_size.return_value = 2
        config = _make_config()
        attn = DotProductAttention(
            config,
            layer_number=1,
            attn_mask_type="padding",
            attention_type="self",
        )
        q = paddle.randn([1, 4, 2, 16])
        k = paddle.randn([1, 4, 2, 16])
        v = paddle.randn([1, 4, 2, 16])
        mask = paddle.zeros([1, 1, 4, 4], dtype="float32")
        with self.assertRaises(AssertionError):
            attn(q, k, v, mask, attention_bias="fake")


if __name__ == "__main__":
    unittest.main()
