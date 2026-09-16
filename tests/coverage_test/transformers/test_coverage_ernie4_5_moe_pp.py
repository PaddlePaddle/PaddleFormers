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

import unittest
from unittest.mock import MagicMock, patch

import paddle
from paddle import nn

from paddleformers.transformers.ernie4_5_moe_vl.model.modeling_moe_pp import (
    EmptyLayer,
    Ernie4_5_EmbeddingPipe,
    create_skip_config_for_refined_recompute,
    get_pp_vp_split_layers,
    parse_args,
)


class TestParseArgs(unittest.TestCase):
    """Tests for parse_args function."""

    def test_single_tensor(self):
        hidden = paddle.randn([2, 3])
        result = parse_args(hidden)
        h, am, pid, npk = result
        self.assertTrue(paddle.equal(h, hidden).item() if h.numel() == 1 else paddle.equal(h, hidden).all().item())
        self.assertIsNone(am)
        self.assertIsNone(pid)
        self.assertIsNone(npk)

    def test_tuple_two(self):
        hidden = paddle.randn([2, 3])
        attn_mask = paddle.ones([2, 3])
        h, am, pid, npk = parse_args((hidden, attn_mask))
        self.assertTrue(paddle.equal(h, hidden).all().item())
        self.assertTrue(paddle.equal(am, attn_mask).all().item())
        self.assertIsNone(pid)

    def test_tuple_three(self):
        hidden = paddle.randn([2, 3])
        attn_mask = paddle.ones([2, 3])
        position_ids = paddle.arange(3)
        h, am, pid, npk = parse_args((hidden, attn_mask, position_ids))
        self.assertTrue(paddle.equal(pid, position_ids).all().item())
        self.assertIsNone(npk)

    def test_tuple_three_mtp(self):
        hidden = paddle.randn([2, 3])
        attn_mask = paddle.ones([2, 3])
        nbatch_pack_offset = paddle.ones([2])
        h, am, pid, npk = parse_args((hidden, attn_mask, nbatch_pack_offset), mtp_enable=True)
        self.assertIsNone(pid)
        self.assertTrue(paddle.equal(npk, nbatch_pack_offset).all().item())

    def test_tuple_four(self):
        hidden = paddle.randn([2, 3])
        attn_mask = paddle.ones([2, 3])
        position_ids = paddle.arange(3)
        nbatch_pack_offset = paddle.ones([2])
        h, am, pid, npk = parse_args((hidden, attn_mask, position_ids, nbatch_pack_offset))
        self.assertTrue(paddle.equal(pid, position_ids).all().item())
        self.assertTrue(paddle.equal(npk, nbatch_pack_offset).all().item())

    def test_stop_gradient(self):
        position_ids = paddle.arange(3)
        attn_mask = paddle.ones([2, 3])
        _, am, pid, _ = parse_args((paddle.randn([2, 3]), attn_mask, position_ids))
        self.assertTrue(pid.stop_gradient)
        self.assertTrue(am.stop_gradient)


@unittest.skip("get_pp_vp_split_layers requires pp_size > 1 which conflicts with CI environment")
class TestGetPPVPSplitLayers(unittest.TestCase):
    """Tests for get_pp_vp_split_layers function."""

    @patch("paddleformers.transformers.ernie4_5_moe_vl.model.modeling_moe_pp.get_hcg")
    def test_basic(self, mock_get_hcg):
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_world_size.return_value = 2
        mock_get_hcg.return_value = mock_hcg

        config = MagicMock()
        config.virtual_pipeline_model_parallel_size = 1
        config.num_hidden_layers = 8
        config.num_empty_layers_add_in_tail = 0

        result = get_pp_vp_split_layers(config)
        self.assertIsInstance(result, set)

    @patch("paddleformers.transformers.ernie4_5_moe_vl.model.modeling_moe_pp.get_hcg")
    def test_pp_size_one_raises(self, mock_get_hcg):
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_world_size.return_value = 1
        mock_get_hcg.return_value = mock_hcg

        config = MagicMock()
        config.virtual_pipeline_model_parallel_size = 1
        config.num_hidden_layers = 8
        config.num_empty_layers_add_in_tail = 0

        with self.assertRaises(AssertionError):
            get_pp_vp_split_layers(config)

    @patch("paddleformers.transformers.ernie4_5_moe_vl.model.modeling_moe_pp.get_hcg")
    def test_skip_recompute_zero(self, mock_get_hcg):
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_world_size.return_value = 2
        mock_get_hcg.return_value = mock_hcg

        config = MagicMock()
        config.virtual_pipeline_model_parallel_size = 1
        config.num_hidden_layers = 8
        config.num_empty_layers_add_in_tail = 0

        result = get_pp_vp_split_layers(config, skip_recompute_num=0)
        self.assertEqual(result, set())

    @patch("paddleformers.transformers.ernie4_5_moe_vl.model.modeling_moe_pp.get_hcg")
    def test_vp_size_one_skip_all(self, mock_get_hcg):
        mock_hcg = MagicMock()
        mock_hcg.get_pipe_parallel_world_size.return_value = 2
        mock_get_hcg.return_value = mock_hcg

        config = MagicMock()
        config.virtual_pipeline_model_parallel_size = 1
        config.num_hidden_layers = 8
        config.num_empty_layers_add_in_tail = 0

        result = get_pp_vp_split_layers(config, skip_recompute_num=1)
        self.assertEqual(result, set(range(8)))


class TestCreateSkipConfigForRefinedRecompute(unittest.TestCase):
    """Tests for create_skip_config_for_refined_recompute function."""

    def test_recompute_granularity_not_none_returns_empty(self):
        """When recompute_granularity is not None, returns empty skip_recompute_ops."""
        config = MagicMock()
        config.recompute_granularity = "full"
        config.recompute_modules = {"attn": 2}
        config.skip_recompute_ops = {}

        create_skip_config_for_refined_recompute(0, config)
        self.assertIn(0, config.skip_recompute_ops)
        self.assertEqual(config.skip_recompute_ops[0], {})

    def test_recompute_granularity_none_with_non_empty_dict_modules_raises(self):
        """When recompute_granularity is None but recompute_modules has items,
        it raises ValueError because granularity != 'full'."""
        config = MagicMock()
        config.recompute_granularity = None
        config.recompute_modules = {"attn": 2}

        with self.assertRaises(ValueError):
            create_skip_config_for_refined_recompute(0, config)

    def test_recompute_granularity_none_with_empty_dict_modules(self):
        """When recompute_granularity is None and recompute_modules is empty dict,
        skip_config is empty dict (loop doesn't run)."""
        config = MagicMock()
        config.recompute_granularity = None
        config.recompute_modules = {}
        config.skip_recompute_ops = {}

        create_skip_config_for_refined_recompute(0, config)
        self.assertIn(0, config.skip_recompute_ops)
        self.assertEqual(config.skip_recompute_ops[0], {})

    def test_recompute_granularity_none_non_dict_modules(self):
        """When recompute_granularity is None and recompute_modules is not dict,
        sets empty skip_recompute_ops (early return)."""
        config = MagicMock()
        config.recompute_granularity = None
        config.recompute_modules = "not_a_dict"
        config.skip_recompute_ops = {}

        create_skip_config_for_refined_recompute(0, config)
        self.assertIn(0, config.skip_recompute_ops)
        self.assertEqual(config.skip_recompute_ops[0], {})

    def test_selective_recompute_granularity_early_return(self):
        """When recompute_granularity is not None (e.g. 'selective'),
        it triggers early return with empty skip_recompute_ops."""
        config = MagicMock()
        config.recompute_granularity = "selective"
        config.recompute_modules = {"attn": 2}
        config.skip_recompute_ops = {}

        create_skip_config_for_refined_recompute(0, config)
        self.assertIn(0, config.skip_recompute_ops)
        self.assertEqual(config.skip_recompute_ops[0], {})


class TestEmptyLayer(unittest.TestCase):
    """Tests for EmptyLayer."""

    def test_forward(self):
        layer = EmptyLayer()
        x = paddle.randn([2, 3])
        result = layer(x)
        self.assertTrue(paddle.equal(result, x).all().item())


class TestErnie45EmbeddingPipe(unittest.TestCase):
    """Tests for Ernie4_5_EmbeddingPipe."""

    def test_init(self):
        config = MagicMock()
        config.vocab_size = 100
        config.hidden_size = 64
        config.tensor_model_parallel_size = 1
        config.sequence_parallel = False
        config.num_nextn_predict_layers = 0
        config.use_moe = False

        emb = Ernie4_5_EmbeddingPipe(config)
        self.assertIsInstance(emb.embed_tokens, nn.Embedding)

    def test_embedding_weight(self):
        config = MagicMock()
        config.vocab_size = 100
        config.hidden_size = 64
        config.tensor_model_parallel_size = 1
        config.sequence_parallel = False
        config.num_nextn_predict_layers = 0
        config.use_moe = False

        emb = Ernie4_5_EmbeddingPipe(config)
        self.assertIsNotNone(emb.embedding_weight)

    def test_forward_basic(self):
        config = MagicMock()
        config.vocab_size = 100
        config.hidden_size = 64
        config.tensor_model_parallel_size = 1
        config.sequence_parallel = False
        config.num_nextn_predict_layers = 0
        config.use_moe = False

        emb = Ernie4_5_EmbeddingPipe(config)
        input_ids = paddle.randint(0, 100, [2, 5])
        result = emb(input_ids)
        self.assertIsNotNone(result)

    def test_forward_with_attention_mask(self):
        config = MagicMock()
        config.vocab_size = 100
        config.hidden_size = 64
        config.tensor_model_parallel_size = 1
        config.sequence_parallel = False
        config.num_nextn_predict_layers = 0
        config.use_moe = False

        emb = Ernie4_5_EmbeddingPipe(config)
        input_ids = paddle.randint(0, 100, [2, 5])
        attn_mask = paddle.ones([2, 5], dtype=paddle.int32)
        result = emb((input_ids, attn_mask))
        self.assertIsNotNone(result)

    def test_forward_with_position_ids(self):
        config = MagicMock()
        config.vocab_size = 100
        config.hidden_size = 64
        config.tensor_model_parallel_size = 1
        config.sequence_parallel = False
        config.num_nextn_predict_layers = 0
        config.use_moe = False

        emb = Ernie4_5_EmbeddingPipe(config)
        input_ids = paddle.randint(0, 100, [2, 5])
        attn_mask = paddle.ones([2, 5], dtype=paddle.int32)
        position_ids = paddle.arange(5).unsqueeze(0).expand([2, 5])
        result = emb((input_ids, attn_mask, position_ids))
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
