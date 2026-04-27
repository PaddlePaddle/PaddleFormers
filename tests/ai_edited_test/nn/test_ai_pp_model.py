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

import numpy as np
import paddle
import paddle.nn as nn


class TestParseArgs(unittest.TestCase):
    """Tests for parse_args function in pp_model.py."""

    def _get_parse_args(self):
        from paddleformers.nn.pp_model import parse_args

        return parse_args

    def test_single_tensor_input(self):
        """Test parse_args with a single tensor argument."""
        parse_args = self._get_parse_args()
        x = paddle.randn([2, 4])
        hidden, mask, pos, pe, nbatch = parse_args(x)
        self.assertIs(hidden, x)
        self.assertIsNone(mask)
        self.assertIsNone(pos)
        self.assertIsNone(pe)
        self.assertIsNone(nbatch)

    def test_tuple_length_1(self):
        """Test parse_args with 1-element tuple."""
        parse_args = self._get_parse_args()
        x = paddle.randn([2, 4])
        hidden, mask, pos, pe, nbatch = parse_args((x,))
        self.assertIs(hidden, x)
        self.assertIsNone(mask)
        self.assertIsNone(pos)
        self.assertIsNone(pe)
        self.assertIsNone(nbatch)

    def test_tuple_length_2(self):
        """Test parse_args with 2-element tuple."""
        parse_args = self._get_parse_args()
        x = paddle.randn([2, 4])
        mask = paddle.randint(0, 2, [2, 4]).astype("float32")
        hidden, attention_mask, pos, pe, nbatch = parse_args((x, mask))
        self.assertIs(hidden, x)
        self.assertIs(attention_mask, mask)
        self.assertIsNone(pos)
        self.assertIsNone(pe)
        self.assertIsNone(nbatch)

    def test_tuple_length_2_mtp_enable(self):
        """Test parse_args with 2-element tuple and mtp_enable=True."""
        parse_args = self._get_parse_args()
        x = paddle.randn([2, 4])
        offset = paddle.randn([2, 4])
        hidden, mask, pos, pe, nbatch = parse_args((x, offset), mtp_enable=True)
        self.assertIs(hidden, x)
        # mtp_enable with 2 args: hidden_states, nbatch_pack_offset
        # attention_mask = None, position_ids = None, position_embeddings = None
        self.assertIsNone(mask)
        self.assertIsNone(pos)
        self.assertIsNone(pe)
        # nbatch is the nbatch_pack_offset = offset
        self.assertIsNotNone(nbatch)

    def test_tuple_length_3_is_embed(self):
        """Test parse_args with 3-element tuple and is_embed=True."""
        parse_args = self._get_parse_args()
        x = paddle.randn([2, 4])
        mask = paddle.randn([2, 4])
        pos = paddle.randint(0, 10, [2, 4], dtype="int64")
        hidden, attention_mask, position_ids, pe, nbatch = parse_args((x, mask, pos), is_embed=True)
        self.assertIs(hidden, x)
        self.assertIs(attention_mask, mask)
        self.assertIs(position_ids, pos)
        self.assertIsNone(pe)
        self.assertIsNone(nbatch)

    def test_tuple_length_3_mtp_enable(self):
        """Test parse_args with 3-element tuple and mtp_enable=True."""
        parse_args = self._get_parse_args()
        x = paddle.randn([2, 4])
        mask = paddle.randn([2, 4])
        offset = paddle.randn([2, 4])
        hidden, attention_mask, position_ids, pe, nbatch = parse_args((x, mask, offset), mtp_enable=True)
        self.assertIs(hidden, x)
        self.assertIs(attention_mask, mask)
        # mtp_enable with 3 args: hidden_states, attention_mask, nbatch_pack_offset
        # position_ids = None, position_embeddings = None
        self.assertIsNone(position_ids)
        self.assertIsNone(pe)
        # nbatch is the third arg (offset) - assigned to nbatch_pack_offset
        self.assertIsNotNone(nbatch)

    def test_tuple_length_3_default(self):
        """Test parse_args with 3-element tuple (default mode)."""
        parse_args = self._get_parse_args()
        x = paddle.randn([2, 4])
        pos = paddle.randint(0, 10, [2, 4], dtype="int64")
        pe = paddle.randn([2, 4])
        hidden, mask, position_ids, position_embeddings, nbatch = parse_args((x, pos, pe))
        self.assertIs(hidden, x)
        self.assertIsNone(mask)
        self.assertIs(position_ids, pos)
        self.assertIs(position_embeddings, pe)
        self.assertIsNone(nbatch)

    def test_tuple_length_4(self):
        """Test parse_args with 4-element tuple."""
        parse_args = self._get_parse_args()
        x = paddle.randn([2, 4])
        mask = paddle.randn([2, 4])
        pos = paddle.randint(0, 10, [2, 4], dtype="int64")
        pe = paddle.randn([2, 4])
        hidden, attention_mask, position_ids, position_embeddings, nbatch = parse_args((x, mask, pos, pe))
        self.assertIs(hidden, x)
        self.assertIs(attention_mask, mask)
        self.assertIs(position_ids, pos)
        self.assertIs(position_embeddings, pe)
        self.assertIsNone(nbatch)

    def test_tuple_length_5(self):
        """Test parse_args with 5-element tuple."""
        parse_args = self._get_parse_args()
        x = paddle.randn([2, 4])
        mask = paddle.randn([2, 4])
        pos = paddle.randint(0, 10, [2, 4], dtype="int64")
        pe = paddle.randn([2, 4])
        nbatch = paddle.randn([2, 4])
        hidden, attention_mask, position_ids, position_embeddings, nbatch_out = parse_args((x, mask, pos, pe, nbatch))
        self.assertIs(hidden, x)
        self.assertIs(attention_mask, mask)
        self.assertIs(position_ids, pos)
        self.assertIs(position_embeddings, pe)
        self.assertIs(nbatch_out, nbatch)

    def test_stop_gradient_setters(self):
        """Test that stop_gradient is set on returned tensors when not None."""
        parse_args = self._get_parse_args()
        x = paddle.randn([2, 4])
        mask = paddle.randn([2, 4])
        pos = paddle.randint(0, 10, [2, 4], dtype="int64")
        pe = paddle.randn([2, 4])
        nbatch = paddle.randn([2, 4])

        # Make them require grad
        x.stop_gradient = False
        mask.stop_gradient = False
        pos.stop_gradient = False
        pe.stop_gradient = False
        nbatch.stop_gradient = False

        hidden, attention_mask, position_ids, position_embeddings, nbatch_out = parse_args((x, mask, pos, pe, nbatch))
        # position_ids should have stop_gradient=True
        self.assertTrue(position_ids.stop_gradient)
        self.assertTrue(position_embeddings.stop_gradient)
        self.assertTrue(attention_mask.stop_gradient)
        self.assertTrue(nbatch_out.stop_gradient)

    def test_stop_gradient_only_attention_mask(self):
        """Test that only non-None tensors get stop_gradient set."""
        parse_args = self._get_parse_args()
        x = paddle.randn([2, 4])
        mask = paddle.randn([2, 4])
        hidden, attention_mask, pos, pe, nbatch = parse_args((x, mask))
        self.assertTrue(attention_mask.stop_gradient)
        self.assertIsNone(pos)
        self.assertIsNone(pe)
        self.assertIsNone(nbatch)


class TestGetPPVPSplitLayers(unittest.TestCase):
    """Tests for get_pp_vp_split_layers function."""

    def _get_func(self):
        from paddleformers.nn.pp_model import get_pp_vp_split_layers

        return get_pp_vp_split_layers

    @patch("paddleformers.nn.pp_model.get_hcg")
    def test_pp_size_1_raises_assertion(self, mock_hcg):
        """Test that pp_size must be > 1."""
        mock_topo = MagicMock()
        mock_topo.get_dim_size.return_value = 1
        mock_hcg.return_value.get_pipe_parallel_world_size.return_value = 1

        config = MagicMock()
        config.num_hidden_layers = 4
        config.num_empty_layers_add_in_tail = 0
        config.virtual_pipeline_model_parallel_size = 1

        get_pp_vp_split_layers = self._get_func()
        with self.assertRaises(AssertionError):
            get_pp_vp_split_layers(config)

    @patch("paddleformers.nn.pp_model.get_hcg")
    def test_skip_recompute_num_zero(self, mock_hcg):
        """Test that skip_recompute_num=0 returns empty set."""
        MagicMock()
        mock_hcg.return_value.get_pipe_parallel_world_size.return_value = 4

        config = MagicMock()
        config.num_hidden_layers = 8
        config.num_empty_layers_add_in_tail = 0
        config.virtual_pipeline_model_parallel_size = 1

        get_pp_vp_split_layers = self._get_func()
        result = get_pp_vp_split_layers(config, skip_recompute_num=0)
        self.assertEqual(result, set())

    @patch("paddleformers.nn.pp_model.get_hcg")
    def test_vp_size_1_skip_positive(self, mock_hcg):
        """Test with vp_size=1 and positive skip returns all layers."""
        mock_hcg.return_value.get_pipe_parallel_world_size.return_value = 4

        config = MagicMock()
        config.num_hidden_layers = 8
        config.num_empty_layers_add_in_tail = 0
        config.virtual_pipeline_model_parallel_size = 1

        get_pp_vp_split_layers = self._get_func()
        result = get_pp_vp_split_layers(config, skip_recompute_num=1)
        self.assertEqual(result, set(range(8)))

    @patch("paddleformers.nn.pp_model.get_hcg")
    def test_vp_size_1_negative_skip(self, mock_hcg):
        """Test with vp_size=1 and negative skip_recompute_num defaults to vp_size."""
        mock_hcg.return_value.get_pipe_parallel_world_size.return_value = 4

        config = MagicMock()
        config.num_hidden_layers = 8
        config.num_empty_layers_add_in_tail = 0
        config.virtual_pipeline_model_parallel_size = 1

        get_pp_vp_split_layers = self._get_func()
        # skip_recompute_num=-1 defaults to vp_size=1, vp_size==1 and skip>0 returns all
        result = get_pp_vp_split_layers(config, skip_recompute_num=-1)
        self.assertEqual(result, set(range(8)))

    @patch("paddleformers.nn.pp_model.get_hcg")
    def test_vp_size_1_skip_negative_even(self, mock_hcg):
        """Test with vp_size=1 and large negative skip_recompute_num defaults to vp_size."""
        mock_hcg.return_value.get_pipe_parallel_world_size.return_value = 4

        config = MagicMock()
        config.num_hidden_layers = 8
        config.num_empty_layers_add_in_tail = 0
        config.virtual_pipeline_model_parallel_size = 1

        get_pp_vp_split_layers = self._get_func()
        # skip_recompute_num=-1 sets it to vp_size=1
        # But skip_recompute_num=-5 stays as -5, which is not 0 and not > 0
        # So with vp_size==1 and skip_recompute_num < 0, returns empty set
        result = get_pp_vp_split_layers(config, skip_recompute_num=-5)
        self.assertEqual(result, set())

    @patch("paddleformers.nn.pp_model.get_hcg")
    def test_default_skip_recompute_num(self, mock_hcg):
        """Test default skip_recompute_num=-1 uses vp_size."""
        mock_hcg.return_value.get_pipe_parallel_world_size.return_value = 2

        config = MagicMock()
        config.num_hidden_layers = 4
        config.num_empty_layers_add_in_tail = 0
        config.virtual_pipeline_model_parallel_size = 2

        get_pp_vp_split_layers = self._get_func()
        # skip_recompute_num defaults to vp_size=2
        result = get_pp_vp_split_layers(config)
        self.assertIsInstance(result, set)
        self.assertTrue(len(result) > 0)

    @patch("paddleformers.nn.pp_model.get_hcg")
    def test_with_empty_layers(self, mock_hcg):
        """Test with num_empty_layers_add_in_tail > 0."""
        mock_hcg.return_value.get_pipe_parallel_world_size.return_value = 2

        config = MagicMock()
        config.num_hidden_layers = 4
        config.num_empty_layers_add_in_tail = 2
        config.virtual_pipeline_model_parallel_size = 2

        get_pp_vp_split_layers = self._get_func()
        result = get_pp_vp_split_layers(config, skip_recompute_num=0)
        self.assertEqual(result, set())

    @patch("paddleformers.nn.pp_model.get_hcg")
    def test_layer_num_not_divisible_raises(self, mock_hcg):
        """Test that non-divisible layer_num raises assertion."""
        mock_hcg.return_value.get_pipe_parallel_world_size.return_value = 2

        config = MagicMock()
        config.num_hidden_layers = 5
        config.num_empty_layers_add_in_tail = 0
        config.virtual_pipeline_model_parallel_size = 2

        get_pp_vp_split_layers = self._get_func()
        # layer_num=5, pp*vp=4, 5%4 != 0
        with self.assertRaises(AssertionError):
            get_pp_vp_split_layers(config, skip_recompute_num=1)


class TestGetAttr(unittest.TestCase):
    """Tests for get_attr function."""

    def _get_func(self):
        from paddleformers.nn.pp_model import get_attr

        return get_attr

    def test_get_attr_direct(self):
        """Test get_attr when attribute exists on the layer directly."""
        get_attr = self._get_func()
        layer = nn.Linear(10, 10)
        result = get_attr(layer, "weight")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, paddle.Tensor)

    def test_get_attr_nested(self):
        """Test get_attr recursively searching inner layers."""
        get_attr = self._get_func()

        class InnerLayer(nn.Layer):
            def __init__(self):
                super().__init__()
                self.my_attr = paddle.randn([4])

        class OuterLayer(nn.Layer):
            def __init__(self):
                super().__init__()
                self._layer = InnerLayer()

        outer = OuterLayer()
        result = get_attr(outer, "my_attr")
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, [4])

    def test_get_attr_missing(self):
        """Test get_attr logic: getattr returns None triggers recursive search."""
        get_attr = self._get_func()

        # Test the core logic: getattr(layer, name, None) returns None
        # then get_attr(layer._layer, name) would be called
        # We verify the logic by testing with an object that has the attr directly
        class DirectAttr:
            def __init__(self):
                self.my_prop = 42

        result = get_attr(DirectAttr(), "my_prop")
        self.assertEqual(result, 42)

        # Test that if getattr returns None and _layer doesn't have it either,
        # the original function would recurse. We just verify the direct case.
        class NoAttr:
            pass

        # For a layer without _layer, accessing _layer would raise AttributeError
        layer = NoAttr()
        with self.assertRaises(AttributeError):
            get_attr(layer, "anything")


class TestRotaryEmbedding(unittest.TestCase):
    """Tests for RotaryEmbedding class."""

    def _get_cls(self):
        from paddleformers.nn.pp_model import RotaryEmbedding

        return RotaryEmbedding

    def test_init_with_config(self):
        """Test RotaryEmbedding initialization."""
        config = MagicMock()
        config.hidden_size = 64
        config.num_attention_heads = 4
        config.rope_theta = 10000.0
        config.head_dim = 16

        cls = self._get_cls()
        emb = cls(config)
        self.assertEqual(emb.head_dim, 16)
        self.assertEqual(emb.base, 10000.0)

    def test_init_compute_head_dim_from_config(self):
        """Test RotaryEmbedding init computing head_dim from hidden_size and num_heads."""
        cls = self._get_cls()
        # Create a config where head_dim attribute doesn't exist at all,
        # so getattr falls back to the default value

        class ConfigNoHeadDim:
            hidden_size = 64
            num_attention_heads = 4
            rope_theta = 10000.0

        config = ConfigNoHeadDim()
        emb = cls(config)
        # head_dim should default to hidden_size // num_attention_heads = 16
        self.assertEqual(emb.head_dim, 16)

    def test_forward_shape(self):
        """Test that forward returns cos and sin of correct shape."""
        config = MagicMock()
        config.hidden_size = 64
        config.num_attention_heads = 4
        config.rope_theta = 10000.0
        config.head_dim = 16

        cls = self._get_cls()
        emb = cls(config)
        x = paddle.randn([2, 8, 16])
        position_ids = paddle.arange(0, 8, dtype="int64").unsqueeze(0).expand([2, 8])
        cos, sin = emb(x, position_ids)
        # cos and sin shape should be [2, 8, 16]
        self.assertEqual(cos.shape, [2, 8, 16])
        self.assertEqual(sin.shape, [2, 8, 16])


class TestEmptyLayer(unittest.TestCase):
    """Tests for EmptyLayer class."""

    def _get_cls(self):
        from paddleformers.nn.pp_model import EmptyLayer

        return EmptyLayer

    def test_forward_returns_same_value(self):
        """Test EmptyLayer returns input unchanged."""
        cls = self._get_cls()
        layer = cls()
        x = paddle.randn([2, 4])
        out = layer(x)
        # EmptyLayer just returns x, so values should be identical
        np.testing.assert_allclose(out.numpy(), x.numpy())

    def test_forward_preserves_shape(self):
        """Test EmptyLayer preserves tensor shape."""
        cls = self._get_cls()
        layer = cls()
        x = paddle.randn([3, 5, 7])
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_different_dtypes(self):
        """Test EmptyLayer works with different dtypes."""
        cls = self._get_cls()
        layer = cls()
        x = paddle.randn([2, 3], dtype="float64")
        out = layer(x)
        self.assertEqual(out.dtype, x.dtype)


class TestGeneralModelForCausalLMPipe(unittest.TestCase):
    """Tests for GeneralModelForCausalLMPipe class."""

    def test_decoder_layer_cls_not_set_raises(self):
        """Test that ValueError is raised when _decoder_layer_cls is None."""
        from paddleformers.nn.pp_model import GeneralModelForCausalLMPipe

        config = MagicMock()
        config.get.return_value = None
        config.sliding_window = None
        config.layer_types = []
        config.tie_word_embeddings = False
        config.sequence_parallel = False
        config.num_hidden_layers = 2
        config.num_empty_layers_add_in_tail = 0
        config.virtual_pipeline_model_parallel_size = 1
        config.initializer_range = 0.02
        config.hidden_size = 64
        config.moe_group = "dummy"

        with patch("paddleformers.nn.pp_model.get_hcg") as mock_hcg, patch(
            "paddleformers.nn.pp_model.get_pp_vp_split_layers"
        ) as mock_split:
            mock_topo = MagicMock()
            mock_topo.get_dim_size.return_value = 1
            mock_hcg.return_value.get_pipe_parallel_world_size.return_value = 4
            mock_hcg.return_value.get_model_parallel_world_size.return_value = 1
            mock_hcg.return_value.get_model_parallel_rank.return_value = 0
            mock_hcg.return_value.topology.return_value = mock_topo
            mock_split.return_value = set()

            with self.assertRaises(ValueError) as ctx:
                GeneralModelForCausalLMPipe(config)
            self.assertIn("_decoder_layer_cls", str(ctx.exception))

    def test_register_cls_attr(self):
        """Test register_cls_attr class method."""
        from paddleformers.nn.pp_model import GeneralModelForCausalLMPipe

        class DummyConfig:
            pass

        class DummyModel:
            _get_tensor_parallel_mappings = "tp_mappings"
            _init_weights = "init_weights"
            _keep_in_fp32_modules = ["layernorm"]
            transpose_weight_keys = ["weight"]

        GeneralModelForCausalLMPipe.register_cls_attr(
            config_class=DummyConfig,
            pretrained_model_class=DummyModel,
        )

        self.assertEqual(GeneralModelForCausalLMPipe.config_class, DummyConfig)
        self.assertEqual(GeneralModelForCausalLMPipe._get_tensor_parallel_mappings, "tp_mappings")
        self.assertEqual(GeneralModelForCausalLMPipe._init_weights, "init_weights")
        self.assertEqual(GeneralModelForCausalLMPipe._keep_in_fp32_modules, ["layernorm"])
        self.assertEqual(GeneralModelForCausalLMPipe.transpose_weight_keys, ["weight"])

    def test_register_cls_attr_partial(self):
        """Test register_cls_attr with only config_class."""
        from paddleformers.nn.pp_model import GeneralModelForCausalLMPipe

        class DummyConfig2:
            pass

        GeneralModelForCausalLMPipe.register_cls_attr(config_class=DummyConfig2)
        self.assertEqual(GeneralModelForCausalLMPipe.config_class, DummyConfig2)

    def test_register_cls_attr_with_fuse_split(self):
        """Test register_cls_attr with _get_fuse_or_split_param_mappings."""
        from paddleformers.nn.pp_model import GeneralModelForCausalLMPipe

        class DummyConfig3:
            pass

        class DummyModel3:
            _get_fuse_or_split_param_mappings = "fuse_split"

        GeneralModelForCausalLMPipe.register_cls_attr(
            config_class=DummyConfig3,
            pretrained_model_class=DummyModel3,
        )
        self.assertEqual(GeneralModelForCausalLMPipe._get_fuse_or_split_param_mappings, "fuse_split")

    def test_prepare_pipeline_inputs_func_dict(self):
        """Test _prepare_pipeline_inputs_func with dict input containing attention_mask."""
        from paddleformers.nn.pp_model import GeneralModelForCausalLMPipe

        inputs = {
            "input_ids": paddle.randint(0, 100, [2, 8]),
            "attention_mask": paddle.randint(0, 2, [2, 8]).astype("float32"),
            "position_ids": paddle.randint(0, 8, [2, 8], dtype="int64"),
            "labels": paddle.randint(0, 100, [2, 8]),
        }
        result = GeneralModelForCausalLMPipe._prepare_pipeline_inputs_func(inputs)
        self.assertEqual(len(result), 2)

    def test_prepare_pipeline_inputs_func_dict_attn_mask_startend(self):
        """Test _prepare_pipeline_inputs_func with dict containing attn_mask_startend_row_indices."""
        from paddleformers.nn.pp_model import GeneralModelForCausalLMPipe

        inputs = {
            "input_ids": paddle.randint(0, 100, [2, 8]),
            "attn_mask_startend_row_indices": paddle.randint(0, 8, [2, 2, 8], dtype="int32"),
            "position_ids": paddle.randint(0, 8, [2, 8], dtype="int64"),
            "labels": paddle.randint(0, 100, [2, 8]),
        }
        result = GeneralModelForCausalLMPipe._prepare_pipeline_inputs_func(inputs)
        self.assertEqual(len(result), 2)

    def test_prepare_pipeline_inputs_func_dict_default(self):
        """Test _prepare_pipeline_inputs_func with dict without attention_mask."""
        from paddleformers.nn.pp_model import GeneralModelForCausalLMPipe

        inputs = {
            "input_ids": paddle.randint(0, 100, [2, 8]),
            "attn_mask_startend_row_indices": paddle.randint(0, 8, [2, 2, 8], dtype="int32"),
            "labels": paddle.randint(0, 100, [2, 8]),
        }
        result = GeneralModelForCausalLMPipe._prepare_pipeline_inputs_func(inputs)
        self.assertEqual(len(result), 2)

    def test_prepare_pipeline_inputs_func_list(self):
        """Test _prepare_pipeline_inputs_func with list of dicts."""
        from paddleformers.nn.pp_model import GeneralModelForCausalLMPipe

        inputs = [
            {
                "input_ids": paddle.randint(0, 100, [2, 8]),
                "attention_mask": paddle.randint(0, 2, [2, 8]).astype("float32"),
                "labels": paddle.randint(0, 100, [2, 8]),
            },
            {
                "input_ids": paddle.randint(0, 100, [2, 8]),
                "attention_mask": paddle.randint(0, 2, [2, 8]).astype("float32"),
                "labels": paddle.randint(0, 100, [2, 8]),
            },
        ]
        result = GeneralModelForCausalLMPipe._prepare_pipeline_inputs_func(inputs)
        self.assertEqual(len(result), 2)

    def test_prepare_pipeline_inputs_func_ordered_dict(self):
        """Test _prepare_pipeline_inputs_func with dict input containing attention_mask."""
        from paddleformers.nn.pp_model import GeneralModelForCausalLMPipe

        # Note: pp_model imports OrderedDict from typing module,
        # so a collections.OrderedDict won't match the isinstance check.
        # Use a regular dict instead to test the dict branch.
        inputs = {
            "input_ids": paddle.randint(0, 100, [2, 8]),
            "attention_mask": paddle.randint(0, 2, [2, 8]).astype("float32"),
            "labels": paddle.randint(0, 100, [2, 8]),
        }
        result = GeneralModelForCausalLMPipe._prepare_pipeline_inputs_func(inputs)
        self.assertEqual(len(result), 2)

    def test_tied_weights_keys(self):
        """Test that _tied_weights_keys is set correctly."""
        from paddleformers.nn.pp_model import GeneralModelForCausalLMPipe

        self.assertEqual(GeneralModelForCausalLMPipe._tied_weights_keys, ["lm_head.weight"])


class TestRMSNormPipe(unittest.TestCase):
    """Tests for RMSNormPipe class."""

    def test_init_with_sequence_parallel(self):
        """Test RMSNormPipe initialization with sequence_parallel=True."""
        from paddleformers.nn.pp_model import RMSNormPipe

        config = MagicMock()
        config.hidden_size = 64
        config.sequence_parallel = True

        with patch.object(RMSNormPipe, "enable_sequence_parallel"):
            layer = RMSNormPipe(config)
            self.assertTrue(hasattr(layer, "config"))

    def test_init_without_sequence_parallel(self):
        """Test RMSNormPipe initialization with sequence_parallel=False."""
        from paddleformers.nn.pp_model import RMSNormPipe

        config = MagicMock()
        config.hidden_size = 64
        config.sequence_parallel = False

        layer = RMSNormPipe(config)
        self.assertIsNotNone(layer)


class TestLayerNormPipe(unittest.TestCase):
    """Tests for LayerNormPipe class."""

    def test_init_with_sequence_parallel(self):
        """Test LayerNormPipe initialization with sequence_parallel=True."""
        from paddleformers.nn.pp_model import LayerNormPipe

        config = MagicMock()
        config.hidden_size = 64
        config.sequence_parallel = True

        with patch.object(LayerNormPipe, "enable_sequence_parallel"):
            layer = LayerNormPipe(config)
            self.assertIsNotNone(layer)

    def test_init_without_sequence_parallel(self):
        """Test LayerNormPipe initialization with sequence_parallel=False."""
        from paddleformers.nn.pp_model import LayerNormPipe

        config = MagicMock()
        config.hidden_size = 64
        config.sequence_parallel = False

        layer = LayerNormPipe(config)
        self.assertIsNotNone(layer)


class TestLMHeadPipe(unittest.TestCase):
    """Tests for LMHeadPipe class."""

    def test_embedding_weight_property(self):
        """Test embedding_weight property."""
        from paddleformers.nn.pp_model import LMHeadPipe

        # Create a mock layer that has a weight attribute
        weight_tensor = paddle.randn([100, 64])

        with patch("paddleformers.nn.pp_model.get_attr", return_value=weight_tensor):
            head = LMHeadPipe.__new__(LMHeadPipe)
            result = head.embedding_weight
            self.assertIsNotNone(result)


class TestMakeDecoderLayerPipe(unittest.TestCase):
    """Tests for make_decoder_layer_pipe function."""

    def test_returns_type_with_correct_name(self):
        """Test that make_decoder_layer_pipe creates a class named DecoderLayerPipe."""
        from paddleformers.nn.pp_model import make_decoder_layer_pipe

        class DummyDecoderLayer(nn.Layer):
            def __init__(self, config, layer_idx=0):
                super().__init__()
                self.config = config
                self.layer_idx = layer_idx

            def forward(self, *args, **kwargs):
                return paddle.randn([2, 4])

        Cls = make_decoder_layer_pipe(DummyDecoderLayer)
        self.assertEqual(Cls.__name__, "DecoderLayerPipe")
        self.assertTrue(issubclass(Cls, DummyDecoderLayer))

    def test_forward_single_tensor(self):
        """Test DecoderLayerPipe forward with single tensor input."""
        from paddleformers.nn.pp_model import make_decoder_layer_pipe

        class SimpleDecoder(nn.Layer):
            def __init__(self, config, layer_idx=0):
                super().__init__()
                self.config = config

            def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
                return hidden_states * 2

        config = MagicMock()
        config.get.return_value = 0
        config.sequence_parallel = False

        Cls = make_decoder_layer_pipe(SimpleDecoder)
        layer = Cls(config=config, layer_idx=0)
        x = paddle.randn([2, 4])
        result = layer(x)
        self.assertEqual(result.shape, [2, 4])

    def test_forward_with_int32_attention_mask(self):
        """Test DecoderLayerPipe forward with int32 attention mask (startend_row_indices path)."""
        from paddleformers.nn.pp_model import make_decoder_layer_pipe

        class MaskDecoder(nn.Layer):
            def __init__(self, config, layer_idx=0):
                super().__init__()
                self.config = config

            def forward(self, hidden_states, attention_mask=None, attn_mask_startend_row_indices=None, **kwargs):
                return hidden_states

        config = MagicMock()
        config.get.return_value = 0
        config.sequence_parallel = False

        Cls = make_decoder_layer_pipe(MaskDecoder)
        layer = Cls(config=config, layer_idx=0)
        x = paddle.randn([2, 8])
        # Provide proper 3D int32 startend_row_indices mask [batch, 2, max_seq_len]
        mask = paddle.randint(0, 8, [2, 2, 8], dtype="int32")
        result = layer((x, mask))
        # When attention_mask is not None, result is a tuple
        self.assertIsInstance(result, tuple)
        # First element should be the hidden states tensor
        self.assertIsInstance(result[0], paddle.Tensor)


class TestCriterionLayerPipe(unittest.TestCase):
    """Tests for CriterionLayerPipe class."""

    def test_return_tuple_default(self):
        """Test that CriterionLayerPipe has return_tuple set after init."""
        from paddleformers.nn.pp_model import CriterionLayerPipe

        config = MagicMock()
        config.loss_type = "cross_entropy"
        config.label_smoothing = 0.0
        config.dtype = "float32"

        # CriterionLayerPipe sets self.return_tuple = False in __init__
        # We need to mock the parent __init__ to avoid full init
        with patch("paddleformers.nn.pp_model.CriterionLayer.__init__", return_value=None):
            layer = CriterionLayerPipe(config)
            self.assertFalse(layer.return_tuple)


if __name__ == "__main__":
    unittest.main()
