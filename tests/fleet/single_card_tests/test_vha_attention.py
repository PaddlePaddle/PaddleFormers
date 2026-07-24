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

"""Unit tests for SelfAttentionVHA (Virtual Head Attention).

Covers: SelfAttentionVHA init, _apply_vha_premix, _apply_vha_postmix,
_post_core_attention_hook, _get_qkv_vha, backward_dw, SelfAttentionVHASublayersSpec.
Also covers: DotProductAttention.expand_attn_mask_startend_row_indices_for_cp,
get_doc_lens, get_doc_starts, startend_row_indices_add_sliding_window (num_vec=2).
"""

import unittest

import paddle

from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.attention import (
    SelfAttentionVHA,
    SelfAttentionVHASublayersSpec,
)
from paddleformers.fleet.transformer.dot_product_attention import (
    DotProductAttention,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.transformer.utils import (
    get_doc_lens,
    get_doc_starts,
    startend_row_indices_add_sliding_window,
)
from paddleformers.fleet.utils import (
    init_method_normal,
    scaled_init_method_normal,
)

strategy = paddle.distributed.fleet.DistributedStrategy()
initialize_fleet(strategy=strategy)


class BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x), self.linear.bias

    def backward_dw(self):
        pass


class RMSNorm(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_size = kwargs.get("normalized_shape", kwargs.get("hidden_size"))
        eps = kwargs.get("norm_eps", kwargs.get("eps"))
        self.weight = paddle.nn.Parameter(paddle.ones([hidden_size]))
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


class TestSelfAttentionVHA(unittest.TestCase):
    """Tests for SelfAttentionVHA layer."""

    def _make_config(self, gated=False, per_layer_norm=False):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=32,
            v_head_dim=32,
            use_vha_attention=True,
            vha_q_lora_rank=32,
            vha_postmix_rank=4,
        )
        config.softmax_scale = None
        config.use_bias = False
        config.attention_bias = False
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.gpt_model_use_experimental_version = False
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.gated_attention = gated
        config.attention_value_scale = None
        config.sliding_window = None
        config.window_attn_skip_freq = None
        if per_layer_norm:
            config.qk_norm_type = "per_layer"
        return config

    def _build_vha(self, config, layer_number=0):
        return SelfAttentionVHA(
            config,
            SelfAttentionVHASublayersSpec(
                q_proj=BiasedLinear,
                k_proj=BiasedLinear,
                v_proj=BiasedLinear,
                gate_proj=BiasedLinear if config.gated_attention else None,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=layer_number,
        )

    def test_init_basic(self):
        config = self._make_config()
        attn = self._build_vha(config)
        self.assertFalse(attn.is_swa)
        self.assertEqual(attn.q_head_dim, 32)
        self.assertEqual(attn.num_attention_heads, 8)
        self.assertEqual(attn.num_key_value_heads, 2)
        self.assertEqual(attn.vha_premix_weight.shape, [2, 32, 32])
        self.assertEqual(attn.vha_postmix_U.shape, [8, 4])
        self.assertEqual(attn.vha_postmix_V.shape, [8, 4])

    def test_init_gated(self):
        config = self._make_config(gated=True)
        attn = self._build_vha(config)
        self.assertIsNotNone(attn.gate_proj)

    def test_premix_shape(self):
        config = self._make_config()
        attn = self._build_vha(config)
        query = paddle.randn([2, 16, 4, 32])
        out = attn._apply_vha_premix(query)
        self.assertEqual(list(out.shape), [2, 16, 8, 32])

    def test_postmix_shape(self):
        config = self._make_config()
        attn = self._build_vha(config)
        attn_out = paddle.randn([2, 16, 256])
        out = attn._apply_vha_postmix(attn_out)
        self.assertEqual(list(out.shape), [2, 16, 256])

    def test_post_core_attention_hook(self):
        config = self._make_config()
        attn = self._build_vha(config)
        attn_out = paddle.randn([2, 16, 256])
        out = attn._post_core_attention_hook(attn_out)
        self.assertEqual(list(out.shape), [2, 16, 256])

    def test_get_qkv_vha_shapes(self):
        config = self._make_config()
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        q, k, v = attn._get_qkv_vha(hidden)
        self.assertEqual(list(q.shape), [2, 16, 8, 32])
        self.assertEqual(list(k.shape), [2, 16, 2, 32])
        self.assertEqual(list(v.shape), [2, 16, 2, 32])

    def test_get_qkv_vha_gated(self):
        config = self._make_config(gated=True)
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        result = attn._get_qkv_vha(hidden)
        self.assertEqual(len(result), 4)
        q, k, v, gate = result
        self.assertEqual(list(gate.shape), [2, 16, 256])

    def test_get_qkv_vha_per_layer_norm(self):
        config = self._make_config(per_layer_norm=True)
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        q, k, v = attn._get_qkv_vha(hidden)
        self.assertEqual(list(q.shape), [2, 16, 8, 32])
        self.assertEqual(list(k.shape), [2, 16, 2, 32])

    def test_forward_shape(self):
        config = self._make_config()
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        rotary_pos_emb = paddle.randn([1, 16, 1, 32])
        output, bias = attn(
            hidden, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )
        self.assertEqual(list(output.shape), [2, 16, 256])

    def test_backward_gradients(self):
        config = self._make_config()
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        hidden.stop_gradient = False
        rotary_pos_emb = paddle.randn([1, 16, 1, 32])
        output, bias = attn(
            hidden, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(attn.vha_premix_weight.grad)
        self.assertIsNotNone(attn.vha_postmix_V.grad)

    def test_backward_dw(self):
        config = self._make_config()
        attn = self._build_vha(config)
        attn.backward_dw()

    def test_backward_dw_gated(self):
        config = self._make_config(gated=True)
        attn = self._build_vha(config)
        attn.backward_dw()

    def test_sublayers_spec_dataclass(self):
        spec = SelfAttentionVHASublayersSpec()
        self.assertIsNone(spec.q_proj)
        self.assertIsNone(spec.k_proj)
        self.assertIsNone(spec.gate_proj)
        self.assertIsNone(spec.core_attention)

    def test_no_norm(self):
        config = self._make_config()
        attn = SelfAttentionVHA(
            config,
            SelfAttentionVHASublayersSpec(
                q_proj=BiasedLinear,
                k_proj=BiasedLinear,
                v_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=None,
                k_norm=None,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,
        )
        self.assertIsNone(attn.q_norm)
        self.assertIsNone(attn.k_norm)
        hidden = paddle.randn([2, 8, 256])
        q, k, v = attn._get_qkv_vha(hidden)
        self.assertEqual(list(q.shape), [2, 8, 8, 32])

    def test_orthogonal_init_path(self):
        config = self._make_config()
        config.vha_q_lora_rank = 16
        attn = SelfAttentionVHA(
            config,
            SelfAttentionVHASublayersSpec(
                q_proj=BiasedLinear,
                k_proj=BiasedLinear,
                v_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,
        )
        self.assertEqual(attn.q_head_dim, 16)
        self.assertEqual(attn.vha_premix_weight.shape, [2, 16, 32])

    def test_postmix_rank_default(self):
        config = self._make_config()
        config.vha_postmix_rank = None
        attn = SelfAttentionVHA(
            config,
            SelfAttentionVHASublayersSpec(
                q_proj=BiasedLinear,
                k_proj=BiasedLinear,
                v_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,
        )
        self.assertEqual(attn.vha_postmix_U.shape, [8, 2])

    def test_v_scale(self):
        config = self._make_config()
        config.attention_value_scale = 0.5
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 8, 256])
        q, k, v = attn._get_qkv_vha(hidden)
        self.assertEqual(list(v.shape), [2, 8, 2, 32])

    def test_vha_debug_env(self):
        import os

        os.environ["VHA_DEBUG"] = "1"
        try:
            config = self._make_config()
            attn = self._build_vha(config)
            hidden = paddle.randn([2, 8, 256])
            attn._get_qkv_vha(hidden)
        finally:
            del os.environ["VHA_DEBUG"]

    def test_swa_vha_init(self):
        config = self._make_config()
        config.sliding_window = (4096, 0)
        config.window_attn_skip_freq = [1, 1, 1, 1]
        config.swa_vha_q_lora_rank = 32
        config.swa_vha_postmix_rank = 2
        attn = SelfAttentionVHA(
            config,
            SelfAttentionVHASublayersSpec(
                q_proj=BiasedLinear,
                k_proj=BiasedLinear,
                v_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,
        )
        self.assertTrue(attn.is_swa)
        self.assertEqual(attn.q_head_dim, 32)


class TestSelfAttentionExperimentalVersion(unittest.TestCase):
    """Tests for SelfAttention with gpt_model_use_experimental_version=True."""

    def _make_config(self, gated=False):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=32,
            v_head_dim=32,
            gpt_model_use_experimental_version=True,
        )
        config.softmax_scale = None
        config.use_bias = True
        config.attention_bias = False
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.max_sequence_length = 16
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.gated_attention = gated
        config.attention_value_scale = None
        config.sliding_window = None
        config.window_attn_skip_freq = None
        return config

    def test_experimental_init(self):
        from paddleformers.fleet.transformer.attention import (
            SelfAttention,
            SelfAttentionSublayersSpec,
        )

        config = self._make_config()
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,
        )
        self.assertIsNotNone(attn.qkv_proj)

    def test_experimental_gated_init(self):
        from paddleformers.fleet.transformer.attention import (
            SelfAttention,
            SelfAttentionSublayersSpec,
        )

        config = self._make_config(gated=True)
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                gate_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,
        )
        self.assertIsNotNone(attn.gate_proj)

    def test_experimental_get_qkv(self):
        from paddleformers.fleet.transformer.attention import (
            SelfAttention,
            SelfAttentionSublayersSpec,
        )

        config = self._make_config()
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,
        )
        hidden = paddle.randn([2, 16, 256])
        q, k, v = attn.get_query_key_value_tensors(hidden)
        self.assertEqual(list(q.shape), [2, 16, 8, 32])
        self.assertEqual(list(k.shape), [2, 16, 2, 32])

    def test_experimental_gated_get_qkv(self):
        from paddleformers.fleet.transformer.attention import (
            SelfAttention,
            SelfAttentionSublayersSpec,
        )

        config = self._make_config(gated=True)
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                gate_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=0,
        )
        hidden = paddle.randn([2, 16, 256])
        result = attn.get_query_key_value_tensors(hidden)
        self.assertEqual(len(result), 4)


class TestExpandAttnMaskForCP(unittest.TestCase):
    """Tests for DotProductAttention.expand_attn_mask_startend_row_indices_for_cp."""

    def _make_dpa(self, cp_size=8, experimental_dataflow=False):
        from unittest.mock import patch as mock_patch

        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,
            context_parallel_size=cp_size,
            experimental_dataflow=experimental_dataflow,
        )
        config.attention_dropout = 0.0
        config.flashmask_use_varlen = False
        with mock_patch(
            "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size",
            return_value=cp_size,
        ):
            dpa = DotProductAttention(
                config=config,
                layer_number=0,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
            )
        return dpa

    def test_none_mask_creates_full_causal(self):
        dpa = self._make_dpa(cp_size=4)
        key = paddle.randn([2, 16, 2, 32])
        result = dpa.expand_attn_mask_startend_row_indices_for_cp(None, key)
        self.assertEqual(result.shape[-1], 2)
        self.assertEqual(result.shape[2], 64)

    def test_dim1_expands_to_dim2(self):
        dpa = self._make_dpa(cp_size=4)
        key = paddle.randn([2, 16, 2, 32])
        mask = paddle.full([2, 1, 64, 1], fill_value=64, dtype=paddle.int32)
        result = dpa.expand_attn_mask_startend_row_indices_for_cp(mask, key)
        self.assertEqual(result.shape[-1], 2)

    def test_dim2_experimental_passthrough(self):
        dpa = self._make_dpa(cp_size=4, experimental_dataflow=True)
        key = paddle.randn([2, 16, 2, 32])
        mask = paddle.full([2, 1, 64, 2], fill_value=64, dtype=paddle.int32)
        result = dpa.expand_attn_mask_startend_row_indices_for_cp(mask, key)
        self.assertTrue(paddle.equal_all(result, mask).item())

    def test_invalid_dim_raises(self):
        dpa = self._make_dpa(cp_size=4, experimental_dataflow=False)
        key = paddle.randn([2, 16, 2, 32])
        mask = paddle.full([2, 1, 64, 3], fill_value=64, dtype=paddle.int32)
        with self.assertRaises(ValueError):
            dpa.expand_attn_mask_startend_row_indices_for_cp(mask, key)


class TestDocMaskUtils(unittest.TestCase):
    """Tests for get_doc_lens and get_doc_starts."""

    def test_get_doc_lens_single_doc(self):
        mask = paddle.to_tensor([4, 4, 4, 4], dtype=paddle.int32).reshape(
            [1, 1, 4, 1]
        )
        doc_lens = get_doc_lens(mask)
        self.assertEqual(doc_lens.numpy().tolist(), [4])

    def test_get_doc_lens_multi_doc(self):
        mask = paddle.to_tensor([3, 3, 3, 6, 6, 6], dtype=paddle.int32).reshape(
            [1, 1, 6, 1]
        )
        doc_lens = get_doc_lens(mask)
        self.assertEqual(doc_lens.numpy().tolist(), [3, 3])

    def test_get_doc_starts(self):
        doc_lens = paddle.to_tensor([3, 4, 2], dtype=paddle.int32)
        starts = get_doc_starts(doc_lens)
        self.assertEqual(starts.numpy().tolist(), [0, 3, 7])

    def test_get_doc_starts_single(self):
        doc_lens = paddle.to_tensor([5], dtype=paddle.int32)
        starts = get_doc_starts(doc_lens)
        self.assertEqual(starts.numpy().tolist(), [0])


class TestStartendRowIndicesNumVec2(unittest.TestCase):
    """Test startend_row_indices_add_sliding_window with num_vec=2."""

    def test_num_vec_2(self):
        bsz, seq, kv_num_heads = 1, 8, 2
        window_size = 3
        indices = paddle.ones([bsz, 1, seq, 2], dtype=paddle.int32) * 10000
        result = startend_row_indices_add_sliding_window(
            indices, (window_size, 0), 0.0, kv_num_heads
        )
        self.assertEqual(result.shape[-1], 2)
        self.assertEqual(result.shape[1], kv_num_heads)


class TestSelfAttentionVHASharedKV(unittest.TestCase):
    """Tests for SharedKV feature in SelfAttentionVHA."""

    def _make_config(self, gated=False, rotary_percent=1.0):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=32,
            v_head_dim=32,
            use_vha_attention=True,
            vha_q_lora_rank=32,
            vha_postmix_rank=4,
            vha_shared_kv=True,
        )
        config.softmax_scale = None
        config.use_bias = False
        config.attention_bias = False
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        config.gpt_model_use_experimental_version = False
        config.rms_norm_eps = 1e-5
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = False
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"
        config.gated_attention = gated
        config.attention_value_scale = None
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.rotary_percent = rotary_percent
        config.sequence_parallel = False
        return config

    def _build_vha(self, config, layer_number=0):
        return SelfAttentionVHA(
            config,
            SelfAttentionVHASublayersSpec(
                q_proj=BiasedLinear,
                k_proj=BiasedLinear,
                v_proj=BiasedLinear,
                gate_proj=BiasedLinear if config.gated_attention else None,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=layer_number,
        )

    def test_shared_kv_init(self):
        config = self._make_config()
        attn = self._build_vha(config)
        self.assertTrue(attn.shared_kv)
        self.assertTrue(hasattr(attn, "shared_kv_proj"))
        self.assertFalse(hasattr(attn, "k_proj"))
        self.assertFalse(hasattr(attn, "v_proj"))

    def test_shared_kv_init_asserts_head_dim_eq_v_head_dim(self):
        config = self._make_config()
        config.v_head_dim = 64  # mismatch with head_dim=32
        with self.assertRaises(AssertionError):
            self._build_vha(config)

    def test_get_qkv_shared_kv_returns_2(self):
        config = self._make_config()
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        result = attn._get_qkv_vha(hidden)
        self.assertEqual(len(result), 2)
        query, shared_kv = result
        self.assertEqual(list(query.shape), [2, 16, 8, 32])
        self.assertEqual(list(shared_kv.shape), [2, 16, 2, 32])

    def test_get_qkv_shared_kv_gated_returns_3(self):
        config = self._make_config(gated=True)
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        result = attn._get_qkv_vha(hidden)
        self.assertEqual(len(result), 3)
        query, shared_kv, gate = result
        self.assertEqual(list(query.shape), [2, 16, 8, 32])
        self.assertEqual(list(shared_kv.shape), [2, 16, 2, 32])

    def test_forward_shared_kv(self):
        config = self._make_config()
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        rotary_pos_emb = paddle.randn([1, 16, 1, 32])
        output, bias = attn(
            hidden, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )
        self.assertEqual(list(output.shape), [2, 16, 256])

    def test_forward_shared_kv_partial_rope(self):
        config = self._make_config(rotary_percent=0.5)
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        rotary_pos_emb = paddle.randn([1, 16, 1, 16])
        output, bias = attn(
            hidden, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )
        self.assertEqual(list(output.shape), [2, 16, 256])

    def test_forward_shared_kv_gated(self):
        config = self._make_config(gated=True)
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        rotary_pos_emb = paddle.randn([1, 16, 1, 32])
        output, bias = attn(
            hidden, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )
        self.assertEqual(list(output.shape), [2, 16, 256])

    def test_backward_shared_kv(self):
        config = self._make_config()
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        hidden.stop_gradient = False
        rotary_pos_emb = paddle.randn([1, 16, 1, 32])
        output, bias = attn(
            hidden, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(hidden.grad)

    def test_backward_dw_shared_kv(self):
        config = self._make_config()
        attn = self._build_vha(config)
        attn.backward_dw()

    def test_backward_dw_shared_kv_gated(self):
        config = self._make_config(gated=True)
        attn = self._build_vha(config)
        attn.backward_dw()

    def test_apply_shared_kv_inverse_rope_full(self):
        config = self._make_config()
        attn = self._build_vha(config)
        core_attn_out = paddle.randn([2, 16, 256])
        k_pos_emb = paddle.randn([1, 16, 1, 32])
        result = attn._apply_shared_kv_inverse_rope(
            core_attn_out, k_pos_emb, None, None, None
        )
        self.assertEqual(list(result.shape), [2, 16, 256])

    def test_apply_shared_kv_inverse_rope_partial(self):
        config = self._make_config(rotary_percent=0.5)
        attn = self._build_vha(config)
        core_attn_out = paddle.randn([2, 16, 256])
        k_pos_emb = paddle.randn([1, 16, 1, 16])
        result = attn._apply_shared_kv_inverse_rope(
            core_attn_out, k_pos_emb, None, None, None
        )
        self.assertEqual(list(result.shape), [2, 16, 256])

    def test_shared_kv_debug_env(self):
        import os

        os.environ["VHA_DEBUG"] = "1"
        try:
            config = self._make_config()
            attn = self._build_vha(config)
            hidden = paddle.randn([2, 8, 256])
            attn._get_qkv_vha(hidden)
        finally:
            del os.environ["VHA_DEBUG"]

    def test_shared_kv_sublayers_spec_field(self):
        spec = SelfAttentionVHASublayersSpec(shared_kv_proj=BiasedLinear)
        self.assertEqual(spec.shared_kv_proj, BiasedLinear)

    def test_shared_kv_per_layer_norm(self):
        config = self._make_config()
        config.qk_norm_type = "per_layer"
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        result = attn._get_qkv_vha(hidden)
        self.assertEqual(len(result), 2)
        query, shared_kv = result
        self.assertEqual(list(query.shape), [2, 16, 8, 32])

    def test_forward_shared_kv_with_attention_mask(self):
        config = self._make_config()
        attn = self._build_vha(config)
        hidden = paddle.randn([2, 16, 256])
        rotary_pos_emb = paddle.randn([1, 16, 1, 32])
        mask = paddle.zeros([2, 1, 16, 16])
        output, bias = attn(
            hidden, attention_mask=mask, rotary_pos_emb=rotary_pos_emb
        )
        self.assertEqual(list(output.shape), [2, 16, 256])


if __name__ == "__main__":
    unittest.main()
