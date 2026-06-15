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
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

from paddleformers.fleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.utils import init_method_normal, scaled_init_method_normal

# ---- Helpers & stubs ----


def _mock_rng_tracker():
    @contextmanager
    def _noop(*a, **kw):
        yield

    t = MagicMock()
    t.fork = _noop
    return t


_RNG = patch(
    "paddleformers.fleet.tensor_parallel.get_cuda_rng_tracker",
    side_effect=lambda *a, **kw: _mock_rng_tracker(),
)
_SCATTER = patch(
    "paddleformers.fleet.models.gpt.gpt_embedding.scatter_to_sequence_parallel_region",
    side_effect=lambda x, group=None: x,
)


def _emb_config(**kw):
    c = TransformerConfig(num_hidden_layers=1, hidden_size=64, num_attention_heads=2)
    defaults = {
        "sequence_parallel": False,
        "multimodal_embedding": False,
        "num_nextn_predict_layers": 0,
        "mtp_load_weight_only": False,
        "clone_scatter_output_in_embedding": True,
        "image_token_id": 1000,
        "video_token_id": 1001,
    }
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(c, k, v)
    return c


def _make_emb(config, rope=None):
    from paddle.distributed.fleet.meta_parallel import LayerSpec

    from paddleformers.fleet.models.common.embeddings import LanguageModelEmbedding
    from paddleformers.fleet.models.gpt.gpt_embedding import (
        GPTEmbedding,
        GPTEmbeddingSpec,
    )

    rope_spec, pos_type, msec = None, "none", None
    if rope == "rope":
        from paddleformers.fleet.models.common.embeddings import RotaryEmbedding

        rope_spec, pos_type = LayerSpec(RotaryEmbedding), "rope"
    elif rope == "mrope":
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            MultimodalRotaryEmbedding,
        )

        rope_spec, pos_type, msec = (
            LayerSpec(MultimodalRotaryEmbedding),
            "mrope",
            [16, 24, 24],
        )
    return GPTEmbedding(
        GPTEmbeddingSpec(
            language_embedding=LayerSpec(LanguageModelEmbedding),
            rope_embedding=rope_spec,
        ),
        config,
        vocab_size=1024,
        max_sequence_length=128,
        position_embedding_type=pos_type,
        mrope_section=msec,
    )


class BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_f, out_f, **kw):
        super().__init__()
        self.linear = paddle.nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.linear(x), self.linear.bias


class RMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps, **kw):
        super().__init__()
        self.weight = paddle.nn.Parameter(paddle.ones([hidden_size]))
        self.eps = eps

    def forward(self, x):
        return x * paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps) * self.weight


def _make_attn(sp=False):
    from paddleformers.fleet.transformer.attention import (
        SelfAttention,
        SelfAttentionSublayersSpec,
    )
    from paddleformers.fleet.transformer.dot_product_attention import (
        DotProductAttention,
    )
    from paddleformers.fleet.transformer.enums import AttnMaskType

    c = TransformerConfig(num_hidden_layers=1, hidden_size=128, num_attention_heads=4)
    for k, v in {
        "num_key_value_heads": 4,
        "head_dim": 32,
        "softmax_scale": None,
        "use_bias": True,
        "no_rope_freq": None,
        "recompute_granularity": None,
        "fused_single_qkv_rope": False,
        "rotary_interleaved": False,
        "multi_latent_attention": False,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "rms_norm_eps": 1e-5,
        "context_parallel_size": 1,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "window_attn_skip_freq": None,
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "attention_dropout": 0.0,
        "softmax_type": "vanilla",
        "high_precision_rope": False,
        "sequence_parallel": sp,
    }.items():
        setattr(c, k, v)
    return SelfAttention(
        c,
        SelfAttentionSublayersSpec(
            qkv_proj=BiasedLinear,
            core_attention=DotProductAttention,
            o_proj=BiasedLinear,
            q_norm=RMSNorm,
            k_norm=RMSNorm,
        ),
        attn_mask_type=AttnMaskType.causal,
        layer_number=1,
    )


# ---- 1. rope_utils: M-RoPE 3D freqs reshape (diff lines 164-171) ----


class TestMRoPEFreqsReshape(unittest.TestCase):
    def test_reshape_and_numerical_correctness(self):
        """freqs [S,B,D] auto-transposed to [B,S,D]; result matches manual transpose."""
        B, S, H, D = 2, 4, 2, 8
        t = paddle.randn([B, S, H, D])
        freqs_sb = paddle.randn([S, B, D])
        result = _apply_rotary_pos_emb_bshd(t, freqs_sb)
        assert result.shape == [B, S, H, D]
        # Manual transpose [S,B,D] -> [B,S,D] should match auto-transpose
        freqs_bs = freqs_sb.transpose([1, 0, 2]).contiguous()
        np.testing.assert_allclose(
            result.numpy(),
            _apply_rotary_pos_emb_bshd(t, freqs_bs).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_no_reshape_for_matching_3d_or_4d(self):
        """No reshape when dims already match (3D) or freqs is 4D."""
        B, S, H, D = 2, 8, 4, 16
        t = paddle.randn([B, S, H, D])
        assert _apply_rotary_pos_emb_bshd(t, paddle.randn([B, S, D])).shape == [
            B,
            S,
            H,
            D,
        ]
        assert _apply_rotary_pos_emb_bshd(t, paddle.randn([1, S, 1, D])).shape == [B, S, H, D]

    def test_partial_rot_dim(self):
        """Reshape works when rot_dim < head_dim."""
        B, S, H = 2, 4, 2
        assert _apply_rotary_pos_emb_bshd(paddle.randn([B, S, H, 16]), paddle.randn([S, B, 8])).shape == [B, S, H, 16]


# ---- 2. gpt_embedding: multimodal SP config (diff lines 68-81) ----


class TestGPTEmbeddingMultimodalSPConfig(unittest.TestCase):
    def test_sp_disables_internal_scatter_for_multimodal(self):
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        from paddleformers.fleet.models.common.embeddings import LanguageModelEmbedding
        from paddleformers.fleet.models.gpt.gpt_embedding import (
            GPTEmbedding,
            GPTEmbeddingSpec,
        )

        config = _emb_config(sequence_parallel=True, multimodal_embedding=True)
        gpt_emb = GPTEmbedding(
            GPTEmbeddingSpec(
                language_embedding=LayerSpec(LanguageModelEmbedding),
                rope_embedding=None,
            ),
            config,
            vocab_size=1024,
            max_sequence_length=128,
        )
        assert gpt_emb.embedding.scatter_to_sequence_parallel is False
        assert gpt_emb.embedding.reduce_scatter_embeddings is False


# ---- 3. gpt_embedding.forward(): RoPE/MRoPE + SP transpose (diff lines 291-344) ----


class TestGPTEmbeddingForward(unittest.TestCase):
    def test_basic_and_decoder_input(self):
        gpt_emb = _make_emb(_emb_config())
        B, S, H = 2, 16, 64
        out = gpt_emb.forward({"input_ids": paddle.randint(0, 500, [B, S])})
        assert out["hidden_states"].shape == [B, S, H]

        dec = paddle.randn([B, S, H])
        out2 = gpt_emb.forward({"input_ids": None, "decoder_input": dec})
        np.testing.assert_allclose(out2["hidden_states"].numpy(), dec.numpy())

    @_RNG
    def test_rope_sp_transpose(self, _):
        """RoPE + SP: output rotary_pos_emb transposed to [S,1,1,D]."""
        gpt_emb = _make_emb(_emb_config(sequence_parallel=True), rope="rope")
        S = 8
        rope = gpt_emb.forward({"input_ids": paddle.randint(0, 500, [1, S])})["rotary_pos_emb"]
        assert rope.ndim == 4 and rope.shape[0] == S and rope.shape[1] == 1

    @_RNG
    def test_mrope_sp_transpose(self, _):
        """MRoPE + SP: output rotary_pos_emb transposed to [S,B,D]."""
        gpt_emb = _make_emb(_emb_config(sequence_parallel=True), rope="mrope")
        B, S = 2, 8
        pos_ids = paddle.arange(S).unsqueeze(0).expand([3, B, S])
        rope = gpt_emb.forward(
            {
                "input_ids": paddle.randint(0, 500, [B, S]),
                "position_ids": pos_ids,
            }
        )["rotary_pos_emb"]
        assert rope.ndim == 3 and rope.shape[0] == S and rope.shape[1] == B


# ---- 4. gpt_embedding.forward(): multimodal + SP scatter ----


class TestGPTEmbeddingForwardMultimodal(unittest.TestCase):
    def _emb(self, sp=False):
        return _make_emb(_emb_config(multimodal_embedding=True, sequence_parallel=sp))

    def test_image_and_video_embed_replace(self):
        gpt_emb = self._emb()
        B, S, H = 1, 16, 64
        ids = paddle.randint(0, 500, [B, S])
        ids[0, 3:6] = 1000  # image
        ids[0, 8:11] = 1001  # video
        out = gpt_emb.forward({"input_ids": ids, "image_embeds": paddle.randn([3, H])})
        assert out["hidden_states"].shape == [B, S, H] and "visual_pos_masks" in out

        ids2 = paddle.randint(0, 500, [B, S])
        ids2[0, 5:8] = 1001
        out2 = gpt_emb.forward({"input_ids": ids2, "video_embeds": paddle.randn([3, H])})
        assert "visual_pos_masks" in out2

    def test_image_and_video_with_deepstack(self):
        gpt_emb = self._emb()
        B, S, H, n_visual = 1, 16, 64, 5
        ids = paddle.randint(0, 500, [B, S])
        ids[0, 2:4] = 1000
        ids[0, 6:9] = 1001
        out = gpt_emb.forward(
            {
                "input_ids": ids,
                "image_embeds": paddle.randn([2, H]),
                "video_embeds": paddle.randn([3, H]),
                "deepstack_image_embeds": [paddle.randn([n_visual, H])],
                "deepstack_video_embeds": [paddle.randn([n_visual, H])],
            }
        )
        assert "deepstack_visual_emb" in out

    @_SCATTER
    @_RNG
    def test_sp_scatter_and_clone(self, _, mock_scatter):
        """SP + multimodal: scatter called; clone path exercised."""
        config = _emb_config(
            multimodal_embedding=True,
            sequence_parallel=True,
            clone_scatter_output_in_embedding=True,
        )
        gpt_emb = _make_emb(config)
        B, S, H = 1, 8, 64
        ids = paddle.randint(0, 500, [B, S])
        ids[0, 2:5] = 1000
        out = gpt_emb.forward({"input_ids": ids, "image_embeds": paddle.randn([3, H])})
        mock_scatter.assert_called_once()
        assert "hidden_states" in out


# ---- 5. gpt_embedding.get_placeholder_mask() ----


class TestGetPlaceholderMask(unittest.TestCase):
    def setUp(self):
        self.gpt_emb = _make_emb(_emb_config(multimodal_embedding=True))

    def test_mask_positions(self):
        B, S, H = 1, 8, 64
        ids = paddle.randint(0, 500, [B, S])
        ids[0, 2:5] = 1000
        ids[0, 6:7] = 1001
        emb = paddle.randn([B, S, H])

        img, vid = self.gpt_emb.get_placeholder_mask(
            ids,
            emb,
            image_features=paddle.randn([3, H]),
            video_features=paddle.randn([1, H]),
        )
        assert int(img[0, 2, 0]) == 1 and int(img[0, 0, 0]) == 0
        assert int(vid[0, 6, 0]) == 1 and int(vid[0, 0, 0]) == 0

    def test_count_mismatch_raises(self):
        B, S, H = 1, 8, 64
        ids = paddle.randint(0, 500, [B, S])
        ids[0, 2:5] = 1000
        with self.assertRaises(ValueError):
            self.gpt_emb.get_placeholder_mask(
                ids,
                paddle.randn([B, S, H]),
                image_features=paddle.randn([5, H]),
            )


# ---- 6. attention.py: mask startend_row_indices slicing (diff lines 390-404) ----


class TestAttentionMaskSlicing(unittest.TestCase):
    def test_no_sp_passthrough(self):
        attn = _make_attn(sp=False)
        B, S, H = 2, 8, 128
        mask = paddle.arange(S, dtype="int32").reshape([1, 1, S, 1]).expand([B, 1, S, 1])
        out, _ = attn(
            paddle.randn([B, S, H]),
            attention_mask=None,
            attn_mask_startend_row_indices=mask,
        )
        assert out.shape == [B, S, H]

    def test_sp_full_mask_triggers_slicing(self):
        """SP + full-size mask triggers slicing code path."""
        attn = _make_attn(sp=True)
        B, S, H = 2, 8, 128
        full_S = S * 2
        mask = paddle.arange(full_S, dtype="int32").reshape([1, 1, full_S, 1]).expand([B, 1, full_S, 1])
        out, _ = attn(
            paddle.randn([B, S, H]),
            attention_mask=None,
            attn_mask_startend_row_indices=mask,
        )
        assert out.shape == [B, S, H]


if __name__ == "__main__":
    unittest.main()
