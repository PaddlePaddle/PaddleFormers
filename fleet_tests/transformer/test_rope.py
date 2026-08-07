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
from types import SimpleNamespace

import numpy as np
import paddle
from paddle.incubate.nn.functional import (
    fused_rotary_position_embedding as fused_rope,
)

from paddleformers.fleet.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)


class TestRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        self.head_dim = 8
        self.rotary_percent = 1.0
        self.rope = RotaryEmbedding(self.head_dim, self.rotary_percent)

    def test_forward(self):
        output = self.rope(64)
        assert output.shape[0] == 1
        assert output.shape[1] == 64
        assert output.shape[2] == 1
        assert output.shape[3] == self.head_dim
        assert output.dtype == paddle.float32
        assert output.place.is_gpu_place()


class TestYarnRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        self.head_dim = 8
        self.rotary_percent = 1.0
        self.rope = YarnRotaryEmbedding(self.head_dim, self.rotary_percent)

    def test_forward(self):
        output, mscale = self.rope(64)
        assert output.shape[0] == 1
        assert output.shape[1] == 64
        assert output.shape[2] == 1
        assert output.shape[3] == self.head_dim
        assert output.dtype == paddle.float32
        assert output.place.is_gpu_place()
        assert mscale == 1.0


class TestYarnRotaryEmbeddingInterleaved(unittest.TestCase):
    def test_forward_returns_interleaved_freqs(self):
        rope = YarnRotaryEmbedding(8, 1.0, rotary_interleaved=True)
        output, mscale = rope(64)
        assert output.shape[0] == 1
        assert output.shape[1] == 64
        assert output.shape[2] == 1
        assert output.shape[3] == 8
        assert output.dtype == paddle.float32
        assert output.place.is_gpu_place()
        assert mscale == 1.0
        np.testing.assert_array_equal(
            output[0, :, 0, 0::2].numpy(),
            output[0, :, 0, 1::2].numpy(),
        )


def build_freqs_half(seq_len, rot_dim, rope_theta=10000.0):
    positions = paddle.arange(seq_len, dtype="float32")
    inv_freq = 1.0 / (
        rope_theta ** (paddle.arange(0, rot_dim, 2, dtype="float32") / rot_dim)
    )
    return paddle.outer(positions, inv_freq)


def build_freqs(freqs_half):
    return paddle.concat([freqs_half, freqs_half], axis=-1)


def build_cos_sin(freqs):
    cos = paddle.cos(freqs).reshape([1, freqs.shape[0], 1, freqs.shape[-1]])
    sin = paddle.sin(freqs).reshape([1, freqs.shape[0], 1, freqs.shape[-1]])
    return cos, sin


def make_config(
    apply_rope_fusion, high_precision_rope, rotary_interleaved=False
):
    return SimpleNamespace(
        apply_rope_fusion=apply_rope_fusion,
        rotary_interleaved=rotary_interleaved,
        multi_latent_attention=False,
        high_precision_rope=high_precision_rope,
        rope_theta=10000.0,
        sequence_parallel=False,
    )


def apply_fused_rope_reference(x, freqs):
    freqs_2d = (
        freqs.reshape([-1, freqs.shape[-1]]) if freqs.ndim == 3 else freqs
    )
    rot_dim = freqs_2d.shape[-1]
    cos, sin = build_cos_sin(freqs_2d)
    rotated, _, _ = fused_rope(
        x[..., :rot_dim].astype("float32"),
        sin=sin,
        cos=cos,
        use_neox_rotary_style=False,
    )
    rotated = rotated.astype(x.dtype)
    return paddle.cat((rotated, x[..., rot_dim:]), axis=-1)


class TestApplyRotaryPosEmbFusedHighPrecision(unittest.TestCase):
    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")
        paddle.set_device("gpu")
        paddle.seed(2026)

    def assertTensorExactEqual(self, actual, expected):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_high_precision_fused_matches_fp32_cast_for_2d_freqs(self):
        batch_size, seq_len, num_heads, hidden_dim = 2, 128, 8, 88
        rot_dim = 72
        x = paddle.randn(
            [batch_size, seq_len, num_heads, hidden_dim], dtype="float32"
        ).astype("bfloat16")
        freqs = build_freqs(build_freqs_half(seq_len, rot_dim))

        fused_config = make_config(
            apply_rope_fusion=True, high_precision_rope=True
        )
        fused_out = apply_rotary_pos_emb(
            x,
            freqs,
            cos=None,
            sin=None,
            config=fused_config,
            mscale=1.0,
        )
        ref_out = apply_fused_rope_reference(x, freqs)

        self.assertTensorExactEqual(fused_out, ref_out)

    def test_high_precision_fused_matches_fp32_cast_for_3d_freqs_and_none_mscale(
        self,
    ):
        batch_size, seq_len, num_heads, hidden_dim = 1, 256, 16, 72
        x = paddle.randn(
            [batch_size, seq_len, num_heads, hidden_dim], dtype="float32"
        ).astype("bfloat16")
        freqs = build_freqs(build_freqs_half(seq_len, hidden_dim)).unsqueeze(0)

        fused_config = make_config(
            apply_rope_fusion=True, high_precision_rope=True
        )
        fused_out = apply_rotary_pos_emb(
            x,
            freqs,
            cos=None,
            sin=None,
            config=fused_config,
            mscale=None,
        )
        ref_out = apply_fused_rope_reference(x, freqs)

        self.assertTensorExactEqual(fused_out, ref_out)

    def test_high_precision_fused_falls_back_to_unfused_path_when_mscale_changes(
        self,
    ):
        batch_size, seq_len, num_heads, hidden_dim = 1, 96, 4, 72
        x = paddle.randn(
            [batch_size, seq_len, num_heads, hidden_dim], dtype="float32"
        ).astype("bfloat16")
        freqs = build_freqs(build_freqs_half(seq_len, hidden_dim))

        fused_config = make_config(
            apply_rope_fusion=True, high_precision_rope=True
        )
        ref_config = make_config(
            apply_rope_fusion=False, high_precision_rope=True
        )

        fused_out = apply_rotary_pos_emb(
            x,
            freqs,
            cos=None,
            sin=None,
            config=fused_config,
            mscale=0.5,
        )
        ref_out = apply_rotary_pos_emb(
            x,
            freqs,
            cos=None,
            sin=None,
            config=ref_config,
            mscale=0.5,
        )

        self.assertTensorExactEqual(fused_out, ref_out)

    def test_high_precision_fused_falls_back_to_unfused_path_when_interleaved(
        self,
    ):
        batch_size, seq_len, num_heads, hidden_dim = 1, 96, 4, 72
        x = paddle.randn(
            [batch_size, seq_len, num_heads, hidden_dim], dtype="float32"
        ).astype("bfloat16")
        freqs = build_freqs(build_freqs_half(seq_len, hidden_dim))

        fused_config = make_config(
            apply_rope_fusion=True,
            high_precision_rope=True,
            rotary_interleaved=True,
        )
        ref_config = make_config(
            apply_rope_fusion=False,
            high_precision_rope=True,
            rotary_interleaved=True,
        )

        fused_out = apply_rotary_pos_emb(
            x,
            freqs,
            cos=None,
            sin=None,
            config=fused_config,
            mscale=1.0,
        )
        ref_out = apply_rotary_pos_emb(
            x,
            freqs,
            cos=None,
            sin=None,
            config=ref_config,
            mscale=1.0,
        )

        self.assertTensorExactEqual(fused_out, ref_out)

    def test_non_high_precision_fused_delegates_to_paddle_fused_rope(self):
        batch_size, seq_len, num_heads, hidden_dim = 2, 64, 8, 72
        q = paddle.randn(
            [batch_size, seq_len, num_heads, hidden_dim], dtype="float32"
        )
        k = paddle.randn(
            [batch_size, seq_len, num_heads, hidden_dim], dtype="float32"
        )
        freqs = build_freqs(build_freqs_half(seq_len, hidden_dim))
        cos, sin = build_cos_sin(freqs)

        config = make_config(apply_rope_fusion=True, high_precision_rope=False)

        fused_q, fused_k, _ = apply_rotary_pos_emb(
            (q, k),
            freqs=None,
            cos=cos,
            sin=sin,
            config=config,
        )
        ref_q, ref_k, _ = fused_rope(
            q,
            k,
            sin=sin,
            cos=cos,
            use_neox_rotary_style=False,
        )

        self.assertTensorExactEqual(fused_q, ref_q)
        self.assertTensorExactEqual(fused_k, ref_k)

    def test_non_high_precision_fused_requires_tuple_input(self):
        x = paddle.randn([1, 32, 4, 72], dtype="float32")
        freqs = build_freqs(build_freqs_half(32, 72))
        cos, sin = build_cos_sin(freqs)
        config = make_config(apply_rope_fusion=True, high_precision_rope=False)

        with self.assertRaisesRegex(
            AssertionError,
            "The input for fused_rope should be a tuple of tensors",
        ):
            apply_rotary_pos_emb(
                x,
                freqs=None,
                cos=cos,
                sin=sin,
                config=config,
            )


if __name__ == "__main__":
    unittest.main()
