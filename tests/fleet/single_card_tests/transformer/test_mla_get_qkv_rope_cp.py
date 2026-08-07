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

import sys
import types
import unittest
from unittest import mock

import paddle

from paddleformers.fleet.transformer import multi_latent_attention as mla_mod
from paddleformers.fleet.transformer.multi_latent_attention import MLASelfAttention


class _IdentityNorm:
    def __call__(self, x):
        return x


class _TileProjection:
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def __call__(self, x):
        repeat = (self.out_dim + x.shape[-1] - 1) // x.shape[-1]
        reps = [1] * x.ndim
        reps[-1] = repeat
        return paddle.tile(x, reps)[..., : self.out_dim], None


def _expected_rotary_seq_len(hidden_states, config):
    if config.sequence_parallel:
        return (
            hidden_states.shape[0]
            * config.tensor_model_parallel_size
            * config.context_parallel_size
        )
    return hidden_states.shape[1] * config.context_parallel_size


class _NoneRotaryEmbedding:
    def get_rotary_seq_len(self, hidden_states, config, packed_seq_params=None):
        return _expected_rotary_seq_len(hidden_states, config)

    def __call__(self, *args, **kwargs):
        return None


class _FakeRotaryEmbedding:
    def __init__(self, dim):
        self.dim = dim

    def get_rotary_seq_len(self, hidden_states, config, packed_seq_params=None):
        return _expected_rotary_seq_len(hidden_states, config)

    def __call__(
        self,
        max_seq_len,
        offset=0,
        packed_seq=False,
        position_ids=None,
    ):
        return self._emb(max_seq_len, offset)

    def get_cached_cos_sin(
        self,
        seq_len,
        offset=0,
        dtype=paddle.get_default_dtype(),
        packed_seq=False,
    ):
        emb = self._emb(seq_len, offset).astype(dtype)
        return emb + 1.0, emb * 0.5 + 0.25

    def _emb(self, seq_len, offset=0):
        pos = paddle.arange(offset, offset + seq_len, dtype="float32").reshape(
            [1, seq_len, 1, 1]
        )
        dim = paddle.arange(self.dim, dtype="float32").reshape(
            [1, 1, 1, self.dim]
        )
        return pos * 0.01 + dim * 0.001


def _scatter_for_rank(tensor, axis, mode, rank, size):
    if size == 1:
        return tensor.clone()

    seq_len = tensor.shape[axis]
    if mode.startswith("contiguous"):
        interval = seq_len // size
        return paddle.slice(
            tensor,
            axes=[axis],
            starts=[interval * rank],
            ends=[interval * (rank + 1)],
        )

    assert seq_len % (size * 2) == 0
    interval = seq_len // size // 2
    chunk_start = paddle.slice(
        tensor,
        axes=[axis],
        starts=[interval * rank],
        ends=[interval * (rank + 1)],
    )
    chunk_end = paddle.slice(
        tensor,
        axes=[axis],
        starts=[seq_len - interval * (rank + 1)],
        ends=[seq_len - interval * rank],
    )
    return paddle.concat([chunk_start, chunk_end], axis=axis)


def _apply_rope_ref(emb, cos, sin):
    emb_dim = emb.shape[-1]
    half = emb_dim // 2
    cos = cos.reshape([-1, emb_dim])
    sin = sin.reshape([-1, emb_dim])
    if emb.ndim == 4:
        cos = cos.reshape([1, emb.shape[1], 1, emb_dim])
        sin = sin.reshape([1, emb.shape[1], 1, emb_dim])
    else:
        cos = cos.reshape([emb.shape[0], 1, emb_dim])
        sin = sin.reshape([emb.shape[0], 1, emb_dim])

    x1 = emb[..., 0::2]
    x2 = emb[..., 1::2]
    out_left = x1 * cos[..., :half] - x2 * sin[..., :half]
    out_right = x2 * cos[..., half:] + x1 * sin[..., half:]
    return paddle.concat([out_left, out_right], axis=-1)


def _fused_apply_mla_rope_for_q_ref(
    q,
    cos,
    sin,
    qk_head_dim,
    emb_dim,
    cu_seqlens_q=None,
    cp_rank=0,
    cp_size=1,
    rotary_interleaved=False,
):
    q_nope = q[..., :qk_head_dim]
    q_pe = q[..., qk_head_dim:]
    return paddle.concat([q_nope, _apply_rope_ref(q_pe, cos, sin)], axis=-1)


def _fused_apply_mla_rope_for_kv_ref(
    kv,
    k_pos_emb,
    cos,
    sin,
    emb_dim,
    k_dim,
    v_dim,
    cu_seqlens_kv=None,
    cp_rank=0,
    cp_size=1,
    rotary_interleaved=False,
):
    k_nope = kv[..., :k_dim]
    value = kv[..., k_dim : k_dim + v_dim]
    k_pe = _apply_rope_ref(k_pos_emb, cos, sin)
    k_pe = k_pe.expand([*k_pe.shape[:-2], kv.shape[-2], emb_dim])
    return paddle.concat([k_nope, k_pe], axis=-1), value


def _apply_rotary_pos_emb_identity(
    t,
    freqs,
    cos,
    sin,
    config,
    cu_seqlens=None,
    total_seq_len=None,
    mscale=1.0,
    cp_group=None,
    sp_group=None,
    position_ids=None,
    inverse=False,
    mla_output_remove_interleaving=False,
):
    return t


class TestMLAGetQKVRopeContextParallel(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)
        try:
            paddle.device.set_device("cpu")
        except Exception:
            pass

    def _make_layer(
        self,
        rope_type,
        apply_rope_fusion,
        context_parallel_size,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
    ):
        heads = 2
        qk_nope = 4
        qk_rope = 8
        v_dim = 4
        kv_lora = 8
        hidden = 16

        layer = types.SimpleNamespace()
        layer.get_query_key_value_tensors = types.MethodType(
            MLASelfAttention.get_query_key_value_tensors, layer
        )
        layer._is_cudagraph_active = types.MethodType(
            MLASelfAttention._is_cudagraph_active, layer
        )
        layer.config = types.SimpleNamespace(
            q_lora_rank=None,
            hidden_size=hidden,
            kv_lora_rank=kv_lora,
            sequence_parallel=sequence_parallel,
            tensor_model_parallel_size=tensor_model_parallel_size,
            context_parallel_size=context_parallel_size,
            cp_balance_mode="dualchunk_allgather",
            rope_type=rope_type,
            apply_rope_fusion=apply_rope_fusion,
            gpt_model_use_experimental_version=False,
        )
        layer.num_attention_heads_per_partition = heads
        layer.qk_nope_head_dim = qk_nope
        layer.qk_rope_head_dim = qk_rope
        layer.v_head_dim = v_dim
        layer.q_head_dim = qk_nope + qk_rope
        layer.q_proj = _TileProjection(heads * layer.q_head_dim)
        layer.kv_a_proj_with_mqa = _TileProjection(kv_lora + qk_rope)
        layer.kv_b_proj = _TileProjection(heads * (qk_nope + v_dim))
        layer.kv_a_layernorm = _IdentityNorm()
        layer.rotary_pos_emb = _FakeRotaryEmbedding(qk_rope)
        layer.pg_collection = types.SimpleNamespace(tp=None, cp=object())
        layer.core_attention = types.SimpleNamespace(
            config=types.SimpleNamespace()
        )
        layer.layer_number = 1
        layer.training = True
        layer.recompute_qkv_up_porj_and_rope = False
        return layer

    def _hidden(self, batch=2, seq=32, hidden=16):
        values = paddle.arange(batch * seq * hidden, dtype="float32")
        return values.reshape([batch, seq, hidden]) / 100.0

    def _fake_fused_module(self):
        module = types.ModuleType(
            "paddleformers.fleet.triton_ops.fused_mla_yarn_rope_apply"
        )
        module.fused_apply_mla_rope_for_q = _fused_apply_mla_rope_for_q_ref
        module.fused_apply_mla_rope_for_kv = _fused_apply_mla_rope_for_kv_ref
        return module

    def _fake_transformer_layer_module(self):
        module = types.ModuleType("paddleformers.fleet.transformer.transformer_layer")

        class TransformerLayer:
            @staticmethod
            def _log_md5(tensor, name, layer_idx):
                pass

        module.TransformerLayer = TransformerLayer
        return module

    def _run_layer(
        self,
        layer,
        hidden,
        cp_size,
        cp_rank,
        scatter_calls=None,
        packed_seq_params=None,
    ):
        def scatter_spy(tensor, axis=0, mode="dualchunk_allgather"):
            if scatter_calls is not None:
                scatter_calls.append((list(tensor.shape), axis, mode))
            return _scatter_for_rank(tensor, axis, mode, cp_rank, cp_size)

        patches = [
            mock.patch.object(
                mla_mod, "get_context_parallel_world_size", return_value=cp_size
            ),
            mock.patch.object(mla_mod, "get_pg_size", return_value=cp_size),
            mock.patch.object(mla_mod, "get_pg_rank", return_value=cp_rank),
            mock.patch.object(
                mla_mod.ContextParallelScatterOp,
                "apply",
                side_effect=scatter_spy,
            ),
            mock.patch.dict(
                sys.modules,
                {
                    "paddleformers.fleet.triton_ops.fused_mla_yarn_rope_apply": self._fake_fused_module(),
                    "paddleformers.fleet.transformer.transformer_layer": self._fake_transformer_layer_module(),
                },
            ),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            return layer.get_query_key_value_tensors(
                hidden, packed_seq_params=packed_seq_params
            )

    def _assert_equal(self, actual, expected, name):
        self.assertTrue(
            bool(paddle.equal_all(actual, expected).item()),
            f"{name} is not bitwise equal",
        )

    def _run_cp1_cp4_bitwise(self, rope_type):
        hidden_full = self._hidden()
        ref_layer = self._make_layer(
            rope_type=rope_type,
            apply_rope_fusion=True,
            context_parallel_size=1,
        )
        q_ref, k_ref, v_ref, *_ = self._run_layer(
            ref_layer, hidden_full, cp_size=1, cp_rank=0
        )

        for rank in range(4):
            cp_layer = self._make_layer(
                rope_type=rope_type,
                apply_rope_fusion=True,
                context_parallel_size=4,
            )
            hidden_local = _scatter_for_rank(
                hidden_full,
                axis=1,
                mode="dualchunk_allgather",
                rank=rank,
                size=4,
            )
            scatter_calls = []
            q_cp, k_cp, v_cp, *_ = self._run_layer(
                cp_layer,
                hidden_local,
                cp_size=4,
                cp_rank=rank,
                scatter_calls=scatter_calls,
            )

            self._assert_equal(
                q_cp,
                _scatter_for_rank(
                    q_ref, axis=1, mode="dualchunk_allgather", rank=rank, size=4
                ),
                f"query rank {rank} rope_type={rope_type}",
            )
            self._assert_equal(
                k_cp,
                _scatter_for_rank(
                    k_ref, axis=1, mode="dualchunk_allgather", rank=rank, size=4
                ),
                f"key rank {rank} rope_type={rope_type}",
            )
            self._assert_equal(
                v_cp,
                _scatter_for_rank(
                    v_ref, axis=1, mode="dualchunk_allgather", rank=rank, size=4
                ),
                f"value rank {rank} rope_type={rope_type}",
            )

            rope_table_calls = [
                call for call in scatter_calls if call[0][0] == 1
            ]
            if rope_type == "yarn":
                self.assertEqual(len(rope_table_calls), 2)
            else:
                self.assertEqual(len(rope_table_calls), 1)
            for shape, axis, mode in rope_table_calls:
                self.assertEqual(shape, [1, hidden_full.shape[1], 1, 8])
                self.assertEqual(axis, 1)
                self.assertEqual(mode, "dualchunk_allgather")

    def test_fused_yarn_cp4_matches_cp1_bitwise(self):
        self._run_cp1_cp4_bitwise("yarn")

    def test_fused_rope_cp4_matches_cp1_bitwise(self):
        self._run_cp1_cp4_bitwise("rope")

    def test_non_fused_cp_uses_prescattered_rotary_pos_emb(self):
        hidden_full = self._hidden()
        ref_layer = self._make_layer(
            rope_type="rope",
            apply_rope_fusion=False,
            context_parallel_size=1,
        )
        with mock.patch.object(
            mla_mod,
            "apply_rotary_pos_emb",
            side_effect=_apply_rotary_pos_emb_identity,
        ):
            q_ref, k_ref, v_ref, *_ = self._run_layer(
                ref_layer, hidden_full, cp_size=1, cp_rank=0
            )

            for rank in range(4):
                cp_layer = self._make_layer(
                    rope_type="rope",
                    apply_rope_fusion=False,
                    context_parallel_size=4,
                )
                hidden_local = _scatter_for_rank(
                    hidden_full,
                    axis=1,
                    mode="dualchunk_allgather",
                    rank=rank,
                    size=4,
                )
                scatter_calls = []
                q_cp, k_cp, v_cp, *_ = self._run_layer(
                    cp_layer,
                    hidden_local,
                    cp_size=4,
                    cp_rank=rank,
                    scatter_calls=scatter_calls,
                )

                self._assert_equal(
                    q_cp,
                    _scatter_for_rank(
                        q_ref,
                        axis=1,
                        mode="dualchunk_allgather",
                        rank=rank,
                        size=4,
                    ),
                    f"non-fused query rank {rank}",
                )
                self._assert_equal(
                    k_cp,
                    _scatter_for_rank(
                        k_ref,
                        axis=1,
                        mode="dualchunk_allgather",
                        rank=rank,
                        size=4,
                    ),
                    f"non-fused key rank {rank}",
                )
                self._assert_equal(
                    v_cp,
                    _scatter_for_rank(
                        v_ref,
                        axis=1,
                        mode="dualchunk_allgather",
                        rank=rank,
                        size=4,
                    ),
                    f"non-fused value rank {rank}",
                )
                self.assertEqual(
                    scatter_calls[0],
                    ([1, hidden_full.shape[1], 1, 8], 1, "dualchunk_allgather"),
                )

    def test_context_parallel_requires_prepared_rope_tensor(self):
        layer = self._make_layer(
            rope_type="rope",
            apply_rope_fusion=True,
            context_parallel_size=4,
        )
        layer.rotary_pos_emb = _NoneRotaryEmbedding()
        with self.assertRaisesRegex(
            ValueError, "Context parallel requires rotary_pos_emb"
        ):
            self._run_layer(layer, self._hidden(seq=8), cp_size=4, cp_rank=0)

    def test_context_parallel_rejects_packed_seq_params(self):
        layer = self._make_layer(
            rope_type="rope",
            apply_rope_fusion=True,
            context_parallel_size=4,
        )
        packed_seq_params = types.SimpleNamespace(qkv_format="thd")
        with self.assertRaisesRegex(
            ValueError, "does not support packed_seq_params"
        ):
            self._run_layer(
                layer,
                self._hidden(seq=8),
                cp_size=4,
                cp_rank=0,
                packed_seq_params=packed_seq_params,
            )

    def test_sequence_parallel_rotary_seq_len_validation(self):
        layer = self._make_layer(
            rope_type="rope",
            apply_rope_fusion=False,
            context_parallel_size=1,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
        )
        hidden = self._hidden(batch=8, seq=2)
        with mock.patch.object(
            mla_mod,
            "apply_rotary_pos_emb",
            side_effect=_apply_rotary_pos_emb_identity,
        ):
            q_ref, *_ = self._run_layer(layer, hidden, cp_size=1, cp_rank=0)
        self.assertEqual(q_ref.shape[0], hidden.shape[0])

    def test_context_parallel_rejects_wrong_rotary_seq_len(self):
        class _BadSeqLenRotaryEmbedding(_FakeRotaryEmbedding):
            def get_rotary_seq_len(
                self, hidden_states, config, packed_seq_params=None
            ):
                return _expected_rotary_seq_len(hidden_states, config) + 8

        layer = self._make_layer(
            rope_type="rope",
            apply_rope_fusion=True,
            context_parallel_size=4,
        )
        layer.rotary_pos_emb = _BadSeqLenRotaryEmbedding(layer.qk_rope_head_dim)
        with self.assertRaisesRegex(
            ValueError, "rotary_seq_len to be the global sequence length"
        ):
            self._run_layer(layer, self._hidden(seq=8), cp_size=4, cp_rank=0)

    def test_context_parallel_sequence_parallel_rejects_wrong_rotary_seq_len(
        self,
    ):
        class _BadSeqLenRotaryEmbedding(_FakeRotaryEmbedding):
            def get_rotary_seq_len(
                self, hidden_states, config, packed_seq_params=None
            ):
                return _expected_rotary_seq_len(hidden_states, config) + 8

        layer = self._make_layer(
            rope_type="rope",
            apply_rope_fusion=False,
            context_parallel_size=4,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
        )
        layer.rotary_pos_emb = _BadSeqLenRotaryEmbedding(layer.qk_rope_head_dim)
        with self.assertRaisesRegex(
            ValueError, "rotary_seq_len to be the global sequence length"
        ):
            self._run_layer(
                layer, self._hidden(batch=8, seq=2), cp_size=4, cp_rank=0
            )

    def test_context_parallel_rejects_wrong_cached_cos_sin_length(self):
        class _LongCachedRotaryEmbedding(_FakeRotaryEmbedding):
            def get_cached_cos_sin(
                self,
                seq_len,
                offset=0,
                dtype=paddle.get_default_dtype(),
                packed_seq=False,
            ):
                emb = self._emb(seq_len + 8, offset).astype(dtype)
                return emb + 1.0, emb * 0.5 + 0.25

        layer = self._make_layer(
            rope_type="yarn",
            apply_rope_fusion=True,
            context_parallel_size=4,
        )
        layer.rotary_pos_emb = _LongCachedRotaryEmbedding(
            layer.qk_rope_head_dim
        )
        with self.assertRaisesRegex(
            ValueError, "rotary_pos_cos/sin sequence length"
        ):
            self._run_layer(layer, self._hidden(seq=8), cp_size=4, cp_rank=0)

    def test_context_parallel_rejects_wrong_rotary_pos_emb_length(self):
        class _LongRotaryEmbedding(_FakeRotaryEmbedding):
            def __call__(
                self,
                max_seq_len,
                offset=0,
                packed_seq=False,
                position_ids=None,
            ):
                return self._emb(max_seq_len + 8, offset)

        layer = self._make_layer(
            rope_type="rope",
            apply_rope_fusion=True,
            context_parallel_size=4,
        )
        layer.rotary_pos_emb = _LongRotaryEmbedding(layer.qk_rope_head_dim)
        with self.assertRaisesRegex(
            ValueError, "rotary_pos_emb sequence length"
        ):
            self._run_layer(layer, self._hidden(seq=8), cp_size=4, cp_rank=0)

    def test_fused_rope_rejects_local_cos_sin_length_mismatch(self):
        layer = self._make_layer(
            rope_type="yarn",
            apply_rope_fusion=True,
            context_parallel_size=1,
        )
        hidden = self._hidden(seq=8)
        with (
            mock.patch.object(
                layer.rotary_pos_emb,
                "get_cached_cos_sin",
                return_value=(
                    layer.rotary_pos_emb._emb(9) + 1.0,
                    layer.rotary_pos_emb._emb(9) * 0.5 + 0.25,
                ),
            ),
            self.assertRaisesRegex(ValueError, "local cos/sin sequence length"),
        ):
            self._run_layer(layer, hidden, cp_size=1, cp_rank=0)

    def test_unfused_rope_rejects_local_rotary_pos_emb_mismatch(self):
        class _ShortRotaryEmbedding(_FakeRotaryEmbedding):
            def __call__(
                self,
                max_seq_len,
                offset=0,
                packed_seq=False,
                position_ids=None,
            ):
                return self._emb(max_seq_len - 1, offset)

        layer = self._make_layer(
            rope_type="rope",
            apply_rope_fusion=False,
            context_parallel_size=1,
        )
        layer.rotary_pos_emb = _ShortRotaryEmbedding(layer.qk_rope_head_dim)
        with self.assertRaisesRegex(
            ValueError, "local rotary_pos_emb sequence"
        ):
            self._run_layer(layer, self._hidden(seq=8), cp_size=1, cp_rank=0)

    def test_unfused_rope_rejects_packed_seq_params(self):
        layer = self._make_layer(
            rope_type="rope",
            apply_rope_fusion=False,
            context_parallel_size=1,
        )
        cu_seqlens = paddle.to_tensor([0, 8], dtype="int32")
        packed_seq_params = types.SimpleNamespace(
            qkv_format="thd",
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
        )
        with self.assertRaisesRegex(
            ValueError, "qkv_up_proj_and_rope_apply does not support"
        ):
            self._run_layer(
                layer,
                self._hidden(seq=8),
                cp_size=1,
                cp_rank=0,
                packed_seq_params=packed_seq_params,
            )


if __name__ == "__main__":
    unittest.main()
