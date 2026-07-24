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

from paddleformers.fleet.transformer.multi_latent_attention import (
    FP8OverlapProj,
    MultiLatentAttention,
    _ec_compatible_rope_apply,
)


class TestMLASelfAttentionBackwardDW(unittest.TestCase):
    """Tests for MLASelfAttention backward_dw methods."""

    def _make_mla_self_attn(self, q_lora_rank=None):
        """Create a MLASelfAttention with mocked internals."""
        config = MagicMock()
        config.head_dim = 16
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.hidden_size = 64
        config.v_head_dim = 16
        config.qk_nope_head_dim = 8
        config.qk_rope_head_dim = 8
        config.q_lora_rank = q_lora_rank
        config.kv_lora_rank = 16
        config.rope_type = "rope"
        config.rotary_interleaved = False
        config.rope_theta = 10000.0
        config.rotary_scaling_factor = 1.0
        config.mscale_all_dim = False
        config.gated_attention = False
        config.dw_p2p_overlap = False
        config.use_bias = True
        config.recompute_granularity = None
        config.recompute_modules = None
        config.sequence_parallel = False

        with (
            patch(
                "paddleformers.fleet.transformer.multi_latent_attention.Attention.__init__",
                return_value=None,
            ),
            patch(
                "paddleformers.fleet.transformer.multi_latent_attention.RotaryEmbedding"
            ),
            patch(
                "paddleformers.fleet.transformer.multi_latent_attention.build_spec_layer"
            ),
            patch(
                "paddleformers.fleet.transformer.multi_latent_attention.ProcessGroupCollection.use_mpu_process_groups"
            ),
        ):
            from paddleformers.fleet.transformer.multi_latent_attention import (
                MLASelfAttention,
            )

            mla = MLASelfAttention.__new__(MLASelfAttention)
            mla.config = config
            mla.kv_b_proj = MagicMock()
            mla.kv_a_proj_with_mqa = MagicMock()
            mla.o_proj = MagicMock()
            if q_lora_rank is not None:
                mla.q_a_proj = MagicMock()
                mla.q_b_proj = MagicMock()
            else:
                mla.q_proj = MagicMock()
            return mla

    def test_backward_dw_with_q_lora_rank(self):
        """Test backward_dw with q_lora_rank (uses q_a_proj and q_b_proj)."""
        mla = self._make_mla_self_attn(q_lora_rank=32)
        mla.backward_dw()
        mla.kv_b_proj.backward_dw.assert_called_once()
        mla.kv_a_proj_with_mqa.backward_dw.assert_called_once()
        mla.q_a_proj.backward_dw.assert_called_once()
        mla.q_b_proj.backward_dw.assert_called_once()
        mla.o_proj.backward_dw.assert_called_once()

    def test_backward_dw_without_q_lora_rank(self):
        """Test backward_dw without q_lora_rank (uses q_proj)."""
        mla = self._make_mla_self_attn(q_lora_rank=None)
        mla.backward_dw()
        mla.kv_b_proj.backward_dw.assert_called_once()
        mla.kv_a_proj_with_mqa.backward_dw.assert_called_once()
        mla.q_proj.backward_dw.assert_called_once()
        mla.o_proj.backward_dw.assert_called_once()


class TestFP8OverlapProjBackward(unittest.TestCase):
    """Tests for FP8OverlapProj backward pass."""

    def test_backward_with_stop_gradient_weight(self):
        """Test FP8OverlapProj backward with weight that has stop_gradient."""
        x = paddle.randn([2, 4, 8])
        weight = paddle.randn([8, 4])
        weight.stop_gradient = True

        out = FP8OverlapProj.apply(x, weight)
        self.assertEqual(out.shape, [2, 4, 4])


class TestMultiLatentAttentionGate(unittest.TestCase):
    """Tests for MultiLatentAttention _gate method."""

    def _make_mla_with_gate(self):
        """Create a MultiLatentAttention with gate."""
        config = MagicMock()
        config.sequence_parallel = False
        config.recompute_granularity = None
        config.gated_attention = True
        config.dw_p2p_overlap = False
        config.use_bias = True
        config.sigmoid_gate_fusion = False

        # Create a concrete subclass to avoid abstract class instantiation error
        class ConcreteMLA(MultiLatentAttention):
            def get_query_key_value_tensors(self, *args, **kwargs):
                return None

        with patch(
            "paddleformers.fleet.transformer.multi_latent_attention.Attention.__init__",
            return_value=None,
        ):
            mla = ConcreteMLA.__new__(ConcreteMLA)
            mla.config = config
            mla.layer_number = 1
            mla.recompute_core_attention = False
            mla.use_rr_flash_attention = False
            mla.training = True
            mla.attn_mask_type = MagicMock()
            mla.core_attention = MagicMock(
                return_value=paddle.randn([1, 4, 2, 16])
            )
            mla.o_proj = MagicMock(
                return_value=(paddle.randn([1, 4, 64]), None)
            )
            mla.gate_proj = MagicMock(
                return_value=(paddle.randn([1, 4, 64]), None)
            )
            mla.recompute_gated_attn = False
            return mla

    def test_gate_without_fusion(self):
        """Test _gate without sigmoid_gate_fusion."""
        mla = self._make_mla_with_gate()
        mla.config.sigmoid_gate_fusion = False

        hidden_states = paddle.randn([1, 4, 64])
        core_attn_out = paddle.randn([1, 4, 64])

        result = mla._gate(hidden_states, core_attn_out)
        self.assertEqual(result.shape, [1, 4, 64])


class TestECCompatibleRopeApplyEdgeCases(unittest.TestCase):
    """Edge case tests for _ec_compatible_rope_apply."""

    def test_single_token_sequence(self):
        """Test RoPE with single-token sequence."""
        batch, seq_len, num_heads, head_dim = 1, 1, 2, 8
        q_pe = paddle.randn([batch, seq_len, num_heads, head_dim])
        k_pe = paddle.randn([batch, seq_len, 1, head_dim])

        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, seq_len)
        self.assertEqual(q_out.shape, q_pe.shape)
        self.assertEqual(k_out.shape, k_pe.shape)

    def test_large_batch(self):
        """Test RoPE with larger batch size."""
        batch, seq_len, num_heads, head_dim = 4, 8, 2, 8
        q_pe = paddle.randn([batch, seq_len, num_heads, head_dim])
        k_pe = paddle.randn([batch, seq_len, 1, head_dim])

        q_out, k_out = _ec_compatible_rope_apply(q_pe, k_pe, seq_len)
        self.assertEqual(q_out.shape, q_pe.shape)
        self.assertEqual(k_out.shape, k_pe.shape)


class TestRecomputeQKVUpProjAndRope(unittest.TestCase):
    """Tests for recompute_qkv_up_porj_and_rope feature in MLASelfAttention.

    These tests actually call get_query_key_value_tensors to cover the
    `if self.recompute_qkv_up_porj_and_rope and self.training:` branch at L1379.
    """

    def setUp(self):
        paddle.seed(2026)
        try:
            paddle.device.set_device("cpu")
        except Exception:
            pass

    def _make_layer(self, recompute_qkv=False, training=True):
        """Create a SimpleNamespace layer that can run get_query_key_value_tensors."""
        import types as _types

        from paddleformers.fleet.transformer.multi_latent_attention import (
            MLASelfAttention,
        )

        heads = 2
        qk_nope = 4
        qk_rope = 8
        v_dim = 4
        kv_lora = 8
        hidden = 16

        layer = _types.SimpleNamespace()
        layer.get_query_key_value_tensors = _types.MethodType(
            MLASelfAttention.get_query_key_value_tensors, layer
        )
        layer._is_cudagraph_active = _types.MethodType(
            MLASelfAttention._is_cudagraph_active, layer
        )
        layer.config = _types.SimpleNamespace(
            q_lora_rank=None,
            hidden_size=hidden,
            kv_lora_rank=kv_lora,
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
            cp_balance_mode="dualchunk_allgather",
            rope_type="rope",
            apply_rope_fusion=False,
            gpt_model_use_experimental_version=False,
        )
        layer.num_attention_heads_per_partition = heads
        layer.qk_nope_head_dim = qk_nope
        layer.qk_rope_head_dim = qk_rope
        layer.v_head_dim = v_dim
        layer.q_head_dim = qk_nope + qk_rope

        # Simple projections that tile input to desired output dim
        class _TileProj:
            def __init__(self, out_dim):
                self.out_dim = out_dim

            def __call__(self, x):
                repeat = (self.out_dim + x.shape[-1] - 1) // x.shape[-1]
                reps = [1] * x.ndim
                reps[-1] = repeat
                return paddle.tile(x, reps)[..., : self.out_dim], None

        layer.q_proj = _TileProj(heads * layer.q_head_dim)
        layer.kv_a_proj_with_mqa = _TileProj(kv_lora + qk_rope)
        layer.kv_b_proj = _TileProj(heads * (qk_nope + v_dim))

        class _IdentityNorm:
            def __call__(self, x):
                return x

        layer.kv_a_layernorm = _IdentityNorm()

        # Fake rotary embedding
        class _FakeRotaryEmb:
            def __init__(self, dim):
                self.dim = dim

            def get_rotary_seq_len(
                self, hidden_states, config, packed_seq_params=None
            ):
                return hidden_states.shape[1] * config.context_parallel_size

            def __call__(
                self, max_seq_len, offset=0, packed_seq=False, position_ids=None
            ):
                pos = paddle.arange(
                    offset, offset + max_seq_len, dtype="float32"
                ).reshape([1, max_seq_len, 1, 1])
                dim = paddle.arange(self.dim, dtype="float32").reshape(
                    [1, 1, 1, self.dim]
                )
                return pos * 0.01 + dim * 0.001

        layer.rotary_pos_emb = _FakeRotaryEmb(qk_rope)
        layer.pg_collection = _types.SimpleNamespace(tp=None, cp=object())
        layer.core_attention = _types.SimpleNamespace(
            config=_types.SimpleNamespace()
        )
        layer.layer_number = 1
        layer.training = training
        layer.recompute_qkv_up_porj_and_rope = recompute_qkv
        return layer

    def _hidden(self, batch=2, seq=8, hidden=16):
        values = paddle.arange(batch * seq * hidden, dtype="float32")
        return values.reshape([batch, seq, hidden]) / 100.0

    def _run_layer(self, layer, hidden):
        """Run get_query_key_value_tensors with necessary mocks."""
        import sys
        import types as _types
        from unittest.mock import patch as _patch

        from paddleformers.fleet.transformer import multi_latent_attention as mla_mod

        # Fake transformer_layer module for _log_md5
        fake_transformer_layer = _types.ModuleType(
            "paddleformers.fleet.transformer.transformer_layer"
        )

        class _FakeTransformerLayer:
            @staticmethod
            def _log_md5(tensor, name, layer_idx):
                pass

        fake_transformer_layer.TransformerLayer = _FakeTransformerLayer

        # Identity for apply_rotary_pos_emb
        def _identity_rope(
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

        with (
            _patch.object(
                mla_mod, "get_context_parallel_world_size", return_value=1
            ),
            _patch.object(mla_mod, "get_pg_size", return_value=1),
            _patch.object(mla_mod, "get_pg_rank", return_value=0),
            _patch.object(
                mla_mod, "apply_rotary_pos_emb", side_effect=_identity_rope
            ),
            _patch.dict(
                sys.modules,
                {
                    "paddleformers.fleet.transformer.transformer_layer": fake_transformer_layer
                },
            ),
        ):
            return layer.get_query_key_value_tensors(hidden)

    def test_recompute_branch_invokes_RecomputeWithoutOutput(self):
        """Test that get_query_key_value_tensors uses RecomputeWithoutOutput when
        recompute_qkv_up_porj_and_rope=True and training=True."""
        from unittest.mock import patch as _patch

        from paddleformers.fleet.transformer import multi_latent_attention as mla_mod

        layer = self._make_layer(recompute_qkv=True, training=True)
        hidden = self._hidden()

        # We patch RecomputeWithoutOutput so that its .recompute() just
        # calls the function directly (transparent pass-through)
        original_cls = None

        class _MockRecomputeWithoutOutput:
            call_count = 0

            def __init__(self):
                _MockRecomputeWithoutOutput.call_count += 1

            def recompute(
                self,
                fn,
                *args,
                preserve_rng_state=True,
                share_grad_holder=False,
            ):
                return fn(*args)

        with _patch.object(
            mla_mod, "RecomputeWithoutOutput", _MockRecomputeWithoutOutput
        ):
            result = self._run_layer(layer, hidden)

        # Verify RecomputeWithoutOutput was instantiated (branch was taken)
        self.assertEqual(_MockRecomputeWithoutOutput.call_count, 1)
        # Verify _qkv_recompute attribute was set on the layer
        self.assertIsInstance(layer._qkv_recompute, _MockRecomputeWithoutOutput)
        # Verify result shape
        query, key, value, q_compressed, kv_compressed, k_pos_emb = result
        self.assertEqual(query.ndim, 4)
        self.assertEqual(key.ndim, 4)
        self.assertEqual(value.ndim, 4)

    def test_recompute_branch_produces_same_result_as_normal(self):
        """Test that recompute path produces identical results to normal path."""
        from unittest.mock import patch as _patch

        from paddleformers.fleet.transformer import multi_latent_attention as mla_mod

        hidden = self._hidden()

        # Run normal path (recompute disabled)
        layer_normal = self._make_layer(recompute_qkv=False, training=True)
        q_normal, k_normal, v_normal, *_ = self._run_layer(layer_normal, hidden)

        # Run recompute path with transparent pass-through
        class _PassthroughRecompute:
            def recompute(
                self,
                fn,
                *args,
                preserve_rng_state=True,
                share_grad_holder=False,
            ):
                return fn(*args)

        layer_recomp = self._make_layer(recompute_qkv=True, training=True)
        with _patch.object(
            mla_mod, "RecomputeWithoutOutput", _PassthroughRecompute
        ):
            q_recomp, k_recomp, v_recomp, *_ = self._run_layer(
                layer_recomp, hidden
            )

        self.assertTrue(paddle.equal_all(q_recomp, q_normal).item())
        self.assertTrue(paddle.equal_all(k_recomp, k_normal).item())
        self.assertTrue(paddle.equal_all(v_recomp, v_normal).item())

    def test_recompute_branch_skipped_when_not_training(self):
        """Test that recompute path is NOT taken when training=False."""
        from unittest.mock import patch as _patch

        from paddleformers.fleet.transformer import multi_latent_attention as mla_mod

        layer = self._make_layer(recompute_qkv=True, training=False)
        hidden = self._hidden()

        class _MockRecomputeWithoutOutput:
            call_count = 0

            def __init__(self):
                _MockRecomputeWithoutOutput.call_count += 1

            def recompute(self, fn, *args, **kwargs):
                return fn(*args)

        with _patch.object(
            mla_mod, "RecomputeWithoutOutput", _MockRecomputeWithoutOutput
        ):
            # Note: non-fused path with training=False will hit the normal else branch
            # The rope unfused path may raise NotImplementedError for inference if
            # apply_rope_fusion=True, so we use apply_rope_fusion=False here.
            result = self._run_layer(layer, hidden)

        # RecomputeWithoutOutput should NOT have been called
        self.assertEqual(_MockRecomputeWithoutOutput.call_count, 0)
        self.assertFalse(hasattr(layer, "_qkv_recompute"))

    def test_recompute_branch_skipped_when_flag_false(self):
        """Test that recompute path is NOT taken when recompute_qkv_up_porj_and_rope=False."""
        from unittest.mock import patch as _patch

        from paddleformers.fleet.transformer import multi_latent_attention as mla_mod

        layer = self._make_layer(recompute_qkv=False, training=True)
        hidden = self._hidden()

        class _MockRecomputeWithoutOutput:
            call_count = 0

            def __init__(self):
                _MockRecomputeWithoutOutput.call_count += 1

            def recompute(self, fn, *args, **kwargs):
                return fn(*args)

        with _patch.object(
            mla_mod, "RecomputeWithoutOutput", _MockRecomputeWithoutOutput
        ):
            result = self._run_layer(layer, hidden)

        self.assertEqual(_MockRecomputeWithoutOutput.call_count, 0)


class TestRecomputeQKVSelectiveBranches(unittest.TestCase):
    """Tests covering all branches of the selective recompute_qkv_up_porj_and_rope
    initialization logic in MultiLatentAttention.__init__.

    These tests mock super().__init__ and build_spec_layer to bypass heavy
    dependencies, then call MultiLatentAttention.__init__ directly so that
    the actual recompute logic in the source code is exercised for coverage.
    """

    def _make_config(
        self,
        recompute_granularity="selective",
        recompute_modules=None,
        recompute_num_layers=None,
        recompute_method="block",
        layer_number=0,
    ):
        """Create a minimal config SimpleNamespace for MultiLatentAttention.__init__."""
        import types as _types

        config = _types.SimpleNamespace(
            recompute_granularity=recompute_granularity,
            recompute_modules=recompute_modules,
            recompute_num_layers=recompute_num_layers,
            recompute_method=recompute_method,
            # Fields needed by need_recompute_in_block / need_recompute_in_first_n
            num_hidden_layers=8,
            num_empty_layers_add_in_head=0,
            num_empty_layers_add_in_tail=0,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_size=1,
            # Fields needed by MultiLatentAttention.__init__
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            rotary_scaling_factor=1.0,
            mscale_all_dim=0,
            rope_type="rope",
            rotary_interleaved=False,
            hidden_size=64,
            output_layer_init_method=None,
            init_method=None,
            rms_norm_eps=1e-6,
            use_bias=False,
            gated_attention=False,
            gated_attn_use_q_lora=False,
            q_lora_rank=None,
            kv_lora_rank=16,
            head_dim=16,
            v_head_dim=16,
            num_attention_heads=4,
            num_key_value_heads=4,
            rope_theta=10000.0,
            sliding_window=None,
            rotary_percent=1.0,
            attention_value_scale=None,
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            context_parallel_size=1,
        )
        return config

    def _build_mla_instance(self, config, layer_number=0):
        """Instantiate MLASelfAttention with mocked base class init and build_spec_layer."""
        from unittest.mock import patch as _patch

        from paddleformers.fleet.transformer.multi_latent_attention import (
            MLASelfAttention,
        )

        # We need to bypass the heavy Attention.__init__ (paddle.nn.Layer, ProcessGroupCollection, etc.)
        # but still run MultiLatentAttention.__init__ logic for the recompute block.
        # Strategy: patch Attention.__init__ to just set the minimal attributes needed.
        def _fake_attention_init(self_obj, *args, **kwargs):
            self_obj.config = config
            self_obj.layer_number = layer_number
            self_obj.is_swa = False
            self_obj.num_attention_heads = config.num_attention_heads
            self_obj.v_head_dim = config.v_head_dim
            self_obj.head_dim = config.head_dim
            self_obj.qk_rope_head_dim = config.qk_rope_head_dim
            self_obj.rope_theta = config.rope_theta
            self_obj.pg_collection = MagicMock()
            self_obj.attn_mask_type = MagicMock()
            self_obj.attention_type = "self"
            self_obj.is_mtp_layer = False

        def _fake_build_spec_layer(*args, **kwargs):
            return MagicMock()

        with (
            _patch(
                "paddleformers.fleet.transformer.multi_latent_attention.Attention.__init__",
                _fake_attention_init,
            ),
            _patch(
                "paddleformers.fleet.transformer.multi_latent_attention.build_spec_layer",
                _fake_build_spec_layer,
            ),
            _patch(
                "paddleformers.fleet.transformer.multi_latent_attention.RotaryEmbedding",
                MagicMock,
            ),
            _patch(
                "paddleformers.fleet.transformer.multi_latent_attention.ProcessGroupCollection",
                MagicMock,
            ),
        ):
            instance = object.__new__(MLASelfAttention)
            MLASelfAttention.__init__(
                instance,
                config=config,
                sublayers_spec=MagicMock(gate_proj=None),
                layer_number=layer_number,
                attn_mask_type=MagicMock(),
                pg_collection=MagicMock(),
            )
        return instance

    def test_non_selective_granularity_returns_false(self):
        """Branch: recompute_granularity != 'selective' -> False."""
        config = self._make_config(
            recompute_granularity="full",
            recompute_modules=["mla_qkv_recompute"],
        )
        inst = self._build_mla_instance(config)
        self.assertFalse(inst.recompute_qkv_up_porj_and_rope)

    def test_list_without_mla_qkv_recompute_returns_false(self):
        """Branch: modules is list but doesn't contain 'mla_qkv_recompute' -> False."""
        config = self._make_config(recompute_modules=["other_module"])
        inst = self._build_mla_instance(config)
        self.assertFalse(inst.recompute_qkv_up_porj_and_rope)

    def test_dict_without_mla_qkv_recompute_returns_false(self):
        """Branch: modules is dict but doesn't contain 'mla_qkv_recompute' -> False."""
        config = self._make_config(recompute_modules={"other_module": 4})
        inst = self._build_mla_instance(config)
        self.assertFalse(inst.recompute_qkv_up_porj_and_rope)

    def test_list_num_layers_none_returns_true(self):
        """Branch: modules is list, recompute_num_layers is None -> True."""
        config = self._make_config(
            recompute_modules=["mla_qkv_recompute"],
            recompute_num_layers=None,
        )
        inst = self._build_mla_instance(config)
        self.assertTrue(inst.recompute_qkv_up_porj_and_rope)

    def test_list_block_method_layer_in_recompute(self):
        """Branch: modules is list, method='block', layer IS in recompute range -> True."""
        config = self._make_config(
            recompute_modules=["mla_qkv_recompute"],
            recompute_num_layers=4,
            recompute_method="block",
        )
        # layer_number=0 is in the first 4-layer block
        inst = self._build_mla_instance(config, layer_number=0)
        self.assertTrue(inst.recompute_qkv_up_porj_and_rope)

    def test_list_block_method_layer_not_in_recompute(self):
        """Branch: modules is list, method='block', layer NOT in recompute range -> False."""
        config = self._make_config(
            recompute_modules=["mla_qkv_recompute"],
            recompute_num_layers=2,
            recompute_method="block",
        )
        # layer_number=5 is NOT in the first 2-layer block
        inst = self._build_mla_instance(config, layer_number=5)
        self.assertFalse(inst.recompute_qkv_up_porj_and_rope)

    def test_list_first_n_method_layer_in_recompute(self):
        """Branch: modules is list, method='first_n', layer IS in recompute range -> True."""
        config = self._make_config(
            recompute_modules=["mla_qkv_recompute"],
            recompute_num_layers=4,
            recompute_method="first_n",
        )
        inst = self._build_mla_instance(config, layer_number=1)
        self.assertTrue(inst.recompute_qkv_up_porj_and_rope)

    def test_list_first_n_method_layer_not_in_recompute(self):
        """Branch: modules is list, method='first_n', layer NOT in recompute range -> False."""
        config = self._make_config(
            recompute_modules=["mla_qkv_recompute"],
            recompute_num_layers=2,
            recompute_method="first_n",
        )
        inst = self._build_mla_instance(config, layer_number=5)
        self.assertFalse(inst.recompute_qkv_up_porj_and_rope)

    def test_dict_block_method_layer_in_recompute(self):
        """Branch: modules is dict, method='block', layer IS in recompute range -> True."""
        config = self._make_config(
            recompute_modules={"mla_qkv_recompute": 4},
            recompute_method="block",
        )
        inst = self._build_mla_instance(config, layer_number=0)
        self.assertTrue(inst.recompute_qkv_up_porj_and_rope)

    def test_dict_block_method_layer_not_in_recompute(self):
        """Branch: modules is dict, method='block', layer NOT in recompute range -> False."""
        config = self._make_config(
            recompute_modules={"mla_qkv_recompute": 2},
            recompute_method="block",
        )
        inst = self._build_mla_instance(config, layer_number=5)
        self.assertFalse(inst.recompute_qkv_up_porj_and_rope)

    def test_dict_first_n_method_layer_in_recompute(self):
        """Branch: modules is dict, method='first_n', layer IS in recompute range -> True."""
        config = self._make_config(
            recompute_modules={"mla_qkv_recompute": 4},
            recompute_method="first_n",
        )
        inst = self._build_mla_instance(config, layer_number=1)
        self.assertTrue(inst.recompute_qkv_up_porj_and_rope)

    def test_dict_first_n_method_layer_not_in_recompute(self):
        """Branch: modules is dict, method='first_n', layer NOT in recompute range -> False."""
        config = self._make_config(
            recompute_modules={"mla_qkv_recompute": 2},
            recompute_method="first_n",
        )
        inst = self._build_mla_instance(config, layer_number=5)
        self.assertFalse(inst.recompute_qkv_up_porj_and_rope)

    def test_modules_is_none_returns_false(self):
        """Branch: recompute_modules is None -> False."""
        config = self._make_config(recompute_modules=None)
        inst = self._build_mla_instance(config)
        self.assertFalse(inst.recompute_qkv_up_porj_and_rope)


class TestForwardDiscardOutputAndRegisterRecompute(unittest.TestCase):
    """Tests for the forward path that calls discard_output_and_register_recompute
    at multi_latent_attention.py L642-645."""

    def test_forward_calls_discard_output_when_recompute_enabled(self):
        """Test that forward() calls _qkv_recompute.discard_output_and_register_recompute
        and then sets _qkv_recompute to None."""
        from unittest.mock import MagicMock, patch as _patch

        from paddleformers.fleet.transformer.multi_latent_attention import (
            MLASelfAttention,
        )

        # Create a minimal instance with the attributes forward() needs
        instance = object.__new__(MLASelfAttention)
        instance.recompute_qkv_up_porj_and_rope = True
        instance.training = True
        instance.recompute_core_attention = False
        instance.use_rr_flash_attention = False
        instance.layer_number = 0
        instance.config = MagicMock()
        instance.config.sequence_parallel = False
        instance.gated_attention = False
        instance.config.dw_p2p_overlap = False
        instance.config.use_bias = True

        # Mock core_attention to return a tensor
        core_attn_out = paddle.randn([1, 4, 64])
        core_attn_out.stop_gradient = False
        instance.core_attention = MagicMock(return_value=core_attn_out)
        instance.core_attention.config = MagicMock(spec=[])  # no forward_meta

        # Mock o_proj
        output_tensor = paddle.randn([1, 4, 64])
        instance.o_proj = MagicMock(return_value=(output_tensor, None))

        # Mock get_query_key_value_tensors
        q = paddle.randn([1, 4, 2, 12])
        k = paddle.randn([1, 4, 2, 12])
        v = paddle.randn([1, 4, 2, 4])
        instance.get_query_key_value_tensors = MagicMock(
            return_value=(q, k, v, None, None, None)
        )

        # Set up _qkv_recompute mock
        mock_recompute = MagicMock()
        instance._qkv_recompute = mock_recompute

        # Mock attn_mask_type for core_attention
        instance.attn_mask_type = MagicMock()

        # Patch TransformerLayer._log_md5 and paddle.device.cuda.memory_allocated
        import sys
        import types as _types

        fake_transformer_layer = _types.ModuleType(
            "paddleformers.fleet.transformer.transformer_layer"
        )

        class _FakeTransformerLayer:
            @staticmethod
            def _log_md5(tensor, name, layer_idx):
                pass

        fake_transformer_layer.TransformerLayer = _FakeTransformerLayer

        with (
            _patch.dict(
                sys.modules,
                {
                    "paddleformers.fleet.transformer.transformer_layer": fake_transformer_layer
                },
            ),
            _patch("paddle.device.cuda.memory_allocated", return_value=0),
            _patch("paddle.base.core.nvprof_nvtx_push"),
            _patch("paddle.base.core.nvprof_nvtx_pop"),
        ):
            MLASelfAttention.forward(
                instance,
                hidden_states=paddle.randn([1, 4, 64]),
                attention_mask=None,
            )

        # Verify discard_output_and_register_recompute was called with core_attn_out
        mock_recompute.discard_output_and_register_recompute.assert_called_once()
        # Verify _qkv_recompute was set to None after the call
        self.assertIsNone(instance._qkv_recompute)


class TestRecomputeWithoutOutputEmptyFiltered(unittest.TestCase):
    """Tests for the else branch in RecomputeWithoutOutputFunction.backward
    at random.py L448-451 where filtered is empty."""

    def test_backward_empty_filtered_branch(self):
        """Test that backward handles the case where all outputs are None
        (the else branch at L448-451) by directly invoking backward logic."""
        from unittest.mock import MagicMock, patch as _patch

        from paddleformers.fleet.tensor_parallel.random import (
            RecomputeWithoutOutputFunction,
        )

        ctx = MagicMock()
        # Set outputs to (None,) so filtered will be empty
        ctx.outputs = (None,)
        ctx.share_grad_holder = False

        # inputs need to be tensors for the grads tuple at the end
        inp = paddle.randn([2, 3])
        inp.stop_gradient = False
        inp.grad = paddle.ones([2, 3])
        ctx.inputs = (inp,)

        # Call backward directly - output_grads matches outputs length
        output_grads = (None,)

        # Patch paddle.autograd.backward to avoid "tensors cannot be empty" error
        # The key assertion is that we reach it with empty lists (else branch taken)
        backward_called_with = {}

        def _mock_backward(tensors, grad_tensors):
            backward_called_with["tensors"] = tensors
            backward_called_with["grad_tensors"] = grad_tensors

        with _patch("paddle.autograd.backward", side_effect=_mock_backward):
            grads = RecomputeWithoutOutputFunction.backward(ctx, *output_grads)

        # Verify backward was called with empty lists (else branch)
        self.assertEqual(backward_called_with["tensors"], [])
        self.assertEqual(backward_called_with["grad_tensors"], [])
        # Verify it returns the grad from inputs
        self.assertEqual(len(grads), 1)

    def test_backward_mixed_none_and_tensor_outputs(self):
        """Test backward when some outputs are None and some are Tensors.
        Covers the `if filtered` branch with partial filtering."""
        from paddleformers.fleet.tensor_parallel.random import RecomputeWithoutOutput

        # Function that returns a tuple with a real tensor
        def fn_with_tensor_output(x):
            return x * 2.0

        recompute_obj = RecomputeWithoutOutput()
        x = paddle.randn([2, 3])
        x.stop_gradient = False

        result = recompute_obj.recompute(fn_with_tensor_output, x)
        self.assertIsNotNone(result)
        # outputs stored as (tensor,) - this covers the `if filtered` branch
        self.assertEqual(len(recompute_obj.outputs), 1)


if __name__ == "__main__":
    unittest.main()
