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

"""Tests for MTP Magic Send feature."""

import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import paddle
from paddle import nn

from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.models.gpt.mtp_embedding_layer import (
    MTPEmbeddingLayer,
    input_ids_for_mtp,
)
from paddleformers.fleet.pipeline_parallel.pp_utils.pp_comm_utils import (
    _DtypeSndShape,
    broadcast_data_obj,
    init_magic_send_comm_group,
    split_group,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

# =====================================================================
# Helpers
# =====================================================================

_BASE_CFG = {
    "vocab_size": 512,
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "tensor_model_parallel_size": 1,
    "expert_model_parallel_size": 1,
}


def _cfg(**kw):
    return GPTConfig(
        **{
            **_BASE_CFG,
            "enable_mtp_magic_send": True,
            "num_nextn_predict_layers": 1,
            "pipeline_model_parallel_size": 2,
            **kw,
        }
    )


class _FakeNorm(nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()

    def forward(self, x):
        return x


class _FakeLinear(nn.Layer):
    def __init__(self, in_f=128, out_f=64, *a, **kw):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias_attr=False)

    def forward(self, x):
        return self.linear(x), None


class _FakeTransformerLayer(nn.Layer):
    def __init__(self, *a, **kw):
        super().__init__()

    def forward(self, d):
        return {"hidden_states": d["hidden_states"]}


@dataclass
class _FakeMTPSpec:
    enorm: object = None
    hnorm: object = None
    eh_proj: object = None
    e_proj: object = None
    h_proj: object = None
    transformer_layer: object = None
    layer_norm: object = None

    def __post_init__(self):
        tl = MagicMock()
        tl.sublayers_spec = MagicMock()
        tl.sublayers_spec.self_attn = MagicMock()
        tl.sublayers_spec.self_attn.extra_kwargs = {"attn_mask_type": AttnMaskType.causal}
        self.transformer_layer = tl


def _build_mtp_layer(config, layer_number=0):
    from paddleformers.fleet.transformer.multi_token_prediction import (
        MultiTokenPredictionLayer,
    )

    spec = _FakeMTPSpec()
    mock_pg = MagicMock(cp=None, tp=None)
    with (
        patch(
            "paddleformers.fleet.transformer.multi_token_prediction.build_spec_layer",
            side_effect=lambda s, *a, **kw: _FakeTransformerLayer() if s is spec.transformer_layer else _FakeNorm(),
        ),
        patch(
            "paddleformers.fleet.transformer.multi_token_prediction.ProcessGroupCollection.use_mpu_process_groups",
            return_value=mock_pg,
        ),
        patch(
            "paddleformers.fleet.transformer.multi_token_prediction.paddle.distributed.get_world_size",
            return_value=1,
        ),
    ):
        layer = MultiTokenPredictionLayer(
            config=config,
            sublayers_spec=spec,
            layer_number=layer_number,
            pg_collection=mock_pg,
        )

    if layer.eh_proj is not None:
        layer.eh_proj = _FakeLinear(config.hidden_size * 2, config.hidden_size)
    layer.enorm = _FakeNorm()
    layer.hnorm = _FakeNorm()
    if hasattr(layer, "norm") and layer.norm is not None:
        layer.norm = _FakeNorm()
    layer.transformer_layer = _FakeTransformerLayer()
    return layer


def _build_gpt_embedding(config):
    from paddleformers.fleet.models.gpt.gpt_embedding import GPTEmbedding

    mock_spec = MagicMock(rope_embedding=None)

    class _Emb(nn.Layer):
        def __init__(self, v, h):
            super().__init__()
            self.embed_tokens = nn.Embedding(v, h)
            self.reduce_scatter_embeddings = self.scatter_to_sequence_parallel = self.sequence_parallel = False

        @property
        def embedding_weight(self):
            return self.embed_tokens.weight

        def forward(self, input_ids, position_ids=None):
            return self.embed_tokens(input_ids)

    emb_layer = _Emb(config.vocab_size, config.hidden_size)
    with (
        patch(
            "paddleformers.fleet.models.gpt.gpt_embedding.build_spec_layer",
            side_effect=lambda s, *a, **kw: emb_layer if s is mock_spec.language_embedding else None,
        ),
        patch("paddleformers.fleet.models.gpt.gpt_embedding.mark_context_parallel_parameter_disable_scale_grad"),
    ):
        return GPTEmbedding(
            sublayers_spec=mock_spec,
            config=config,
            vocab_size=config.vocab_size,
            max_sequence_length=128,
            position_embedding_type="rope",
        )


def _mtp_forward_ctx(
    cp_world_size=1,
    scatter_fn=None,
    cp_scatter_fn=None,
    proj_override=None,
    layer=None,
):
    """Returns a context manager that sets up all MTP forward mocks."""
    stack = ExitStack()

    def _enter(stack_ref):
        stack_ref.enter_context(
            patch(
                "paddleformers.fleet.transformer.multi_token_prediction.get_context_parallel_world_size",
                return_value=cp_world_size,
            )
        )
        tp = stack_ref.enter_context(patch("paddleformers.fleet.transformer.multi_token_prediction.tensor_parallel"))
        tp.get_cuda_rng_tracker.return_value.fork.return_value = nullcontext()

        if scatter_fn is not None:
            so = stack_ref.enter_context(patch("paddleformers.fleet.transformer.multi_token_prediction.ScatterOp"))
            so.apply = scatter_fn
        if cp_scatter_fn is not None:
            co = stack_ref.enter_context(
                patch("paddleformers.fleet.transformer.multi_token_prediction.ContextParallelScatterOp")
            )
            co.apply = cp_scatter_fn
        if proj_override is not None and layer is not None:
            stack_ref.enter_context(
                patch.object(
                    layer,
                    "_proj_and_transformer_layer",
                    side_effect=proj_override,
                )
            )

    class _Ctx:
        def __enter__(self):
            _enter(stack)
            return stack

        def __exit__(self, *a):
            stack.__exit__(*a)

    return _Ctx()


# =====================================================================
# Tests
# =====================================================================


class TestTransformerConfig(unittest.TestCase):
    def test_magic_send_validation(self):
        self.assertFalse(TransformerConfig().enable_mtp_magic_send)
        self.assertTrue(
            TransformerConfig(
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
            ).enable_mtp_magic_send
        )
        with self.assertRaises(AssertionError):
            TransformerConfig(
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=2,
                pipeline_model_parallel_size=2,
            )
        with self.assertRaises(AssertionError):
            TransformerConfig(
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=1,
            )

    def test_magic_send_vpp_requires_overlap_and_variable_seq(self):
        # vpp + overlap_p2p_comm + variable_seq_lengths => OK
        cfg = TransformerConfig(
            enable_mtp_magic_send=True,
            num_nextn_predict_layers=1,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            overlap_p2p_comm=True,
            variable_seq_lengths=True,
        )
        self.assertTrue(cfg.enable_mtp_magic_send)

        # vpp without overlap_p2p_comm => AssertionError
        with self.assertRaises(AssertionError):
            TransformerConfig(
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=2,
                overlap_p2p_comm=False,
                variable_seq_lengths=True,
            )

        # vpp without variable_seq_lengths => AssertionError
        with self.assertRaises(AssertionError):
            TransformerConfig(
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=2,
                overlap_p2p_comm=True,
                variable_seq_lengths=False,
            )

        # vpp without both => AssertionError
        with self.assertRaises(AssertionError):
            TransformerConfig(
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=2,
                overlap_p2p_comm=False,
                variable_seq_lengths=False,
            )


class TestMTPEmbeddingLayer(unittest.TestCase):
    def setUp(self):
        self.config = _cfg(vocab_size=1024, hidden_size=64)

    def test_forward_and_errors(self):
        layer = MTPEmbeddingLayer(config=self.config)
        self.assertEqual(layer.embedding_weight.shape, [1024, 64])

        # Normal forward
        input_ids_for_mtp.clear()
        input_ids_for_mtp.append(paddle.randint(0, 1024, [2, 10]))
        result = layer.forward({"hidden_states": paddle.randn([2, 9, 64])})
        self.assertEqual(result["mtp_input_embeds"].shape, [2, 10, 64])

        # Empty deque error
        input_ids_for_mtp.clear()
        with self.assertRaises(RuntimeError):
            layer.forward({"hidden_states": paddle.randn([2, 9, 64])})

        # Disabled flag error
        layer.config.enable_mtp_magic_send = False
        input_ids_for_mtp.append(paddle.randint(0, 1024, [2, 10]))
        with self.assertRaises(RuntimeError):
            layer.forward({"hidden_states": paddle.randn([2, 9, 64])})

    def test_fill_feature_with_ep_gt1(self):
        layer = MTPEmbeddingLayer(config=_cfg(vocab_size=1024, hidden_size=64, expert_model_parallel_size=4))
        ids = paddle.randint(1, 1024, [1, 5])
        ids[0, 2] = 0
        input_ids_for_mtp.clear()
        input_ids_for_mtp.append(ids)
        result = layer.forward({"hidden_states": paddle.randn([1, 4, 64])})
        self.assertTrue(paddle.allclose(result["mtp_input_embeds"][0, 2, :], paddle.zeros([64])).item())

    def test_fill_feature_uses_custom_pad_token_id(self):
        """When pad_token_id != 0, only that id triggers fill_feature zeroing."""
        layer = MTPEmbeddingLayer(
            config=_cfg(
                vocab_size=1024,
                hidden_size=64,
                expert_model_parallel_size=4,
                pad_token_id=42,
            )
        )
        # Force non-zero weights so fill_feature's effect is observable
        # (perform_initialization=False otherwise leaves weights at zero).
        layer.embed_tokens.weight.set_value(paddle.ones_like(layer.embed_tokens.weight))
        # Position 1 holds id 0 (must remain non-zero), position 3 holds id 42 (must be zeroed).
        ids = paddle.to_tensor([[5, 0, 7, 42, 9]])
        input_ids_for_mtp.clear()
        input_ids_for_mtp.append(ids)
        result = layer.forward({"hidden_states": paddle.randn([1, 4, 64])})
        embeds = result["mtp_input_embeds"]
        self.assertTrue(paddle.allclose(embeds[0, 3, :], paddle.zeros([64])).item())
        # id == 0 must NOT be treated as padding when pad_token_id == 42.
        self.assertFalse(paddle.allclose(embeds[0, 1, :], paddle.zeros([64])).item())

    def test_fill_feature_handles_none_pad_token_id(self):
        """A runtime-None pad_token_id falls back to 0 instead of erroring."""
        layer = MTPEmbeddingLayer(config=_cfg(vocab_size=1024, hidden_size=64, expert_model_parallel_size=4))
        # Override after construction to simulate external config injecting None.
        layer.config.pad_token_id = None
        layer.embed_tokens.weight.set_value(paddle.ones_like(layer.embed_tokens.weight))
        ids = paddle.to_tensor([[5, 0, 7, 8, 9]])
        input_ids_for_mtp.clear()
        input_ids_for_mtp.append(ids)
        result = layer.forward({"hidden_states": paddle.randn([1, 4, 64])})
        # Fallback treats id 0 as padding.
        self.assertTrue(paddle.allclose(result["mtp_input_embeds"][0, 1, :], paddle.zeros([64])).item())
        # id != 0 stays non-zero.
        self.assertFalse(paddle.allclose(result["mtp_input_embeds"][0, 0, :], paddle.zeros([64])).item())


class TestWrappedPaddleNormPipe(unittest.TestCase):
    def test_magic_send_vs_non_magic_send(self):
        from paddleformers.fleet.transformer.paddle_norm import WrappedPaddleNormPipe

        cfg_on = TransformerConfig(
            enable_mtp_magic_send=True,
            num_nextn_predict_layers=1,
            pipeline_model_parallel_size=2,
            hidden_size=64,
            normalization="RMSNorm",
            tensor_model_parallel_size=1,
        )
        r = WrappedPaddleNormPipe(cfg_on, hidden_size=64).forward({"hidden_states": paddle.ones([1, 4, 64]) * 2.0})
        self.assertTrue(paddle.allclose(r["hidden_states"], paddle.ones([1, 4, 64]), atol=1e-5).item())

        cfg_off = TransformerConfig(
            enable_mtp_magic_send=False,
            num_nextn_predict_layers=1,
            hidden_size=64,
            normalization="RMSNorm",
            tensor_model_parallel_size=1,
        )
        r = WrappedPaddleNormPipe(cfg_off, hidden_size=64).forward({"hidden_states": paddle.randn([4, 8, 64])})
        self.assertEqual(r["hidden_states"].shape, [4, 8, 64])


class TestMTPLayerForward(unittest.TestCase):
    """MultiTokenPredictionLayer.forward() magic send branch."""

    def test_basic_forward_all_fields(self):
        """Non-SP: correct shape, all aux keys stripped."""
        layer = _build_mtp_layer(_cfg())
        B, S, H = 2, 8, 64
        with _mtp_forward_ctx():
            result = layer.forward(
                {
                    "hidden_states": paddle.randn([B, S, H]),
                    "mtp_input_embeds": paddle.randn([B, S + 1, H]),
                    "attn_mask_startend_row_indices": paddle.randn([B, S, 4]),
                    "mtp_startend_row_indices_all": paddle.randn([B, 1, S, 4]),
                    "mtp_hidden_inputs_mask_all": paddle.ones([B, 1, S]),
                    "mtp_input_ids_for_moe_mask": paddle.randint(0, 512, [B, 1, S]),
                    "input_ids": paddle.randint(0, 512, [B, S]),
                    "rotary_pos_emb": paddle.randn([1, S + 2, 32]),
                    "rotary_pos_cos": paddle.randn([1, S + 2, 32]),
                    "rotary_pos_sin": paddle.randn([1, S + 2, 32]),
                    "labels": paddle.randint(0, 100, [B, S]),
                }
            )
        self.assertEqual(result["hidden_states"].shape, [2 * B, S, H])
        self.assertEqual(set(result.keys()), {"hidden_states", "labels"})

    def test_optional_fields_exp_versions(self):
        B, S, H = 2, 8, 64
        for exp_ver in [True, False]:
            layer = _build_mtp_layer(_cfg(gpt_model_use_experimental_version=exp_ver))
            with _mtp_forward_ctx():
                result = layer.forward(
                    {
                        "hidden_states": paddle.randn([B, S, H]),
                        "mtp_input_embeds": paddle.randn([B, S + 1, H]),
                        "mtp_startend_row_indices_all": paddle.randn([B, 1, S, 4]),
                        "mtp_hidden_inputs_mask_all": paddle.ones([B, 1, S]),
                        "labels": paddle.randint(0, 100, [B, S]),
                    }
                )
            self.assertIn("hidden_states", result)

    def test_missing_embeds_raises(self):
        layer = _build_mtp_layer(_cfg())
        with self.assertRaises(RuntimeError), _mtp_forward_ctx():
            layer.forward(
                {
                    "hidden_states": paddle.randn([2, 8, 64]),
                    "labels": paddle.randint(0, 100, [2, 8]),
                }
            )

    def test_recompute(self):
        config = _cfg(
            normalization="RMSNorm",
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=1,
        )
        layer = _build_mtp_layer(config)
        layer.training = True
        with (
            patch.object(
                layer,
                "_checkpointed_forward",
                side_effect=lambda fn, **kw: fn(**kw),
            ),
            _mtp_forward_ctx(),
        ):
            result = layer.forward(
                {
                    "hidden_states": paddle.randn([2, 8, 64]),
                    "mtp_input_embeds": paddle.randn([2, 9, 64]),
                    "labels": paddle.randint(0, 100, [2, 8]),
                }
            )
        self.assertIn("hidden_states", result)

    def test_sp_scatter(self):
        """SP mode: ScatterOp applied, decoder_input becomes [S/tp, B, H]."""
        config = _cfg(sequence_parallel=True, tensor_model_parallel_size=2)
        layer = _build_mtp_layer(config)
        S_local, B, H = 4, 2, 64
        captured = {}

        def proj(hidden_states, decoder_input, **kw):
            captured["di"] = decoder_input
            return hidden_states

        scatter_count = [0]

        def scatter_fn(x):
            scatter_count[0] += 1
            return x[: x.shape[0] // 2]

        with _mtp_forward_ctx(scatter_fn=scatter_fn, proj_override=proj, layer=layer):
            layer.forward(
                {
                    "hidden_states": paddle.randn([S_local, B, H]),
                    "mtp_input_embeds": paddle.randn([B, S_local * 2 + 1, H]),
                    "rotary_pos_emb": paddle.randn([S_local * 2 + 4, 32]),
                    "labels": paddle.randint(0, 100, [B, S_local * 2]),
                }
            )
        self.assertEqual(scatter_count[0], 1)
        self.assertEqual(list(captured["di"].shape), [S_local, B, H])

    def test_cp_scatter(self):
        """CP mode: hidden_states is CP-local [B, S/CP, H]; decoder_input sliced at global S then scattered to [B, S/CP, H]."""
        config = _cfg(experimental_dataflow=True)
        layer = _build_mtp_layer(config)
        B, S_global, H, CP = 2, 8, 64, 2
        S_local = S_global // CP  # hidden_states arriving is already CP-local
        captured = {}

        def proj(hidden_states, decoder_input, **kw):
            captured["di"] = decoder_input
            return hidden_states

        cp_count = [0]

        def cp_fn(x, axis=0, **kwargs):
            cp_count[0] += 1
            return x[:, : x.shape[1] // 2, :] if axis == 1 else x[: x.shape[0] // 2]

        with _mtp_forward_ctx(
            cp_world_size=CP,
            cp_scatter_fn=cp_fn,
            proj_override=proj,
            layer=layer,
        ):
            layer.forward(
                {
                    "hidden_states": paddle.randn([B, S_local, H]),  # CP-local
                    "mtp_input_embeds": paddle.randn([B, S_global + 1, H]),  # full global
                    "labels": paddle.randint(0, 100, [B, S_global]),
                }
            )
        self.assertEqual(cp_count[0], 1)
        # After scatter: [B, S_global, H] -> [B, S_global/CP, H] = [B, S_local, H]
        self.assertEqual(list(captured["di"].shape), [B, S_local, H])

    def test_cp_and_sp_combined(self):
        """CP + SP: hidden_states is [S/(TP*CP), B, H]; global seq_len recovered, then CP scatter + SP scatter."""
        config = _cfg(
            sequence_parallel=True,
            tensor_model_parallel_size=2,
            experimental_dataflow=True,
        )
        layer = _build_mtp_layer(config)
        CP, TP = 2, 2
        S_global = 16
        S_sp_cp_local = S_global // (TP * CP)  # = 4, what hidden_states.shape[0] is
        B, H = 2, 64
        captured = {}

        def proj(hidden_states, decoder_input, **kw):
            captured["di"] = decoder_input
            return hidden_states

        def cp_fn(x, axis=0, **kwargs):
            return x[:, : x.shape[1] // 2, :] if axis == 1 else x[: x.shape[0] // 2]

        def scatter_fn(x):
            return x[: x.shape[0] // 2]

        with _mtp_forward_ctx(
            cp_world_size=CP,
            scatter_fn=scatter_fn,
            cp_scatter_fn=cp_fn,
            proj_override=proj,
            layer=layer,
        ):
            layer.forward(
                {
                    "hidden_states": paddle.randn([S_sp_cp_local, B, H]),  # SP+CP local
                    "mtp_input_embeds": paddle.randn([B, S_global + 1, H]),  # full global
                    "labels": paddle.randint(0, 100, [B, S_global]),
                }
            )
        # Global slice: [B, S_global, H] = [B, 16, H]
        # CP scatter: [B, 16, H] -> [B, 8, H]
        # SP: flatten [B*8, H] -> scatter [B*8/TP, H] = [B*4, H] -> reshape [B, 4, H] -> permute [4, B, H]
        self.assertEqual(list(captured["di"].shape), [S_sp_cp_local, B, H])


class TestGPTEmbeddingForward(unittest.TestCase):
    def _run_emb(self, config, cp_world_size=1, mock_scatter=False, mock_cp=False):
        emb = _build_gpt_embedding(config)
        B, S = 2, 10
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "paddleformers.fleet.models.gpt.gpt_embedding.get_context_parallel_world_size",
                    return_value=cp_world_size,
                )
            )
            if mock_scatter:
                sc = stack.enter_context(patch("paddleformers.fleet.models.gpt.gpt_embedding.ScatterOp"))
                sc.apply = lambda x: x
            if mock_cp:
                cp = stack.enter_context(
                    patch("paddleformers.fleet.models.gpt.gpt_embedding.ContextParallelScatterOp")
                )
                cp.apply = lambda x, axis=0, **kwargs: x
            return emb.forward({"input_ids": paddle.randint(0, 512, [B, S])})

    def test_magic_send_paths(self):
        B, S, H = 2, 10, 64
        # Basic truncation
        r = self._run_emb(_cfg())
        self.assertEqual(r["hidden_states"].shape, [B, S - 1, H])
        # CP path
        r = self._run_emb(_cfg(experimental_dataflow=True), cp_world_size=2, mock_cp=True)
        self.assertEqual(r["hidden_states"].shape, [B, S - 1, H])
        # SP path
        r = self._run_emb(
            _cfg(sequence_parallel=True, tensor_model_parallel_size=2),
            mock_scatter=True,
        )
        self.assertEqual(r["hidden_states"].shape, [S - 1, B, H])

    def test_non_magic_send_paths(self):
        B, S, H, num_mtp = 2, 10, 64, 1
        r = self._run_emb(_cfg(enable_mtp_magic_send=False))
        self.assertEqual(r["hidden_states"].shape, [B * (num_mtp + 1), S - 1, H])
        # CP
        r = self._run_emb(
            _cfg(enable_mtp_magic_send=False, experimental_dataflow=True),
            cp_world_size=2,
            mock_cp=True,
        )
        self.assertEqual(r["hidden_states"].shape, [B * (num_mtp + 1), S - 1, H])
        # SP
        r = self._run_emb(
            _cfg(
                enable_mtp_magic_send=False,
                sequence_parallel=True,
                tensor_model_parallel_size=2,
            ),
            mock_scatter=True,
        )
        self.assertEqual(r["hidden_states"].shape, [(num_mtp + 1) * (S - 1), B, H])

    def test_ep_moe_mask(self):
        r = self._run_emb(_cfg(expert_model_parallel_size=4))
        self.assertEqual(r["mtp_input_ids_for_moe_mask"].shape, [2, 1, 9])


class TestPPCommUtils(unittest.TestCase):
    def test_split_group(self):
        self.assertEqual(
            len(list(split_group([(0, _DtypeSndShape("float32", [10]))], 2**18))),
            1,
        )
        self.assertGreater(
            len(
                list(
                    split_group(
                        [(i, _DtypeSndShape("float32", [200000])) for i in range(5)],
                        2**18,
                    )
                )
            ),
            1,
        )
        self.assertEqual(list(split_group([], 2**18)), [])

    def test_broadcast_data_obj(self):
        # Sender
        with (
            patch("paddle.distributed.get_rank", return_value=0),
            patch("paddle.distributed.broadcast_object_list"),
            patch("paddle.distributed.broadcast"),
        ):
            r = broadcast_data_obj(
                {"a": paddle.ones([2, 3]), "b": None},
                src_rank=0,
                group=MagicMock(),
            )
        self.assertTrue(paddle.allclose(r["a"], paddle.ones([2, 3])).item())
        self.assertIsNone(r["b"])

        # Receiver
        tmpl = _DtypeSndShape("float32", [3, 4])
        data = paddle.arange(12, dtype="float32")
        with (
            patch("paddle.distributed.get_rank", return_value=1),
            patch(
                "paddle.distributed.broadcast_object_list",
                side_effect=lambda o, s, group: o.__setitem__(0, tmpl),
            ),
            patch(
                "paddle.distributed.broadcast",
                side_effect=lambda t, s, group: t.set_value(data),
            ),
        ):
            r = broadcast_data_obj(paddle.zeros([3, 4]), src_rank=0, group=MagicMock())
        self.assertEqual(r.shape, [3, 4])

    def test_init_magic_send_comm_group(self):
        def _run(pp_ranks, rank):
            topo = MagicMock()
            topo.get_comm_list.return_value = [pp_ranks]
            hcg = MagicMock()
            hcg._topo = topo
            created = []
            with (
                patch(
                    "paddleformers.fleet.pipeline_parallel.pp_utils.pp_comm_utils.fleet.get_hybrid_communicate_group",
                    return_value=hcg,
                ),
                patch(
                    "paddle.distributed.new_group",
                    side_effect=lambda ranks: (
                        created.append(ranks),
                        MagicMock(),
                    )[1],
                ),
                patch("paddle.distributed.get_rank", return_value=rank),
            ):
                return init_magic_send_comm_group(), created

        r, c = _run([0, 1, 2, 3], 0)
        self.assertEqual(c[0], [0, 2, 3])
        self.assertIsNotNone(r)
        r, _ = _run([0, 1, 2, 3], 1)
        self.assertIsNone(r)
        r, c = _run([0, 1], 0)
        self.assertEqual(c[0], [0, 1])


class TestTransformerLayerCondition(unittest.TestCase):
    def test_condition_logic(self):
        def should_split(cfg, is_mtp=False):
            return (
                cfg.num_nextn_predict_layers is not None
                and cfg.num_nextn_predict_layers > 0
                and not is_mtp
                and not cfg.mtp_load_weight_only
                and not cfg.enable_mtp_magic_send
            )

        self.assertFalse(should_split(_cfg(enable_mtp_magic_send=True)))
        self.assertTrue(should_split(_cfg(enable_mtp_magic_send=False)))
        self.assertFalse(should_split(_cfg(enable_mtp_magic_send=False), is_mtp=True))


class TestHyperConnectionModuleInitWeights(unittest.TestCase):
    """HyperConnectionModule._init_weights: RNG tracker usage by world_size."""

    def test_init_weights_rng_tracker_dispatch(self):
        from paddleformers.fleet.transformer.hyper_connection import (
            HyperConnectionModule,
        )

        config = TransformerConfig(
            hidden_size=64,
            num_residual_streams=4,
            enable_hyper_connections=True,
            tensor_model_parallel_size=1,
            use_fused_mhc=False,
        )
        tracker_mock = MagicMock()
        tracker_mock.fork.return_value = nullcontext()

        # world_size=1: no RNG tracker
        with (
            patch(
                "paddleformers.fleet.transformer.hyper_connection.paddle.distributed.get_world_size",
                return_value=1,
            ),
            patch(
                "paddleformers.fleet.transformer.hyper_connection.get_cuda_rng_tracker",
                return_value=tracker_mock,
            ),
        ):
            m1 = HyperConnectionModule(config, layer_number=0)
        tracker_mock.fork.assert_not_called()
        self.assertFalse((m1.mapping_proj.weight.numpy() == 0).all())

        # world_size=2: uses RNG tracker
        tracker_mock.reset_mock()
        with (
            patch(
                "paddleformers.fleet.transformer.hyper_connection.paddle.distributed.get_world_size",
                return_value=2,
            ),
            patch(
                "paddleformers.fleet.transformer.hyper_connection.get_cuda_rng_tracker",
                return_value=tracker_mock,
            ),
        ):
            m2 = HyperConnectionModule(config, layer_number=0)
        tracker_mock.fork.assert_called_once()
        self.assertFalse((m2.mapping_proj.weight.numpy() == 0).all())


class TestHyperConnectionContractLayerMagicSend(unittest.TestCase):
    """HyperConnectionContractLayer.forward: magic_send branch."""

    def _build(self, magic_send, num_mtp=1):
        from paddleformers.fleet.transformer.hyper_connection import (
            HyperConnectionContractLayer,
        )

        return HyperConnectionContractLayer(
            TransformerConfig(
                hidden_size=64,
                num_residual_streams=4,
                enable_mtp_magic_send=magic_send,
                num_nextn_predict_layers=num_mtp,
                pipeline_model_parallel_size=2 if magic_send else 1,
                tensor_model_parallel_size=1,
            )
        )

    def test_magic_send_contracts_entire_tensor(self):
        layer = self._build(magic_send=True)
        B, S, H, n = 2, 8, 64, 4
        result = layer.forward({"hidden_states": paddle.randn([B, S, H * n])})
        self.assertEqual(result["hidden_states"].shape, [B, S, H])
        self.assertEqual(result["mhc_multistream"].shape, [B, S, H * n])
        self.assertTrue(layer.magic_send)

    def test_non_magic_send_splits_then_contracts(self):
        layer = self._build(magic_send=False)
        B, S, H, n = 2, 8, 64, 4
        result = layer.forward({"hidden_states": paddle.randn([B * 2, S, H * n])})
        self.assertEqual(result["hidden_states"].shape, [B * 2, S, H])
        self.assertIn("mhc_multistream", result)
        self.assertFalse(layer.magic_send)

    def test_magic_send_matches_learned_output_contract(self):
        from paddleformers.fleet.transformer.hyper_connection import (
            HyperConnectionModule,
        )

        layer = self._build(magic_send=True)
        x = paddle.randn([1, 4, 64 * 4])
        expected = HyperConnectionModule.learned_output_contract(
            x,
            layer.hc_head_fn,
            layer.hc_head_base,
            layer.hc_head_scale,
            4,
            layer.config.rms_norm_eps,
        )
        result = layer.forward({"hidden_states": x.clone()})
        self.assertTrue(paddle.allclose(result["hidden_states"], expected, atol=1e-5).item())


class TestMTPLayerMHC(unittest.TestCase):
    """MultiTokenPredictionLayer: mHC init + forward with mhc_multistream."""

    def test_hc_head_fn_init_rng_dispatch(self):
        """world_size=1 uses direct Xavier; world_size>1 uses RNG tracker fork."""
        from paddleformers.fleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )

        config = _cfg(enable_hyper_connections=True, num_residual_streams=4)

        # world_size=1: direct init
        layer = _build_mtp_layer(config)
        self.assertFalse((layer.hc_head_fn.numpy() == 0).all())

        # world_size=2: tracker.fork() called
        spec = _FakeMTPSpec()
        mock_pg = MagicMock(cp=None, tp=None)
        tracker_mock = MagicMock()
        tracker_mock.fork.return_value = nullcontext()
        with (
            patch(
                "paddleformers.fleet.transformer.multi_token_prediction.build_spec_layer",
                side_effect=lambda s, *a, **kw: _FakeTransformerLayer()
                if s is spec.transformer_layer
                else _FakeNorm(),
            ),
            patch(
                "paddleformers.fleet.transformer.multi_token_prediction.ProcessGroupCollection.use_mpu_process_groups",
                return_value=mock_pg,
            ),
            patch(
                "paddleformers.fleet.transformer.multi_token_prediction.paddle.distributed.get_world_size",
                return_value=2,
            ),
            patch(
                "paddleformers.fleet.transformer.multi_token_prediction.get_cuda_rng_tracker",
                return_value=tracker_mock,
            ),
        ):
            MultiTokenPredictionLayer(
                config=config,
                sublayers_spec=spec,
                layer_number=0,
                pg_collection=mock_pg,
            )
        tracker_mock.fork.assert_called()

    def test_mhc_multistream_forward(self):
        """mhc_multistream replaces hidden_states, triggers _postprocess, gets popped."""
        config = _cfg(enable_hyper_connections=True, num_residual_streams=4)
        layer = _build_mtp_layer(config)
        B, S, H, n = 2, 8, 64, 4
        captured = {}
        postprocess_called = [False]
        orig_postprocess = layer._postprocess

        def proj_override(hidden_states, decoder_input, **kw):
            captured["hs_shape"] = list(hidden_states.shape)
            return hidden_states

        def mock_postprocess(hs):
            postprocess_called[0] = True
            return orig_postprocess(hs)

        with (
            patch.object(layer, "_postprocess", side_effect=mock_postprocess),
            _mtp_forward_ctx(proj_override=proj_override, layer=layer),
        ):
            result = layer.forward(
                {
                    "hidden_states": paddle.randn([B, S, H]),
                    "mhc_multistream": paddle.randn([B, S, H * n]),
                    "mtp_input_embeds": paddle.randn([B, S + 1, H]),
                    "labels": paddle.randint(0, 100, [B, S]),
                }
            )

        self.assertEqual(captured["hs_shape"], [B, S, H * n])
        self.assertTrue(postprocess_called[0])
        self.assertNotIn("mhc_multistream", result)
        self.assertEqual(set(result.keys()), {"hidden_states", "labels"})

    def test_mhc_multistream_absent_skips_postprocess(self):
        """Without mhc_multistream, regular hidden_states used, _postprocess skipped."""
        config = _cfg(enable_hyper_connections=True, num_residual_streams=4)
        layer = _build_mtp_layer(config)
        B, S, H = 2, 8, 64
        captured = {}
        postprocess_called = [False]

        def proj_override(hidden_states, decoder_input, **kw):
            captured["hs_shape"] = list(hidden_states.shape)
            return hidden_states

        def mock_pp(hs):
            postprocess_called[0] = True
            return hs

        with (
            patch.object(layer, "_postprocess", side_effect=mock_pp),
            _mtp_forward_ctx(proj_override=proj_override, layer=layer),
        ):
            layer.forward(
                {
                    "hidden_states": paddle.randn([B, S, H]),
                    "mtp_input_embeds": paddle.randn([B, S + 1, H]),
                    "labels": paddle.randint(0, 100, [B, S]),
                }
            )

        self.assertEqual(captured["hs_shape"], [B, S, H])
        self.assertFalse(postprocess_called[0])


if __name__ == "__main__":
    unittest.main()
