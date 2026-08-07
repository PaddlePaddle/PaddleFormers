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
    MagicInstance,
    mtp_magic_instance,
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
        tl.sublayers_spec.self_attn.extra_kwargs = {
            "attn_mask_type": AttnMaskType.causal
        }
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
    if layer.mtp_embed is not None:
        layer.mtp_embed = nn.Embedding(config.vocab_size, config.hidden_size)
    return layer


def _setup_magic(layer, input_ids_list):
    mtp_magic_instance.set_data({"input_ids": input_ids_list})
    mtp_magic_instance.set_magic_count(layer.magic_key, -1)


def _build_gpt_embedding(config):
    from paddleformers.fleet.models.gpt.gpt_embedding import GPTEmbedding

    mock_spec = MagicMock(rope_embedding=None)

    class _Emb(nn.Layer):
        def __init__(self, v, h):
            super().__init__()
            self.embed_tokens = nn.Embedding(v, h)
            self.reduce_scatter_embeddings = (
                self.scatter_to_sequence_parallel
            ) = self.sequence_parallel = False

        @property
        def embedding_weight(self):
            return self.embed_tokens.weight

        def forward(self, input_ids, position_ids=None):
            return self.embed_tokens(input_ids)

    emb_layer = _Emb(config.vocab_size, config.hidden_size)
    with (
        patch(
            "paddleformers.fleet.models.gpt.gpt_embedding.build_spec_layer",
            side_effect=lambda s, *a, **kw: emb_layer
            if s is mock_spec.language_embedding
            else None,
        ),
        patch(
            "paddleformers.fleet.models.gpt.gpt_embedding.mark_context_parallel_parameter_disable_scale_grad"
        ),
    ):
        return GPTEmbedding(
            sublayers_spec=mock_spec,
            config=config,
            vocab_size=config.vocab_size,
            max_sequence_length=128,
            position_embedding_type="rope",
        )


def _fwd_ctx(
    cp_world_size=1,
    scatter_fn=None,
    cp_scatter_fn=None,
    proj_override=None,
    layer=None,
):
    """Context manager that sets up all MTP forward mocks."""
    stack = ExitStack()

    def _enter(s):
        s.enter_context(
            patch(
                "paddleformers.fleet.transformer.multi_token_prediction.get_context_parallel_world_size",
                return_value=cp_world_size,
            )
        )
        tp = s.enter_context(
            patch(
                "paddleformers.fleet.transformer.multi_token_prediction.tensor_parallel"
            )
        )
        tp.get_cuda_rng_tracker.return_value.fork.return_value = nullcontext()
        if scatter_fn is not None:
            so = s.enter_context(
                patch(
                    "paddleformers.fleet.transformer.multi_token_prediction.ScatterOp"
                )
            )
            so.apply = scatter_fn
        if cp_scatter_fn is not None:
            co = s.enter_context(
                patch(
                    "paddleformers.fleet.transformer.multi_token_prediction.ContextParallelScatterOp"
                )
            )
            co.apply = cp_scatter_fn
        if proj_override is not None and layer is not None:
            s.enter_context(
                patch.object(
                    layer,
                    "_proj_and_transformer_layer",
                    side_effect=proj_override,
                )
            )

    class _Ctx:
        def __enter__(self_):
            _enter(stack)
            return stack

        def __exit__(self_, *a):
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
                variable_seq_lengths=False,
            )
        cfg = TransformerConfig(
            enable_mtp_magic_send=True,
            num_nextn_predict_layers=3,
            pipeline_model_parallel_size=2,
            variable_seq_lengths=True,
        )
        self.assertEqual(cfg.num_nextn_predict_layers, 3)
        with self.assertRaises(AssertionError):
            TransformerConfig(
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=1,
            )

    def test_magic_send_vpp(self):
        self.assertTrue(
            TransformerConfig(
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=2,
                overlap_p2p_comm=True,
                variable_seq_lengths=True,
            ).enable_mtp_magic_send
        )
        with self.assertRaises(AssertionError):
            TransformerConfig(
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=2,
                overlap_p2p_comm=False,
                variable_seq_lengths=True,
            )
        with self.assertRaises(AssertionError):
            TransformerConfig(
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=2,
                overlap_p2p_comm=True,
                variable_seq_lengths=False,
            )

    def test_magic_send_rejects_shared_last_layer(self):
        with self.assertRaises(AssertionError):
            TransformerConfig(
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
                mtp_shared_last_layer=True,
            )


class TestMagicInstance(unittest.TestCase):
    def test_lifecycle(self):
        inst = MagicInstance()
        ids_list = [paddle.randint(0, 100, [2, 10]) for _ in range(3)]
        inst.set_data({"input_ids": ids_list})
        self.assertEqual(len(inst.get("input_ids")), 3)
        with self.assertRaises(AssertionError):
            inst.get("nonexistent")

        inst.set_magic_count("k", -1)
        self.assertEqual(inst.get_magic_count("k"), -1)
        with self.assertRaises(AssertionError):
            inst.get_magic_count("missing")

        inst.set_magic_count("k", 5)
        inst.set_magic_count("k2", 3)
        inst.clear_count_dict()
        self.assertEqual(inst.get_magic_count("k"), -1)
        self.assertEqual(inst.get_magic_count("k2"), -1)

    def test_global_singleton(self):
        mtp_magic_instance.set_magic_count("test_key", 42)
        self.assertEqual(mtp_magic_instance.get_magic_count("test_key"), 42)
        mtp_magic_instance.magic_cnt_dict.pop("test_key", None)


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
        r = WrappedPaddleNormPipe(cfg_on, hidden_size=64).forward(
            {"hidden_states": paddle.ones([1, 4, 64]) * 2.0}
        )
        self.assertTrue(
            paddle.allclose(
                r["hidden_states"], paddle.ones([1, 4, 64]), atol=1e-5
            ).item()
        )

        cfg_off = TransformerConfig(
            enable_mtp_magic_send=False,
            num_nextn_predict_layers=1,
            hidden_size=64,
            normalization="RMSNorm",
            tensor_model_parallel_size=1,
        )
        r = WrappedPaddleNormPipe(cfg_off, hidden_size=64).forward(
            {"hidden_states": paddle.randn([4, 8, 64])}
        )
        self.assertEqual(r["hidden_states"].shape, [4, 8, 64])


class TestMTPLayerForward(unittest.TestCase):
    """MultiTokenPredictionLayer.forward() magic send branch."""

    def test_basic_forward(self):
        """Shape, keys, stop_gradient, and rotary passthrough."""
        layer = _build_mtp_layer(_cfg())
        B, S, H = 2, 8, 64
        input_ids = paddle.randint(0, 512, [B, S + 1])
        _setup_magic(layer, [input_ids])
        with _fwd_ctx():
            result = layer.forward(
                {
                    "hidden_states": paddle.randn([B, S, H]),
                    "mtp_startend_row_indices_all": paddle.randn([B, 1, S, 4]),
                    "mtp_hidden_inputs_mask_all": paddle.ones([B, 1, S]),
                    "rotary_pos_emb": paddle.randn([1, S + 2, 32]),
                    "rotary_pos_cos": paddle.randn([1, S + 2, 32]),
                    "rotary_pos_sin": paddle.randn([1, S + 2, 32]),
                    "labels": paddle.randint(0, 100, [B, S]),
                }
            )
        self.assertEqual(result["hidden_states"].shape, [2 * B, S, H])
        expected_keys = {
            "hidden_states",
            "labels",
            "input_ids",
            "rotary_pos_emb",
            "rotary_pos_cos",
            "rotary_pos_sin",
            "mtp_startend_row_indices_all",
            "mtp_hidden_inputs_mask_all",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        for key in (
            "mtp_startend_row_indices_all",
            "mtp_hidden_inputs_mask_all",
            "rotary_pos_emb",
            "labels",
        ):
            self.assertTrue(result[key].stop_gradient)

    def test_exp_versions(self):
        B, S, H = 2, 8, 64
        for exp_ver in [True, False]:
            layer = _build_mtp_layer(
                _cfg(gpt_model_use_experimental_version=exp_ver)
            )
            _setup_magic(layer, [paddle.randint(0, 512, [B, S + 1])])
            with _fwd_ctx():
                result = layer.forward(
                    {
                        "hidden_states": paddle.randn([B, S, H]),
                        "mtp_startend_row_indices_all": paddle.randn(
                            [B, 1, S, 4]
                        ),
                        "mtp_hidden_inputs_mask_all": paddle.ones([B, 1, S]),
                        "labels": paddle.randint(0, 100, [B, S]),
                    }
                )
            self.assertIn("hidden_states", result)

    def test_missing_input_ids_raises(self):
        layer = _build_mtp_layer(_cfg())
        mtp_magic_instance.magic_send_dict.clear()
        with self.assertRaises(AssertionError), _fwd_ctx():
            layer.forward(
                {
                    "hidden_states": paddle.randn([2, 8, 64]),
                    "labels": paddle.randint(0, 100, [2, 8]),
                }
            )

    def test_recompute(self):
        layer = _build_mtp_layer(
            _cfg(
                normalization="RMSNorm",
                recompute_granularity="full",
                recompute_method="block",
                recompute_num_layers=1,
            )
        )
        layer.training = True
        B, S, H = 2, 8, 64
        _setup_magic(layer, [paddle.randint(0, 512, [B, S + 1])])
        with (
            patch.object(
                layer,
                "_checkpointed_forward",
                side_effect=lambda fn, **kw: fn(**kw),
            ),
            _fwd_ctx(),
        ):
            result = layer.forward(
                {
                    "hidden_states": paddle.randn([B, S, H]),
                    "labels": paddle.randint(0, 100, [B, S]),
                }
            )
        self.assertIn("hidden_states", result)

    def test_sp_scatter(self):
        layer = _build_mtp_layer(
            _cfg(sequence_parallel=True, tensor_model_parallel_size=2)
        )
        S_local, B, H, S_global = 4, 2, 64, 8
        _setup_magic(layer, [paddle.randint(0, 512, [B, S_global + 1])])
        captured = {}

        def proj(hidden_states=None, decoder_input=None, **kw):
            captured["di"] = decoder_input
            return hidden_states

        cnt = [0]

        def scatter_fn(x):
            cnt[0] += 1
            return x[: x.shape[0] // 2]

        with _fwd_ctx(scatter_fn=scatter_fn, proj_override=proj, layer=layer):
            layer.forward(
                {
                    "hidden_states": paddle.randn([S_local, B, H]),
                    "rotary_pos_emb": paddle.randn([S_global + 4, 32]),
                    "labels": paddle.randint(0, 100, [B, S_global]),
                }
            )
        self.assertEqual(
            cnt[0], 1
        )  # decoder_input only (mtp_input_ids_local is NOT scattered here; router does it)
        self.assertEqual(list(captured["di"].shape), [S_local, B, H])

    def test_cp_scatter(self):
        layer = _build_mtp_layer(_cfg(experimental_dataflow=True))
        B, S_global, H, CP = 2, 8, 64, 2
        S_local = S_global // CP
        _setup_magic(layer, [paddle.randint(0, 512, [B, S_global + 1])])
        captured = {}

        def proj(hidden_states=None, decoder_input=None, **kw):
            captured["di"] = decoder_input
            return hidden_states

        def cp_fn(x, axis=0, **kw):
            return (
                x[:, : x.shape[1] // 2, :]
                if axis == 1
                else x[: x.shape[0] // 2]
            )

        with _fwd_ctx(
            cp_world_size=CP,
            cp_scatter_fn=cp_fn,
            proj_override=proj,
            layer=layer,
        ):
            layer.forward(
                {
                    "hidden_states": paddle.randn([B, S_local, H]),
                    "labels": paddle.randint(0, 100, [B, S_global]),
                }
            )
        self.assertEqual(list(captured["di"].shape), [B, S_local, H])

    def test_multi_layer_cumulative_concat(self):
        config = _cfg(num_nextn_predict_layers=2, variable_seq_lengths=True)
        B, S, H = 2, 8, 64
        ids = paddle.randint(0, 512, [B, S + 2])
        layer0 = _build_mtp_layer(config, layer_number=0)
        _setup_magic(layer0, [ids])
        with _fwd_ctx():
            r0 = layer0.forward(
                {
                    "hidden_states": paddle.randn([B, S, H]),
                    "labels": paddle.randint(0, 100, [B, S]),
                }
            )
        self.assertEqual(r0["hidden_states"].shape, [2 * B, S, H])
        layer1 = _build_mtp_layer(config, layer_number=1)
        _setup_magic(layer1, [ids])
        with _fwd_ctx():
            r1 = layer1.forward(r0)
        self.assertEqual(r1["hidden_states"].shape, [3 * B, S, H])

    def test_ep_fill_feature(self):
        layer = _build_mtp_layer(_cfg(expert_model_parallel_size=4))
        B, S, H = 1, 4, 64
        ids = paddle.randint(1, 512, [B, S + 1])
        ids[0, 0] = 0
        _setup_magic(layer, [ids])
        with _fwd_ctx():
            self.assertIn(
                "hidden_states",
                layer.forward(
                    {
                        "hidden_states": paddle.randn([B, S, H]),
                        "labels": paddle.randint(0, 100, [B, S]),
                    }
                ),
            )

    def test_multiple_microbatches(self):
        layer = _build_mtp_layer(_cfg())
        B, S, H = 2, 8, 64
        _setup_magic(
            layer, [paddle.randint(0, 512, [B, S + 1]) for _ in range(3)]
        )
        with _fwd_ctx():
            for _ in range(3):
                layer.forward(
                    {
                        "hidden_states": paddle.randn([B, S, H]),
                        "labels": paddle.randint(0, 100, [B, S]),
                    }
                )
        self.assertEqual(mtp_magic_instance.get_magic_count(layer.magic_key), 2)


class TestGPTEmbeddingForward(unittest.TestCase):
    def _run_emb(
        self, config, cp_world_size=1, mock_scatter=False, mock_cp=False
    ):
        emb = _build_gpt_embedding(config)
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "paddleformers.fleet.models.gpt.gpt_embedding.get_context_parallel_world_size",
                    return_value=cp_world_size,
                )
            )
            if mock_scatter:
                sc = stack.enter_context(
                    patch("paddleformers.fleet.models.gpt.gpt_embedding.ScatterOp")
                )
                sc.apply = lambda x: x
            if mock_cp:
                cp = stack.enter_context(
                    patch(
                        "paddleformers.fleet.models.gpt.gpt_embedding.ContextParallelScatterOp"
                    )
                )
                cp.apply = lambda x, axis=0, **kwargs: x
            return emb.forward({"input_ids": paddle.randint(0, 512, [2, 10])})

    def test_magic_send_paths(self):
        B, S, H = 2, 10, 64
        self.assertEqual(
            self._run_emb(_cfg())["hidden_states"].shape, [B, S - 1, H]
        )
        self.assertEqual(
            self._run_emb(
                _cfg(experimental_dataflow=True), cp_world_size=2, mock_cp=True
            )["hidden_states"].shape,
            [B, S - 1, H],
        )
        self.assertEqual(
            self._run_emb(
                _cfg(sequence_parallel=True, tensor_model_parallel_size=2),
                mock_scatter=True,
            )["hidden_states"].shape,
            [S - 1, B, H],
        )

    def test_non_magic_send_paths(self):
        B, S, H = 2, 10, 64
        self.assertEqual(
            self._run_emb(_cfg(enable_mtp_magic_send=False))[
                "hidden_states"
            ].shape,
            [B * 2, S - 1, H],
        )
        self.assertEqual(
            self._run_emb(
                _cfg(
                    enable_mtp_magic_send=False,
                    sequence_parallel=True,
                    tensor_model_parallel_size=2,
                ),
                mock_scatter=True,
            )["hidden_states"].shape,
            [2 * (S - 1), B, H],
        )

    def test_ep_moe_mask(self):
        self.assertEqual(
            self._run_emb(_cfg(expert_model_parallel_size=4))[
                "mtp_input_ids_for_moe_mask"
            ].shape,
            [2, 1, 9],
        )


class TestPPCommUtils(unittest.TestCase):
    def test_split_group(self):
        self.assertEqual(
            len(
                list(split_group([(0, _DtypeSndShape("float32", [10]))], 2**18))
            ),
            1,
        )
        self.assertGreater(
            len(
                list(
                    split_group(
                        [
                            (i, _DtypeSndShape("float32", [200000]))
                            for i in range(5)
                        ],
                        2**18,
                    )
                )
            ),
            1,
        )
        self.assertEqual(list(split_group([], 2**18)), [])

    def test_broadcast_data_obj(self):
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
            r = broadcast_data_obj(
                paddle.zeros([3, 4]), src_rank=0, group=MagicMock()
            )
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
        self.assertIsNone(_run([0, 1, 2, 3], 1)[0])
        self.assertEqual(_run([0, 1], 0)[1][0], [0, 1])


class TestHyperConnection(unittest.TestCase):
    def test_contract_layer_magic_send(self):
        from paddleformers.fleet.transformer.hyper_connection import (
            HyperConnectionContractLayer,
            HyperConnectionModule,
        )

        B, S, H, n = 2, 8, 64, 4
        for num_mtp in [1, 2]:
            layer = HyperConnectionContractLayer(
                TransformerConfig(
                    hidden_size=64,
                    num_residual_streams=4,
                    enable_mtp_magic_send=True,
                    num_nextn_predict_layers=num_mtp,
                    pipeline_model_parallel_size=2,
                    tensor_model_parallel_size=1,
                    variable_seq_lengths=True if num_mtp > 1 else False,
                )
            )
            result = layer.forward(
                {"hidden_states": paddle.randn([B, S, H * n])}
            )
            self.assertEqual(result["hidden_states"].shape, [B, S, H])
            self.assertEqual(
                result["mhc_multistream"].shape, [(num_mtp + 1) * B, S, H * n]
            )

        # Verify learned_output_contract match
        layer = HyperConnectionContractLayer(
            TransformerConfig(
                hidden_size=64,
                num_residual_streams=4,
                enable_mtp_magic_send=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
                tensor_model_parallel_size=1,
            )
        )
        x = paddle.randn([1, 4, 64 * 4])
        expected = HyperConnectionModule.learned_output_contract(
            x,
            layer.hc_head_fn,
            layer.hc_head_base,
            layer.hc_head_scale,
            4,
            layer.config.rms_norm_eps,
        )
        self.assertTrue(
            paddle.allclose(
                layer.forward({"hidden_states": x.clone()})["hidden_states"],
                expected,
                atol=1e-5,
            ).item()
        )

    def test_contract_layer_non_magic_send(self):
        from paddleformers.fleet.transformer.hyper_connection import (
            HyperConnectionContractLayer,
        )

        layer = HyperConnectionContractLayer(
            TransformerConfig(
                hidden_size=64,
                num_residual_streams=4,
                enable_mtp_magic_send=False,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=1,
                tensor_model_parallel_size=1,
            )
        )
        result = layer.forward({"hidden_states": paddle.randn([4, 8, 64 * 4])})
        self.assertEqual(result["hidden_states"].shape, [4, 8, 64])

    def test_module_init_weights_rng(self):
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
        tracker = MagicMock()
        tracker.fork.return_value = nullcontext()
        with (
            patch(
                "paddleformers.fleet.transformer.hyper_connection.paddle.distributed.get_world_size",
                return_value=1,
            ),
            patch(
                "paddleformers.fleet.transformer.hyper_connection.get_cuda_rng_tracker",
                return_value=tracker,
            ),
        ):
            HyperConnectionModule(config, layer_number=0)
        tracker.fork.assert_not_called()
        tracker.reset_mock()
        with (
            patch(
                "paddleformers.fleet.transformer.hyper_connection.paddle.distributed.get_world_size",
                return_value=2,
            ),
            patch(
                "paddleformers.fleet.transformer.hyper_connection.get_cuda_rng_tracker",
                return_value=tracker,
            ),
        ):
            HyperConnectionModule(config, layer_number=0)
        tracker.fork.assert_called_once()


class TestMTPLayerMHC(unittest.TestCase):
    def test_hc_head_fn_init(self):
        from paddleformers.fleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )

        config = _cfg(enable_hyper_connections=True, num_residual_streams=4)
        self.assertFalse(
            (_build_mtp_layer(config).hc_head_fn.numpy() == 0).all()
        )

        spec = _FakeMTPSpec()
        mock_pg = MagicMock(cp=None, tp=None)
        tracker = MagicMock()
        tracker.fork.return_value = nullcontext()
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
                return_value=tracker,
            ),
        ):
            MultiTokenPredictionLayer(
                config=config,
                sublayers_spec=spec,
                layer_number=0,
                pg_collection=mock_pg,
            )
        tracker.fork.assert_called()

    def test_mhc_forward_and_passthrough(self):
        config = _cfg(
            enable_hyper_connections=True,
            num_residual_streams=4,
            num_nextn_predict_layers=2,
            variable_seq_lengths=True,
        )
        layer = _build_mtp_layer(config)
        B, S, H, n = 2, 8, 64, 4
        _setup_magic(layer, [paddle.randint(0, 512, [B, S + 2])])
        captured = {}
        pp_called = [False]
        orig_pp = layer._postprocess

        def proj(hidden_states=None, decoder_input=None, **kw):
            captured["hs_shape"] = list(hidden_states.shape)
            return hidden_states

        def mock_pp(hs):
            pp_called[0] = True
            return orig_pp(hs)

        with (
            patch.object(layer, "_postprocess", side_effect=mock_pp),
            _fwd_ctx(proj_override=proj, layer=layer),
        ):
            result = layer.forward(
                {
                    "hidden_states": paddle.randn([B, S, H]),
                    "mhc_multistream": paddle.randn([3 * B, S, H * n]),
                    "labels": paddle.randint(0, 100, [B, S]),
                }
            )
        self.assertEqual(captured["hs_shape"], [B, S, H * n])
        self.assertTrue(pp_called[0])
        self.assertIn("mhc_multistream", result)

    def test_mhc_absent_skips_postprocess(self):
        layer = _build_mtp_layer(
            _cfg(enable_hyper_connections=True, num_residual_streams=4)
        )
        B, S, H = 2, 8, 64
        _setup_magic(layer, [paddle.randint(0, 512, [B, S + 1])])
        pp_called = [False]

        def mock_pp(hs):
            pp_called[0] = True
            return hs

        def proj(hidden_states=None, decoder_input=None, **kw):
            return hidden_states

        with (
            patch.object(layer, "_postprocess", side_effect=mock_pp),
            _fwd_ctx(proj_override=proj, layer=layer),
        ):
            layer.forward(
                {
                    "hidden_states": paddle.randn([B, S, H]),
                    "labels": paddle.randint(0, 100, [B, S]),
                }
            )
        self.assertFalse(pp_called[0])

    def test_mhc_last_layer_no_passthrough(self):
        layer = _build_mtp_layer(
            _cfg(
                enable_hyper_connections=True,
                num_residual_streams=4,
                num_nextn_predict_layers=1,
            ),
            layer_number=0,
        )
        B, S, H, n = 2, 8, 64, 4
        _setup_magic(layer, [paddle.randint(0, 512, [B, S + 1])])

        def proj(hidden_states=None, decoder_input=None, **kw):
            return hidden_states

        with _fwd_ctx(proj_override=proj, layer=layer):
            result = layer.forward(
                {
                    "hidden_states": paddle.randn([B, S, H]),
                    "mhc_multistream": paddle.randn([2 * B, S, H * n]),
                    "labels": paddle.randint(0, 100, [B, S]),
                }
            )
        self.assertNotIn("mhc_multistream", result)


class TestGPTModelMTPMethods(unittest.TestCase):
    """Tests for GPTModel MTP magic send methods."""

    def _make_model(self, num_mtp=2, has_heads=False):
        from paddleformers.fleet.models.gpt.gpt_model import GPTModel
        from paddleformers.fleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )

        class FakeEmbed:
            def __init__(self):
                self._parameters = {
                    "weight": paddle.create_parameter(
                        shape=[512, 64], dtype="float32"
                    )
                }

            @property
            def weight(self):
                return self._parameters["weight"]

        class FakeMTPLayer(MultiTokenPredictionLayer):
            def __new__(cls, ln):
                return object.__new__(cls)

            def __init__(self, ln):
                self.layer_number = ln
                self.mtp_embed = FakeEmbed()

        mtp_layers = [FakeMTPLayer(i) for i in range(num_mtp)]
        model = MagicMock(spec=GPTModel)
        model.config = MagicMock()
        model.config.enable_mtp_magic_send = True
        model.config.num_nextn_predict_layers = num_mtp
        model.config.vocab_size = 512
        model.config.hidden_size = 64
        model.config.tensor_model_parallel_size = 1
        model.config.make_vocab_size_divisible_by = 1
        model.config.params_dtype = "float32"
        model._num_virtual_pipeline_stages = 1
        model.shared_layers = {}
        model.shared_comm = {}

        layers = list(mtp_layers)
        if has_heads:
            from paddleformers.fleet.models.gpt.lm_head import (
                GPTMainLMHead,
                GPTMTPLMHead,
            )

            main_head = MagicMock(spec=GPTMainLMHead)
            main_head.weight = paddle.create_parameter(
                shape=[512, 64], dtype="float32"
            )
            mtp_head = MagicMock(spec=GPTMTPLMHead)
            mtp_head.weight = paddle.create_parameter(
                shape=[512, 64], dtype="float32"
            )
            mtp_head._parameters = {"weight": mtp_head.weight}
            layers += [main_head, mtp_head]
        model.run_function = layers

        for name in (
            "_get_all_mtp_layers",
            "_get_mtp_embed_primary_weight",
            "_tie_mtp_embed_weights_intra_rank",
            "_create_mtp_embed_global_group",
            "_synchronize_mtp_embed_weight",
            "_mark_mtp_embed_shared_flags",
            "_assert_mtp_depth_contiguous",
            "_tie_mtp_lm_head_weight",
            "allreduce_shared_weight_gradients",
        ):
            setattr(model, name, getattr(GPTModel, name).__get__(model))
        return model, mtp_layers

    def test_get_all_mtp_layers(self):
        model, layers = self._make_model(num_mtp=3)
        self.assertEqual(len(model._get_all_mtp_layers()), 3)
        # VPP path
        model._num_virtual_pipeline_stages = 2
        c1, c2 = MagicMock(), MagicMock()
        c1.run_function = [layers[0]]
        c2.run_function = [layers[1]]
        model._model_chunks = [c1, c2]
        self.assertEqual(len(model._get_all_mtp_layers()), 2)

    def test_primary_weight_and_tie(self):
        model, layers = self._make_model(num_mtp=3)
        self.assertIs(
            model._get_mtp_embed_primary_weight(), layers[0].mtp_embed.weight
        )
        model._tie_mtp_embed_weights_intra_rank()
        for l in layers:
            self.assertIs(l.mtp_embed.weight, layers[0].mtp_embed.weight)

        # Shared layers override
        shared = MagicMock()
        shared.embedding_weight = paddle.create_parameter(
            shape=[512, 64], dtype="float32"
        )
        model.shared_layers = {"mtp_embed": shared}
        self.assertIs(
            model._get_mtp_embed_primary_weight(), shared.embedding_weight
        )
        model._tie_mtp_embed_weights_intra_rank()
        for l in layers:
            self.assertIs(l.mtp_embed.weight, shared.embedding_weight)

    def test_create_global_group(self):
        model, _ = self._make_model(num_mtp=2)
        hcg = MagicMock()
        hcg.get_pipe_parallel_group.return_value = MagicMock(ranks=[0, 1, 2, 3])

        def mock_gather(out, obj, group=None):
            if isinstance(obj, bool):
                out.extend([True, False, True, True])
            else:
                out.extend([[0, 2, 3]] * 4)

        with (
            patch(
                "paddleformers.fleet.models.gpt.gpt_model.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            patch("paddle.distributed.get_rank", return_value=0),
            patch("paddle.distributed.get_world_size", return_value=4),
            patch(
                "paddle.distributed.all_gather_object", side_effect=mock_gather
            ),
            patch("paddle.distributed.new_group", return_value=MagicMock()),
        ):
            model._create_mtp_embed_global_group()
        self.assertIsNotNone(model._mtp_embed_global_group)

    def test_synchronize_weight(self):
        model, _ = self._make_model(num_mtp=1)
        hcg = MagicMock()
        hcg.get_rank_from_stage.return_value = 0
        hcg.get_pipe_parallel_group.return_value = MagicMock()
        with (
            patch(
                "paddleformers.fleet.models.gpt.gpt_model.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            patch("paddle.distributed.broadcast") as bcast,
        ):
            model._synchronize_mtp_embed_weight()
        bcast.assert_called_once()

        # Without weight + string dtype
        model2, _ = self._make_model(num_mtp=0)
        model2.run_function = []
        model2.config.params_dtype = "float16"
        with (
            patch(
                "paddleformers.fleet.models.gpt.gpt_model.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            patch("paddle.distributed.broadcast") as bcast2,
        ):
            model2._synchronize_mtp_embed_weight()
        bcast2.assert_called_once()

    def test_mark_shared_flags(self):
        model, layers = self._make_model(num_mtp=1)
        model._mark_mtp_embed_shared_flags()
        self.assertFalse(layers[0].mtp_embed.weight.is_firstly_shared)
        shared = MagicMock()
        shared.embedding_weight = layers[0].mtp_embed.weight
        model.shared_layers = {"mtp_embed": shared}
        model._mark_mtp_embed_shared_flags()
        self.assertTrue(layers[0].mtp_embed.weight.is_firstly_shared)

    def test_assert_depth_contiguous(self):
        model, _ = self._make_model(num_mtp=2)
        hcg = MagicMock()
        hcg.get_pipe_parallel_group.return_value = MagicMock()
        with (
            patch(
                "paddleformers.fleet.models.gpt.gpt_model.fleet.get_hybrid_communicate_group",
                return_value=hcg,
            ),
            patch(
                "paddle.distributed.all_gather_object",
                side_effect=lambda out, obj, group=None: out.extend([[0, 1]]),
            ),
        ):
            model._assert_mtp_depth_contiguous()

    def test_tie_lm_head_weight(self):
        model, _ = self._make_model(num_mtp=1, has_heads=True)
        model._tie_mtp_lm_head_weight()
        # VPP path
        model._num_virtual_pipeline_stages = 2
        chunk = MagicMock()
        chunk.run_function = model.run_function
        model._model_chunks = [chunk]
        model._tie_mtp_lm_head_weight()

    def test_allreduce(self):
        model, layers = self._make_model(num_mtp=1)
        model._mtp_embed_global_group = MagicMock()
        w = layers[0].mtp_embed.weight
        w.grad = paddle.ones_like(w)
        other = paddle.create_parameter(shape=[64], dtype="float32")
        other.grad = paddle.ones([64])
        ol = MagicMock()
        ol.p = other
        model.shared_comm = {
            "x": {"weight_attr": ["p"], "layer": ol, "group": MagicMock()}
        }
        with patch("paddle.distributed.all_reduce") as ar:
            model.allreduce_shared_weight_gradients()
        self.assertEqual(ar.call_count, 2)

        # No grad => no call
        w.grad = None
        model.shared_comm = {}
        with patch("paddle.distributed.all_reduce") as ar2:
            model.allreduce_shared_weight_gradients()
        ar2.assert_not_called()

    def test_allreduce_main_grad(self):
        model, layers = self._make_model(num_mtp=1)
        model._mtp_embed_global_group = MagicMock()
        layers[0].mtp_embed.weight.main_grad = paddle.ones_like(
            layers[0].mtp_embed.weight
        )
        model.shared_comm = {}
        with patch("paddle.distributed.all_reduce") as ar:
            model.allreduce_shared_weight_gradients()
        ar.assert_called_once()

    def test_edge_cases_no_layers(self):
        model, _ = self._make_model(num_mtp=0)
        model.run_function = []
        self.assertIsNone(model._get_mtp_embed_primary_weight())
        model._tie_mtp_embed_weights_intra_rank()
        model._mark_mtp_embed_shared_flags()


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
        self.assertFalse(
            should_split(_cfg(enable_mtp_magic_send=False), is_mtp=True)
        )


if __name__ == "__main__":
    unittest.main()
