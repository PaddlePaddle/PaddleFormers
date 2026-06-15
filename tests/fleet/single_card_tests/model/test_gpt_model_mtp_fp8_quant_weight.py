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

"""Unit tests for GPTModel.fp8_quant_weight with MultiTokenPredictionLayer.

Covers commits:
  7f2936f5 - add MultiTokenPredictionLayer handling in fp8_quant_weight
             (both VPP and non-VPP paths)
  91ba77d6 - import style cleanup (parenthesised multi-line import,
             trailing comma in fp8_quant_weight call)
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import functools
import random
import unittest
from unittest.mock import MagicMock

import numpy as np
import paddle
from paddle.distributed import fleet

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)
from paddleformers.fleet.transformer.transformer_layer import TransformerLayer

# ---------------------------------------------------------------------------
# Module-level one-time fleet initialisation
# (fleet.init may only be called once per process)
# ---------------------------------------------------------------------------
_strategy = None


def _ensure_fleet_init():
    global _strategy
    if _strategy is not None:
        return _strategy
    seed = 46
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    ps.initialize_model_parallel(hcg)
    _strategy = strategy
    return strategy


def _make_config(**overrides):
    seed = 46
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 512,
        "rotary_base": 10000,
        "vocab_size": 100,
        "rotary_percent": 1.0,
        "rope_scaling": 1.0,
        "position_embedding_type": "rope",
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "max_sequence_length": 64,
        "normalization": "RMSNorm",
        "hidden_dropout_prob": 0.0,
        "attention_dropout": 0.0,
        "init_method": functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        "output_layer_init_method": functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        "tie_word_embeddings": True,
        "use_qk_norm": True,
        "num_nextn_predict_layers": 1,
    }
    defaults.update(overrides)
    return GPTConfig(**defaults)


# ---------------------------------------------------------------------------
# Non-VPP path tests
# ---------------------------------------------------------------------------


class TestFp8QuantWeightNonVPP(unittest.TestCase):
    """Tests for the non-VPP path (pipeline_model_parallel_size == 1)."""

    @classmethod
    def setUpClass(cls):
        cls.strategy = _ensure_fleet_init()
        cls.config = _make_config()
        cls.gpt_model = gpt_builder(cls.config, num_stages=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_mtp_layers(self):
        return [layer for layer in self.gpt_model.run_function if isinstance(layer, MultiTokenPredictionLayer)]

    def _get_transformer_layers(self):
        return [layer for layer in self.gpt_model.run_function if isinstance(layer, TransformerLayer)]

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    def test_model_contains_mtp_layer(self):
        """Model with num_nextn_predict_layers=1 should contain MultiTokenPredictionLayer."""
        mtp_layers = self._get_mtp_layers()
        self.assertGreater(
            len(mtp_layers),
            0,
            "GPTModel should contain at least one MultiTokenPredictionLayer",
        )

    def test_mtp_layer_has_transformer_layer_attr(self):
        """Each MultiTokenPredictionLayer must expose a .transformer_layer attribute."""
        for mtp in self._get_mtp_layers():
            self.assertTrue(
                hasattr(mtp, "transformer_layer"),
                "MultiTokenPredictionLayer must have .transformer_layer",
            )
            self.assertIsInstance(mtp.transformer_layer, TransformerLayer)

    # ------------------------------------------------------------------
    # Core: fp8_quant_weight dispatches into MTP.transformer_layer
    # ------------------------------------------------------------------

    def test_fp8_quant_weight_calls_mtp_transformer_layer(self):
        """fp8_quant_weight should forward to transformer_layer on each MTP layer."""
        mtp_layers = self._get_mtp_layers()
        self.assertGreater(len(mtp_layers), 0)

        mocks = []
        for mtp in mtp_layers:
            m = MagicMock()
            mtp.transformer_layer.fp8_quant_weight = m
            mocks.append(m)

        self.gpt_model.fp8_quant_weight(batch_mode=False, quant_transpose=True)

        for m in mocks:
            m.assert_called_once_with(
                batch_mode=False,
                quant_transpose=True,
            )

    def test_fp8_quant_weight_custom_args_forwarded(self):
        """batch_mode and quant_transpose must be forwarded verbatim."""
        mtp_layers = self._get_mtp_layers()
        self.assertGreater(len(mtp_layers), 0)

        mocks = []
        for mtp in mtp_layers:
            m = MagicMock()
            mtp.transformer_layer.fp8_quant_weight = m
            mocks.append(m)

        self.gpt_model.fp8_quant_weight(batch_mode=True, quant_transpose=False)

        for m in mocks:
            m.assert_called_once_with(
                batch_mode=True,
                quant_transpose=False,
            )

    def test_fp8_quant_weight_default_args(self):
        """fp8_quant_weight() with no args should use batch_mode=False, quant_transpose=True."""
        mtp_layers = self._get_mtp_layers()
        self.assertGreater(len(mtp_layers), 0)

        mocks = []
        for mtp in mtp_layers:
            m = MagicMock()
            mtp.transformer_layer.fp8_quant_weight = m
            mocks.append(m)

        self.gpt_model.fp8_quant_weight()

        for m in mocks:
            m.assert_called_once_with(
                batch_mode=False,
                quant_transpose=True,
            )

    def test_fp8_quant_weight_called_once_per_invocation(self):
        """Each MTP transformer_layer.fp8_quant_weight is called exactly once per invocation."""
        mtp_layers = self._get_mtp_layers()
        self.assertGreater(len(mtp_layers), 0)

        mocks = []
        for mtp in mtp_layers:
            m = MagicMock()
            mtp.transformer_layer.fp8_quant_weight = m
            mocks.append(m)

        self.gpt_model.fp8_quant_weight()
        self.gpt_model.fp8_quant_weight()

        for m in mocks:
            self.assertEqual(
                m.call_count,
                2,
                "fp8_quant_weight on transformer_layer should be called once per invocation",
            )

    def test_fp8_quant_weight_does_not_define_method_on_mtp(self):
        """MultiTokenPredictionLayer itself should not define fp8_quant_weight.

        The commit routes through mtp.transformer_layer.fp8_quant_weight, so
        fp8_quant_weight must NOT exist directly on MultiTokenPredictionLayer.
        """
        for mtp in self._get_mtp_layers():
            self.assertFalse(
                "fp8_quant_weight" in type(mtp).__dict__,
                "MultiTokenPredictionLayer should not define fp8_quant_weight itself; "
                "the call must be forwarded to .transformer_layer",
            )

    def test_fp8_quant_weight_transformer_layers_also_called(self):
        """TransformerLayer.fp8_quant_weight must still be called for non-MTP layers."""
        transformer_layers = self._get_transformer_layers()
        self.assertGreater(len(transformer_layers), 0)

        trans_mocks = []
        for layer in transformer_layers:
            m = MagicMock()
            layer.fp8_quant_weight = m
            trans_mocks.append(m)

        # Patch MTP transformer_layers too so they don't error
        for mtp in self._get_mtp_layers():
            mtp.transformer_layer.fp8_quant_weight = MagicMock()

        self.gpt_model.fp8_quant_weight(batch_mode=False, quant_transpose=True)

        for m in trans_mocks:
            m.assert_called_once_with(
                batch_mode=False,
                quant_transpose=True,
            )

    def test_fp8_quant_weight_is_non_vpp_path(self):
        """Verify this test class exercises the non-VPP code path."""
        self.assertEqual(
            self.gpt_model._num_virtual_pipeline_stages,
            1,
            "TestFp8QuantWeightNonVPP expects _num_virtual_pipeline_stages == 1",
        )


# ---------------------------------------------------------------------------
# VPP path tests (simulated via patching)
# ---------------------------------------------------------------------------


class TestFp8QuantWeightVPP(unittest.TestCase):
    """Tests for the VPP path (_num_virtual_pipeline_stages > 1).

    We simulate VPP by patching _num_virtual_pipeline_stages and _model_chunks
    directly, since spinning up real multi-card VPP is not feasible in a
    single-card unit test.
    """

    @classmethod
    def setUpClass(cls):
        cls.strategy = _ensure_fleet_init()
        cls.config = _make_config()
        cls.gpt_model = gpt_builder(cls.config, num_stages=1)

    def _make_mock_mtp(self):
        mtp = MagicMock(spec=MultiTokenPredictionLayer)
        mtp.transformer_layer = MagicMock(spec=TransformerLayer)
        mtp.transformer_layer.fp8_quant_weight = MagicMock()
        return mtp

    def _make_mock_transformer(self):
        layer = MagicMock(spec=TransformerLayer)
        layer.fp8_quant_weight = MagicMock()
        return layer

    def _run_as_vpp(self, model, chunks, **kwargs):
        """Temporarily set model to appear as VPP, call fp8_quant_weight, then restore."""
        original_stages = model._num_virtual_pipeline_stages
        original_chunks = getattr(model, "_model_chunks", None)
        model._num_virtual_pipeline_stages = 2
        model._model_chunks = chunks
        try:
            model.fp8_quant_weight(**kwargs)
        finally:
            model._num_virtual_pipeline_stages = original_stages
            if original_chunks is None:
                del model._model_chunks
            else:
                model._model_chunks = original_chunks

    # ------------------------------------------------------------------

    def test_vpp_mtp_layers_are_dispatched(self):
        """VPP path: MTP layers in each chunk must have fp8_quant_weight forwarded."""
        mtp1 = self._make_mock_mtp()
        mtp2 = self._make_mock_mtp()
        trans1 = self._make_mock_transformer()
        chunks = [[trans1, mtp1], [mtp2]]

        self._run_as_vpp(self.gpt_model, chunks, batch_mode=False, quant_transpose=True)

        mtp1.transformer_layer.fp8_quant_weight.assert_called_once_with(batch_mode=False, quant_transpose=True)
        mtp2.transformer_layer.fp8_quant_weight.assert_called_once_with(batch_mode=False, quant_transpose=True)

    def test_vpp_transformer_layers_also_called(self):
        """VPP path: TransformerLayer.fp8_quant_weight must also be forwarded."""
        trans1 = self._make_mock_transformer()
        mtp1 = self._make_mock_mtp()
        chunks = [[trans1, mtp1]]

        self._run_as_vpp(self.gpt_model, chunks, batch_mode=True, quant_transpose=False)

        trans1.fp8_quant_weight.assert_called_once_with(batch_mode=True, quant_transpose=False)

    def test_vpp_custom_args_forwarded(self):
        """VPP path: batch_mode and quant_transpose are forwarded correctly."""
        mtp = self._make_mock_mtp()
        chunks = [[mtp]]

        self._run_as_vpp(self.gpt_model, chunks, batch_mode=True, quant_transpose=False)

        mtp.transformer_layer.fp8_quant_weight.assert_called_once_with(
            batch_mode=True,
            quant_transpose=False,
        )

    def test_vpp_multiple_chunks_each_mtp_called_once(self):
        """VPP path: each MTP layer across all chunks is dispatched exactly once per call."""
        mtps = [self._make_mock_mtp() for _ in range(3)]
        chunks = [[mtps[0], mtps[1]], [mtps[2]]]

        self._run_as_vpp(self.gpt_model, chunks)

        for mtp in mtps:
            mtp.transformer_layer.fp8_quant_weight.assert_called_once()

    def test_vpp_empty_chunk_does_not_crash(self):
        """VPP path: an empty virtual pipeline chunk must not raise."""
        mtp = self._make_mock_mtp()
        chunks = [[], [mtp]]

        self._run_as_vpp(self.gpt_model, chunks)

        mtp.transformer_layer.fp8_quant_weight.assert_called_once()

    def test_vpp_no_mtp_in_chunks_does_not_crash(self):
        """VPP path: chunks with only TransformerLayers must not raise."""
        trans1 = self._make_mock_transformer()
        trans2 = self._make_mock_transformer()
        chunks = [[trans1], [trans2]]

        self._run_as_vpp(self.gpt_model, chunks)

        trans1.fp8_quant_weight.assert_called_once()
        trans2.fp8_quant_weight.assert_called_once()


# ---------------------------------------------------------------------------
# Model without MTP (regression: must not break existing behaviour)
# ---------------------------------------------------------------------------


class TestFp8QuantWeightNoMTP(unittest.TestCase):
    """Verify models without MTP are unaffected by the new isinstance check."""

    @classmethod
    def setUpClass(cls):
        cls.strategy = _ensure_fleet_init()
        cls.config = _make_config(num_nextn_predict_layers=0)
        cls.gpt_model = gpt_builder(cls.config, num_stages=1)

    def test_no_mtp_layer_in_model(self):
        """Model with num_nextn_predict_layers=0 should not contain MultiTokenPredictionLayer."""
        mtp_layers = [layer for layer in self.gpt_model.run_function if isinstance(layer, MultiTokenPredictionLayer)]
        self.assertEqual(
            len(mtp_layers),
            0,
            "Model with num_nextn_predict_layers=0 should have no MTP layers",
        )

    def test_fp8_quant_weight_no_mtp_does_not_raise(self):
        """fp8_quant_weight on a model without MTP layers must not raise."""
        try:
            self.gpt_model.fp8_quant_weight(batch_mode=False, quant_transpose=True)
        except Exception as e:
            self.fail(f"fp8_quant_weight raised unexpectedly on a model without MTP: {e}")


if __name__ == "__main__":
    unittest.main()
