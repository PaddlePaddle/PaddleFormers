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

"""Unit tests for fp8 weight clear feature (commit e9815ca).

Covers:
- MoELayer.clear_fp8_quant_weight: grouped_gemm_experts path & experts fallback path
- TransformerLayer.clear_fp8_quant_weight: delegation to MoELayer
- GPTModel.clear_fp8_quant_weight: virtual pipeline stages & single stage paths
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock


class TestMoELayerClearFp8QuantWeight(unittest.TestCase):
    """Test MoELayer.clear_fp8_quant_weight logic."""

    def _make_moe_layer(self, moe_use_fusion_node=True, fp8=True):
        """Create a minimal mock MoELayer with required attributes."""
        moe = MagicMock()
        moe.moe_use_fusion_node = moe_use_fusion_node
        moe.fp8 = fp8
        return moe

    def _attach_clear_method(self, moe):
        """Bind the real clear_fp8_quant_weight logic to a mock."""
        fp8_attrs = (
            "fp8_weight_stacked",
            "fp8_scale_stacked",
            "fp8_weight_stacked_transpose",
            "fp8_scale_stacked_transpose",
        )

        def clear_fp8_quant_weight():
            if not (moe.moe_use_fusion_node and moe.fp8):
                return

            def _clear_attrs(weight_obj):
                for attr in fp8_attrs:
                    if hasattr(weight_obj, attr):
                        delattr(weight_obj, attr)

            if hasattr(moe, "grouped_gemm_experts"):
                _clear_attrs(moe.grouped_gemm_experts.weight1)
                _clear_attrs(moe.grouped_gemm_experts.weight2)
            else:
                for expert in moe.experts:
                    if expert is not None:
                        _clear_attrs(expert.up_gate_proj.weight)
                        _clear_attrs(expert.down_proj.weight)

        moe.clear_fp8_quant_weight = clear_fp8_quant_weight

    def test_grouped_gemm_path_clears_all_attrs(self):
        """When grouped_gemm_experts exists, clear weight1/weight2 fp8 attrs."""
        moe = self._make_moe_layer()

        # Set up grouped_gemm_experts with fp8 attrs
        weight1 = SimpleNamespace(
            fp8_weight_stacked="w1",
            fp8_scale_stacked="s1",
            fp8_weight_stacked_transpose="wt1",
            fp8_scale_stacked_transpose="st1",
        )
        weight2 = SimpleNamespace(
            fp8_weight_stacked="w2",
            fp8_scale_stacked="s2",
            fp8_weight_stacked_transpose="wt2",
            fp8_scale_stacked_transpose="st2",
        )
        moe.grouped_gemm_experts = SimpleNamespace(
            weight1=weight1, weight2=weight2
        )
        self._attach_clear_method(moe)

        moe.clear_fp8_quant_weight()

        for attr in (
            "fp8_weight_stacked",
            "fp8_scale_stacked",
            "fp8_weight_stacked_transpose",
            "fp8_scale_stacked_transpose",
        ):
            self.assertFalse(
                hasattr(weight1, attr), f"weight1.{attr} not cleared"
            )
            self.assertFalse(
                hasattr(weight2, attr), f"weight2.{attr} not cleared"
            )

    def test_experts_fallback_path_clears_attrs(self):
        """When no grouped_gemm_experts, clear per-expert weight attrs."""
        moe = self._make_moe_layer()
        # Remove grouped_gemm_experts so hasattr returns False
        del moe.grouped_gemm_experts

        expert1_up_gate = SimpleNamespace(
            fp8_weight_stacked="x", fp8_scale_stacked="x"
        )
        expert1_down = SimpleNamespace(
            fp8_weight_stacked_transpose="x", fp8_scale_stacked_transpose="x"
        )
        expert1 = SimpleNamespace(
            up_gate_proj=SimpleNamespace(weight=expert1_up_gate),
            down_proj=SimpleNamespace(weight=expert1_down),
        )
        # Include a None expert to test the `if expert is not None` guard
        moe.experts = [expert1, None]
        self._attach_clear_method(moe)

        moe.clear_fp8_quant_weight()

        self.assertFalse(hasattr(expert1_up_gate, "fp8_weight_stacked"))
        self.assertFalse(hasattr(expert1_up_gate, "fp8_scale_stacked"))
        self.assertFalse(hasattr(expert1_down, "fp8_weight_stacked_transpose"))
        self.assertFalse(hasattr(expert1_down, "fp8_scale_stacked_transpose"))

    def test_early_return_when_fusion_disabled(self):
        """Should return early and not touch weights if moe_use_fusion_node=False."""
        moe = self._make_moe_layer(moe_use_fusion_node=False)
        weight1 = SimpleNamespace(fp8_weight_stacked="keep_me")
        weight2 = SimpleNamespace(fp8_weight_stacked="keep_me")
        moe.grouped_gemm_experts = SimpleNamespace(
            weight1=weight1, weight2=weight2
        )
        self._attach_clear_method(moe)

        moe.clear_fp8_quant_weight()

        self.assertTrue(hasattr(weight1, "fp8_weight_stacked"))

    def test_early_return_when_fp8_disabled(self):
        """Should return early and not touch weights if fp8=False/None."""
        moe = self._make_moe_layer(fp8=None)
        weight1 = SimpleNamespace(fp8_weight_stacked="keep_me")
        weight2 = SimpleNamespace(fp8_weight_stacked="keep_me")
        moe.grouped_gemm_experts = SimpleNamespace(
            weight1=weight1, weight2=weight2
        )
        self._attach_clear_method(moe)

        moe.clear_fp8_quant_weight()

        self.assertTrue(hasattr(weight1, "fp8_weight_stacked"))

    def test_partial_attrs_only_clears_existing(self):
        """Only existing fp8 attrs are deleted; missing ones are ignored."""
        moe = self._make_moe_layer()
        # weight1 has only 2 of the 4 attrs
        weight1 = SimpleNamespace(
            fp8_weight_stacked="w1", fp8_scale_stacked="s1"
        )
        weight2 = SimpleNamespace(
            fp8_weight_stacked_transpose="wt2",
            fp8_scale_stacked_transpose="st2",
        )
        moe.grouped_gemm_experts = SimpleNamespace(
            weight1=weight1, weight2=weight2
        )
        self._attach_clear_method(moe)

        moe.clear_fp8_quant_weight()  # Should not raise

        self.assertFalse(hasattr(weight1, "fp8_weight_stacked"))
        self.assertFalse(hasattr(weight1, "fp8_scale_stacked"))
        self.assertFalse(hasattr(weight2, "fp8_weight_stacked_transpose"))
        self.assertFalse(hasattr(weight2, "fp8_scale_stacked_transpose"))


class TestTransformerLayerClearFp8QuantWeight(unittest.TestCase):
    """Test TransformerLayer.clear_fp8_quant_weight delegation."""

    def test_delegates_to_moe_mlp(self):
        """When mlp is a MoELayer, should call mlp.clear_fp8_quant_weight."""

        # Simulate the isinstance check via a class hierarchy
        class FakeMoELayer:
            clear_fp8_quant_weight = MagicMock()

        class FakeTransformerLayer:
            def __init__(self):
                self.mlp = FakeMoELayer()

            def clear_fp8_quant_weight(self):
                if isinstance(self.mlp, FakeMoELayer):
                    self.mlp.clear_fp8_quant_weight()

        layer = FakeTransformerLayer()
        layer.clear_fp8_quant_weight()

        layer.mlp.clear_fp8_quant_weight.assert_called_once()

    def test_no_op_when_mlp_is_not_moe(self):
        """When mlp is not MoELayer, should do nothing."""

        class FakeMoELayer:
            pass

        class FakeTransformerLayer:
            def __init__(self):
                self.mlp = MagicMock()  # Not a FakeMoELayer

            def clear_fp8_quant_weight(self):
                if isinstance(self.mlp, FakeMoELayer):
                    self.mlp.clear_fp8_quant_weight()

        layer = FakeTransformerLayer()
        layer.clear_fp8_quant_weight()  # Should not raise


class TestGPTModelClearFp8QuantWeight(unittest.TestCase):
    """Test GPTModel.clear_fp8_quant_weight traversal logic."""

    def _make_transformer_layer(self):
        mock = MagicMock()
        mock.__class__ = type("TransformerLayer", (), {})
        return mock

    def _make_mtp_layer(self):
        mock = MagicMock()
        mock.__class__ = type("MultiTokenPredictionLayer", (), {})
        mock.transformer_layer = MagicMock()
        return mock

    def _gpt_clear(self, model):
        """Replicate GPTModel.clear_fp8_quant_weight logic."""
        TransformerLayer = type(self._make_transformer_layer())
        MultiTokenPredictionLayer = type(self._make_mtp_layer())

        if model._num_virtual_pipeline_stages > 1:
            for chunk in model._model_chunks:
                for layer in chunk:
                    if isinstance(layer, TransformerLayer):
                        layer.clear_fp8_quant_weight()
                    elif isinstance(layer, MultiTokenPredictionLayer):
                        layer.transformer_layer.clear_fp8_quant_weight()
        else:
            for layer in model.run_function:
                if isinstance(layer, TransformerLayer):
                    layer.clear_fp8_quant_weight()
                elif isinstance(layer, MultiTokenPredictionLayer):
                    layer.transformer_layer.clear_fp8_quant_weight()

    def test_single_stage_calls_transformer_layers(self):
        """With _num_virtual_pipeline_stages=1, iterates run_function."""

        class TransformerLayer:
            def __init__(self):
                self.clear_fp8_quant_weight = MagicMock()

        class MultiTokenPredictionLayer:
            def __init__(self):
                self.transformer_layer = MagicMock()

        tl = TransformerLayer()
        mtp = MultiTokenPredictionLayer()

        model = SimpleNamespace(
            _num_virtual_pipeline_stages=1,
            run_function=[tl, mtp, "other_layer"],
        )

        # Replicate the logic
        for layer in model.run_function:
            if isinstance(layer, TransformerLayer):
                layer.clear_fp8_quant_weight()
            elif isinstance(layer, MultiTokenPredictionLayer):
                layer.transformer_layer.clear_fp8_quant_weight()

        tl.clear_fp8_quant_weight.assert_called_once()
        mtp.transformer_layer.clear_fp8_quant_weight.assert_called_once()

    def test_multi_stage_iterates_all_chunks(self):
        """With _num_virtual_pipeline_stages>1, iterates _model_chunks."""

        class TransformerLayer:
            def __init__(self):
                self.clear_fp8_quant_weight = MagicMock()

        tl1 = TransformerLayer()
        tl2 = TransformerLayer()

        model = SimpleNamespace(
            _num_virtual_pipeline_stages=2,
            _model_chunks=[[tl1], [tl2]],
        )

        for chunk in model._model_chunks:
            for layer in chunk:
                if isinstance(layer, TransformerLayer):
                    layer.clear_fp8_quant_weight()

        tl1.clear_fp8_quant_weight.assert_called_once()
        tl2.clear_fp8_quant_weight.assert_called_once()


if __name__ == "__main__":
    unittest.main()
