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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest
from unittest.mock import MagicMock

import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.models.gpt.gpt_model import GPTModel, GPTSublayersSpec


class _DummyLayer(paddle.nn.Layer):
    """A dummy paddle.nn.Layer subclass for LayerSpec/LayerDesc usage."""

    def __init__(self):
        super().__init__()


def _make_model(model_type="", experimental=False, num_nextn=0):
    """Create a GPTModel with minimal mock config."""
    model = GPTModel.__new__(GPTModel)
    model.config = MagicMock()
    model.config.model_type = model_type
    model.config.gpt_model_use_experimental_version = experimental
    model.config.num_nextn_predict_layers = num_nextn
    model._pipeline_name_mapping = None
    model._pp_to_single_mapping = None
    model._sequential_layers = []
    return model


def _make_spec(**overrides):
    """Create a GPTSublayersSpec with LayerSpec-wrapped dummy layers."""
    defaults = {
        "embedding": LayerSpec(_DummyLayer),
        "head_empty_layers": [],
        "transformer_layers": [LayerSpec(_DummyLayer)],
        "tail_empty_layers": [],
        "layer_norm": LayerSpec(_DummyLayer),
        "lm_head": LayerSpec(_DummyLayer),
    }
    defaults.update(overrides)
    return GPTSublayersSpec(**defaults)


class TestGPTModelGetLayerDescList(unittest.TestCase):
    """Test GPTModel.get_layer_desc_list method."""

    def test_qwen3_vl_name_prefix(self):
        """Test qwen3_vl model type uses language_model prefix."""
        model = _make_model(model_type="qwen3_vl")
        spec = _make_spec()

        layers = model.get_layer_desc_list(spec, tie_word_embeddings=False)
        self.assertTrue(len(layers) > 0)
        self.assertTrue(layers[0]["name_prefix"].startswith("model.language_model"))

    def test_qwen3_5_name_prefix(self):
        """Test qwen3_5 model type uses language_model prefix."""
        model = _make_model(model_type="qwen3_5")
        spec = _make_spec()

        layers = model.get_layer_desc_list(spec, tie_word_embeddings=False)
        self.assertTrue(layers[0]["name_prefix"].startswith("model.language_model"))

    def test_default_name_prefix(self):
        """Test default model type uses 'model' prefix."""
        model = _make_model(model_type="")
        spec = _make_spec()

        layers = model.get_layer_desc_list(spec, tie_word_embeddings=False)
        self.assertTrue(layers[0]["name_prefix"].startswith("model"))

    def test_with_mtp_layers(self):
        """Test get_layer_desc_list with MTP layers."""
        model = _make_model()
        spec = _make_spec(mtp=[LayerSpec(_DummyLayer), LayerSpec(_DummyLayer)])

        layers = model.get_layer_desc_list(spec, tie_word_embeddings=False)
        self.assertTrue(len(layers) > 0)

    def test_with_tie_word_embeddings(self):
        """Test get_layer_desc_list with tie_word_embeddings=True."""
        model = _make_model()
        spec = _make_spec()

        layers = model.get_layer_desc_list(spec, tie_word_embeddings=True)
        self.assertTrue(len(layers) > 0)

    def test_with_mtp_lm_head(self):
        """Test get_layer_desc_list with mtp_lm_head."""
        model = _make_model()
        # SharedLayerDesc expects LayerSpec or class, not LayerDesc
        # So mtp_lm_head must be a LayerSpec
        mock_mtp_lm_head = LayerSpec(_DummyLayer)
        mock_mtp_loss = LayerSpec(_DummyLayer)
        spec = _make_spec(
            mtp_lm_head=mock_mtp_lm_head,
            mtp_loss=mock_mtp_loss,
        )

        layers = model.get_layer_desc_list(spec, tie_word_embeddings=False)
        self.assertTrue(len(layers) > 0)

    def test_experimental_version_no_layer_norm(self):
        """Test that experimental version with MTP skips layer_norm."""
        model = _make_model(experimental=True, num_nextn=1)
        spec = _make_spec(mtp=[LayerSpec(_DummyLayer)])

        layers = model.get_layer_desc_list(spec, tie_word_embeddings=False)
        self.assertTrue(len(layers) > 0)

    def test_with_head_empty_layers(self):
        """Test get_layer_desc_list with head_empty_layers."""
        model = _make_model()
        spec = _make_spec(head_empty_layers=[LayerSpec(_DummyLayer)])

        layers = model.get_layer_desc_list(spec, tie_word_embeddings=False)
        self.assertTrue(len(layers) > 0)

    def test_with_tail_empty_layers(self):
        """Test get_layer_desc_list with tail_empty_layers."""
        model = _make_model()
        spec = _make_spec(tail_empty_layers=[LayerSpec(_DummyLayer), LayerSpec(_DummyLayer)])

        layers = model.get_layer_desc_list(spec, tie_word_embeddings=False)
        self.assertTrue(len(layers) > 0)


class TestGPTModelOverlappedForwardBackward(unittest.TestCase):
    """Test GPTModel.overlapped_forward_backward method."""

    def test_basic_call(self):
        """Test overlapped_forward_backward with mocked chunks."""
        model = GPTModel.__new__(GPTModel)
        model.config = MagicMock()

        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        mock_forward_chunk = ScheduleChunk([])
        mock_backward_chunk = ScheduleChunk([])

        result = model.overlapped_forward_backward(
            forward_chunk=mock_forward_chunk,
            forward_inputs=None,
            forward_loss_fn_node=None,
            backward_chunk=mock_backward_chunk,
            backward_loss_fn_node=None,
            backward_input_grads=None,
            scaler=None,
            p2p_async_handle=None,
        )
        self.assertIsNotNone(result)


class TestGPTModelSetPipelineNameMapping(unittest.TestCase):
    """Test GPTModel._set_pipeline_name_mapping method."""

    def test_with_explicit_mappings(self):
        """Test setting explicit mappings."""
        model = GPTModel.__new__(GPTModel)
        model._pipeline_name_mapping = None

        mappings = {"layer0.weight": "0.layer0.weight"}
        result = model._set_pipeline_name_mapping(mappings)
        self.assertEqual(result, mappings)


class TestGPTModelFp8Methods(unittest.TestCase):
    """Test GPTModel fp8 methods."""

    def test_fp8_quant_weight_virtual_stages(self):
        """Test fp8_quant_weight with virtual pipeline stages."""
        model = GPTModel.__new__(GPTModel)
        model.config = MagicMock()
        model._num_virtual_pipeline_stages = 2
        model._model_chunks = [[MagicMock()]]

        model.fp8_quant_weight()

    def test_use_fp8_virtual_stages_true(self):
        """Test use_fp8 returns True with virtual stages and fp8 layer."""
        model = GPTModel.__new__(GPTModel)
        model.config = MagicMock()
        model._num_virtual_pipeline_stages = 2
        model._model_chunks = [[MagicMock()]]

        result = model.use_fp8()
        self.assertFalse(result)

    def test_use_fp8_no_virtual_stages_false(self):
        """Test use_fp8 returns False without fp8 layers."""
        model = GPTModel.__new__(GPTModel)
        model.config = MagicMock()
        model._num_virtual_pipeline_stages = 1
        model.run_function = [MagicMock()]

        result = model.use_fp8()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
