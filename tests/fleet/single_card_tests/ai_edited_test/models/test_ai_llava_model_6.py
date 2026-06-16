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

from paddleformers.fleet.models.multimodal.llava_model import (
    LLaVAModel,
    pixel_shuffle,
)


class TestLLaVAModelSetInputTensor(unittest.TestCase):
    """Test LLaVAModel set_input_tensor additional paths."""

    def test_set_input_tensor_encoder_only(self):
        """Test set_input_tensor when only encoder is added (not decoder)."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_encoder = True
        model.add_decoder = False
        model.pre_process = True
        model.vision_model = MagicMock()

        mock_tensor = paddle.randn([10, 64])
        model.set_input_tensor([mock_tensor])
        model.vision_model.set_input_tensor.assert_called_once_with(mock_tensor)

    def test_set_input_tensor_pre_process_no_encoder(self):
        """Test set_input_tensor when pre_process and no encoder."""
        model = LLaVAModel.__new__(LLaVAModel)
        # Initialize Paddle Layer internals so __setattr__ works
        model.__dict__.setdefault("_parameters", {})
        model.__dict__.setdefault("_buffers", {})
        model.__dict__.setdefault("_sub_layers", {})
        model.__dict__.setdefault("_loaddict_holder", {})
        model.__dict__.setdefault("_non_persistable_buffers", set())
        model.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        model.add_encoder = False
        model.add_decoder = True
        model.pre_process = True

        mock_tensor = paddle.randn([10, 64])
        model.set_input_tensor([mock_tensor])
        # Should set encoder_hidden_state
        self.assertIsNotNone(model.encoder_hidden_state)

    def test_set_input_tensor_not_list(self):
        """Test set_input_tensor with non-list input wraps in list."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_encoder = True
        model.add_decoder = True
        model.vision_model = MagicMock()

        mock_tensor = paddle.randn([10, 64])
        model.set_input_tensor(mock_tensor)
        model.vision_model.set_input_tensor.assert_called_once_with(mock_tensor)


class TestLLaVAModelFreeze(unittest.TestCase):
    """Test LLaVAModel freeze method additional paths."""

    def test_freeze_vision_projection(self):
        """Test freezing vision projection module."""
        model = LLaVAModel.__new__(LLaVAModel)
        param = MagicMock()
        param.stop_gradient = False
        mock_proj = MagicMock()
        mock_proj.parameters.return_value = [param]
        model.language_model = None
        model.vision_model = None
        model.vision_projection = mock_proj

        model.freeze(False, False, True)
        self.assertTrue(param.stop_gradient)

    def test_freeze_language_model(self):
        """Test freezing language model module."""
        model = LLaVAModel.__new__(LLaVAModel)
        param = MagicMock()
        param.stop_gradient = False
        mock_lm = MagicMock()
        mock_lm.parameters.return_value = [param]
        model.language_model = mock_lm
        model.vision_model = None
        model.vision_projection = None

        model.freeze(True, False, False)
        self.assertTrue(param.stop_gradient)

    def test_freeze_none_vision_projection(self):
        """Test freeze when vision_projection is None doesn't crash."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.language_model = None
        model.vision_model = None
        model.vision_projection = None

        # Should not raise
        model.freeze(False, False, True)


class TestLLaVAModelSharedEmbeddingWeight(unittest.TestCase):
    """Test LLaVAModel shared_embedding_or_output_weight."""

    def test_no_decoder_returns_none(self):
        """Test shared_embedding_or_output_weight returns None without decoder."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_decoder = False
        result = model.shared_embedding_or_output_weight()
        self.assertIsNone(result)

    def test_with_decoder_delegates(self):
        """Test shared_embedding_or_output_weight delegates to language_model."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_decoder = True
        mock_weight = paddle.randn([100, 64])
        mock_lm = MagicMock()
        mock_lm.shared_embedding_or_output_weight.return_value = mock_weight
        model.language_model = mock_lm
        result = model.shared_embedding_or_output_weight()
        self.assertIs(result, mock_weight)


class TestLLaVAModelInitVisionModelType(unittest.TestCase):
    """Test LLaVAModel init with different vision_model_type."""

    @patch(
        "paddleformers.fleet.models.multimodal.llava_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.multimodal.llava_model.log_single_rank")
    @patch(
        "paddleformers.fleet.models.multimodal.llava_model.get_num_image_embeddings",
        return_value=16,
    )
    @patch("paddleformers.fleet.models.multimodal.llava_model.GPTModel")
    @patch("paddleformers.fleet.models.multimodal.llava_model.CLIPViTModel")
    @patch(
        "paddleformers.fleet.models.multimodal.llava_model.MultimodalProjector"
    )
    def test_unsupported_vision_model_raises(
        self, mock_proj, mock_clip, mock_gpt, mock_num, mock_log, mock_cfg
    ):
        """Test unsupported vision_model_type raises ValueError."""
        mock_lang_config = MagicMock()
        mock_lang_config.hidden_size = 64
        mock_lang_config.params_dtype = "float32"
        mock_lang_config.sequence_parallel = False
        mock_lang_config.context_parallel_size = 1
        mock_lang_config.tensor_model_parallel_size = 1
        mock_lang_config.pipeline_model_parallel_size = 1
        mock_lang_config.language_model_type = ""

        mock_vision_config = MagicMock()
        mock_vision_config.hidden_size = 64
        mock_vision_config.params_dtype = "float32"
        mock_vision_config.vision_model_type = "unsupported_type"

        mock_proj_config = MagicMock()

        with self.assertRaises(ValueError):
            LLaVAModel(
                language_transformer_config=mock_lang_config,
                language_transformer_layer_spec=MagicMock(),
                language_vocab_size=1000,
                language_max_sequence_length=512,
                vision_transformer_config=mock_vision_config,
                vision_transformer_layer_spec=MagicMock(),
                drop_vision_class_token=False,
                vision_projection_config=mock_proj_config,
                vision_projection_layer_spec=MagicMock(),
                vision_projection_type="mlp",
            )


class TestLLaVAModelImgSeqLen(unittest.TestCase):
    """Test LLaVAModel img_seq_len computation."""

    @patch(
        "paddleformers.fleet.models.multimodal.llava_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.multimodal.llava_model.log_single_rank")
    @patch(
        "paddleformers.fleet.models.multimodal.llava_model.get_num_image_embeddings",
        return_value=576,
    )
    @patch("paddleformers.fleet.models.multimodal.llava_model.GPTModel")
    @patch("paddleformers.fleet.models.multimodal.llava_model.CLIPViTModel")
    @patch(
        "paddleformers.fleet.models.multimodal.llava_model.MultimodalProjector"
    )
    def test_img_seq_len_stored(
        self, mock_proj, mock_clip, mock_gpt, mock_num, mock_log, mock_cfg
    ):
        """Test img_seq_len is stored from get_num_image_embeddings."""
        mock_lang_config = MagicMock()
        mock_lang_config.hidden_size = 64
        mock_lang_config.params_dtype = "float32"
        mock_lang_config.sequence_parallel = False
        mock_lang_config.context_parallel_size = 1
        mock_lang_config.tensor_model_parallel_size = 1
        mock_lang_config.pipeline_model_parallel_size = 1
        mock_lang_config.language_model_type = ""

        mock_vision_config = MagicMock()
        mock_vision_config.hidden_size = 64
        mock_vision_config.params_dtype = "float32"
        mock_vision_config.vision_model_type = "clip"

        mock_proj_config = MagicMock()

        mock_gpt.return_value = MagicMock()
        mock_clip.return_value = MagicMock()
        mock_proj.return_value = MagicMock()
        mock_proj.return_value.state_dict.return_value = {}

        model = LLaVAModel(
            language_transformer_config=mock_lang_config,
            language_transformer_layer_spec=MagicMock(),
            language_vocab_size=1000,
            language_max_sequence_length=512,
            vision_transformer_config=mock_vision_config,
            vision_transformer_layer_spec=MagicMock(),
            drop_vision_class_token=False,
            vision_projection_config=mock_proj_config,
            vision_projection_layer_spec=MagicMock(),
            vision_projection_type="mlp",
        )
        self.assertEqual(model.img_seq_len, 576)
        self.assertFalse(model._pixel_shuffle)


class TestLLaVAModelSequenceParallelAssertion(unittest.TestCase):
    """Test LLaVAModel assertion for sequence_parallel."""

    @patch(
        "paddleformers.fleet.models.multimodal.llava_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.multimodal.llava_model.log_single_rank")
    def test_sequence_parallel_raises(self, mock_log, mock_cfg):
        """Test that sequence_parallel raises AssertionError."""
        mock_lang_config = MagicMock()
        mock_lang_config.hidden_size = 64
        mock_lang_config.params_dtype = "float32"
        mock_lang_config.sequence_parallel = True
        mock_lang_config.context_parallel_size = 1

        with self.assertRaises(AssertionError):
            LLaVAModel(
                language_transformer_config=mock_lang_config,
                language_transformer_layer_spec=MagicMock(),
                language_vocab_size=1000,
                language_max_sequence_length=512,
                vision_transformer_config=MagicMock(),
                vision_transformer_layer_spec=MagicMock(),
                drop_vision_class_token=False,
                vision_projection_config=MagicMock(),
                vision_projection_layer_spec=MagicMock(),
                vision_projection_type="mlp",
            )

    @patch(
        "paddleformers.fleet.models.multimodal.llava_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.multimodal.llava_model.log_single_rank")
    def test_context_parallel_raises(self, mock_log, mock_cfg):
        """Test that context_parallel_size > 1 raises AssertionError."""
        mock_lang_config = MagicMock()
        mock_lang_config.hidden_size = 64
        mock_lang_config.params_dtype = "float32"
        mock_lang_config.sequence_parallel = False
        mock_lang_config.context_parallel_size = 2

        with self.assertRaises(AssertionError):
            LLaVAModel(
                language_transformer_config=mock_lang_config,
                language_transformer_layer_spec=MagicMock(),
                language_vocab_size=1000,
                language_max_sequence_length=512,
                vision_transformer_config=MagicMock(),
                vision_transformer_layer_spec=MagicMock(),
                drop_vision_class_token=False,
                vision_projection_config=MagicMock(),
                vision_projection_layer_spec=MagicMock(),
                vision_projection_type="mlp",
            )


class TestPixelShuffleVersions(unittest.TestCase):
    """Test pixel_shuffle with different versions."""

    def test_pixel_shuffle_version1_shape(self):
        """Test pixel_shuffle version 1 shape."""
        x = paddle.randn([2, 16, 128])
        result = pixel_shuffle(x, scale_factor=0.5, version=1)
        self.assertIsNotNone(result)

    def test_pixel_shuffle_version2_shape(self):
        """Test pixel_shuffle version 2 shape."""
        x = paddle.randn([2, 16, 128])
        result = pixel_shuffle(x, scale_factor=0.5, version=2)
        self.assertIsNotNone(result)

    def test_pixel_shuffle_identity(self):
        """Test pixel_shuffle with scale=1.0 is identity-like."""
        x = paddle.randn([2, 16, 128])
        result = pixel_shuffle(x, scale_factor=1.0, version=2)
        self.assertEqual(result.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
