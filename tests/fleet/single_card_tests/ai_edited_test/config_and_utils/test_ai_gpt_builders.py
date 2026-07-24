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
from functools import partial
from unittest.mock import MagicMock, patch


class TestGptBuilder(unittest.TestCase):
    """Tests for gpt_builder and _get_transformer_layer_spec_func in paddleformers.fleet.gpt_builders."""

    def test_get_transformer_layer_spec_func_returns_partial(self):
        from paddleformers.fleet.gpt_builders import (
            _get_transformer_layer_spec_func,
        )

        mock_config = MagicMock()
        mock_config.use_qk_norm = False
        mock_config.n_routed_experts = None
        mock_config.multi_latent_attention = False
        mock_config.normalization = "RMSNorm"

        func = _get_transformer_layer_spec_func(mock_config)
        self.assertIsInstance(func, partial)

    def test_get_transformer_layer_spec_func_mla(self):
        from paddleformers.fleet.gpt_builders import (
            _get_transformer_layer_spec_func,
        )

        mock_config = MagicMock()
        mock_config.use_qk_norm = True
        mock_config.n_routed_experts = None
        mock_config.multi_latent_attention = True
        mock_config.normalization = "RMSNorm"

        func = _get_transformer_layer_spec_func(mock_config)
        self.assertIsInstance(func, partial)

    def test_get_transformer_layer_spec_func_with_experts(self):
        from paddleformers.fleet.gpt_builders import (
            _get_transformer_layer_spec_func,
        )

        mock_config = MagicMock()
        mock_config.use_qk_norm = False
        mock_config.n_routed_experts = 8
        mock_config.multi_latent_attention = False
        mock_config.normalization = "LayerNorm"

        func = _get_transformer_layer_spec_func(mock_config)
        self.assertIsInstance(func, partial)

    def test_gpt_builder_without_routed_experts(self):
        """Test gpt_builder path when n_routed_experts is falsy."""
        from paddleformers.fleet.gpt_builders import gpt_builder

        mock_config = MagicMock()
        mock_config.n_routed_experts = None
        mock_config.num_hidden_layers = 2
        mock_config.num_empty_layers_add_in_head = 0
        mock_config.num_empty_layers_add_in_tail = 0
        mock_config.num_nextn_predict_layers = None
        mock_config.vocab_size = 1000
        mock_config.tie_word_embeddings = False
        mock_config.max_sequence_length = 128
        mock_config.position_embedding_type = "rope"
        mock_config.rotary_percent = 1.0
        mock_config.rope_theta = 10000.0
        mock_config.rope_scaling = None
        mock_config.parallel_output = True
        mock_config.hidden_size = 64
        mock_config.use_qk_norm = False
        mock_config.multi_latent_attention = False
        mock_config.normalization = "RMSNorm"
        mock_config.separate_mtp_headloss = False

        mock_loss = MagicMock()

        with (
            patch(
                "paddleformers.fleet.gpt_builders.get_gpt_spec"
            ) as mock_get_spec,
            patch(
                "paddleformers.fleet.gpt_builders.build_spec_layer"
            ) as mock_build,
            patch(
                "paddleformers.fleet.gpt_builders.get_gpt_layer_local_spec"
            ) as mock_layer_spec,
        ):
            mock_layer_spec.return_value = MagicMock()
            mock_build.return_value = MagicMock()

            result = gpt_builder(mock_config, loss_fn=mock_loss)
            mock_build.assert_called_once()
            self.assertIsNotNone(result)

    def test_gpt_builder_with_routed_experts(self):
        """Test gpt_builder path when n_routed_experts is set."""
        from paddleformers.fleet.gpt_builders import gpt_builder

        mock_config = MagicMock()
        mock_config.n_routed_experts = 8
        mock_config.normalization = "RMSNorm"
        mock_config.num_empty_layers_add_in_head = 0
        mock_config.num_empty_layers_add_in_tail = 0
        mock_config.num_nextn_predict_layers = None
        mock_config.vocab_size = 512
        mock_config.tie_word_embeddings = False
        mock_config.max_sequence_length = 64
        mock_config.position_embedding_type = "rope"
        mock_config.rotary_percent = 1.0
        mock_config.rope_theta = 10000.0
        mock_config.rope_scaling = None
        mock_config.parallel_output = True
        mock_config.separate_mtp_headloss = False

        mock_loss = MagicMock()

        with (
            patch(
                "paddleformers.fleet.gpt_builders.get_gpt_spec"
            ) as mock_get_spec,
            patch(
                "paddleformers.fleet.gpt_builders.build_spec_layer"
            ) as mock_build,
            patch(
                "paddleformers.fleet.gpt_builders.get_gpt_decoder_layers_spec"
            ) as mock_decoder,
        ):
            mock_decoder.return_value = [MagicMock()]
            mock_build.return_value = MagicMock()

            result = gpt_builder(mock_config, loss_fn=mock_loss)
            mock_build.assert_called_once()

    def test_gpt_builder_no_loss_fn_creates_default(self):
        """Test gpt_builder creates default LanguageLoss when no loss_fn is provided."""
        from paddleformers.fleet.gpt_builders import gpt_builder

        mock_config = MagicMock()
        mock_config.n_routed_experts = None
        mock_config.num_hidden_layers = 1
        mock_config.num_empty_layers_add_in_head = 0
        mock_config.num_empty_layers_add_in_tail = 0
        mock_config.num_nextn_predict_layers = None
        mock_config.vocab_size = 512
        mock_config.tie_word_embeddings = False
        mock_config.max_sequence_length = 64
        mock_config.position_embedding_type = "rope"
        mock_config.rotary_percent = 1.0
        mock_config.rope_theta = 10000.0
        mock_config.rope_scaling = None
        mock_config.parallel_output = True
        mock_config.hidden_size = 32
        mock_config.use_qk_norm = False
        mock_config.multi_latent_attention = False
        mock_config.normalization = "RMSNorm"
        mock_config.separate_mtp_headloss = False

        with (
            patch(
                "paddleformers.fleet.gpt_builders.get_gpt_spec"
            ) as mock_get_spec,
            patch(
                "paddleformers.fleet.gpt_builders.build_spec_layer"
            ) as mock_build,
            patch(
                "paddleformers.fleet.gpt_builders.get_gpt_layer_local_spec"
            ) as mock_layer_spec,
            patch(
                "paddleformers.fleet.gpt_builders.LanguageLoss"
            ) as mock_loss_cls,
        ):
            mock_layer_spec.return_value = MagicMock()
            mock_loss_instance = MagicMock()
            mock_loss_cls.return_value = mock_loss_instance
            mock_build.return_value = MagicMock()

            result = gpt_builder(mock_config)
            # Verify LanguageLoss was called with config
            mock_loss_cls.assert_called_once_with(mock_config)
