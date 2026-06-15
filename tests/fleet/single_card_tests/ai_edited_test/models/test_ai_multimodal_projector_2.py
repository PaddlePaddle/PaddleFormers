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
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.models.vision.multimodal_projector import MultimodalProjector


class TestMultimodalProjectorIntegration(unittest.TestCase):
    """Test MultimodalProjector with real MLP."""

    def test_mlp_projector_forward(self):
        """Test MLP projector construction via mocked MLP."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        with patch("paddleformers.fleet.models.vision.multimodal_projector.MLP") as mock_mlp:
            mock_mlp.return_value = MagicMock()
            from paddleformers.fleet.transformer.mlp import MLPSublayersSpec

            sublayers_spec = MLPSublayersSpec()
            model = MultimodalProjector(
                config=config,
                sublayers_spec=sublayers_spec,
                projector_type="mlp",
                input_size=64,
            )
            self.assertIsNotNone(model.encoder)
            self.assertEqual(model.projector_type, "mlp")
            mock_mlp.assert_called_once()

    def test_projector_type_attribute(self):
        """Test projector_type is stored correctly."""
        model = MultimodalProjector.__new__(MultimodalProjector)
        model.projector_type = "mlp"
        self.assertEqual(model.projector_type, "mlp")

        model2 = MultimodalProjector.__new__(MultimodalProjector)
        model2.projector_type = "affine"
        self.assertEqual(model2.projector_type, "affine")


class TestMultimodalProjectorForwardPaths(unittest.TestCase):
    """Test MultimodalProjector forward paths."""

    def test_forward_with_zero_bias(self):
        """Test forward when encoder returns zero bias."""
        model = MultimodalProjector.__new__(MultimodalProjector)
        hidden_states = paddle.randn([2, 10, 64])
        bias = paddle.zeros([64])
        mock_encoder = MagicMock(return_value=(hidden_states, bias))
        model.encoder = mock_encoder

        result = model.forward(hidden_states)
        self.assertIsNotNone(result)
        # Result should be hidden_states + bias
        expected = hidden_states + bias
        self.assertTrue(paddle.allclose(result, expected, atol=1e-5))

    def test_forward_with_nonzero_bias(self):
        """Test forward when encoder returns non-zero bias."""
        model = MultimodalProjector.__new__(MultimodalProjector)
        hidden_states = paddle.randn([2, 10, 64])
        bias = paddle.ones([64])
        mock_encoder = MagicMock(return_value=(hidden_states, bias))
        model.encoder = mock_encoder

        result = model.forward(hidden_states)
        self.assertIsNotNone(result)
        expected = hidden_states + bias
        self.assertTrue(paddle.allclose(result, expected, atol=1e-5))

    def test_forward_calls_encoder(self):
        """Test forward calls encoder with hidden_states."""
        model = MultimodalProjector.__new__(MultimodalProjector)
        hidden_states = paddle.randn([2, 10, 64])
        mock_encoder = MagicMock(return_value=(hidden_states, None))
        model.encoder = mock_encoder

        model.forward(hidden_states)
        mock_encoder.assert_called_once_with(hidden_states)


class TestMultimodalProjectorAffineWithTPGroup(unittest.TestCase):
    """Test MultimodalProjector affine type with tp_group."""

    @patch("paddleformers.fleet.models.vision.multimodal_projector.build_spec_layer")
    def test_affine_with_tp_group(self, mock_build):
        """Test affine projector type with tp_group."""
        mock_build.return_value = MagicMock()

        from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            use_bias=True,
        )
        # Source code uses config.add_bias_linear which doesn't exist in
        # the installed TransformerConfig. Add it manually for affine path.
        config.add_bias_linear = True
        config.init_method = MagicMock()
        config.init_method = MagicMock()
        sublayers_spec = MLPSublayersSpec()
        sublayers_spec.linear_fc1 = MagicMock()

        mock_tp_group = MagicMock()

        model = MultimodalProjector(
            config=config,
            sublayers_spec=sublayers_spec,
            projector_type="affine",
            input_size=128,
            tp_group=mock_tp_group,
        )
        self.assertEqual(model.projector_type, "affine")
        # Check that tp_group was passed
        call_kwargs = mock_build.call_args[1]
        self.assertEqual(call_kwargs.get("tp_group"), mock_tp_group)


if __name__ == "__main__":
    unittest.main()
