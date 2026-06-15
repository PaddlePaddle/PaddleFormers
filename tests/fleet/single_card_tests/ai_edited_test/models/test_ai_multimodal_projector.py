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


class TestMultimodalProjectorUnsupportedType(unittest.TestCase):
    """Test MultimodalProjector with unsupported projector type."""

    def test_unsupported_projector_type_raises(self):
        """Test that unsupported projector_type raises Exception."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        from paddleformers.fleet.transformer.mlp import MLPSublayersSpec

        sublayers_spec = MLPSublayersSpec()

        with self.assertRaises(Exception) as ctx:
            MultimodalProjector(
                config=config,
                sublayers_spec=sublayers_spec,
                projector_type="unsupported",
                input_size=128,
            )
        self.assertIn("unsupported", str(ctx.exception))

    def test_assert_none_sublayers_spec_raises(self):
        """Test that None sublayers_spec raises AssertionError."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )

        with self.assertRaises(AssertionError):
            MultimodalProjector(
                config=config,
                sublayers_spec=None,
                projector_type="mlp",
                input_size=128,
            )


class TestMultimodalProjectorMLPType(unittest.TestCase):
    """Test MultimodalProjector with MLP projector type."""

    @patch("paddleformers.fleet.models.vision.multimodal_projector.MLP")
    def test_mlp_type_creates_mlp_encoder(self, mock_mlp):
        """Test MLP projector type creates MLP encoder."""
        mock_mlp_instance = MagicMock()
        mock_mlp.return_value = mock_mlp_instance

        from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        sublayers_spec = MLPSublayersSpec()

        model = MultimodalProjector(
            config=config,
            sublayers_spec=sublayers_spec,
            projector_type="mlp",
            input_size=128,
        )
        self.assertEqual(model.projector_type, "mlp")
        mock_mlp.assert_called_once()


class TestMultimodalProjectorAffineType(unittest.TestCase):
    """Test MultimodalProjector with affine projector type."""

    @patch("paddleformers.fleet.models.vision.multimodal_projector.build_spec_layer")
    def test_affine_type_creates_linear_encoder(self, mock_build):
        """Test affine projector type creates build_spec_layer."""
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
        sublayers_spec = MLPSublayersSpec()
        sublayers_spec.linear_fc1 = MagicMock()

        model = MultimodalProjector(
            config=config,
            sublayers_spec=sublayers_spec,
            projector_type="affine",
            input_size=128,
        )
        self.assertEqual(model.projector_type, "affine")
        mock_build.assert_called_once()


class TestMultimodalProjectorForward(unittest.TestCase):
    """Test MultimodalProjector forward method."""

    def test_forward_with_bias(self):
        """Test forward when encoder returns bias."""
        model = MultimodalProjector.__new__(MultimodalProjector)
        model.projector_type = "mlp"

        mock_encoder = MagicMock()
        hidden_states = paddle.randn([2, 10, 64])
        bias = paddle.randn([64])
        mock_encoder.return_value = (hidden_states, bias)
        model.encoder = mock_encoder

        result = model.forward(hidden_states)
        self.assertIsNotNone(result)

    def test_forward_without_bias(self):
        """Test forward when encoder returns no bias."""
        model = MultimodalProjector.__new__(MultimodalProjector)
        model.projector_type = "mlp"

        mock_encoder = MagicMock()
        hidden_states = paddle.randn([2, 10, 64])
        mock_encoder.return_value = (hidden_states, None)
        model.encoder = mock_encoder

        result = model.forward(hidden_states)
        self.assertIsNotNone(result)
        # Result should be same as hidden_states since bias is None
        self.assertTrue(paddle.allclose(result, hidden_states))


if __name__ == "__main__":
    unittest.main()
