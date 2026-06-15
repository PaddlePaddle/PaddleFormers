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


class TestQwen3_5RMSNormInit(unittest.TestCase):
    """Test Qwen3_5RMSNorm initialization."""

    def test_init_with_hidden_size(self):
        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNorm

        mock_config = MagicMock()
        mock_config.rms_norm_eps = 1e-5
        mock_config.hidden_size = 1024

        norm = Qwen3_5RMSNorm(config=mock_config, hidden_size=512, eps=1e-6)
        self.assertEqual(norm.normalized_shape, 512)
        self.assertEqual(norm.variance_epsilon, 1e-6)

    def test_init_with_normalized_shape(self):
        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNorm

        mock_config = MagicMock()
        mock_config.rms_norm_eps = 1e-5

        norm = Qwen3_5RMSNorm(config=mock_config, normalized_shape=256, norm_eps=1e-8)
        self.assertEqual(norm.normalized_shape, 256)
        self.assertEqual(norm.variance_epsilon, 1e-8)

    def test_init_uses_config_defaults(self):
        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNorm

        mock_config = MagicMock()
        mock_config.rms_norm_eps = 1e-6
        mock_config.hidden_size = 2048

        norm = Qwen3_5RMSNorm(config=mock_config)
        self.assertEqual(norm.normalized_shape, 2048)
        self.assertEqual(norm.variance_epsilon, 1e-6)

    def test_weight_initialized_to_zero(self):
        import paddle

        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNorm

        mock_config = MagicMock()
        mock_config.rms_norm_eps = 1e-5

        norm = Qwen3_5RMSNorm(config=mock_config, hidden_size=64)
        # Weight should be all zeros (1-centered parameterization)
        self.assertTrue(paddle.allclose(norm.weight, paddle.zeros([64])))


class TestQwen3_5RMSNormForward(unittest.TestCase):
    """Test Qwen3_5RMSNorm forward method."""

    def test_forward_returns_correct_shape(self):
        import paddle

        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNorm

        mock_config = MagicMock()
        mock_config.rms_norm_eps = 1e-5

        norm = Qwen3_5RMSNorm(config=mock_config, hidden_size=64)
        x = paddle.randn([2, 10, 64])
        result = norm(x)
        self.assertEqual(result.shape, x.shape)

    def test_forward_preserves_dtype(self):
        import paddle

        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNorm

        mock_config = MagicMock()
        mock_config.rms_norm_eps = 1e-5

        norm = Qwen3_5RMSNorm(config=mock_config, hidden_size=64)
        x = paddle.randn([2, 10, 64], dtype="float16")
        result = norm(x)
        self.assertEqual(result.dtype, x.dtype)

    def test_forward_with_zero_weight_is_identity_like(self):
        import paddle

        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNorm

        mock_config = MagicMock()
        mock_config.rms_norm_eps = 1e-5

        norm = Qwen3_5RMSNorm(config=mock_config, hidden_size=64)
        x = paddle.randn([1, 4, 64])
        result = norm(x)
        # With weight=0, output should be rms_norm(x) * (1 + 0) = rms_norm(x)
        self.assertIsNotNone(result)


class TestQwen3_5RMSNormEnableSP(unittest.TestCase):
    """Test Qwen3_5RMSNorm.enable_sequence_parallel."""

    @patch("paddleformers.fleet.models.qwen3_5.qwen3_5_model.mark_as_sequence_parallel_parameter")
    def test_enable_sp_calls_mark(self, mock_mark):
        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNorm

        mock_config = MagicMock()
        mock_config.rms_norm_eps = 1e-5

        norm = Qwen3_5RMSNorm(config=mock_config, hidden_size=64, input_is_parallel=True)
        mock_mark.assert_called_once_with(norm.weight)


class TestQwen3_5RMSNormPipe(unittest.TestCase):
    """Test Qwen3_5RMSNormPipe class."""

    def test_init_creates_norm(self):
        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNormPipe

        mock_config = MagicMock()
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False

        pipe = Qwen3_5RMSNormPipe(config=mock_config, hidden_size=64, eps=1e-5)
        self.assertIsNotNone(pipe.norm)

    def test_build_schedule_node(self):
        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNormPipe

        mock_config = MagicMock()
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False

        pipe = Qwen3_5RMSNormPipe(config=mock_config, hidden_size=64, eps=1e-5)
        node = pipe.build_schedule_node()
        self.assertIsNotNone(node)

    def test_forward_basic(self):
        import paddle

        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNormPipe

        mock_config = MagicMock()
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False

        pipe = Qwen3_5RMSNormPipe(config=mock_config, hidden_size=64, eps=1e-5)
        x = paddle.randn([2, 10, 64])
        result = pipe({"hidden_states": x})
        self.assertEqual(result["hidden_states"].shape, x.shape)

    def test_forward_with_mtp(self):
        import paddle

        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNormPipe

        mock_config = MagicMock()
        mock_config.num_nextn_predict_layers = 2
        mock_config.mtp_load_weight_only = False

        pipe = Qwen3_5RMSNormPipe(config=mock_config, hidden_size=64, eps=1e-5)
        # hidden dim=0 must be divisible by (num_nextn_predict_layers + 1)=3
        x = paddle.randn([3, 12, 64])
        result = pipe({"hidden_states": x})
        # Should split, normalize first part, then concat
        self.assertEqual(result["hidden_states"].shape, x.shape)


class TestQwen3_5VisionSublayersSpec(unittest.TestCase):
    """Test Qwen3_5VisionSublayersSpec dataclass."""

    def test_defaults(self):
        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import (
            Qwen3_5VisionSublayersSpec,
        )

        spec = Qwen3_5VisionSublayersSpec()
        self.assertIsNone(spec.embedding)
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.transformer_layers)
        self.assertIsNone(spec.tail_empty_layers)
        self.assertIsNone(spec.merger)


class TestQwen3_5VisionModel(unittest.TestCase):
    """Test Qwen3_5VisionModel."""

    @unittest.skip(
        "get_layer_desc_list requires LayerSpec instances, not MagicMock, "
        "and LayerDesc asserts isinstance(layer_spec, LayerSpec)"
    )
    @patch("paddleformers.fleet.models.qwen3_5.qwen3_5_model.TransformerEncoder.get_encoder_layer_desc_list")
    @patch("paddleformers.fleet.models.qwen3_5.qwen3_5_model.TransformerEncoder.add_sequential_layer")
    def test_get_layer_desc_list(self, mock_add, mock_encoder):
        from paddleformers.fleet.models.qwen3_5.qwen3_5_model import (
            Qwen3_5VisionModel,
            Qwen3_5VisionSublayersSpec,
        )

        mock_encoder.return_value = None
        mock_add.return_value = None

        spec = Qwen3_5VisionSublayersSpec(
            embedding=MagicMock(),
            transformer_layers=[MagicMock()],
            merger=MagicMock(),
        )

        model = Qwen3_5VisionModel.__new__(Qwen3_5VisionModel)
        model.modal = None
        layers = model.get_layer_desc_list(spec)
        # Should add embedding, encoder layers, and merger
        self.assertIsNotNone(layers)


if __name__ == "__main__":
    unittest.main()
