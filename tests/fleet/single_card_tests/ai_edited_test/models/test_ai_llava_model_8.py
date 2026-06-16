# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on applicable law or agreed to in writing, software
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


class TestPixelShuffleVersion2(unittest.TestCase):
    """Tests for pixel_shuffle with version parameter."""

    def test_pixel_shuffle_version_1(self):
        """pixel_shuffle with version=1 should work."""
        x = paddle.randn([2, 16, 64])
        result = pixel_shuffle(x, scale_factor=0.5, version=1)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.numel(), x.numel())

    def test_pixel_shuffle_version_2_default(self):
        """pixel_shuffle should default to version=2."""
        x = paddle.randn([2, 16, 64])
        result_v2 = pixel_shuffle(x, scale_factor=0.5, version=2)
        result_default = pixel_shuffle(x, scale_factor=0.5)
        self.assertTrue(paddle.allclose(result_v2, result_default))


class TestLLaVAModelSetInputTensorBoth(unittest.TestCase):
    """Tests for LLaVAModel.set_input_tensor with both encoder and decoder."""

    def _make_model(self, **attrs):
        with patch.object(LLaVAModel, "__init__", lambda self, *a, **kw: None):
            model = LLaVAModel.__new__(LLaVAModel)
            object.__setattr__(model, "_sub_layers", {})
            object.__setattr__(model, "_parameters", {})
            object.__setattr__(model, "_buffers", {})
            object.__setattr__(model, "_non_persistable_buffers", set())
            for k, v in attrs.items():
                object.__setattr__(model, k, v)
            return model

    def test_sets_both_encoder_and_decoder(self):
        """set_input_tensor with both add_encoder and add_decoder should set vision_model input."""
        mock_vision = MagicMock()
        model = self._make_model(
            add_encoder=True, add_decoder=True, vision_model=mock_vision
        )
        mock_tensor = MagicMock()
        model.set_input_tensor(mock_tensor)
        mock_vision.set_input_tensor.assert_called_once()


class TestLLaVAModelFreezeMethod(unittest.TestCase):
    """Tests for LLaVAModel.freeze method."""

    def test_freeze_method_exists(self):
        """freeze should be a callable method."""
        self.assertTrue(callable(LLaVAModel.freeze))


if __name__ == "__main__":
    unittest.main()
