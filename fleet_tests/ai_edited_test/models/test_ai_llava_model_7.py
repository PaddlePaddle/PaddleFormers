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
    IGNORE_INDEX,
    LLaVAModel,
    pixel_shuffle,
)


class TestPixelShuffle(unittest.TestCase):
    """Tests for pixel_shuffle function."""

    def test_pixel_shuffle_basic(self):
        """pixel_shuffle should produce valid output with scale_factor=0.5."""
        x = paddle.randn([2, 16, 64])
        result = pixel_shuffle(x, scale_factor=0.5)
        self.assertEqual(list(result.shape), [2, 4, 256])

    def test_pixel_shuffle_preserves_batch(self):
        """pixel_shuffle should preserve batch dimension."""
        x = paddle.randn([3, 16, 64])
        result = pixel_shuffle(x, scale_factor=0.5)
        self.assertEqual(result.shape[0], 3)

    def test_pixel_shuffle_output_elements_match(self):
        """pixel_shuffle should preserve total number of elements."""
        x = paddle.randn([2, 16, 64])
        result = pixel_shuffle(x, scale_factor=0.5)
        self.assertEqual(result.numel(), x.numel())


class TestLLaVAModelMethods(unittest.TestCase):
    """Tests for LLaVAModel method existence."""

    def test_has_set_input_tensor(self):
        """LLaVAModel should have set_input_tensor method."""
        self.assertTrue(hasattr(LLaVAModel, "set_input_tensor"))

    def test_has_freeze_method(self):
        """LLaVAModel should have freeze method."""
        self.assertTrue(hasattr(LLaVAModel, "freeze"))


class TestLLaVAModelSetInputTensor(unittest.TestCase):
    """Tests for LLaVAModel.set_input_tensor."""

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

    def test_set_input_tensor_with_encoder_only(self):
        """set_input_tensor with add_encoder True should set vision_model input."""
        mock_vision = MagicMock()
        model = self._make_model(
            add_encoder=True, add_decoder=False, vision_model=mock_vision
        )
        mock_tensor = MagicMock()
        model.set_input_tensor(mock_tensor)
        mock_vision.set_input_tensor.assert_called_once()

    def test_set_input_tensor_with_pre_process(self):
        """set_input_tensor with pre_process should store encoder_hidden_state."""
        model = self._make_model(
            add_encoder=False, add_decoder=False, pre_process=True
        )
        mock_tensor = MagicMock()
        model.set_input_tensor(mock_tensor)
        self.assertEqual(model.encoder_hidden_state, mock_tensor)

    def test_set_input_tensor_with_decoder_only(self):
        """set_input_tensor with no encoder should set language_model input."""
        mock_lang = MagicMock()
        model = self._make_model(
            add_encoder=False,
            add_decoder=False,
            pre_process=False,
            language_model=mock_lang,
        )
        mock_tensor = MagicMock()
        model.set_input_tensor(mock_tensor)
        mock_lang.set_input_tensor.assert_called_once()


class TestIgnoreIndex(unittest.TestCase):
    """Tests for IGNORE_INDEX constant."""

    def test_ignore_index_value(self):
        """IGNORE_INDEX should be -100."""
        self.assertEqual(IGNORE_INDEX, -100)


if __name__ == "__main__":
    unittest.main()
