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


# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import unittest
from collections import namedtuple
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.models.multimodal.llava_model import (
    DEFAULT_IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
    IMAGE_TOKEN,
    VIDEO_TOKEN,
    LLaVAModel,
    _load_state_dict_hook_ignore_extra_state,
    _load_state_dict_hook_ignore_param_names,
    pixel_shuffle,
)


class TestLLaVAModelConstants(unittest.TestCase):
    """Test module-level constants."""

    def test_ignore_index(self):
        self.assertEqual(IGNORE_INDEX, -100)

    def test_default_image_token_index(self):
        self.assertEqual(DEFAULT_IMAGE_TOKEN_INDEX, -200)

    def test_image_token(self):
        self.assertEqual(IMAGE_TOKEN, "<image>")

    def test_video_token(self):
        self.assertEqual(VIDEO_TOKEN, "<video>")


class TestPixelShuffle(unittest.TestCase):
    """Test pixel_shuffle function."""

    def test_pixel_shuffle_basic(self):
        """Test basic pixel_shuffle function."""
        x = paddle.randn([2, 196, 512])
        result = pixel_shuffle(x, scale_factor=0.5, version=2)
        self.assertEqual(result.shape[0], 2)
        # output seq_len = (sqrt(196) * 0.5)^2 * 4 = 98 / 4 * 4 ... actually:
        # h = w = sqrt(196) = 14
        # After shuffle: seq_len = int(14 * 0.5) * int(14 * 0.5) = 7 * 7 = 49
        # hidden = 512 / (0.5 * 0.5) = 512 / 0.25 = 2048
        self.assertEqual(result.shape[1], 49)
        self.assertEqual(result.shape[2], 2048)

    def test_pixel_shuffle_version1(self):
        """Test pixel_shuffle with version=1."""
        x = paddle.randn([2, 196, 512])
        result = pixel_shuffle(x, scale_factor=0.5, version=1)
        self.assertEqual(result.shape[0], 2)

    def test_pixel_shuffle_identity_scale(self):
        """Test pixel_shuffle with scale_factor=1.0."""
        x = paddle.randn([2, 196, 512])
        result = pixel_shuffle(x, scale_factor=1.0, version=2)
        self.assertEqual(result.shape, [2, 196, 512])


class TestLoadStateDictHooks(unittest.TestCase):
    """Test state dict hook functions."""

    def test_ignore_param_names_removes_missing(self):
        """Test _load_state_dict_hook_ignore_param_names removes matching keys."""
        incompatible_keys = namedtuple(
            "IncompatibleKeys", ["missing_keys", "unexpected_keys"]
        )
        keys = incompatible_keys(
            missing_keys=["vision_projection.fc1.weight", "other.weight"],
            unexpected_keys=[],
        )
        _load_state_dict_hook_ignore_param_names(
            ["vision_projection.fc1.weight"],
            None,
            keys,
        )
        self.assertEqual(keys.missing_keys, ["other.weight"])

    def test_ignore_param_names_no_match(self):
        """Test hook when param_name is not in missing keys."""
        incompatible_keys = namedtuple(
            "IncompatibleKeys", ["missing_keys", "unexpected_keys"]
        )
        keys = incompatible_keys(
            missing_keys=["other.weight"],
            unexpected_keys=[],
        )
        _load_state_dict_hook_ignore_param_names(
            ["vision_projection.fc1.weight"],
            None,
            keys,
        )
        self.assertEqual(len(keys.missing_keys), 1)

    def test_ignore_extra_state_removes(self):
        """Test _load_state_dict_hook_ignore_extra_state removes _extra_state keys."""
        incompatible_keys = namedtuple(
            "IncompatibleKeys", ["missing_keys", "unexpected_keys"]
        )
        keys = incompatible_keys(
            missing_keys=["layer._extra_state"],
            unexpected_keys=["other._extra_state"],
        )
        _load_state_dict_hook_ignore_extra_state(None, keys)
        self.assertEqual(len(keys.missing_keys), 0)
        self.assertEqual(len(keys.unexpected_keys), 0)

    def test_ignore_extra_state_no_extra(self):
        """Test hook when no _extra_state keys present."""
        incompatible_keys = namedtuple(
            "IncompatibleKeys", ["missing_keys", "unexpected_keys"]
        )
        keys = incompatible_keys(
            missing_keys=["layer.weight"],
            unexpected_keys=[],
        )
        _load_state_dict_hook_ignore_extra_state(None, keys)
        self.assertEqual(len(keys.missing_keys), 1)


class TestLLaVAModelConstruction(unittest.TestCase):
    """Test LLaVAModel construction logic (unit-level, no GPU)."""


class TestLLaVAModelMethods(unittest.TestCase):
    """Test LLaVAModel methods with mocked internals."""

    def test_set_input_tensor_list(self):
        """Test set_input_tensor with list input."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_encoder = True
        model.add_decoder = True
        model.pre_process = True
        model.vision_model = MagicMock()

        mock_tensor = paddle.randn([10, 64])
        model.set_input_tensor([mock_tensor])
        model.vision_model.set_input_tensor.assert_called_once_with(mock_tensor)

    def test_set_input_tensor_decoder_only(self):
        """Test set_input_tensor when only decoder is added."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_encoder = False
        model.pre_process = False
        model.language_model = MagicMock()

        mock_tensor = paddle.randn([10, 64])
        model.set_input_tensor([mock_tensor])
        model.language_model.set_input_tensor.assert_called_once_with(
            mock_tensor
        )

    def test_freeze_all(self):
        """Test freeze method freezes all modules."""
        model = LLaVAModel.__new__(LLaVAModel)
        lang_model = MagicMock()
        vision_model = MagicMock()
        vision_proj = MagicMock()
        param = MagicMock()
        param.stop_gradient = False
        lang_model.parameters.return_value = [param]
        vision_model.parameters.return_value = [param]
        vision_proj.parameters.return_value = [param]

        model.language_model = lang_model
        model.vision_model = vision_model
        model.vision_projection = vision_proj

        model.freeze(True, True, True)
        self.assertTrue(param.stop_gradient)

    def test_freeze_none(self):
        """Test freeze method with all False."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.language_model = None
        model.vision_model = None
        # Should not raise
        model.freeze(False, False, False)

    def test_freeze_partial(self):
        """Test freeze with only vision frozen."""
        model = LLaVAModel.__new__(LLaVAModel)
        param = MagicMock()
        param.stop_gradient = False
        vision_model = MagicMock()
        vision_model.parameters.return_value = [param]
        model.language_model = None
        model.vision_model = vision_model
        model.freeze(False, True, False)
        self.assertTrue(param.stop_gradient)

    def test_shared_embedding_or_output_weight_no_decoder(self):
        """Test shared_embedding_or_output_weight returns None without decoder."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_decoder = False
        result = model.shared_embedding_or_output_weight()
        self.assertIsNone(result)

    def test_shared_embedding_or_output_weight_with_decoder(self):
        """Test shared_embedding_or_output_weight delegates to language_model."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_decoder = True
        mock_weight = paddle.randn([100, 64])
        mock_lm = MagicMock()
        mock_lm.shared_embedding_or_output_weight.return_value = mock_weight
        model.language_model = mock_lm
        result = model.shared_embedding_or_output_weight()
        self.assertIs(result, mock_weight)


if __name__ == "__main__":
    unittest.main()
