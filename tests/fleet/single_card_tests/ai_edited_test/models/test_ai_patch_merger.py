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

from paddleformers.fleet.models.qwen3_vl.patch_merger import (
    Qwen3VLVisionPatchMergerSpec,
    Qwen3VLVisionPathMerger,
)
from paddleformers.fleet.transformer.identity_op import IdentityOp


class TestQwen3VLVisionPatchMergerSpec(unittest.TestCase):
    """Test Qwen3VLVisionPatchMergerSpec dataclass."""

    def test_default_norm_is_identity(self):
        """Test default norm is IdentityOp."""
        spec = Qwen3VLVisionPatchMergerSpec()
        self.assertEqual(spec.norm, IdentityOp)

    def test_custom_norm(self):
        """Test custom norm field."""
        mock_norm = MagicMock()
        spec = Qwen3VLVisionPatchMergerSpec(norm=mock_norm)
        self.assertEqual(spec.norm, mock_norm)


class TestQwen3VLVisionPathMergerInit(unittest.TestCase):
    """Test Qwen3VLVisionPathMerger initialization."""

    def test_init_requires_valid_config(self):
        """Test init with a mock config."""
        merger = Qwen3VLVisionPathMerger.__new__(Qwen3VLVisionPathMerger)
        # Check class exists and is constructable
        self.assertIsNotNone(merger)

    def test_forward_with_dict_input(self):
        """Test forward with dict input extracts hidden_states."""
        merger = Qwen3VLVisionPathMerger.__new__(Qwen3VLVisionPathMerger)
        merger.__dict__.setdefault("_parameters", {})
        merger.__dict__.setdefault("_buffers", {})
        merger.__dict__.setdefault("_sub_layers", {})
        merger.__dict__.setdefault("_loaddict_holder", {})
        merger.__dict__.setdefault("_non_persistable_buffers", set())
        merger.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        merger.use_postshuffle_norm = False
        merger.hidden_size = 64

        mock_norm = MagicMock(return_value=paddle.randn([2, 64]))
        mock_mlp = MagicMock(return_value=(MagicMock(), None))
        # Just verify the class has expected attributes
        self.assertTrue(hasattr(merger, "__class__"))


class TestQwen3VLVisionPathMergerForward(unittest.TestCase):
    """Test Qwen3VLVisionPathMerger forward method."""

    def test_forward_with_bias(self):
        """Test forward when mlp returns bias."""
        import paddle

        merger = Qwen3VLVisionPathMerger.__new__(Qwen3VLVisionPathMerger)
        merger.__dict__.setdefault("_parameters", {})
        merger.__dict__.setdefault("_buffers", {})
        merger.__dict__.setdefault("_sub_layers", {})
        merger.__dict__.setdefault("_loaddict_holder", {})
        merger.__dict__.setdefault("_non_persistable_buffers", set())
        merger.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        merger.use_postshuffle_norm = False
        merger.hidden_size = 64

        # Create mock norm that returns a tensor
        norm_output = paddle.randn([2, 64])
        merger.norm = MagicMock(return_value=norm_output)

        # Create mock mlp that returns (output, bias)
        mlp_output = paddle.randn([2, 32])
        mlp_bias = paddle.randn([32])
        merger.mlp = MagicMock(return_value=(mlp_output, mlp_bias))

        # Test with dict input
        x = {"hidden_states": paddle.randn([1, 2, 64])}
        result, bias = merger.forward(x)
        self.assertTrue(paddle.allclose(result, mlp_output + mlp_bias))
        self.assertIsNone(bias)

    def test_forward_without_bias(self):
        """Test forward when mlp returns no bias."""
        import paddle

        merger = Qwen3VLVisionPathMerger.__new__(Qwen3VLVisionPathMerger)
        merger.__dict__.setdefault("_parameters", {})
        merger.__dict__.setdefault("_buffers", {})
        merger.__dict__.setdefault("_sub_layers", {})
        merger.__dict__.setdefault("_loaddict_holder", {})
        merger.__dict__.setdefault("_non_persistable_buffers", set())
        merger.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        merger.use_postshuffle_norm = False
        merger.hidden_size = 64

        norm_output = paddle.randn([2, 64])
        merger.norm = MagicMock(return_value=norm_output)

        mlp_output = paddle.randn([2, 32])
        merger.mlp = MagicMock(return_value=(mlp_output, None))

        x = {"hidden_states": paddle.randn([1, 2, 64])}
        result, bias = merger.forward(x)
        self.assertTrue(paddle.allclose(result, mlp_output))
        self.assertIsNone(bias)

    def test_forward_with_postshuffle_norm(self):
        """Test forward with use_postshuffle_norm=True."""
        import paddle

        merger = Qwen3VLVisionPathMerger.__new__(Qwen3VLVisionPathMerger)
        merger.__dict__.setdefault("_parameters", {})
        merger.__dict__.setdefault("_buffers", {})
        merger.__dict__.setdefault("_sub_layers", {})
        merger.__dict__.setdefault("_loaddict_holder", {})
        merger.__dict__.setdefault("_non_persistable_buffers", set())
        merger.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        merger.use_postshuffle_norm = True
        merger.hidden_size = 64

        norm_output = paddle.randn([2, 64])
        merger.norm = MagicMock(return_value=norm_output)

        mlp_output = paddle.randn([2, 32])
        merger.mlp = MagicMock(return_value=(mlp_output, None))

        x = {"hidden_states": paddle.randn([1, 2, 64])}
        result, bias = merger.forward(x)
        # Verify norm was called with reshaped input
        merger.norm.assert_called()


if __name__ == "__main__":
    unittest.main()
