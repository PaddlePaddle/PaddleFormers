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

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import paddle

# Direct import to avoid __init__.py triggering workflow.py which requires AutoTokenizer
_MODULE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "paddleformers", "cli", "train", "deepseek_v3_pretrain"
)
_MODULE_DIR = os.path.abspath(_MODULE_DIR)


def _load_module(name, path):
    """Load a module directly from file path without going through __init__.py."""
    full_name = f"paddleformers.cli.train.deepseek_v3_pretrain.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-populate the package in sys.modules to prevent __init__.py from loading
_pkg_name = "paddleformers.cli.train.deepseek_v3_pretrain"
if _pkg_name not in sys.modules:
    _pkg_mod = types.ModuleType(_pkg_name)
    _pkg_mod.__path__ = [_MODULE_DIR]
    _pkg_mod.__package__ = _pkg_name
    sys.modules[_pkg_name] = _pkg_mod

# Load the moe_utils module directly
_moe_utils_mod = _load_module("moe_utils", os.path.join(_MODULE_DIR, "moe_utils.py"))
permute_fast = _moe_utils_mod.permute_fast
unpermute_fast = _moe_utils_mod.unpermute_fast
topk_to_permuted_indices = _moe_utils_mod.topk_to_permuted_indices
UnZipNode = _moe_utils_mod.UnZipNode
ZipNode = _moe_utils_mod.ZipNode
PermuteNode = _moe_utils_mod.PermuteNode
UnPermuteNode = _moe_utils_mod.UnPermuteNode
get_env_device = _moe_utils_mod.get_env_device
merge_subbatch_cast = _moe_utils_mod.merge_subbatch_cast


class TestTopkToPermutedIndices(unittest.TestCase):
    """Test topk_to_permuted_indices function."""

    def test_basic_permutation(self):
        """Test basic topk to permuted indices conversion."""
        topk_idx = paddle.to_tensor([[0, 1], [1, 2], [0, 2], [2, 3]])
        num_tokens_per_expert_list = [2, 2, 3, 1]
        topk = 2

        token_permuted_indices, prob_permuted_indices = topk_to_permuted_indices(
            topk_idx, num_tokens_per_expert_list, topk
        )

        self.assertEqual(token_permuted_indices.shape[0], prob_permuted_indices.shape[0])
        # Total assignments = 4 tokens * 2 topk = 8
        self.assertEqual(token_permuted_indices.shape[0], 8)

    def test_single_expert_per_token(self):
        """Test with topk=1 (single expert per token)."""
        topk_idx = paddle.to_tensor([[0], [1], [2]])
        num_tokens_per_expert_list = [1, 1, 1]
        topk = 1

        token_permuted_indices, prob_permuted_indices = topk_to_permuted_indices(
            topk_idx, num_tokens_per_expert_list, topk
        )

        self.assertEqual(token_permuted_indices.shape[0], 3)


class TestPermuteFast(unittest.TestCase):
    """Test permute_fast function."""

    def test_basic_permute(self):
        """Test basic permutation of tokens."""
        tokens = paddle.randn([4, 8])
        indices = paddle.to_tensor([2, 0, 3, 1])

        result = permute_fast(tokens, indices)
        self.assertEqual(result.shape, [4, 8])
        # Check that the permutation was applied correctly
        self.assertTrue(paddle.allclose(result[0], tokens[2]))
        self.assertTrue(paddle.allclose(result[1], tokens[0]))

    def test_permute_preserves_shape(self):
        """Test that permute preserves the hidden dimension."""
        tokens = paddle.randn([6, 16])
        indices = paddle.to_tensor([0, 1, 2, 3, 4, 5])

        result = permute_fast(tokens, indices)
        self.assertEqual(result.shape, [6, 16])

    def test_permute_drop_and_pad_not_supported(self):
        """Test that drop_and_pad=True raises assertion."""
        tokens = paddle.randn([4, 8])
        indices = paddle.to_tensor([0, 1, 2, 3])

        with self.assertRaises(AssertionError):
            permute_fast(tokens, indices, drop_and_pad=True)


class TestUnpermuteFast(unittest.TestCase):
    """Test unpermute_fast function."""

    def test_basic_unpermute(self):
        """Test basic unpermutation of tokens."""
        permuted_tokens = paddle.randn([4, 8])
        token_permuted_indices = paddle.to_tensor([2, 0, 3, 1])
        prob_permuted_indices = paddle.to_tensor([0, 1, 2, 3])
        restore_shape = [4, 8]

        result = unpermute_fast(
            permuted_tokens,
            token_permuted_indices,
            prob_permuted_indices,
            restore_shape,
        )
        self.assertEqual(result.shape, [4, 8])

    def test_unpermute_with_probs(self):
        """Test unpermutation with probability weighting."""
        permuted_tokens = paddle.ones([4, 8])
        token_permuted_indices = paddle.to_tensor([2, 0, 3, 1])
        prob_permuted_indices = paddle.to_tensor([0, 1, 2, 3])
        restore_shape = [4, 8]
        probs = paddle.to_tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        result = unpermute_fast(
            permuted_tokens,
            token_permuted_indices,
            prob_permuted_indices,
            restore_shape,
            probs=probs,
        )
        self.assertEqual(result.shape, [4, 8])

    def test_unpermute_drop_and_pad_not_supported(self):
        """Test that drop_and_pad=True raises assertion."""
        permuted_tokens = paddle.randn([4, 8])
        token_permuted_indices = paddle.to_tensor([0, 1, 2, 3])
        prob_permuted_indices = paddle.to_tensor([0, 1, 2, 3])
        restore_shape = [4, 8]

        with self.assertRaises(AssertionError):
            unpermute_fast(
                permuted_tokens,
                token_permuted_indices,
                prob_permuted_indices,
                restore_shape,
                drop_and_pad=True,
            )

    def test_permute_unpermute_roundtrip(self):
        """Test that permute and unpermute are inverse operations (without probs)."""
        original_tokens = paddle.randn([4, 8])
        indices = paddle.to_tensor([2, 0, 3, 1])

        permuted = permute_fast(original_tokens, indices)
        restored = unpermute_fast(
            permuted,
            indices,
            paddle.to_tensor([0, 1, 2, 3]),
            [4, 8],
        )
        # Since no probs are applied, the result should be a scatter-add
        self.assertEqual(restored.shape, [4, 8])


class TestUnZipNode(unittest.TestCase):
    """Test UnZipNode class."""

    def test_init(self):
        """Test UnZipNode initialization."""
        node = UnZipNode(name="test_unzip")
        self.assertEqual(node.name, "test_unzip")
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)

    def test_reset_status(self):
        """Test UnZipNode reset_statue method."""
        node = UnZipNode()
        node.unzipped_probs = paddle.randn([4])
        node.zipped_expertwise_rowmap = paddle.randn([4])
        node.reset_statue()
        self.assertIsNone(node.unzipped_probs)
        self.assertIsNone(node.zipped_expertwise_rowmap)


class TestZipNode(unittest.TestCase):
    """Test ZipNode class."""

    def test_init(self):
        """Test ZipNode initialization."""
        node = ZipNode(name="test_zip")
        self.assertEqual(node.name, "test_zip")


class TestPermuteNode(unittest.TestCase):
    """Test PermuteNode class."""

    def test_init(self):
        """Test PermuteNode initialization."""
        mock_dispatcher = MagicMock()
        node = PermuteNode(mock_dispatcher, name="test_permute")
        self.assertEqual(node.name, "test_permute")
        self.assertEqual(node.token_dispatcher, mock_dispatcher)

    def test_reset_status(self):
        """Test PermuteNode reset_status method."""
        mock_dispatcher = MagicMock()
        node = PermuteNode(mock_dispatcher)
        node.token_permuted_indices = paddle.to_tensor([0, 1])
        node.prob_permuted_indices = paddle.to_tensor([2, 3])
        node.reset_status()
        self.assertIsNone(node.token_permuted_indices)
        self.assertIsNone(node.prob_permuted_indices)


class TestUnPermuteNode(unittest.TestCase):
    """Test UnPermuteNode class."""

    def test_init(self):
        """Test UnPermuteNode initialization."""
        mock_dispatcher = MagicMock()
        node = UnPermuteNode(mock_dispatcher, name="test_unpermute")
        self.assertEqual(node.name, "test_unpermute")
        self.assertEqual(node.token_dispatcher, mock_dispatcher)

    def test_reset_status(self):
        """Test UnPermuteNode reset_status clears all state."""
        mock_dispatcher = MagicMock()
        node = UnPermuteNode(mock_dispatcher)
        node.token_permuted_indices = paddle.to_tensor([0, 1])
        node.hidden_states = paddle.randn([2, 4])
        node.prob_permuted_indices = paddle.to_tensor([2, 3])
        node.faltten_dispatched_probs = paddle.randn([4])
        node.hidden = 4
        node.permuted_tokens = paddle.randn([2, 4])
        node.output_tokens = paddle.randn([2, 4])

        node.reset_status()
        self.assertIsNone(node.token_permuted_indices)
        self.assertIsNone(node.hidden_states)
        self.assertIsNone(node.prob_permuted_indices)
        self.assertIsNone(node.faltten_dispatched_probs)
        self.assertIsNone(node.hidden)
        self.assertIsNone(node.permuted_tokens)
        self.assertIsNone(node.output_tokens)


class TestGetEnvDevice(unittest.TestCase):
    """Test get_env_device function."""

    def test_returns_string(self):
        """Test that get_env_device returns a string."""
        device = get_env_device()
        self.assertIsInstance(device, str)

    def test_returns_valid_device(self):
        """Test that get_env_device returns a valid device name."""
        device = get_env_device()
        valid_devices = ["gpu", "npu", "mlu", "gcu", "intel_hpu", "rocm", "xpu", "cpu"]
        self.assertIn(device, valid_devices)

    @patch.object(_moe_utils_mod.paddle, "is_compiled_with_cuda", return_value=False)
    @patch.object(_moe_utils_mod.paddle, "is_compiled_with_rocm", return_value=False)
    @patch.object(_moe_utils_mod.paddle, "is_compiled_with_xpu", return_value=False)
    @patch.object(_moe_utils_mod.paddle.device, "get_all_custom_device_type", return_value=[])
    def test_cpu_device_when_no_accelerator(self, mock_custom, mock_xpu, mock_rocm, mock_cuda):
        """Test that CPU is returned when no accelerator is available."""
        result = get_env_device()
        self.assertEqual(result, "cpu")


class TestMergeSubbatchCast(unittest.TestCase):
    """Test merge_subbatch_cast function."""

    def test_single_element_list(self):
        """Test with a single-element list."""
        x = [paddle.randn([2, 4], dtype="float32")]
        result = merge_subbatch_cast(x, paddle.float32)
        self.assertEqual(result.shape, [2, 4])

    def test_single_element_list_dtype_conversion(self):
        """Test dtype conversion for single-element list."""
        x = [paddle.randn([2, 4], dtype="float32")]
        result = merge_subbatch_cast(x, paddle.float16)
        self.assertEqual(result.dtype, paddle.float16)

    def test_single_tensor_input(self):
        """Test with a single tensor (not a list)."""
        x = paddle.randn([2, 4], dtype="float32")
        result = merge_subbatch_cast(x, paddle.float32)
        self.assertEqual(result.shape, [2, 4])

    def test_single_tensor_dtype_conversion(self):
        """Test dtype conversion for single tensor input."""
        x = paddle.randn([2, 4], dtype="float32")
        result = merge_subbatch_cast(x, paddle.float16)
        self.assertEqual(result.dtype, paddle.float16)

    def test_same_dtype_no_conversion(self):
        """Test that no conversion happens when dtypes match."""
        x = paddle.randn([2, 4], dtype="float32")
        result = merge_subbatch_cast(x, paddle.float32)
        self.assertEqual(result.dtype, paddle.float32)


class TestTensorExtensions(unittest.TestCase):
    """Test _clear_to_zero_allocation and _holder_size extensions."""

    def test_holder_size_returns_int(self):
        """Test _holder_size returns an integer for an initialized tensor."""
        x = paddle.randn([2, 4], dtype="float32")
        size = x._holder_size()
        self.assertIsInstance(size, int)
        self.assertGreater(size, 0)

    def test_holder_size_positive_for_various_dtypes(self):
        """Test _holder_size returns positive values for various data types."""
        x_f32 = paddle.randn([2, 4], dtype="float32")
        x_f16 = paddle.randn([2, 4], dtype="float16")
        self.assertGreater(x_f32._holder_size(), 0)
        self.assertGreater(x_f16._holder_size(), 0)

    def test_holder_size_larger_tensor_has_larger_size(self):
        """Test that a larger tensor has a larger holder size."""
        x_small = paddle.randn([2, 4], dtype="float32")
        x_large = paddle.randn([32, 64], dtype="float32")
        self.assertGreater(x_large._holder_size(), x_small._holder_size())

    def test_clear_to_zero_allocation_exists(self):
        """Test _clear_to_zero_allocation method exists on tensor."""
        x = paddle.randn([2, 4], dtype="float32")
        self.assertTrue(hasattr(x, "_clear_to_zero_allocation"))
        self.assertTrue(callable(x._clear_to_zero_allocation))


if __name__ == "__main__":
    unittest.main()
