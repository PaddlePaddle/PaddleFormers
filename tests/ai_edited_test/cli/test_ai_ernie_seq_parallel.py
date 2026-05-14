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

import unittest
from unittest.mock import MagicMock, patch

from paddleformers.cli.train.ernie_pretrain.models.sequence_parallel_utils import (
    get_hcg,
    is_fused_matmul_bias_supported,
    is_sequence_parallel_parameter,
    mark_as_sequence_parallel_parameter,
)


class TestGetHcg(unittest.TestCase):
    """Tests for get_hcg function."""

    @patch("paddleformers.cli.train.ernie_pretrain.models.sequence_parallel_utils.fleet")
    def test_get_hcg_calls_fleet(self, mock_fleet):
        """Test that get_hcg calls fleet.get_hybrid_communicate_group."""
        mock_hcg = MagicMock()
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg
        result = get_hcg()
        self.assertEqual(result, mock_hcg)


class TestIsFusedMatmulBiasSupported(unittest.TestCase):
    """Tests for is_fused_matmul_bias_supported function."""

    @patch("paddleformers.cli.train.ernie_pretrain.models.sequence_parallel_utils.paddle")
    def test_returns_false_on_cpu(self, mock_paddle):
        """Test that function returns False when not compiled with CUDA."""
        mock_paddle.is_compiled_with_cuda.return_value = False
        result = is_fused_matmul_bias_supported()
        self.assertFalse(result)

    @patch("paddleformers.cli.train.ernie_pretrain.models.sequence_parallel_utils.paddle")
    def test_returns_false_on_rocm(self, mock_paddle):
        """Test that function returns False when compiled with ROCm."""
        mock_paddle.is_compiled_with_cuda.return_value = True
        mock_paddle.is_compiled_with_rocm.return_value = True
        result = is_fused_matmul_bias_supported()
        self.assertFalse(result)


class TestSequenceParallelParameter(unittest.TestCase):
    """Tests for mark_as_sequence_parallel_parameter and is_sequence_parallel_parameter."""

    def test_mark_and_check(self):
        """Test marking a parameter and checking it."""
        param = MagicMock()
        mark_as_sequence_parallel_parameter(param)
        self.assertTrue(is_sequence_parallel_parameter(param))

    def test_unmarked_parameter(self):
        """Test that unmarked parameter returns False."""
        param = MagicMock(spec=[])
        self.assertFalse(is_sequence_parallel_parameter(param))

    def test_explicit_false(self):
        """Test parameter with sequence_parallel=False."""
        param = MagicMock()
        param.sequence_parallel = False
        self.assertFalse(is_sequence_parallel_parameter(param))


if __name__ == "__main__":
    unittest.main()
