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
from unittest.mock import MagicMock

import paddleformers.cli.train.ernie_pretrain.models.sequence_parallel_utils as sp_mod
from paddleformers.cli.train.ernie_pretrain.models.sequence_parallel_utils import (
    get_hcg,
    is_fused_matmul_bias_supported,
    is_sequence_parallel_parameter,
    mark_as_sequence_parallel_parameter,
)


class TestGetHcg(unittest.TestCase):
    """Tests for get_hcg function."""

    def setUp(self):
        self._original_fleet = sp_mod.fleet
        mock_fleet = MagicMock()
        mock_hcg = MagicMock()
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg
        self._mock_fleet = mock_fleet
        self._mock_hcg = mock_hcg
        sp_mod.fleet = mock_fleet

    def tearDown(self):
        sp_mod.fleet = self._original_fleet

    def test_get_hcg_calls_fleet(self):
        """Test that get_hcg calls fleet.get_hybrid_communicate_group."""
        result = get_hcg()
        self.assertEqual(result, self._mock_hcg)


class TestIsFusedMatmulBiasSupported(unittest.TestCase):
    """Tests for is_fused_matmul_bias_supported function."""

    def setUp(self):
        self._original_paddle = sp_mod.paddle
        mock_paddle = MagicMock()
        sp_mod.paddle = mock_paddle
        self._mock_paddle = mock_paddle

    def tearDown(self):
        sp_mod.paddle = self._original_paddle

    def test_returns_false_on_cpu(self):
        """Test that function returns False when not compiled with CUDA."""
        self._mock_paddle.is_compiled_with_cuda.return_value = False
        result = is_fused_matmul_bias_supported()
        self.assertFalse(result)

    def test_returns_false_on_rocm(self):
        """Test that function returns False when compiled with ROCm."""
        self._mock_paddle.is_compiled_with_cuda.return_value = True
        self._mock_paddle.is_compiled_with_rocm.return_value = True
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
