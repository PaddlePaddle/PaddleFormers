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

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.transformers.embedding_utils import dist_gather_tensor_with_gradient

IS_SINGLE_CARD = paddle.distributed.get_world_size() <= 1


class TestDistGatherTensorWithGradient(unittest.TestCase):
    """Tests for dist_gather_tensor_with_gradient function."""

    def test_none_input_returns_none(self):
        result = dist_gather_tensor_with_gradient(None)
        self.assertIsNone(result)

    def test_single_card_returns_tensor(self):
        tensor = paddle.randn([2, 3])
        result = dist_gather_tensor_with_gradient(tensor)
        self.assertTrue(paddle.allclose(result, tensor))

    @unittest.skipIf(
        IS_SINGLE_CARD,
        "multi-card sharding test requires distributed environment",
    )
    @patch("paddleformers.transformers.embedding_utils.fleet")
    @patch("paddleformers.transformers.embedding_utils.paddle")
    def test_multi_card_sharding_and_data_both_1(self, mock_paddle, mock_fleet):
        mock_paddle.distributed.get_world_size.return_value = 2
        mock_hcg = MagicMock()
        mock_sharding_group = MagicMock()
        mock_sharding_group.nranks = 1
        mock_data_group = MagicMock()
        mock_data_group.nranks = 1
        mock_hcg.get_sharding_parallel_group.return_value = mock_sharding_group
        mock_hcg.get_data_parallel_group.return_value = mock_data_group
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg

        tensor = paddle.randn([2, 3])
        result = dist_gather_tensor_with_gradient(tensor)
        self.assertTrue(paddle.allclose(result, tensor))


if __name__ == "__main__":
    unittest.main()
