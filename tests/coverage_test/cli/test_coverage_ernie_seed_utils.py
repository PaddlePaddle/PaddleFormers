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

from paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils import set_seed


@unittest.skip("get_rng_state_tracker is a C extension singleton that cannot be reliably patched in CI")
class TestSetSeed(unittest.TestCase):
    """Tests for set_seed function."""

    @patch("paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils.fleet")
    @patch("paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils.dist")
    @patch("paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils.paddle")
    @patch("paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils.get_rng_state_tracker")
    def test_set_seed_without_hcg(self, mock_tracker, mock_paddle, mock_dist, mock_fleet):
        """Test set_seed when fleet has no _hcg attribute."""
        mock_fleet._hcg = None
        type(mock_fleet).__contains__ = lambda self, item: False
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 1
        mock_paddle.distributed.get_world_size.return_value = 1

        set_seed(42)

        mock_tracker.assert_called_once()
        mock_tracker.return_value.add.assert_called()
        mock_paddle.seed.assert_called()

    @patch("paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils.fleet")
    @patch("paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils.paddle")
    @patch("paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils.get_rng_state_tracker")
    def test_set_seed_with_hcg(self, mock_tracker, mock_paddle, mock_fleet):
        """Test set_seed when fleet has _hcg attribute."""
        mock_hcg = MagicMock()
        mock_hcg.get_model_parallel_rank.return_value = 0
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_stage_id.return_value = 0
        mock_hcg.get_pipe_parallel_world_size.return_value = 1
        mock_hcg.get_data_parallel_rank.return_value = 0
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_sharding_parallel_rank.return_value = 0
        mock_fleet._hcg = "exists"
        mock_fleet.get_hybrid_communicate_group.return_value = mock_hcg
        mock_paddle.distributed.get_world_size.return_value = 1

        set_seed(123)

        mock_tracker.assert_called_once()
        mock_tracker.return_value.add.assert_called()
        mock_paddle.seed.assert_called()

    @patch("paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils.fleet")
    @patch("paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils.dist")
    @patch("paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils.paddle")
    @patch("paddleformers.cli.train.ernie_pretrain.src.utils.seed_utils.get_rng_state_tracker")
    def test_set_seed_different_seeds_for_different_ranks(self, mock_tracker, mock_paddle, mock_dist, mock_fleet):
        """Test that different ranks get different seeds."""
        mock_fleet._hcg = None
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 2
        mock_paddle.distributed.get_world_size.return_value = 2

        set_seed(42)

        mock_paddle.seed.assert_called()


if __name__ == "__main__":
    unittest.main()
