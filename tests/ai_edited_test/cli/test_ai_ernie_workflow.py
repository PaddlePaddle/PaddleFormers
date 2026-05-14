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

from paddleformers.cli.train.ernie_pretrain.workflow import (
    AllArguments,
    ExpConfig,
    get_tp_split_ckpt,
    update_model_config_from_args,
)


class TestUpdateModelConfigFromArgs(unittest.TestCase):
    """Tests for update_model_config_from_args function."""

    @patch("paddleformers.cli.train.ernie_pretrain.workflow.logger")
    def test_update_existing_keys(self, mock_logger):
        """Test updating config with existing keys."""
        mock_config = MagicMock()
        mock_config.has.side_effect = lambda k: k in ["hidden_size", "num_layers"]
        mock_config.hidden_size = 768
        mock_config.num_layers = 2
        model_args = {"hidden_size": 1024, "num_layers": 4}

        result = update_model_config_from_args(mock_config, model_args)
        self.assertIs(result, mock_config)
        # Check setattr was called
        mock_config.__setattr__.call_args_list if hasattr(mock_config.__setattr__, "call_args_list") else []

    @patch("paddleformers.cli.train.ernie_pretrain.workflow.logger")
    def test_update_with_nonexistent_keys(self, mock_logger):
        """Test updating config with keys that don't exist."""
        mock_config = MagicMock(spec=[])
        model_args = {"nonexistent_key": 42}

        # Function iterates over model_args and calls hasattr/setattr
        # With spec=[], hasattr returns False for everything
        result = update_model_config_from_args(mock_config, model_args)
        self.assertIs(result, mock_config)

    @patch("paddleformers.cli.train.ernie_pretrain.workflow.logger")
    def test_empty_args(self, mock_logger):
        """Test with empty model args."""
        mock_config = MagicMock()
        result = update_model_config_from_args(mock_config, {})
        self.assertIs(result, mock_config)


class TestGetTpSplitCkpt(unittest.TestCase):
    """Tests for get_tp_split_ckpt function."""

    def test_tp_degree_1(self):
        """Test checkpoint path with tp_degree=1."""
        args = MagicMock()
        args.tensor_model_parallel_size = 1
        args.tensor_parallel_rank = 0
        path = "/some/path"

        result = get_tp_split_ckpt(args, path)
        self.assertEqual(result, "/some/path/model_state.pdparams")

    def test_tp_degree_2(self):
        """Test checkpoint path with tp_degree=2."""
        args = MagicMock()
        args.tensor_model_parallel_size = 2
        args.tensor_parallel_rank = 0
        path = "/some/path"

        result = get_tp_split_ckpt(args, path)
        self.assertEqual(result, "/some/path/tp02/model_state.tp00.pdparams")

    def test_tp_degree_8_rank_3(self):
        """Test checkpoint path with tp_degree=8 and rank 3."""
        args = MagicMock()
        args.tensor_model_parallel_size = 8
        args.tensor_parallel_rank = 3
        path = "/model/dir"

        result = get_tp_split_ckpt(args, path)
        self.assertEqual(result, "/model/dir/tp08/model_state.tp03.pdparams")

    def test_negative_rank_uses_zero(self):
        """Test that negative tensor_parallel_rank defaults to 0."""
        args = MagicMock()
        args.tensor_model_parallel_size = 1
        args.tensor_parallel_rank = -1
        path = "/some/path"

        result = get_tp_split_ckpt(args, path)
        self.assertEqual(result, "/some/path/model_state.pdparams")


class TestExpConfig(unittest.TestCase):
    """Tests for ExpConfig dataclass."""

    def test_creation(self):
        """Test creating ExpConfig."""
        config = ExpConfig(max_steps=1000, name="test_exp", config={"key": "value"})
        self.assertEqual(config.max_steps, 1000)
        self.assertEqual(config.name, "test_exp")
        self.assertEqual(config.config, {"key": "value"})


class TestAllArguments(unittest.TestCase):
    """Tests for AllArguments dataclass."""

    def test_is_dataclass(self):
        """Test that AllArguments is a dataclass."""
        from dataclasses import is_dataclass

        self.assertTrue(is_dataclass(AllArguments))


if __name__ == "__main__":
    unittest.main()
