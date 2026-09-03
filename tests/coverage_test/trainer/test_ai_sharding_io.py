# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/utils/sharding_io.py"""

import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.trainer.utils.sharding_io import (
    ParameterNameRemapper,
    ShardingIO,
    filter_sharded_params,
    to_device,
)


class TestToDevice(unittest.TestCase):
    """Tests for to_device function."""

    def test_same_device(self):
        """Test that tensor on same device is returned unchanged."""
        tensor = paddle.randn([2, 3])
        result = to_device(tensor, paddle.CPUPlace())
        self.assertIsNotNone(result)

    def test_none_place_uses_default(self):
        """Test that None place uses default device."""
        tensor = paddle.randn([2, 3])
        result = to_device(tensor)
        self.assertIsNotNone(result)

    def test_string_place(self):
        """Test with string place."""
        tensor = paddle.randn([2, 3])
        result = to_device(tensor, "cpu")
        self.assertIsNotNone(result)


class TestFilterShardedParams(unittest.TestCase):
    """Tests for filter_sharded_params function."""

    def test_non_sharding_optimizer(self):
        """Test with non-sharding optimizer returns original dict."""
        state_dict = OrderedDict([("w1", paddle.randn([2, 2])), ("w2", paddle.randn([3, 3]))])
        optimizer = MagicMock()
        mock_group = MagicMock()
        mock_group.rank = 0
        mock_group.nranks = 1

        with patch("paddleformers.trainer.utils.sharding_io.reshard_util.is_sharding_opt", return_value=False):
            result = filter_sharded_params(state_dict, optimizer, mock_group)
            self.assertEqual(len(result), 2)


class TestParameterNameRemapper(unittest.TestCase):
    """Tests for ParameterNameRemapper class."""

    def test_init(self):
        """Test ParameterNameRemapper initialization."""
        old_mapping = {"layer1": "param_0"}
        new_mapping = {"layer1": "param_1"}
        remapper = ParameterNameRemapper(old_mapping, new_mapping, "/test/checkpoint")
        self.assertEqual(remapper.p_name_map["param_0"], "param_1")

    def test_init_missing_key_raises(self):
        """Test that missing key in new_mapping raises assertion."""
        old_mapping = {"layer1": "param_0"}
        new_mapping = {}  # Missing "layer1"
        with self.assertRaises(AssertionError):
            ParameterNameRemapper(old_mapping, new_mapping, "/test/checkpoint")

    def test_map_tensor(self):
        """Test _map_tensor method."""
        old_mapping = {"layer1": "param_0"}
        new_mapping = {"layer1": "param_1"}
        remapper = ParameterNameRemapper(old_mapping, new_mapping, "/test/checkpoint")
        tensor = paddle.randn([2, 2])
        tensor.name = "param_0_moment1_0"
        new_name, result = remapper._map_tensor(tensor, "param_0")
        self.assertEqual(new_name, "param_1_moment1_0")

    def test_remap_model_state(self):
        """Test remap_model_state method."""
        old_mapping = {"layer1": "param_0"}
        new_mapping = {"layer1": "param_1"}
        remapper = ParameterNameRemapper(old_mapping, new_mapping, "/test/checkpoint")
        tensor = paddle.randn([2, 2])
        tensor.name = "param_0"
        model_state = {"layer1": tensor}
        result = remapper.remap_model_state(model_state)
        self.assertIn("layer1", result)


class TestShardingIO(unittest.TestCase):
    """Tests for ShardingIO class."""

    def test_init_basic(self):
        """Test ShardingIO initialization without distributed."""
        args = MagicMock()
        args.use_hybrid_parallel = False
        model = MagicMock()

        with patch("paddle.distributed.get_world_size", return_value=1):
            io = ShardingIO(args, model)
            self.assertEqual(io.args, args)
            self.assertEqual(io.model, model)

    def test_set_optimizer(self):
        """Test set_optimizer method."""
        args = MagicMock()
        args.use_hybrid_parallel = False
        model = MagicMock()
        optimizer = MagicMock()

        with patch("paddle.distributed.get_world_size", return_value=1):
            io = ShardingIO(args, model)
            io.set_optimizer(optimizer)
            self.assertEqual(io.optimizer, optimizer)


if __name__ == "__main__":
    unittest.main()
