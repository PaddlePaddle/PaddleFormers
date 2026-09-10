# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/utils/reshard/common.py"""

import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.trainer.utils.reshard.common import (
    SHARDING_STRATEGY_V1,
    SHARDING_STRATEGY_V2,
    NodeModelState,
    convert_opt_name_to_tname,
    is_sharding_opt,
    merge_model_state,
    merge_opt_state,
    split_model_state,
    split_opt_state,
    split_structure_name_mapping,
)


class TestShardingStrategyConstants(unittest.TestCase):
    """Tests for sharding strategy constants."""

    def test_v1_constant(self):
        """Test V1 constant value."""
        self.assertEqual(SHARDING_STRATEGY_V1, "ShardingV1")

    def test_v2_constant(self):
        """Test V2 constant value."""
        self.assertEqual(SHARDING_STRATEGY_V2, "ShardingV2")


class TestIsShardingOpt(unittest.TestCase):
    """Tests for is_sharding_opt function."""

    def test_non_sharding_optimizer_returns_false(self):
        """Test that a regular optimizer returns False."""
        optimizer = MagicMock(spec=[])
        with patch("paddleformers.trainer.utils.reshard.common.unwrap_optimizer", return_value=None):
            result = is_sharding_opt(optimizer)
            self.assertFalse(result)


class TestConvertOptNameToTname(unittest.TestCase):
    """Tests for convert_opt_name_to_tname function."""

    def test_basic_conversion(self):
        """Test basic optimizer name to tensor name conversion."""
        tensor_names = ["linear.weight"]
        opt_names = ["linear.weight_moment1_0", "linear.weight_moment2_0"]
        result = convert_opt_name_to_tname(tensor_names, opt_names)
        self.assertEqual(result["linear.weight_moment1_0"], "linear.weight")
        self.assertEqual(result["linear.weight_moment2_0"], "linear.weight")

    def test_beta_pow_conversion(self):
        """Test beta power accumulator name conversion."""
        tensor_names = ["linear.weight"]
        opt_names = ["linear.weight_beta1_pow_acc_0", "linear.weight_beta2_pow_acc_0"]
        result = convert_opt_name_to_tname(tensor_names, opt_names)
        self.assertEqual(result["linear.weight_beta1_pow_acc_0"], "linear.weight")
        self.assertEqual(result["linear.weight_beta2_pow_acc_0"], "linear.weight")

    def test_fp32_master_conversion(self):
        """Test fp32 master weight name conversion."""
        tensor_names = ["linear.weight"]
        opt_names = ["linear.weight_fp32_master_0_moment1_0"]
        result = convert_opt_name_to_tname(tensor_names, opt_names)
        self.assertEqual(result["linear.weight_fp32_master_0_moment1_0"], "linear.weight")

    def test_multiple_params(self):
        """Test conversion with multiple parameters."""
        tensor_names = ["layer1.weight", "layer2.weight"]
        opt_names = ["layer1.weight_moment1_0", "layer2.weight_moment2_0"]
        result = convert_opt_name_to_tname(tensor_names, opt_names)
        self.assertEqual(len(result), 2)


class TestNodeModelState(unittest.TestCase):
    """Tests for NodeModelState class."""

    def test_init(self):
        """Test NodeModelState initialization."""
        state = NodeModelState(group=None)
        self.assertIsNotNone(state.model_weights)
        self.assertIsNotNone(state.opt_state)
        self.assertIsNotNone(state.master_weights)
        self.assertIsNone(state.lr_scheduler)

    def test_add_weight(self):
        """Test adding a weight."""
        state = NodeModelState(group=None)
        tensor = paddle.randn([4, 4])
        state.add_weight("test.weight", tensor)
        self.assertIn("test.weight", state.model_weights)

    def test_add_weights_with_rank(self):
        """Test adding weights with rank."""
        state = NodeModelState(group=None)
        model_state_dict = {"w1": paddle.randn([2, 2]), "w2": paddle.randn([3, 3])}
        state.add_weights(model_state_dict, rank=0)
        self.assertIn(("w1", 0), state.model_weights)
        self.assertIn(("w2", 0), state.model_weights)

    def test_set_weights(self):
        """Test setting weights."""
        state = NodeModelState(group=None)
        new_weights = OrderedDict([("w1", paddle.randn([2, 2]))])
        state.set_weights(new_weights)
        self.assertEqual(state.model_weights, new_weights)

    def test_add_opt(self):
        """Test adding optimizer state."""
        state = NodeModelState(group=None)
        tensor = paddle.randn([4, 4])
        state.add_opt("moment1", tensor)
        self.assertIn("moment1", state.opt_state)

    def test_add_opts_with_master_weights(self):
        """Test adding optimizer state dict with master_weights."""
        state = NodeModelState(group=None)
        opt_dict = OrderedDict(
            [
                ("moment1", paddle.randn([4, 4])),
                ("master_weights", {"mw1": paddle.randn([4, 4])}),
                ("LR_Scheduler", MagicMock()),
            ]
        )
        state.add_opts(opt_dict)
        self.assertIn("moment1", state.opt_state)
        self.assertIn("mw1", state.master_weights)
        self.assertIsNotNone(state.lr_scheduler)

    def test_add_master_weight(self):
        """Test adding master weight."""
        state = NodeModelState(group=None)
        tensor = paddle.randn([4, 4])
        state.add_master_weight("mw1", tensor)
        self.assertIn("mw1", state.master_weights)

    def test_set_lr_scheduler(self):
        """Test setting LR scheduler."""
        state = NodeModelState(group=None)
        scheduler = MagicMock()
        state.set_lr_scheduler(scheduler)
        self.assertEqual(state.lr_scheduler, scheduler)

    def test_set_lr_scheduler_none(self):
        """Test setting None LR scheduler does not override."""
        state = NodeModelState(group=None)
        scheduler = MagicMock()
        state.set_lr_scheduler(scheduler)
        state.set_lr_scheduler(None)
        self.assertEqual(state.lr_scheduler, scheduler)

    def test_get_opt_state_dict(self):
        """Test getting optimizer state dict."""
        state = NodeModelState(group=None)
        state.add_opt("moment1", paddle.randn([4, 4]))
        state.set_lr_scheduler(MagicMock())
        result = state.get_opt_state_dict()
        self.assertIn("moment1", result)
        self.assertIn("LR_Scheduler", result)
        self.assertIn("master_weights", result)

    def test_group_property(self):
        """Test group property."""
        mock_group = MagicMock()
        state = NodeModelState(group=mock_group)
        self.assertEqual(state.group, mock_group)

    def test_duplicate_key_raises(self):
        """Test that adding duplicate key raises assertion."""
        state = NodeModelState(group=None)
        tensor = paddle.randn([4, 4])
        state.add_weight("test.weight", tensor)
        with self.assertRaises(AssertionError):
            state.add_weight("test.weight", tensor)


class TestSplitModelState(unittest.TestCase):
    """Tests for split_model_state function."""

    def test_basic_split(self):
        """Test splitting model state by group."""
        model_state = OrderedDict([("w1", paddle.randn([2, 2])), ("w2", paddle.randn([3, 3]))])
        mock_group = MagicMock()
        mock_group.id = 0
        mock_getter = MagicMock()
        mock_getter.get_group.return_value = mock_group
        result = split_model_state(model_state, mock_getter)
        self.assertIn(0, result)
        self.assertIn("w1", result[0])
        self.assertIn("w2", result[0])


class TestMergeModelState(unittest.TestCase):
    """Tests for merge_model_state function."""

    def test_basic_merge(self):
        """Test merging model states."""
        state1 = OrderedDict([("w1", paddle.randn([2, 2]))])
        state2 = OrderedDict([("w2", paddle.randn([3, 3]))])
        model_state_map = {0: state1, 1: state2}
        result = merge_model_state(model_state_map)
        self.assertIn("w1", result)
        self.assertIn("w2", result)


class TestSplitOptState(unittest.TestCase):
    """Tests for split_opt_state function."""

    def test_basic_split(self):
        """Test splitting optimizer state by group."""
        opt_state = OrderedDict(
            [
                ("moment1", paddle.randn([2, 2])),
                ("master_weights", OrderedDict([("mw1", paddle.randn([2, 2]))])),
                ("LR_Scheduler", MagicMock()),
            ]
        )
        mock_group = MagicMock()
        mock_group.id = 0
        mock_getter = MagicMock()
        mock_getter.get_group.return_value = mock_group
        result = split_opt_state(opt_state, mock_getter)
        self.assertIn(0, result)
        self.assertIn("moment1", result[0])
        self.assertIn("master_weights", result[0])


class TestMergeOptState(unittest.TestCase):
    """Tests for merge_opt_state function."""

    def test_basic_merge(self):
        """Test merging optimizer states."""
        state1 = OrderedDict(
            [
                ("moment1", paddle.randn([2, 2])),
                ("master_weights", OrderedDict([("mw1", paddle.randn([2, 2]))])),
                ("LR_Scheduler", None),
            ]
        )
        state2 = OrderedDict(
            [
                ("moment2", paddle.randn([3, 3])),
                ("master_weights", OrderedDict([("mw2", paddle.randn([3, 3]))])),
                ("LR_Scheduler", MagicMock()),
            ]
        )
        opt_state_map = {0: state1, 1: state2}
        result = merge_opt_state(opt_state_map)
        self.assertIn("moment1", result)
        self.assertIn("moment2", result)
        self.assertIn("master_weights", result)
        self.assertIsNotNone(result["LR_Scheduler"])


class TestSplitStructureNameMapping(unittest.TestCase):
    """Tests for split_structure_name_mapping function."""

    def test_basic_split(self):
        """Test splitting structure name mapping by group."""
        mapping = OrderedDict([("w1", "param_0"), ("w2", "param_1")])
        mock_group = MagicMock()
        mock_group.id = 0
        mock_getter = MagicMock()
        mock_getter.get_group.return_value = mock_group
        result = split_structure_name_mapping(mapping, mock_getter)
        self.assertIn(0, result)
        self.assertEqual(result[0]["w1"], "param_0")


if __name__ == "__main__":
    unittest.main()
