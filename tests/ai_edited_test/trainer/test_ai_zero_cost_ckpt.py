# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
"""Tests for trainer/utils/zero_cost_checkpoint.py"""

import hashlib
import unittest
from unittest.mock import MagicMock

import paddle

from paddleformers.trainer.utils.zero_cost_checkpoint import (
    ZCCTaskType,
    ZCCWorkerStatus,
    _unwrap_opt_for_fused_states,
    md5,
    sharded_state_dict_compatibility,
)


class TestZCCTaskType(unittest.TestCase):
    """Tests for ZCCTaskType enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        self.assertEqual(ZCCTaskType.UPDATE.value, 0)
        self.assertEqual(ZCCTaskType.PREPARE.value, 1)
        self.assertEqual(ZCCTaskType.OFFLOAD.value, 2)
        self.assertEqual(ZCCTaskType.FINISH.value, 3)
        self.assertEqual(ZCCTaskType.SET_EMA_STATE_DICT.value, 5)


class TestZCCWorkerStatus(unittest.TestCase):
    """Tests for ZCCWorkerStatus enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        self.assertEqual(ZCCWorkerStatus.IDLE.value, 0)
        self.assertEqual(ZCCWorkerStatus.OFFLOADING.value, 1)
        self.assertEqual(ZCCWorkerStatus.DUMPING.value, 2)
        self.assertEqual(ZCCWorkerStatus.ERROR.value, 3)


class TestMd5(unittest.TestCase):
    """Tests for md5 function."""

    def test_basic_md5(self):
        """Test that md5 returns correct hash for a tensor."""
        tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        result = md5(tensor)
        expected = hashlib.md5(tensor.numpy().tobytes()).hexdigest()
        self.assertEqual(result, expected)

    def test_consistent_md5(self):
        """Test that md5 is consistent for same tensor."""
        tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        result1 = md5(tensor)
        result2 = md5(tensor)
        self.assertEqual(result1, result2)


class TestUnwrapOptForFusedStates(unittest.TestCase):
    """Tests for _unwrap_opt_for_fused_states function."""

    def test_no_inner_opt(self):
        """Test with optimizer that has no _inner_opt."""
        optimizer = MagicMock(spec=[])
        result = _unwrap_opt_for_fused_states(optimizer)
        self.assertEqual(result, optimizer)

    def test_single_inner_opt(self):
        """Test unwrapping one level of _inner_opt."""
        inner_opt = MagicMock(spec=[])
        optimizer = MagicMock(spec=["_inner_opt"])
        optimizer._inner_opt = inner_opt
        result = _unwrap_opt_for_fused_states(optimizer)
        self.assertEqual(result, inner_opt)

    def test_nested_inner_opt_stops_at_sharding(self):
        """Test unwrapping stops at DygraphShardingOptimizer."""
        from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
            DygraphShardingOptimizer,
        )

        sharding_opt = MagicMock(spec=DygraphShardingOptimizer)
        sharding_opt.__class__ = DygraphShardingOptimizer
        outer_opt = MagicMock(spec=["_inner_opt"])
        outer_opt._inner_opt = sharding_opt
        result = _unwrap_opt_for_fused_states(outer_opt)
        self.assertEqual(result, sharding_opt)


class TestShardedStateDictCompatibility(unittest.TestCase):
    """Tests for sharded_state_dict_compatibility decorator."""

    def test_normal_dict_passthrough(self):
        """Test that normal dict is passed through unchanged."""

        @sharded_state_dict_compatibility
        def test_func(state_dict):
            return state_dict

        state_dict = {"key1": paddle.randn([2, 3])}
        result = test_func(state_dict)
        self.assertEqual(set(result.keys()), {"key1"})

    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves the original function name."""

        @sharded_state_dict_compatibility
        def my_function(state_dict):
            return state_dict

        self.assertEqual(my_function.__name__, "my_function")


if __name__ == "__main__":
    unittest.main()
