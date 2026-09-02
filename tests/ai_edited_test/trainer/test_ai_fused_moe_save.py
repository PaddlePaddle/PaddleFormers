# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
"""Focused tests for fused-MoE save at sharding_parallel_size=1 and HF export cadence."""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from paddleformers.trainer.trainer_callback import (
    DefaultFlowCallback,
    TrainerControl,
    TrainerState,
)
from paddleformers.trainer.trainer_utils import IntervalStrategy
from paddleformers.trainer.training_args import TrainingArguments


class TestFusedMoEShardingOneSaveAssert(unittest.TestCase):
    """The fusion + sharding=1 + save_strategy=steps combination must not abort."""

    def test_post_init_parallel_degree_does_not_assert_at_sharding_one(self):
        source = inspect.getsource(TrainingArguments._post_init_parallel_degree)
        self.assertNotIn("please set moe_expert_fusion to false", source)
        self.assertNotIn("Checkpoint will fail to save when moe_expert_fusion is true", source)
        self.assertIn("keeps 3-D grouped_gemm weights", source)
        self.assertNotIn("sharding_parallel_size=%s", source)


class TestRestoreFusedExpert3DLayout(unittest.TestCase):
    def test_restores_flattened_grouped_gemm_weight(self):
        import paddle
        from paddle.distributed import ShardedWeight

        from paddleformers.trainer.trainer import restore_fused_expert_3d_layout

        key = "model.layers.3.mlp.grouped_gemm_experts.weight1"
        param = paddle.zeros([2, 4, 6], dtype="float32")
        flat = param.reshape([8, 6])
        shard = ShardedWeight(
            key=key,
            local_tensor=flat,
            local_shape=tuple(flat.shape),
            global_shape=tuple(flat.shape),
            global_offset=(0, 0),
        )
        model = MagicMock()
        model.named_parameters.return_value = [(key, param)]

        restore_fused_expert_3d_layout(model, {key: shard})

        self.assertEqual(tuple(shard.local_tensor.shape), (2, 4, 6))
        self.assertEqual(shard.local_shape, (2, 4, 6))
        self.assertEqual(shard.global_shape, (2, 4, 6))

    def test_optimizer_path_passes_optimizer_into_restore(self):
        source = inspect.getsource(
            __import__("paddleformers.trainer.trainer", fromlist=["Trainer"]).Trainer._save_flex_optimizer_state
        )
        self.assertIn("optimizer=self.optimizer", source)
        # Rank-invariant skip: fused experts live only on some PP stages.
        self.assertIn('getattr(self.args, "moe_expert_fusion", False)', source)
        self.assertIn("Fused-expert optimizer FlexCheckpoint save is not supported", source)


class TestDefaultFlowCallbackSaveHf(unittest.TestCase):
    def test_save_to_hf_reuses_save_steps_when_save_hf_steps_default(self):
        args = SimpleNamespace(
            logging_first_step=False,
            logging_strategy=IntervalStrategy.NO,
            logging_steps=1,
            evaluation_strategy=IntervalStrategy.NO,
            eval_steps=1,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=5,
            flash_device_save_steps=0,
            save_last_step=False,
            save_hf_steps=-1,
            save_to_hf=True,
        )
        state = TrainerState(global_step=5, max_steps=5)
        control = TrainerControl()
        DefaultFlowCallback().on_step_end(args, state, control)
        self.assertTrue(control.should_save_hf)
        self.assertTrue(control.should_save)

    def test_save_hf_stays_off_when_save_to_hf_false(self):
        args = SimpleNamespace(
            logging_first_step=False,
            logging_strategy=IntervalStrategy.NO,
            logging_steps=1,
            evaluation_strategy=IntervalStrategy.NO,
            eval_steps=1,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=5,
            flash_device_save_steps=0,
            save_last_step=False,
            save_hf_steps=-1,
            save_to_hf=False,
        )
        state = TrainerState(global_step=5, max_steps=5)
        control = TrainerControl()
        DefaultFlowCallback().on_step_end(args, state, control)
        self.assertFalse(control.should_save_hf)


if __name__ == "__main__":
    unittest.main()
