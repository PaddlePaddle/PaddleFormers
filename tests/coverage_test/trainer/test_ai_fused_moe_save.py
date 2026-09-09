# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
"""Focused tests for fused-MoE save at sharding_parallel_size=1 and HF export cadence."""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from paddleformers.trainer.trainer import Trainer, maybe_zero_max_grad_norm_for_uac
from paddleformers.trainer.trainer_callback import (
    DefaultFlowCallback,
    TrainerControl,
    TrainerState,
)
from paddleformers.trainer.trainer_utils import IntervalStrategy
from paddleformers.trainer.training_args import TrainingArguments


class TestDeferredTokenNormalizationWiring(unittest.TestCase):
    def test_train_loop_resolves_then_applies_before_optimizer_step(self):
        src = inspect.getsource(Trainer._inner_training_loop)
        resolve_at = src.find("self._resolve_deferred_token_normalization()")
        begin_at = src.find("self.callback_handler.on_optimizer_begin(")
        apply_at = src.find("self._apply_deferred_token_normalization(model)")
        step_at = src.find("self.optimizer_step(")
        self.assertNotEqual(resolve_at, -1)
        self.assertNotEqual(begin_at, -1)
        self.assertNotEqual(apply_at, -1)
        self.assertNotEqual(step_at, -1)
        self.assertLess(resolve_at, begin_at)
        self.assertLess(begin_at, apply_at)
        self.assertLess(apply_at, step_at)

    def test_apply_scales_main_grad_then_clears_divisor(self):
        src = inspect.getsource(Trainer._apply_deferred_token_normalization)
        self.assertIn("clear_pending_gradient_divisor", src)
        self.assertIn("grad.scale_(scale)", src)
        self.assertIn("main_grad", src)

    def test_resolve_skips_collectives_when_model_accuracy_mode_off(self):
        from unittest.mock import patch

        trainer = object.__new__(Trainer)
        trainer.model = SimpleNamespace(config=SimpleNamespace(use_accuracy_compatible=False))
        with patch("paddle.distributed.all_reduce") as reduce, patch("paddle.full") as allocate:
            trainer._resolve_deferred_token_normalization()
        reduce.assert_not_called()
        allocate.assert_not_called()


class TestFlexSaveWithoutMtpNumLayers(unittest.TestCase):
    def test_flex_save_uses_getattr_for_mtp_num_layers(self):
        from paddleformers.transformers.model_utils import PretrainedModel

        src = inspect.getsource(PretrainedModel.save_pretrained)
        self.assertIn('getattr(model_to_save.config, "mtp_num_layers", 0)', src)


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


class TestFusedExpertOptimizerSave(unittest.TestCase):
    def make_trainer(self, dtype="bfloat16"):
        import paddle
        from paddle.distributed import ShardedWeight

        class Model(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.grouped_gemm_experts = paddle.nn.Layer()
                self.grouped_gemm_experts.add_parameter(
                    "weight1",
                    self.create_parameter(
                        [2, 4, 6], dtype=dtype, default_initializer=paddle.nn.initializer.Constant(0.25)
                    ),
                )
                self.add_parameter(
                    "unrelated",
                    self.create_parameter(
                        [2, 3, 4], dtype=dtype, default_initializer=paddle.nn.initializer.Constant(0.5)
                    ),
                )

            def sharded_state_dict(self):
                result = {}
                for key, param in self.named_parameters():
                    tensor = param.reshape([-1, param.shape[-1]]) if key.startswith("grouped_gemm_experts") else param
                    tensor.name = param.name
                    result[key] = ShardedWeight(
                        key, tensor, tuple(tensor.shape), tuple(tensor.shape), (0,) * tensor.ndim
                    )
                return result

        trainer = object.__new__(Trainer)
        trainer.model = Model()
        trainer.optimizer = paddle.optimizer.AdamW(
            learning_rate=0.01, parameters=trainer.model.parameters(), multi_precision=True
        )
        trainer.args = SimpleNamespace(replicate_saved_into_local=False)
        self.step(trainer)
        return trainer

    @staticmethod
    def step(trainer):
        loss = sum((param.astype("float32") ** 2).sum() for param in trainer.model.parameters())
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.clear_grad()

    @staticmethod
    def snapshot(optimizer):
        return [
            (mapping, key, value, tuple(value.shape))
            for mapping in [*optimizer._accumulators.values(), optimizer._master_weights]
            for key, value in mapping.items()
        ]

    def assert_unchanged(self, snapshot):
        for mapping, key, tensor, shape in snapshot:
            self.assertIs(mapping[key], tensor)
            self.assertEqual(tuple(tensor.shape), shape)

    def test_save_load_then_step_matches_uninterrupted(self):
        import tempfile
        from pathlib import Path

        import numpy as np
        import paddle.distributed as dist

        from paddleformers.trainer.trainer import (
            MASTER_WEIGHT_DIC,
            OPTIMIZER_STATE_DIC,
            _fused_expert_optimizer_save_views,
        )

        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                trainer = self.make_trainer(dtype)
                snapshot = self.snapshot(trainer.optimizer)
                trainer._save_flex_optimizer_state(directory)
                self.assert_unchanged(snapshot)
                self.assertTrue((Path(directory) / "saved_signal_0").is_file())
                resumed = self.make_trainer(dtype)
                self.step(resumed)
                resumed.model.set_state_dict(trainer.model.state_dict())
                resumed_snapshot = self.snapshot(resumed.optimizer)
                with _fused_expert_optimizer_save_views(
                    resumed.model, resumed.model.sharded_state_dict(), resumed.optimizer
                ):
                    shards = resumed.optimizer.sharded_state_dict(resumed.model.sharded_state_dict())
                    dist.load_state_dict(
                        {k: v for k, v in shards.items() if not k.endswith(".w_0")},
                        str(Path(directory) / OPTIMIZER_STATE_DIC),
                    )
                    if dtype == "bfloat16":
                        dist.load_state_dict(
                            {k: v for k, v in shards.items() if k.endswith(".w_0")},
                            str(Path(directory) / MASTER_WEIGHT_DIC),
                        )
                self.assert_unchanged(resumed_snapshot)
                self.step(trainer)
                self.step(resumed)
                for left, right in zip(trainer.model.parameters(), resumed.model.parameters()):
                    np.testing.assert_array_equal(left.numpy(), right.numpy())
                for left_mapping, right_mapping in zip(
                    [*trainer.optimizer._accumulators.values(), trainer.optimizer._master_weights],
                    [*resumed.optimizer._accumulators.values(), resumed.optimizer._master_weights],
                ):
                    for left, right in zip(left_mapping.values(), right_mapping.values()):
                        np.testing.assert_array_equal(left.numpy(), right.numpy())

    def test_failures_restore_original_objects_and_shapes(self):
        import tempfile
        from unittest.mock import patch

        for location in ("sharded_state_dict", "save_state_dict"):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as directory:
                trainer = self.make_trainer()
                snapshot = self.snapshot(trainer.optimizer)
                target = (
                    trainer.optimizer
                    if location == "sharded_state_dict"
                    else __import__("paddle.distributed", fromlist=["save_state_dict"])
                )
                with patch.object(target, location, side_effect=RuntimeError("save failure")):
                    with self.assertRaisesRegex(RuntimeError, "save failure"):
                        trainer._save_flex_optimizer_state(directory)
                self.assert_unchanged(snapshot)
                self.step(trainer)


class TestUacMaxGradNormOverride(unittest.TestCase):
    def test_trainer_init_zeros_max_grad_norm_under_uac(self):
        args = SimpleNamespace(max_grad_norm=1.0)
        model = SimpleNamespace(config=SimpleNamespace(use_accuracy_compatible=True))
        maybe_zero_max_grad_norm_for_uac(args, model)
        self.assertEqual(args.max_grad_norm, 0.0)

    def test_non_uac_keeps_configured_max_grad_norm(self):
        args = SimpleNamespace(max_grad_norm=1.0)
        model = SimpleNamespace(config=SimpleNamespace(use_accuracy_compatible=False))
        maybe_zero_max_grad_norm_for_uac(args, model)
        self.assertEqual(args.max_grad_norm, 1.0)

    def test_trainer_init_calls_shipped_uac_helper(self):
        source = inspect.getsource(Trainer.__init__)
        self.assertIn("maybe_zero_max_grad_norm_for_uac(self.args, model)", source)


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
