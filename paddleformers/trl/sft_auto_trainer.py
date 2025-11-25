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

import os
import time

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.intermediate.parallelize import (
    parallelize_optimizer,
)

from ..data import DataCollatorForSeq2Seq
from ..trainer.trainer_callback import TrainerState
from ..trainer.trainer_utils import (  # set_hyrbid_parallel_seed,
    ShardingOption,
    TrainOutput,
    _exec_mode_guard,
    speed_metrics,
)
from ..trainer.utils.helper import (  # nested_truncate,
    distributed_file,
    distributed_isfile,
)
from ..utils.batch_sampler import DistributedBatchSampler as NlpDistributedBatchSampler
from ..utils.env import TRAINER_STATE_NAME
from ..utils.log import logger
from .sft_trainer import SFTTrainer

try:
    from ...quantization.quantization_linear import QuantizationLinear
except:
    QuantizationLinear = None

MODEL_NAME = "model"
OPTIMIZER_NAME = "optimizer"
DIST_CKPT_PATH = "dist_ckpt"
DIST_MODEL_PATH = "dist_model"
FREE_SVAE_LOAD_KEY_PATTERNS = ["learning_rate_", "gradient_merge_", "@GRAD@MERG", "eager_tmp"]

__all__ = ["SFTAutoTrainer"]


class SFTAutoTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):

        if kwargs.get("args", None) is not None and kwargs["args"].to_static:
            if kwargs.get("criterion", None) is None:

                def loss_func(loss, outputs):
                    return loss

                kwargs.update({"criterion": loss_func})

        sequence_parallel = False
        if kwargs.get("model_args", None) is not None:
            model_args = kwargs.pop("model_args")
            if hasattr(model_args, "sequence_parallel"):
                sequence_parallel = model_args.sequence_parallel
        print(" use_intermediate_api ", kwargs["args"].use_intermediate_api)
        if kwargs.get("args", None) is not None and kwargs["args"].use_intermediate_api:
            model = kwargs.get("model", None)
            assert model is not None
            # NOTE(zhangwl): some param_init_func is not support lazy init
            auto_dist_degree = {
                "tensor_parallel": kwargs["args"].tensor_parallel_degree > 1,
                "sequence_parallel": sequence_parallel,
                "pipeline_parallel": kwargs["args"].pipeline_parallel_degree > 1,
                "data_sharding_parallel": kwargs["args"].dataset_world_size > 1,
                "sharding": kwargs["args"].sharding,
                "sharding_mesh_dim": kwargs["args"].sharding_parallel_mesh_dimension,
            }
            auto_dist_config = model._generate_auto_dist_config(auto_dist_degree)
            self.auto_dist_config = auto_dist_config
            logger.info(f"auto_dist_config: {self.auto_dist_config}")
            kwargs["model"] = model

        trainable_parameters = [p for p in model.parameters() if not p.stop_gradient]
        self.set_optimizer_grouped_parameters(trainable_parameters)

        assert kwargs["args"].max_seq_len is not None, "max_seq_len must be specified in auto_parallel"

        if kwargs.get("data_collator", None) is None:
            data_collator = DataCollatorForSeq2Seq(
                max_length=kwargs["args"].max_seq_len,
                max_label_length=kwargs["args"].max_seq_len,
                padding="max_length",
            )
            kwargs["data_collator"] = data_collator

        super().__init__(*args, **kwargs)
        assert self.args.enable_auto_parallel

        self._in_pir_mode = paddle.base.framework.get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]

    def _wrap_for_auto(self, model, train_dataloader):
        logger.info(f"Wrapping model for auto parallel using intermediate api {self.args.use_intermediate_api} ")
        dist_loader = train_dataloader

        if self.args.use_intermediate_api:
            assert self.auto_dist_config is not None
            self.optimizer = parallelize_optimizer(
                self.optimizer,
                config=self.auto_dist_config,
            )
        else:
            sharding_parallel_mesh_dimension = self.args.sharding_parallel_mesh_dimension
            if ShardingOption.SHARD_OP in self.args.sharding:
                self.optimizer = dist.shard_optimizer(
                    self.optimizer,
                    dist.ShardingStage1(sharding_mesh_dim=sharding_parallel_mesh_dimension),
                    self.args.gradient_accumulation_steps,
                )
            elif ShardingOption.SHARD_GRAD_OP in self.args.sharding:
                self.optimizer = dist.shard_optimizer(
                    self.optimizer,
                    dist.ShardingStage2(sharding_mesh_dim=sharding_parallel_mesh_dimension),
                    self.args.gradient_accumulation_steps,
                )
            elif ShardingOption.FULL_SHARD in self.args.sharding:
                self.optimizer = dist.shard_optimizer(
                    self.optimizer,
                    dist.ShardingStage3(sharding_mesh_dim=sharding_parallel_mesh_dimension),
                    self.args.gradient_accumulation_steps,
                )
            else:
                self.optimizer = dist.shard_optimizer(self.optimizer, None, self.args.gradient_accumulation_steps)

        if self.args.to_static:
            unified_strategy = dist.Strategy()
            unified_strategy._from_legacy_strategy(self.args.strategy)

            # same logic as autocast_smart_context_manager() in trainer.py
            if self.enable_autocast_context_manager:
                unified_strategy.amp.custom_black_list.extend(["reduce_sum", "c_softmax_with_cross_entropy"])
                if self.args.fp16_opt_level == "O2":
                    unified_strategy.amp.custom_white_list.extend(["lookup_table", "lookup_table_v2"])

            # dist.to_static() obtains the input spec information through next(dataloader), but this has side effects
            # on the passed-in dataloader, altering the state of the sampler of the dataloader. In some cases, once
            # the state of the sampler is changed, it cannot be reverted. Therefore, a temporary dataloader is
            # constructed here to avoid side effects on the dataloader used for actual training.
            temp_loader = self._wrap_for_dist_loader(self.get_train_dataloader())
            model = dist.to_static(model, temp_loader, self.criterion, self.optimizer, strategy=unified_strategy)

        self.model_wrapped = model
        return model, dist_loader

    def _wrap_amp_model(self, args, model):
        self.amp_dtype = "float16" if self.args.fp16 else "bfloat16"
        if self.args.fp16_opt_level == "O2":
            paddle.amp.decorate(
                models=model,
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
                master_grad=self.args.amp_master_grad,
                excluded_layers=QuantizationLinear,
            )

    def _inner_training_loop(
        self,
        args,
        model,
        train_dataloader,
        len_dataloader,
        max_steps,
        num_train_epochs,
        num_update_steps_per_epoch,
        num_train_samples,
        resume_from_checkpoint,
        ignore_keys_for_eval,
    ):
        start_time = time.time()
        self._globalstep_last_start_time = time.time()
        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if (
            resume_from_checkpoint is not None
            and distributed_isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            and not self.args.ignore_load_lr_and_optim
        ):
            self.state = TrainerState.load_from_json(
                distributed_file(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            )
            if self.args.world_size > 1:
                global_step_list = []
                paddle.distributed.all_gather(
                    global_step_list, paddle.to_tensor([self.state.global_step], dtype="int64")
                )
                assert (
                    paddle.sum(paddle.stack(global_step_list) - global_step_list[0]) == 0
                ), f"Error, get different global step, please check! step list: {[x.item() for x in global_step_list]}"

            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    consumed_samples = (
                        self.state.global_step
                        * args.train_batch_size
                        * args.gradient_accumulation_steps
                        * args.dataset_world_size
                    )
                    train_dataloader.batch_sampler.set_epoch(consumed_samples=consumed_samples)
                    logger.info(f"Set DistributedBatchSampler consumed_samples to {consumed_samples}")

        epoch_iterator = train_dataloader
        # steps_in_epoch = len(epoch_iterator)
        steps_in_epoch = (
            len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
        )
        if len_dataloader is not None:
            if self.args.gradient_accumulation_steps > len(epoch_iterator):
                logger.warning(
                    f"changing accumulation step from `{self.args.gradient_accumulation_steps}` to `{len(epoch_iterator)}` to avoid, cross epoch accumulate"
                )
                self.args.gradient_accumulation_steps = len(epoch_iterator)

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        self.state.max_steps = int(max_steps)
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        tr_loss = paddle.to_tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        if self.args.device == "npu" and self.args.flatten_param_grads:
            from .plugins.npu_plugin import npu_accelerate_plugin

            npu_accelerate_plugin(self.optimizer)

        model, dist_loader = model, train_dataloader
        train_dataloader = dist_loader()
        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)

        self.timers and self.timers("read-data").start()

        for epoch in range(epochs_trained, num_train_epochs):

            step_control = 0  # used in loop control, reset to 0 after every step
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            # read global-batch from dist_loader
            for step, inputs in enumerate(train_dataloader):
                self.timers and self.timers("read-data").stop()
                os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
                self.callback_handler.on_load_data_end(args, self.state, self.control, inputs=inputs)

                # Skip past any already trained steps if resuming training
                # We use consumed_samples to reset the status
                if isinstance(train_dataloader._dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader._dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    if step == 0:
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(steps_trained_in_current_epoch)
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None
                        self._load_rng_state(resume_from_checkpoint)
                    step += steps_trained_in_current_epoch
                elif steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    self.timers and self.timers("read-data").start()
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                inputs_list = self._split_batches_for_accumulation(inputs)
                for input in inputs_list:
                    if step_control % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                        self.timers and self.timers("forward-backward").start()

                    tr_loss_step = self.training_step(model, input)
                    with _exec_mode_guard("dynamic"):
                        tr_loss += tr_loss_step

                    disable_accumulation = False
                    if self.args.pipeline_parallel_degree > 1 and self.args.to_static:
                        disable_accumulation = True
                    if self.args.to_static and self._in_pir_mode and self.args.gradient_accumulation_steps > 1:
                        disable_accumulation = True
                    # disable_accumulation = self.args.to_static

                    if (step_control + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                        or disable_accumulation
                    ):

                        self.timers and self.timers("forward-backward").stop()

                        self.timers and self.timers("optimizer-step").start()

                        if self.args.gradient_accumulation_steps > 1 and self._enable_delay_scale_loss():
                            tr_loss /= self.args.gradient_accumulation_steps

                        # Optimizer step
                        self.callback_handler.on_optimizer_begin(
                            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                        )

                        self.optimizer_step()

                        self.timers and self.timers("optimizer-step").stop()

                        self.callback_handler.on_optimizer_end(
                            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                        )

                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=input)
                        self._print_timer()
                        step_control = 0
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                        step_control += 1

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                self.timers and self.timers("read-data").start()

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\nTraining completed. \n")

        self._total_loss_scalar += self._get_item_from_loss(tr_loss)
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def optimizer_step(self):
        if not self.args.to_static:
            optimizer_was_run = True
            if self.do_grad_scaling:
                scale_before = paddle.assign(self.scaler._scale)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler._scale
                # Compatible with paddlepaddle 2.6.0 using typo word.
                if hasattr(self.scaler, "_cache_founf_inf"):
                    optimizer_was_run = not self.scaler._cache_founf_inf
                else:
                    optimizer_was_run = not self.scaler._cache_found_inf
                if not optimizer_was_run:
                    scale_before_value = scale_before.cpu().numpy()
                    scale_after_value = scale_after.cpu().numpy()
                    logger.warning(
                        f"optimizer not run, scale_before: {scale_before_value[0]}, scale_after: {scale_after_value[0]}"
                    )
            else:
                self.optimizer.step()

            if optimizer_was_run:
                self.lr_scheduler.step()

            self.optimizer.clear_grad()
        else:
            # TODO: support optimizer_was_run in static mode
            self.lr_scheduler.step()
