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
import random
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.intermediate.parallelize import (
    parallelize_model,
    parallelize_optimizer,
)

from ..data import DataCollatorForSeq2Seq
from ..trainer.argparser import strtobool
from ..trainer.trainer_callback import TrainerState
from ..trainer.trainer_utils import (  # set_hyrbid_parallel_seed,
    PREFIX_CHECKPOINT_DIR,
    ShardingOption,
    TrainOutput,
    _exec_mode_guard,
    get_last_checkpoint,
    has_length,
    speed_metrics,
)
from ..trainer.utils.ckpt_converter import CheckpointConverter
from ..trainer.utils.helper import (  # nested_truncate,
    distributed_file,
    distributed_isfile,
)
from ..utils.batch_sampler import DistributedBatchSampler as NlpDistributedBatchSampler
from ..utils.env import (
    SCALER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
)
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
            model = parallelize_model(
                model,
                config=self.auto_dist_config,
            )
            for p in model.parameters():
                print(f"param {p.name} stop_gradient {p.stop_gradient}")
            kwargs["model"] = model

        model = kwargs["model"]
        for param in model.parameters():
            if not param._is_initialized() and param._init_func is not None:
                param.initialize()
        kwargs["model"] = model

        trainable_parameters = [p for p in model.parameters() if not p.stop_gradient]
        self.set_optimizer_grouped_parameters(trainable_parameters)

        assert kwargs["args"].max_seq_length is not None, "max_seq_length must be specified in auto_parallel"

        if kwargs.get("data_collator", None) is None:
            data_collator = DataCollatorForSeq2Seq(
                max_length=kwargs["args"].max_seq_length,
                max_label_length=kwargs["args"].max_seq_length,
                padding="max_length",
            )
            kwargs["data_collator"] = data_collator

        super().__init__(*args, **kwargs)
        assert self.args.enable_auto_parallel

        self.global_mesh = fleet.auto.get_mesh()
        self.comm_group_in_pp = fleet.get_hybrid_communicate_group().get_pipe_parallel_group()
        self._in_pir_mode = paddle.base.framework.get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]

    def _nested_gather(self, tensors):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        with _exec_mode_guard("dynamic"):
            if isinstance(tensors, paddle.Tensor):
                tr_loss = tensors._local_value() if tensors.is_dist() else tensors
            else:
                tr_loss = paddle.to_tensor([tensors])

        if self.args.pipeline_parallel_degree <= 1:
            return super()._nested_gather(tr_loss)

        paddle.distributed.broadcast(tr_loss, src=self.comm_group_in_pp.ranks[-1], group=self.comm_group_in_pp)

        return super()._nested_gather(tr_loss)

    def _wrap_model(self, model, training=True):
        return model

    def _get_meshes_for_loader(self, train_dataloader):
        def _get_mesh(pp_idx=0):
            return self.global_mesh.get_mesh_with_dim("pp")[pp_idx]

        # Note(lizhiyu): If the values returned by `DataLoader` don't have the format `[images, labels]`,
        # error may occurs here.
        meshes = []
        meshes.append(_get_mesh(0))
        data = next(train_dataloader())
        if isinstance(data, dict):
            data_num = len(list(data.values()))
        elif isinstance(data, (list, tuple)):
            data_num = len(data)
        assert data_num >= 2
        if self.args.pipeline_parallel_degree > 1:
            for i in range(1, data_num):
                meshes.append(_get_mesh(0))
            meshes[-1] = _get_mesh(self.args.pipeline_parallel_degree - 1)
        return meshes

    def _wrap_for_dist_loader(self, train_dataloader):
        dist_loader = dist.shard_dataloader(
            dataloader=train_dataloader,
            meshes=self._get_meshes_for_loader(train_dataloader),
            shard_dims="dp",
        )
        return dist_loader

    def _wrap_for_auto(self, model, train_dataloader):
        logger.info(f"Wrapping model for auto parallel using intermediate api {self.args.use_intermediate_api} ")
        dist_loader = self._wrap_for_dist_loader(train_dataloader)

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
        logger.info("Using half precision")
        self.amp_dtype = "float16" if self.args.fp16 else "bfloat16"
        if self.args.fp16_opt_level == "O2":
            paddle.amp.decorate(
                models=model,
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
                master_grad=self.args.amp_master_grad,
                excluded_layers=QuantizationLinear,
            )
        self.enable_autocast_context_manager = True

        if args.to_static:
            return
        self.do_grad_scaling = True if self.args.fp16 else False
        self.scaler = dist.shard_scaler(paddle.amp.GradScaler(init_loss_scaling=self.args.scale_loss))

    def _get_item_from_loss(self, loss):
        if isinstance(loss, paddle.Tensor):
            if loss.is_dist():
                return loss._local_value().item() if loss._is_initialized() else 0.0
            else:
                return loss.item() if loss._is_initialized() else 0.0
        else:
            return loss

    def _split_batches_for_accumulation(self, inputs):
        if self.args.gradient_accumulation_steps == 1:
            return [inputs]

        if self.args.to_static and self.args.pipeline_parallel_degree > 1:
            return [inputs]

        if self.args.to_static and self._in_pir_mode and self.args.gradient_accumulation_steps > 1:
            return [inputs]

        global_micro_batchs = [{} for i in range(self.args.gradient_accumulation_steps)]
        assert isinstance(inputs, dict)

        def split_dtensor_by_axis(dtensor, axis=0):
            if not dtensor._is_initialized():
                return dtensor.split(self.args.gradient_accumulation_steps, axis=axis)

            micro_batch_shape = dtensor.shape
            micro_batch_shape[axis] = int(dtensor.shape[axis] / self.args.gradient_accumulation_steps)

            global_micro_batchs = [
                paddle.zeros(micro_batch_shape, dtype=dtensor.dtype)
                for _ in range(self.args.gradient_accumulation_steps)
            ]
            global_micro_batchs = [
                dist.shard_tensor(b, dtensor.process_mesh, dtensor.placements) for b in global_micro_batchs
            ]

            local_micro_batchs = dtensor._local_value().split(self.args.gradient_accumulation_steps, axis=axis)
            for local_micro_batch, global_micro_batch in zip(local_micro_batchs, global_micro_batchs):
                paddle.assign(local_micro_batch, global_micro_batch._local_value())
            return global_micro_batchs

        for key, dtensors in inputs.items():
            if isinstance(dtensors, paddle.Tensor):
                mesh, placements = dtensors.process_mesh, dtensors.placements
                global_datas = split_dtensor_by_axis(dtensors, 0)
                for index, data in enumerate(global_datas):
                    global_micro_batchs[index].update({key: dist.reshard(data, mesh, placements)})
            elif isinstance(dtensors, (list, tuple)):
                if len(dtensors) == 0:
                    for i in range(self.args.gradient_accumulation_steps):
                        global_micro_batchs[i].update({key: []})
                else:
                    for dtensor in dtensors:
                        if isinstance(dtensor, paddle.Tensor):
                            mesh, placements = dtensor.process_mesh, dtensor.placements
                            global_datas = split_dtensor_by_axis(dtensor, 0)
                            for index, data in enumerate(global_datas):
                                if key in global_micro_batchs[index].keys():
                                    global_micro_batchs[index][key].append(dist.reshard(data, mesh, placements))
                                else:
                                    global_micro_batchs[index].update({key: [dist.reshard(data, mesh, placements)]})
                        else:
                            raise ValueError(f"unsupported type: {type(dtensor)}")
            else:
                raise ValueError(f"unsupported type: {type(dtensors)}")
        return global_micro_batchs

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

        model, dist_loader = self._wrap_for_auto(model, train_dataloader)
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

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        total_batch_size_per_acc_step = self.args.per_device_train_batch_size * self.args.dataset_world_size
        total_batch_size = total_batch_size_per_acc_step * self.args.gradient_accumulation_steps

        return paddle.io.BatchSampler(
            dataset=self.train_dataset,
            shuffle=False,
            batch_size=total_batch_size,
            drop_last=self.args.dataloader_drop_last,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.criterion is not None:
            if "labels" in inputs:
                labels = inputs.pop("labels")
            elif "start_positions" in inputs and "end_positions" in inputs:
                labels = (inputs.pop("start_positions"), inputs.pop("end_positions"))
            elif self.args.label_names is not None:
                labels = []
                for label in self.label_names:
                    labels.append(inputs.pop(label))
                labels = tuple(labels)
            elif "generator_labels" in inputs:
                labels = inputs["generator_labels"]
        else:
            labels = None

        outputs = model(**inputs)

        if self.criterion is not None:

            def to_list(value):
                if value is None:
                    return value
                if isinstance(value, (list, tuple)):
                    return list(value)
                return [value]

            criterion_inputs = to_list(outputs)
            criterion_labels = to_list(labels)
            loss = self.criterion(*(criterion_inputs + criterion_labels))
            outputs = (loss, outputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        elif isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            loss = outputs

        return (loss, outputs) if return_outputs else loss

    def dynamic_training(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if loss is not None and self.args.gradient_accumulation_steps > 1 and not self._enable_delay_scale_loss():
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss

    def static_training(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        # NOTE(zhangwl):need support input attention_mask in static mode
        input_data = list(inputs.values())
        loss = model(*input_data)
        # inputs = list(inputs.values())
        # loss = model(*inputs)
        if loss is not None and self.args.gradient_accumulation_steps > 1 and not self._enable_delay_scale_loss():
            loss = loss / self.args.gradient_accumulation_steps

        return loss

    def training_step(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        model.train()

        inputs = self._prepare_inputs(inputs)

        if not self.args.to_static:
            loss = self.dynamic_training(model, inputs)
        else:
            loss = self.static_training(model, inputs)

        if isinstance(loss, paddle.Tensor):
            return loss.detach() if loss._is_initialized() else float(0.0)
        elif isinstance(loss, np.ndarray):
            return np.sum(loss)
        elif loss is None:
            return float(0.0)
        else:
            return float(loss)

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

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        with _exec_mode_guard("dynamic"):
            self.control.should_evaluate = False
            super()._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, **kwargs)

    def _save_model(self):
        if not self.args.to_static:
            return
        with _exec_mode_guard("static"):
            output_dir = f"{self.args.output_dir}/{DIST_MODEL_PATH}"
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model files into {output_dir}")
            model_file = os.path.join(output_dir, "rank_" + str(paddle.distributed.get_rank()) + ".pd_dist_model")
            if os.path.exists(model_file):
                os.remove(model_file)
            paddle.save(self.model_wrapped.dist_main_program("train"), model_file)

    def _save_checkpoint(self, model, metrics=None):

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir
        output_dir = f"{run_dir}/{checkpoint_folder}"

        if self.args.should_save or self.args.should_save_model_state:
            os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            logger.info(f"Saving checkpoinit files into {output_dir}")

            if self.args.should_save_model_state:
                if self.args.to_static:
                    opt_state_dict = {
                        key: value
                        for key, value in model.state_dict("opt").items()
                        if not any(keyword in key for keyword in FREE_SVAE_LOAD_KEY_PATTERNS)
                    }
                    state_dict = {
                        MODEL_NAME: model.state_dict("param"),
                        OPTIMIZER_NAME: opt_state_dict,
                    }
                else:
                    optim_state_dict = self.optimizer.state_dict()
                    optim_state_dict.pop("LR_Scheduler", None)
                    opt_state_keys = ["_moment1_0", "_moment2_0", "_beta1_pow_acc_0", "_beta2_pow_acc_0"]
                    for p_name, p in model.state_dict().items():
                        if paddle.distributed.get_rank() not in p.process_mesh.process_ids:
                            var_name = p.name
                            for key in opt_state_keys:
                                if (
                                    var_name + key in optim_state_dict
                                    and not optim_state_dict[var_name + key].is_dist()
                                ):
                                    optim_state_dict.pop(var_name + key)

                    state_dict = {
                        MODEL_NAME: model.state_dict(),
                        OPTIMIZER_NAME: optim_state_dict,
                    }

                self._save_ckpt_func(state_dict, os.path.join(output_dir, DIST_CKPT_PATH))
                logger.info(f"Model weights and optimizer states saved in {output_dir}/{DIST_CKPT_PATH}")

                # FIXME: maybe only save one copy
                paddle.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

                if self.do_grad_scaling:
                    paddle.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": paddle.get_rng_state(),
            "cpu": paddle.framework.core.default_cpu_generator().get_state(),
        }

        if self.args.world_size > 1:
            rng_states_list = []
            paddle.distributed.all_gather_object(rng_states_list, rng_states)
            if self.args.should_save:
                os.makedirs(output_dir, exist_ok=True)
                paddle.save(rng_states_list, os.path.join(output_dir, f"rng_state_{self.args.world_size}.pth"))
        else:
            os.makedirs(output_dir, exist_ok=True)
            paddle.save(rng_states, os.path.join(output_dir, "rng_state.pth"))

        if strtobool(os.getenv("FLAG_LLM_PDC", "False")):
            # save checkpoint_done file to ensure checkpoint is complete
            if self.args.should_save_model_state and self.args.should_save:
                # For ckpt integrity
                paddle.save(self.state.global_step, os.path.join(output_dir, ".checkpoint_done"))

    def _save(
        self,
        output_dir: Optional[str] = None,
        state_dict=None,
        merge_tensor_parallel=False,
    ):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if self.args.should_save:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            # Good practice: save your training arguments together with the trained model
            paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.args.should_save_model_state:
            self._save_ckpt_func(self.model.state_dict(), os.path.join(output_dir, MODEL_NAME))
            logger.info(f"Model weights saved in {output_dir}/{MODEL_NAME}")

    def _load_from_checkpoint(self, resume_from_checkpoint=None):

        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if resume_from_checkpoint is not None:

            logger.info(f"Loading model from {resume_from_checkpoint} .")

            if not self.args.ignore_load_lr_and_optim:
                with _exec_mode_guard("dynamic"):
                    if distributed_isfile(os.path.join(resume_from_checkpoint, SCHEDULER_NAME)):
                        self.lr_scheduler.set_state_dict(
                            paddle.load(distributed_file(os.path.join(resume_from_checkpoint, SCHEDULER_NAME)))
                        )
                    else:
                        raise ValueError(
                            f"scheduler-file not found, scheduler:{os.path.join(resume_from_checkpoint, SCHEDULER_NAME)}"
                        )

                    if self.do_grad_scaling and distributed_isfile(os.path.join(resume_from_checkpoint, SCALER_NAME)):
                        self.scaler.load_state_dict(
                            paddle.load(
                                distributed_file(os.path.join(resume_from_checkpoint, SCALER_NAME)), return_numpy=True
                            )
                        )

            if self.args.to_static:
                if self.model_wrapped._mode is None:
                    self.model_wrapped.train()
                model_state_dict = {
                    key: value
                    for key, value in self.model_wrapped.state_dict("param").items()
                    if not any(keyword in key for keyword in FREE_SVAE_LOAD_KEY_PATTERNS)
                }
                optim_state_dict = {
                    key: value
                    for key, value in self.model_wrapped.state_dict("opt").items()
                    if not any(keyword in key for keyword in FREE_SVAE_LOAD_KEY_PATTERNS)
                }
            else:
                model_state_dict = self.model_wrapped.state_dict()
                optim_state_dict = self.optimizer.state_dict()
                optim_state_dict.pop("LR_Scheduler", None)
                if len(optim_state_dict) == 0:
                    self.optimizer._create_accumulators(
                        paddle.base.framework.default_main_program().global_block(), self.optimizer._parameter_list
                    )
                    optim_state_dict = self.optimizer.state_dict()
                    optim_state_dict.pop("LR_Scheduler", None)

            state_dict = {
                MODEL_NAME: model_state_dict,
                OPTIMIZER_NAME: optim_state_dict,
            }

            parameter_to_structured_name = {}
            if self.args.to_static:
                parameter_to_structured_name = self.model_wrapped._parameter_to_structured_name
            else:
                for state_name, state_value in self.model_wrapped.state_dict().items():
                    parameter_to_structured_name[state_value.name] = state_name

            if self.args.auto_parallel_resume_form_hybrid_parallel:
                CheckpointConverter(
                    resume_from_checkpoint, state_dict, parameter_to_structured_name, self.args
                ).load_from_hybrid_parallel_checkpoint()
            else:
                ckpt_path = os.path.join(resume_from_checkpoint, DIST_CKPT_PATH)
                if not os.path.isdir(ckpt_path):
                    raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")
                self._load_ckpt_func(state_dict, ckpt_path)

            if self.args.to_static:
                self.model_wrapped.set_state_dict(model_state_dict)
                self.model_wrapped.set_state_dict(optim_state_dict)
            # release memory
            del state_dict
