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
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.auto_parallel.intermediate.parallelize as parallelize
import paddle.distributed.auto_parallel.static.utils as auto_utils
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.dist_input_spec import DistributedInputSpec
from paddle.static import InputSpec

from ..peft.lora.lora_model import AVAILABLE_LAYERS
from ..trainer import Trainer
from ..trainer.trainer_callback import TrainerState
from ..trainer.trainer_utils import (
    ShardingOption,
    TrainOutput,
    _exec_mode_guard,
    speed_metrics,
)
from ..trainer.utils.helper import (  # nested_truncate,
    distributed_file,
    distributed_isfile,
)
from ..transformers.dpo_criterion import AutoDPOCriterion
from ..transformers.model_utils import unwrap_model
from ..utils import infohub
from ..utils.batch_sampler import DistributedBatchSampler as NlpDistributedBatchSampler
from ..utils.env import TRAINER_STATE_NAME
from ..utils.log import logger

DPO_INFO_KEYS = [
    "reference_chosen_logps",
    "reference_rejected_logps",
    "sft_loss",
    "policy_chosen_logps",
    "policy_rejected_logps",
    "dpo_loss",
]


def disable_dropout_in_model(model: paddle.nn.Layer) -> None:
    """ "disable dropout"""
    for module in model.children():
        if isinstance(module, paddle.nn.Dropout):
            module.p = 0


class DPOAutoTrainer(Trainer):
    """
    Initialize DPOAutoTrainer.
    """

    def __init__(
        self,
        model,
        data_collator,
        dpo_criterion=None,
        ref_model=None,
        dpo_config=None,
        disable_dropout: bool = True,
        padding_value: int = 0,
        model_with_dpo_criterion: bool = False,
        ignore_eos_token: bool = False,
        **kwargs
    ):
        super().__init__(model, data_collator=data_collator, **kwargs)
        sequence_parallel = False
        if kwargs.get("model_args", None) is not None:
            model_args = kwargs.pop("model_args")
            if hasattr(model_args, "sequence_parallel"):
                sequence_parallel = model_args.sequence_parallel

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
        self._in_pir_mode = paddle.base.framework.get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]
        model = parallelize.parallelize_model(
            model,
            config=self.auto_dist_config,
        )
        ref_model = parallelize.parallelize_model(
            ref_model,
            config=self.auto_dist_config,
        )
        kwargs["model"] = model

        if dpo_config is None:
            raise ValueError("dpo_config is None")
        else:
            self.dpo_config = dpo_config
        if not model_with_dpo_criterion:
            if dpo_criterion is None:
                self.dpo_criterion = AutoDPOCriterion(
                    self.model.config, dpo_config=dpo_config, ignore_eos_token=ignore_eos_token
                )
                self.dpo_criterion_dy = AutoDPOCriterion(
                    self.model.config, dpo_config=dpo_config, ignore_eos_token=ignore_eos_token
                )
            elif isinstance(dpo_criterion, AutoDPOCriterion):
                self.dpo_criterion = dpo_criterion
            else:
                raise ValueError(
                    "dpo_criterion should be None or AutoDPOCriterion. Got {}".format(type(dpo_criterion))
                )
        # model_with_dpo_criterion will save memory (logits part)
        self.model_with_dpo_criterion = model_with_dpo_criterion
        if self.dpo_config.loss_type not in [
            "sigmoid",
            "hinge",
            "ipo",
            "kto_pair",
            "sppo_hard",
            "nca_pair",
            "dpop",
            "or",
            "simpo",
        ]:
            raise ValueError(f"Unknown loss type: {self.dpo_config.loss_type}")
        if self.dpo_config.reference_free:
            if ref_model is not None:
                raise ValueError("reference_free set to True. No need to pass ref_model")
            if self.dpo_config.loss_type not in ["sigmoid", "hinge", "ipo", "or", "simpo"]:
                raise ValueError(f"{self.dpo_config.loss_type} does not support reference_free")
            self.ref_model = None
            self.ref_model_wrapped = None
        elif ref_model:
            if self.dpo_config.loss_type in ["or", "simpo"]:
                raise ValueError(f"{self.dpo_config.loss_type} loss type does not support ref_model")
            self.ref_model = ref_model
            self.ref_model_wrapped = self.ref_model
            self.ref_model_wrapped.eval()
        elif self.dpo_config.lora:
            self.ref_model = None
            self.ref_model_wrapped = None
        else:
            raise ValueError("reference_free set to False. ref_model is None")
        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.padding_value = padding_value
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.train_step_count = 0
        if self.compute_metrics is not None:
            raise NotImplementedError("compute_metrics is not supported for DPOAutoTrainer")
        self.reset_dpo_infohub()
        self.global_mesh = fleet.auto.get_mesh()

    def get_batch_metrics(self, ref_model, model, batch, train_eval="train"):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        dpo_inputs = {
            "input_ids": batch["input_ids"],
            "position_ids": batch["position_ids"],
        }
        if "attention_mask" in batch:
            dpo_inputs["attention_mask"] = batch["attention_mask"]
        elif "attn_mask_start_row_indices" in batch:
            dpo_inputs["attn_mask_start_row_indices"] = batch["attn_mask_start_row_indices"]
        elif "attn_mask_startend_row_indices" in batch:
            dpo_inputs["attn_mask_startend_row_indices"] = batch["attn_mask_startend_row_indices"]

        if self.model_with_dpo_criterion:
            dpo_inputs["chosen_labels"] = batch["chosen_labels"]
            dpo_inputs["rejected_labels"] = batch["rejected_labels"]
            dpo_inputs["response_indexs"] = batch["response_indexs"]
            if self.dpo_config.reference_free:
                reference_chosen_logps = paddle.zeros([1])
                reference_rejected_logps = paddle.zeros([1])
            else:
                if self.dpo_config.lora:
                    with paddle.no_grad():
                        self.disable_lora(model)
                        model.eval()
                        reference_chosen_logps, reference_rejected_logps = model(**dpo_inputs)
                        self.enable_lora(model)
                        model.train()
                else:
                    with paddle.no_grad():
                        reference_chosen_logps, reference_rejected_logps = ref_model(**dpo_inputs)
            dpo_inputs["reference_chosen_logps"] = reference_chosen_logps
            dpo_inputs["reference_rejected_logps"] = reference_rejected_logps
            policy_chosen_logps, policy_rejected_logps, sft_loss, dpo_loss, loss = model(**dpo_inputs)
        else:
            labels = (batch["chosen_labels"], batch["rejected_labels"], batch["response_indexs"], None, None)
            if self.dpo_config.reference_free:
                reference_chosen_logps = paddle.zeros([1])
                reference_rejected_logps = paddle.zeros([1])
            else:
                if self.dpo_config.lora:
                    with paddle.no_grad():
                        self.disable_lora(model)
                        model.eval()
                        logits = model(**dpo_inputs)
                        self.enable_lora(model)
                        model.train()
                else:
                    with paddle.no_grad():
                        logits = ref_model(**dpo_inputs)
                reference_chosen_logps, reference_rejected_logps = self.dpo_criterion(logits, labels)
            labels = labels[:3] + (reference_chosen_logps, reference_rejected_logps)
            logits = model(**dpo_inputs)
            policy_chosen_logps, policy_rejected_logps, sft_loss, dpo_loss, loss = self.dpo_criterion(logits, labels)

        return loss

    def compute_loss(self, model, inputs):
        """Compute the DPO loss for the given batch of inputs."""
        if (
            self.dpo_config.ref_model_update_steps > 0
            and self.train_step_count > 0
            and self.train_step_count % self.dpo_config.ref_model_update_steps == 0
            and not self.dpo_config.reference_free
        ):
            self.ref_model.set_state_dict(self.model.state_dict())
        self.train_step_count += 1
        loss = self.get_batch_metrics(self.ref_model_wrapped, model, inputs, train_eval="train")
        return loss

    def _wrap_ref_model(self, model):
        """Wrap reference model."""
        if unwrap_model(model) is not model:
            return model
        self.amp_dtype = "float16" if self.args.fp16 else "bfloat16"
        model = paddle.amp.decorate(
            models=model,
            level=self.args.fp16_opt_level,
            dtype=self.amp_dtype,
        )
        model = fleet.distributed_model(model)
        if self.args.pipeline_parallel_degree > 1:
            model._prepare_pipeline_inputs_func = prepare_pipeline_dpo_inputs_func

        return model

    def _wrap_model(self, model, training=True):
        """Wrap model."""
        model = super()._wrap_model(model, training)
        if self.args.pipeline_parallel_degree > 1:
            model._prepare_pipeline_inputs_func = prepare_pipeline_dpo_inputs_func
        return model

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """evaluate"""
        self.model_wrapped = self._wrap_ref_model(self.model_wrapped)
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):

        """prediction_step"""
        if self.args.pipeline_parallel_degree > 1:
            # hack for pipeline mode
            inputs = self._prepare_inputs(inputs)
            return self.prediction_pipeline_step(self.ref_model_wrapped, model, inputs)
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with paddle.no_grad():
            with self.autocast_smart_context_manager():
                loss = self.get_batch_metrics(self.ref_model_wrapped, model, inputs, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)
        else:
            raise NotImplementedError("DPOAutoTrainer only supports prediction_loss_only=True for now.")

    def store_metrics(self, metrics, train_eval="train"):
        """store_metrics"""
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs, **kwargs):
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = paddle.to_tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        if self.state.epoch is not None and train_eval == "train":
            self.state.epoch *= self.args.num_train_epochs
        return super().log(logs, **kwargs)

    def prediction_pipeline_step(
        self,
        ref_model,
        model,
        batch,
    ):
        """
        prediction_step function for pipeline parallel mode.
        """
        model._p2p_helper.clear_meta_cache()
        concatenated_inputs = {}
        # consider no drop last
        per_device_train_batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        # preprocess inputs: tuple(List[Tensor])
        for key in batch.keys():
            if key not in "response_indexs":
                concatenated_inputs[key] = [
                    batch[key][i * per_device_train_batch_size : (i + 1) * per_device_train_batch_size]
                    for i in range(gradient_accumulation_steps)
                ]
            else:
                concatenated_inputs["response_indexs"] = [[] for _ in range(gradient_accumulation_steps)]
                for i in range(gradient_accumulation_steps):
                    for response_index in batch[key]:
                        if response_index[0] in list(
                            range(i * per_device_train_batch_size, (i + 1) * per_device_train_batch_size)
                        ):
                            response_index[0] -= i * per_device_train_batch_size
                            concatenated_inputs["response_indexs"][i].append(response_index)
                    concatenated_inputs["response_indexs"][i] = paddle.stack(concatenated_inputs["response_indexs"][i])
                    if model._layers.config.use_sparse_head_and_loss_fn:
                        last_batch_response_length = concatenated_inputs["response_indexs"][i][0, 1]
                        concatenated_inputs["response_indexs"][i][:, 1:] -= last_batch_response_length

        concatenated_inputs["reference_chosen_logps"] = None
        concatenated_inputs["reference_rejected_logps"] = None

        self._pp_data_buffer = []
        inputs, labels = model._prepare_pipeline_inputs_func(concatenated_inputs)
        if not self.dpo_config.reference_free:
            if self.dpo_config.lora:
                self.disable_lora(model)
                model.eval()
                with paddle.no_grad():
                    with self.autocast_smart_context_manager():
                        model.eval_batch(data=[inputs, labels], compute_loss=True)
                self.enable_lora(model)
                model._p2p_helper.clear_meta_cache()
                model.train()
            else:
                ref_model = self.ref_model_wrapped
                with paddle.no_grad():
                    with self.autocast_smart_context_manager():
                        ref_model.eval_batch(data=[inputs, labels], compute_loss=True)
            reference_chosen_logps = infohub.reference_chosen_logps
            reference_rejected_logps = infohub.reference_rejected_logps
        else:
            reference_chosen_logps = [paddle.zeros([1]) for _ in range(model.accumulate_steps)]
            reference_rejected_logps = [paddle.zeros([1]) for _ in range(model.accumulate_steps)]
        if model.is_pipeline_last_stage(ignore_virtual=model._layers._num_virtual_pipeline_stages > 1):
            labels = labels[:3] + (reference_chosen_logps, reference_rejected_logps)
        with paddle.no_grad():
            with self.autocast_smart_context_manager():
                loss = model.eval_batch(data=[inputs, labels], compute_loss=True)

        # broadcast DPO_INFO_KEYS
        self.broadcast_last_stage_infohub_tensor()
        # metrics
        metric_inputs = dict(
            reference_chosen_logps=infohub.reference_chosen_logps,
            reference_rejected_logps=infohub.reference_rejected_logps,
            policy_chosen_logps=infohub.policy_chosen_logps,
            policy_rejected_logps=infohub.policy_rejected_logps,
            dpo_loss=infohub.dpo_loss,
            sft_loss=infohub.sft_loss,
            train_eval="eval",
        )
        self.log_metric(**metric_inputs)
        self.reset_dpo_infohub()
        model._p2p_helper.clear_meta_cache()
        return (loss, None, None)

    def log_metric(
        self,
        reference_chosen_logps,
        reference_rejected_logps,
        policy_chosen_logps,
        policy_rejected_logps,
        dpo_loss,
        sft_loss,
        train_eval,
    ):
        metrics = {}
        chosen_rewards = self.dpo_config.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.dpo_config.beta * (policy_rejected_logps - reference_rejected_logps)
        reward_accuracies = (chosen_rewards > rejected_rewards).astype(paddle.float32)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean()
        metrics[f"{prefix}{self.dpo_config.loss_type}_loss"] = dpo_loss
        metrics[f"{prefix}sft_loss"] = sft_loss
        if self.dpo_config.loss_type == "or":
            log_odds = (policy_chosen_logps - policy_rejected_logps) - (
                paddle.log1p(-paddle.exp(policy_chosen_logps)) - paddle.log1p(-paddle.exp(policy_rejected_logps))
            )
            ratio = F.log_sigmoid(log_odds)
            metrics[f"{prefix}log_odds_ratio"] = log_odds.mean()
            metrics[f"{prefix}log_odds_chosen"] = ratio.mean()

        for key in metrics:
            metrics[key] = self._nested_gather(paddle.tile(metrics[key], repeat_times=[1, 1])).mean().cpu()
        if self.args.should_save:
            self.store_metrics(metrics, train_eval=train_eval)

    def training_pipeline_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.
        """
        # accumulation data
        if not hasattr(self, "_pp_data_buffer"):
            self._pp_data_buffer = []
        self._pp_data_buffer.append(inputs)
        if len(self._pp_data_buffer) != self.args.gradient_accumulation_steps:
            return paddle.zeros([])

        concatenated_inputs = {}
        for key in self._pp_data_buffer[0].keys():
            concatenated_inputs[key] = [
                self._pp_data_buffer[i][key] for i in range(self.args.gradient_accumulation_steps)
            ]
        concatenated_inputs["reference_chosen_logps"] = None
        concatenated_inputs["reference_rejected_logps"] = None
        self._pp_data_buffer = []
        inputs, labels = model._prepare_pipeline_inputs_func(concatenated_inputs)
        model_config_backup = model.micro_batch_size, model.accumulate_steps
        model.micro_batch_size = self.args.per_device_train_batch_size
        model.accumulate_steps = self.args.gradient_accumulation_steps

        if not self.dpo_config.reference_free:
            if self.dpo_config.lora:
                self.disable_lora(model)
                model.eval()
                with paddle.no_grad():
                    with self.autocast_smart_context_manager():
                        model.eval_batch(data=[inputs, labels], compute_loss=True)
                self.enable_lora(model)
                model._p2p_helper.clear_meta_cache()
                model.train()
            else:
                ref_model = self.ref_model_wrapped
                ref_model_config_backup = ref_model.micro_batch_size, ref_model.accumulate_steps
                ref_model.accumulate_steps = model.accumulate_steps
                ref_model.micro_batch_size = model.micro_batch_size
                with paddle.no_grad():
                    with self.autocast_smart_context_manager():
                        ref_model.eval_batch(data=[inputs, labels], compute_loss=True)
                ref_model.micro_batch_size, ref_model.accumulate_steps = ref_model_config_backup
            reference_chosen_logps = infohub.reference_chosen_logps
            reference_rejected_logps = infohub.reference_rejected_logps
        else:
            reference_chosen_logps = [paddle.zeros([1]) for _ in range(model.accumulate_steps)]
            reference_rejected_logps = [paddle.zeros([1]) for _ in range(model.accumulate_steps)]
        if model.is_pipeline_last_stage(ignore_virtual=model._layers._num_virtual_pipeline_stages > 1):
            labels = labels[:3] + (reference_chosen_logps, reference_rejected_logps)
        train_inputs = [inputs, labels]
        train_inputs = model._prepare_training(train_inputs, self.optimizer, self.lr_scheduler)
        model.optimizer = None  # we do not use `PipelineParallel` to handler optimizer step
        model.lr_scheduler = None
        with self.autocast_smart_context_manager():
            loss = model.forward_backward_pipeline(train_inputs, self.scaler if self.do_grad_scaling else None)
        model.micro_batch_size, model.accumulate_steps = model_config_backup

        # broadcast DPO_INFO_KEYS
        self.broadcast_last_stage_infohub_tensor()

        # metrics
        metric_inputs = dict(
            reference_chosen_logps=infohub.reference_chosen_logps,
            reference_rejected_logps=infohub.reference_rejected_logps,
            policy_chosen_logps=infohub.policy_chosen_logps,
            policy_rejected_logps=infohub.policy_rejected_logps,
            dpo_loss=infohub.dpo_loss,
            sft_loss=infohub.sft_loss,
            train_eval="train",
        )
        self.log_metric(**metric_inputs)
        self.reset_dpo_infohub()
        return loss.detach()

    def disable_lora(self, model):
        """Disable LORA layers."""
        for _, layer in model.named_sublayers():
            if any(isinstance(layer, lora_layer) for lora_layer in AVAILABLE_LAYERS):
                layer.disable_lora = True

    def enable_lora(self, model):
        """Enable LORA layers."""
        for _, layer in model.named_sublayers():
            if any(isinstance(layer, lora_layer) for lora_layer in AVAILABLE_LAYERS):
                layer.disable_lora = False

    def reset_dpo_infohub(self):
        """Initialize infohub"""
        for key in DPO_INFO_KEYS:
            setattr(infohub, key, [])

    def broadcast_last_stage_infohub_tensor(self):
        for key in DPO_INFO_KEYS:
            if self.model_wrapped.is_pipeline_last_stage(
                ignore_virtual=self.model_wrapped._layers._num_virtual_pipeline_stages > 1
            ):
                if "loss" in key:
                    tensor = paddle.stack(getattr(infohub, key)).mean().detach()
                elif "logps" in key:
                    if len(getattr(infohub, key)) == 0:
                        tensor = paddle.zeros([1])
                    else:
                        tensor = paddle.concat(getattr(infohub, key), axis=0).detach()
                    tensor_shape = paddle.to_tensor(tensor.shape, dtype="int64")
                    paddle.distributed.broadcast(
                        tensor_shape, src=self.model_wrapped.global_rank, group=self.model_wrapped.pp_group
                    )
                else:
                    raise ValueError(f"Invalid key: {key}")
                paddle.distributed.broadcast(
                    tensor, src=self.model_wrapped.global_rank, group=self.model_wrapped.pp_group
                )
            else:
                if "loss" in key:
                    tensor = paddle.zeros([], "float32")
                elif "logps" in key:
                    tensor_shape = paddle.empty([1], dtype="int64")
                    paddle.distributed.broadcast(
                        tensor_shape,
                        src=self.model_wrapped._hcg.get_rank_from_stage(self.model_wrapped.num_stages - 1),
                        group=self.model_wrapped.pp_group,
                    )
                    tensor = paddle.zeros(tensor_shape, "float32")
                else:
                    raise ValueError(f"Invalid key: {key}")
                paddle.distributed.broadcast(
                    tensor,
                    src=self.model_wrapped._hcg.get_rank_from_stage(self.model_wrapped.num_stages - 1),
                    group=self.model_wrapped.pp_group,
                )
            setattr(infohub, key, tensor)

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
                for inputs in inputs_list:
                    if step_control % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                        self.timers and self.timers("forward-backward").start()

                    tr_loss_step = self.training_step(model, inputs)

                    with _exec_mode_guard("dynamic"):
                        tr_loss += tr_loss_step

                    disable_accumulation = False
                    if self.args.pipeline_parallel_degree > 1 and self.args.to_static:
                        disable_accumulation = True
                    if self.args.to_static and self._in_pir_mode and self.args.gradient_accumulation_steps > 1:
                        disable_accumulation = True

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
                        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
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
        dist_loader._input_keys = ["input_ids", "labels"]  # 动转静要求模型输入只能是2个字段
        return dist_loader

    def _wrap_for_auto(self, model, train_dataloader):
        logger.info(f"Wrapping model for auto parallel using intermediate api {self.args.use_intermediate_api} ")
        dist_loader = self._wrap_for_dist_loader(train_dataloader)

        if self.args.use_intermediate_api:
            assert self.auto_dist_config is not None
            self.optimizer = parallelize.parallelize_optimizer(
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

            # create inputs_spec and labels_spec
            inputs_spec = []
            labels_spec = []
            data = next(temp_loader())
            if hasattr(temp_loader, "batch_sampler"):
                batch_sampler = temp_loader.batch_sampler
            else:
                batch_sampler = temp_loader._dataloader.batch_sampler
            if hasattr(batch_sampler, "set_epoch"):
                # Get data from DataLoader iterator directly may affect data generation randomness
                # of BatchSampler when `Shuffle=True`. It may cause difference of data feeding
                # between dynamic and to_static mode.
                batch_sampler.set_epoch(0)
            if isinstance(data, dict):
                data = list(data.values())
                if len(data) >= 2:
                    labels = data.pop()
                    inputs = data
                else:
                    raise ValueError(f"Data should be a dict at least two keys, but received {len(data)}.")
            elif isinstance(data, (list, tuple)):
                if len(data) >= 2:
                    labels = data.pop()
                    inputs = data
                else:
                    raise ValueError(f"Data should be a dict or list at list two element, but received {len(data)}.")
            else:
                raise TypeError(f"Data should be a dict or list, but received {type(data)}.")
            if not isinstance(inputs, (list, tuple)):
                inputs = auto_utils.to_list(inputs)
            labels = auto_utils.to_list(labels)

            def flatten_list(nested_list):
                flat_list = []
                for item in nested_list:
                    if isinstance(item, (list, tuple)):
                        flat_list.extend(flatten_list(item))
                    else:
                        flat_list.append(item)
                return flat_list

            inputs = flatten_list(inputs)
            if inputs is not None:
                for i, item in enumerate(inputs):
                    assert item is not None, "Receive None input."
                    name = "input" + str(i)
                    spec = DistributedInputSpec.from_dtensor(item, name)
                    if i == 2:
                        spec.shape = (-1, 4)
                    inputs_spec.append(spec)
            if labels is not None:
                for i, item in enumerate(labels):
                    assert item is not None, "Receive None input."
                    name = "label" + str(i)
                    spec = DistributedInputSpec.from_dtensor(item, name)
                    if i == 2:
                        spec.shape = (-1, 4)
                    labels_spec.append(spec)

            def _validate_spec(specs):
                specs = auto_utils.to_list(specs)
                if specs is not None:
                    for i, spec in enumerate(specs):
                        if not isinstance(spec, InputSpec) and not isinstance(spec, DistributedInputSpec):
                            raise TypeError(
                                "'spec' must be object of class `paddle.static.InputSpec` or `DistributedInputSpec`."
                            )
                        if spec.name is None:
                            raise ValueError(f"Requires Input[{i}].name != None, but receive `None` with {spec}.")

                return specs or []

            inputs_spec = _validate_spec(inputs_spec)
            labels_spec = _validate_spec(labels_spec)
            input_spec = (inputs_spec, labels_spec)
            model = dist.to_static(
                model,
                temp_loader,
                self.dpo_criterion,
                self.optimizer,
                strategy=unified_strategy,
                input_spec=input_spec,
            )
        self.model_wrapped = model
        return model, dist_loader

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
        dpo_inputs = {
            "input_ids": inputs["input_ids"][0],
            "position_ids": inputs["input_ids"][1],
        }
        dpo_inputs["attention_mask"] = inputs["input_ids"][3]
        if len(inputs["input_ids"]) > 6:
            dpo_inputs["attn_mask_startend_row_indices"] = inputs["input_ids"][6]

        input_ids, labels = tuple(inputs.values())
        labels = (labels[0], labels[1], labels[2], None, None)

        ref_model = self.ref_model_wrapped
        with paddle.no_grad():
            logits = ref_model(**dpo_inputs)
            reference_chosen_logps, reference_rejected_logps = self.dpo_criterion_dy(
                logits, labels[0], labels[1], labels[2], None, None
            )

        labels = labels[:3] + (reference_chosen_logps, reference_rejected_logps)
        loss = model(input_ids, labels)

        return loss


def prepare_pipeline_dpo_inputs_func(inputs):
    """Prepare pipeline inputs"""
    if "attention_mask" in inputs:
        first_stage_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
        ]
    else:
        first_stage_keys = [
            "input_ids",
            "attn_mask_start_row_indices",
            "attn_mask_startend_row_indice",
            "position_ids",
        ]

    last_stage_keys = [
        "chosen_labels",
        "rejected_labels",
        "response_indexs",
        "reference_chosen_logps",
        "reference_rejected_logps",
    ]

    def get_expected_keys(inputs, keys):
        ret = tuple([inputs.pop(k) for k in keys if k in inputs])
        if len(ret) == 1:
            ret = ret[0]
        return ret

    if type(inputs) is dict or type(inputs) is OrderedDict:
        return [
            get_expected_keys(inputs, first_stage_keys),
            get_expected_keys(inputs, last_stage_keys),
        ]

    keys = list(inputs[0].keys())
    inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
    return [
        get_expected_keys(inputs_batch, first_stage_keys),
        get_expected_keys(inputs_batch, last_stage_keys),
    ]
