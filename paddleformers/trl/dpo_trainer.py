# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from collections import OrderedDict, defaultdict

import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet

from ..peft.lora.lora_model import AVAILABLE_LAYERS
from ..trainer import Trainer
from ..transformers.dpo_criterion import DPOCriterion
from ..transformers.model_utils import unwrap_model
from ..utils import infohub

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


class DPOTrainer(Trainer):
    """
    Initialize DPOTrainer.
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
        if dpo_config is None:
            raise ValueError("dpo_config is None")
        else:
            self.dpo_config = dpo_config
        if not model_with_dpo_criterion:
            if dpo_criterion is None:
                self.dpo_criterion = DPOCriterion(
                    self.model.config, dpo_config=dpo_config, ignore_eos_token=ignore_eos_token
                )
            elif isinstance(dpo_criterion, DPOCriterion):
                self.dpo_criterion = dpo_criterion
            else:
                raise ValueError("dpo_criterion should be None or DPOCriterion. Got {}".format(type(dpo_criterion)))
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
            self.ref_model_wrapped = self._wrap_ref_model(self.ref_model)
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
            raise NotImplementedError("compute_metrics is not supported for DPOTrainer")
        self.reset_dpo_infohub()

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
            if "score_deltas" in batch:
                dpo_inputs["score_deltas"] = batch["score_deltas"]
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
            labels = labels[:-2] + (reference_chosen_logps, reference_rejected_logps)
            logits = model(**dpo_inputs)
            policy_chosen_logps, policy_rejected_logps, sft_loss, dpo_loss, loss = self.dpo_criterion(logits, labels)

        # metrics
        metric_inputs = dict(
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            dpo_loss=dpo_loss,
            sft_loss=sft_loss,
            train_eval=train_eval,
        )
        self.log_metric(**metric_inputs)
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
            raise NotImplementedError("DPOTrainer only supports prediction_loss_only=True for now.")

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
            labels = labels[:-2] + (reference_chosen_logps, reference_rejected_logps)
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
            labels = labels[:-2] + (reference_chosen_logps, reference_rejected_logps)
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
        "score_deltas",
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
