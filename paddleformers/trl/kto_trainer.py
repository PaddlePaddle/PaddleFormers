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

""" KTO Trainer """
from collections import OrderedDict, defaultdict

import paddle
from paddle.distributed import fleet

from ..trainer import Trainer
from ..transformers.kto_criterion import KTOCriterion
from ..transformers.model_utils import unwrap_model
from ..utils import infohub


def disable_dropout_in_model(model: paddle.nn.Layer) -> None:
    """ "disable dropout"""
    for module in model.children():
        if isinstance(module, paddle.nn.Dropout):
            module.p = 0


try:
    from ..peft.lora.lora_model import AVAILABLE_LAYERS
except:
    from ..peft.lora.lora_model import AVALIABLE_LAYERS

    AVAILABLE_LAYERS = AVALIABLE_LAYERS

KTO_INFO_KEYS = [
    "reference_chosen_logps",
    "reference_rejected_logps",
    "reference_kl_logps",
    "policy_chosen_logps",
    "policy_rejected_logps",
    "policy_kl_logps",
    "kl",
]


class KTOTrainer(Trainer):
    """
    Initialize KTOTrainer.
    """

    def __init__(
        self,
        model,
        data_collator,
        ref_model=None,
        kto_config=None,
        disable_dropout: bool = True,
        padding_value: int = 0,
        kto_criterion=None,
        ignore_label: int = 0,
        **kwargs,
    ):
        super().__init__(model, data_collator=data_collator, **kwargs)
        if kto_config is None:
            raise ValueError("kto_config is None")
        else:
            self.kto_config = kto_config
        if ref_model:
            self.ref_model = ref_model
            self.ref_model_wrapped = self._wrap_ref_model(self.ref_model)
            self.ref_model_wrapped.eval()
        elif self.kto_config.lora:
            self.ref_model = None
            self.ref_model_wrapped = None
        else:
            raise ValueError("ref_model is None! KTO requires a reference model")
        if not self.args.pipeline_parallel_degree > 1:
            if kto_criterion is None:
                self.kto_criterion = KTOCriterion(self.model.config, kto_config=kto_config, ignore_label=ignore_label)
            elif isinstance(kto_criterion, KTOCriterion):
                self.kto_criterion = kto_criterion
            else:
                raise ValueError(f"kto_criterion should be None or KTOCriterion. Got {type(kto_criterion)}")
        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.padding_value = padding_value
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.reset_dpo_infohub()

    def get_batch_metrics(self, ref_model, model, batch, train_eval="train"):
        """Compute the KTO loss and other metrics for the given batch of inputs for train or test."""
        inputs = {
            "input_ids": batch["input_ids"],
            "position_ids": batch["position_ids"],
        }
        if "attention_mask" in batch:
            inputs["attention_mask"] = batch["attention_mask"]
        elif "attn_mask_start_row_indices" in batch:
            inputs["attn_mask_start_row_indices"] = batch["attn_mask_start_row_indices"]
        elif "attn_mask_startend_row_indices" in batch:
            inputs["attn_mask_startend_row_indices"] = batch["attn_mask_startend_row_indices"]
        else:
            raise ValueError("No attention mask found in batch.")
        labels = (
            batch["response_labels"],
            batch["response_kl_labels"],
            batch["response_indexs"],
            None,
            None,
            None,
        )
        with paddle.no_grad():
            if self.kto_config.lora:
                self.disable_lora(model)
                model.eval()
                logits = model(**inputs)
                self.enable_lora(model)
                model.train()
            else:
                logits = ref_model(**inputs)
            (
                reference_chosen_logps,
                reference_rejected_logps,
                reference_kl_logps,
            ) = self.kto_criterion(logits, labels)
        labels = labels[:3] + (
            reference_chosen_logps,
            reference_rejected_logps,
            reference_kl_logps,
        )
        logits = model(**inputs)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_kl_logps,
            loss,
            kl,
        ) = self.kto_criterion(logits, labels)

        # metrics
        metric_inputs = dict(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            kl=kl,
            train_eval=train_eval,
        )
        self.log_metric(**metric_inputs)
        return loss

    def log_metric(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        kl,
        train_eval,
    ):
        metrics = {}
        chosen_rewards = self.kto_config.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.kto_config.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}count/chosen"] = paddle.to_tensor(chosen_rewards.shape[0])
        metrics[f"{prefix}count/rejected"] = paddle.to_tensor(rejected_rewards.shape[0])

        if policy_chosen_logps.shape[0] == 0 or len(reference_chosen_logps.shape) == 0:
            metrics[f"{prefix}rewards/chosen"] = paddle.zeros([])
            metrics[f"{prefix}logps/chosen"] = paddle.zeros([])
        else:
            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean()
            metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean()
        if policy_rejected_logps.shape[0] == 0 or reference_rejected_logps.shape[0] == 0:
            metrics[f"{prefix}rewards/rejected"] = paddle.zeros([])
            metrics[f"{prefix}logps/rejected"] = paddle.zeros([])
        else:
            metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean()
            metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean()

        for key in metrics:
            if "count" in key:
                metrics[key] = self._nested_gather(paddle.tile(metrics[key], repeat_times=[1, 1])).sum().cpu()
                metrics[key] /= max(self.args.tensor_parallel_degree, 1)
            else:
                metrics[key] = self._nested_gather(paddle.tile(metrics[key], repeat_times=[1, 1])).mean().cpu()
        metrics[f"{prefix}kl"] = kl
        metrics[f"{prefix}rewards/margins"] = metrics[f"{prefix}rewards/chosen"] - metrics[f"{prefix}rewards/rejected"]
        if self.args.should_save:
            self.store_metrics(metrics, train_eval=train_eval)

    def compute_loss(self, model, inputs):
        """Compute the KTO loss for the given batch of inputs."""
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
            raise NotImplementedError("KTOTrainer only supports prediction_loss_only=True for now.")

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
            if "count" in key:
                logs[key] = paddle.to_tensor(metrics).sum().item()
            else:
                logs[key] = paddle.to_tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        if self.state.epoch is not None and train_eval == "train":
            self.state.epoch *= self.args.num_train_epochs
        return super().log(logs, **kwargs)

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
        concatenated_inputs["reference_kl_logps"] = None
        self._pp_data_buffer = []
        inputs, labels = model._prepare_pipeline_inputs_func(concatenated_inputs)
        model_config_backup = model.micro_batch_size, model.accumulate_steps
        model.micro_batch_size = self.args.per_device_train_batch_size
        model.accumulate_steps = self.args.gradient_accumulation_steps

        if self.kto_config.lora:
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
            ref_model_config_backup = (
                ref_model.micro_batch_size,
                ref_model.accumulate_steps,
            )
            ref_model.accumulate_steps = model.accumulate_steps
            ref_model.micro_batch_size = model.micro_batch_size
            with paddle.no_grad():
                with self.autocast_smart_context_manager():
                    ref_model.eval_batch(data=[inputs, labels], compute_loss=True)
            ref_model.micro_batch_size, ref_model.accumulate_steps = ref_model_config_backup
        reference_chosen_logps = infohub.reference_chosen_logps
        reference_rejected_logps = infohub.reference_rejected_logps
        reference_kl_logps = infohub.reference_kl_logps

        if model.is_pipeline_last_stage(ignore_virtual=model._layers._num_virtual_pipeline_stages > 1):
            labels = labels[:3] + (
                reference_chosen_logps,
                reference_rejected_logps,
                reference_kl_logps,
            )
        train_inputs = [inputs, labels]
        train_inputs = model._prepare_training(train_inputs, self.optimizer, self.lr_scheduler)
        model.optimizer = None  # we do not use `PipelineParallel` to handler optimizer step
        model.lr_scheduler = None
        with self.autocast_smart_context_manager():
            loss = model.forward_backward_pipeline(train_inputs, self.scaler if self.do_grad_scaling else None)
        model.micro_batch_size, model.accumulate_steps = model_config_backup

        # broadcast KTO_INFO_KEYS
        self.broadcast_last_stage_infohub_tensor()

        # metrics
        metric_inputs = dict(
            policy_chosen_logps=infohub.policy_chosen_logps,
            policy_rejected_logps=infohub.policy_rejected_logps,
            reference_chosen_logps=infohub.reference_chosen_logps,
            reference_rejected_logps=infohub.reference_rejected_logps,
            kl=infohub.kl,
            train_eval="train",
        )
        self.log_metric(**metric_inputs)
        self.reset_dpo_infohub()
        return loss.detach()

    def prediction_pipeline_step(
        self,
        ref_model,
        model,
        batch,
    ):
        """
        prediction_step function for pipeline parallel mode.
        """
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
                            range(
                                i * per_device_train_batch_size,
                                (i + 1) * per_device_train_batch_size,
                            )
                        ):
                            response_index[0] -= i * per_device_train_batch_size
                            concatenated_inputs["response_indexs"][i].append(response_index)
                    concatenated_inputs["response_indexs"][i] = paddle.stack(concatenated_inputs["response_indexs"][i])
                    if model._layers.config.use_sparse_head_and_loss_fn:
                        last_batch_response_length = concatenated_inputs["response_indexs"][i][0, 1]
                        concatenated_inputs["response_indexs"][i][:, 1:] -= last_batch_response_length

        concatenated_inputs["reference_chosen_logps"] = None
        concatenated_inputs["reference_rejected_logps"] = None
        concatenated_inputs["reference_kl_logps"] = None

        self._pp_data_buffer = []
        inputs, labels = model._prepare_pipeline_inputs_func(concatenated_inputs)

        if self.kto_config.lora:
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
        reference_kl_logps = infohub.reference_kl_logps

        if model.is_pipeline_last_stage(ignore_virtual=model._layers._num_virtual_pipeline_stages > 1):
            labels = labels[:3] + (
                reference_chosen_logps,
                reference_rejected_logps,
                reference_kl_logps,
            )
        with paddle.no_grad():
            with self.autocast_smart_context_manager():
                loss = model.eval_batch(data=[inputs, labels], compute_loss=True)

        # broadcast KTO_INFO_KEYS
        self.broadcast_last_stage_infohub_tensor()
        # metrics
        metric_inputs = dict(
            policy_chosen_logps=infohub.policy_chosen_logps,
            policy_rejected_logps=infohub.policy_rejected_logps,
            reference_chosen_logps=infohub.reference_chosen_logps,
            reference_rejected_logps=infohub.reference_rejected_logps,
            kl=infohub.kl,
            train_eval="eval",
        )
        self.log_metric(**metric_inputs)
        self.reset_dpo_infohub()
        return (loss, None, None)

    def reset_dpo_infohub(self):
        """Initialize infohub"""
        for key in KTO_INFO_KEYS:
            setattr(infohub, key, [])

    def broadcast_last_stage_infohub_tensor(self):
        for key in KTO_INFO_KEYS:
            if self.model_wrapped.is_pipeline_last_stage(
                ignore_virtual=self.model_wrapped._layers._num_virtual_pipeline_stages > 1
            ):
                if key == "kl":
                    tensor = paddle.stack(getattr(infohub, key)).mean().detach()
                elif "logps" in key:
                    logps_list = getattr(infohub, key)
                    if all(logps.shape == [0] for logps in logps_list):
                        tensor = paddle.zeros([1])
                    else:
                        tensor = paddle.concat(getattr(infohub, key), axis=0).detach()
                    tensor_shape = paddle.to_tensor(tensor.shape, dtype="int64")
                    paddle.distributed.broadcast(
                        tensor_shape,
                        src=self.model_wrapped.global_rank,
                        group=self.model_wrapped.pp_group,
                    )
                else:
                    raise ValueError(f"Invalid key: {key}")
                paddle.distributed.broadcast(
                    tensor,
                    src=self.model_wrapped.global_rank,
                    group=self.model_wrapped.pp_group,
                )
            else:
                if key == "kl":
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
            "attn_mask_startend_row_indices",
            "position_ids",
        ]

    last_stage_keys = [
        "response_labels",
        "response_kl_labels",
        "response_indexs",
        "reference_chosen_logps",
        "reference_rejected_logps",
        "reference_kl_logps",
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
