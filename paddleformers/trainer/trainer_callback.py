# Copyright 2020-present the HuggingFace Inc. team.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# This file is modified from
#  https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py
"""
Callbacks to use with the Trainer class and customize the training loop.
"""
import dataclasses
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.fleet import fleet
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients_with_group,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    is_sequence_parallel_parameter,
)
from tqdm.auto import tqdm

from ..transformers.moe_gate import PretrainedMoEGate
from ..transformers.moe_utils import offload, reload
from ..utils.log import logger
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments

__all__ = [
    "TrainerState",
    "TrainerControl",
    "TrainerCallback",
    "CallbackHandler",
    "DefaultFlowCallback",
    "ProgressCallback",
    "PrinterCallback",
    "EarlyStoppingCallback",
    "StepFlexToken",
    "FP8QuantWeightCallback",
    "MoECorrectionBiasAdjustCallback",
    "MoeExpertsGradScaleCallback",
    "MoEGateSpGradSyncCallBack",
    "SPGradSyncCallback",
]


@dataclass
class TrainerState:
    """
    A class containing the [`Trainer`] inner state that will be saved along the model and optimizer when checkpointing
    and passed to the [`TrainerCallback`].

    <Tip>

    In all this class, one step is to be understood as one update step. When using gradient accumulation, one update
    step may require several forward and backward passes: if you use `gradient_accumulation_steps=n`, then one update
    step requires going through *n* batches.

    </Tip>

    Args:
        epoch (`float`, *optional*):
            Only set during training, will represent the epoch the training is at (the decimal part being the
            percentage of the current epoch completed).
        global_step (`int`, *optional*, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (`int`, *optional*, defaults to 0):
            The number of update steps to do during the current training.
        total_flos (`float`, *optional*, defaults to 0):
            The total number of floating operations done by the model since the beginning of training (stored as floats
            to avoid overflow).
        log_history (`List[Dict[str, float]]`, *optional*):
            The list of logs done since the beginning of training.
        best_metric (`float`, *optional*):
            When tracking the best model, the value of the best metric encountered so far.
        best_model_checkpoint (`str`, *optional*):
            When tracking the best model, the value of the name of the checkpoint for the best model encountered so
            far.
        is_local_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
            several machines) main process.
        is_world_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the global main process (when training in a distributed fashion on several
            machines, this is only going to be `True` for one process).
    """

    epoch: Optional[float] = None
    global_step: int = 0
    consumed_samples: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    total_flos: float = 0
    log_history: List[Dict[str, float]] = None
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    trial_name: str = None
    trial_params: Dict[str, Union[str, float, int, bool]] = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))


@dataclass
class TrainerControl:
    """
    A class that handles the [`Trainer`] control flow. This class is used by the [`TrainerCallback`] to activate some
    switches in the training loop.

    Args:
        should_training_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the training should be interrupted.

            If `True`, this variable will not be set back to `False`. The training will just stop.
        should_epoch_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the current epoch should be interrupted.

            If `True`, this variable will be set back to `False` at the beginning of the next epoch.
        should_save (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be saved at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_evaluate (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be evaluated at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_log (`bool`, *optional*, defaults to `False`):
            Whether or not the logs should be reported at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
    """

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_save_hf: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        self.should_epoch_stop = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        self.should_save = False
        self.should_save_hf = False
        self.should_evaluate = False
        self.should_log = False


class TrainerCallback:
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:

    Args:
        args ([`TrainingArguments`]):
            The training arguments used to instantiate the [`Trainer`].
        state ([`TrainerState`]):
            The current state of the [`Trainer`].
        control ([`TrainerControl`]):
            The object that is returned to the [`Trainer`] and can be used to make some decisions.
        model ([`PreTrainedModel`] or `paddle.nn.Layer`):
            The model being trained.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for encoding the data.
        optimizer (`paddle.optimizer.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (`paddle.optimizer.lr.LRScheduler`):
            The scheduler used for setting the learning rate.
        train_dataloader (`paddle.io.DataLoader`, *optional*):
            The current dataloader used for training.
        eval_dataloader (`paddle.io.DataLoader`, *optional*):
            The current dataloader used for training.
        metrics (`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event `on_evaluate`.
        logs  (`Dict[str, float]`):
            The values to log.

            Those are only accessible in the event `on_log`.

    The `control` object is the only one that can be changed by the callback, in which case the event that changes it
    should return the modified version.

    The argument `args`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`.
    You can unpack the ones you need in the signature of the event using them. As an example, see the code of the
    simple [`~transformer.PrinterCallback`].

    Example:

    ```python
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                logger.info(logs)
    ```"""

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_load_data_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_optimizer_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_optimizer_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        pass

    def on_save_hf(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a huggingface checkpoint save.
        """
        pass


class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""

    def __init__(self, callbacks, model, tokenizer, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                "The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n"
                + "should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of"
                + "callbacks is\n:"
                + self.callback_list
            )

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event("on_train_end", args, state, control, **kwargs)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        control.should_save_hf = False
        return self.call_event("on_step_begin", args, state, control)

    def on_load_data_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, inputs: Dict):
        return self.call_event("on_load_data_end", args, state, control, inputs=inputs)

    def on_optimizer_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, scaler):
        return self.call_event("on_optimizer_begin", args, state, control, scaler=scaler)

    def on_optimizer_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, scaler):
        return self.call_event("on_optimizer_end", args, state, control, scaler=scaler)

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_substep_end", args, state, control)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_save_hf(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_save_hf = False
        return self.call_event("on_save_hf", args, state, control)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs, **kwargs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs, **kwargs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_prediction_step", args, state, control)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class DefaultFlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % args.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step % args.eval_steps == 0:
            control.should_evaluate = True

        # Save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and args.save_steps > 0
            and state.global_step % args.save_steps == 0
        ):
            control.should_save = True

        # For Flash save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and args.flash_device_save_steps > 0
            and state.global_step % args.flash_device_save_steps == 0
        ):
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        # Save hf
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and args.save_hf_steps > 0
            and state.global_step % args.save_hf_steps == 0
        ):
            control.should_save_hf = True

        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.EPOCH:
            control.should_evaluate = True

        # Save
        if args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True

        return control


class ProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(total=state.max_steps, desc="TrainProcess")
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_local_process_zero and has_length(eval_dataloader.dataset):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader), leave=self.training_bar is None, desc="PredictProcess"
                )
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            if type(logs) is dict:
                logs_str = ", ".join(f"{k}: {v}" for k, v in logs.items())
            else:
                logs_str = str(logs)
            logger.info(logs_str)

    def on_train_end(self, args, state, control, **kwargs):
        metrics_dumper = kwargs.get("metrics_dumper", None)
        if metrics_dumper is not None:
            metrics_dumper.close()
        if state.is_local_process_zero:
            self.training_bar.close()
            self.training_bar = None


class PrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            if type(logs) is dict:
                logger.info(", ".join(f"{k}: {v}" for k, v in logs.items()))
                metrics_dumper = kwargs.get("metrics_dumper", None)
                if metrics_dumper is not None:
                    metrics_dumper.append(logs)
            else:
                logger.info(logs)


class EarlyStoppingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.

    Args:
       early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
       early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`].
    """

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
            args.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True


class StepFlexToken(TrainerCallback):
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        model = kwargs.pop("model")
        if hasattr(model, "step_flex_token"):
            model.step_flex_token(state.global_step)


g_shard_bypass_dygraph_optimizer = int(os.environ.get("FLAGS_shard_bypass_dygraph_optimizer", 0))


def enable_in_dict_config(config, key):
    """enable_in_dict_config"""
    return key in config and config[key]


skip_count = 0


class FP8QuantWeightCallback(TrainerCallback):
    """
    Callback for FP8 weight quantization during training
    """

    def on_step_begin(self, args, state, control, **kwargs):
        """
        Quantize expert weights to FP8 before each training step
        """
        model = kwargs["model"]
        optimizer = kwargs["optimizer"]
        global skip_count

        if (not g_shard_bypass_dygraph_optimizer or skip_count == 0) and hasattr(model, "fp8_quant_weight"):
            model.fp8_quant_weight(True, quant_transpose=True)
            optimizer.clear_param_storage("moe_expert")
            optimizer.clear_param_storage("rms_linear")
            optimizer.clear_param_storage("memory_attn")
            optimizer.clear_param_storage("attn_out_project")
            optimizer.clear_param_storage("shared_expert")

            self.moe_weights_name = []
            for param in optimizer._inner_opt._parameter_list:
                color = getattr(param, "color", -1)
                if isinstance(color, dict) and color["color"] == "moe_expert":
                    self.moe_weights_name.append(param.name)

            for name in self.moe_weights_name:
                offload(optimizer._master_weights[name])

        skip_count += 1

    def on_optimizer_begin(self, args, state, control, **kwargs):
        """
        Reload weights before optimizer step
        """
        model = kwargs["model"]
        optimizer = kwargs["optimizer"]
        global skip_count

        if (not g_shard_bypass_dygraph_optimizer) and hasattr(model, "fp8_quant_weight"):
            for name in self.moe_weights_name:
                reload(optimizer._master_weights[name])


class MoECorrectionBiasAdjustCallback(TrainerCallback):
    """
    used for moe aux loss free balance
    """

    def __init__(self, lr=0.001, use_mp=False):
        super().__init__()
        self.update_lr = lr
        self.use_mp = use_mp

    def on_optimizer_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        biases = []
        usages = []

        def get_stat(layer):
            if isinstance(layer, PretrainedMoEGate) and layer.topk_method == "noaux_tc":
                biases.append(layer.e_score_correction_bias)
                usages.append(layer.expert_usage)

        model.apply(get_stat)

        if not usages:
            return
        usages_tensor = paddle.stack(usages, 0)  # [num_layers, num_local_experts]
        if not hasattr(fleet, "_hcg"):
            dist.all_reduce(usages_tensor)
            return

        hcg = fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        dp_group = hcg.get_data_parallel_group()
        sd_group = hcg.get_sharding_parallel_group()

        if self.use_mp and mp_group.nranks > 1:
            dist.all_reduce(usages_tensor, group=mp_group)
        if dp_group.nranks > 1:
            dist.all_reduce(usages_tensor, group=dp_group)
        if sd_group.nranks > 1:
            dist.all_reduce(usages_tensor, group=sd_group)

        usages_mean = usages_tensor.mean(-1, keepdim=True)
        update = paddle.sign(usages_mean - usages_tensor) * self.update_lr
        update = update.astype(paddle.float32)
        update_list = list(update)

        # print('on_optimizer_end bias:', [bias.tolist() for bias in biases])
        # print('on_optimizer_end usage:', usages_tensor.tolist())
        # print('on_optimizer_end update:', update.tolist())

        def update_bias(layer):
            if isinstance(layer, PretrainedMoEGate) and layer.topk_method == "noaux_tc":
                with paddle.no_grad():
                    if not layer.weight.stop_gradient:
                        biases.pop(0).add_(update_list.pop(0))
                    usages.pop(0).zero_()

        model.apply(update_bias)


class MoeExpertsGradScaleCallback(TrainerCallback):
    """
    This hook is used to correct the issue where the gradients of expert parameters are amplified by a factor of N.
    """

    def __init__(self, args):
        """_summary_
        Args:
            args (_type_): _description_
        """
        if not args.use_expert_parallel:
            raise ValueError("This callback should be used with expert parallel")
        if args.expert_parallel_degree > 1:
            self.expert_gradient_scaling_factor = 1.0 / args.expert_parallel_degree
            if args.tensor_parallel_degree > 1:
                self.expert_gradient_scaling_factor *= args.tensor_parallel_degree
            logger.info(
                f"EP-MoE is used, expert gradient scaling factor is set to {self.expert_gradient_scaling_factor}"
            )

    def on_optimizer_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        param_count = 0
        for p in model.parameters():
            if not getattr(p, "no_sync", False):
                continue
            if hasattr(p, "is_moe_param") and p.is_moe_param:
                with paddle.no_grad():
                    if hasattr(p, "main_grad") and p.main_grad is not None:
                        p.main_grad.scale_(self.expert_gradient_scaling_factor)
                        param_count += 1
                    elif p.grad is not None:
                        p.grad.scale_(self.expert_gradient_scaling_factor)
                        param_count += 1
        logger.info("correct ep grad count:{}".format(param_count))


class MoEGateSpGradSyncCallBack(TrainerCallback):
    """
    用于绕过sp allreduce hook被错误调用多次的bug，此bug是框架内部机制的问题，将来会进行修复。
    目前仅gate的梯度在开启moe_subbatch_token_num存在这个问题，因此这里只添加gate的梯度聚合。
    但保险起见mark_as_sequence_parallel_parameter的参数最好都通过类似的hook处理。
    """

    def __init__(self):
        logger.info("MoEGateSpGradSyncCallBack Created")

    def on_optimizer_begin(self, args, state, control, **kwargs):
        if args.tensor_parallel_degree > 1 and args.sequence_parallel:
            model = kwargs["model"]
            hcg = fleet.get_hybrid_communicate_group()
            pg = hcg.get_model_parallel_group().process_group
            for param in model.parameters():
                if not getattr(param, "is_gate", False):
                    continue
                grad = getattr(param, "main_grad", None)
                if grad is None:
                    grad = getattr(param, "grad", None)
                if grad is None:
                    continue
                pg.allreduce(grad).wait()

            logger.info("MoEGate grad allreduced done")


class SPGradSyncCallback(TrainerCallback):
    """
    SPGradSyncCallback
    只能在非 sharding stage2 的情况下使用。
    开启sharding stage2 时，在 `on_optimizer_begin` 的时候 grad 已经被清空了
    """

    def __init__(self, model):
        assert hasattr(fleet, "_hcg"), "must use MP when calling this Callback"
        logger.info("using sp callback")
        params = []
        self.model = model
        for n, p in model.named_parameters():
            if is_sequence_parallel_parameter(p):
                logger.info(f"register bw hook for:{n}")
                params.append(p)

        logger.info(f"#-sp-sync param:{len(params)}")
        self._sp_params = params

    def on_optimizer_begin(self, args, state, control, **kwargs):
        """on_optimizer_begin"""
        if self._sp_params:
            now = time.time()
            mp_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
            fused_allreduce_gradients_with_group(self._sp_params, group=mp_group, scale=1.0)  # sum not mean
            another_time = time.time()
            logger.info(f"sync gradients takes {another_time - now} time")
