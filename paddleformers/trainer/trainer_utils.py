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
#  https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_utils.py

"""
Utilities for the Trainer class.
"""
import datetime
import gc
import inspect
import json
import math
import os
import random
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.io import IterableDataset
from paddle.optimizer.lr import LambdaDecay

from ..ops import Topology
from ..trainer.argparser import strtobool
from ..transformers.tokenizer_utils_base import BatchEncoding
from ..utils.env import PREFIX_CHECKPOINT_DIR, _re_checkpoint  # noqa for compatibility
from ..utils.fault_tolerance import PDC_DOWNLOAD_ERROR
from ..utils.import_utils import is_paddle_cuda_available, is_psutil_available
from ..utils.log import logger
from ..utils.pdc_sdk import PDCErrorCode, PDCErrorMessageMap, pdc_tool
from .utils.helper import distributed_file

__all__ = [
    "TrainOutput",
    "PredictionOutput",
    "EvalPrediction",
    "IntervalStrategy",
    "SchedulerType",
    "set_seed",
    "speed_metrics",
    "get_last_checkpoint",
    "get_scheduler",
    "set_hyrbid_parallel_seed",
    "log_trainer_start",
]


def log_trainer_start():
    if "MAIN_PROCESS_STARTED" not in os.environ:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"The Training Main Process Started Successfully. time: {start_time}, pid: {os.getpid()}")
        os.environ["MAIN_PROCESS_STARTED"] = "1"


def _get_distributed_seeds(seed: int = 1234, topo: Topology = None):
    """
    Get the seeds from distributed environment strategy.
    Args:
        seed (:obj:`int`, `optional`, defaults to 1234): The seeds for initializing distributed training.
        topo (:obj:`Topology`, `optional`, defaults to None): The topology of hybrid parallel in semi-auto mode.
    Returns:
        Tuple[int, int]: The global seed and local seed respectively.
    """

    # NOTE: For parameter init seed:
    # seed: dp/mp_undistributed_parameter/sharding is same; others is different
    # For compute seed(dropout):
    # global seed: only mp group is same.
    # local seed: all groups are different
    hcg = None
    if hasattr(fleet.fleet, "_hcg") and topo is None:
        hcg = fleet.get_hybrid_communicate_group()

    if topo is not None and paddle.distributed.get_world_size() > 1:
        dp_rank = topo.dp_info.rank
        dp_size = topo.dp_info.size

        pp_rank = topo.pp_info.rank
        pp_size = topo.pp_info.size

        mp_rank = topo.mp_info.rank
        mp_size = topo.mp_info.size

        sep_rank = topo.sep_info.rank
        sep_size = topo.sep_info.size

        sharding_rank = topo.sharding_info.rank
    elif hcg is not None and paddle.distributed.get_world_size() > 1:
        # obtain rank message of hybrid parallel

        mp_rank = hcg.get_model_parallel_rank()
        mp_size = hcg.get_model_parallel_world_size()

        if hasattr(hcg, "get_sep_parallel_rank"):
            sep_rank = hcg.get_sep_parallel_rank()
            sep_size = hcg.get_sep_parallel_world_size()
        else:
            sep_rank, sep_size = 0, 1

        pp_rank = hcg.get_stage_id()
        pp_size = hcg.get_pipe_parallel_world_size()

        dp_rank = hcg.get_data_parallel_rank()
        dp_size = hcg.get_data_parallel_world_size()

        sharding_rank = hcg.get_sharding_parallel_rank()
    else:
        mp_rank, mp_size = 0, 1
        sep_rank, sep_size = 0, 1
        pp_rank, pp_size = 0, 1
        dp_rank, dp_size = 0, 1
        sharding_rank, _ = 0, 1

    seed_offset = seed
    global_seed = (
        seed_offset
        + sep_rank * (mp_size)
        + pp_rank * (mp_size * sep_size)
        + dp_rank * (mp_size * sep_size * pp_size)
        + sharding_rank * (mp_size * sep_size * pp_size * dp_size)
    )

    seed_offset += paddle.distributed.get_world_size()
    local_seed = (
        seed_offset
        + mp_rank
        + sep_rank * (mp_size)
        + pp_rank * (mp_size * sep_size)
        + dp_rank * (mp_size * sep_size * pp_size)
        + sharding_rank * (mp_size * sep_size * pp_size * dp_size)
    )

    # NOTE: the commented seeds are set only for precision validation
    random_seed = seed + 100 * pp_rank

    return global_seed, local_seed, random_seed


def set_seed(seed: int = 1234, topo=None):
    global_seed, local_seed, random_seed = _get_distributed_seeds(seed, topo)

    tracker = get_rng_state_tracker()
    if "global_seed" not in tracker.states_ and global_seed not in tracker.seeds_:
        tracker.add("global_seed", global_seed)

    if "local_seed" not in tracker.states_ and local_seed not in tracker.seeds_:
        tracker.add("local_seed", local_seed)

    paddle.seed(global_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    logger.info(
        "The global seed is set to {}, local seed is set to {} and "
        "random seed is set to {}.".format(global_seed, local_seed, random_seed)
    )


def _switch_mode(mode="dynamic"):
    assert mode in ["dynamic", "static"]
    if mode == "dynamic":
        paddle.disable_static()
    else:
        paddle.enable_static()


@contextmanager
def _exec_mode_guard(mode="dynamic"):
    origin_mode = "dynamic" if paddle.in_dynamic_mode() else "static"
    _switch_mode(mode)
    try:
        yield
    finally:
        _switch_mode(origin_mode)


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]


def _check_checkpoint_files(
    folder_path, world_size, ignore_save_lr_and_optim, skip_save_model_weight, remove_master_weight
):
    files = os.listdir(folder_path)
    model_weight_files = [f for f in files if f.startswith(".model_weight")]
    a = len(model_weight_files) == world_size
    if not ignore_save_lr_and_optim:
        b = True
        if not skip_save_model_weight or not remove_master_weight:
            master_weight_file = [f for f in files if f.startswith(".master_weight")]
            b = len(master_weight_file) == world_size
        optimizer_file = [f for f in files if f.startswith(".optimizer_weight")]
        c = len(optimizer_file) == world_size
        return a and b and c
    else:
        return a


def get_last_checkpoint(folder, signal_folder=None, uc_async_save=False):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return

    if uc_async_save:
        assert signal_folder is not None

    if strtobool(os.getenv("FLAG_LLM_PDC", "False")):
        for i in sorted(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0]), reverse=True):
            current_path = os.path.join(folder, i)
            # make sure the checkpoint is valid
            if not uc_async_save:
                if os.path.exists(os.path.join(current_path, ".checkpoint_done")):
                    return current_path
            else:
                saving_info = paddle.load(distributed_file(os.path.join(current_path, ".saving_info")))
                current_signal_path = os.path.join(signal_folder, i)
                pre_world_size = saving_info.get("world_size", 1)
                ignore_save_lr_and_optim = saving_info.get("ignore_save_lr_and_optim", False)
                skip_save_model_weight = saving_info.get("skip_save_model_weight", False)
                remove_master_weight = saving_info.get("remove_master_weight", False)
                if _check_checkpoint_files(
                    current_signal_path,
                    pre_world_size,
                    ignore_save_lr_and_optim,
                    skip_save_model_weight,
                    remove_master_weight,
                ):
                    return current_path
        return
    else:
        return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class EvaluationStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAMW = "adamw"
    ADAFACTOR = "adafactor"
    ADAMW_MINI = "adamw_mini"
    ADAMW_CUSTOM = "adamw_custom"


class ShardingOption(ExplicitEnum):
    """
    Sharding Option
    OP for sharding optimizer state
    GRAD for sharding gradients
    FULL_SHARD for sharding optimizer gradient and parameter
    OFFLOAD means offload to cpu.
    """

    SHARD_OP = "stage1"
    SHARD_GRAD_OP = "stage2"
    FULL_SHARD = "stage3"
    # NO_SHARD = "no"
    OFFLOAD = "offload"


def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `xm.get_ordinal()` (for TPUs) first, then on
    `local_rank`.
    """

    return local_rank in [-1, 0]


def total_processes_number(local_rank):
    """
    Return the number of processes launched in parallel. Works with `paddle.distributed` and TPUs.
    """
    if local_rank != -1:
        import paddle

        return paddle.distributed.get_world_size()
    return 1


def speed_metrics(split, start_time, num_samples=None, num_steps=None, seq_length=None, model_flops_per_token=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:

    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 4)
        if seq_length is not None:
            tokens_per_second_per_device = samples_per_second * seq_length / paddle.distributed.get_world_size()
            result[f"{split}_tokens_per_second_per_device"] = round(tokens_per_second_per_device, 4)
        if model_flops_per_token is not None:
            result[f"{split}_hardware_tflops_per_device"] = round(
                tokens_per_second_per_device * model_flops_per_token / 2**40, 2
            )

    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 4)
    return result


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    POLYNOMIAL = "polynomial"


def get_constant_schedule(learning_rate: float, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.
    Args:
        learning_rate (float)
            The initial learning rate. It is a python float number.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `paddle.optimizer.lr.LambdaDecay` with the appropriate schedule.
    """
    return LambdaDecay(learning_rate, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(learning_rate: float, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    Args:
        learning_rate (float)
            The initial learning rate. It is a python float number.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `paddle.optimizer.lr.LambdaDecay` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaDecay(learning_rate, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(learning_rate: float, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        learning_rate (float)
            The initial learning rate. It is a python float number.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `paddle.optimizer.lr.LambdaDecay` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaDecay(learning_rate, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    learning_rate: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float = 0.0,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        learning_rate (float)
            The initial learning rate. It is a python float number.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `paddle.optimizer.lr.LambdaDecay` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return ratio * (1 - min_lr / learning_rate) + min_lr / learning_rate

    return LambdaDecay(learning_rate, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    learning_rate: float,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.
    Args:
        learning_rate (`float`):
            The base learning rate. It is a python float number.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37
    Return:
        `paddle.optimizer.lr.LambdaDecay` with the appropriate schedule.
    """

    lr_init = learning_rate
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaDecay(learning_rate, lr_lambda, last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    learning_rate: float,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: Optional[float] = 0.5,
    lr_end: Optional[float] = 1e-7,
    power: Optional[float] = 1.0,
    min_lr: Optional[float] = 0.0,
):
    """
    Unified API to get any scheduler from its name.
    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        learning_rate (float)
            The initial learning rate. It is a python float number.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles (``float``, *optional*):
            The number of waves in the cosine scheduler (the defaults is to just decrease from the max value to 0
            following a half-cosine). This is not required by all schedulers (hence the argument being optional)
        lr_end (``float``, *optional*):
            The end LR in the polynomial scheduler. This is not required by all schedulers (hence the argument
            being optional).
        power (``float``, *optional*):
            The power factor in the polynomial scheduler. This is not required by all schedulers (hence the argument
            being optional).
        min_lr (``float``, *optional*):
            The minimum LR in the cosine scheduler. This is not required by all schedulers (hence the argument
            being optional).
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(learning_rate)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(learning_rate, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE:
        return schedule_func(
            learning_rate,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            min_lr=min_lr,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            learning_rate,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_end=lr_end,
            power=power,
        )

    return schedule_func(learning_rate, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


def _secs2timedelta(secs):
    """
    convert seconds to hh:mm:ss.msec, msecs rounded to 2 decimals
    """

    msec = int(abs(secs - int(secs)) * 100)
    return f"{datetime.timedelta(seconds=int(secs))}.{msec:02d}"


def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Reformat Trainer metrics values to a human-readable format
    Args:
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predict
    Returns:
        metrics (`Dict[str, float]`): The reformatted metrics
    """

    metrics_copy = metrics.copy()
    for k, v in metrics_copy.items():
        if "_mem_" in k:
            metrics_copy[k] = f"{ v >> 20 }MB"
        elif "_runtime" in k:
            metrics_copy[k] = _secs2timedelta(v)
        elif k == "total_flos":
            metrics_copy[k] = f"{ int(v) >> 30 }GF"
        elif isinstance(metrics_copy[k], float):
            metrics_copy[k] = round(v, 4)

    return metrics_copy


def log_metrics(self, split, metrics):
    """
    Log metrics in a specially formatted way
    Under distributed environment this is done only for a process with rank 0.
    Args:
        split (`str`):
            Mode/split name: one of `train`, `eval`, `test`
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predictmetrics: metrics dict
    """
    if not self.is_world_process_zero():
        return

    logger.info(f"***** {split} metrics *****")
    metrics_formatted = self.metrics_format(metrics)
    k_width = max(len(str(x)) for x in metrics_formatted.keys())
    v_width = max(len(str(x)) for x in metrics_formatted.values())
    for key in sorted(metrics_formatted.keys()):
        logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")


def save_metrics(self, split, metrics, combined=True):
    """
    Save metrics into a json file for that split, e.g. `train_results.json`.
    Under distributed environment this is done only for a process with rank 0.
    Args:
        split (`str`):
            Mode/split name: one of `train`, `eval`, `test`, `all`
        metrics (`Dict[str, float]`):
            The metrics returned from train/evaluate/predict
        combined (`bool`, *optional*, defaults to `True`):
            Creates combined metrics by updating `all_results.json` with metrics of this call
    To understand the metrics please read the docstring of [`~Trainer.log_metrics`]. The only difference is that raw
    unformatted numbers are saved in the current method.
    """
    if not self.is_world_process_zero():
        return

    path = os.path.join(self.args.output_dir, f"{split}_results.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)

    if combined:
        path = os.path.join(self.args.output_dir, "all_results.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        all_metrics.update(metrics)
        with open(path, "w") as f:
            json.dump(all_metrics, f, indent=4, sort_keys=True)


def save_state(self):
    """
    Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model
    Under distributed environment this is done only for a process with rank 0.
    """
    if not self.is_world_process_zero():
        return

    path = os.path.join(self.args.output_dir, "trainer_state.json")
    self.state.save_to_json(path)


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except (TypeError, ValueError, RuntimeError):
        # TypeError: len() of unsized object
        return False


class TrainerMemoryTracker:
    """
    A helper class that tracks cpu and gpu memory.

    This class will silently skip unless `psutil` is available. Install with `pip install psutil`.

    When a stage completes, it can pass metrics dict to update with the memory metrics gathered during this stage.

    Example :

    ```python
    self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
    self._memory_tracker.start()
    # code ...
    metrics = {"train_runtime": 10.5}
    self._memory_tracker.stop_and_update_metrics(metrics)
    ```

    At the moment GPU tracking is only for `paddle`.

    # To understand this class' intricacies please read the documentation of [`~Trainer.log_metrics`].
    """

    # map trainer methods to metrics prefix
    stages = {
        "__init__": "init",
        "train": "train",
        "_inner_training_loop": "train",
        "evaluate": "eval",
        "predict": "test",
    }

    def __init__(self, skip_memory_metrics=False):

        self.skip_memory_metrics = skip_memory_metrics

        if not is_psutil_available():
            # soft dependency on psutil
            self.skip_memory_metrics = True

        if self.skip_memory_metrics:
            return

        import psutil  # noqa

        if is_paddle_cuda_available():
            import paddle

            self.paddle = paddle
            self.gpu = {}
        else:
            self.paddle = None

        self.process = psutil.Process()

        self.cur_stage = None
        self.cpu = {}
        self.init_reported = False

    def derive_stage(self):
        """derives the stage/caller name automatically"""
        caller = inspect.currentframe().f_back.f_back.f_code.co_name
        if caller in self.stages:
            return self.stages[caller]
        else:
            raise ValueError(
                f"was called from {caller}, but only expect to be called from one of {self.stages.keys()}"
            )

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_mem_used_peak = -1

        while True:
            self.cpu_mem_used_peak = max(self.cpu_mem_used(), self.cpu_mem_used_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def start(self):
        """start tracking for the caller's stage"""
        if self.skip_memory_metrics:
            return

        stage = self.derive_stage()
        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        self.cur_stage = stage

        gc.collect()

        if self.paddle is not None:
            # self.paddle.cuda.reset_peak_memory_stats()?
            self.paddle.device.cuda.empty_cache()

        # gpu
        if self.paddle is not None:
            self.gpu_mem_used_at_start = self.paddle.device.cuda.memory_allocated()

        # cpu
        self.cpu_mem_used_at_start = self.cpu_mem_used()

        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def stop(self, stage):
        """stop tracking for the passed stage"""

        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # this sends a signal to peak_monitor_func to complete its loop
        self.peak_monitoring = False

        # first ensure all objects get collected and their memory is freed
        gc.collect()

        if self.paddle is not None:
            self.paddle.device.cuda.empty_cache()

        # concepts:
        # - alloc_delta:  the difference of allocated memory between the end and the start
        # - peaked_delta: the difference between the peak memory and the current memory
        # in order to know how much memory the measured code consumed one needs to sum these two

        # gpu
        if self.paddle is not None:
            self.gpu_mem_used_now = self.paddle.device.cuda.memory_allocated()
            self.gpu_mem_used_peak = self.paddle.device.cuda.max_memory_allocated()
            self.gpu[self.cur_stage] = dict(
                begin=self.gpu_mem_used_at_start,
                end=self.gpu_mem_used_now,
                alloc=(self.gpu_mem_used_now - self.gpu_mem_used_at_start),
                peaked=max(0, self.gpu_mem_used_peak - self.gpu_mem_used_now),
            )

        # cpu
        self.cpu_mem_used_now = self.cpu_mem_used()
        self.cpu[self.cur_stage] = dict(
            begin=self.cpu_mem_used_at_start,
            end=self.cpu_mem_used_now,
            alloc=(self.cpu_mem_used_now - self.cpu_mem_used_at_start),
            peaked=max(0, self.cpu_mem_used_peak - self.cpu_mem_used_now),
        )

        # reset - cycle finished
        self.cur_stage = None

    def update_metrics(self, stage, metrics):
        """updates the metrics"""
        if self.skip_memory_metrics:
            return

        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        if hasattr(self, "gpu_mem_used_peak"):
            metrics["gpu_mem_max_memory_allocated"] = self.gpu_mem_used_peak
            metrics["gpu_mem_max_memory_reserved"] = self.paddle.device.cuda.max_memory_reserved()

        # since we don't have a way to return init metrics, we push them into the first of train/val/predict
        stages = [stage]
        if not self.init_reported:
            stages.insert(0, "init")
            self.init_reported = True

        for stage in stages:
            for t in ["alloc", "peaked"]:
                if stage in self.cpu and t in self.cpu[stage]:
                    metrics[f"{stage}_mem_cpu_{t}_delta"] = self.cpu[stage][t]
                if self.paddle is not None and stage in self.gpu and t in self.gpu[stage]:
                    metrics[f"{stage}_mem_gpu_{t}_delta"] = self.gpu[stage][t]
            # if we need additional debug info, enable the following
            # for t in ["begin", "end"]:
            #     if stage in self.cpu and t in self.cpu[stage]:
            #         metrics[f"{stage}_mem_cpu_{t}"] = self.cpu[stage][t]
            #     if self.paddle is not None and stage in self.gpu and t in self.gpu[stage]:
            #         metrics[f"{stage}_mem_gpu_{t}"] = self.gpu[stage][t]

        # since memory can be allocated before init, and it might be difficult to track overall
        # memory usage, in particular for GPU, let's report memory usage at the point init was called
        if stages[0] == "init":
            metrics["before_init_mem_cpu"] = self.cpu["init"]["begin"]
            if self.paddle is not None:
                metrics["before_init_mem_gpu"] = self.gpu["init"]["begin"]
            # if we also wanted to report any additional memory allocations in between init and
            # whatever the next stage was we could also report this:
            # if self.cpu["init"]["end"] != self.cpu[stage]["begin"]:
            #     metrics[f"after_init_mem_cpu_delta"] = self.cpu[stage]["begin"] - self.cpu["init"]["end"]
            # if self.paddle is not None and self.gpu["init"]["end"] != self.gpu[stage]["begin"]:
            #     metrics[f"after_init_mem_gpu_delta"] = self.gpu[stage]["begin"] - self.gpu["init"]["end"]

    def stop_and_update_metrics(self, metrics=None):
        """combine stop and metrics update in one call for simpler code"""
        if self.skip_memory_metrics:
            return

        stage = self.derive_stage()
        self.stop(stage)

        # init doesn't have metrics to update so we just save that data for later stages to retrieve
        if metrics is not None:
            self.update_metrics(stage, metrics)


class IterableDatasetShard(IterableDataset):
    """
    Wraps a Paddle `IterableDataset` to generate samples for one of the processes only. Instances of this class will
    always yield a number of samples that is a round multiple of the actual batch size (which is `batch_size x
    num_processes`). Depending on the value of the `drop_last` attribute, it will either stop the iteration at the
    first batch that would be too small or loop with indices from the beginning.
    On two processes with an iterable dataset yielding of `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` with a batch size of
    2:
    - the shard on process 0 will yield `[0, 1, 4, 5, 8, 9]` so will see batches `[0, 1]`, `[4, 5]`, `[8, 9]`
    - the shard on process 1 will yield `[2, 3, 6, 7, 10, 11]` so will see batches `[2, 3]`, `[6, 7]`, `[10, 11]`
    Args:
        dataset (`paddle.io.IterableDataset`):
            The batch sampler to split in several shards.
        batch_size (`int`, *optional*, defaults to 1):
            The size of the batches per shard.
        drop_last (`bool`, *optional*, defaults to `False`):
            Whether or not to drop the last incomplete batch or complete the last batches by using the samples from the
            beginning.
        num_processes (`int`, *optional*, defaults to 1):
            The number of processes running concurrently.
        process_index (`int`, *optional*, defaults to 0):
            The index of the current process.
        seed (`int`, *optional*, defaults to 0):
            A random seed that will be used for the random number generation in
            [`~trainer_utils.IterableDatasetShard.set_epoch`].
    """

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index
        self.seed = seed
        self.epoch = 0
        self.num_examples = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __iter__(self):
        self.num_examples = 0
        # TODO: support generator seed in sampling.
        #
        # if (
        #     not hasattr(self.dataset, "set_epoch")
        #     and hasattr(self.dataset, "generator")
        #     and isinstance(self.dataset.generator, paddle.fluid.Generator)
        # ):
        #     self.dataset.generator.manual_seed(self.seed + self.epoch)
        real_batch_size = self.batch_size * self.num_processes
        process_slice = range(self.process_index * self.batch_size, (self.process_index + 1) * self.batch_size)

        first_batch = None
        current_batch = []
        for element in self.dataset:
            self.num_examples += 1
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield current_batch[i]
                if first_batch is None:
                    first_batch = current_batch.copy()
                current_batch = []

        # Finished if drop_last is True, otherwise complete the last batch with elements from the beginning.
        if not self.drop_last and len(current_batch) > 0:
            if first_batch is None:
                first_batch = current_batch.copy()
            while len(current_batch) < real_batch_size:
                current_batch += first_batch
            for i in process_slice:
                yield current_batch[i]

    def __len__(self):
        # Will raise an error if the underlying dataset is not sized.
        if self.drop_last:
            return (len(self.dataset) // (self.batch_size * self.num_processes)) * self.batch_size
        else:
            return math.ceil(len(self.dataset) / (self.batch_size * self.num_processes)) * self.batch_size


class LastBatchPaddingSampler(paddle.io.DistributedBatchSampler):
    """The sampler which pads the first batch to the last batch"""

    def __iter__(self):
        local_batch_size = self.batch_size * self._acc_steps
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        global_eval_batch_size = self.batch_size * self.nranks
        last_batch_size = num_samples % global_eval_batch_size

        # Padding the first batch if the last batch is not full
        if last_batch_size > 0:
            padding_size = global_eval_batch_size - last_batch_size
            # Select the first batch of indices for padding
            if global_eval_batch_size <= len(indices):
                first_batch_idx = indices[:global_eval_batch_size]
            else:
                first_batch_idx = indices.copy()
            while padding_size > 0:
                # Repeatedly pad the indices until the padding size is fulfilled
                if padding_size > len(first_batch_idx):
                    indices += first_batch_idx
                    padding_size -= len(first_batch_idx)
                else:
                    indices += first_batch_idx[:padding_size]
                    padding_size = 0

        # Update the total number of indices
        self.total_size = len(indices)
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            # Iterate over the indices and extract batches that belong to the current device
            for i in range(
                self.local_rank * self.batch_size,
                len(indices),
                self.batch_size * self.nranks,
            ):
                subsampled_indices.extend(indices[i : i + self.batch_size])

            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        _sample_iter = iter(indices)
        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == local_batch_size:
                yield batch_indices
                batch_indices = []
        # Ensure that there are no leftover indices after batching
        assert len(batch_indices) == 0


def find_batch_size(tensors):
    """
    Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
    """
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            result = find_batch_size(t)
            if result is not None:
                return result
    elif isinstance(tensors, (dict, BatchEncoding)):
        for key, value in tensors.items():
            result = find_batch_size(value)
            if result is not None:
                return result
    elif isinstance(tensors, paddle.Tensor):
        return tensors.shape[0] if len(tensors.shape) >= 1 else None
    elif isinstance(tensors, np.ndarray):
        return tensors.shape[0] if len(tensors.shape) >= 1 else None


class RemoveColumnsCollator:
    """Wrap the data collator to remove unused columns before they are passed to the collator."""

    def __init__(
        self,
        data_collator,
        signature_columns,
        logger=None,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.data_collator = data_collator
        self.signature_columns = signature_columns
        self.logger = logger
        self.description = description
        self.model_name = model_name
        self.message_logged = False

    def _remove_columns(self, feature: dict) -> dict:
        if not isinstance(feature, dict):
            return feature
        if not self.message_logged and self.logger and self.model_name:
            ignored_columns = list(set(feature.keys()) - set(self.signature_columns))
            if len(ignored_columns) > 0:
                dset_description = "" if self.description is None else f"in the {self.description} set"
                self.logger.info(
                    f"The following columns {dset_description} don't have a corresponding argument in "
                    f"`{self.model_name}.forward` and have been ignored: {', '.join(ignored_columns)}."
                    f" If {', '.join(ignored_columns)} are not expected by `{self.model_name}.forward`, "
                    " you can safely ignore this message."
                )
                self.message_logged = True
        return {k: v for k, v in feature.items() if k in self.signature_columns}

    def __call__(self, features: List[dict]):
        features = [self._remove_columns(feature) for feature in features]
        return self.data_collator(features)


def set_hyrbid_parallel_seed(basic_seed, dataset_rank, tp_rank, pp_rank=0):
    from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

    random.seed(basic_seed + dataset_rank)
    np.random.seed(basic_seed + dataset_rank)
    paddle.seed(basic_seed + dataset_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = basic_seed + 59999 + tp_rank * 10 + pp_rank * 1000
    global_seed = basic_seed + 100003 + dataset_rank

    tracker = get_rng_state_tracker()

    if "global_seed" not in tracker.states_ and global_seed not in tracker.seeds_:
        tracker.add("global_seed", global_seed)
    if "local_seed" not in tracker.states_ and local_seed not in tracker.seeds_:
        tracker.add("local_seed", local_seed)


def should_skip_data(global_step, skip_data_intervals):
    """Whether to skip current step data"""

    if skip_data_intervals is None:
        return False
    skip_flag = False
    for interval in skip_data_intervals:
        if len(interval) != 2 or interval[0] > interval[1] or interval[0] <= 0:
            raise ValueError(f"Please check your skip interval {interval}")
        start_global_step, end_global_step = interval[0], interval[1]
        # start_global_step and end_global_step start from 1, while global_step start from 0
        if start_global_step <= global_step + 1 <= end_global_step:
            skip_flag = True
            break
    return skip_flag


def split_parallel_config(parallel_config):
    if "," in parallel_config:
        parallel_config = set(parallel_config.split(","))
    else:
        parallel_config = set(parallel_config.split(" "))
    return parallel_config


def download_recovery_ckpt_from_pdc(recovery_checkpoint_path, timeout):
    """Download checkpoint from PDC for resuming training after failover. Longjob environment is necessary.

    Args:
        recovery_checkpoint_path (`str`):
            local path to load checkpoint for training recovery
        timeout (`int`):
            max wait time for download
    """

    try:
        base_dir, download_dir = os.path.split(os.path.normpath(recovery_checkpoint_path))
        if not os.path.exists(base_dir) and base_dir != "":
            os.makedirs(base_dir, exist_ok=True)
        download_step = int(_re_checkpoint.search(download_dir).groups()[0])
    except Exception as e:
        raise RuntimeError(f"{PDC_DOWNLOAD_ERROR}; Failed to parse checkpoint path, details: {e}")
    start_time = time.time()
    # TODO(@gexiao): temporary workaround for environment variable conflicts.
    original_trainer_id = os.getenv("PADDLE_TRAINER_ID")
    original_trainers_num = os.getenv("PADDLE_TRAINERS_NUM")
    cards_per_node = int(os.getenv("PADDLE_LOCAL_SIZE", "8"))
    os.environ["PADDLE_TRAINER_ID"] = str(dist.get_rank() // cards_per_node)
    os.environ["PADDLE_TRAINERS_NUM"] = str(dist.get_world_size() // cards_per_node)
    result = pdc_tool.pdc_download_checkpoint(download_step, timeout)
    os.environ["PADDLE_TRAINER_ID"] = original_trainer_id
    os.environ["PADDLE_TRAINERS_NUM"] = original_trainers_num
    end_time = time.time()
    if result == PDCErrorCode.Success:
        logger.info(f"Successfully downloaded checkpoint from PDC, total time cost: {end_time - start_time} seconds.")
    elif result == PDCErrorCode.LocalPathExist:
        logger.warning(
            f"Skipping download checkpoint since file exists at local, total time cost: {end_time - start_time} seconds."
        )
    else:
        raise RuntimeError(
            f"{PDC_DOWNLOAD_ERROR}; Error occurred when trying to download checkpoint from PDC, recovery_checkpoint_path: {recovery_checkpoint_path}, timeout: {timeout}; error details: {PDCErrorMessageMap[result]}"
        )
