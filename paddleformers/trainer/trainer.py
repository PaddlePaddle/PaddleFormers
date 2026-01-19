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
#  https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
from __future__ import annotations

import collections
import contextlib
import gc
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import time
import types
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.amp.auto_cast as autocast
import paddle.distributed as dist
import paddle.distributed.auto_parallel.intermediate.parallelize as parallelize
import paddle.nn as nn
import psutil
from packaging import version
from paddle import framework
from paddle.base import core
from paddle.distributed.auto_parallel._utils import _patch_grads_for_step
from paddle.distributed.fleet.meta_parallel import PipelineLayer

try:
    from paddle.distributed.fleet.meta_parallel import PipelineDatasetPreprocessor
except:
    PipelineDatasetPreprocessor = None

from ..utils.import_utils import is_paddlefleet_available

# Conditionally import paddlefleet modules
if is_paddlefleet_available():
    import paddlefleet.distributed.model as paddlefleet_dist_model
    from paddlefleet.models.gpt import GPTModel as FleetGPTModel
    from paddlefleet.pipeline_parallel import ParallelBase as PaddleFleetParallelBase
    from paddlefleet.pipeline_parallel import PipelineLayer as PaddleFleetPipelineLayer

    from paddleformers.transformers.gpt_provider import GPTModel
else:
    paddlefleet_dist_model = None
    FleetGPTModel = None
    PaddleFleetParallelBase = None
    PaddleFleetPipelineLayer = None

from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_optimizer_stage2 import (
    GroupShardedOptimizerStage2,
)
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    obtain_optimizer_parameters_list,
)

_obtain_optimizer_parameters_list = obtain_optimizer_parameters_list


from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    register_sequence_parallel_allreduce_hooks,
)
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
from tqdm.auto import tqdm

from ..data import (
    DataCollator,
    DataCollatorWithPadding,
    DistDataLoader,
    default_data_collator,
    init_dataloader_comm_group,
)
from ..peft import LoRAModel
from ..peft.lora import QuantizationLoRABaseLinear
from ..quantization.quantization_linear import (
    ColumnParallelQuantizationLinear,
    QuantizationLinear,
    RowParallelQuantizationLinear,
)

if is_paddlefleet_available():
    from paddlefleet.utils import get_batch_on_this_cp_rank
else:
    get_batch_on_this_cp_rank = None
if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

from paddle.framework.recall_error import LOSS_INF_ERROR, LOSS_NAN_ERROR

from ..transformers.context_parallel_utils import auto_split_sequence_dim_load_balance
from ..transformers.image_processing_utils import ImageProcessingMixin
from ..transformers.model_utils import (
    PretrainedModel,
    _add_variant,
    load_sharded_checkpoint,
    unwrap_model,
)
from ..transformers.segment_parallel_utils import (
    auto_split_inputs_sequence_dim,
    split_inputs_sequence_dim,
)
from ..utils import empty_device_cache, perf_utils
from ..utils.batch_sampler import DistributedBatchSampler as NlpDistributedBatchSampler
from ..utils.env import (
    EMA_STATE_DIC,
    LORA_WEIGHTS_NAME,
    MASTER_WEIGHT_DIC,
    MODEL_META_NAME,
    MODEL_STATE_DIC,
    OPTIMIZER_STATE_DIC,
    PADDLE_MASTER_WEIGHTS_INDEX_NAME,
    PADDLE_OPTIMIZER_NAME,
    PADDLE_PEFT_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    PREFIX_CHECKPOINT_DIR,
    PREFIX_HF_CHECKPOINT_DIR,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_PEFT_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
)
from ..utils.import_utils import is_datasets_available, is_paddle_cuda_available
from ..utils.log import MetricsDumper, logger
from ..utils.pdc_sdk import FLASH_DEVICE
from ..utils.tools import get_env_device, paddle_device
from .argparser import strtobool
from .integrations import get_reporting_integration_callbacks
from .plugins.timer import RuntimeTimer, get_timers, set_timers
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    InterleaveGateUpCallback,
    PrinterCallback,
    ProgressCallback,
    SPGradSyncCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_utils import (  # set_hyrbid_parallel_seed,
    EvalLoopOutput,
    EvalPrediction,
    IntervalStrategy,
    IterableDatasetShard,
    OptimizerNames,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardingOption,
    TrainerMemoryTracker,
    TrainOutput,
    _exec_mode_guard,
    _insert_sync,
    download_recovery_ckpt_from_pdc,
    find_batch_size,
    get_last_checkpoint,
    get_scheduler,
    has_length,
    init_optimizer,
    mock_offload_optimizer,
    set_random_seed,
    set_seed,
    should_skip_data,
    speed_metrics,
)
from .training_args import TrainingArguments
from .unified_checkpoint import UnifiedCheckpointHandler
from .utils import reshard as reshard_util
from .utils.async_save import AsyncSaver
from .utils.ckpt_converter import CheckpointConverter
from .utils.reshard import SHARDING_STRATEGY_V1, split_opt_state
from .utils.sharding_io import GroupGetter, to_device

try:
    from .utils.zero_cost_checkpoint import (
        DistInfoCollectorValidator,
        NonZCCEMACallback,
        ZeroCostCheckpointCallback,
        ZeroCostCheckpointCallbackFcBased,
        ZeroCostCheckpointManager,
        ZeroCostCheckpointWorker,
        ZeroCostCheckpointWorkerFcBased,
    )
except (ImportError, ModuleNotFoundError):
    ZeroCostCheckpointManager, NonZCCEMACallback = None, None
from .utils.helper import (  # nested_truncate,
    broadcast_dataset_rank0_model,
    broadcast_dp_optimizer,
    broadcast_moe_optimizer,
    distributed_concat,
    distributed_file,
    distributed_isfile,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
)
from .utils.sharding_io import ShardingIO

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_datasets_available():
    PADDLEFORMERS_TESTING = os.environ.get("PADDLEFORMERS_TESTING", False)
    if "torch" not in sys.modules and not PADDLEFORMERS_TESTING:
        sys.modules["torch"] = None
        import datasets

        del sys.modules["torch"]
    else:
        import datasets

from paddle.distributed import in_auto_parallel_align_mode
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.io.dataloader.dataloader_iter import _DataLoaderIterBase

__all__ = ["Trainer"]

MODEL_NAME = "model"
OPTIMIZER_NAME = "optimizer"
DIST_CKPT_PATH = "dist_ckpt"
DIST_MODEL_PATH = "dist_model"


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PaddlePaddle, optimized for PaddleFormers.

    Args:
        model ([`PretrainedModel`] or `paddle.nn.Layer`, *optional*):
            The model to train, evaluate or use for predictions.

            [`Trainer`] is optimized to work with the [`PretrainedModel`] provided by the library. You can still use
            your own models defined as `paddle.nn.Layer` as long as they work the same way as the PaddleFormers
            models.
        criterion(`paddle.nn.Layer`, *optional*):
            The model may only output the loggit, if you want do more computation for the output of model, you can
            add the criterion Layer.
        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise.
        train_dataset (`paddle.io.Dataset` or `paddle.io.IterableDataset`, *optional*):
            The dataset to use for training. If it is an `datasets.Dataset`, columns not accepted by the
            `model.forward()` method are automatically removed.
        eval_dataset (Union[`paddle.io.Dataset`, Dict[str, `paddle.io.Dataset`]],  *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values.
        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks.
            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler]`, *optional*): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your model
            and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        preprocess_logits_for_metrics (`Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

    Important attributes:

        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PretrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, the inner model is
          wrapped in `paddle.DataParallel`. If model hasn't been wrapped, then `self.model_wrapped` is the same
          as `self.model`.

    """

    from .trainer_utils import log_metrics, metrics_format, save_metrics, save_state

    def __init__(
        self,
        model: Union[PretrainedModel, nn.Layer] = None,
        criterion: nn.Layer = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Union[Dataset, Dict[str, Dataset]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[paddle.optimizer.Optimizer, paddle.optimizer.lr.LRScheduler] = (None, None),
        preprocess_logits_for_metrics: Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor] = None,
        processing_class: Optional[ImageProcessingMixin] = None,
        resume_from_custom_func: Optional[Callable] = None,
    ):

        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)

        self.args = args
        self.is_in_train = False
        # self.do_grad_scaling = args.fp16

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # Seed must be set before instantiating the model when using model
        if is_paddlefleet_available():
            if not self.args.enable_auto_parallel:
                set_random_seed(seed_=self.args.seed)
            else:
                logger.warning("set_seed not support yet in auto_parallel mode")

        set_seed(seed=self.args.seed)

        self._skip_global_steps = 0  # total skip global steps
        self._skip_steps_since_last_logged = 0  # skip steps since last logged
        if model is None:
            logger.warning("Model is None.")
            self.model = None
            self.train_dataset = train_dataset
            self.tokenizer = tokenizer
            default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
            self.data_collator = data_collator if data_collator is not None else default_collator
            return

        if self.args.to_static:
            model = paddle.jit.to_static(model)
            logger.info("Successfully to apply @to_static to the whole model.")

        if self.args.should_save or self.args.should_save_model_state:
            os.makedirs(self.args.output_dir, exist_ok=True)

        self.sharding = None
        if len(args.sharding) > 0:
            if args.local_rank == -1:
                raise ValueError("Using sharding only works in distributed training.")
            self.sharding = True

        # Initialize the parallel environment.
        # Parallelism includes enabling auto-parallel or not, and the auto-parallel implementation
        # consists of two modes: the intermediate API and the basic API.
        if paddle.distributed.get_world_size() > 1:
            if self.args.enable_auto_parallel:
                # Process intermediate API.
                if self.args.use_intermediate_api:
                    auto_dist_flag = self.args.get_auto_dist_flag()
                    self.auto_dist_config = model._generate_auto_dist_config(auto_dist_flag)
                    assert (
                        not parallelize.has_parallelized_model
                    ), "Current init model should not be in parallel process."

                    for param in model.parameters():
                        if param._is_initialized():
                            logger.warning(
                                "intermediate_api needs lazy init because if param init before parallelize_model ,"
                                + " param will be allocated the full amount of memory"
                                + " We recommend reallocating memory after paralleliz-model to reduce the peak of memory allocation"
                            )

                # Process basic API.
                else:
                    raise ValueError("Auto parallel only support basic API now.")

                self.global_mesh = fleet.auto.get_mesh()
                self.comm_group_in_pp = fleet.get_hybrid_communicate_group().get_pipe_parallel_group()

            elif self.args.use_hybrid_parallel:
                self.hcg = fleet.get_hybrid_communicate_group()
                self.dp_group = self.hcg.get_data_parallel_group()
                self.sharding_group = self.hcg.get_sharding_parallel_group()

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)

        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.resume_from_custom_func = resume_from_custom_func
        if not args.skip_profile_timer:
            set_timers()
        self.timers = get_timers()
        self.runtime_timer = RuntimeTimer("RuntimeTimer")

        self.model_wrapped = model
        self.model = model
        self.criterion = criterion

        # Set use_cache for the model
        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = self.args.use_cache

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.processing_class = processing_class
        self.optimizer, self.lr_scheduler = optimizers

        self.label_smoother = None
        self.state = TrainerState()
        self.control = TrainerControl()
        self._signature_columns = None
        self.optimizer_grouped_parameters = None
        self.sharding_io = None
        if self.args.should_save_sharding_stage1_model or self.args.should_load_sharding_stage1_model:
            self.sharding_io = ShardingIO(
                self.args,
                self.model,
                self.optimizer,
                remap_parameter_name=self.args.load_sharded_model_remap_parameter_name,
                is_ema=self.args.sharded_model_from_ema,
            )

        if (
            self.args.save_checkpoint_format == "unified_checkpoint"
            or self.args.load_checkpoint_format == "unified_checkpoint"
        ):
            self.unified_checkpoint_handler = UnifiedCheckpointHandler(self.args)

        if self.sharding is not None and self.optimizer is not None:
            raise RuntimeError(
                "Passing `optimizers` is not allowed if sharding is enabled."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )

        if self.args.pipeline_model_parallel_size > 1 and self.args.use_hybrid_parallel:
            if is_paddlefleet_available() and PaddleFleetPipelineLayer is not None:
                assert (
                    isinstance(model, LoRAModel) and isinstance(model.model, (PaddleFleetPipelineLayer, PipelineLayer))
                ) or isinstance(
                    model, (PaddleFleetPipelineLayer, PipelineLayer)
                ), f"Only support pipeline parallel mode when model is PaddleFleetPipelineLayer or PipelineLayer!!! but get {type(model.model)}"
            else:
                assert (isinstance(model, LoRAModel) and isinstance(model.model, PipelineLayer)) or isinstance(
                    model, PipelineLayer
                ), f"Only support pipeline parallel mode when model is PipelineLayer!!! but get {type(model.model)}"
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        def _save_ckpt_func(state_dict, path, signal_path=None):
            if self.args.enable_auto_parallel:
                dist.save_state_dict(state_dict, path)
            else:
                paddle.save(state_dict, path)

            if signal_path is not None:
                with open(signal_path, mode="w+") as f:
                    f.write("1")

        self.metrics_dumper = None
        if self.args.metrics_output_path is not None:
            if not os.path.exists(self.args.metrics_output_path):
                os.makedirs(self.args.metrics_output_path, exist_ok=True)
            metrics_output_file = os.path.join(self.args.metrics_output_path, f"metrics_rank{dist.get_rank()}.json")
            logger.info(f"create/append metrics dumper at {metrics_output_file}")
            self.metrics_dumper = MetricsDumper(metrics_output_file)

        self._save_ckpt_func = _save_ckpt_func
        self._load_ckpt_func = dist.load_state_dict if self.args.enable_auto_parallel else paddle.load

        if ZeroCostCheckpointManager is None and self.args.enable_zero_cost_checkpoint:
            logger.warning(
                "enable_zero_cost_checkpoint has been set as True, but paddle version is too old to support this function, please upgrade it."
            )
            self.args.enable_zero_cost_checkpoint = False

        if self.args.enable_zero_cost_checkpoint:
            # Currently, zero cost checkpoint only support pretraining mode with hybrid parallel enabled
            assert (
                not self.args.ignore_save_lr_and_optim
            ), "ignore_save_lr_and_optim should be False when using zero cost checkpoint"
            assert self.args.use_hybrid_parallel, "use_hybrid_parallel must be True when using zero cost checkpoint"
            assert not (
                self.args.save_checkpoint_format == "unified_checkpoint"
                or self.args.load_checkpoint_format == "unified_checkpoint"
            ), "use_unified_checkpoint should be False when using zero cost checkpoint"
            assert not strtobool(
                os.getenv("FLAG_LLM_PDC", "False")
            ), "Dont support FLAG_LLM_PDC when using zero cost checkpoint"
            assert (
                self.args.should_save_sharding_stage1_model
            ), "should_save_sharding_stage1_model should be True when using zero cost checkpoint"
            assert (
                ShardingOption.FULL_SHARD not in self.args.sharding
            ), "FULL_SHARD is not supported when using zero cost checkpoint"
            assert not self.args.save_tokenizer, "save_tokenizer is not supported when using zero cost checkpoint"

            # init attributes for zero cost checkpoint mode
            self.zcc_manager = None

        if self.args.ordered_save_group_size > 0:
            logger.info(f"using save in order, its group size is {self.args.ordered_save_group_size}")
            assert not self.args.use_async_save, "Not support async save in ordered save"
            assert self.args.tensor_model_parallel_size % self.args.ordered_save_group_size == 0
            self._save_ckpt_func = self._ordered_save

        if self.args.use_async_save:
            self._async_optimizer_saver = AsyncSaver()

        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not isinstance(train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")

        if isinstance(self.model, LoRAModel):
            if (
                self.args.save_checkpoint_format == "unified_checkpoint"
                and "skip_save_model_weight" in self.args.unified_checkpoint_config
            ):
                self.args.unified_checkpoint_config.remove("skip_save_model_weight")
                logger.warning(
                    "We do not support skip_save_model_weight in peft model when using unified checkpoint, remove this config."
                )

        self.do_grad_scaling = True if args.fp16 else False
        self.enable_autocast_context_manager = False
        self.amp_dtype = "float16" if args.fp16 else "bfloat16"
        if not self.args.enable_auto_parallel:
            if args.fp16 or args.bf16:
                # set do_grad_scaling, enable_autocast_context_manager
                self._wrap_amp_model(args, model)

        if isinstance(model, nn.Layer) and args.recompute_granularity is not None:

            def fn(layer):
                if hasattr(layer, "enable_recompute") and (
                    layer.enable_recompute is False or layer.enable_recompute == 0
                ):
                    layer.enable_recompute = True

            model.apply(fn)

        self._pp_data_group = None
        if self.args.pipeline_model_parallel_size > 1 and self.args.distributed_dataloader:
            self._pp_data_group = init_dataloader_comm_group()

        default_label_names = (
            ["start_positions", "end_positions"]
            if "QusetionAnswering" in type(self.model).__name__ or "UIE" in type(self.model).__name__
            else ["labels"]
        )
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names

        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)
        self.print_config()

        # very last
        self._memory_tracker.stop_and_update_metrics()
        if self.args.count_trained_tokens:
            self.trained_effective_tokens = 0
            self.trained_tokens = 0

    def _wrap_amp_model(self, args, model):
        logger.info("Using half precision")
        self.enable_autocast_context_manager = True
        self.scaler = paddle.amp.GradScaler(init_loss_scaling=self.args.scale_loss)
        # fix for load saved fp16 or bf16 ckpt, decorate model first.
        if self.args.enable_auto_parallel:
            self.scaler = dist.shard_scaler(self.scaler)
        if self.args.fp16_opt_level == "O2":
            excluded_layers = [
                QuantizationLinear,
                ColumnParallelQuantizationLinear,
                RowParallelQuantizationLinear,
                QuantizationLoRABaseLinear,
            ]
            if not self.args.enable_auto_parallel:
                excluded_layers += self._decorate_exclude_layers(model)
            paddle.amp.decorate(
                models=model,
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
                excluded_layers=excluded_layers,
            )
        if args.enable_auto_parallel:
            return
        # for pipeline mode and pure tensor parallel
        if self.args.pipeline_model_parallel_size > 1 or (
            self.args.tensor_model_parallel_size > 1 and self.sharding is None
        ):
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionScaler(self.scaler)  # return value has no use
            self.scaler = fleet.distributed_scaler(self.scaler)
        elif self.sharding is not None:
            if self.amp_dtype == "float16" or self.amp_dtype == "bfloat16":
                if ShardingOption.SHARD_OP in self.args.sharding:
                    if self.args.amp_master_grad:
                        mix_precision_utils.MixPrecisionScaler(self.scaler)  # return value has no use
                    self.scaler = fleet.distributed_scaler(self.scaler)
                else:
                    # scaler for stage2 and stage3
                    from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_utils import (
                        GroupShardedScaler,
                    )

                    if self.args.amp_master_grad:
                        mix_precision_utils.MixPrecisionScaler(self.scaler)  # return value has no use

                    self.scaler = GroupShardedScaler(self.scaler)
            else:
                self.do_grad_scaling = False
                self.use_cuda_amp = False
                self.amp_dtype = None

    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~TrainerCallback`].

        Args:
           callback (`type` or [`~TrainerCallback`]):
               A [`~TrainerCallback`] class or an instance of a [`~TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        """
        Remove a callback from the current list of [`~TrainerCallback`] and returns it.
        If the callback is not found, returns `None` (and no error is raised).
        Args:
           callback (`type` or [`~TrainerCallback`]):
               A [`~TrainerCallback`] class or an instance of a [`~TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.
        Returns:
            [`~TrainerCallback`]: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        """
        Remove a callback from the current list of [`~TrainerCallback`].
        Args:
           callback (`type` or [`~TrainerCallback`]):
               A [`~TrainerCallback`] class or an instance of a [`~TrainerCallback`]. In the
               first case, will remove the first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    def _load_from_peft_checkpoint(self, resume_from_checkpoint=None):
        """load state_dict from checkpoint, Only for PEFT Model.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. Only load model state dict.
        """

        if resume_from_checkpoint is not None:
            convert_tp = False
            if isinstance(self.model, LoRAModel):
                if self.model.quantized or self.args.pipeline_model_parallel_size > 1:
                    weights_file = os.path.join(
                        resume_from_checkpoint, _add_variant(LORA_WEIGHTS_NAME, self.args.weight_name_suffix)
                    )
                else:
                    weights_file = os.path.join(resume_from_checkpoint, LORA_WEIGHTS_NAME)
                    if self.model.lora_config.tensor_model_parallel_size > 1:
                        convert_tp = True

            if self.args.dataset_rank == 0:
                logger.info(f"Loading model from {resume_from_checkpoint} .")

                if os.path.isfile(weights_file):
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = paddle.load(weights_file, return_numpy=True)
                    if convert_tp:
                        state_dict = self.model._convert_tensor_parallel(state_dict)

                    # If the model is on the GPU, it still works!
                    self._set_state_dict_in_model(state_dict)
                    # release memory
                    del state_dict
        elif resume_from_checkpoint is not None:
            logger.info(f"not loading ckpt :{self.args.dataset_rank}")

    def _load_from_checkpoint(self, resume_from_checkpoint=None):
        """load state_dict from_checkpoint, Only load model state dict.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. Only load model state dict.
        """
        self.runtime_timer.start("checkpoint loading time")
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            uc_async_save = False
            if self.args.enable_auto_parallel:
                uc_async_save = (
                    self.args.load_checkpoint_format == "unified_checkpoint"
                    and "async_save" in self.args.unified_checkpoint_config
                )
            resume_from_checkpoint = get_last_checkpoint(
                self.args.output_dir, signal_folder=self.args.output_signal_dir, uc_async_save=uc_async_save
            )
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if self.args.enable_auto_parallel:
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

                        if self.do_grad_scaling and distributed_isfile(
                            os.path.join(resume_from_checkpoint, SCALER_NAME)
                        ):
                            self.scaler.load_state_dict(
                                paddle.load(
                                    distributed_file(os.path.join(resume_from_checkpoint, SCALER_NAME)),
                                    return_numpy=True,
                                )
                            )
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

                # release memory
                del state_dict
            self.runtime_timer.stop()
        else:
            if self.args.load_checkpoint_format == "unified_checkpoint":
                if resume_from_checkpoint is not None:
                    use_unified_checkpoint = False
                    if self.is_unified_checkpoint(resume_from_checkpoint):
                        use_unified_checkpoint = True
                    else:
                        logger.info(
                            "Loading origin checkpoint, the next checkpoint will be saved as unified checkpoint"
                        )

                    if use_unified_checkpoint:
                        self.unified_checkpoint_handler.load_unified_checkpoint(
                            self.model,
                            resume_from_checkpoint,
                            convert_from_hf=self.args.convert_from_hf,
                        )
                        if isinstance(self.model, LoRAModel) and self.model.lora_config.loraga:
                            self.model.reinit_base_model = True
                        logger.info(f"Loading model from {resume_from_checkpoint} using unified checkpoint.")
                        self.runtime_timer.stop()
                        return

            if isinstance(self.model, LoRAModel):
                self._load_from_peft_checkpoint(resume_from_checkpoint)
                if isinstance(self.model, LoRAModel) and self.model.lora_config.loraga:
                    self.model.reinit_base_model = True
                self.runtime_timer.stop()
                return

            weight_name = PADDLE_WEIGHTS_NAME
            weight_index_name = (
                PADDLE_WEIGHTS_INDEX_NAME  # currently set paddle as default, do not support safetensors.
            )

            if self.args.should_load_sharding_stage1_model:
                state_dict = self.sharding_io.load_state_dict_from_checkpoint_with_reshard(
                    resume_from_checkpoint,
                    base_weight_name=weight_name,
                    model_wrapped=self.model_wrapped,
                )
                old_state_dict = self.model.state_dict()
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k not in old_state_dict or id(v) != id(old_state_dict[k]):
                        new_state_dict[k] = v
                self.model.set_state_dict(new_state_dict)
                if self.args.offload_optim:
                    self._offload_optimizer()
            else:
                if resume_from_checkpoint is not None and (
                    self.args.dataset_rank == 0 or self.args.use_expert_parallel
                ):

                    weights_file = os.path.join(
                        resume_from_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)
                    )
                    weights_index_file = os.path.join(
                        resume_from_checkpoint, _add_variant(weight_index_name, self.args.weight_name_suffix)
                    )

                    if not any(
                        os.path.isfile(f)
                        for f in [
                            weights_file,
                            weights_index_file,
                        ]
                    ):
                        raise ValueError(
                            f"Can't find a valid checkpoint at {resume_from_checkpoint} -- {weights_file}"
                        )

                    logger.info(f"Loading model from {resume_from_checkpoint} .")

                    if os.path.isfile(weights_file):
                        # We load the model state dict on the CPU to avoid an OOM error.
                        state_dict = paddle.load(weights_file, return_numpy=True)
                        # If the model is on the GPU, it still works!
                        self._set_state_dict_in_model(state_dict)
                        # release memory
                        del state_dict
                    else:
                        # We load the sharded checkpoint.
                        missing_keys, unexpected_keys = load_sharded_checkpoint(
                            self.model, resume_from_checkpoint, self.args.weight_name_suffix, prefer_safe=False
                        )
                        logger.info(f"set state_dict: {missing_keys, unexpected_keys}")

                elif resume_from_checkpoint is not None:
                    logger.info(f"not loading ckpt :{self.args.dataset_rank}")
            self.runtime_timer.stop()

    def _wrap_model_and_load_sharded_checkpoint(self, resume_from_checkpoint):
        # In the sharded mode, should invoke _load_from_checkpoint after _wrap_model.
        # In this mode, each sharding rank load sharded params, do not need to implement the broadcast logic.
        model = self._wrap_model(self.model_wrapped)
        if self.sharding_io is not None:
            # the self.optimizer should be wrapped and it is done in _wrap_model
            self.sharding_io.set_optimizer(self.optimizer)
        if model is not self.model:
            self.model_wrapped = model
        # Should invoke _load_from_checpoint after _load_optimizer_and_scheduler
        # because the _load_from_checkpoint method rely on the optimizer in the shareded mode.
        if resume_from_checkpoint:
            self._load_optimizer_and_scheduler(resume_from_checkpoint)
            self._load_from_checkpoint(resume_from_checkpoint)
        return model

    def _get_zcc_implementation_classes(self):
        """Get appropriate ZCC implementation classes based on checkpoint format."""
        if self.args.save_checkpoint_format == "flex_checkpoint":
            return ZeroCostCheckpointCallbackFcBased, ZeroCostCheckpointWorkerFcBased
        return ZeroCostCheckpointCallback, ZeroCostCheckpointWorker

    def _create_zcc_manager_instance(self, unwrapped_model, zcc_worker_class):
        """Create ZCC manager instance with appropriate configuration."""
        if isinstance(self.model, PipelineLayer):
            pipeline_hooks_capacity = (
                unwrapped_model.forward_pipeline_parallel_hook_capacity
                + unwrapped_model.backward_pipeline_parallel_hook_capacity
            )
        else:
            pipeline_hooks_capacity = self.args.gradient_accumulation_steps

        return ZeroCostCheckpointManager(
            worker_num=self.args.zcc_workers_num,
            pipeline_hooks_capacity=pipeline_hooks_capacity,
            capacity_usage=self.args.zcc_pipeline_hooks_capacity_usage,
            use_expert_parallel=self.args.use_expert_parallel,
            ema_coef=self.args.zcc_save_ema_coef,
            zcc_worker_class=zcc_worker_class,
            save_hf_steps=self.args.save_hf_steps,
        )

    def _register_pipeline_hooks(self, unwrapped_model):
        """Register forward and backward pipeline hooks."""
        # Register forward hooks
        for i in range(unwrapped_model.forward_pipeline_parallel_hook_capacity):
            unwrapped_model.register_forward_pipeline_parallel_hook(
                location=i, hook=self.zcc_manager.zcc_pipeline_hook
            )

        # Register backward hooks
        for i in range(unwrapped_model.backward_pipeline_parallel_hook_capacity):
            unwrapped_model.register_backward_pipeline_parallel_hook(
                location=i, hook=self.zcc_manager.zcc_pipeline_hook
            )

    def _setup_zcc_callback(self, zcc_callback_class):
        """Setup ZCC callback with required dependencies."""
        callback = zcc_callback_class(self.args, self.zcc_manager, self.runtime_timer, self.sharding_io)
        self.add_callback(callback)

    def _handle_checkpoint_resume(self, resume_from_checkpoint):
        """Handle resumption from previous checkpoint if provided."""
        if resume_from_checkpoint is None:
            return

        ema_state_path = self._get_ema_state_path(resume_from_checkpoint)

        if not os.path.exists(ema_state_path):
            logger.info(f"ZCC EMA state dict not found at: {ema_state_path}")
            return

        # Validate distributed strategy compatibility
        should_load_ema = self._should_load_ema_state(resume_from_checkpoint, ema_state_path)

        if should_load_ema:
            logger.info(f"Loading ZCC EMA state from: {ema_state_path}")
            self.zcc_manager.set_ema_state_dict(ema_state_path)

    def _get_ema_state_path(self, checkpoint_path):
        """Get the path to EMA state based on checkpoint format."""
        if self.args.save_checkpoint_format == "flex_checkpoint":
            return os.path.join(checkpoint_path, "ema_state", f"{dist.get_rank()}_0.distcp")
        else:
            optimizer_name = _add_variant(PADDLE_OPTIMIZER_NAME, self.args.optimizer_name_suffix)
            return os.path.join(checkpoint_path, optimizer_name).replace("optimizer", "ema")

    def _should_load_ema_state(self, checkpoint_path, ema_state_path):
        """Determine if EMA state should be loaded based on configuration and compatibility."""
        if self.args.zcc_save_ema_coef is None:
            logger.info("EMA coefficient is None, skipping EMA state loading")
            return False

        success, err_msg = DistInfoCollectorValidator(self.args, self.hcg).check_same_strategy(checkpoint_path)

        if not success:
            logger.warning(f"Cannot load EMA state due to strategy mismatch: {err_msg}")
            return False

        return True

    def create_zcc_manager(self, unwrapped_model, resume_from_checkpoint=None):
        """
        Create zero cost checkpoint manager.
        Has to be called after pipeline model is created.
        resume_from_checkpoint: if use Flash checkpoint EMA, load previous checkpoint status
        """
        assert isinstance(
            self.model, PretrainedModel
        ), "model should be a PretrainedModel when using zero cost checkpoint"
        logger.info("Create zero cost checkpoint manager...")

        zcc_callback_class, zcc_worker_class = self._get_zcc_implementation_classes()

        # Create ZCC manager with appropriate configuration
        self.zcc_manager = self._create_zcc_manager_instance(unwrapped_model, zcc_worker_class)

        # Register pipeline hooks if using pipeline parallelism
        if isinstance(self.model, PipelineLayer):
            self._register_pipeline_hooks(unwrapped_model)

        # Add callback and handle checkpoint resumption
        self._setup_zcc_callback(zcc_callback_class)

        self._handle_checkpoint_resume(resume_from_checkpoint)

        logger.info("Zero cost checkpoint manager created successfully.")

    def add_non_zcc_ema_callback(self, resume_from_checkpoint, ema_state_assembler=None):

        non_zcc_ema_callback = NonZCCEMACallback.create_nonzcc_callback(
            args=self.args,
            resume_from_checkpoint=resume_from_checkpoint,
            sharding_io=self.sharding_io,
            model=self.model,
            optimizer=self.optimizer,
            hcg=self.hcg,
            ema_state_assembler=ema_state_assembler,
        )

        self.add_callback(non_zcc_ema_callback)

    def _save_flex_model_state(self, output_dir):
        model_sharded_state_dict = self.model.sharded_state_dict()
        model_state_dict_path = os.path.join(output_dir, MODEL_STATE_DIC)
        os.makedirs(model_state_dict_path, exist_ok=True)
        dist.save_state_dict(
            model_sharded_state_dict,
            model_state_dict_path,
            save_replicas=self.args.replicate_saved_into_local,
        )

    def _save_flex_optimizer_state(self, output_dir):
        optimizer_state_dict_path = os.path.join(output_dir, OPTIMIZER_STATE_DIC)
        optimizer_states = {}
        master_weights = {}
        model_sharded_state_dict = self.model.sharded_state_dict()
        optimizer_sharded_state_dict = self.optimizer.sharded_state_dict(model_sharded_state_dict)
        for k, v in optimizer_sharded_state_dict.items():
            if k.endswith(".w_0"):
                master_weights[k] = v
            else:
                optimizer_states[k] = v

        dist.save_state_dict(
            optimizer_states,
            optimizer_state_dict_path,
            save_replicas=self.args.replicate_saved_into_local,
        )

        master_weights_path = os.path.join(output_dir, MASTER_WEIGHT_DIC)
        dist.save_state_dict(
            master_weights,
            master_weights_path,
            save_replicas=self.args.replicate_saved_into_local,
        )

        saved_signal_path = os.path.join(output_dir, f"saved_signal_{dist.get_rank()}")
        with open(saved_signal_path, mode="w+") as f:
            f.write("1")

    def _load_flex_checkpoint(self, resume_from_checkpoint):
        def get_metadata_file_name(path):
            files = os.listdir(path)
            metadata_files = [f for f in files if f.endswith(".metadata")]
            assert len(metadata_files) > 0, f"Found no metadata files in {path}"
            assert len(metadata_files) == 1, f"Found multiple metadata files in {path}"
            return metadata_files[0]

        model_sharded_state_dict = self.model.sharded_state_dict()
        master_weights_path = os.path.join(resume_from_checkpoint, MASTER_WEIGHT_DIC)
        opt_states_path = os.path.join(resume_from_checkpoint, OPTIMIZER_STATE_DIC)
        model_states_path = os.path.join(resume_from_checkpoint, MODEL_STATE_DIC)

        if self.args.load_from_hf:
            hf_aoa_config = self.model._gen_aoa_config(self.model.config)
            hcg = dist.fleet.get_hybrid_communicate_group()
            assert (
                self.args.ignore_load_lr_and_optim
            ), "Loading from HuggingFace format is only allowed when learning rate and optimizer state are ignored."
            try:
                moe_sharding_group = hcg.get_moe_sharding_parallel_group()
            except Exception:
                moe_sharding_group = None

            if moe_sharding_group is None or moe_sharding_group.nranks <= 1:
                # when moe_sharding_group is None, we use the default process_group
                logger.info(f"Loading model weights from '{resume_from_checkpoint}' in safetensors format.")
                dist.load_state_dict(
                    model_sharded_state_dict,
                    resume_from_checkpoint,
                    aoa_config=hf_aoa_config,
                    offload=self.args.load_via_cpu,
                    safetensors=True,
                    process_group=None,
                    comm_method=self.args.flex_ckpt_comm_method,
                )
            else:
                try:
                    pp_group = hcg.get_pipe_parallel_group()
                    if pp_group is None or pp_group.nranks < 1:
                        raise NotImplementedError("Only support when pp_group is not None.")
                except Exception:
                    raise RuntimeError("Only support when pp_group is not None.")

                try:
                    moe_group = hcg.get_expert_parallel_group()
                    if moe_group is None or moe_group.nranks < 1:
                        raise NotImplementedError("Only support when moe_group is not None.")
                except Exception:
                    raise RuntimeError("Only support when moe_group is not None.")
                moe_sharding_rank = moe_sharding_group.rank
                cur_rank = dist.get_rank()
                if moe_sharding_rank == 0:
                    moe_group_ranks = []
                    dist.all_gather_object(moe_group_ranks, cur_rank, group=moe_group)
                    pp_group_ranks = []
                    dist.all_gather_object(pp_group_ranks, moe_group_ranks, group=pp_group)
                    process_group_ranks = [rank for ranks in pp_group_ranks for rank in ranks]
                else:
                    process_group_ranks = [0] * (pp_group.nranks * moe_group.nranks)
                src_rank = hcg.get_moe_sharding_parallel_group_src_rank()
                dist.broadcast_object_list(process_group_ranks, src=src_rank, group=moe_sharding_group)
                assert any(process_group_ranks), "process_group_ranks should not be all 0"
                logger.info(f"Creating a temporary process group with ranks: {process_group_ranks}")
                process_group = dist.new_group(process_group_ranks)

                if moe_sharding_rank == 0:
                    logger.info(f"Loading model weights from '{resume_from_checkpoint}' in safetensors format.")
                    # Only the first moe_sharding process is allowed to load the model weights.
                    dist.load_state_dict(
                        model_sharded_state_dict,
                        resume_from_checkpoint,
                        aoa_config=hf_aoa_config,
                        offload=self.args.load_via_cpu,
                        safetensors=True,
                        process_group=process_group,
                        comm_method=self.args.flex_ckpt_comm_method,
                    )

                dist.barrier()
                logger.info("Destroying the temporary process group.")
                dist.destroy_process_group(process_group)
                # The first moe_sharding group loads the model weights and then broadcasts them to all other moe_sharding groups.
                logger.info(
                    "First shard (moe_sharding_group) has loaded safetensors weights, starting broadcast on moe_sharding_groups."
                )
                for param_name, param in self.model.state_dict().items():
                    dist.broadcast(param, src=src_rank, group=moe_sharding_group)
            return

        if not self.args.ignore_load_lr_and_optim:
            state_dict_metadata = {}
            metadata_paths = [
                os.path.join(model_states_path, get_metadata_file_name(model_states_path)),
                os.path.join(opt_states_path, get_metadata_file_name(opt_states_path)),
                os.path.join(master_weights_path, get_metadata_file_name(master_weights_path)),
            ]

            for metadata_file in metadata_paths:
                if not os.path.exists(metadata_file):
                    raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
                metadata = paddle.load(metadata_file)
                state_dict_metadata.update(metadata.state_dict_metadata)

            if not self.args.sharded_model_from_ema:
                init_optimizer(self.optimizer, model_sharded_state_dict, state_dict_metadata)

                optimizer_sharded_state_dict = self.optimizer.sharded_state_dict(model_sharded_state_dict)

                opt_states = {}
                master_weights = {}
                for k, v in optimizer_sharded_state_dict.items():
                    if k.endswith(".w_0"):
                        master_weights[k] = v
                    else:
                        opt_states[k] = v

                dist.load_state_dict(
                    opt_states,
                    opt_states_path,
                    aoa_config=self.args.aoa_config,
                    offload=self.args.load_via_cpu,
                    comm_method=self.args.flex_ckpt_comm_method,
                )

                dist.load_state_dict(
                    master_weights,
                    master_weights_path,
                    aoa_config=self.args.aoa_config,
                    offload=self.args.load_via_cpu,
                    comm_method=self.args.flex_ckpt_comm_method,
                )

            self._load_scheduler(resume_from_checkpoint)
            if self.args.tensorwise_offload_optimizer:
                logger.info("Offloading optimizer state for FC...")
                self._offload_optimizer()

        enable_bf16_opt = (
            not isinstance(self.model, LoRAModel) and self.args.enable_zero_cost_checkpoint and self.args.bf16
        )
        logger.debug(f"sharded_model_from_ema: {self.args.sharded_model_from_ema}")
        logger.debug(f"enable_bf16_opt: {enable_bf16_opt}")

        if self.args.sharded_model_from_ema:
            ema_states_path = os.path.join(resume_from_checkpoint, EMA_STATE_DIC, f"{dist.get_rank()}_0.distcp")
            ema_state_dict = paddle.load(ema_states_path)
            ema_master_weights = ema_state_dict.pop("master_weights", None)
            opt_state_dict = {"master_weights": ema_master_weights}
            self.optimizer.set_state_dict(opt_state_dict)

            self.model.set_state_dict(ema_state_dict)
        else:

            def bf16_filtered_sharded_state_dict(sharded_state_dict):
                new_state_dict = {}
                for k, v in sharded_state_dict.items():
                    if v.local_tensor.dtype == paddle.bfloat16:
                        continue
                    new_state_dict[k] = v
                return new_state_dict

            if enable_bf16_opt:
                model_sharded_state_dict = bf16_filtered_sharded_state_dict(model_sharded_state_dict)

            dist.load_state_dict(
                model_sharded_state_dict,
                model_states_path,
                aoa_config=self.args.aoa_config,
                offload=self.args.load_via_cpu,
                comm_method=self.args.flex_ckpt_comm_method,
            )

        if enable_bf16_opt and (not self.args.ignore_load_lr_and_optim):
            opt_state_dict = self.optimizer.state_dict()

            def recover_params_from_master_weight(opt_state_dict, group):
                master_weights = opt_state_dict["master_weights"]
                tmp = OrderedDict()
                (master_weights, tmp) = (tmp, master_weights)
                # cast to before
                for (k, v) in tmp.items():
                    name = v.name
                    master_weights[k] = paddle.cast(to_device(v), paddle.bfloat16).cpu()
                    master_weights[k].name = name

                structure_name_map = {k: v.name for (k, v) in self.model.state_dict().items()}
                node_model_state = reshard_util.NodeModelState(group=group)
                node_model_state_tmp = reshard_util.NodeModelState(group=group)
                node_model_state_tmp.add_master_weights(master_weights)
                node_model_state_tmp.pack_keys(structure_name_map)
                node_model_state.merge_from(node_model_state_tmp, max(group.rank, 0))
                del node_model_state_tmp
                sharding_strategy = reshard_util.get_sharding_strategy(self.optimizer)
                logger.debug(f"sharding_strategy: {sharding_strategy}")
                restore_func = (
                    reshard_util.sharding_v1.restore
                    if sharding_strategy == SHARDING_STRATEGY_V1
                    else reshard_util.sharding_v2.restore
                )
                node_model_state = restore_func(node_model_state, self.model, self.optimizer)
                node_model_state.unpack_keys()
                master_weights = node_model_state.master_weights

                master_weights = reshard_util.all_gather_state_dict(master_weights, lambda x: True, group)

                model_state_dict = self.model.state_dict()
                for key, param in model_state_dict.items():
                    if param.name in master_weights and param.dtype == paddle.bfloat16:
                        logger.debug(
                            f"key {key}, convert master weights {param.name} shape {master_weights[param.name].shape} to param {param.name} shape{param.shape}"
                        )
                        assert (
                            param.shape == master_weights[param.name].shape
                        ), f"got {param.shape} vs {master_weights[param.name].shape}"
                        master_weight = paddle.reshape(master_weights[param.name], param.shape)
                        paddle.assign(paddle.cast(to_device(master_weight), paddle.bfloat16), model_state_dict[key])

            with paddle.no_grad():
                if paddle.distributed.is_initialized():
                    group_getter = GroupGetter(self.model)
                    opt_state_dict = split_opt_state(opt_state_dict, group_getter)
                    for gid in group_getter.get_group_ids():
                        sub_opt_state_dict = opt_state_dict[gid]
                        group = group_getter.get_group_by_id(gid)
                        if self.args.bf16:
                            recover_params_from_master_weight(sub_opt_state_dict, group)
                else:
                    master_weights = opt_state_dict["master_weights"]
                    model_state_dict = self.model.state_dict()
                    for key, param in model_state_dict.items():
                        if param.name in master_weights and param.dtype == paddle.bfloat16:
                            logger.debug(
                                f"key {key}, convert master weights {param.name} shape {master_weights[param.name].shape} to param {param.name} shape{param.shape}"
                            )
                            assert (
                                param.shape == master_weights[param.name].shape
                            ), f"got {param.shape} vs {master_weights[param.name].shape}"
                            master_weight = paddle.reshape(master_weights[param.name], param.shape)
                            paddle.assign(
                                paddle.cast(to_device(master_weight), paddle.bfloat16), model_state_dict[key]
                            )

    def prepare_resume_from_checkpoint(self, args, resume_from_checkpoint):
        logger.info(f"Starting training from resume_from_checkpoint : {resume_from_checkpoint}")

        # The resume_from_checkpoint could be None in some machine node.
        # Here we reset None to temp directory.
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")
        if args.world_size > 1:
            is_resume_from_checkpoint = paddle.to_tensor([resume_from_checkpoint is not None], dtype="int32")
            paddle.distributed.all_reduce(is_resume_from_checkpoint)
            is_resume_from_checkpoint = is_resume_from_checkpoint.item()
            if is_resume_from_checkpoint > 0 and is_resume_from_checkpoint < paddle.distributed.get_world_size():
                if resume_from_checkpoint is None:
                    resume_from_checkpoint = os.path.join(self.args.output_dir, "local_tempdir")
                    if os.path.exists(resume_from_checkpoint) and self.args.local_rank == 0:
                        shutil.rmtree(resume_from_checkpoint)
                    os.makedirs(resume_from_checkpoint, exist_ok=True)
                    logger.info(f"Reset resume_from_checkpoint to temp directory : {resume_from_checkpoint}")

        if (
            resume_from_checkpoint is not None
            and self.args.pdc_download_ckpt
            and FLASH_DEVICE not in resume_from_checkpoint
        ):
            if self.is_local_process_zero():
                download_recovery_ckpt_from_pdc(resume_from_checkpoint, self.args.pdc_download_timeout)
            if self.args.world_size > 1:
                logger.info("Wait all processes finish downloading...")
                paddle.distributed.barrier()
            logger.info("All processes finished downloading from pdc")

    def cal_epoch_step_samples(self, args, train_dataloader, total_train_batch_size):
        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = len(self.train_dataset)

            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = int(len(self.train_dataset) * args.num_train_epochs)

            if args.minimum_eval_times is not None and args.minimum_eval_times > 0:
                if max_steps // args.eval_steps < args.minimum_eval_times:
                    exp_step = max_steps / args.minimum_eval_times
                    exp_step = max(int(exp_step - exp_step % 10), 10)
                    logger.info("Reset eval step by minimum_eval_times to %d" % exp_step)
                    args.eval_steps = exp_step
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        logger.info(f"{self.runtime_timer.log()}")
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Total num train samples = {num_train_samples:,}")

        return max_steps, num_train_epochs, num_update_steps_per_epoch, num_examples, num_train_samples, len_dataloader

    def _wrap_optimizer(self, model):
        # Do nothing when not in auto parallel mode.
        if not self.args.enable_auto_parallel:
            return
        self.optimizer = parallelize.parallelize_optimizer(
            self.optimizer,
            config=self.auto_dist_config,
        )
        if hasattr(self.optimizer, "_enable_tensor_fusion") and self.args.tensor_fusion:
            self.optimizer._enable_tensor_fusion()
        if hasattr(self.optimizer, "_enable_sharding_overlap") and self.args.overlap:
            self.optimizer._enable_sharding_overlap(model)

        if dist.in_auto_parallel_align_mode():
            orig_step = (
                self.optimizer.step.__func__ if hasattr(self.optimizer.step, "__func__") else self.optimizer.step
            )
            decorator = _patch_grads_for_step(amp_master_grad=self.args.amp_master_grad)
            new_step = decorator(orig_step)
            self.optimizer.__dict__["step"] = types.MethodType(new_step, self.optimizer)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
        """
        args = self.args
        self.is_in_train = True

        # The resume_from_checkpoint could be None in some machine node.
        # Here we reset None to temp directory.
        self.prepare_resume_from_checkpoint(args, resume_from_checkpoint)

        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size
        (
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            len_dataloader,
        ) = self.cal_epoch_step_samples(args, train_dataloader, total_train_batch_size)

        # delay_optimizer_creation = (
        #     self.sharding is not None
        #     and ShardingOption.SHARD_OP in self.args.sharding
        # )
        delay_optimizer_creation = False

        if not self.args.enable_auto_parallel:
            if not delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # produce flex_checkpoint.
        if not self.args.enable_auto_parallel:
            if (
                not self.args.should_load_sharding_stage1_model
                and not self.args.load_checkpoint_format == "flex_checkpoint"
            ):
                self._load_from_checkpoint(resume_from_checkpoint)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        model = self._wrap_model(self.model_wrapped)
        # wrap optimizer and amp only for auto parallel. Non-auto model is processed in _wrap_model.
        if self.args.enable_auto_parallel:
            if not delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            self._wrap_optimizer(model)

            if self.args.fp16 or self.args.bf16:
                self._wrap_amp_model(self.args, model)

        if model is not self.model:
            self.model_wrapped = model

        if self.args.enable_auto_parallel:
            if resume_from_checkpoint is not None:
                if self.args.convert_from_hf:
                    model_sharded_state_dict = model.sharded_state_dict()
                    aoa_config = model._gen_aoa_config(model.config)
                    dist.load_state_dict(
                        model_sharded_state_dict,
                        resume_from_checkpoint,
                        aoa_config=aoa_config,
                        offload=False,
                        safetensors=True,
                    )
                else:
                    self._load_flex_checkpoint(resume_from_checkpoint)
        else:
            if self.args.should_load_sharding_stage1_model and self.args.load_checkpoint_format != "flex_checkpoint":
                if self.sharding_io is not None:
                    # the self.optimizer should be wrapped and it is done in _wrap_model
                    self.sharding_io.set_optimizer(self.optimizer)
                # Should invoke _load_from_checpoint after _load_optimizer_and_scheduler
                # because the _load_from_checkpoint method rely on the optimizer in the shareded mode.
                if resume_from_checkpoint:
                    self._load_optimizer_and_scheduler(resume_from_checkpoint)
                    self._load_from_checkpoint(resume_from_checkpoint)

            elif self.args.should_save_sharding_stage1_model and not (
                self.args.load_checkpoint_format == "flex_checkpoint"
            ):
                if self.sharding_io is not None:
                    assert delay_optimizer_creation is False, "delay_optimizer_creation should be False"
                    # the self.optimizer should be wrapped and it is done in _wrap_model
                    self.sharding_io.set_optimizer(self.optimizer)

                if delay_optimizer_creation:
                    self.create_optimizer_and_scheduler(num_training_steps=max_steps)
                self._load_optimizer_and_scheduler(resume_from_checkpoint)
            elif self.args.load_checkpoint_format == "flex_checkpoint" and resume_from_checkpoint is not None:
                if delay_optimizer_creation:
                    self.create_optimizer_and_scheduler(num_training_steps=max_steps)
                if ShardingOption.FULL_SHARD in self.args.sharding:
                    model.init_slice_param()
                    model.init_optimizer_for_slice_param()
                self._load_flex_checkpoint(resume_from_checkpoint)
                if ShardingOption.FULL_SHARD in self.args.sharding:
                    model.align_param_to_buffer_and_clear_slice_param()

            else:
                if delay_optimizer_creation:
                    self.create_optimizer_and_scheduler(num_training_steps=max_steps)
                self._load_optimizer_and_scheduler(resume_from_checkpoint)

        if self.args.enable_zero_cost_checkpoint:
            self.create_zcc_manager(model, resume_from_checkpoint)

        elif self.args.zcc_save_ema_coef is not None:
            self.add_non_zcc_ema_callback(resume_from_checkpoint)

        if self.args.using_sonic_moe:
            callback = InterleaveGateUpCallback(self.model, resume_from_checkpoint, self.args.output_dir)
            self.add_callback(callback)

        self.log_trainable_numel(model)

        return self._inner_training_loop(
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
        )

    def log_trainable_numel(self, model):
        # per_device_trainable_numel = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
        # TODO: Temporary fix since Tensor.numel() not supported in distributed mode
        if self.args.enable_auto_parallel:
            per_device_trainable_numel = 0
            for p in model.parameters():
                if not p.stop_gradient:
                    per_device_trainable_numel += np.prod(p._local_shape) if p.is_dist() else np.prod(p.shape)
        else:
            per_device_trainable_numel = sum(np.prod(p.shape) for p in model.parameters() if not p.stop_gradient)
        logger.debug(f"  Number of trainable parameters = {per_device_trainable_numel:,} (per device)")
        if self.args.use_hybrid_parallel:
            # todo fix for pipeline_model_parallel_size
            parts_num = max(self.args.tensor_model_parallel_size, 1) * max(self.args.pipeline_model_parallel_size, 1)
            if parts_num > 1:
                all_reduce_dtype = "int64"
                if paddle.get_device().split(":")[0] in ["npu", "xpu"]:
                    # TODO(duanyanhui): fix when NPU all_reduce supports int64
                    all_reduce_dtype = "float32"
                trainable_numel_tensor = paddle.to_tensor(per_device_trainable_numel, dtype=all_reduce_dtype)
                paddle.distributed.all_reduce(trainable_numel_tensor)
                trainable_numel = int(trainable_numel_tensor.item()) // self.args.dataset_world_size
                if self.args.sep_parallel_size > 0:
                    trainable_numel = trainable_numel // self.args.sep_parallel_size
                # the numel is roughly, because the tensor parallel still hold own bias or layer_norm weight without splited
                # so, the trainable numel is a little bigger than real.
                logger.debug(f"  Number of trainable parameters = {trainable_numel:,} (all devices, roughly)")

    def _split_batches_for_accumulation(self, inputs):
        if self.args.gradient_accumulation_steps == 1:
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

        skip_next_i = False
        for i, (key, dtensors) in enumerate(inputs.items()):
            if skip_next_i:
                skip_next_i = False
                continue
            if isinstance(dtensors, paddle.Tensor):
                if self.dense_tensor_idx is not None and self.dense_tensor_idx[i] != []:
                    next_dtensor = dtensors[i + 1]
                    if isinstance(next_dtensor, paddle.Tensor):
                        next_dtensor_list = (
                            paddle.prod(next_dtensor, axis=-1) if len(next_dtensor.shape) != 1 else next_dtensor
                        )
                        global_datas = dtensors.split(next_dtensor_list.cast("int64").tolist(), axis=0)
                        for index in range(self.args.gradient_accumulation_steps):
                            tensor_list = []
                            for offset in range(self.args.per_device_train_batch_size):
                                tensor_list.append(
                                    global_datas[index * self.args.per_device_train_batch_size + offset]
                                )
                            concat_tensor = paddle.cat(tensor_list, axis=0)
                            global_micro_batchs[index].update({key: [concat_tensor]})
                        global_datas_next = next_dtensor.split(self.args.gradient_accumulation_steps, axis=0)
                        for index, data in enumerate(global_datas):
                            global_micro_batchs[index].update({key: data})
                    elif isinstance(next_dtensor, int):
                        global_datas = dtensors.split(next_dtensor, axis=0)
                        for index, data in enumerate(global_datas):
                            global_micro_batchs[index].update({key: data})
                        for index in range(self.args.gradient_accumulation_steps):
                            global_micro_batchs[index].update({key: next_dtensor})
                    else:
                        raise ValueError(f"unsupported split dense_tensor with type: {type(next_dtensor)}")
                    skip_next_i = True
                else:
                    mesh, placements = dtensors.process_mesh, dtensors.placements
                    global_datas = split_dtensor_by_axis(dtensors, 0)
                    for index, data in enumerate(global_datas):
                        global_micro_batchs[index].update({key: dist.reshard(data, mesh, placements)})
            elif isinstance(dtensors, (list, tuple)):
                if len(dtensors) == 0:
                    for j in range(self.args.gradient_accumulation_steps):
                        global_micro_batchs[j].update({key: []})
                else:
                    skip_next_j = False
                    for j, dtensor in enumerate(dtensors):
                        if skip_next_j:
                            skip_next_j = False
                            continue
                        if isinstance(dtensor, paddle.Tensor):
                            if self.dense_tensor_idx is not None and j in self.dense_tensor_idx[i]:
                                next_dtensor = dtensors[j + 1]
                                if isinstance(next_dtensor, paddle.Tensor):
                                    next_dtensor_list = (
                                        paddle.prod(next_dtensor, axis=-1)
                                        if len(next_dtensor.shape) != 1
                                        else next_dtensor
                                    )
                                    global_datas = dtensor.split(next_dtensor_list.cast("int64").tolist(), axis=0)
                                    for index in range(self.args.gradient_accumulation_steps):
                                        tensor_list = []
                                        for offset in range(self.args.per_device_train_batch_size):
                                            tensor_list.append(
                                                global_datas[index * self.args.per_device_train_batch_size + offset]
                                            )
                                        concat_tensor = paddle.cat(tensor_list, axis=0)
                                        if key in global_micro_batchs[index].keys():
                                            global_micro_batchs[index][key].append(concat_tensor)
                                        else:
                                            global_micro_batchs[index].update({key: [concat_tensor]})

                                    global_datas_next = next_dtensor.split(
                                        self.args.gradient_accumulation_steps, axis=0
                                    )
                                    for index, data in enumerate(global_datas_next):
                                        if key in global_micro_batchs[index].keys():
                                            global_micro_batchs[index][key].append(data)
                                        else:
                                            global_micro_batchs[index].update({key: [data]})
                                elif isinstance(next_dtensor, int):
                                    global_datas = dtensor.split(next_dtensor, axis=0)
                                    for index, data in enumerate(global_datas):
                                        if key in global_micro_batchs[index].keys():
                                            global_micro_batchs[index][key].append(data)
                                        else:
                                            global_micro_batchs[index].update({key: [data]})
                                    for index in range(self.args.gradient_accumulation_steps):
                                        if key in global_micro_batchs[index].keys():
                                            global_micro_batchs[index][key].append(next_dtensor)
                                        else:
                                            global_micro_batchs[index].update({key: next_dtensor})
                                else:
                                    raise ValueError(f"unsupported split dense_tensor with type: {type(next_dtensor)}")
                                skip_next_j = True
                            else:
                                mesh, placements = dtensor.process_mesh, dtensor.placements
                                global_datas = split_dtensor_by_axis(dtensor, 0)
                                for index, data in enumerate(global_datas):
                                    if key in global_micro_batchs[index].keys():
                                        global_micro_batchs[index][key].append(dist.reshard(data, mesh, placements))
                                    else:
                                        global_micro_batchs[index].update(
                                            {key: [dist.reshard(data, mesh, placements)]}
                                        )
                        else:
                            raise ValueError(f"unsupported type: {type(dtensor)}")
            else:
                raise ValueError(f"unsupported type: {type(dtensors)}")
        return global_micro_batchs

    def optimizer_step(self, args, model, parameters_list=None):
        if parameters_list is None:
            parameters_list = []

        optimizer_was_run = True
        if not args.enable_auto_parallel and self.args.offload_optim:
            self._reload_optimizer()

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
        elif isinstance(self.optimizer, HybridParallelOptimizer):
            self.optimizer._step(parameters_list)
        else:
            self.optimizer.step()

        if not args.enable_auto_parallel and self.args.offload_optim:
            self._offload_optimizer()

        if optimizer_was_run:
            self.lr_scheduler.step()

        enable_release_grads = False
        if args.sharding_parallel_size > 1:
            enable_release_grads = args.sd_release_grads
        if not enable_release_grads and args.pipeline_model_parallel_size > 1:
            enable_release_grads = args.pp_release_grads

        if not args.enable_auto_parallel and (args.release_grads or enable_release_grads):
            self.optimizer.clear_grad(set_to_zero=False)
            if args.pipeline_model_parallel_size > 1:
                for _, buffers in model._chunk_2_comm_buffers.items():
                    for buffer in buffers:
                        buffer._clear_grad_storage()
        else:
            self.optimizer.clear_grad()

    def _get_meshes_for_loader(self):
        return self.global_mesh.get_mesh_with_dim("pp")[0]

    def _get_inputs_list(self, inputs):
        inputs_list = [inputs]
        if self.args.enable_auto_parallel:
            inputs_list = self._split_batches_for_accumulation(inputs)
            for inputs in inputs_list:
                if self.args.sep_parallel_size > 1 and self.args.split_inputs_sequence_dim:
                    inputs = auto_split_inputs_sequence_dim(inputs)
                if self.args.context_parallel_size > 1 and self.args.split_inputs_sequence_dim:
                    inputs = auto_split_sequence_dim_load_balance(inputs)
        else:
            if (
                self.args.use_hybrid_parallel
                and self.args.sep_parallel_size > 1
                and self.args.split_inputs_sequence_dim
            ):
                inputs = split_inputs_sequence_dim(inputs)
            if (
                self.args.use_hybrid_parallel
                and self.args.context_parallel_size > 1
                and is_paddlefleet_available()
                and FleetGPTModel is not None
                and isinstance(self.model, FleetGPTModel)
                and get_batch_on_this_cp_rank is not None
            ):
                inputs = get_batch_on_this_cp_rank(inputs)

            if self.args.ignore_data_skip:
                self.timers and self.timers("read-data").stop()

            inputs_list = [inputs]
        return inputs_list

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
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")
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
        self.state.consumed_samples = 0

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        tr_loss = paddle.to_tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        if self.args.device == "npu" and self.args.flatten_param_grads:
            from .plugins.npu_plugin import npu_accelerate_plugin

            npu_accelerate_plugin(self.optimizer)

        if self.args.ignore_data_skip:
            self.timers and self.timers("read-data").start()

        if self.resume_from_custom_func is not None:
            self.resume_from_custom_func(model)

        for epoch in range(epochs_trained, num_train_epochs):
            if (
                not args.enable_auto_parallel
                and isinstance(train_dataloader, paddle.io.DataLoader)
                and isinstance(train_dataloader.batch_sampler, DistributedBatchSampler)
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            step_control = 0  # used in loop control, reset to 0 after every step
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1

            for step, inputs in enumerate(epoch_iterator):

                if self.args.profile:
                    perf_utils.switch_profile(
                        self.state.global_step,
                        self.args.profile_step_start,
                        self.args.profile_step_end,
                        enable_layerwise_event=True,
                    )
                os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
                self.callback_handler.on_load_data_end(args, self.state, self.control, inputs=inputs)

                # Skip past any already trained steps if resuming training
                # for paddleformers.utils.batch_sampler.DistributedBatchSampler
                # We use consumed_samples to reset the status
                dataloader = train_dataloader
                if self.args.enable_auto_parallel:
                    dataloader = train_dataloader._dataloader

                if isinstance(dataloader, paddle.io.DataLoader) and isinstance(
                    dataloader.batch_sampler, NlpDistributedBatchSampler
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

                inputs_list = self._get_inputs_list(inputs)

                if not self.args.enable_auto_parallel and should_skip_data(
                    self.state.global_step, self.args.skip_data_intervals
                ):
                    # skip this step

                    if (step_control + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                    ):
                        # update current global step and skip step
                        self.state.global_step += 1
                        self._skip_global_steps += 1
                        self._skip_steps_since_last_logged += 1

                        self.state.epoch = epoch + (step + 1) / steps_in_epoch

                        # For ZCC EMA
                        if self.args.enable_zero_cost_checkpoint or self.args.zcc_save_ema_coef is not None:
                            tr_loss_for_zcc = tr_loss.clone()
                            dist.all_reduce(
                                tr_loss_for_zcc, dist.ReduceOp.SUM
                            )  # 3级并行时，每个pp下的loss会广播，全局reduce-mean的时候，分子分母都会乘以pp_world_size，结果会被约掉
                            tr_loss_for_zcc_scalar = tr_loss_for_zcc.item() / dist.get_world_size()
                            self.state.loss = tr_loss_for_zcc_scalar

                        self.state.consumed_samples = (
                            self.state.global_step
                            * args.per_device_train_batch_size
                            * args.gradient_accumulation_steps
                            * args.dataset_world_size
                        )

                        if self.state.global_step == 1 and self.args.logging_first_step:
                            self.control.should_log = True
                        if (
                            self.args.logging_strategy == IntervalStrategy.STEPS
                            and self.state.global_step % self.args.logging_steps == 0
                        ):
                            self.control.should_log = True

                        self.control.should_evaluate = False
                        self.control.should_save = False

                        # log loss and memory usage
                        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
                        self._print_timer()
                        step_control = 0
                    else:
                        step_control += 1
                    if self.state.global_step >= self.state.max_steps:
                        break

                    self.timers and self.timers("read-data").start()
                    continue

                for inputs in inputs_list:
                    if step_control % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                        self.timers and self.timers("forward-backward").start()

                    sync_context = None
                    if not self.args.enable_auto_parallel:
                        # stage2 and stage3 should not no_sync, because the is no DDP wrapper and no_sync API
                        # hybrid_parallel (tp or pp or sharding stage 1) should not no_sync
                        available_no_sync = hasattr(model, "no_sync")
                        is_no_sync = (
                            (
                                ((step_control + 1) % args.gradient_accumulation_steps != 0)
                                and args._no_sync_in_gradient_accumulation
                            )
                            or args.recompute_granularity is not None
                            or args.use_expert_parallel
                        ) and available_no_sync
                        # sharding
                        # stage1. the same as ddp
                        # stage2. manually collect gradient on dp group

                        dp_master_grad = (
                            self.args.world_size > 1
                            and self.args.amp_master_grad
                            and not self.args.use_hybrid_parallel
                        )
                        if dp_master_grad:
                            is_no_sync = True

                        sync_context = model.no_sync() if is_no_sync else contextlib.nullcontext()
                        if self.args.count_trained_tokens:
                            self.trained_effective_tokens += (inputs["input_ids"] != self.args.pad_token_id).sum()
                            self.trained_tokens += inputs["input_ids"].numel()

                    if not self.args.enable_auto_parallel:
                        with sync_context:
                            if "step_control" in inspect.signature(self.training_step).parameters:
                                tr_loss_step = self.training_step(model, inputs, step_control=step_control)
                            else:
                                tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                    if self.args.enable_auto_parallel:
                        with _exec_mode_guard("dynamic"):
                            tr_loss += tr_loss_step
                    else:
                        tr_loss += tr_loss_step

                    def fused_allreduce_gradients_no_sync(paramlist, hcg):
                        paramlist = list(paramlist)
                        nonmoe_list = [p for p in paramlist if not getattr(p, "no_sync", False)]
                        moelist = [p for p in paramlist if getattr(p, "no_sync", False)]
                        if moelist and not self.args.use_expert_parallel:
                            logger.warning("found `no sync` param when `use_expert_parallel=False`")
                        fused_allreduce_gradients(nonmoe_list, hcg)

                    def hybrid_parallel_scale_param_grad(paramlist, hcg):
                        if not hasattr(hcg, "get_context_parallel_world_size"):
                            cp_worldsize = 1
                        else:
                            cp_worldsize = hcg.get_context_parallel_world_size()

                        for p in paramlist:
                            color = getattr(p, "color", -1)
                            is_expert = isinstance(color, dict) and color.get("color", -1) == "moe_expert"
                            disable_scale_grad = getattr(p, "context_parallel_disable_scale_grad", False)
                            if not (disable_scale_grad or is_expert) and cp_worldsize > 1:
                                grad = getattr(p, "main_grad", p.grad)
                                if grad is not None:
                                    with paddle.no_grad():
                                        coeff = cp_worldsize
                                        grad.scale_(coeff)
                            elif is_expert and self.args.hybrid_parallel_expert_grad_scale != 1.0:
                                grad = getattr(p, "main_grad", p.grad)
                                if grad is not None:
                                    with paddle.no_grad():
                                        coeff = self.args.hybrid_parallel_expert_grad_scale
                                        grad.scale_(coeff)

                    disable_accumulation = False

                    if (step_control + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                        or disable_accumulation
                    ):
                        if self._enable_delay_scale_loss():
                            if self.args.enable_auto_parallel and self.args.gradient_accumulation_steps > 1:
                                tr_loss /= self.args.gradient_accumulation_steps
                            if not self.args.enable_auto_parallel and self.args.pipeline_model_parallel_size <= 1:
                                tr_loss /= self.args.gradient_accumulation_steps

                        # assert if loss is invalid
                        self._check_loss_valid(tr_loss)

                        self.timers and self.timers("forward-backward").stop()

                        parameters_list = None
                        if not args.enable_auto_parallel:
                            # Manually collect gradients
                            # Case 1: Use recompute and dp
                            # Case 2: Hack dp with master_grad
                            # Case 3: Pipeline or sharding overlap
                            # local_rank != -1 don't means dp in networks.
                            self.timers and self.timers("all-reduce").start()
                            if hasattr(self.optimizer, "_hcg"):
                                hybrid_parallel_scale_param_grad(list(model.parameters()), self.optimizer._hcg)

                            # Case 1: Use recompute and dp / sharding stage1,
                            # manually collect gradient for dp.
                            if (
                                args.recompute_granularity is not None or args.use_expert_parallel
                            ) and available_no_sync:
                                fused_allreduce_gradients_no_sync(list(model.parameters()), None)

                            # Case 2: hack dp with master_grad
                            elif dp_master_grad:
                                fused_allreduce_gradients_no_sync(list(model.parameters()), None)

                            # Pipeline parallel mode,  handle gradient reduce here to overlap
                            enable_dp_comm_overlap = (
                                self.args.pipeline_model_parallel_size > 1 and args.dp_comm_overlap
                            )

                            enable_release_grads = False
                            if args.sharding_parallel_size > 1:
                                enable_release_grads = args.sd_release_grads
                            if not enable_release_grads and args.pipeline_model_parallel_size > 1:
                                enable_release_grads = args.pp_release_grads

                            # Case 3: Pipeline parallel mode, overlap with dp
                            if isinstance(self.optimizer, HybridParallelOptimizer) and not self.do_grad_scaling:
                                parameters_list = _obtain_optimizer_parameters_list(self.optimizer._inner_opt)

                                if not enable_dp_comm_overlap:
                                    if self.optimizer._sharding_enable:
                                        assert reshard_util.is_sharding_opt(self.optimizer)
                                        self.optimizer._inner_opt.reduce_gradients(
                                            list(parameters_list), self.optimizer._hcg
                                        )

                                    if self.optimizer._dp_enable or getattr(self.optimizer, "_sep_enable", False):
                                        fused_allreduce_gradients_no_sync(list(parameters_list), self.optimizer._hcg)
                            self.timers and self.timers("all-reduce").stop()
                            self.timers and self.timers("optimizer-step").start()

                        if (
                            not args.enable_auto_parallel
                            and self.args.gradient_accumulation_steps > 1
                            and self._enable_delay_scale_loss()
                        ):
                            paddle.device.synchronize()
                            for p in model._layers.parameters():
                                with paddle.no_grad():
                                    if hasattr(p, "main_grad") and p.main_grad is not None:
                                        assert p.grad is None
                                        p.main_grad.scale_(1.0 / self.args.gradient_accumulation_steps)
                                    elif p.grad is not None:
                                        p.grad.scale_(1.0 / self.args.gradient_accumulation_steps)

                        # Optimizer step
                        self.callback_handler.on_optimizer_begin(
                            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                        )

                        self.optimizer_step(args, model=model, parameters_list=parameters_list)

                        self.timers and self.timers("optimizer-step").stop()

                        self.callback_handler.on_optimizer_end(
                            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                        )

                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1) / steps_in_epoch
                        self.state.consumed_samples = (
                            self.state.global_step
                            * args.per_device_train_batch_size
                            * args.gradient_accumulation_steps
                            * args.dataset_world_size
                        )
                        # For ZCC EMA
                        if self.args.enable_zero_cost_checkpoint or self.args.zcc_save_ema_coef is not None:
                            tr_loss_for_zcc = tr_loss.clone()
                            dist.all_reduce(
                                tr_loss_for_zcc, dist.ReduceOp.SUM
                            )  # 3级并行时，每个pp下的loss会广播，全局reduce-mean的时候，分子分母都会乘以pp_world_size，结果会被约掉
                            tr_loss_for_zcc_scalar = tr_loss_for_zcc.item() / dist.get_world_size()
                            self.state.loss = tr_loss_for_zcc_scalar

                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
                        self._print_timer()
                        step_control = 0
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                        step_control += 1

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                if self.args.ignore_data_skip:
                    self.timers and self.timers("read-data").start()

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            if self.args.enable_auto_parallel:
                with _exec_mode_guard("dynamic"):
                    self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
            else:
                self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        if not self.args.enable_auto_parallel and self.args.enable_zero_cost_checkpoint:
            self.zcc_manager.finalize()
        logger.info("\nTraining completed. \n")

        if not self.args.enable_auto_parallel:
            # unlink shared_memory if used.
            if (
                self.args.save_checkpoint_format == "unified_checkpoint"
                or self.args.load_checkpoint_format == "unified_checkpoint"
            ):
                self.unified_checkpoint_handler.unlink_shared_memory()

            if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
                if args.local_rank != -1:
                    dist.barrier()

                logger.info(
                    f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
                )
                if isinstance(self.model, LoRAModel):
                    self._load_best_model_from_peft_checkpoint()
                else:
                    if self.args.load_checkpoint_format == "unified_checkpoint":
                        self.unified_checkpoint_handler.load_unified_checkpoint(
                            self.model,
                            self.state.best_model_checkpoint,
                            convert_from_hf=self.args.convert_from_hf,
                        )
                        if self.args.sharding_parallel_size > 1 or self.args.data_parallel_size > 1:
                            broadcast_dataset_rank0_model(self.model)
                    else:
                        weight_name = PADDLE_WEIGHTS_NAME
                        best_model_path = os.path.join(
                            self.state.best_model_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)
                        )
                        if os.path.exists(best_model_path):
                            # We load the model state dict on the CPU to avoid an OOM error.
                            state_dict = paddle.load(best_model_path, return_numpy=True)
                            # If the model is on the GPU, it still works!
                            self._set_state_dict_in_model(state_dict)
                        else:
                            logger.warning(
                                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                                "on multiple nodes, you should activate `--save_on_each_node`."
                            )

        self._total_loss_scalar += self._get_item_from_loss(tr_loss)

        # In case all steps were skipped, the total loss is set to 0.
        if self.state.global_step == self._skip_global_steps:
            logger.info("All steps were skipped, the total loss is set to 0.")
            train_loss = 0.0
        else:
            train_loss = self._total_loss_scalar / (self.state.global_step - self._skip_global_steps)

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        kwargs = {
            "metrics_dumper": self.metrics_dumper,
        }
        self.control = self.callback_handler.on_train_end(args, self.state, self.control, **kwargs)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _load_best_model_from_peft_checkpoint(self):
        if self.args.save_checkpoint_format == "unified_checkpoint":
            self.unified_checkpoint_handler.load_unified_checkpoint(
                self.model,
                self.state.best_model_checkpoint,
                convert_from_hf=self.args.convert_from_hf,
            )
            if self.args.sharding_parallel_size > 1 or self.args.data_parallel_size > 1:
                broadcast_dataset_rank0_model(self.model)
            return

        convert_tp = False
        if isinstance(self.model, LoRAModel):
            if self.model.quantized or self.args.pipeline_model_parallel_size > 1:
                best_model_path = os.path.join(
                    self.state.best_model_checkpoint, _add_variant(LORA_WEIGHTS_NAME, self.args.weight_name_suffix)
                )
            else:
                best_model_path = os.path.join(self.state.best_model_checkpoint, LORA_WEIGHTS_NAME)
                if self.model.lora_config.tensor_model_parallel_size > 1:
                    convert_tp = True

        if os.path.exists(best_model_path):
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = paddle.load(best_model_path, return_numpy=True)
            if convert_tp:
                state_dict = self.model._convert_tensor_parallel(state_dict)
            # If the model is on the GPU, it still works!
            self._set_state_dict_in_model(state_dict)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        shuffle = True if self.args.enable_auto_parallel else self.args.dataloader_shuffle
        total_batch_size = self.args.per_device_train_batch_size
        if self.args.enable_auto_parallel:
            total_batch_size = total_batch_size * self.args.dataset_world_size * self.args.gradient_accumulation_steps

        if self.args.enable_auto_parallel or self.args.world_size <= 1:
            return paddle.io.BatchSampler(
                dataset=self.train_dataset,
                shuffle=shuffle,
                batch_size=total_batch_size,
                drop_last=self.args.dataloader_drop_last,
            )

        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=total_batch_size,
            shuffle=shuffle,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )

    def _set_state_dict_in_model(self, state_dict):
        # TODO  @ZHUI paddle need return the results of set_state_dict.
        logger.info(f"set state-dict :{self.model.set_state_dict(state_dict)}")

    def _print_timer(self):
        """print timer and clear states"""
        paddle_timer_info = ""
        try:
            from paddle.distributed.fleet.utils.timer_helper import (
                get_timers as paddle_get_timers,
            )

            paddle_pipeline_timers = paddle_get_timers()
            for name, timer in paddle_pipeline_timers.timers.items():
                elapsed_time = timer.elapsed(reset=False) * 1000.0
                paddle_timer_info += f" | {name}: {elapsed_time:.2f}"
            paddle_pipeline_timers.log(paddle_pipeline_timers.timers.keys(), reset=True)
        except AssertionError:  # paddle timer not enabled
            pass

        if self.timers is not None:
            timer_info = self.timers.log(self.timers.timers.keys(), reset=True)
        else:
            timer_info = ""

        if timer_info or paddle_timer_info:
            logger.info(f"[Profile global_step: {self.state.global_step}] {timer_info} {paddle_timer_info}")

    def _check_loss_valid(self, loss):
        assert isinstance(loss, paddle.Tensor) and loss._is_initialized()
        loss_value = loss.item()
        if not self.args.fp16:
            if not np.isfinite(loss_value).all():
                err_msg = LOSS_NAN_ERROR if np.isnan(loss_value).any() else LOSS_INF_ERROR
                raise ValueError(f"{err_msg}. Loss contains inf or nan values, its value is {loss_value}")

    def _get_item_from_loss(self, loss):
        assert isinstance(loss, paddle.Tensor) and loss._is_initialized()
        if isinstance(loss, paddle.Tensor):
            if loss.is_dist():
                return loss._local_value().item() if loss._is_initialized() else 0.0
            else:
                return loss.item() if loss._is_initialized() else 0.0
        else:
            return loss

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        if self.control.should_log:

            logs: Dict[str, float] = {}
            num_steps = self.state.global_step - self._globalstep_last_logged - self._skip_steps_since_last_logged
            self._skip_steps_since_last_logged = 0
            # all_gather + mean() to get average loss over all processes
            avg_loss = self._nested_gather(tr_loss).mean()
            tr_loss_scalar = self._get_item_from_loss(avg_loss)

            # reset tr_loss to zero
            tr_loss.subtract_(tr_loss)
            # set loss to zero if all steps are skipped since last log
            if num_steps == 0:
                logs["loss"] = 0.0
            else:
                logs["loss"] = round(tr_loss_scalar / num_steps, 8)

            logs["learning_rate"] = float("{0:.3e}".format(self._get_learning_rate()))
            logs["global_step"] = int(self.state.global_step)
            if in_auto_parallel_align_mode():
                logs["loss_md5"] = avg_loss._md5sum()

            divisor = 2**30
            # TODO(@gexiao): replace these codes with unified APIs in Paddle
            current_device = framework._current_expected_place_()
            if str(current_device) != "Place(cpu)":
                device_id = current_device.get_device_id()
                current_memory_allocated = core.device_memory_stat_current_value("Allocated", device_id)
                current_memory_reserved = core.device_memory_stat_current_value("Reserved", device_id)
                max_memory_allocated = core.device_memory_stat_peak_value("Allocated", device_id)
                max_memory_reserved = core.device_memory_stat_peak_value("Reserved", device_id)
                logs["current_memory_allocated"] = current_memory_allocated / divisor
                logs["current_memory_reserved"] = current_memory_reserved / divisor
                logs["max_memory_allocated"] = max_memory_allocated / divisor
                logs["max_memory_reserved"] = max_memory_reserved / divisor

            total_train_batch_size = (
                self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.dataset_world_size
            )

            seq_length = None
            model_flops_per_token = None
            if (getattr(self, "is_pretraining", False) or getattr(self.args, "benchmark", False)) and hasattr(
                self.model, "config"
            ):
                seq_length = getattr(self.model.config, "seq_length", None)
                try:
                    model_flops_per_token = self.model.get_hardware_flops()
                except NotImplementedError:
                    model_flops_per_token = None

            # Do not log speed metrics if all steps are skipped since last log.
            if num_steps > 0:
                logs.update(
                    speed_metrics(
                        "interval",
                        self._globalstep_last_start_time,
                        num_samples=total_train_batch_size * num_steps,
                        num_steps=num_steps,
                        seq_length=seq_length,
                        model_flops_per_token=model_flops_per_token,
                    )
                )

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            # Add additional memory in log.
            if not self.args.skip_memory_metrics:
                logs.update(
                    {
                        "cpu_mem_used": self._memory_tracker.cpu_mem_used() >> 20,
                        "cpu_mem_used_peak": self._memory_tracker.cpu_mem_used_peak >> 20,
                    }
                )
                if is_paddle_cuda_available():
                    logs.update(
                        {
                            "gpu_max_memory_allocated": paddle_device.max_memory_allocated() >> 20,
                            "gpu_max_memory_reserved": paddle_device.max_memory_reserved() >> 20,
                        }
                    )

            self.log(logs, **kwargs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.optimizer, GroupShardedOptimizerStage2) and self.optimizer._broadcast_overlap:
                paddle.device.synchronize()

            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            if isinstance(self.optimizer, GroupShardedOptimizerStage2) and self.optimizer._broadcast_overlap:
                paddle.device.synchronize()

            self._save_checkpoint(model, metrics=metrics)
            logger.info(f"{self.runtime_timer.log()}")
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            self.log_trained_tokens()

        if self.control.should_save_hf:
            if self.args.save_checkpoint_format == "flex_checkpoint":
                is_main_process = paddle.distributed.get_rank() == 0
                run_dir = self.args.output_dir
                checkpoint_folder = f"{PREFIX_HF_CHECKPOINT_DIR}-{self.state.global_step}"
                ckpt_path = os.path.join(run_dir, checkpoint_folder)
                if isinstance(self.model, LoRAModel):
                    self.model.save_pretrained(
                        ckpt_path,
                        merge_tensor_parallel=True,
                        variant=self.args.weight_name_suffix,
                        save_checkpoint_format=self.args.save_checkpoint_format,
                    )
                else:
                    self.model.save_pretrained(
                        ckpt_path, is_main_process, save_checkpoint_format=self.args.save_checkpoint_format
                    )
                if self.tokenizer is not None and self.args.save_tokenizer:
                    self.tokenizer.save_pretrained(ckpt_path)
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(ckpt_path)
                self.control = self.callback_handler.on_save_hf(self.args, self.state, self.control)

    def log_trained_tokens(self):
        if self.args.count_trained_tokens:
            token_list = []
            for token_num in [self.trained_effective_tokens, self.trained_tokens]:
                tensors = token_num.reshape([1])
                if self.hcg._sharding_degree > 1:
                    output_tensors = []
                    paddle.distributed.all_gather(output_tensors, tensors, group=self.hcg._sharding_comm_group)
                    tensors = paddle.cat(output_tensors).sum().reshape([1])
                if self.hcg._dp_degree > 1:
                    output_tensors = []
                    paddle.distributed.all_gather(output_tensors, tensors, group=self.hcg._dp_comm_group)
                    tensors = paddle.cat(output_tensors).sum().reshape([1])
                token_list.append(tensors.item())
            if self.is_local_process_zero():

                logger.info(
                    f"Update to now, trained_effective_tokens: {token_list[0]}, trained_tokens: {token_list[1]}."
                )

    def _get_learning_rate(self):
        return self.optimizer.get_lr()

    def get_train_dataloader(self, dense_tensor_idx=None):
        """
        Returns the training [`~paddle.io.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.args.should_load_dataset and self.train_dataset is None:
            raise ValueError("Training requires a train_dataset when should_load_dataset is True.")
        if not self.args.should_load_dataset and self.train_dataset is not None:
            raise ValueError("We don't need train_dataset when should_load_dataset is False.")

        train_dataset = self.train_dataset
        if self.args.distributed_dataloader:
            is_iterable_dataset = self._is_iterable_dataset_distributed(train_dataset)
        else:
            is_iterable_dataset = self._is_iterable_dataset(train_dataset)
        if is_datasets_available() and train_dataset is not None and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        _DataLoader = DistDataLoader if self.args.distributed_dataloader else DataLoader

        additional_configs = {}
        if is_iterable_dataset:  # For iterable dataset
            if self.args.dataset_world_size > 1 and train_dataset is not None:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.dataset_world_size,
                    process_index=self.args.dataset_rank,
                )

            if self.args.distributed_dataloader:
                logger.info("Training using DistDataLoader.")
                additional_configs = {"is_iterable_dataset": True, "pp_data_group": self._pp_data_group}
            train_dataloader = _DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                persistent_workers=self.args.dataloader_num_workers > 0,
                prefetch_factor=self.args.prefetch_factor,
                reader_buffer_size=max(self.args.gradient_accumulation_steps, 2),
                **additional_configs,
            )
        else:
            train_sampler = self._get_train_sampler()
            if self.args.distributed_dataloader:
                logger.info("Training using DistDataLoader.")
                additional_configs = {"pp_data_group": self._pp_data_group}
            train_dataloader = _DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                persistent_workers=self.args.dataloader_num_workers > 0,
                prefetch_factor=self.args.prefetch_factor,
                reader_buffer_size=max(self.args.gradient_accumulation_steps, 2),
                **additional_configs,
            )

        if self.args.enable_auto_parallel:
            self.dense_tensor_idx = dense_tensor_idx
            train_dataloader = dist.shard_dataloader(
                dataloader=train_dataloader,
                meshes=self._get_meshes_for_loader(),
                shard_dims="dp",
                dense_tensor_idx=dense_tensor_idx,
            )

        return train_dataloader

    def _get_eval_sampler(self, eval_dataset: Dataset):
        if eval_dataset is None or not has_length(eval_dataset):
            return None
        if self.args.world_size <= 1:
            return paddle.io.BatchSampler(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                drop_last=False,
            )
        else:
            if (
                is_paddlefleet_available() and isinstance(self.model, PaddleFleetPipelineLayer)
            ) or self.args.pipeline_model_parallel_size > 1:
                # In pipeline parallelism, batch size will be strictly checked
                # Use LastBatchPaddingSampler to pad the last batch with the first batch
                from .trainer_utils import LastBatchPaddingSampler

                return LastBatchPaddingSampler(
                    eval_dataset,
                    num_replicas=self.args.dataset_world_size,
                    rank=self.args.dataset_rank,
                    batch_size=self.args.per_device_eval_batch_size,
                    shuffle=False,
                    drop_last=False,
                )
            else:
                return DistributedBatchSampler(
                    eval_dataset,
                    num_replicas=self.args.dataset_world_size,
                    rank=self.args.dataset_rank,
                    batch_size=self.args.per_device_eval_batch_size,
                    shuffle=False,
                    drop_last=False,
                )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~paddle.io.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`paddle.io.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if self.args.should_load_dataset and eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset when should_load_dataset is True.")
        if not self.args.should_load_dataset and not (eval_dataset is None and self.eval_dataset is None):
            raise ValueError("We don't need eval_dataset when should_load_dataset is False.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if self.args.distributed_dataloader:
            is_iterable_dataset = self._is_iterable_dataset_distributed(eval_dataset)
        else:
            is_iterable_dataset = self._is_iterable_dataset(eval_dataset)
        if is_datasets_available() and eval_dataset is not None and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        _DataLoader = DistDataLoader if self.args.distributed_dataloader else DataLoader

        additional_configs = {}
        if is_iterable_dataset:
            if (
                self.args.dataset_world_size > 1 or self.args.pipeline_model_parallel_size > 1
            ) and eval_dataset is not None:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.dataset_world_size,
                    process_index=self.args.dataset_rank,
                )

            if self.args.distributed_dataloader:
                logger.info("Eval using DistDataLoader.")
                additional_configs = {"eval": True, "is_iterable_dataset": True, "pp_data_group": self._pp_data_group}
            return _DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=0,
                **additional_configs,
            )
        else:
            eval_sampler = self._get_eval_sampler(eval_dataset)
            if self.args.distributed_dataloader:
                logger.info("Eval using DistDataLoader.")
                additional_configs = {"eval": True, "pp_data_group": self._pp_data_group}
            return _DataLoader(
                eval_dataset,
                batch_sampler=eval_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                persistent_workers=self.args.dataloader_num_workers > 0,
                prefetch_factor=self.args.prefetch_factor,
                reader_buffer_size=max(self.args.gradient_accumulation_steps, 2),
                **additional_configs,
            )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~paddle.io.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`paddle.io.Dataset`, *optional*):
                The test dataset to use. If it is an `datasets.Dataset`, columns not accepted by the `model.forward()`
                method are automatically removed. It must implement `__len__`.
        """
        if self.args.should_load_dataset and not test_dataset:
            raise ValueError("Test requires an test_dataset when should_load_dataset is True.")
        if not self.args.should_load_dataset and test_dataset is not None:
            raise ValueError("We don't need test_dataset when should_load_dataset is False.")

        if self.args.distributed_dataloader:
            is_iterable_dataset = self._is_iterable_dataset_distributed(test_dataset)
        else:
            is_iterable_dataset = self._is_iterable_dataset(test_dataset)
        if is_datasets_available() and test_dataset is not None and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        _DataLoader = DistDataLoader if self.args.distributed_dataloader else DataLoader

        additional_config = {}
        if is_iterable_dataset:
            if self.args.dataset_world_size > 1 and test_dataset is not None:
                test_dataset = IterableDatasetShard(
                    test_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.dataset_world_size,
                    process_index=self.args.dataset_rank,
                )

            if self.args.distributed_dataloader:
                logger.info("Test using DistDataLoader.")
                additional_config = {"eval": True, "is_iterable_dataset": True, "pp_data_group": self._pp_data_group}
            return _DataLoader(
                test_dataset,
                batch_size=self.args.per_device_eval_batch_size * self.world_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                persistent_workers=self.args.dataloader_num_workers > 0,
                prefetch_factor=self.args.prefetch_factor,
                reader_buffer_size=max(self.args.gradient_accumulation_steps, 2),
                **additional_config,
            )
        else:
            test_sampler = self._get_eval_sampler(test_dataset)
            if self.args.distributed_dataloader:
                logger.info("Test using DistDataLoader.")
                additional_config = {"eval": True, "pp_data_group": self._pp_data_group}
            # We use the same batch_size as for eval.
            return _DataLoader(
                test_dataset,
                batch_sampler=test_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                **additional_config,
            )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_scheduler(num_training_steps=num_training_steps)
        self.create_optimizer(self.lr_scheduler)

    def create_optimizer(self, lr_scheduler=None):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            if self.optimizer_grouped_parameters is not None:
                params = self.optimizer_grouped_parameters
                apply_decay_param_fun = None
            else:
                params = self.model.parameters()
                decay_parameters = [
                    p.name for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])
                ]

                def apply_decay_param_fun(x):
                    return x in decay_parameters

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            if self.args.optim == OptimizerNames.ADAMW_CUSTOM:
                optimizer_kwargs["quantization_config"] = self.model.config.quantization_config
                optimizer_kwargs["use_lowprecision_moment"] = self.args.use_lowprecision_moment
                optimizer_kwargs["tensorwise_offload_optimizer"] = self.args.tensorwise_offload_optimizer

            if hasattr(optimizer_cls, "_create_master_weight") and self.args.fp16_opt_level == "O2":
                optimizer_kwargs["multi_precision"] = True

            self.optimizer = optimizer_cls(
                learning_rate=self.lr_scheduler if lr_scheduler is None else lr_scheduler,
                apply_decay_param_fun=apply_decay_param_fun,
                parameters=params,
                weight_decay=self.args.weight_decay,
                grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm) if self.args.max_grad_norm > 0 else None,
                **optimizer_kwargs,
            )

            if self.args.tensorwise_offload_optimizer:
                mock_offload_optimizer()

        return self.optimizer

    def _apply_to_optimizer(self, action):
        attributes = [
            ("_accumulators", "_moment1_acc_str"),
            ("_accumulators", "_moment2_acc_str"),
            ("_master_weights",),
            ("_accumulators_holder",),
        ]
        for attr in attributes:
            if all(hasattr(self.optimizer, a) for a in attr):
                target_attr = getattr(self.optimizer, attr[0])
                if len(attr) == 2:
                    target_attr = target_attr[getattr(self.optimizer, attr[1])]

                for key, value in target_attr.items():
                    if get_env_device() == "gpu":
                        target_attr[key] = getattr(value, action)()
                    elif get_env_device() == "xpu" and action in ["cpu", "pin_memory"]:
                        target_attr[key] = getattr(value, action)()
                    else:
                        target_attr[key] = getattr(value, "to")(action)

    def _offload_optimizer(self):
        if get_env_device() == "gpu":
            self._apply_to_optimizer("pin_memory")
        elif get_env_device() == "xpu":
            self._apply_to_optimizer("pin_memory")
        else:
            self._apply_to_optimizer("cpu")

    def _reload_optimizer(self):
        if get_env_device() == "gpu":
            self._apply_to_optimizer("cuda")
        else:
            self._apply_to_optimizer(get_env_device())

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        rng_file = os.path.join(checkpoint, f"rng_state_{dist.get_rank()}.pth")
        if not os.path.isfile(rng_file):
            logger.info(
                "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                "fashion, reproducibility is not guaranteed."
            )
            return

        checkpoint_rng_state = paddle.load(rng_file, return_numpy=True)
        if checkpoint_rng_state.get("world_size", None) != self.args.world_size:
            logger.warning("Cannot load rng states when changing world size of training job.")
            return

        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])

        core.default_cpu_generator().set_state(checkpoint_rng_state["cpu"])
        if core.is_compiled_with_cuda():
            if not len(checkpoint_rng_state["cuda"]) == core.get_cuda_device_count():
                raise ValueError("Length of gpu state list should be equal to the gpu device count")
            for i in range(core.get_cuda_device_count()):
                core.default_cuda_generator(i).set_state(checkpoint_rng_state["cuda"][i])

        if core.is_compiled_with_xpu():
            if not len(checkpoint_rng_state["cuda"]) == core.get_xpu_device_count():
                raise ValueError("Length of xpu state list should be equal to the xpu device count")
            for i in range(core.get_xpu_device_count()):
                core.default_xpu_generator(i).set_state(checkpoint_rng_state["cuda"][i])

        if paddle.device.get_all_custom_device_type() is not None:
            custom_device_type = paddle.device.get_all_custom_device_type()
            for device in custom_device_type:
                if not len(checkpoint_rng_state["cuda"]) == core.get_custom_device_count(device):
                    raise ValueError("Length of custom device state list should be equal to the custom device count")
                for i in range(core.get_custom_device_count(device)):
                    core.default_custom_device_generator(paddle.CustomPlace(device, i)).set_state(
                        checkpoint_rng_state["cuda"][i]
                    )

        if self.args.use_hybrid_parallel:
            if "hybrid_parallel_rng_state_tracker" in checkpoint_rng_state:
                if self.args.tensor_model_parallel_size <= 1:
                    checkpoint_rng_state["hybrid_parallel_rng_state_tracker"].pop("model_parallel_rng", None)
                try:
                    fleet.meta_parallel.get_rng_state_tracker().set_states_tracker(
                        checkpoint_rng_state["hybrid_parallel_rng_state_tracker"]
                    )
                except:
                    logger.warning(
                        "Hybrid parallel rng states change when training environment differs, so we dot not set state tracker here."
                    )
            else:
                logger.warning("Not found hybrid parallel RNG state.")

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`paddleformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """
        # optimizer_kwargs = {"lr": args.learning_rate}
        optimizer_kwargs = {}
        adam_kwargs = {
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
            "epsilon": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAMW:
            from paddle.optimizer import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.ADAMW_MINI:
            from ..utils import AdamWMini

            optimizer_cls = AdamWMini
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.ADAMW_CUSTOM:
            from ..utils import AdamWCustom

            optimizer_cls = AdamWCustom
            optimizer_kwargs.update(adam_kwargs)
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")

        return optimizer_cls, optimizer_kwargs

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        warmup = (
            self.args.warmup_steps if self.args.warmup_steps > 0 else int(self.args.warmup_ratio * num_training_steps)
        )
        decay_steps = num_training_steps
        if getattr(self.args, "decay_steps", None) and self.args.decay_steps > 0:
            decay_steps = self.args.decay_steps

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                learning_rate=self.args.learning_rate,
                num_warmup_steps=warmup,
                num_training_steps=decay_steps,
                num_cycles=self.args.num_cycles,
                lr_end=self.args.lr_end,
                power=self.args.power,
                min_lr=self.args.min_lr,
            )

        return self.lr_scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~paddle.io.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def _decorate_exclude_layers(self, model: nn.Layer):
        """
        Exclude layers from the model for paddle.amp.decorate.
        Args:
            model (`nn.Layer`): The model to exclude layers from.
        Returns:
            A list of excluded layers.
        """
        exclude_layers = []
        return exclude_layers

    def _wrap_distributed_optimizer(self, optimizer):
        """
        In hybrid expert parallel, use customized optimizer and grad clip
        """
        if (
            self.args.use_expert_parallel
            and self.args.moe_sharding_parallel_size >= 1
            and self.args.expert_model_parallel_size > 1
            and self.args.sharding_parallel_size > 1
        ):
            from ..utils import MoEHybridParallelOptimizer

            fleet_env = fleet.fleet
            fleet_env.user_defined_optimizer = optimizer
            hp_optim = MoEHybridParallelOptimizer(optimizer, fleet_env._hcg, fleet_env._user_defined_strategy)

            if fleet_env._user_defined_strategy.hybrid_configs["pp_configs"].dp_comm_overlap:
                hp_optim._dp_enable = False

            if fleet_env._user_defined_strategy.hybrid_configs["pp_configs"].sharding_comm_overlap:
                hp_optim._sharding_enable = False
            return hp_optim
        else:
            return fleet.distributed_optimizer(optimizer)

    def _wrap_model(self, model, training=True):
        if self.args.enable_auto_parallel:
            model = parallelize.parallelize_model(
                model,
                config=self.auto_dist_config,
            )
            for param in self.model.parameters():
                if not param._is_initialized() and param._init_func is not None:
                    param.initialize()

            return model

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Note: in paddle.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Mixed precision training
        if training and self.do_grad_scaling:  # self.args.fp16_opt_level=="O2":
            # model, self.optimizer
            decorated = paddle.amp.decorate(
                models=model,
                optimizers=self.optimizer,
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
                excluded_layers=[
                    QuantizationLinear,
                    ColumnParallelQuantizationLinear,
                    RowParallelQuantizationLinear,
                    QuantizationLoRABaseLinear,
                ]
                + self._decorate_exclude_layers(model),
            )

            if self.optimizer is None:
                model = decorated
            else:
                model, self.optimizer = decorated

        if self.args.tensor_model_parallel_size > 1 and self.args.sequence_parallel:
            # use callback for sp grad sync in case of unexpected behaviour (except sharding stage 2&3)
            if ShardingOption.SHARD_GRAD_OP in self.args.sharding or ShardingOption.FULL_SHARD in self.args.sharding:
                # stage 2 or stage 3
                register_sequence_parallel_allreduce_hooks(
                    model, self.args.gradient_accumulation_steps, self.args.fuse_sequence_parallel_allreduce
                )
            else:
                # stage 1 or dp
                self.add_callback(SPGradSyncCallback(model))

        if self.args.world_size == 1:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)
                assert self.optimizer is not None, "optimizer is empty!"
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)

        if isinstance(model, LoRAModel):
            model = model.model
        if (
            is_paddlefleet_available()
            and PaddleFleetPipelineLayer is not None
            and isinstance(model, PaddleFleetPipelineLayer)
        ):
            in_pipeline_parallel_mode = True
        else:
            in_pipeline_parallel_mode = self.args.pipeline_model_parallel_size > 1
        in_sharding_parallel_mode = self.sharding is not None
        in_tensor_parallel_mode = self.args.tensor_model_parallel_size > 1
        in_sep_parallel_mode = self.args.sep_parallel_size > 1
        in_cp_parallel_mode = self.args.context_parallel_size > 1

        # Multi-gpu training
        if self.args.world_size > 1 and (not self.args.use_hybrid_parallel):
            # MOE use DDP to broadcaset parameters.
            ddp_kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                ddp_kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PretrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                ddp_kwargs["find_unused_parameters"] = not any(
                    hasattr(m, "enable_recompute") and m.enable_recompute for m in model.sublayers(include_self=True)
                )
            else:
                ddp_kwargs["find_unused_parameters"] = True
            model = paddle.DataParallel(model, **ddp_kwargs)
            # Distributed training (should be after fp16 initialization)

            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)
                assert self.optimizer is not None, "optimizer is empty!"
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)

        # Pipeline mode
        if in_pipeline_parallel_mode:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
            # hack for pipeline model mini batch to batch
            # need batter solution @ZHUI
            # make batch_fn compatible for fleet.distributed_model decorate.
            prepare_pipeline_inputs_func = (
                model._prepare_pipeline_inputs_func if hasattr(model, "_prepare_pipeline_inputs_func") else None
            )

            if (
                is_paddlefleet_available()
                and paddlefleet_dist_model is not None
                and PaddleFleetPipelineLayer is not None
                and isinstance(model, PaddleFleetPipelineLayer)
            ):
                model = paddlefleet_dist_model.distributed_model(model)
            else:
                model = fleet.distributed_model(model)
            if prepare_pipeline_inputs_func is not None:
                model._prepare_pipeline_inputs_func = prepare_pipeline_inputs_func
            else:

                def _prepare_pipeline_inputs_func(inputs):
                    first_stage_keys = ["input_ids", "attention_mask", "position_ids"]
                    last_stage_keys = ["labels"]

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
                    if (
                        is_paddlefleet_available()
                        and PaddleFleetParallelBase is not None
                        and isinstance(model, PaddleFleetParallelBase)
                    ):
                        first_stage_inputs_batch = inputs_batch
                        last_stage_inputs = first_stage_inputs_batch.pop("labels")
                        outputs = (
                            first_stage_inputs_batch,
                            last_stage_inputs,
                        )
                        return outputs
                    return [
                        get_expected_keys(inputs_batch, first_stage_keys),
                        get_expected_keys(inputs_batch, last_stage_keys),
                    ]

                logger.warning(
                    "Using default prepare pipeline inputs func, only support input_ids and labels as inputs."
                )
                model._prepare_pipeline_inputs_func = _prepare_pipeline_inputs_func

            assert self.optimizer is not None, "Pipeline mode need decorate optimizer, pelease init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
            self.optimizer = self._wrap_distributed_optimizer(self.optimizer)

            if (
                hasattr(self.args, "enable_sharding_comm_overlap")
                and self.args.enable_sharding_comm_overlap
                and (
                    self.args.save_checkpoint_format == "unified_checkpoint"
                    or self.args.load_checkpoint_format == "unified_checkpoint"
                )
                and self.args.split_param
            ):
                model.register_sharding_comm_overlap_hook(self.optimizer)

        # No pipeline mode, sharding only
        if not in_pipeline_parallel_mode and in_sharding_parallel_mode:
            # Sharded DDP!
            if self.args.tensor_model_parallel_size > 1:
                hcg = fleet.get_hybrid_communicate_group()
                assert (
                    ShardingOption.SHARD_GRAD_OP in self.args.sharding or ShardingOption.SHARD_OP in self.args.sharding
                ), "Only support tensor parallel + sharding stage1/stage2 hybrid parallel now."
                # NOTE: TensorParallel will be called in distributed_model when sharding stage1, so no need to call here
                if ShardingOption.SHARD_GRAD_OP in self.args.sharding:
                    model = paddle.distributed.fleet.meta_parallel.TensorParallel(
                        model, hcg, strategy=fleet.fleet._user_defined_strategy
                    )

            if ShardingOption.SHARD_OP in self.args.sharding:
                if self.args.amp_master_grad:
                    mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
                model = fleet.distributed_model(model)

                if self.args.amp_master_grad:
                    self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
                self.optimizer = self._wrap_distributed_optimizer(self.optimizer)
            else:
                cpu_offload = ShardingOption.OFFLOAD in self.args.sharding
                assert self.optimizer is not None, "optimizer is empty!"
                level = None
                if ShardingOption.SHARD_GRAD_OP in self.args.sharding:
                    level = "os_g"
                if ShardingOption.FULL_SHARD in self.args.sharding:
                    level = "p_g_os"

                from paddle.distributed.sharding import group_sharded_parallel

                # add dp_group and exclude_layer params
                # https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/sharding/group_sharded_parallel_cn.html#group-sharded-parallel
                extra_kwargs = {}
                extra_kwargs["dp_group"] = self.dp_group
                extra_kwargs["exclude_layer"] = ["GroupNorm"]

                if self.args.amp_master_grad:
                    assert (
                        self.args.data_parallel_size == 1
                    ), "Sharding stage 2 / Sharding stage 3 main grad is not compatible with dp for now."
                    mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
                    self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)

                model, optimizer, _ = group_sharded_parallel(
                    model,
                    self.optimizer,
                    level=level,
                    scaler=None,
                    group=self.sharding_group,
                    offload=cpu_offload,
                    **extra_kwargs,
                )
                if ShardingOption.SHARD_GRAD_OP in self.args.sharding and self.args.amp_master_grad:
                    assert hasattr(optimizer, "use_main_grad"), (
                        "Current installed paddle doesn't support sharding stage 2 with main grad, "
                        "please upgrade your paddle (using nightly version)."
                    )

                if level == "os_g" and self.args.stage2_overlap:
                    model._set_reduce_overlap(True)
                    optimizer._set_broadcast_overlap(True, model)

                self.optimizer = optimizer
        # pure tesnor parallel mode, no pipeline_parallel, no sharding.
        if (
            not in_pipeline_parallel_mode
            and not in_sharding_parallel_mode
            and (in_tensor_parallel_mode or in_sep_parallel_mode or in_cp_parallel_mode)
        ):
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use

            model = fleet.distributed_model(model)
            assert self.optimizer is not None, "Tensor parallel mode need decorate optimizer, pelease init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
            self.optimizer = self._wrap_distributed_optimizer(self.optimizer)

        # stage1 has v1 and v2 version
        if in_sharding_parallel_mode and ShardingOption.SHARD_OP in self.args.sharding:
            if self.args.split_param:
                if hasattr(self.optimizer, "_set_all_gather_overlap_forward") and self.args.stage1_allgather_overlap:
                    self.optimizer._set_all_gather_overlap_forward(True, model)
            else:
                if hasattr(self.optimizer, "_set_broadcast_overlap") and self.args.stage1_broadcast_overlap:
                    self.optimizer._set_broadcast_overlap(True, model)

        # To solve DPO pin-memory problem, temporarily modify the _insert_sync method.
        self.optimizer._insert_sync = types.MethodType(_insert_sync, self.optimizer)

        return model

    def _prepare_input(self, data: Union[paddle.Tensor, Any]) -> Union[paddle.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, paddle.Tensor):
            # kwargs = dict(device=self.args.current_device)
            # update data type for pure fp16
            if data.place.is_cuda_pinned_place():
                return data.cuda()
            elif data.place.is_xpu_pinned_place():
                return data.to(paddle.device.get_device())
            return data
            # return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> Dict[str, Union[paddle.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def autocast_smart_context_manager(self):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        if self.enable_autocast_context_manager:
            custom_black_list = ["reduce_sum", "c_softmax_with_cross_entropy"]
            custom_white_list = []
            if self.args.fp16_opt_level == "O2":
                # https://github.com/PaddlePaddle/Paddle/blob/eb97f4f0adca40b16a309b927e480178beb8ae96/python/paddle/amp/amp_lists.py#L85-L86
                # the lookup_table is in black_list, but in O2, we need it return fp16
                custom_white_list.extend(["lookup_table", "lookup_table_v2"])

            if self.args.amp_custom_white_list is not None:
                custom_white_list.extend(self.args.amp_custom_white_list)
            if self.args.amp_custom_black_list is not None:
                custom_black_list.extend(self.args.amp_custom_black_list)

            ctx_manager = autocast(
                True,
                custom_black_list=set(custom_black_list),
                custom_white_list=set(custom_white_list),
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
            )
        else:
            ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

        return ctx_manager

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
            if self.args.enable_auto_parallel:

                def to_list(value):
                    if value is None:
                        return value
                    if isinstance(value, (list, tuple)):
                        return list(value)
                    return [value]

                criterion_inputs = to_list(outputs)
                criterion_labels = to_list(labels)
                loss = self.criterion(*(criterion_inputs + criterion_labels))
            else:
                loss = self.criterion(outputs, labels)
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

    def _enable_delay_scale_loss(self):
        if in_auto_parallel_align_mode():
            return True

        if self.args.pipeline_model_parallel_size > 1:
            return self.args.pp_delay_scale_loss
        elif self.args.tensor_model_parallel_size > 1:
            return self.args.tp_delay_scale_loss
        else:
            return False

    def training_step(
        self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]], step_control=0
    ) -> paddle.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to train.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `paddle.Tensor`: The tensor with training loss on this batch.
        """
        if (
            is_paddlefleet_available()
            and PaddleFleetParallelBase is not None
            and isinstance(model, PaddleFleetParallelBase)
        ):
            return self.training_pipeline_step(model, inputs)

        if self.args.pipeline_model_parallel_size > 1:
            return self.training_pipeline_step(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1 and not self._enable_delay_scale_loss():
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if not self.args.enable_auto_parallel:
            return loss.detach()

        if isinstance(loss, paddle.Tensor):
            return loss.detach() if loss._is_initialized() else float(0.0)
        elif isinstance(loss, np.ndarray):
            return np.sum(loss)
        elif loss is None:
            return float(0.0)
        else:
            return float(loss)

    def training_pipeline_step(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to train.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `paddle.Tensor`: The tensor with training loss on this batch.
        """
        # accumulation data
        if not hasattr(self, "_pp_data_buffer"):
            self._pp_data_buffer = []
        self._pp_data_buffer.append(inputs)
        if len(self._pp_data_buffer) != self.args.gradient_accumulation_steps:
            return paddle.zeros([])

        model.train()
        if model._dp_comm_overlap or model._sharding_comm_overlap:
            for _, buffers in model._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer._acc_steps = self.args.gradient_accumulation_steps

        model.optimizer = None  # we do not use `PipelineParallel` to handler optimizer step
        model.lr_scheduler = None

        def _dataset_process_function():
            # Pass a local function to forward_backward_pipeline instead of the dataset itself.
            # This prevents the dataset from being passed as a direct argument to forward_backward_pipeline,
            # which would create additional reference counts that cannot be cleared, leading to GPU memory leaks.
            with self.autocast_smart_context_manager():
                inputs = model._prepare_pipeline_inputs_func(self._pp_data_buffer)
            self._pp_data_buffer = []

            return model._prepare_training(
                inputs, self.optimizer, self.lr_scheduler
            )  # None, None => [optimizer, lr_scheduler]

        if PipelineDatasetPreprocessor is None or self.args.use_dualpipev:
            inputs = _dataset_process_function()
        else:
            inputs = PipelineDatasetPreprocessor(_dataset_process_function)

        with self.autocast_smart_context_manager():
            loss = model.forward_backward_pipeline(inputs, self.scaler if self.do_grad_scaling else None)

        return loss.detach()

    def save_model(
        self,
        output_dir: Optional[str] = None,
        merge_tensor_parallel: Optional[bool] = False,
        last_fc_to_hf: Optional[bool] = False,
    ):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if PREFIX_CHECKPOINT_DIR in os.path.split(output_dir)[-1]:
            signal_dir = os.path.join(self.args.output_signal_dir, os.path.split(output_dir)[-1])
        else:
            signal_dir = self.args.output_signal_dir

        if ShardingOption.FULL_SHARD in self.args.sharding:
            self.model_wrapped.get_all_parameters(convert2cpu=True, with_freeze_param=True)

        if self.args.should_save_model_state:
            self._save(output_dir=output_dir, merge_tensor_parallel=merge_tensor_parallel, last_fc_to_hf=last_fc_to_hf)
        else:
            if (
                self.args.save_checkpoint_format == "unified_checkpoint"
                and "async_save" in self.args.unified_checkpoint_config
            ):
                os.makedirs(signal_dir, exist_ok=True)
                if self.is_in_train:
                    global_rank = paddle.distributed.get_rank() if paddle.distributed.get_world_size() > 1 else -1
                    paddle.save(global_rank, os.path.join(signal_dir, f".model_weight.done.{global_rank}"))

        if strtobool(os.getenv("FLAG_LLM_PDC", "False")):
            # save model_done file to ensure model is complete
            if (
                self.args.should_save_model_state
                and self.args.should_save
                and not ("async_save" in self.args.unified_checkpoint_config)
            ):
                # For ckpt integrity
                paddle.save(self.state.global_step, os.path.join(output_dir, ".model_done"))
        if (
            self.args.save_checkpoint_format == "unified_checkpoint"
            and "async_save" in self.args.unified_checkpoint_config
            and not self.is_in_train
        ):
            os.makedirs(signal_dir, exist_ok=True)
            global_rank = paddle.distributed.get_rank() if paddle.distributed.get_world_size() > 1 else -1
            paddle.save(self.state.global_step, os.path.join(signal_dir, f".model_weight.done.{global_rank}"))

    def _filter_moe_no_sync_optimizer_params(self):
        """
        filter optimizer params which should not sync
        """
        state_dict = self.model.state_dict()
        optimzier_state_dict = self.optimizer.state_dict()
        filter_optimzier_state_dict = OrderedDict()
        param_names_in_master_weights = list(optimzier_state_dict["master_weights"].keys()) if self.args.bf16 else []
        filter_optimzier_state_dict["master_weights"] = OrderedDict()
        for _, v in state_dict.items():
            if getattr(v, "no_sync", False):
                if v.name in param_names_in_master_weights:
                    filter_optimzier_state_dict["master_weights"][v.name] = optimzier_state_dict["master_weights"][
                        v.name
                    ]
                for op_k, op_v in optimzier_state_dict.items():
                    if op_k.startswith(v.name):
                        filter_optimzier_state_dict[op_k] = op_v
        return filter_optimzier_state_dict

    def _ordered_save(self, state_dict, save_path, signal_path=None):
        group_size = self.args.ordered_save_group_size
        hcg = fleet.get_hybrid_communicate_group()
        if hcg.get_sharding_parallel_world_size() > 1 or hcg.get_model_parallel_world_size() <= 1:
            return paddle.save(state_dict, save_path)

        mp_group = hcg.get_model_parallel_group()
        ranks = list(mp_group.ranks)
        n = len(ranks)

        group_num = (n + group_size - 1) // group_size
        groups = []
        for i in range(group_num):
            groups.append([ranks[j] for j in range(i, n, group_num)])

        for group in groups:
            if dist.get_rank() in group:
                paddle.save(state_dict, save_path)
            dist.barrier(mp_group)

        if signal_path is not None:
            with open(signal_path, mode="w+") as f:
                f.write("1")

    def _save_checkpoint(self, model, metrics=None):
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"
        if self.args.enable_zero_cost_checkpoint:
            return

        self.runtime_timer.start("checkpoint saving time")

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self.args.output_dir
        run_signal_dir = self.args.output_signal_dir

        output_dir = os.path.join(run_dir, checkpoint_folder)
        signal_dir = os.path.join(run_signal_dir, checkpoint_folder)

        if not self.args.enable_auto_parallel:
            if isinstance(self.model, LoRAModel) and (
                self.model.quantized or self.args.pipeline_model_parallel_size > 1
            ):
                self.save_model(output_dir)
            elif isinstance(self.model, LoRAModel):
                self.save_model(output_dir, True)
            else:
                self.save_model(output_dir)

        else:
            # Save model checkpoint
            if self.args.should_save or self.args.should_save_model_state:
                os.makedirs(output_dir, exist_ok=True)

            if self.args.should_save:
                logger.info(f"Saving checkpoinit files into {output_dir}")

                if self.args.should_save_model_state:
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

                    self._save(output_dir=output_dir, state_dict=state_dict)
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

        if self.args.save_rng_states:
            # Save RNG state in non-distributed training
            rng_states = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cuda": paddle.get_rng_state(),
                "cpu": paddle.framework.core.default_cpu_generator().get_state(),
                "world_size": self.args.world_size,
            }
            if not self.args.enable_auto_parallel:
                if self.args.use_hybrid_parallel:
                    rng_states[
                        "hybrid_parallel_rng_state_tracker"
                    ] = fleet.meta_parallel.get_rng_state_tracker().get_states_tracker()
                rng_state_file = os.path.join(output_dir, f"rng_state_{dist.get_rank()}.pth")
                os.makedirs(output_dir, exist_ok=True)
                paddle.save(rng_states, rng_state_file)
            else:
                rng_states_list = []
                paddle.distributed.all_gather_object(rng_states_list, rng_states)
                if self.args.should_save:
                    os.makedirs(output_dir, exist_ok=True)
                    paddle.save(rng_states_list, os.path.join(output_dir, f"rng_state_{self.args.world_size}.pth"))

        if not self.args.enable_auto_parallel:
            # only save model state dict, ignore optimizer and scheduler
            if not self.args.ignore_save_lr_and_optim:
                optimizer_name = _add_variant(PADDLE_OPTIMIZER_NAME, self.args.optimizer_name_suffix)
                saved_signal_path = os.path.join(output_dir, f"saved_signal_{dist.get_rank()}")

                if self.args.unified_checkpoint and self.args.offload_optim:
                    self._reload_optimizer()

                if self.args.use_hybrid_parallel:
                    if self.dp_group.rank <= 0 or self.args.use_expert_parallel:
                        os.makedirs(output_dir, exist_ok=True)
                        logger.info("Saving optimizer files.")
                        if self.args.save_checkpoint_format == "unified_checkpoint":
                            self.unified_checkpoint_handler.save_unified_optimizer(
                                self.model,
                                self.optimizer,
                                output_dir,
                                signal_dir,
                                self.args.optim_shard_num,
                            )
                        elif self.args.save_checkpoint_format == "flex_checkpoint":
                            self._save_flex_optimizer_state(output_dir)
                        else:
                            if self.dp_group.rank > 0:  # this should only work for MoE saving
                                self._save_ckpt_func(
                                    self._filter_moe_no_sync_optimizer_params(),
                                    os.path.join(output_dir, optimizer_name),
                                    saved_signal_path,
                                )
                            else:
                                state_dict = self.optimizer.state_dict()
                                save_path = os.path.join(output_dir, optimizer_name)
                                if self.args.use_async_save:
                                    assert not strtobool(
                                        os.getenv("FLAG_LLM_PDC", "False")
                                    ), "Dont support FLAG_LLM_PDC"
                                    self._async_optimizer_saver.run(
                                        state_dict, save_path, saved_signal_path=saved_signal_path
                                    )
                                else:
                                    self._save_ckpt_func(state_dict, save_path, saved_signal_path)
                    else:
                        if (
                            self.args.save_checkpoint_format == "unified_checkpoint"
                            and "async_save" in self.args.unified_checkpoint_config
                        ):
                            global_rank = (
                                paddle.distributed.get_rank() if paddle.distributed.get_world_size() > 1 else -1
                            )
                            os.makedirs(signal_dir, exist_ok=True)
                            paddle.save(global_rank, os.path.join(signal_dir, f".optimizer_weight.done.{global_rank}"))
                            if (
                                "skip_save_model_weight" not in self.args.unified_checkpoint_config
                                or "remove_master_weight" not in self.args.unified_checkpoint_config
                            ):
                                paddle.save(
                                    global_rank, os.path.join(signal_dir, f".master_weight.done.{global_rank}")
                                )
                if (
                    self.args.should_save
                    or self.args.use_expert_parallel
                    or (self.args.data_parallel_size > 1 and self.args.save_checkpoint_format == "flex_checkpoint")
                ):
                    if not self.args.use_hybrid_parallel:
                        logger.info("Saving optimizer files.")
                        if self.args.save_checkpoint_format == "unified_checkpoint":
                            self.unified_checkpoint_handler.save_unified_optimizer(
                                self.model,
                                self.optimizer,
                                output_dir,
                                signal_dir,
                            )
                        elif self.args.save_checkpoint_format == "flex_checkpoint":
                            self._save_flex_model_state(output_dir)
                            self._save_flex_optimizer_state(output_dir)
                        else:
                            if self.args.data_parallel_rank > 0 and self.args.use_expert_parallel:
                                self._save_ckpt_func(
                                    self._filter_moe_no_sync_optimizer_params(),
                                    os.path.join(output_dir, optimizer_name),
                                    saved_signal_path,
                                )
                            else:
                                self._save_ckpt_func(
                                    self.optimizer.state_dict(),
                                    os.path.join(output_dir, optimizer_name),
                                    saved_signal_path,
                                )
                    # FIXME: maybe only save one copy
                    paddle.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

                    if self.do_grad_scaling:
                        paddle.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
                else:
                    if self.args.save_checkpoint_format == "unified_checkpoint" and not self.args.use_hybrid_parallel:
                        if "async_save" in self.args.unified_checkpoint_config:
                            global_rank = (
                                paddle.distributed.get_rank() if paddle.distributed.get_world_size() > 1 else -1
                            )
                            os.makedirs(signal_dir, exist_ok=True)
                            paddle.save(global_rank, os.path.join(signal_dir, f".optimizer_weight.done.{global_rank}"))
                            if (
                                "skip_save_model_weight" not in self.args.unified_checkpoint_config
                                or "remove_master_weight" not in self.args.unified_checkpoint_config
                            ):
                                paddle.save(
                                    global_rank, os.path.join(signal_dir, f".master_weight.done.{global_rank}")
                                )

                if self.args.save_checkpoint_format == "unified_checkpoint" and (
                    self.args.offload_optim or self.args.tensorwise_offload_optimizer
                ):
                    self._offload_optimizer()
            self.runtime_timer.stop()

            # Maybe delete some older checkpoints.
            # For hybrid parallel training, the checkpoint files maybe on different node.
            need_to_rotate_checkpoints = False
            if self.args.use_hybrid_parallel:
                if self.dp_group.rank <= 0 or self.args.use_expert_parallel:
                    need_to_rotate_checkpoints = True
            else:
                need_to_rotate_checkpoints = self.args.should_save_model_state

            # Delete only by one process
            need_to_rotate_checkpoints = need_to_rotate_checkpoints and self.args.local_rank in [0, -1]
            if need_to_rotate_checkpoints:
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
                self._rotate_checkpoints(use_mtime=True, output_dir=run_signal_dir)

        if strtobool(os.getenv("FLAG_LLM_PDC", "False")) and not ("async_save" in self.args.unified_checkpoint_config):
            # save checkpoint_done file to ensure checkpoint is complete
            if self.args.should_save_model_state and self.args.should_save:
                # For ckpt integrity
                paddle.save(self.state.global_step, os.path.join(output_dir, ".checkpoint_done"))

    def set_optimizer_grouped_parameters(self, optimizer_grouped_parameters=None):
        """
        set optimizer grouped parameters:

        you can set optimizer_grouped_parameters with whatever arguments on whatever parameters to train.
        """
        self.optimizer_grouped_parameters = optimizer_grouped_parameters

    def disable_autocast_context_manager(self):
        """
        For pure fp16 or pure bf16 training, the paddle.amp.autocast is annoy for always cast fp32 to fp16.
        if you networks cast fp16 to fp32 manually to get higher precision, autocast make it not work, since it cast fp32 to fp16 back.

        """
        assert self.args.fp16_opt_level == "O2", "disable_autocast_context_manager should only work for pure fp16/bf16"
        self.enable_autocast_context_manager = False

    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            # ignore_errors for shared disks between train nodes.
            shutil.rmtree(checkpoint, ignore_errors=True)

    def _save(
        self,
        output_dir: Optional[str] = None,
        state_dict=None,
        merge_tensor_parallel=False,
        last_fc_to_hf=False,
    ):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # signal_dir is used for asynchronous saving situations.
        signal_dir = self.args.output_signal_dir
        if (
            self.args.save_checkpoint_format == "unified_checkpoint"
            and "async_save" in self.args.unified_checkpoint_config
        ):
            if PREFIX_CHECKPOINT_DIR in os.path.split(output_dir)[-1]:
                signal_dir = os.path.join(signal_dir, os.path.split(output_dir)[-1])
            os.makedirs(signal_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint finish signal to {signal_dir}")

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if (
            strtobool(os.getenv("FLAG_LLM_PDC", "False"))
            and paddle.distributed.get_rank() == 0
            and self.args.save_checkpoint_format == "unified_checkpoint"
            and "async_save" in self.args.unified_checkpoint_config
        ):
            world_size = paddle.distributed.get_world_size()
            save_info = {
                "world_size": world_size,
                "ignore_save_lr_and_optim": self.args.ignore_save_lr_and_optim,
                "skip_save_model_weight": "skip_save_model_weight" in self.args.unified_checkpoint_config,
                "remove_master_weight": "remove_master_weight" in self.args.unified_checkpoint_config,
            }
            if os.path.exists(
                os.path.join(self.args.output_signal_dir, "async_save_info.json")
            ):  # afs cannot overwrite
                os.remove(os.path.join(self.args.output_signal_dir, "async_save_info.json"))
            with open(os.path.join(self.args.output_signal_dir, "async_save_info.json"), "w") as f:
                json.dump(save_info, f)

        if self.args.enable_auto_parallel:
            if self.args.save_to_hf:
                is_main_process = paddle.distributed.get_rank() == 0
                self.model.save_pretrained(
                    output_dir,
                    variant=self.args.weight_name_suffix,
                    save_function=dist.save_state_dict,
                    merge_tensor_parallel=merge_tensor_parallel,
                    is_main_process=is_main_process,
                    max_shard_size="1024GB",
                    save_to_hf=True,
                    enable_auto_parallel=True,
                    save_checkpoint_format=self.args.save_checkpoint_format,
                )
            else:
                self._save_flex_model_state(output_dir)
                self._save_flex_optimizer_state(output_dir)
            return

        else:
            if self.args.should_save:
                if self.tokenizer is not None and self.args.save_tokenizer:
                    self.tokenizer.save_pretrained(output_dir)
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(output_dir)
                # Good practice: save your training arguments together with the trained model
                paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            if self.args.save_checkpoint_format == "unified_checkpoint":
                unified_checkpoint_config_backup = self.args.unified_checkpoint_config
                # backup and remove unified_checkpoint_config for not trine stage
                if not self.is_in_train:
                    self.args.unified_checkpoint_config = []
                self.unified_checkpoint_handler.save_unified_checkpoint(
                    self.model, self.optimizer, output_dir, signal_dir, save_to_hf=self.args.save_to_hf
                )

                # recover unified_checkpoint_config for not trine stage
                if not self.is_in_train:
                    self.args.unified_checkpoint_config = unified_checkpoint_config_backup

                return
            if self.args.save_checkpoint_format == "flex_checkpoint":
                if last_fc_to_hf:
                    is_main_process = paddle.distributed.get_rank() == 0
                    if isinstance(self.model, LoRAModel):
                        self.model.save_pretrained(
                            output_dir,
                            merge_tensor_parallel=merge_tensor_parallel,
                            variant=self.args.weight_name_suffix,
                            save_checkpoint_format=self.args.save_checkpoint_format,
                        )
                    else:
                        self.model.save_pretrained(
                            output_dir, is_main_process, save_checkpoint_format=self.args.save_checkpoint_format
                        )
                else:
                    self._save_flex_model_state(output_dir)

                return
            merge_tensor_parallel = merge_tensor_parallel and self.args.use_hybrid_parallel
            # peft model
            if isinstance(self.model, LoRAModel):
                self.model.save_pretrained(
                    output_dir,
                    variant=self.args.weight_name_suffix,
                    save_function=self._save_ckpt_func,
                    merge_tensor_parallel=merge_tensor_parallel,
                    is_main_process=self.args.should_save,
                    max_shard_size="1024GB",
                    save_to_hf=self.args.save_to_hf,
                    save_checkpoint_format=self.args.save_checkpoint_format,
                )
            # TODO: @ZHUI unify unwrap_model(self.model) and self.model
            elif not isinstance(self.model, PretrainedModel):
                if isinstance(unwrap_model(self.model), PretrainedModel):
                    if self.args.should_save_sharding_stage1_model:
                        config_to_save = None
                        self.sharding_io.set_optimizer(self.optimizer)
                        (
                            state_dict,
                            config_to_save,
                            weight_name_suffix,
                        ) = self.sharding_io.manipulate_state_dict_and_config(
                            unwrap_model(self.model), merge_tensor_parallel=merge_tensor_parallel
                        )
                        unwrap_model(self.model).save_pretrained(
                            output_dir,
                            state_dict=state_dict,
                            config_to_save=config_to_save,
                            merge_tensor_parallel=merge_tensor_parallel,
                            variant=weight_name_suffix,
                            save_function=self._save_ckpt_func,
                            is_main_process=self.args.should_save,
                            max_shard_size="1024GB",
                            save_to_hf=self.args.save_to_hf,
                            save_checkpoint_format=self.args.save_checkpoint_format,
                        )
                    else:
                        unwrap_model(self.model).save_pretrained(
                            output_dir,
                            merge_tensor_parallel=merge_tensor_parallel,
                            variant=self.args.weight_name_suffix,
                            save_function=self._save_ckpt_func,
                            is_main_process=self.args.should_save,
                            max_shard_size="1024GB",
                            save_to_hf=self.args.save_to_hf,
                            save_checkpoint_format=self.args.save_checkpoint_format,
                        )
                else:
                    logger.info("Trainer.model is not a `PretrainedModel`, only saving its state dict.")
                    if merge_tensor_parallel:
                        logger.warning(
                            "Trainer.model is not a `PretrainedModel`, not support for merge_tensor_parallel."
                        )
                    if state_dict is None:
                        state_dict = self.model.state_dict()

                    if self.args.should_save_sharding_stage1_model:
                        state_dict, _, _ = self.sharding_io.manipulate_state_dict_and_config(
                            unwrap_model(self.model), merge_tensor_parallel=False, state_dict=state_dict
                        )
                        variant = _add_variant(PADDLE_WEIGHTS_NAME, self.args.sharded_name_suffix())
                    else:
                        variant = _add_variant(PADDLE_WEIGHTS_NAME, self.args.weight_name_suffix)

                    self._save_ckpt_func(state_dict, os.path.join(output_dir, variant))
            else:
                if isinstance(self.model, PretrainedModel) and self.args.should_save_sharding_stage1_model:
                    config_to_save = None
                    state_dict, config_to_save, weight_name_suffix = self.sharding_io.manipulate_state_dict_and_config(
                        self.model, merge_tensor_parallel=merge_tensor_parallel
                    )
                    self.model.save_pretrained(
                        output_dir,
                        state_dict=state_dict,
                        config_to_save=config_to_save,
                        merge_tensor_parallel=merge_tensor_parallel,
                        variant=weight_name_suffix,
                        save_function=self._save_ckpt_func,
                        is_main_process=self.args.should_save,
                        max_shard_size="1024GB",
                        save_to_hf=self.args.save_to_hf,
                        save_checkpoint_format=self.args.save_checkpoint_format,
                    )
                else:
                    self.model.save_pretrained(
                        output_dir,
                        merge_tensor_parallel=merge_tensor_parallel,
                        variant=self.args.weight_name_suffix,
                        save_function=self._save_ckpt_func,
                        is_main_process=self.args.should_save,
                        max_shard_size="1024GB",
                        save_to_hf=self.args.save_to_hf,
                        save_checkpoint_format=self.args.save_checkpoint_format,
                    )
            if self.args.should_save_sharding_stage1_model:
                model_meta = self.sharding_io.gather_distributed_model_meta()
                if self.args.should_save:
                    path = os.path.join(output_dir, MODEL_META_NAME)
                    with open(path, "w") as f:
                        json.dump(model_meta, f)

    def _load_scheduler(self, checkpoint):
        if checkpoint is None:
            self.runtime_timer.stop()
            return

        if not self.args.ignore_load_lr_and_optim:
            if distributed_isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
                self.lr_scheduler.set_state_dict(
                    paddle.load(distributed_file(os.path.join(checkpoint, SCHEDULER_NAME)))
                )
            else:
                raise ValueError(f"scheduler-file not found, scheduler:{os.path.join(checkpoint, SCHEDULER_NAME)}")

            if self.do_grad_scaling and distributed_isfile(os.path.join(checkpoint, SCALER_NAME)):
                self.scaler.load_state_dict(
                    paddle.load(distributed_file(os.path.join(checkpoint, SCALER_NAME)), return_numpy=True)
                )

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        self.runtime_timer.start("checkpoint loading time")
        if checkpoint is None:
            self.runtime_timer.stop()
            return

        logger.info("Loading optimizer and scheduler...")
        if (not self.args.should_load_sharding_stage1_model) and self.args.ignore_load_lr_and_optim:
            self.runtime_timer.stop()
            return

        opt_state_dict = None
        if self.args.should_load_sharding_stage1_model:
            opt_state_dict = self.sharding_io.load_optimizer_state_with_reshard(
                checkpoint, PADDLE_OPTIMIZER_NAME, self.model_wrapped
            )
        else:
            use_unified_checkpoint = False
            if self.args.save_checkpoint_format == "unified_checkpoint":
                if self.is_unified_checkpoint(checkpoint):
                    use_unified_checkpoint = True
                else:
                    logger.info("Loading checkpoint, the next checkpoint will be saved as unified checkpoint")

            if not use_unified_checkpoint:
                if self.args.data_parallel_rank == 0 or self.args.use_expert_parallel:
                    optimizer_name = _add_variant(PADDLE_OPTIMIZER_NAME, self.args.optimizer_name_suffix)
                    path = os.path.join(checkpoint, optimizer_name)
                    if os.path.isfile(path):
                        opt_state_dict = paddle.load(path)
                else:
                    opt_state_dict = None
            else:
                model = self.model
                if (
                    hasattr(self.args, "enable_sharding_comm_overlap")
                    and self.args.enable_sharding_comm_overlap
                    and self.args.split_param
                ):
                    model = self.model_wrapped
                opt_state_dict = self.unified_checkpoint_handler.load_unified_optimizer(
                    model=model,
                    optimizer=self.optimizer,
                    resume_from_checkpoint=checkpoint,
                    offload=self.args.tensorwise_offload_optimizer,
                )

        if self.args.ignore_load_lr_and_optim and opt_state_dict:
            tmp = self.optimizer.state_dict()
            tmp["master_weights"] = opt_state_dict["master_weights"]
            opt_state_dict = tmp

        # broadcast optimizer state in dp group
        if self.args.local_rank != -1:
            dist.barrier()
        if self.args.use_expert_parallel:
            opt_state_dict = broadcast_moe_optimizer(
                opt_state_dict,
                model_state_dict=self.model.state_dict(),
                broadcast_dp=not self.args.should_load_sharding_stage1_model,
            )
        else:
            if not self.args.should_load_sharding_stage1_model:
                opt_state_dict = broadcast_dp_optimizer(opt_state_dict)

        if opt_state_dict is not None:
            # Load in optimizer and scheduler states
            self.optimizer.set_state_dict(opt_state_dict)
        else:
            optimizer_name = _add_variant(PADDLE_OPTIMIZER_NAME, self.args.optimizer_name_suffix)
            raise ValueError(f"optimizer-state-dict not found, opt: {os.path.join(checkpoint, optimizer_name)}.")
        gc.collect()
        empty_device_cache()

        self._load_scheduler(checkpoint)

        if self.args.offload_optim:
            logger.info("Offloading optimizer state...")
            self._offload_optimizer()

        self.runtime_timer.stop()

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """

        try:
            from paddle.distributed.fleet.utils.timer_helper import (
                get_timers as paddle_get_timers,
            )

            paddle_pipeline_timers = paddle_get_timers()
        except AssertionError:
            paddle_pipeline_timers = None
        kwargs.update(
            timer=self.timers, paddle_pipeline_timers=paddle_pipeline_timers, metrics_dumper=self.metrics_dumper
        )

        if self.state.epoch is not None:
            logs["progress_or_epoch"] = round(self.state.epoch, 4)

        if self.timers:
            logs.update(self.timers.info(self.timers.timers.keys()))

        mem_info = psutil.virtual_memory()
        logs["cpu_used_memory"] = round(mem_info.used / (1024**3), 2)
        logs["cpu_available_memory"] = round(mem_info.available / (1024**3), 2)

        self.state.log_history = []
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            max_eval_iters=self.args.max_evaluate_steps,
        )

        total_batch_size = self.args.eval_batch_size * self.args.dataset_world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        if self.args.pipeline_model_parallel_size > 1:
            from paddle.distributed.fleet.meta_parallel import PipelineLayer

            _prepare_pipeline_inputs_func = getattr(self.model_wrapped, "_prepare_pipeline_inputs_func", None)
            # Only accept wrapped model for pipeline_parallel mode
            if self.model is self.model_wrapped and isinstance(self.model_wrapped, PipelineLayer):
                # NOTE(gongenlei): when do_train=False, do_eval=True, we need to wrap model for pipeline
                self.model_wrapped = fleet.distributed_model(self.model_wrapped)
            if isinstance(self.model_wrapped, LoRAModel) and isinstance(self.model_wrapped.model, PipelineLayer):
                # NOTE(liuting): when do_train=False, do_eval=True, lora=True, we need to wrap model for pipeline
                self.model_wrapped = fleet.distributed_model(self.model_wrapped.model)
            model = self.model_wrapped
            if _prepare_pipeline_inputs_func is not None:
                model._prepare_pipeline_inputs_func = _prepare_pipeline_inputs_func
        elif is_paddlefleet_available() and isinstance(self.model, GPTModel):
            model = self.model_wrapped
        else:
            model = self.model

        if isinstance(dataloader, paddle.io.DataLoader):
            batch_size = dataloader.batch_sampler.batch_size
        elif isinstance(dataloader, _DataLoaderIterBase):
            # support for inner dataloader
            batch_size = dataloader._batch_sampler.batch_size
            # alias for inner dataloader
            dataloader.dataset = dataloader._dataset
        else:
            raise ValueError("Only support for paddle.io.DataLoader")

        num_samples = None
        if max_eval_iters > 0:
            # on eval limit steps
            num_samples = batch_size * self.args.dataset_world_size * max_eval_iters
            if isinstance(dataloader, _DataLoaderIterBase) and isinstance(
                dataloader._batch_sampler, NlpDistributedBatchSampler
            ):
                consumed_samples = (
                    ((self.state.global_step) // args.eval_steps)
                    * max_eval_iters
                    * args.per_device_eval_batch_size
                    * args.dataset_world_size
                )
                dataloader._batch_sampler.set_epoch(consumed_samples=consumed_samples)

        logger.info(f"***** Running {description} *****")

        if not self.args.distributed_dataloader or (
            self.args.distributed_dataloader and self.args.should_load_dataset
        ):
            if has_length(dataloader):
                logger.info(f"  Num examples = {self.num_examples(dataloader)}")
                if max_eval_iters > 0:
                    logger.info(f"  Total prediction steps = {max_eval_iters}")
                else:
                    logger.info(f"  Total prediction steps = {len(dataloader)}")
            else:
                logger.info("  Num examples: Unknown")
                if max_eval_iters > 0:
                    logger.info(f"  Total prediction steps = {max_eval_iters}")

            logger.info(f"  Pre device batch size = {batch_size}")
            logger.info(f"  Total Batch size = {batch_size * self.args.dataset_world_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        losses = []
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                # losses = self._nested_gather(loss.repeat(batch_size))
                losses = self._nested_gather(paddle.tile(loss, repeat_times=[batch_size, 1]))
                losses_host = losses if losses_host is None else paddle.cat((losses_host, losses), axis=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)

                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

            if max_eval_iters > 0 and step >= max_eval_iters - 1:
                break

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if num_samples is not None:
            pass
        elif has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[: num_samples * int(self.args.world_size / self.args.dataset_world_size)]
        if all_preds is not None:
            all_preds = nested_truncate(
                all_preds, num_samples * int(self.args.world_size / self.args.dataset_world_size)
            )
        if all_labels is not None:
            all_labels = nested_truncate(
                all_labels, num_samples * int(self.args.world_size / self.args.dataset_world_size)
            )

        model.train()

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            # all_labels maybe is a tuple when prediction_steps output label_mask
            batch_labels = all_labels[0] if isinstance(all_labels, (list, tuple)) else all_labels
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=batch_labels))
        else:
            metrics = {}

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)
        <Tip>
        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.
        </Tip>
        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            prediction_loss_only=True if self.compute_metrics is None else None,
            metric_key_prefix=metric_key_prefix,
            max_eval_iters=self.args.max_evaluate_steps,
        )
        total_batch_size = self.args.per_device_eval_batch_size * self.args.dataset_world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def prediction_pipeline_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        """
        prediction_step function for pipeline parallel mode.
        """
        if hasattr(model, "_prepare_pipeline_inputs_func"):
            inputs, labels = model._prepare_pipeline_inputs_func(inputs)
            has_labels = labels is not None
        else:
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)
            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None
            inputs = inputs.pop("input_ids")
        # train & eval share the same p2p_helper, so clear it before and after each step
        if hasattr(model, "_p2p_helper"):
            model._p2p_helper.clear_meta_cache()

        with paddle.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():

                    def _prepare_inputs_for_fleet(inputs):
                        if isinstance(inputs, (tuple, list)) and len(inputs) > 1:
                            inputs = {"input_ids": inputs[0], "position_ids": inputs[1]}
                        return inputs

                    if (
                        is_paddlefleet_available()
                        and PaddleFleetParallelBase is not None
                        and isinstance(model, PaddleFleetParallelBase)
                    ):
                        inputs = _prepare_inputs_for_fleet(inputs)
                    loss = model.eval_batch([inputs, labels], compute_loss=True)
                    # loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            else:
                raise ValueError("pipeline mode eval need label!")
        # train & eval share the same p2p_helper, so clear it before and after each step
        if hasattr(model, "_p2p_helper"):
            model._p2p_helper.clear_meta_cache()

        return (loss, None, labels)

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to evaluate.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        if self.args.pipeline_model_parallel_size > 1 or (
            is_paddlefleet_available()
            and PaddleFleetParallelBase is not None
            and isinstance(model, PaddleFleetParallelBase)
        ):
            # hack for pipeline mode
            inputs = self._prepare_inputs(inputs)
            return self.prediction_pipeline_step(model, inputs, prediction_loss_only, ignore_keys)

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with paddle.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if isinstance(logits, (list, tuple)) and len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        return self.args.process_index == 0

    def _nested_gather(self, tensors):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if self.args.enable_auto_parallel:
            with _exec_mode_guard("dynamic"):
                if isinstance(tensors, paddle.Tensor):
                    tr_loss = tensors._local_value() if tensors.is_dist() else tensors
                else:
                    tr_loss = paddle.to_tensor([tensors])
        else:
            tr_loss = tensors

        if tr_loss is None:
            return
        if self.args.local_rank != -1:
            tr_loss = distributed_concat(tr_loss)
        return tr_loss

    def _pad_across_processes(self, tensor, pad_index=-100):
        """
        Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
        they can safely be gathered.
        """
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(self._pad_across_processes(t, pad_index=pad_index) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: self._pad_across_processes(v, pad_index=pad_index) for k, v in tensor.items()})
        elif not isinstance(tensor, paddle.Tensor):
            raise TypeError(
                f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
            )

        if len(tensor.shape) < 2:
            return tensor
        # Gather all sizes
        size = paddle.to_tensor(tensor.shape)[None]
        sizes = self._nested_gather(size).cpu()

        max_size = max(s[1] for s in sizes)
        if tensor.shape[1] == max_size:
            return tensor

        # Then pad to the maximum size
        old_size = tensor.shape
        new_size = list(old_size)
        new_size[1] = max_size
        # new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        new_tensor = paddle.zeros(tuple(new_size), dtype=tensor.dtype) + pad_index
        new_tensor[:, : old_size[1]] = tensor
        return new_tensor

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns or self.model is None:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids", "labels", "start_positions", "end_positions"]

        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                f" you can safely ignore this message."
            )

        columns = [k for k in self._signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _is_iterable_dataset(self, dataset):
        return isinstance(dataset, paddle.io.IterableDataset)

    def _is_iterable_dataset_distributed(self, dataset):
        # For distributed dataloaer.
        is_iterable_dataset_tensor = paddle.to_tensor(self._is_iterable_dataset(dataset)).astype("int32").reshape([1])
        if dist.get_world_size() > 1:
            dist.all_reduce(is_iterable_dataset_tensor, op=dist.ReduceOp.MAX)
        if is_iterable_dataset_tensor.item() == 1:
            return True
        return False

    def print_config(self, args=None, key=""):
        """
        print config values
        """
        logger.debug("=" * 60)
        if args is None:
            args = self.args
            key = "Training"
        import paddleformers

        logger.debug("{:^40}".format("{} Configuration Arguments".format(key)))
        logger.debug("{:30}: {}".format("paddle commit id", paddle.version.commit))
        logger.debug("{:30}: {}".format("paddleformers commit id", paddleformers.version.commit))

        for a in dir(args):
            if a[:2] != "__":  # don't print double underscore methods
                v = getattr(args, a)
                if not isinstance(v, types.MethodType):
                    logger.debug("{:30}: {}".format(a, v))

        logger.debug("")

    def is_unified_checkpoint(self, resume_from_checkpoint, safe_serialization=True):
        is_unified_checkpoint_type = False
        if isinstance(self.model, LoRAModel):
            weights_index_name = (
                PADDLE_PEFT_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_PEFT_WEIGHTS_INDEX_NAME
            )
        else:
            weights_index_name = PADDLE_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME
        master_weights_index_name = (
            PADDLE_MASTER_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_MASTER_WEIGHTS_INDEX_NAME
        )
        weights_index_file = os.path.join(
            resume_from_checkpoint,
            weights_index_name,
        )
        master_weights_index_file = os.path.join(
            resume_from_checkpoint,
            master_weights_index_name,
        )

        if distributed_isfile(weights_index_file) or distributed_isfile(master_weights_index_file):
            is_unified_checkpoint_type = True

        return is_unified_checkpoint_type
