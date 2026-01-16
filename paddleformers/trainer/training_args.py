# Copyright 2020-present the HuggingFace Inc. team.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
#  https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py

import contextlib
import json
import math
import os
import sys
import types
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet, in_auto_parallel_align_mode

from ..utils.env import PREFIX_CHECKPOINT_DIR
from ..utils.import_utils import is_paddlefleet_available
from ..utils.log import logger
from ..utils.pdc_sdk import FLASH_DEVICE
from ..utils.tools import paddle_device
from .trainer_utils import (
    IntervalStrategy,
    OptimizerNames,
    SchedulerType,
    ShardingOption,
    init_nccl_config,
    split_parallel_config,
)

# Conditionally import paddlefleet modules
if paddle.device.is_compiled_with_cuda() and is_paddlefleet_available():
    from paddlefleet.parallel_state import get_tensor_model_parallel_group
    from paddlefleet.training import initialize_fleet
else:

    def get_tensor_model_parallel_group(*args, **kwargs):
        return None

    def initialize_fleet(*args, **kwargs):
        pass


__all__ = [
    "default_logdir",
    "TrainingArguments",
]


def default_logdir() -> str:
    """
    Same default
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop
    itself**.

    Using [`PdArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        output_dir (`str`):
            The output directory where the model predictions and checkpoints will be written.
        overwrite_output_dir (`bool`, *optional*, defaults to `False`):
            If `True`, overwrite the content of the output directory. Use this to continue training if `output_dir`
            points to a checkpoint directory.
        do_train (`bool`, *optional*, defaults to `False`):
            Whether to run training or not. This argument is not directly used by [`Trainer`], it's intended to be used
            by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/PaddlePaddle/PaddleFormers/tree/develop/examples) for more details.
        do_eval (`bool`, *optional*):
            Whether to run evaluation on the validation set or not. Will be set to `True` if `evaluation_strategy` is
            different from `"no"`. This argument is not directly used by [`Trainer`], it's intended to be used by your
            training/evaluation scripts instead. See the [example
            scripts](https://github.com/PaddlePaddle/PaddleFormers/tree/develop/examples) for more details.
        do_predict (`bool`, *optional*, defaults to `False`):
            Whether to run predictions on the test set or not. This argument is not directly used by [`Trainer`], it's
            intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/PaddlePaddle/PaddleFormers/tree/develop/examples) for more details.
        do_export (`bool`, *optional*, defaults to `False`):
            Whether to export inference model or not. This argument is not directly used by [`Trainer`], it's
            intended to be used by your training/evaluation scripts instead.
        evaluation_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"no"`):
            The evaluation strategy to adopt during training. Possible values are:

                - `"no"`: No evaluation is done during training.
                - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
                - `"epoch"`: Evaluation is done at the end of each epoch.

        prediction_loss_only (`bool`, *optional*, defaults to `False`):
            When performing evaluation and generating predictions, only returns the loss.
        per_device_train_batch_size (`int`, *optional*, defaults to 8):
            The batch size per GPU core/CPU for training.
        per_device_eval_batch_size (`int`, *optional*, defaults to 8):
            The batch size per GPU core/CPU for evaluation.
        gradient_accumulation_steps (`int`, *optional*, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

            <Tip warning={true}>

            When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging,
            evaluation, save will be conducted every `gradient_accumulation_steps * xxx_step` training examples.

            </Tip>

        eval_accumulation_steps (`int`, *optional*):
            Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. If
            left unset, the whole predictions are accumulated on GPU/TPU before being moved to the CPU (faster but
            requires more memory).
        learning_rate (`float`, *optional*, defaults to 5e-5):
            The initial learning rate for [`AdamW`] optimizer.
        weight_decay (`float`, *optional*, defaults to 0):
            The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`]
            optimizer.
        adam_beta1 (`float`, *optional*, defaults to 0.9):
            The beta1 hyperparameter for the [`AdamW`] optimizer.
        adam_beta2 (`float`, *optional*, defaults to 0.999):
            The beta2 hyperparameter for the [`AdamW`] optimizer.
        adam_epsilon (`float`, *optional*, defaults to 1e-8):
            The epsilon hyperparameter for the [`AdamW`] optimizer.
        max_grad_norm (`float`, *optional*, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(`float`, *optional*, defaults to 1.0):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents of
            the last epoch before stopping training).
        max_steps (`int`, *optional*, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
            In case of using a finite iterable dataset the training may stop before reaching the set number of steps
            when all data is exhausted
        lr_scheduler_type (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
            The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
        warmup_ratio (`float`, *optional*, defaults to 0.0):
            Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
        warmup_steps (`int`, *optional*, defaults to 0):
            Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine scheduler.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR used in the polynomial scheduler.
        power (`float`, *optional*, defaults to 1.0):
            The power factor used in the polynomial scheduler.
        min_lr (`float`, *optional*, defaults to 0.0):
            The minimum learning rate used in the cosine scheduler.

        log_on_each_node (`bool`, *optional*, defaults to `True`):
            In multinode distributed training, whether to log using `log_level` once per node, or only on the main
            node.
        logging_dir (`str`, *optional*):
            log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
        logging_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The logging strategy to adopt during training. Possible values are:

                - `"no"`: No logging is done during training.
                - `"epoch"`: Logging is done at the end of each epoch.
                - `"steps"`: Logging is done every `logging_steps`.

        logging_first_step (`bool`, *optional*, defaults to `False`):
            Whether to log and evaluate the first `global_step` or not.
        logging_steps (`int`, *optional*, defaults to 500):
            Number of update steps between two logs if `logging_strategy="steps"`.
        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: No save is done during training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`.
        save_steps (`int`, *optional*, defaults to 500):
            Number of updates steps before two checkpoint saves if `save_strategy="steps"`.
        save_total_limit (`int`, *optional*):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            `output_dir`.
        save_on_each_node (`bool`, *optional*, defaults to `False`):
            When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on
            the main one.

            This should not be activated when the different nodes use the same storage as the files will be saved with
            the same names for each node.
        no_cuda (`bool`, *optional*, defaults to `False`):
            Whether to not use CUDA even when it is available or not.
        seed (`int`, *optional*, defaults to 42):
            Random seed that will be set at the beginning of training. To ensure reproducibility across runs, use the
            [`~Trainer.model_init`] function to instantiate the model if it has some randomly initialized parameters.
        fp16 (`bool`, *optional*, defaults to `False`):
            Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        fp16_opt_level (`str`, *optional*, defaults to 'O1'):
            For `fp16` training,  AMP optimization level selected in ['O0', 'O1', 'O2']. See details at
            https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/auto_cast_cn.html
        amp_custom_black_list (`List[str]`, *optional*, defaults to ["reduce_sum", "softmax_with_cross_entropy", "c_softmax_with_cross_entropy", "elementwise_div", "sin", "cos"]):
            The custom black_list. The set of ops that support fp16/bf16 calculation and are considered numerically-dangerous
            and whose effects may also be observed in downstream ops. These ops will not be converted to fp16/bf16.
        amp_custom_white_list (`List[str]`, *optional*, defaults to ["lookup_table", "lookup_table_v2", "flash_attn", "matmul", "matmul_v2", "fused_gemm_epilogue"]):
            The custom white_list. It’s the set of ops that support fp16/bf16 calculation and are considered numerically-safe and
             performance-critical. These ops will be converted to fp16/bf16.
        amp_master_grad (`bool`, *optional*, defaults to `True`):
            For amp opt level=’O2’, whether to use float32 weight gradients
            for calculations such as gradient clipping, weight decay, and weight updates. If master_grad is enabled,
            the weight gradients will be float32 dtype after the backpropagation. Default is False, there is only float16 weight gradients.
            Note: only support model parallel and pipeline parallel for now !!!
        sharding (`str`, *optional*, defaults to `stage1`):
            Whether or not to use Paddle Sharding Data Parallel training (in distributed training
            only). The base option should be `stage1`, `stage2` or `stage3` and you can add
            CPU-offload to `stage2` or `stage3` like this: `stage2 offload` or `stage3 offload`.
            Each stage means:
                stage1 : optimizer state segmentation
                stage2 : optimizer state + gradient segmentation
                stage3 : parameter + gradient + optimizer state segmentation
                offload : offload parameters to cpu
        sharding_parallel_size (`int`, *optional*, defaults to `-1`)
            Sharding parameter in certain cards group. For example, aussume we use 2 machines each with 8 cards,
            then set sharding_parallel_size=8, sharding will only communication inside machine.
            default -1 means sharding parameters between all workers.
        sharding_parallel_mesh_dimension (`str`, *optional*, defaults to `dp`)
            Specifies the name of the dimension in a multi-dimensional parallelism mesh that is responsible for sharding.
            default `dp` for default parallelism mesh.
        tensor_model_parallel_size (`int`, *optional*, defaults to `-1`)
            Tensor parallelism is parallel technique proposed in (https://arxiv.org/pdf/2104.04473.pdf see 2.3 Tensor Model Parallelism).
            This technique splits one transformer layer into multi-cards (For examples, tensor_model_parallel_size=4, will split a layer to 4-parts)
            tensor_model_parallel_size means split the transformer layer to how many parts.
            default -1 for not use tensor parallel,  Suggest tensor_model_parallel_size<=8 for better proformance.
            Note, this need model support in source code, currently GPT/BLOOM/LLAMA/BLOOM/CLM/CHATGLM is supported.
        pipeline_model_parallel_size (`int`, *optional*, defaults to `-1`)
            Pipeline parallelism is parallel technique proposed in (https://arxiv.org/pdf/2104.04473.pdf see 2.2 Pipeline Model Parallelism).
            Pipeline parallelism assigns multi-transformer layers to different cards, the micro batch data stream passed between cards like pipelines.
            pipeline_model_parallel_size means split all transformer layers to how many stages.
            default -1 for not use pipeline parallel.
            Note. this need model support in source code, see llama modeling_pp.py file
        sep_parallel_size (`int`, *optional*, defaults to `-1`):
            The paddle sequence parallel strategy. It can reduce the GPU memory of activation to 1/sep, and it is orthogonal to
            data parallel, sharding stage1, tensor parallel and pipeline parallel strategy.
        )
        context_parallel_size (`int`, *optional*, defaults to `-1`):
            Context parallelism is a parallel method that segments training data in the sequence dimension.
            This method uses Ring FlashAttention to ensure the correctness of the Attention result after segmentation. The complete attention score is obtained through ring communication and iterative updates.
        )
        data_parallel_config (`str`, *optional*) **[Deprecated]** :
            Some additional configs which affect data parallel performance, we provide some option to config it.
            following config is support:
              enable_allreduce_avg_in_gradinent_scale, it replace `allreduce_sum + scale` pattern with `allreduce_avg` when scale gradient in data_parallel, which improve the performance. ONLY supported for auto mode now.
              gradient_sync_after_accumulate, move gradient sync operations from backward into optimizer step when gradient accumulate enabling, which reduce the sync times to improve performance, but will increase the memory usage. ONLY supported for auto mode now.
        tensor_parallel_config (`str`, *optional*) **[Deprecated]** :
            Some additional configs which affect model parallel performance, we provide some option to config it.
            following config is support:
              enable_mp_async_allreduce, it supports all_reduce(dx) overlap with matmul(dw) in ColumnParallelLinear backward when it set True, which can accelerate model parallel performance.
              enable_mp_skip_c_identity, it supports skip c_identity in ColumnParallelLinear and RowParallelLinear. It only works when set mp_async_allreduce is True. It can accelerate model parallel further.
              enable_mp_fused_linear_param_grad_add, it supports fused_linear_param_grad_add in ColumnParallelLinear (cuda >= 11.6). It only works when mp_async_allreduce is true. It can accelerate model parallel further.
              enable_sp_async_reduce_scatter, it supports async reduce_scatter in ColumnSequenceParallelLinear. It only works when set sp_async_reduce_scatter is True. It can accelerate sequence parallel further.
              enable_delay_scale_loss, accumulate gradients until optimizer step, all gradients div by accumute step. instead of div accumute step on loss directly.
              sync_param, in optimizer step, use broadcast to sync parameters those attr 'is_distributed' is False.
              sync_grad, in optimizer step, use broadcast to sync gradients those attr 'is_distributed' is False.
              sync_moment, in optimizer step, use broadcast to sync momentums those attr 'is_distributed' is False.
              replace_with_c_embedding, it supports replacing col-sliced embedding with row-sliced c_embedding when it set True, which is used in PIR auto_parallel.
              replace_with_parallel_cross_entropy, it replaces 'cross_entropy_with_softmax' OP with 'c_softmax_with_cross_entropy' OP in PIR static graph, which can improve model parallel performance.
        pipeline_parallel_config (`str`, *optional*) **[Deprecated]** :
            Some additional config it highly affect the usage of pipeline parallel, we provide some option to config it.
            following config is support:
              disable_p2p_cache_shape, if you max sequence length is varying, please set disable_p2p_cache_shape.
              disable_partial_send_recv, optimize send speed for tensor parallel.
              enable_delay_scale_loss, accumulate gradients until optimizer step, all gradients div by inner pipeline accumute step. instead of div accumute step on loss directly.
              enable_dp_comm_overlap, fuse data parallel gradient communication.
              enable_sharding_comm_overlap, fuse sharding stage 1 parallel gradient communication.
              enable_release_grads, reduce peak memory usage by releasing gradients after each iteration. The creation of gradients will be postponed until backward propagation of the next iteration.
              enable_overlap_p2p_comm, overlap p2p communication with computation.
              enable_clear_every_step_cache, clear every step cache for pipeline parallel.
              disable_non_batch_p2p_comm, disable batched send/recv in pipeline parallel mode.
              auto_parallel_sync_shared_params, optimize the parameter sharing between two stages in a pipeline parallel scenario.
        sharding_parallel_config (`str`, *optional*) **[Deprecated]** :
            Some additional config it highly affect the usage of sharding parallel, we provide some option to config it.
            following config is support:
              enable_stage1_tensor_fusion, fuse small tensors into big tensor chunks to accelerate communications, may increase memory occupation
              enable_tensor_fusion, fuse small tensors into big tensor chunks to accelerate communications, may increase memory occupation only used for semi auto mode.
              enable_stage1_overlap, fuse small tensors into big tensor chunks to accelerate communications and do communication overlap with backward computation, may harm the backward speed
              enable_overlap, fuse small tensors into big tensor chunks to accelerate communications and do communication overlap with backward computation, may harm the backward speed only used for semi auto mode.
              enable_stage2_overlap, overlap stage2 NCCL communication with computation. There are some constraints for the overlap, such as the logging_step should be bigger than 1 for broadcast overlap and no other sync could be called during the training for broadcast overlap.
              enable_stage1_broadcast_overlap, overlap stage1 V1 broadcast with next step forward computation. There are some constraints for the overlap, such as the logging_step should be bigger than 1 for broadcast overlap forward compute and no other sync could be called during the training for broadcast overlap.
              enable_stage1_allgather_overlap, overlap stage1 V2 allgather with next step forward computation. There are some constraints for the overlap, such as the logging_step should be bigger than 1 for allgather overlap forward compute and no other sync could be called during the training for allgather overlap.
              disable_stage1_reduce_avg, replace reduce_avg with original reduce_sum+scale in stage1, which can be used for accuracy verification.
              enable_release_grads, reduce peak memory usage by releasing gradients after each iteration. The creation of gradients will be postponed until backward propagation of the next iteration.
              enable_fuse_optimizer_states, fuse optimizer states to a single storage.
        scale_loss (`float`,  *optional*, defaults to 32768):
            The value of initial scale_loss for fp16. (default: 32768)
        local_rank (`int`, *optional*, defaults to -1):
            Rank of the process during distributed training.
        dataloader_drop_last (`bool`, *optional*, defaults to `False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (`int`, *optional*):
            Number of update steps between two evaluations if `evaluation_strategy="steps"`. Will default to the same
            value as `logging_steps` if not set.
        max_evaluate_steps (`int`, *optional*, defaults to -1):
            If set to a positive number, the total number of evaluation steps to perform.
        dataloader_num_workers (`int`, *optional*, defaults to 0):
            Number of subprocesses to use for data loading. 0 means that the data will be loaded in the
            main process.
        past_index (`int`, *optional*, defaults to -1):
            Some models like TransformerXL or XLNet can make use of the past hidden states for their predictions.
            If this argument is set to a positive int, the `Trainer` will use the corresponding output (usually index 2) as
            the past state and feed it to the model at the next training step under the keyword argument `mems`.
        run_name (`str`, *optional*):
            A descriptor for the run. Typically used for logging.
        disable_tqdm (`bool`, *optional*):
            Whether or not to disable the tqdm progress bars and table of metrics. Will default to `True` if the logging
            level is set to warn or lower (default), `False` otherwise.
        remove_unused_columns (`bool`, *optional*, defaults to `True`):
            If using `datasets.Dataset` datasets, whether or not to automatically remove the columns unused by the
            model forward method.
        label_names (`List[str]`, *optional*):
            The list of keys in your dictionary of inputs that correspond to the labels.
            Will eventually default to `["labels"]` except if the model used is one of the `XxxForQuestionAnswering` in
            which case it will default to `["start_positions", "end_positions"]`.
        load_best_model_at_end (`bool`, *optional*, defaults to `False`):
            Whether or not to load the best model found during training at the end of training.

            <Tip>

            When set to `True`, the parameters `save_strategy` needs to be the same as `eval_strategy`, and in the case
            it is "steps", `save_steps` must be a round multiple of `eval_steps`.

            </Tip>

        metric_for_best_model (`str`, *optional*):
            Use in conjunction with `load_best_model_at_end` to specify the metric to use to compare two different
            models. Must be the name of a metric returned by the evaluation with or without the prefix `"eval_"`. Will
            default to `"loss"` if unspecified and `load_best_model_at_end=True` (to use the evaluation loss).

            If you set this value, `greater_is_better` will default to `True`. Don't forget to set it to `False` if
            your metric is better when lower.
        greater_is_better (`bool`, *optional*):
            Use in conjunction with `load_best_model_at_end` and `metric_for_best_model` to specify if better models
            should have a greater metric or not. Will default to:

            - `True` if `metric_for_best_model` is set to a value that isn't `"loss"` or `"eval_loss"`.
            - `False` if `metric_for_best_model` is not set, or set to `"loss"` or `"eval_loss"`.
        ignore_data_skip (`bool`, *optional*, defaults to `False`):
            When resuming training, whether or not to skip the epochs and batches to get the data loading at the same
            stage as in the previous training. If set to `True`, the training will begin faster (as that skipping step
            can take a long time) but will not yield the same results as the interrupted training would have.
        optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw"`):
            The optimizer to use: adamw, or adafactor.
        length_column_name (`str`, *optional*, defaults to `"length"`):
            Column name for precomputed lengths. If the column exists, grouping by length will use these values rather
            than computing them on train startup. Ignored unless `group_by_length` is `True` and the dataset is an
            instance of `Dataset`.
        report_to (`str` or `List[str]`, *optional*, defaults to `"visualdl"`):
            The list of integrations to report the results and logs to.
            Supported platforms are `"visualdl"`/`"wandb"`/`"tensorboard"`/`"swanlab"`.
            `"none"` for no integrations.
        ddp_find_unused_parameters (`bool`, *optional*):
            When using distributed training, the value of the flag `find_unused_parameters` passed to
            `paddle.DataParallel`. Will default to `False` if recompute is used, `True` otherwise.
        wandb_api_key (`str`, *optional*):
            Weights & Biases (WandB) API key(s) for authentication with the WandB service.
        wandb_http_proxy (`str`, *optional*):
            Weights & Biases (WandB) http proxy for connecting with the WandB service.
        resume_from_checkpoint (`str`, *optional*):
            The path to a folder with a valid checkpoint for your model. This argument is not directly used by
            [`Trainer`], it's intended to be used by your training/evaluation scripts instead. See the [example
            scripts](https://github.com/PaddlePaddle/PaddleFormers/tree/develop/examples) for more details.
        auto_parallel_resume_form_hybrid_parallel (`bool`, *optional*):
            Whether hybrid parallel checkpoints be loaded in auto parallel mode.
        flatten_param_grads (`bool`, *optional*):
            Whether use flatten_param_grads method in optimizer, only used on NPU devices. Default is `False`.
        skip_profile_timer (`bool`, *optional*):
            Whether skip profile timer, timer will record time usage of forward/ backward/ step, etc.
        distributed_dataloader (`bool`, *optional*):
            Whether to use distributed dataloader. Default is `False`.
        release_grads (`bool`, *optional*):
            Whether to release gradients during training. Default is `False`.
        ckpt_quant_stage (`str`, *optional*):
            Whether activate checkpoint quantization. O0: deactivate, O1: Int8 compression, O2: Int4 compression. (default: O0).

        save_checkpoint_format (`str`, *optional*):
            Specifies the format for saving checkpoints. Options are: None, 'sharding_io', 'unified_checkpoint', 'flex_checkpoint'. (default: None). This setting is ignored if the corresponding switch is configured.
        load_checkpoint_format (`str`, *optional*):
            Specifies the format for loading checkpoints. Options are: None, 'sharding_io', 'unified_checkpoint', 'flex_checkpoint'. (default: None). This setting is ignored if the corresponding switch is configured.
        aoa_config (`Optional[dict[str, list[str]]]`, *optional*):
            The AoA configuration of FlexCheckpoint, used to describe the mapping between model weights and the checkpoint content. Default is None.
        load_via_cpu (bool, optional):
            Whether to load checkpoint data into CPU memory first before transferring to GPU.
            This helps mitigate GPU memory shortage by staging data on the CPU and only moving required parts to the GPU on demand during communication.
            Defaults to True.
        save_hf_steps (`int`, *optional*, defaults to -1):
            Number of updates steps before two huggingface checkpoint saves if `save_strategy="steps"`.
        hybrid_parallel_expert_grad_scale (float, optional, defaults to None)(
            Scaling factor for expert gradients when Expert Parallel is enabled.

            When Expert Parallel is enabled, the number of tokens processed by each MoE expert
            may change due to variations in context parallel degree, tensor parallel degree, or
            expert parallel degree. To ensure that the gradient scale for each expert remains
            consistent and is not affected by such changes, this factor should be applied to
            the expert gradients.

            The value is defined as the ratio of the current configuration to a reference (base) configuration:

            .. math::
                hybrid\\_parallel\\_expert\\_grad\\_scale} =
                    \\frac{tensor\\_parallel\\_degree}{expert\\_parallel\\_degree}
        )
        enable_auto_parallel (`bool`, *optional*, defaults to `False`):
            whether to run distributed training in auto parallel mode.
        use_intermediate_api (`bool`, *optional*, defaults to `True`):
            whether to use auto_parallel intermediate API if `enable_auto_parallel=True`.

        use_cache (`bool`, *optional*, defaults to `False`):
            Whether or not to enable cache for the model. For training, this is usually not needed apart from some PEFT methods that uses `past_key_values`.

        load_from_hf (bool, optional):
            Whether to load a checkpoint in the HuggingFace format.
            Defaults to False.

        flex_ckpt_comm_method (str, optional):
            Communication method used for checkpoint resharding.
            Choices are "send_recv", "broadcast", "multi_group_broadcast", and "grouped_send_recv".
            Defaults to "broadcast".

        replicate_saved_into_local (bool, optional):
            Whether to save checkpoint replicas into local files in a distributed save/load system.
            If set to True, replicas will be stored locally on each node/machine.
            Defaults to False.

    """

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    do_export: bool = field(default=False, metadata={"help": "Whether to export infernece model."})
    evaluation_strategy: IntervalStrategy = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU core/CPU for training."})
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=1.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={"help": "The scheduler type to use. support linear, cosine, constant, constant_with_warmup"},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    num_cycles: float = field(default=0.5, metadata={"help": "The number of waves in the cosine scheduler."})
    lr_end: float = field(default=1e-7, metadata={"help": "The end LR in the polynomial scheduler."})
    power: float = field(default=1.0, metadata={"help": "The power factor in the polynomial scheduler."})
    min_lr: float = field(default=0.0, metadata={"help": "The minimum learning rate in cosine scheduler."})

    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": "When doing a multinode distributed training, whether to log once per node or just once on the main node."
        },
    )
    logging_dir: Optional[str] = field(default=None, metadata={"help": "VisualDL log dir."})
    output_signal_dir: Optional[str] = field(default=None, metadata={"help": "Asynchronous saving signal dir."})
    logging_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})

    save_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": "When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on the main one"
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (no_cuda). This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: AMP optimization level selected in ['O0', 'O1', and 'O2']. "
                "See details at https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/auto_cast_cn.html"
            )
        },
    )
    amp_master_grad: bool = field(
        default=True,
        metadata={
            "help": "amp_master_grad (bool, optional) – For amp opt level=’O2’, whether to use float32 weight gradients "
            " for calculations such as gradient clipping, weight decay, and weight updates. If master_grad is enabled,"
            " the weight gradients will be float32 dtype after the backpropagation. Default is False, there is only float16 weight gradients."
            "Note: only support model parallel and pipeline parallel for now !!!"
        },
    )
    bf16_full_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may"
                " change."
            )
        },
    )
    fp16_full_eval: bool = field(
        default=False,
        metadata={"help": "Whether to use full float16 evaluation instead of 32-bit"},
    )

    amp_custom_black_list: Optional[List[str]] = field(
        default_factory=lambda: [
            "reduce_sum",
            "softmax_with_cross_entropy",
            "c_softmax_with_cross_entropy",
            "elementwise_div",
            "sin",
            "cos",
        ],
        metadata={
            "help": "The set of ops that support fp16/bf16 calculation and are considered numerically-dangerous and whose effects may also be observed in downstream ops."
        },
    )
    amp_custom_white_list: Optional[List[str]] = field(
        default_factory=lambda: [
            "lookup_table",
            "lookup_table_v2",
            "flash_attn",
            "matmul",
            "matmul_v2",
            "fused_gemm_epilogue",
        ],
        metadata={
            "help": "The the set of ops that support fp16/bf16 calculation and are considered numerically-safe and performance-critical. These ops will be converted to fp16/bf16."
        },
    )

    sharding: str = field(
        default="stage1",
        metadata={
            "help": (
                "Whether or not to use Paddle Sharding Data Parallel training (in distributed training"
                " only). The base option should be `stage1`, `stage2` or `stage3` and you can add"
                " CPU-offload to `stage2` or `stage3` like this: stage2 offload` or `stage3"
                " offload`. "
            )
        },
    )
    sharding_degree: int = field(  # Alias for sharding_parallel_size
        default=-1,
        metadata={"help": ("@deprecated Please use sharding_parallel_size. ")},
    )
    sharding_parallel_size: int = field(
        default=-1,
        metadata={
            "help": (
                "Sharding parameter in certain cards group. For example, aussume we use 2 machines each with 8 cards, "
                "then set sharding_degree=8, sharding will only communication inside machine. "
                "default -1 means sharding parameters between all workers."
            )
        },
    )
    sharding_parallel_mesh_dimension: str = field(
        default="dp",
        metadata={
            "help": (
                "Specifies the name of the dimension in a multi-dimensional parallelism mesh that is responsible for sharding. "
                "default `dp` for default parallelism mesh. "
            )
        },
    )
    sharding_comm_buffer_size_MB: int = field(
        default=-1,
        metadata={
            "help": (
                "Set the size of the fuse gradient in sharding communication. This option only takes effect when "
                "the sharding option is turned on.The default value is -1, which means that the gradient size of "
                "all communication fuses follows the default configuration, which is 256MB. "
            )
        },
    )

    sharding_offload_opt_buffersize_GB: int = field(
        default=-1,
        metadata={
            "help": (
                "Set the size of the optimizer offload buffer when need_hack_offload_optimizer() is True. This option only takes effect when "
                "use DygraphShardingOptimizerV2. The default value is -1, which means that all of the optimizer states will be offloaded. Only "
                "works when export HACK_OFFLOAD_OPTIMIZER=1. "
            )
        },
    )

    save_sharded_model: bool = field(
        default=False,
        metadata={
            "help": (
                "When use sharding stage1 and set save_sharded_model True, each shanding rank only save part of the model. It reduce time to save the model."
            )
        },
    )

    load_sharded_model: bool = field(
        default=False,
        metadata={
            "help": (
                "When use sharding stage1 and set load_sharded_model True, it means loading the sharded model. The sharded model is saved when we set save_sharded_model True."
            )
        },
    )

    load_sharded_model_remap_parameter_name: bool = field(
        default=False,
        metadata={"help": "Whether to remap parameter name when load_sharded_model = true."},
    )

    sharded_model_from_ema: bool = field(
        default=False,
        metadata={"help": "Whether to load sharded model from EMA."},
    )

    tensor_model_parallel_size: int = field(
        default=-1,
        metadata={
            "help": (
                "Tensor parallelism is parallel technique proposed in (https://arxiv.org/pdf/2104.04473.pdf see 2.3 Tensor Model Parallelism). "
                "This technique splits one transformer layer into multi-cards (For examples, tensor_model_parallel_size=4, will split a layer to 4-parts) "
                "tensor_model_parallel_size means split the transformer layer to how many parts."
                "default -1 for not use tensor parallel,  Suggest tensor_model_parallel_size<=8 for better proformance."
                "Note, this need model support in source code, currently GPT/BLOOM/LLAMA/BLOOM/CLM/CHATGLM is supported. "
            )
        },
    )
    pipeline_model_parallel_size: int = field(
        default=-1,
        metadata={
            "help": (
                "Pipeline parallelism is parallel technique proposed in (https://arxiv.org/pdf/2104.04473.pdf see 2.2 Pipeline Model Parallelism). "
                "Pipeline parallelism assigns multi-transformer layers to different cards, the micro batch data stream passed between cards like pipelines."
                "pipeline_model_parallel_size means split all transformer layers to how many stages."
                "default -1 for not use pipeline parallel."
                "Note. this need model support in source code, see llama modeling_pp.py file"
            )
        },
    )
    sep_parallel_size: int = field(
        default=-1,
        metadata={
            "help": (
                "The paddle sequence parallel strategy. It can reduce the GPU memory of activation to 1/sep, and it is orthogonal to "
                "data parallel, sharding stage1, tensor parallel and pipeline parallel strategy. "
            )
        },
    )
    split_inputs_sequence_dim: bool = field(
        default=True,
        metadata={
            "help": (
                "The paddle sequence parallel strategy can reduce the GPU memory of activation to 1/sep .If it is true, trainer will cut input in sequence dim "
            )
        },
    )
    context_parallel_size: int = field(
        default=-1,
        metadata={
            "help": (
                "The paddle context parallel strategy. It can reduce the GPU memory of activation to 1/cp, and it is orthogonal to "
                "data parallel, sharding stage1, tensor parallel and pipeline parallel strategy. "
            )
        },
    )
    expert_model_parallel_size: int = field(
        default=-1,
        metadata={"help": ("The paddle expert data parallel strategy.")},
    )
    expert_tensor_model_parallel_size: int = field(
        default=-1,
        metadata={"help": ("The paddle expert tensor parallel strategy. Currently is not supported. DO NOT SET.")},
    )
    # Deprecated
    data_parallel_config: str = field(
        default="",
        metadata={
            "help": (
                "@deprecated Please promote the secondary switchs to primary switchs."
                "Some additional configs which affect data parallel performance, we provide some option to config it."
                "following config is support:\n"
                "enable_allreduce_avg_in_gradinent_scale, it replace `allreduce_sum + scale` pattern with `allreduce_avg` when scale gradient in data_parallel, which improve the performance. ONLY supported for auto mode now. \n"
                "gradient_sync_after_accumulate, move gradient sync operations from backward into optimizer step when gradient accumulate enabling, which reduce the sync times to improve performance, but will increase the memory usage. ONLY supported for auto mode now. \n"
            )
        },
    )
    sequence_parallel: bool = field(
        default=False,
        metadata={"help": "Whether to enable sequence parallel."},
    )
    fuse_sequence_parallel_allreduce: bool = field(
        default=False, metadata={"help": "Whether to use fuse sequence parallel allreduce."}
    )
    # Deprecated
    sequence_parallel_config: str = field(
        default="",
        metadata={
            "help": (
                "@deprecated Please promote the secondary switchs to primary switchs."
                "Some additional configs which affect sequence parallel performance, we provide some option to config it."
                "following config is support:\n"
                "enable_allreduce_avg_in_gradinent_scale, it replace `allreduce_sum + scale` pattern with `allreduce_avg` when scale gradient in sequence_parallel, which improve the performance. ONLY supported for auto mode now. \n"
            )
        },
    )
    # Deprecated
    tensor_parallel_config: str = field(
        default="",
        metadata={
            "help": (
                "@deprecated Please promote the secondary switchs to primary switchs."
                "Some additional configs which affect model parallel performance, we provide some option to config it."
                "following config is support:\n"
                "enable_mp_async_allreduce, it supports all_reduce(dx) overlap with matmul(dw) in ColumnParallelLinear backward when it set True, which can accelerate model parallel performance. \n"
                "enable_mp_skip_c_identity, it supports skip c_identity in ColumnParallelLinear and RowParallelLinear. It only works when set mp_async_allreduce is True. It can accelerate model parallel further.\n"
                "enable_mp_fused_linear_param_grad_add, it supports fused_linear_param_grad_add in ColumnParallelLinear (cuda >= 11.6). It only works when mp_async_allreduce is true.  It can accelerate model parallel further.\n"
                "enable_sp_async_reduce_scatter, it supports async reduce_scatter in ColumnSequenceParallelLinear. It only works when set sp_async_reduce_scatter is True. It can accelerate sequence parallel further.\n"
                "enable_delay_scale_loss, accumulate gradients until optimizer step, all gradients div by accumute step. instead of div accumute step on loss directly.\n"
                "sync_param, in optimizer step, use broadcast to sync parameters those attr 'is_distributed' is False.\n"
                "sync_grad, in optimizer step, use broadcast to sync gradients those attr 'is_distributed' is False.\n"
                "sync_moment, in optimizer step, use broadcast to sync momentums those attr 'is_distributed' is False.\n"
                "replace_with_c_embedding, it supports replacing col-sliced embedding with row-sliced c_embedding when it set True, which is used in PIR auto_parallel.\n"
                "replace_with_parallel_cross_entropy, it replaces 'cross_entropy_with_softmax' OP with 'c_softmax_with_cross_entropy' OP in PIR static graph, which can improve model parallel performance.\n"
            )
        },
    )
    # Deprecated
    pipeline_parallel_config: str = field(
        default="",
        metadata={
            "help": (
                "@deprecated Please promote the secondary switchs to primary switchs."
                "Some additional config it highly affect the usage of pipeline parallel, we provide some option to config it."
                "following config is support:\n"
                "disable_p2p_cache_shape, if you max sequence length is varying, please set disable_p2p_cache_shape. \n"
                "disable_partial_send_recv, optimize send speed for tensor parallel.\n"
                "enable_delay_scale_loss, accumulate gradients until optimizer step, all gradients div by inner pipeline accumute step. instead of div accumute step on loss directly.\n"
                "enable_dp_comm_overlap, fuse data parallel gradient communication. \n"
                "enable_sharding_comm_overlap, fuse sharding stage 1 parallel gradient communication. \n"
                "enable_overlap_p2p_comm, overlap p2p communication with computation. \n"
                "enable_clear_every_step_cache, clear every step cache for pipeline parallel. \n"
                "disable_batch_p2p_comm, disable batched send/recv in pipeline parallel mode. \n"
                "enable_split_backward, only can be used in StaticGraph-AutoParallel! split the `backward` program into `backward_b` and `backward_w` to decrease the bubble in VPP pipeline mode when `acc_step == pp_degree`. it increase the memory! \n"
                "auto_parallel_sync_shared_params, optimize the parameter sharing between two stages in a pipeline parallel scenario."
            )
        },
    )
    # Deprecated
    sharding_parallel_config: str = field(
        default="",
        metadata={
            "help": (
                "@deprecated Please promote the secondary switchs to primary switchs."
                "Some additional config it highly affect the usage of sharding parallel, we provide some option to config it."
                "following config is support: \n"
                "enable_stage1_tensor_fusion, fuse small tensors into big tensor chunks to accelerate communications, may increase memory occupation\n"
                "enable_tensor_fusion, fuse small tensors into big tensor chunks to accelerate communications, may increase memory occupation only used for semi auto mode.\n"
                "enable_stage1_overlap, fuse small tensors into big tensor chunks to accelerate communications and do communication overlap with backward computation, may harm the backward speed\n"
                "enable_overlap, fuse small tensors into big tensor chunks to accelerate communications and do communication overlap with backward computation, may harm the backward speed only used for semi auto mode.\n"
                "disable_stage1_reduce_avg, replace reduce_avg with original reduce_sum+scale in stage1, which can be used for accuracy verification.\n"
                "enable_stage2_overlap, overlap stage2 NCCL communication with computation. There are some constraints for the overlap, such as the logging_step should be bigger than 1 for broadcast overlap and no other sync could be called during the training for broadcast overlap\n"
                "enable_stage1_broadcast_overlap, overlap stage1 V1 broadcast with next step forward computation. There are some constraints for the overlap, such as the logging_step should be bigger than 1 for broadcast overlap forward compute and no other sync could be called during the training for broadcast overlap.\n"
                "enable_stage1_allgather_overlap, overlap stage1 V2 allgather with next step forward computation. There are some constraints for the overlap, such as the logging_step should be bigger than 1 for allgather overlap forward compute and no other sync could be called during the training for allgather overlap.\n"
            )
        },
    )
    hybrid_parallel_topo_order: str = field(
        default=None,
        metadata={
            "help": (
                "In hybrid parallelism, the order of communication groups may affect efficiency.\n"
                "Following options are supported:\n"
                "- pp_first. the topo order is dp, pp, sharding, mp \n"
                "- sharding_first. the topo order is dp, sharding, pp, mp \n"
                "Default is None, for sharding_first"
            )
        },
    )

    recompute_granularity: Optional[str] = field(
        default=None, metadata={"help": "Determines which type of activation recompute to use"}
    )

    recompute_method: Optional[str] = field(
        default=None, metadata={"help": "Determines which transformer layers will be recomputed"}
    )

    recompute_modules: Optional[Any] = field(default=None, metadata={"help": "The submodules to recompute"})

    recompute_num_layers: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "When recompute_method is uniform, recompute_num_layers is the number of transformer layers in"
                "each uniformly divided recompute unit.  When recompute_method is block, recompute_num_layers is"
                "the number of transformer layers to recompute within each pipeline stage."
            )
        },
    )

    recompute_mtp_granularity: Optional[str] = field(
        default=None, metadata={"help": "Determines which type of activation recompute to use in MTP layer"}
    )

    recompute_mtp_method: Optional[str] = field(
        default=None, metadata={"help": "Determines which layers will be recomputed in MTP layer"}
    )

    recompute_mtp_modules: Optional[Any] = field(
        default=None, metadata={"help": "The submodules to recompute in MTP layer"}
    )

    scale_loss: float = field(default=2**15, metadata={"help": "The value of initial scale_loss for fp16."})

    minimum_eval_times: int = field(
        default=None,
        metadata={
            "help": "If under eval_steps, the valid time is less then minimum_eval_times, the config of override eval_steps."
        },
    )

    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    dataloader_shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the train dataloder."})
    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    max_evaluate_steps: int = field(
        default=-1, metadata={"help": "If set to a positive number, the total number of evaluation steps to perform."}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        },
    )
    prefetch_factor: int = field(
        default=2,
        metadata={
            "help": "Number of batch data the DataLoader would prefetch if use_buffer_reader=True. " "Default 2."
        },
    )

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    run_name: Optional[str] = field(default=None, metadata={"help": "An optional descriptor for the run."})

    device: Optional[str] = field(default="gpu", metadata={"help": "select cpu, gpu, xpu, npu devices."})

    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": "When resuming training, whether or not to skip the first epochs and batches to get to the same training data."
        },
    )
    optim: str = field(
        default="adamw",
        metadata={"help": "The optimizer to use."},
    )
    use_lowprecision_moment: bool = field(
        default=False,
        metadata={"help": "AdamW use 16bit moment as model parameter."},
    )
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DataParallel`."
            )
        },
    )
    wandb_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases (WandB) API key(s) for authentication with the WandB service."},
    )
    wandb_http_proxy: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases (WandB) http proxy for connecting with the WandB service."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    resume_from_huggingface_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid huggingface checkpoint for your model."},
    )
    auto_parallel_resume_form_hybrid_parallel: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether hybrid parallel checkpoints be loaded in auto parallel mode."},
    )
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    flatten_param_grads: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether use flatten_param_grads method in optimizer, only used on NPU devices."},
    )
    lazy_data_processing: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether use lazy data processing."},
    )
    use_async_save: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use async_save instead of paddle.save."},
    )
    ordered_save_group_size: int = field(
        default=0,
        metadata={
            "help": "Select ordered_save_group_size to save checkpoint in ordered. if ordered_save_group_size=0, not used ordered save"
        },
    )
    metrics_output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Where to save training metrics (None for skipping save)."},
    )
    skip_profile_timer: Optional[bool] = field(
        default=True,
        metadata={"help": "enable framework timer, will output timeline information in logging and visualdl."},
    )
    distributed_dataloader: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use distributed dataloader."}
    )
    unified_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to unify hybrid parallel checkpoint."},
    )
    to_static: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to train model under static mode by jit.to_static or distributed.to_static.")},
    )
    unified_checkpoint_config: Optional[str] = field(
        default="",
        metadata={
            "help": (
                "Configs to unify hybrid parallel checkpoint.\n"
                "Following options are supports:\n"
                "- skip_save_model_weight: do not save model weights when the masters weight exist\n"
                "- master_weight_compatible: 1. if the master weights exist, only load when needed\n"
                "                            2. if master weights does not exist, convert model weights to master weights when needed\n"
                "- remove_master_weight: same with `master_weight_compatible`, use in checkpoint quantization.\n"
                "- async_save: enable asynchronous saving checkpoints to disk\n"
                "- enable_all_options: enable all optimization configurations\n"
            )
        },
    )
    ckpt_quant_stage: str = field(
        default="O0",
        metadata={
            "help": "checkpoint quantization stage. O0: deactivate, O1: Int8 compression, O2: Int4 compression. (default: O0)"
        },
    )
    ignore_load_lr_and_optim: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to ignore load optimizer and scheduler."},
    )
    ignore_save_lr_and_optim: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to ignore save optimizer and scheduler."},
    )
    force_reshard_pp: Optional[bool] = field(
        default=False,
        metadata={"help": "reshard pp even if pp degree in the model and pp degree in script match"},
    )
    enable_auto_parallel: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to run distributed training in auto parallel mode"},
    )
    use_expert_parallel: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable MoE (Mixture of Experts) expert parallel training"},
    )
    aux_loss_alpha: Optional[float] = field(
        default=0.0001,
        metadata={"help": "MoE (Mixture of Experts) Auxiliary loss weight coefficient"},
    )
    release_grads: Optional[bool] = field(
        default=False, metadata={"help": "Whether to release gradients during training. Default is `False`."}
    )
    skip_data_intervals: Optional[List[List[int]]] = field(
        default=None,
        metadata={"help": "The intervals to skip, pass start global step and end global step at each interval"},
    )
    offload_optim: Optional[bool] = field(
        default=False,
        metadata={"help": "Offload optimizer after optimizer.step()"},
    )
    tensorwise_offload_optimizer: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Offload all optimizer states to CPU memory. "
                "The corresponding parameters will only be loaded to GPU during optimizer step, "
                "which reduces GPU memory usage but may increase step time."
            )
        },
    )
    optim_shard_num: int = field(
        default=1,
        metadata={"help": ("Number of shards to split the optimizer into.")},
    )
    save_sharding_stage1_model_include_freeze_params: Optional[bool] = field(
        default=False, metadata={"help": "Save Sharding Stage1 Model Exclude Freeze Params"}
    )
    enable_zero_cost_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable Flash Save Mode"},
    )
    zcc_workers_num: Optional[int] = field(
        default=3,
        metadata={
            "help": "The workers num for zero cost checkpoint save mode. Increase to gain performance but cost more memory and cpu usage."
        },
    )
    zcc_pipeline_hooks_capacity_usage: Optional[float] = field(
        default=0.6,
        metadata={
            "help": "Set pipeline hook capacity usage ratio. Lower value brings faster save speed but may effect calculation speed."
        },
    )
    zcc_save_ema_coef: Optional[float] = field(
        default=None,
        metadata={
            "help": "The coefficient of EMA parameters in zero cost checkpoint save mode. if set to 0, skip EMA process"
        },
    )
    zcc_ema_interval: Optional[int] = field(
        default=1,
        metadata={"help": "Interval between updating EMA parameters."},
    )
    zcc_ema_loss_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "If set not None, only do EMA when the training loss is smaller than the threshold value"},
    )
    save_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Save tokenizer to output_dir."},
    )
    save_rng_states: Optional[bool] = field(
        default=True,
        metadata={"help": "Save rng states to output_dir."},
    )
    pdc_download_ckpt: Optional[bool] = field(
        default=False,
        metadata={"help": "Download checkpoint in paddlecloud longjob environment"},
    )
    pdc_download_timeout: Optional[int] = field(
        default=300,
        metadata={"help": "Timeout seconds for downloading checkpoint from remote cluster."},
    )
    count_trained_tokens: bool = field(
        default=False,
        metadata={"help": "Whether to count trained tokens."},
    )
    pad_token_id: int = field(
        default=0,
        metadata={"help": "The id of the padding token."},
    )
    flash_device_save_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Save checkpoints on flash device every this many steps. Default is 0 which disables it"},
    )
    split_norm_comm: Optional[bool] = field(
        default=False,
        metadata={"help": "是否开启单路sharding时global norm通信拆分全局通信组为pp通信和mp通信分别做"},
    )
    convert_from_hf: Optional[bool] = field(
        default=True,
        metadata={"help": "Load model from HuggingFace safetensors."},
    )
    save_to_hf: Optional[bool] = field(
        default=True,
        metadata={"help": "Save model to HuggingFace safetensors."},
    )
    nccl_comm_group_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "supporting fine-grained control of communication groups in NCCL. "
                "The default value is None, indicating that this configuration is not enabled"
            )
        },
    )

    reorder_pipeline_priority: Optional[bool] = field(
        default=False,
        metadata={"help": "Controls the parallel execution order. False (pp first), True (sharding first)."},
    )
    pre_alloc_memory: int = field(
        default=0,
        metadata={"help": "pre allocate memory size GB"},
    )
    num_nextn_predict_layers: int = field(default=0, metadata={"help": "Number of nextn predict layers."})
    profile: bool = field(default=False, metadata={"help": "Enable nsys profiling."})
    profile_step_start: int = field(default=10, metadata={"help": "Step to start nsys profiling."})
    profile_step_end: int = field(default=12, metadata={"help": "Step to end nsys profiling."})
    save_checkpoint_format: Optional[str] = field(
        default="flex_checkpoint",
        metadata={
            "help": (
                "Specifies the format used to save checkpoints. "
                "Available options: 'sharding_io', 'unified_checkpoint', "
                "'flex_checkpoint'."
                "This setting is ignored if the corresponding switch is configured."
            )
        },
    )

    load_checkpoint_format: Optional[str] = field(
        default="flex_checkpoint",
        metadata={
            "help": (
                "Specifies the format used to load checkpoints. "
                "Available options: 'sharding_io', 'unified_checkpoint', "
                "'flex_checkpoint'."
                "This setting is ignored if the corresponding switch is configured."
            )
        },
    )

    aoa_config: Optional[dict[str, list[str]]] = field(
        default=None,
        metadata={
            "help": "The AoA configuration of FlexCheckpoint, used to describe the mapping between model weights and the checkpoint content. Default is None."
        },
    )

    load_via_cpu: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If True, loads checkpoint data to CPU first, then transfers required parts to GPU on demand to reduce GPU memory usage. Defaults to False."
        },
    )

    save_hf_steps: int = field(default=-1, metadata={"help": "Save huggingface checkpoint every X updates steps."})

    hybrid_parallel_expert_grad_scale: Optional[float] = field(
        default=None,
        metadata={"help": ("Scaling factor for expert gradients.")},
    )
    use_intermediate_api: bool = field(
        default=True,
        metadata={"help": "whether to use auto_parallel intermediate API."},
    )
    offload_fp8_expert_master_weight: bool = field(
        default=True,
        metadata={"help": "Offload FP8 expert weights."},
    )
    use_cache: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use cache for the model For training, this is usually not needed apart from some PEFT methods that uses `past_key_values`."
        },
    )
    load_from_hf: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to load a checkpoint in the HuggingFace format."},
    )
    flex_ckpt_comm_method: Optional[str] = field(
        default="broadcast",
        metadata={
            "help": (
                "Communication method used by FlexCheckpoint for checkpoint resharding. "
                'Choices are "send_recv", "broadcast", "multi_group_broadcast", and "grouped_send_recv". '
                'Default is "broadcast".'
            )
        },
    )
    deterministic_mode: bool = field(
        default=False,
        metadata={"help": "Whether to use deterministic mode."},
    )
    cp_comm_type: Optional[str] = field(
        default=None,
        metadata={"help": "Communication type."},
    )
    replicate_saved_into_local: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to save replicas cross files in distributed save load system."},
    )
    dp_comm_overlap: bool = field(
        default=False, metadata={"help": "Whether to overlap data parallelism (DP) communication with computation."}
    )
    pp_sharding_comm_overlap: bool = field(
        default=False,
        metadata={
            "help": "Whether to overlap sharding parallelism (SP) communication with computation. Reduces latency for sharded models. Defaults to True."
        },
    )
    tp_async_allreduce: bool = field(
        default=False, metadata={"help": "Whether to use asynchronous allreduce for tensor parallelism (TP)."}
    )
    sp_async_reduce_scatter: bool = field(
        default=False, metadata={"help": "Whether to use asynchronous reduce-scatter for sharding parallelism (SP)."}
    )
    overlap_p2p_comm: bool = field(
        default=False,
        metadata={"help": "Whether to overlap point-to-point (P2P) communication with computation. Defaults to True."},
    )
    batch_p2p_comm: bool = field(
        default=True, metadata={"help": "Whether to batch point-to-point (P2P) communication requests."}
    )
    variable_seq_lengths: bool = field(
        default=False,
        metadata={
            "help": "Whether to support dynamic input shapes (variable sequence lengths). Critical for LLM inference with varying prompt lengths. Defaults to True (standard for LLM pipelines)."
        },
    )
    mtp_loss_scaling_factor: float = field(
        default=1.0,
        metadata={
            "help": "Loss scaling factor for MTP (Mixture of Token-Parallel) training. Adjusts for imbalanced token distributions. Defaults to 1.0 (no scaling; tune for MTP-specific stability issues)."
        },
    )
    dp_allreduce_avg_in_gradinent_scale: bool = field(
        default=False,
        metadata={
            "help": "Replace `allreduce_sum + scale` pattern with `allreduce_avg` when scaling gradient in data_parallel/sequence_parallel, which improves performance. ONLY supported for auto mode now."
        },
    )
    sp_allreduce_avg_in_gradinent_scale: bool = field(
        default=False,
        metadata={
            "help": "Replace `allreduce_sum + scale` pattern with `allreduce_avg` when scaling gradient in data_parallel/sequence_parallel, which improves performance. ONLY supported for auto mode now."
        },
    )
    gradient_sync_after_accumulate: bool = field(
        default=False,
        metadata={
            "help": "Move gradient sync operations from backward into optimizer step when gradient accumulate is enabled, which reduces sync times to improve performance but increases memory usage. ONLY supported for auto mode now."
        },
    )
    mp_async_allreduce: bool = field(
        default=False,
        metadata={
            "help": "Support all_reduce(dx) overlap with matmul(dw) in ColumnParallelLinear backward when set to True, which can accelerate model parallel performance."
        },
    )
    mp_skip_c_identity: bool = field(
        default=False,
        metadata={
            "help": "Support skipping c_identity in ColumnParallelLinear and RowParallelLinear. Only works when mp_async_allreduce is True. Can accelerate model parallel further."
        },
    )
    mp_fused_linear_param_grad_add: bool = field(
        default=False,
        metadata={
            "help": "Support fused_linear_param_grad_add in ColumnParallelLinear (requires cuda >= 11.6). Only works when mp_async_allreduce is True. Can accelerate model parallel further."
        },
    )
    tp_delay_scale_loss: bool = field(
        default=False,
        metadata={
            "help": "Accumulate gradients until optimizer step, all gradients divided by accumulate step (instead of dividing accumulate step on loss directly). Also applies to inner pipeline accumulate step in relevant scenarios."
        },
    )
    pp_delay_scale_loss: bool = field(
        default=False,
        metadata={
            "help": "Accumulate gradients until optimizer step, all gradients divided by accumulate step (instead of dividing accumulate step on loss directly). Also applies to inner pipeline accumulate step in relevant scenarios."
        },
    )
    pp_sync_param: bool = field(
        default=False,
        metadata={
            "help": "In optimizer step, use broadcast to sync parameters whose attribute 'is_distributed' is False."
        },
    )
    tp_sync_param: bool = field(
        default=False,
        metadata={
            "help": "In optimizer step, use broadcast to sync parameters whose attribute 'is_distributed' is False."
        },
    )
    sync_grad: bool = field(
        default=False,
        metadata={
            "help": "In optimizer step, use broadcast to sync gradients whose attribute 'is_distributed' is False."
        },
    )
    tp_sync_moment: bool = field(
        default=False,
        metadata={
            "help": "In optimizer step, use broadcast to sync momentums whose attribute 'is_distributed' is False."
        },
    )
    pp_sync_moment: bool = field(
        default=False,
        metadata={
            "help": "In optimizer step, use broadcast to sync momentums whose attribute 'is_distributed' is False."
        },
    )
    replace_with_c_embedding: bool = field(
        default=False,
        metadata={
            "help": "Support replacing col-sliced embedding with row-sliced c_embedding when set to True, which is used in PIR auto_parallel."
        },
    )
    replace_with_parallel_cross_entropy: bool = field(
        default=False,
        metadata={
            "help": "Replace 'cross_entropy_with_softmax' OP with 'c_softmax_with_cross_entropy' OP in PIR static graph, which can improve model parallel performance."
        },
    )
    p2p_cache_shape: bool = field(
        default=True,
        metadata={"help": "Set this when maximum sequence length is varying (disables p2p cache shape)."},
    )
    partial_send_recv: bool = field(
        default=True, metadata={"help": "Optimize send speed for tensor parallel (disables partial send/recv)."}
    )
    pp_release_grads: bool = field(
        default=False,
        metadata={
            "help": "Reduce peak memory usage by releasing gradients after each iteration. The creation of gradients will be postponed until backward propagation of the next iteration."
        },
    )
    clear_every_step_cache: bool = field(
        default=False, metadata={"help": "Clear every step cache for pipeline parallel."}
    )
    non_batch_p2p_comm: bool = field(
        default=False, metadata={"help": "Disable batched send/recv in pipeline parallel mode."}
    )
    auto_parallel_sync_shared_params: bool = field(
        default=False,
        metadata={"help": "Optimize parameter sharing between two stages in a pipeline parallel scenario."},
    )
    best_unbalanced_scheduler: bool = field(
        default=False,
        metadata={
            "help": "Enable dynamic parameter sharding scheduler to optimize load balancing across GPUs during uneven computation patterns."
        },
    )
    offload_queue: bool = field(
        default=False,
        metadata={
            "help": "Enable asynchronous offload queue to overlap CPU-GPU memory transfers with computation, reducing GPU memory pressure."
        },
    )
    use_dualpipev: bool = field(
        default=False,
        metadata={
            "help": "Enable dual-pipeline virtual stages to overlap forward/backward computation and reduce pipeline bubbles in hybrid parallelism."
        },
    )
    forward_backward_overlap_scheduler: bool = field(
        default=False,
        metadata={
            "help": "Enable dynamic scheduler to overlap forward/backward computation with gradient communication, reducing training latency in distributed setups."
        },
    )
    send_recv_overlap: bool = field(
        default=False,
        metadata={
            "help": "Enable asynchronous overlap of gradient send/recv with GPU computation to minimize communication overhead in distributed training."
        },
    )
    split_backward: bool = field(
        default=False,
        metadata={
            "help": "Enable splitting backward pass into stages to balance computation and reduce peak memory usage in model parallelism."
        },
    )
    timer: bool = field(
        default=False,
        metadata={
            "help": "Enable timing for pipeline parallel stages to profile and optimize communication/computation overlap."
        },
    )
    stage1_tensor_fusion: bool = field(
        default=False,
        metadata={
            "help": "Fuse small tensors into big tensor chunks to accelerate communications. May increase memory occupation."
        },
    )
    tensor_fusion: bool = field(
        default=False,
        metadata={
            "help": "Fuse small tensors into big tensor chunks to accelerate communications. May increase memory occupation. Only used for semi auto mode."
        },
    )
    stage1_overlap: bool = field(
        default=False,
        metadata={
            "help": "Fuse small tensors into big tensor chunks to accelerate communications and overlap communication with backward computation. May harm backward speed."
        },
    )
    overlap: bool = field(
        default=False,
        metadata={
            "help": "Fuse small tensors into big tensor chunks to accelerate communications and overlap communication with backward computation. May harm backward speed. Only used for semi auto mode."
        },
    )
    stage2_overlap: bool = field(
        default=False,
        metadata={
            "help": "Overlap stage2 NCCL communication with computation. Constraints: logging_step should be bigger than 1 for broadcast overlap, and no other sync should be called during training for broadcast overlap."
        },
    )
    stage1_broadcast_overlap: bool = field(
        default=False,
        metadata={
            "help": "Overlap stage1 V1 broadcast with next step forward computation. Constraints: logging_step should be bigger than 1 for broadcast overlap forward compute, and no other sync should be called during training for broadcast overlap."
        },
    )
    stage1_allgather_overlap: bool = field(
        default=False,
        metadata={
            "help": "Overlap stage1 V2 allgather with next step forward computation. Constraints: logging_step should be bigger than 1 for allgather overlap forward compute, and no other sync should be called during training for allgather overlap."
        },
    )
    stage1_reduce_avg: bool = field(
        default=True,
        metadata={
            "help": "Replace reduce_avg with original reduce_sum+scale in stage1, which can be used for accuracy verification (disables stage1 reduce_avg)."
        },
    )
    fuse_optimizer_states: bool = field(default=False, metadata={"help": "Fuse optimizer states to a single storage."})
    sd_release_grads: bool = field(
        default=False,
        metadata={
            "help": "Reduce peak memory usage by releasing gradients after each iteration. The creation of gradients will be postponed until backward propagation of the next iteration."
        },
    )
    split_param: bool = field(
        default=True,
        metadata={
            "help": "Enable parameter sharding to distribute model parameters across devices, reducing memory footprint per GPU (ZeRO-style optimization)."
        },
    )
    sd_sharding_comm_overlap: bool = field(
        default=False,
        metadata={
            "help": "Whether to overlap sharding parallelism (SP) communication with computation. Reduces latency for sharded models. Defaults to True."
        },
    )
    sd_shardingv1_comm_overlap: bool = field(
        default=False,
        metadata={
            "help": "Whether to overlap sharding parallelism (SP) communication with computation. Reduces latency for sharded models. Defaults to True."
        },
    )
    fa_version: int = field(
        default=2, metadata={"help": "FlashAttention or FlashMask version. Can be set to 2 or 3. Default is 2."}
    )

    using_sonic_moe: bool = field(
        default=False,
        metadata={
            "help": "When enabled, the computation part of the moelayer will use the implementation provided by SonicMoE."
        },
    )

    def __post_init__(self):
        world_size = paddle.distributed.get_world_size()
        if in_auto_parallel_align_mode():
            # self.max_grad_norm = 0.0
            # The current auto_hybrid_pp has aligned the handling of ClipGradByGlobalNorm with the original dygraph semi-auto parallel and dynamic manual-parallel modes and can correctly handle grad_clip, so it is no longer necessary to set max_grad_norm=0.0.
            if self.max_grad_norm != 0.0:
                warnings.warn(
                    "max_grad_norm is not 0.0,We will execute ClipGradByGlobalNorm,if you want to disable it,please set max_grad_norm=0.0"
                )
            self.max_grad_norm = 0.0
            os.environ["FLAGS_max_inplace_grad_add"] = "65536"
            os.environ["FLAGS_embedding_deterministic"] = "1"
            os.environ["FLAGS_cudnn_deterministic"] = "1"

        if self.deterministic_mode:
            os.environ["FLAGS_cudnn_deterministic"] = "1"
            os.environ["FLAGS_embedding_deterministic"] = "1"

        if self.fa_version == 2 or self.fa_version == 3:
            if paddle.base.core.is_compiled_with_cuda():
                is_sm90 = (
                    paddle_device.get_device_capability()[0] == 9 and paddle_device.get_device_capability()[1] == 0
                )
                if is_sm90:
                    paddle.set_flags({"FLAGS_flash_attn_version": 3})
                    self.fa_version = 3
                    warnings.warn("sm90 automatic set fa_version to fa3")
                else:
                    paddle.set_flags({"FLAGS_flash_attn_version": self.fa_version})
                    logger.info(f"fa_version = {self.fa_version} set FLAGS_flash_attn_version to {self.fa_version}")
        else:
            raise ValueError(f"--fa_version should be 2 or 3, but got {self.fa_version}")

        env_local_rank = int(os.environ.get("PADDLE_RANK_IN_NODE", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank and paddle.distributed.get_world_size() > 1:
            self.local_rank = env_local_rank

        # NOTE(gongenlei): new add, disable sharding when we have only single gpu
        if paddle.distributed.get_world_size() <= 1:
            self.sharding = ""
            self.sharding_degree = -1
            self.sharding_parallel_size = -1
            self.tensor_model_parallel_size = -1
            self.pipeline_model_parallel_size = -1

        # convert to int
        self.log_level = -1
        self.log_level_replica = -1

        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)
        if self.output_signal_dir is None and self.output_dir is not None:
            self.output_signal_dir = self.output_dir
        if self.output_signal_dir is not None:
            self.output_signal_dir = os.path.expanduser(self.output_signal_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = False  # logger.getEffectiveLevel() > logging.WARN

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        if self.do_eval and self.evaluation_strategy == IntervalStrategy.NO:
            logger.warning(
                "evaluation_strategy reset to IntervalStrategy.STEPS for do_eval is True. you can also set evaluation_strategy='epoch'."
            )
            self.evaluation_strategy = IntervalStrategy.STEPS

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")

        if self.fp16_full_eval and self.bf16_full_eval:
            raise ValueError("At most one of fp16 and bf16 can be True for full eval, but not both")

        self.optim = OptimizerNames(self.optim)
        if self.optim == OptimizerNames.ADAMW_MINI and self.tensor_model_parallel_size > 1:
            raise ValueError("AdamW Mini currently doesn't support tensor parallelism.")

        self._post_init_parallel_degree()

        # check recompute
        if (
            self.recompute_modules is not None
            and not isinstance(self.recompute_modules, list)
            and not isinstance(self.recompute_modules, dict)
        ):
            raise ValueError("recompute_modules must be list, dict or None")
        # check recompute:
        if (
            self.recompute_mtp_modules is not None
            and not isinstance(self.recompute_mtp_modules, list)
            and not isinstance(self.recompute_mtp_modules, dict)
        ):
            raise ValueError("recompute_mtp_modules must be list, dict or None")

        if getattr(self, "moe_subbatch_token_num_before_dispatch", 0) > 0 and self.recompute_granularity == "full":
            raise ValueError(
                "When moe_subbatch_token_num_before_dispatch > 0, please set recompute_granularity='selective and add corresponding module name to recompute_modules"
            )
        self._post_init_save_checkpoint_format()
        self._post_init_load_checkpoint_format()
        if self.tensorwise_offload_optimizer and self.data_parallel_size > 1:
            raise NotImplementedError(
                f"Optimizer offload is not supported under data parallel. Please use sharding by setting --sharding stage1 --sharding_parallel_size {self.sharding_parallel_size * self.data_parallel_size}."
            )

        if self.to_static:
            assert world_size == 1 or self.enable_auto_parallel, (
                "It's not supported for training in static mode except the following cases : "
                "1. world_size == 1, which means single-card training while no parallelism is used; "
                "2. enable_auto_parallel is set to True, which means the training will be executed in static mode of auto parallel."
            )

        if self.distributed_dataloader and not (
            self.tensor_model_parallel_size > 1 or self.pipeline_model_parallel_size > 1
        ):
            warnings.warn("We set `distributed_dataloader` to False if tp_degree <= 1 and pp_degree <= 1")
            self.distributed_dataloader = False

        if self.amp_master_grad:
            if not (self.bf16 or self.fp16):
                logger.warning("set amp_master_grad to false since amp is disabled.")
                self.amp_master_grad = False

        # use_hybrid_parallel
        if self.use_hybrid_parallel:
            if ShardingOption.OFFLOAD in self.sharding:
                warnings.warn("`offload` is not supported NOW!")

            if self.pipeline_model_parallel_size > 1:
                if ShardingOption.FULL_SHARD in self.sharding or ShardingOption.SHARD_GRAD_OP in self.sharding:
                    raise ValueError(
                        "pipeline parallel is not compatible for sharding stage2 or stage3, please using sharding stage1"
                    )

            # TODO use paddle.distributed.is_initialized() after paddle 2.4rc
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
                strategy = fleet.DistributedStrategy()
                assert self.data_parallel_config == "", "data_parallle_config is not supported in hybrid parallel"
                if self.pipeline_model_parallel_size > 1 or is_paddlefleet_available():
                    pipeline_parallel_config = split_parallel_config(self.pipeline_parallel_config)
                    for x in pipeline_parallel_config:
                        if len(x) > 0:
                            if x not in [
                                "disable_p2p_cache_shape",
                                "disable_partial_send_recv",
                                "enable_delay_scale_loss",
                                "enable_dp_comm_overlap",
                                "enable_sharding_comm_overlap",
                                "enable_timer",
                                "enable_release_grads",
                                "enable_clear_every_step_cache",
                                "enable_overlap_p2p_comm",
                                "disable_batch_p2p_comm",
                                "best_unbalanced_scheduler",
                                "enable_offload_queue",
                                "use_dualpipev",
                                "forward_backward_overlap_scheduler",
                                "enable_dynamic_shape",
                                "sync_moment",
                                "sync_param",
                            ]:
                                raise ValueError(
                                    f"Found unknown pipeline mode config {x}, accept config is disable_p2p_cache_shape, disable_partial_send_recv."
                                )
                            warnings.warn(
                                f"The field pipeline_parallel_config is deprecated. "
                                f"Please promote this secondary switch {x} to a primary switch."
                            )

                    enable_partial_send_recv = self.partial_send_recv
                    if self.sequence_parallel and enable_partial_send_recv:
                        logger.warning(
                            "When use pipeline parallel and sequence parallel simultaneously, we should turn off partial send recv."
                        )
                        enable_partial_send_recv = False

                    strategy.pipeline_configs = {
                        "accumulate_steps": self.gradient_accumulation_steps,
                        "micro_batch_size": self.per_device_train_batch_size,
                        "enable_partial_send_recv": enable_partial_send_recv,
                        "p2p_cache_shape": self.p2p_cache_shape,
                        # "delay_scale_loss": True, Fix ME
                    }
                    logger.info(f"PP configs:{strategy.pipeline_configs}, use master_grad: {self.amp_master_grad}")

                    using_comm_overlap = self.pp_sharding_comm_overlap or self.dp_comm_overlap
                    enable_dp_comm_overlap = using_comm_overlap and self.data_parallel_size > 1
                    self.enable_sharding_comm_overlap = using_comm_overlap and self.sharding_parallel_size > 1
                    assert not (
                        enable_dp_comm_overlap and self.enable_sharding_comm_overlap
                    ), "dp_comm_overlap and sharding_comm_overlap cannot be enabled at the same time"

                    if self.enable_sharding_comm_overlap and not self.amp_master_grad:
                        raise ValueError(
                            "If `sharding_comm_overlap` in training_args, `amp_master_grad` must be True."
                        )

                    dygraph_pp_configs = {
                        "delay_scale_loss": self.pp_delay_scale_loss,
                        "dp_comm_overlap": enable_dp_comm_overlap,
                        "sharding_comm_overlap": self.enable_sharding_comm_overlap,
                        "enable_timer": self.timer,
                        "release_gradients": self.pp_release_grads or self.release_grads,
                        "overlap_p2p_comm": self.overlap_p2p_comm,
                        "clear_every_step_cache": self.clear_every_step_cache,
                        "use_batch_p2p_comm": self.batch_p2p_comm,
                        "best_unbalanced_scheduler": self.best_unbalanced_scheduler,
                        "enable_offload_queue": self.offload_queue,
                        "use_dualpipev": self.use_dualpipev,
                        "forward_backward_overlap_scheduler": self.forward_backward_overlap_scheduler,
                        "enable_dynamic_shape": self.variable_seq_lengths,
                    }

                    pp_sync_param = self.pp_sync_param
                    pp_sync_moment = self.pp_sync_moment

                    if pp_sync_param:
                        logger.info("setting pp sync_param")
                        strategy.hybrid_configs["pp_configs"].sync_param = True

                    if pp_sync_moment:
                        logger.info("setting pp sync_moment")
                        strategy.hybrid_configs["pp_configs"].sync_moment = True

                    if dygraph_pp_configs["dp_comm_overlap"]:
                        raise ValueError("overlap has accuracy issue")  # TODO: fix `overalap` + `delay_scale` issue

                    if self.do_eval:
                        if (
                            self.per_device_train_batch_size * self.gradient_accumulation_steps
                            != self.per_device_eval_batch_size
                        ):
                            logger.warning(
                                "In pipeline model, the evaluation also shares same setting with training. "
                                "We will enforce that per_device_eval_batch_size=per_device_train_batch_size * gradient_accumulation_steps."
                            )

                            self.per_device_eval_batch_size = (
                                self.per_device_train_batch_size * self.gradient_accumulation_steps
                            )

                if self.tensor_model_parallel_size > 1:
                    strategy.tensor_parallel_configs = {"tensor_init_seed": self.seed}

                    mp_config = split_parallel_config(self.tensor_parallel_config)

                    for x in mp_config:
                        if len(x) > 0:
                            if x not in [
                                "enable_mp_async_allreduce",
                                "enable_mp_skip_c_identity",
                                "enable_mp_fused_linear_param_grad_add",
                                "enable_sp_async_reduce_scatter",
                                "enable_delay_scale_loss",
                                "sync_param",
                                "sync_grad",
                                "sync_moment",
                            ]:
                                raise ValueError(
                                    f"Found unknown tensor parallel config {x}, "
                                    f"accept config is enable_mp_async_allreduce, enable_mp_skip_c_identity, enable_mp_fused_linear_param_grad_add, enable_sp_async_reduce_scatter, enable_delay_scale_loss, sync_param, sync_grad and sync_moment."
                                )
                            warnings.warn(
                                f"The field tensor_parallel_config is deprecated. "
                                f"Please promote this secondary switch {x} to a primary switch."
                            )
                    try:
                        if self.mp_async_allreduce:
                            strategy.hybrid_configs["mp_configs"].mp_async_allreduce = True
                            if self.mp_skip_c_identity:
                                strategy.hybrid_configs["mp_configs"].mp_skip_c_identity = True
                            if self.mp_fused_linear_param_grad_add:
                                strategy.hybrid_configs["mp_configs"].mp_fused_linear_param_grad_add = True
                        else:
                            if self.mp_skip_c_identity:
                                warnings.warn(
                                    "mp_skip_c_identity only works with mp_async_allreduce. It will not work."
                                )
                            if self.mp_fused_linear_param_grad_add:
                                warnings.warn(
                                    "mp_fused_linear_param_grad_add only works with mp_async_allreduce. It will not work."
                                )
                        if self.sp_async_reduce_scatter:
                            strategy.hybrid_configs["mp_configs"].sp_async_reduce_scatter = True

                        sync_param = self.tp_sync_param
                        sync_grad = self.sync_grad
                        sync_moment = self.tp_sync_moment

                        # sync_param_name = [""] matches any parameter name.
                        # If sync_param, sync_grad and sync_moment are not set, the default value in Paddle is :
                        # sync_param = True, sync_grad = False, sync_moment = False, sync_param_name = ["embedding", "layer_norm", ".b_"].

                        if sync_param or sync_grad or sync_moment:
                            logger.info("setting sync_param_name")
                            strategy.sync_param_name = [""]

                        if sync_param:
                            logger.info("setting sync_param")
                            strategy.hybrid_configs["mp_configs"].sync_param = True

                        if sync_grad:
                            logger.info("setting sync_grad")
                            strategy.hybrid_configs["mp_configs"].sync_grad = True

                        if sync_moment:
                            logger.info("setting sync_moment")
                            strategy.hybrid_configs["mp_configs"].sync_moment = True

                    except:
                        warnings.warn(
                            "The enable_mp_async_allreduce, enable_mp_skip_c_identity and enable_mp_fused_linear_param_grad_add are not supported "
                            "by current version of Paddle. Please try latest develop Paddle."
                        )

                def is_segment_parallel_supported():
                    import inspect

                    members = [name for (name, date) in inspect.getmembers(fleet.HybridCommunicateGroup)]
                    support_sep = "get_sep_parallel_world_size" in members
                    if not support_sep:
                        logger.warning("segment parallel is not supported!!!, Ignore it.")
                    return support_sep

                def is_context_parallel_supported():
                    import inspect

                    members = [
                        name for (name, date) in inspect.getmembers(fleet.base.topology.EPHybridCommunicateGroup)
                    ]
                    support_cp = "get_context_parallel_world_size" in members
                    if not support_cp:
                        logger.warning("context parallel is not supported!!! Ignore it.")
                    return support_cp

                if self.hybrid_parallel_topo_order == "pp_first":
                    if is_segment_parallel_supported():
                        order = ["dp", "pp", "sharding", "sep", "mp"]
                    else:
                        order = ["dp", "pp", "sharding", "mp"]
                if self.hybrid_parallel_topo_order == "sharding_first":
                    if is_segment_parallel_supported():
                        order = ["dp", "sharding", "pp", "sep", "mp"]
                    else:
                        order = ["dp", "sharding", "pp", "mp"]
                if self.use_expert_parallel:
                    if not self.reorder_pipeline_priority:
                        if self.moe_sharding_parallel_size >= 1 and self.expert_model_parallel_size > 1:
                            order.insert(-1, "ep")
                            sd_idx = order.index("sharding")
                            # if pp_first, the order = ["dp", "pp", "moe_sharding", "sharding", "sep", "ep", "mp"]
                            # if sharding_first, the order is ["dp", "moe_sharding", "sharding", "pp", "sep", "ep", "mp"]
                            order.insert(sd_idx, "moe_sharding")
                            if is_context_parallel_supported():
                                sd_idx = order.index("sharding")
                                order.insert(sd_idx, "cp")
                    else:
                        if self.moe_sharding_parallel_size >= 1 and self.expert_model_parallel_size > 1:
                            if is_context_parallel_supported():
                                order = ["sharding", "moe_sharding", "pp", "sep", "cp", "dp", "ep", "mp"]
                            else:
                                order = ["sharding", "moe_sharding", "pp", "sep", "dp", "ep", "mp"]
                        else:
                            order = ["sharding", "pp", "sep", "dp", "mp"]

                if is_context_parallel_supported():
                    hybrid_configs = {
                        "dp_degree": self.data_parallel_size,
                        "mp_degree": self.tensor_model_parallel_size,
                        "pp_degree": self.pipeline_model_parallel_size,
                        "sharding_degree": self.sharding_parallel_size,
                        "sep_degree": self.sep_parallel_size,
                        "cp_degree": self.context_parallel_size,
                        "order": order,
                    }
                elif is_segment_parallel_supported():
                    hybrid_configs = {
                        "dp_degree": self.data_parallel_size,
                        "mp_degree": self.tensor_model_parallel_size,
                        "pp_degree": self.pipeline_model_parallel_size,
                        "sharding_degree": self.sharding_parallel_size,
                        "sep_degree": self.sep_parallel_size,
                        "order": order,
                    }
                else:
                    hybrid_configs = {
                        "dp_degree": self.data_parallel_size,
                        "mp_degree": self.tensor_model_parallel_size,
                        "pp_degree": self.pipeline_model_parallel_size,
                        "sharding_degree": self.sharding_parallel_size,
                        "order": order,
                    }

                if self.expert_model_parallel_size > 1:
                    assert (
                        self.use_expert_parallel is True and self.moe_sharding_parallel_size >= 0
                    ), f"invalid expert_model_parallel_size {self.expert_model_parallel_size} and use_expert_paralle:{self.use_expert_parallel}."
                    hybrid_configs["ep_degree"] = self.expert_model_parallel_size
                    hybrid_configs["moe_sharding_degree"] = self.moe_sharding_parallel_size

                try:
                    if self.split_norm_comm:
                        hybrid_configs["split_norm_comm"] = True
                except (KeyError, AttributeError):
                    warnings.warn(
                        "The split_norm_comm is not supported "
                        "by current version of Paddle. Please try latest develop Paddle."
                    )

                if self.pipeline_model_parallel_size > 1:
                    hybrid_configs["pp_configs"] = dygraph_pp_configs
                    logger.info(f"using pipeline configs:{dygraph_pp_configs}")

                # setter once https://github.com/PaddlePaddle/Paddle/blob/b7295120b0e78b293cd7ae29706e21769d06a3cc/python/paddle/distributed/fleet/base/distributed_strategy.py#L1692
                strategy.hybrid_configs = hybrid_configs

                if self.sharding_parallel_size > 1:
                    sharding_parallel_config = split_parallel_config(self.sharding_parallel_config)

                    for x in sharding_parallel_config:
                        if len(x) > 0:
                            if x not in [
                                "enable_stage1_tensor_fusion",
                                "enable_stage1_overlap",
                                "enable_stage2_overlap",
                                "split_param",
                                "disable_stage1_reduce_avg",
                                "enable_stage1_broadcast_overlap",
                                "enable_stage1_allgather_overlap",
                                "enable_release_grads",
                                "enable_fuse_optimizer_states",
                            ]:
                                raise ValueError(
                                    f"Found unknown sharding mode config {x}, "
                                    f"accept config is enable_stage1_tensor_fusion, enable_stage1_overlap, enable_stage2_overlap, split_param, disable_stage1_reduce_avg, enable_stage1_broadcast_overlap, enable_stage1_allgather_overlap, enable_release_grads, enable_fuse_optimizer_states."
                                )
                            warnings.warn(
                                f"The field sharding_parallel_config is deprecated. "
                                f"Please promote this secondary switch {x} to a primary switch."
                            )
                    if not self.stage1_reduce_avg:
                        assert self.sharding == [
                            ShardingOption.SHARD_OP
                        ], "Only sharding stage1 supports to disable reduce_avg strategy."
                        try:
                            strategy.hybrid_configs["sharding_configs"].use_reduce_avg = False
                        except:
                            warnings.warn(
                                "The reduce_avg strategy is not supported by current version of Paddle so you don't need to disable it. The nccl comm in sharding still use reduce_sum and scale of gradients."
                            )

                    try:
                        if self.sharding_comm_buffer_size_MB > 0:
                            strategy.hybrid_configs["sharding_configs"].comm_buffer_size_MB = int(
                                self.sharding_comm_buffer_size_MB
                            )

                        if hasattr(strategy.hybrid_configs["sharding_configs"], "offload_opt_buffer_size"):
                            strategy.hybrid_configs["sharding_configs"].offload_opt_buffer_size = int(
                                self.sharding_offload_opt_buffersize_GB
                            )

                        if self.split_param:
                            strategy.hybrid_configs["sharding_configs"].split_param = True
                            assert self.amp_master_grad, "Currently sharding stage1 v2 only support amp_master_grad"

                        if self.sd_release_grads:
                            strategy.hybrid_configs["sharding_configs"].release_gradients = True

                        if self.fuse_optimizer_states:
                            strategy.hybrid_configs["sharding_configs"].enable_fuse_optimizer_states = True

                        if self.pipeline_model_parallel_size == 1:
                            strategy.hybrid_configs["sharding_configs"].tensor_fusion = self.stage1_tensor_fusion
                            if self.stage1_overlap:
                                strategy.hybrid_configs["sharding_configs"].comm_overlap = True
                                strategy.hybrid_configs[
                                    "sharding_configs"
                                ].accumulate_steps = self.gradient_accumulation_steps

                        else:
                            warnings.warn(
                                "For pipeline parallel with sharding, the sharding overlap and tensor fusion "
                                "should be configured in training_args."
                                '"enable_stage1_tensor_fusion" and "enable_stage1_overlap" in sharding_parallel_config will be ignored.'
                            )
                    except (KeyError, AttributeError):
                        warnings.warn(
                            "The enable_stage1_tensor_fusion or enable_stage1_overlap is not supported "
                            "by current version of Paddle. Please try latest develop Paddle."
                        )
                    if self.stage2_overlap:
                        assert (
                            ShardingOption.SHARD_GRAD_OP in self.sharding
                        ), f"stage2_overlap expects sharding=stage2, but got {self.sharding}."
                        assert self.logging_steps > 1, (
                            "The logging_steps should be greater than 1 for stage2 overlap, "
                            f"but got logging_steps={self.logging_steps}."
                        )
                    if self.stage1_broadcast_overlap:
                        assert (
                            ShardingOption.SHARD_OP in self.sharding
                        ), f"stage1_broadcast_overlap expects sharding=stage1, but got {self.sharding}."

                        assert (
                            not self.split_param
                        ), "split_param should not be set when stage1_broadcast_overlap is True."

                    if self.stage1_allgather_overlap:
                        assert (
                            ShardingOption.SHARD_OP in self.sharding
                        ), f"stage1_allgather_overlap expects sharding=stage1, but got {self.sharding}."

                        assert self.split_param, "split_param should be set when stage1_allgather_overlap is True."

                    if self.split_param:
                        if ShardingOption.SHARD_OP not in self.sharding:
                            logger.warning("Only sharding stage1 support split_param.")
                        assert self.amp_master_grad, "If `split_param` is True, `amp_master_grad` must be True."

                if self.nccl_comm_group_config is not None:
                    strategy = init_nccl_config(self.nccl_comm_group_config, strategy)

                fleet.init(is_collective=True, strategy=strategy)

                # In PaddleFleet, we should use the following code to initialize.
                if (
                    is_paddlefleet_available()
                    and get_tensor_model_parallel_group is not None
                    and get_tensor_model_parallel_group(False) is None
                ):
                    initialize_fleet(strategy)
                logger.info(strategy)

                if self.reorder_pipeline_priority:
                    if self.expert_model_parallel_size > 1:
                        self.add_moe_comm_group()

        elif self.enable_auto_parallel:

            assert paddle.distributed.get_world_size() > 1, "Auto parallel mode needs world size > 1."
            assert self.use_intermediate_api, "Auto parallel is only supported with intermediate API now."
            assert (
                not self.to_static
            ), "Auto parallel only support dyanmic parallel now. Static parallel will be supported later."

            self.tensor_model_parallel_size = max(self.tensor_model_parallel_size, 1)
            self.sep_parallel_size = max(self.sep_parallel_size, 1)
            self.context_parallel_size = max(self.context_parallel_size, 1)
            self.pipeline_model_parallel_size = max(self.pipeline_model_parallel_size, 1)

            assert (
                self.pipeline_model_parallel_size == 1
            ), "Current not support pipeline parallel in auto parallel mode."
            assert (
                world_size % (self.tensor_model_parallel_size * self.pipeline_model_parallel_size) == 0
            ), f"Total world_size:{world_size} should be divided by tensor_model_parallel_size: {self.tensor_model_parallel_size} and pipeline_model_parallel_size: {self.pipeline_model_parallel_size}."

            if self.sharding_parallel_size == -1:
                if len(self.sharding) > 0:
                    self.sharding_parallel_size = world_size // (
                        self.tensor_model_parallel_size * self.sep_parallel_size * self.pipeline_model_parallel_size
                    )

            self.sharding_parallel_size = max(self.sharding_parallel_size, 1)
            if self.sharding_parallel_size == 1 and len(self.sharding) > 0:
                logger.warning("sharding_parallel_size=1 means no sharding, please set sharding to empty!")
                self.sharding = []

            self.data_parallel_size = world_size // (
                self.sharding_parallel_size
                * self.tensor_model_parallel_size
                * self.sep_parallel_size
                * self.pipeline_model_parallel_size
            )

            if ShardingOption.OFFLOAD in self.sharding:
                warnings.warn("`offload` is not supported NOW!")

            strategy = fleet.auto.Strategy()
            if self.dataset_world_size > 1:
                data_parallel_config = set(self.data_parallel_config.split(" "))
                for x in data_parallel_config:
                    if len(x) > 0:
                        if x not in ["enable_allreduce_avg_in_gradinent_scale", "gradient_sync_after_accumulate"]:
                            raise ValueError(
                                f"Found unknown data parallel config {x}, accept config is enable_allreduce_avg_in_gradinent_scale."
                            )
                        warnings.warn(
                            f"The field data_parallel_config is deprecated. "
                            f"Please promote this secondary switch {x} to a primary switch."
                        )
                if self.dp_allreduce_avg_in_gradinent_scale:
                    strategy.gradient_scale_using_allreduce_avg = True
                if self.gradient_sync_after_accumulate:
                    strategy.dp_optimization.gradient_sync_after_accumulate = True
            sequence_parallel_config = set(self.sequence_parallel_config.split(" "))
            for x in sequence_parallel_config:
                if len(x) > 0:
                    if x not in ["enable_allreduce_avg_in_gradinent_scale"]:
                        raise ValueError(
                            f"Found unknown sequence parallel config {x}, accept config is enable_allreduce_avg_in_gradinent_scale."
                        )
                    warnings.warn(
                        f"The field sequence_parallel_config is deprecated. "
                        f"Please promote this secondary switch {x} to a primary switch."
                    )
            if self.sp_allreduce_avg_in_gradinent_scale:
                strategy.gradient_scale_using_allreduce_avg = True

            # navie-pp: pipeline_model_parallel_size > 1 and gradient_accumulation_steps == 1
            if self.pipeline_model_parallel_size > 1 and self.gradient_accumulation_steps > 1:
                pipeline_parallel_config = split_parallel_config(self.pipeline_parallel_config)
                for x in pipeline_parallel_config:
                    if len(x) > 0:
                        if x not in [
                            "enable_send_recv_overlap",
                            # "disable_p2p_cache_shape",      # no need for auto_parallel
                            # "disable_partial_send_recv",    # no implementation for auto_parallel
                            "enable_delay_scale_loss",
                            # "enable_dp_comm_overlap",       # no implementation for auto_parallel
                            # "enable_sharding_comm_overlap", # no implementation for auto_parallel
                            # "enable_timer",                 # no implementation for auto_parallel
                            # "disable_batch_p2p_comm",       # no implementation for auto_parallel
                            "enable_split_backward",
                            "auto_parallel_sync_shared_params",
                        ]:
                            raise ValueError(
                                f"Found unknown pipeline mode config {x}, accept config is enable_send_recv_overlap."
                            )
                        warnings.warn(
                            f"The field pipeline_parallel_config is deprecated. "
                            f"Please promote this secondary switch {x} to a primary switch."
                        )

                pipeline = strategy.pipeline
                pipeline.enable = True
                pipeline.enable_send_recv_overlap = self.send_recv_overlap
                pipeline.split_backward = self.split_backward
                pipeline.accumulate_steps = self.gradient_accumulation_steps
                pipeline.micro_batch_size = self.per_device_train_batch_size
                pipeline.schedule_mode = self.pipeline_schedule_mode
                pipeline.pp_degree = self.pipeline_model_parallel_size

                logger.info(f"PP configs:{strategy.pipeline}, use master_grad: {self.amp_master_grad}")

                if self.do_eval:
                    if (
                        self.per_device_train_batch_size * self.gradient_accumulation_steps
                        != self.per_device_eval_batch_size
                    ):
                        logger.warning(
                            "In pipeline model, the evaluation also shares same setting with training. "
                            "We will enforce that per_device_eval_batch_size=per_device_train_batch_size * gradient_accumulation_steps."
                        )
                        self.per_device_eval_batch_size = (
                            self.per_device_train_batch_size * self.gradient_accumulation_steps
                        )

            elif self.gradient_accumulation_steps > 1:
                gradient_merge = strategy.gradient_merge
                gradient_merge.enable = True
                gradient_merge.k_steps = self.gradient_accumulation_steps
                gradient_merge.avg = True

            if self.tensor_model_parallel_size > 1:
                mp_optimization = strategy.mp_optimization
                mp_config = split_parallel_config(self.tensor_parallel_config)

                for x in mp_config:
                    if len(x) > 0:
                        if x not in [
                            "enable_mp_async_allreduce",  # allreduce_matmul_grad_overlapping in auto_parallel
                            "enable_delay_scale_loss",
                            "replace_with_c_embedding",
                            # "enable_mp_fused_linear_param_grad_add",
                            "replace_with_parallel_cross_entropy",
                            "enable_sp_async_reduce_scatter",
                        ]:
                            raise ValueError(
                                f"Found unknown tensor parallel config {x}, "
                                f"accept config is enable_mp_async_allreduce, replace_with_c_embedding, and enable_mp_fused_linear_param_grad_add"
                            )
                        warnings.warn(
                            f"The field tensor_parallel_config is deprecated. "
                            f"Please promote this secondary switch {x} to a primary switch."
                        )
                try:
                    if self.mp_async_allreduce:
                        mp_optimization.allreduce_matmul_grad_overlapping = True
                    if self.replace_with_c_embedding:
                        mp_optimization.replace_with_c_embedding = True
                except:
                    warnings.warn(
                        "The enable_mp_async_allreduce, replace_with_c_embedding, enable_mp_skip_c_identity and enable_mp_fused_linear_param_grad_add are not supported "
                        "by current version of Paddle. Please try latest develop Paddle."
                    )

            if self.sharding_parallel_size > 1:
                sharding = strategy.sharding
                sharding.enable = True
                sharding.degree = self.sharding_parallel_size
                if ShardingOption.SHARD_OP in self.sharding:
                    sharding.stage = 1
                elif ShardingOption.SHARD_GRAD_OP in self.sharding:
                    sharding.stage = 2
                elif ShardingOption.FULL_SHARD in self.sharding:
                    sharding.stage = 3
                if self.sharding_comm_buffer_size_MB > 0:
                    sharding.comm_buffer_size_MB = int(self.sharding_comm_buffer_size_MB)

                sharding_parallel_config = split_parallel_config(self.sharding_parallel_config)
                for x in sharding_parallel_config:
                    if len(x) > 0:
                        if x not in [
                            "enable_tensor_fusion",
                            "enable_overlap",
                            "enable_release_grads",
                        ]:
                            if x in ["enable_stage1_overlap", "enable_stage2_overlap"]:
                                raise ValueError(
                                    "enable_stage1_overlap and enable_stage2_overlap are not supported in "
                                    "auto_parallel mode. Please use training_args.overlap instead."
                                )
                            elif x == "enable_stage1_tensor_fusion":
                                raise ValueError(
                                    "enable_stage1_tensor_fusion is not supported in auto_parallel mode. "
                                    "Please use training_args.tensor_fusion instead."
                                )
                            raise ValueError(
                                f"Found unknown sharding mode config {x}, "
                                f"accept config is training_args.tensor_fusion, "
                                "training_args.overlap, training_args.sd_release_grads."
                            )
                        warnings.warn(
                            f"The field sharding_parallel_config is deprecated. "
                            f"Please promote this secondary switch {x} to a primary switch."
                        )

                    if self.overlap:
                        sharding.enable_overlap = True

                    if self.tensor_fusion:
                        sharding.grad_bucket_size_numel = 210355872
                        sharding.enable_tensor_fusion = True

                    if self.sd_release_grads:
                        sharding.release_gradients = True

            if self.bf16 or self.fp16:
                amp = strategy.amp
                amp.enable = True
                amp.dtype = "bfloat16" if self.bf16 else "float16"
                amp.level = self.fp16_opt_level.lower()
                amp.use_master_grad = self.amp_master_grad
                amp.init_loss_scaling = self.scale_loss
                amp.custom_black_list = self.amp_custom_black_list if self.amp_custom_black_list is not None else []
                amp.custom_white_list = self.amp_custom_white_list if self.amp_custom_white_list is not None else []

            self.strategy = strategy
            if self.hybrid_parallel_topo_order == "pp_first":
                order = ["pp", "dp", "mp"]
                degree = [self.pipeline_model_parallel_size, self.dataset_world_size, self.tensor_model_parallel_size]
            elif self.hybrid_parallel_topo_order == "sharding_first":
                order = ["dp", "pp", "mp"]
                degree = [self.dataset_world_size, self.pipeline_model_parallel_size, self.tensor_model_parallel_size]
            mesh_dims = list(zip(order, degree))
            fleet.auto.create_mesh(mesh_dims)

            # init hcg for communication in trainer
            if self.hybrid_parallel_topo_order == "pp_first":
                order = ["pp", "dp", "sharding", "sep", "mp"]
            elif self.hybrid_parallel_topo_order == "sharding_first":
                order = ["dp", "sharding", "pp", "sep", "mp"]
                if self.expert_model_parallel_size > 1:
                    logger.warning(
                        "Currently using sharding_first topo order, but pp_first is recommended when using experts parallel for performance."
                    )

            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": self.dataset_world_size,
                "mp_degree": self.tensor_model_parallel_size,
                "pp_degree": self.pipeline_model_parallel_size,
                "order": order,
            }
            fleet.init(is_collective=True, strategy=strategy)

        else:
            if world_size > 1:
                if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
                    if self.save_checkpoint_format in [
                        "unified_checkpoint",
                        "flex_checkpoint",
                    ] or self.load_checkpoint_format in ["unified_checkpoint", "flex_checkpoint"]:
                        # DP use hybrid group
                        strategy = fleet.DistributedStrategy()
                        fleet.init(is_collective=True, strategy=strategy)
                    else:
                        paddle.distributed.init_parallel_env()
            if (
                world_size == 1
                and is_paddlefleet_available()
                and get_tensor_model_parallel_group is not None
                and get_tensor_model_parallel_group(False) is None
            ):
                single_card_strategy = fleet.DistributedStrategy()
                single_card_strategy.hybrid_configs = {
                    "dp_degree": 1,
                    "mp_degree": 1,
                    "pp_degree": 1,
                    "sharding_degree": 1,
                    "sep_degree": 1,
                    "cp_degree": 1,
                    "ep_degree": 1,
                    "moe_sharding_degree": 1,
                }
                initialize_fleet(single_card_strategy)

        if (
            self.unified_checkpoint
            and (
                self.save_checkpoint_format == "unified_checkpoint"
                or self.load_checkpoint_format == "unified_checkpoint"
            )
            and ShardingOption.FULL_SHARD in self.sharding
        ):
            logger.warning(
                "Unified checkpoint currently do not support sharding stage3, disabling unified_checkpoint format."
            )
            self.save_checkpoint_format = None
            self.load_checkpoint_format = None

        if self.save_checkpoint_format == "unified_checkpoint" or self.load_checkpoint_format == "unified_checkpoint":
            unified_checkpoint_config = set(self.unified_checkpoint_config.split(" "))
            if sys.platform.startswith("win") and "async_save" in self.unified_checkpoint_config:
                raise ValueError("Currently do not support asynchronous saving for Windows system!")
            if (
                "skip_save_model_weight" in self.unified_checkpoint_config
                and "ignore_merge_optimizer" in self.unified_checkpoint_config
            ):
                raise ValueError("`skip_save_model_weight` and `ignore_merge_optimizer` cannot both be True.")
            for x in unified_checkpoint_config:
                if len(x) > 0:
                    if x not in [
                        "skip_save_model_weight",
                        "master_weight_compatible",
                        "remove_master_weight",
                        "async_save",
                        "enable_all_options",
                        "ignore_merge_optimizer",
                    ]:
                        raise ValueError(
                            f"Found unknown unified_checkpoint config {x}, accept config is skip_save_model_weight, "
                            + "master_weight_compatible, async_save, enable_all_options, ignore_merge_optimizer."
                        )
            if "enable_all_options" in unified_checkpoint_config:
                self.unified_checkpoint_config = [
                    "skip_save_model_weight",
                    "master_weight_compatible",
                    # "async_save",
                ]
            else:
                self.unified_checkpoint_config = self.unified_checkpoint_config.split(" ")

        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "visualdl"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from .integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio during training"
            )

        if self.flatten_param_grads and self.device != "npu":
            raise ValueError("flatten_param_grads can only be used on npu devices in temporary.")

        if self.world_size != paddle.distributed.get_world_size():
            raise ValueError(
                f"The local_ran: {self.local_rank} should be consistent with the world size: {paddle.distributed.get_world_size()}."
            )

        # process fault tolerance settings
        pdc_zcc_init_step = os.getenv("PDC_FC_INIT_STEP")
        if pdc_zcc_init_step is not None and int(pdc_zcc_init_step) > 0:
            self.resume_from_checkpoint = os.path.join(FLASH_DEVICE, f"{PREFIX_CHECKPOINT_DIR}-{pdc_zcc_init_step}")
            logger.warning(
                f"PDC_FC_INIT_STEP {pdc_zcc_init_step} has been specified, automatically resume from FLASH_DEVICE: {self.resume_from_checkpoint}"
            )
        if self.flash_device_save_steps > 0:
            assert (
                self.enable_zero_cost_checkpoint
            ), "flash_device_save_steps should only be set in zero cost checkpoint save mode with flash device mounted."

        if self.enable_zero_cost_checkpoint:
            assert (
                self.fuse_optimizer_states
            ), "zero cost checkpoint must be used when fuse_optimizer_states is enabled in sharding parallel config"

        assert (
            self.flash_device_save_steps % self.zcc_ema_interval == 0
        ), f"flash_device_save_steps[{self.flash_device_save_steps}] must be divisible by zcc_ema_interval[{self.zcc_ema_interval}]"
        assert (
            self.save_steps % self.zcc_ema_interval == 0
        ), f"save_steps[{self.save_steps}] must be divisible by zcc_ema_interval[{self.zcc_ema_interval}]"
        if self.enable_zero_cost_checkpoint and self.zcc_save_ema_coef is not None:
            assert (
                self.zcc_workers_num == 1
            ), "EMA function in zero cost checkpoint mode does not support zcc_workers_num > 1 for now."

        if self.hybrid_parallel_expert_grad_scale is None:
            tensor_model_parallel_size = max(self.tensor_model_parallel_size, 1)
            expert_model_parallel_size = max(self.expert_model_parallel_size, 1)
            context_parallel_size = max(self.context_parallel_size, 1)
            self.hybrid_parallel_expert_grad_scale = (
                tensor_model_parallel_size * context_parallel_size / expert_model_parallel_size
            )
            logger.info(f"Auto set hybrid_parallel_expert_grad_scale = {self.hybrid_parallel_expert_grad_scale}")
        else:
            logger.info(f"Set hybrid_parallel_expert_grad_scale = {self.hybrid_parallel_expert_grad_scale}")

    def _post_init_parallel_degree(self):
        self.use_hybrid_parallel = False

        if isinstance(self.sharding, bool):
            self.sharding = "stage1" if self.sharding else ""
        if isinstance(self.sharding, str):
            self.sharding = [ShardingOption(s) for s in self.sharding.split()]
        if self.sharding == [ShardingOption.OFFLOAD]:
            raise ValueError(
                "`--sharding offload` can't work on its own. It needs to be added to `--sharding stage2` or "
                '`--sharding stage3`. For example, `--sharding "stage2 offload"`.'
            )
        elif len(self.sharding) > (ShardingOption.OFFLOAD in self.sharding) + 1:
            raise ValueError("`--sharding` received too many arguments.")

        if self.sharding_degree > 0:
            warnings.warn("`sharding_degree` is deprecated, please use `sharding_parallel_size`")
            self.sharding_parallel_size = max(self.sharding_degree, self.sharding_parallel_size)
        self.data_parallel_size = 1

        try:
            delattr(self, "sharding_degree")
        except AttributeError:
            pass

        if len(self.sharding) == 0 and self.sharding_parallel_size > 0:
            warnings.warn("`--sharding_parallel_size` is useful only when `--sharding` is specified.")

        world_size = paddle.distributed.get_world_size()

        if world_size > 1:
            tensor_model_parallel_size = max(self.tensor_model_parallel_size, 1)
            sep_parallel_size = max(self.sep_parallel_size, 1)
            context_parallel_size = max(self.context_parallel_size, 1)
            pipeline_model_parallel_size = max(self.pipeline_model_parallel_size, 1)
            expert_model_parallel_size = max(self.expert_model_parallel_size, 1)
            expert_tensor_model_parallel_size = max(self.expert_tensor_model_parallel_size, 1)

            # TODO(@gexiao): support expert_tensor_model_parallel_size > 1 in the future
            assert (
                expert_tensor_model_parallel_size == 1
            ), f"Currently only support expert_tensor_model_parallel_size=1, but got expert_tensor_model_parallel_size of {expert_tensor_model_parallel_size}"

            assert (
                world_size % (self.tensor_model_parallel_size * self.pipeline_model_parallel_size) == 0
            ), f"Total world_size:{world_size} should be divided by tensor_model_parallel_size: {self.tensor_model_parallel_size} and pipeline_model_parallel_size: {self.pipeline_model_parallel_size}."

            assert not (
                sep_parallel_size > 1 and context_parallel_size > 1
            ), f"sep parallel and context parallel cannot be used together, sep_parallel_size:{sep_parallel_size}, context_parallel_size:{context_parallel_size}."

            if self.sharding_parallel_size == -1:
                if len(self.sharding) > 0:
                    self.sharding_parallel_size = world_size // (
                        tensor_model_parallel_size * sep_parallel_size * pipeline_model_parallel_size
                    )

            sharding_parallel_size = max(self.sharding_parallel_size, 1)
            if sharding_parallel_size == 1 and len(self.sharding) > 0:
                logger.warning("sharding_parallel_size=1 means no sharding, please set sharding to empty!")
                self.sharding = []

            self.data_parallel_size = world_size // (
                sharding_parallel_size * tensor_model_parallel_size * sep_parallel_size * pipeline_model_parallel_size
            )

            if expert_model_parallel_size > 1:
                moe_sharding_parallel_size = world_size // (pipeline_model_parallel_size * expert_model_parallel_size)
                assert (
                    self.expert_tensor_model_parallel_size <= 1
                ), "expert_tensor_model_parallel_size > 1 is not supported when expert_model_parallel_size > 1"
            else:
                moe_sharding_parallel_size = 1
            moe_sharding_parallel_size = max(moe_sharding_parallel_size, 1)
            if moe_sharding_parallel_size > 1 and self.data_parallel_size > 1:
                raise NotImplementedError(
                    f"Currently only support use expert_data_parallel strategy together with sharding_parallel strategy, but not with data_parallel strategy. But got data_parallel_size: {self.data_parallel_size}, expert_model_parallel_size: {expert_model_parallel_size}, moe_sharding_parallel_size: {moe_sharding_parallel_size}."
                )

            if sharding_parallel_size > 1 and moe_sharding_parallel_size > 1:
                assert (
                    sharding_parallel_size % moe_sharding_parallel_size == 0
                ), f"sharding_parallel_size should be divided by moe_sharding_parallel_size, current sharding_parallel_size: {sharding_parallel_size}, moe_sharding_parallel_size: {moe_sharding_parallel_size}."

            assert not (
                self.data_parallel_size > 1 and expert_model_parallel_size > 1
            ), f"Currently only support use expert_data_parallel strategy together with sharding_parallel strategy, but not with data_parallel strategy. Currently data_parallel_size is {self.data_parallel_size}."

            if (
                sharding_parallel_size > 1
                or tensor_model_parallel_size > 1
                or pipeline_model_parallel_size > 1
                or self.sep_parallel_size > 1
                or self.context_parallel_size > 1
                or expert_model_parallel_size > 1
                or expert_tensor_model_parallel_size > 1
            ):
                self.use_hybrid_parallel = True
                self.sharding_parallel_size = sharding_parallel_size
                self.tensor_model_parallel_size = tensor_model_parallel_size
                self.pipeline_model_parallel_size = pipeline_model_parallel_size
                self.sep_parallel_size = sep_parallel_size
                self.context_parallel_size = context_parallel_size
                self.expert_model_parallel_size = expert_model_parallel_size
                self.expert_tensor_model_parallel_size = expert_tensor_model_parallel_size
                self.moe_sharding_parallel_size = moe_sharding_parallel_size

            if not self.use_hybrid_parallel:
                self.sharding = []
                self.sharding_parallel_size = -1
                self.tensor_model_parallel_size = -1
                self.pipeline_model_parallel_size = -1
                self.sep_parallel_size = -1
                self.context_parallel_size = -1
                self.expert_model_parallel_size = -1
                self.expert_tensor_model_parallel_size = -1

        if self.hybrid_parallel_topo_order is None:
            self.hybrid_parallel_topo_order = "sharding_first"
        assert self.hybrid_parallel_topo_order in ["pp_first", "sharding_first"]

        if self.use_hybrid_parallel and self.enable_auto_parallel:
            self.use_hybrid_parallel = False

    def _post_init_save_checkpoint_format(self):
        if self.save_checkpoint_format:
            valid_modes = ["unified_checkpoint", "sharding_io", "flex_checkpoint"]
            assert (
                self.save_checkpoint_format in valid_modes
            ), f"Invalid save_checkpoint_format: {self.save_checkpoint_format}, Only these formats are allowed: {valid_modes}."
        else:
            if self.unified_checkpoint:
                self.save_checkpoint_format = "unified_checkpoint"
            elif self.save_sharded_model:
                self.save_checkpoint_format = "sharding_io"

    def _post_init_load_checkpoint_format(self):
        if self.load_checkpoint_format:
            valid_modes = ["unified_checkpoint", "sharding_io", "flex_checkpoint"]
            assert (
                self.load_checkpoint_format in valid_modes
            ), f"Invalid load_checkpoint_format: {self.load_checkpoint_format}, Only these formats are allowed: {valid_modes}."
        else:
            if self.unified_checkpoint:
                self.load_checkpoint_format = "unified_checkpoint"
            elif self.load_sharded_model:
                self.load_checkpoint_format = "sharding_io"

    def add_moe_comm_group(self):
        # NOTE(zhangweilong):move init_moe_group logic to paddle fleet.init
        moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
        moe_grad_group = fleet.get_hybrid_communicate_group().get_moe_sharding_parallel_group()
        hcg = fleet.get_hybrid_communicate_group()
        setattr(hcg, "expert_parallel_group", moe_group)
        setattr(hcg, "expert_grad_comm_group", moe_grad_group)
        return

    def __str__(self):
        self_as_dict = asdict(self)
        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training.
        """
        train_batch_size = self.per_device_train_batch_size
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation.
        """
        eval_batch_size = self.per_device_eval_batch_size
        return eval_batch_size

    @property
    def current_device(self) -> "paddle.device":
        """
        The device used by this process.
        """
        return paddle.device.get_device()

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        if self.local_rank != -1:
            return paddle.distributed.get_world_size()
        return 1

    @property
    def data_parallel_rank(self):
        if self.use_hybrid_parallel:
            hcg = fleet.get_hybrid_communicate_group()
            dp_group = hcg.get_data_parallel_group()
            if dp_group.rank == -1:
                return 0
            return dp_group.rank
        elif self.enable_auto_parallel:
            mesh = fleet.auto.get_mesh()
            return mesh.get_rank_by_dim_and_process_id("dp", dist.get_rank())
        else:
            return paddle.distributed.get_rank()

    @property
    def cp_sharding_degree(self):
        """cp_sharding_degree"""
        assert self.sharding_parallel_size % self.context_parallel_size == 0, (
            f"sharding parallel degree {self.sharding_parallel_size} "
            f"is not divisible by context parallel degree {self.context_parallel_size}"
        )
        hcg = None
        if hasattr(fleet.fleet, "_hcg"):
            hcg = fleet.fleet.get_hybrid_communicate_group()
        if hasattr(hcg, "get_context_parallel_world_size"):
            return hcg.get_sharding_parallel_world_size(with_context_parallel=True)
        else:
            if self.context_parallel_size < 0:
                self.context_parallel_size = 1
            assert self.context_parallel_size == 1, (
                "context_parallel_size > 1 requires 'get_context_parallel_world_size' in hcg. "
                "Please upgrade your PaddlePaddle version."
            )
            return 1

    @property
    def cp_sharding_rank(self):
        """cp_sharding_rank"""
        if self.use_hybrid_parallel:
            hcg = fleet.get_hybrid_communicate_group()
            if hasattr(hcg, "get_context_parallel_world_size") and self.context_parallel_size > 1:
                sharding_rank = hcg.get_sharding_parallel_rank(with_context_parallel=True)
            else:
                sharding_rank = hcg.get_sharding_parallel_rank()
            return max(sharding_rank, 0)
        else:
            return 0

    @property
    def dataset_rank(self):
        if self.use_hybrid_parallel:
            sharding_parallel_size = (
                self.cp_sharding_degree if self.context_parallel_size > 1 else self.sharding_parallel_size
            )
            sharding_parallel_rank = (
                self.cp_sharding_rank if self.context_parallel_size > 1 else self.sharding_parallel_rank
            )
            return max(sharding_parallel_size, 1) * self.data_parallel_rank + sharding_parallel_rank
        elif self.enable_auto_parallel:
            return self.data_parallel_rank
        else:
            return paddle.distributed.get_rank()

    @property
    def dataset_world_size(self):
        if self.use_hybrid_parallel:
            if self.context_parallel_size > 1:
                assert self.use_hybrid_parallel, "context parallel only support with use_hybrid_parallel"
                assert (
                    self.data_parallel_size == 1
                ), f"context parallel can not coexist with data parallel, but got self.data_parallel_size == {self.data_parallel_size}"
                sharding_parallel_size = self.cp_sharding_degree
            else:
                sharding_parallel_size = self.sharding_parallel_size
            return max(sharding_parallel_size, 1) * max(self.data_parallel_size, 1)
        elif self.enable_auto_parallel:
            return max(self.sharding_parallel_size, 1) * max(self.data_parallel_size, 1)
        else:
            return paddle.distributed.get_world_size()

    @property
    def sharding_parallel_rank(self):
        if self.use_hybrid_parallel:
            hcg = fleet.get_hybrid_communicate_group()
            sharding_group = hcg.get_sharding_parallel_group()
            return max(sharding_group.rank, 0)
        else:
            return 0

    @property
    def tensor_parallel_rank(self):
        if self.use_hybrid_parallel:
            hcg = fleet.get_hybrid_communicate_group()
            tp_group = hcg.get_model_parallel_group()
            return max(tp_group.rank, 0)
        elif self.enable_auto_parallel:
            mesh = fleet.auto.get_mesh()
            return mesh.get_rank_by_dim_and_process_id("mp", dist.get_rank())
        else:
            return 0

    @property
    def pipeline_parallel_rank(self):
        if self.use_hybrid_parallel:
            hcg = fleet.get_hybrid_communicate_group()
            rank = hcg.get_stage_id()
            return max(rank, 0)
        elif self.enable_auto_parallel:
            mesh = fleet.auto.get_mesh()
            return mesh.get_rank_by_dim_and_process_id("pp", dist.get_rank())
        else:
            return 0

    @property
    def expert_parallel_rank(self):
        if self.use_hybrid_parallel:
            hcg = fleet.get_hybrid_communicate_group()
            if hasattr(hcg, "get_expert_parallel_rank"):
                return max(hcg.get_expert_parallel_rank(), 0)
            else:
                return 0
        else:
            return 0

    @property
    def moe_sharding_parallel_rank(self):
        if self.use_hybrid_parallel:
            hcg = fleet.get_hybrid_communicate_group()
            if hasattr(hcg, "get_moe_sharding_parallel_group"):
                return max(hcg.get_moe_sharding_parallel_group().rank, 0)
            else:
                return 0
        else:
            return 0

    @property
    def context_parallel_rank(self):
        if self.use_hybrid_parallel:
            hcg = fleet.get_hybrid_communicate_group()
            if hasattr(hcg, "get_context_parallel_rank"):
                return max(hcg.get_context_parallel_rank(), 0)
            else:
                return 0
        else:
            return 0

    def _format_name(self, prefix, rank, degree):
        size = 2
        return f"{prefix}{rank:0>{size}d}"

    @property
    def optimizer_name_suffix(self):
        if self.use_hybrid_parallel:
            name = []
            if self.tensor_model_parallel_size > 1:
                name.append(self._format_name("tp", self.tensor_parallel_rank, self.tensor_model_parallel_size))
            if self.pipeline_model_parallel_size > 1:
                name.append(self._format_name("pp", self.pipeline_parallel_rank, self.pipeline_model_parallel_size))
            if self.sharding_parallel_size > 1:
                name.append(self._format_name("shard", self.sharding_parallel_rank, self.sharding_parallel_size))
            if self.use_expert_parallel and self.expert_model_parallel_size <= 1:
                name.append(self._format_name("moe", self.data_parallel_rank, self.data_parallel_size))
            return "_".join(name)
        else:
            if self.use_expert_parallel:
                return self._format_name("moe", self.data_parallel_rank, self.data_parallel_size)
            return None

    @property
    def weight_name_suffix(self):
        if self.use_hybrid_parallel:
            name = []
            if self.tensor_model_parallel_size > 1:
                name.append(self._format_name("tp", self.tensor_parallel_rank, self.tensor_model_parallel_size))
            if self.pipeline_model_parallel_size > 1:
                name.append(self._format_name("pp", self.pipeline_parallel_rank, self.pipeline_model_parallel_size))
            if self.use_expert_parallel and self.expert_model_parallel_size <= 1:
                name.append(self._format_name("moe", self.data_parallel_rank, self.data_parallel_size))
            if self.use_expert_parallel and self.expert_model_parallel_size > 1:
                name.append(
                    self._format_name("moe_sharding", self.expert_parallel_rank, self.expert_model_parallel_size)
                )
            return "_".join(name)

        else:
            if self.use_expert_parallel:
                return self._format_name("moe", self.data_parallel_rank, self.data_parallel_size)
            return None

    def sharded_name_suffix(self, shard_id=None, pp_id=None, moe_id=None, sharding_parallel_size=None):
        if sharding_parallel_size is None:
            sharding_parallel_size = self.sharding_parallel_size
        if self.use_hybrid_parallel:
            name = []
            if self.tensor_model_parallel_size > 1:
                name.append(self._format_name("tp", self.tensor_parallel_rank, self.tensor_model_parallel_size))
            if self.pipeline_model_parallel_size > 1:
                if pp_id is None:
                    pp_id = self.pipeline_parallel_rank
                assert isinstance(pp_id, int)
                name.append(self._format_name("pp", pp_id, self.pipeline_model_parallel_size))
            if sharding_parallel_size > 1:
                if shard_id is None:
                    shard_id = self.sharding_parallel_rank
                assert isinstance(shard_id, int)
                name.append(self._format_name("shard", shard_id, sharding_parallel_size))
            if self.use_expert_parallel and self.expert_model_parallel_size <= 1:
                if moe_id is None:
                    moe_id = self.data_parallel_rank
                assert isinstance(moe_id, int)
                name.append(self._format_name("moe", moe_id, self.data_parallel_size))
            return "_".join(name)
        else:
            if self.use_expert_parallel:
                if moe_id is None:
                    moe_id = self.data_parallel_rank
                return self._format_name("moe", moe_id, self.data_parallel_size)
            return None

    @property
    def process_index(self):
        """
        The index of the current process used.
        """
        if self.local_rank != -1:
            return paddle.distributed.get_rank()
        return 0

    @property
    def logical_process_index(self):
        """
        The index of the current process used.
        """
        if self.local_rank != -1:
            sd_size = max(self.sharding_parallel_size, 1)
            pp_size = max(self.pipeline_model_parallel_size, 1)
            tp_size = max(self.tensor_model_parallel_size, 1)

            dp_rank = max(self.data_parallel_rank, 0)
            sd_rank = max(self.sharding_parallel_rank, 0)
            pp_rank = max(self.pipeline_parallel_rank, 0)
            tp_rank = max(self.tensor_parallel_rank, 0)

            rank = (
                dp_rank * (sd_size * pp_size * tp_size) + sd_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank
            )

            return rank
        return 0

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        if self.local_rank != -1:
            return self.local_rank
        return 0

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        if self.enable_auto_parallel:
            return True
        elif self.log_on_each_node:
            return self.local_process_index == 0
        else:
            return self.process_index == 0

    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.

        For model state:
            work for data parallel, tensor parallel, sharding
        For optimizer state:
            work for data parallel, tensor parallel
            not work for sharding
        """
        if self.save_on_each_node:
            return self.local_process_index == 0
        else:
            if self.enable_auto_parallel:
                return True
            return self.process_index == 0

    @property
    def should_save_model_state(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.

        For model state:
            work for data parallel, tensor parallel, sharding
        For optimizer state:
            work for data parallel, tensor parallel
            not work for sharding
        """
        if self.save_on_each_node:
            return self.local_process_index == 0
        else:
            if self.should_save_sharding_stage1_model:
                return True
            elif self.enable_auto_parallel:
                return True
            elif self.save_checkpoint_format == "flex_checkpoint":
                return True
            elif self.use_hybrid_parallel:
                # save on dataset rank 0
                return (
                    self.sharding_parallel_rank == 0 and (self.data_parallel_rank == 0 or self.use_expert_parallel)
                ) or (self.expert_model_parallel_size > 1 and self.moe_sharding_parallel_rank == 0)
            else:
                return self.process_index == 0 or self.use_expert_parallel

    @property
    def _no_sync_in_gradient_accumulation(self):
        """
        Whether or not to use no_sync for the gradients when doing gradient accumulation.
        """
        return True

    @property
    def should_save_sharding_stage1_model(self):
        if self.enable_auto_parallel:
            return False
        return (
            ShardingOption.SHARD_OP in self.sharding
            and self.sharding_parallel_size > 1
            and (self.save_checkpoint_format == "sharding_io" or self.save_checkpoint_format == "flex_checkpoint")
        )

    @property
    def should_load_sharding_stage1_model(self):
        if self.enable_auto_parallel:
            return False
        return self.load_sharded_model

    @property
    def should_load_dataset(self):
        if not self.distributed_dataloader:
            return True
        else:
            if self.tensor_parallel_rank == 0 and self.pipeline_parallel_rank == 0:
                return True
            else:
                return False

    def get_auto_dist_flag(self):
        """
        Get the auto distributed flags for auto_parallel intermediate API.
        """
        auto_dist_flag = {
            "tensor_parallel": self.tensor_model_parallel_size > 1,
            "sequence_parallel": self.sequence_parallel,
            "pipeline_parallel": self.pipeline_model_parallel_size > 1,
            "data_sharding_parallel": self.dataset_world_size > 1,
            "sharding": self.sharding,
            "sharding_mesh_dim": self.sharding_parallel_mesh_dimension,
        }
        return auto_dist_flag

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        """
        A context manager for paddle distributed environment where on needs to do something on the main process, while
        blocking replicas, and when it's finished releasing the replicas.

        One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
        which upon completion saves a cached version of results and which then automatically gets loaded by the
        replicas.

        Args:
            local (`bool`, *optional*, defaults to `True`):
                if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
                rank 0 In multi-node environment with a shared filesystem you most likely will want to use
                `local=False` so that only the main process of the first node will do the processing. If however, the
                filesystem is not shared, then the main process of each node will need to do the processing, which is
                the default behavior.
            desc (`str`, *optional*, defaults to `"work"`):
                a work description to be used in debug logs

        """
        if self.world_size > 1:
            if local:
                is_main_process = self.local_process_index == 0
                main_process_desc = "main local process"
            else:
                is_main_process = self.process_index == 0
                main_process_desc = "main process"

            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")
                    paddle.distributed.barrier()
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    paddle.distributed.barrier()
        else:
            yield

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(str(self.to_dict()), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        valid_types.append(paddle.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

    def print_config(self, args=None, key=""):
        """
        print all config values.
        """
        logger.debug("=" * 60)
        if args is None:
            args = self
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

    @property
    def should_save_model_with_tensor_fusion(self):
        return (
            self.enable_auto_parallel
            and self.to_static
            and ShardingOption.SHARD_OP in self.sharding
            and self.sharding_parallel_size > 1
            and self.tensor_fusion
        )
