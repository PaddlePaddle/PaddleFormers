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

from dataclasses import dataclass, field
from typing import Any, Optional

from paddle.distributed import fleet

from paddleformers.trainer import TrainingArguments
from paddleformers.trainer.trainer_utils import ShardingOption
from paddleformers.transformers.configuration_utils import llmmetaclass
from paddleformers.utils.log import logger

try:
    from paddle.distributed import in_auto_parallel_align_mode
except Exception:

    def in_auto_parallel_align_mode():
        """
        hack for paddle develop branch.
        """
        return False


@dataclass
class CheckPointArguments(TrainingArguments):
    """CheckPoint Arguments"""

    disable_ckpt_quant: bool = field(
        default=False,
        metadata={"help": "Whether disable checkpoint quantization."},
    )


@dataclass
class DistributedArguments(TrainingArguments):
    use_sp_callback: bool = field(
        default=False,
        metadata={
            "help": "Using the SP callback will skip the implementation of SPHook "
            "to avoid redundant gradient computation."
        },
    )
    # server deploy
    server_tp_degree: int = field(
        default=1,
        metadata={"help": "Tensor parallelism degree use for server deploy"},
    )


@dataclass
class PreTrainingArguments(TrainingArguments):
    """pretraining arguments"""

    multimodal: bool = field(default=False, metadata={"help": "whether training with multimodal"})
    vision_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from "
            "https://paddleformers.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "global random seed factor."},
    )
    eval_iters: int = field(
        default=-1,
        metadata={"help": "eval iteration for every evaluation."},
    )
    train_emb_only: int = field(
        default=0,
        metadata={"help": "train emb only flag"},
    )
    data_filelist: tuple = field(default=None, metadata={"help": "data file list"})
    data_weights: tuple = field(default=None, metadata={"help": "data weights"})
    from_scratch: Optional[int] = field(default=1, metadata={"help": "if set, ignore init_ckpt"})
    record_optimizer_stat: Optional[bool] = field(
        default=False, metadata={"help": "whether record optimizer momentum info"}
    )
    same_data: Optional[bool] = field(default=None, metadata={"help": "whether keep the same data with previous run"})
    adaptive_norm_clip: Optional[bool] = field(default=False, metadata={"help": "whether enable AdaptiveNormClip"})
    use_async_save: Optional[bool] = field(default=False, metadata={"help": "whether enable async save"})
    pre_alloc_memory: float = field(
        default=0.0,
        metadata={
            "help": "Pre-allocate one specific-capacity empty tensor "
            "and release it for avoiding memory fragmentation"
        },
    )
    enable_global_training_logs: bool = field(default=False, metadata={"help": "whether enable global_training_logs"})
    reshard_save_then_exit: Optional[bool] = field(default=False, metadata={"help": "whether reshard save then exit"})
    use_moe: Optional[bool] = field(default=False, metadata={"help": "whether enable moe"})
    log_global_grad_norm: Optional[bool] = field(
        default=False,
        metadata={"help": "whether print global grad-norm, only valid when `enable_global_training_logs` is True"},
    )
    enable_mtp_magic_send: Optional[bool] = field(default=False, metadata={"help": ""})
    lr_scheduler: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use. support linear, cosine, constant, constant_with_warmup"},
    )
    decay_function: str = field(
        default="half_life",
        metadata={"help": "The decay function for WSD LR scheduler. support half_life(default), 1-sqrt"},
    )
    freeze_config: str = field(
        default="",
        metadata={
            "help": (
                "Some additional config for freeze params, we provide some option to config it."
                "following config is support: freeze_vision,freeze_lm"
            )
        },
    )
    moe_gate_lr_ratio: float = field(
        default=None,
        metadata={"help": ("when using MoE, we need a special way to handle gate/router's LR")},
    )
    vit_lr_ratio: float = field(
        default=None,
        metadata={"help": ("when use vit, we need a special way to handle vit's LR")},
    )
    visual_ld: float = field(
        default=None,
        metadata={"help": ("when use vit, we need a special way to handle vit's LR")},
    )
    modality_interleave: str = field(default="acc", metadata={"help": "acc"})
    modality_ratio: tuple = field(
        default=None,
        metadata={"help": "ratio of modality tokens to be masked out"},
    )
    pp_need_data_degree: int = field(
        default=0,
        metadata={"help": "pipline need data degree"},
    )
    pp_need_data: bool = field(default=False, metadata={"help": "pipline need fetch data"})
    balanced_image_preprocess: bool = field(default=False, metadata={"help": "balanced image preprocess"})
    gc_interval: int = field(default=0, metadata={"help": "gc interval"})
    vit_second_fwd_batch_size: int = field(default=None, metadata={"help": "vit second forward batch size"})
    moe_use_aux_free_update_coef: float = field(
        default=1.0e-3,
        metadata={"help": "moe aux free update coef"},
    )
    disable_pipeline_warmup: bool = field(
        default=False,
        metadata={"help": "whether to disable pipeline warmup"},
    )
    global_logging_interval: int = field(
        default=1,
        metadata={"help": "the logging interval of global_training_logs"},
    )
    train_moe_only: int = field(default=None, metadata={"help": "train moe params only"})
    use_ortho_loss_callback: bool = field(default=False, metadata={"help": "whether use ortho loss callback"})

    @property
    def need_data(self):
        """
        whether need load data
        return True
        """
        # only mp0、pp0 need data
        if self.pp_need_data_degree:
            assert self.pipeline_parallel_degree > 1
            assert self.pp_need_data_degree >= 2 and self.pp_need_data_degree <= self.pipeline_parallel_degree, (
                self.pp_need_data_degree,
                self.pipeline_parallel_degree,
            )
            # shift by 1 to avoid last pp no nee data
            no_need_data_range = list(range(self.pp_need_data_degree - 1, self.pipeline_parallel_degree - 1))
            return self.tensor_parallel_rank == 0 and (self.pipeline_parallel_rank not in no_need_data_range)
        return self.pipeline_parallel_rank == 0 and self.tensor_parallel_rank == 0

    @property
    def reeao_dataset_rank(self):
        """
        pp /sharding/ dp sum data stream rank
        """
        if not self.pp_need_data_degree:
            return super().dataset_rank
        no_need_data_range = list(range(self.pp_need_data_degree - 1, self.pipeline_parallel_degree - 1))
        ranks = [i for i in range(self.pipeline_parallel_degree) if i not in no_need_data_range]
        if self.pipeline_parallel_rank not in ranks:
            return None
        reeao_pp_rank = ranks.index(self.pipeline_parallel_rank)
        return (
            max(self.sharding_parallel_degree, 1) * max(self.pp_need_data_degree, 1) * self.data_parallel_rank
            + max(self.pp_need_data_degree, 1) * self.sharding_parallel_rank
            + reeao_pp_rank
        )

    @property
    def reeao_dataset_world_size(self):
        """
        pp /sharding/ dp sum data stream worldsize
        """
        if not self.pp_need_data_degree:
            return super().dataset_world_size
        return (
            max(self.sharding_parallel_degree, 1)
            * max(self.pp_need_data_degree, 1)
            * max(self.data_parallel_degree, 1)
        )


@dataclass
class VLSFTTrainingArguments(PreTrainingArguments):
    factor: int = field(default=20, metadata={"help": "Pretrained model name or path to local model."})
    example_from_same_task_prob: float = field(default=0.0, metadata={"help": "."})
    pseudo_sampling_prob: float = field(default=0.5, metadata={"help": "."})
    trigger_data_prob: float = field(default=0.5, metadata={"help": "."})
    drop_history_with_k: bool = field(default=False, metadata={"help": "drop history"})
    add_sys_token: bool = field(default=False, metadata={"help": "use <sys> </sys> tokens segment system info"})
    min_shot: int = field(default=2, metadata={"help": "min shot"})
    max_shot: int = field(default=8, metadata={"help": "max shot"})

    sampling_wo_replacement_data_resuming: Optional[bool] = field(
        default=True,
        metadata={"help": "save and load state of SFT data, support resuming without replacement"},
    )
    hidden_dropout_prob: float = field(default=0.0, metadata={"help": "hidden dropout rate"})
    moe_dropout_prob: float = field(default=0.0, metadata={"help": "moe dropout rate"})
    token_balance_loss: bool = field(default=False, metadata={"help": "use token_loss_equal_weight or not."})
    use_train_part_sharding: Optional[bool] = field(default=True, metadata={"help": "use_train_part_sharding"})
    text_use_train_part_sharding: Optional[bool] = field(
        default=True, metadata={"help": "text dataset use_train_part_sharding"}
    )


@dataclass
class SFTTrainingArguments(TrainingArguments):
    """SFT Training Arguments"""

    max_estimate_samples: int = field(
        default=1e5,
        metadata={"help": "Maximum number of samples used in estimation."},
    )
    sft_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to calculate effective token per second"},
    )


@dataclass
class DPOTrainingArguments(TrainingArguments):
    """DPOTrainingArguments"""

    # dpo estimate parameters
    num_of_gpus: int = field(
        default=-1,
        metadata={"help": "Number of gpus used in dpo estimate training."},
    )
    # base
    normalize_logps: bool = field(
        default=False,
        metadata={"help": "Apply logprobs normalization."},
    )
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "label_smoothing ratio"},
    )
    dpo_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )
    ignore_eos_token: bool = field(
        default=False,
        metadata={"help": "Ignore EOS token during training."},
    )
    # reference model
    ref_model_update_steps: int = field(
        default=-1,
        metadata={"help": "Update ref model state dict "},
    )
    reference_free: bool = field(
        default=False,
        metadata={"help": "No reference model."},
    )
    # dpo loss
    loss_type: str = field(
        default="sigmoid",
        metadata={"help": "DPO loss type"},
    )
    pref_loss_ratio: float = field(
        default=1.0,
        metadata={"help": "DPO loss ratio"},
    )
    sft_loss_ratio: float = field(
        default=0.0,
        metadata={"help": "SFT loss ratio"},
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "the beta parameter for DPO loss"},
    )
    offset_alpha: float = field(
        default=0.0,
        metadata={"help": "the offset coefficient for score-based DPO loss"},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "the gamma parameter for SimPO loss"},
    )
    dpop_lambda: float = field(
        default=50,
        metadata={"help": "dpop_lambda"},
    )


@dataclass
@llmmetaclass
class FinetuningArguments(
    SFTTrainingArguments,
    VLSFTTrainingArguments,
    DPOTrainingArguments,
    CheckPointArguments,
    DistributedArguments,
):
    """Finetuning Argument"""

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    # base
    batch_size: int = field(default=1, metadata={"help": "Batch size per GPU."})
    layerwise_lr_decay_bound: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Use a large learning rate for the top layers and "
            "a small learning rate for the bottom layers. 1.0: Do not use this strategy."
        },
    )
    decay_steps: int = field(
        default=None,
        metadata={
            "help": "The steps use to control the learing rate. If the step > decay_steps, "
            "will use the min_learning_rate."
        },
    )
    dropout_warmup_steps: int = field(
        default=0,
        metadata={"help": "dropout warmup steps"},
    )
    hidden_dropout_prob: float = field(
        default=0.0,
        metadata={"help": "dropout probability for hidden layers"},
    )
    attention_probs_dropout_prob: float = field(
        default=0.0,
        metadata={"help": "dropout probability for attention layers"},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )

    # performance
    compute_type: str = field(
        default="bf16",
        metadata={"help": "The compute type."},
    )
    weight_quantize_algo: str = field(
        default=None,
        metadata={"help": "Model weight quantization algorithm including 'nf4'(qlora), 'weight_only_int8'."},
    )

    # fp8
    use_fp8: bool = field(
        default=False,
        metadata={"help": "whether to use fp8 training"},
    )
    fp8_force_clear_state: bool = field(
        default=False,
        metadata={"help": "whether to force clear TE FP8 amax state when resume"},
    )
    enable_fp8_quantize_analysis: bool = field(
        default=False,
        metadata={"help": "whether to enable FP8 quantize analysis"},
    )
    apply_hadamard: bool = field(default=True, metadata={"help": "Whether to apply hadamard"})
    hadamard_block_size: int = field(default=32, metadata={"help": "hadamard block size"})
    quant_input_grad: bool = field(default=False, metadata={"help": "Whether to quantize input grad"})
    quant_weight_grad: bool = field(default=False, metadata={"help": "Whether to quantize weight grad"})
    apply_online_actscale_step: int = field(
        default=200,
        metadata={"help": "Use online activation scale for first N step to keep stable training."},
    )
    actscale_moving_rate: float = field(default=0.01, metadata={"help": "EMA moving_rate for activation scale"})
    fp8_format_type: str = field(default="hybrid", metadata={"help": "FP8 Format"})
    multi_token_pred_lambda: float = field(default=0.3, metadata={"help": "multi token pred lambda"})
    use_recompute_mtp: bool = field(default=False, metadata={"help": "Whether to use recompute_mtp"})

    # training pytorch models from huggingFace
    use_huggingface_model: bool = field(
        default=False,
        metadata={"help": "Whether to use huggingface model to finetune."},
    )

    # NOTE(gongenlei): new add autotuner_benchmark
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Weather to run benchmark by autotuner. True for from_scratch and pad_max_length."},
    )
    dataset_num_proc: Optional[int] = None
    dataset_batch_size: int = 1000
    dataset_kwargs: Optional[dict[str, Any]] = None
    dataset_text_field: str = "text"

    enable_linear_fused_grad_add: bool = field(
        default=False,
        metadata={
            "help": "Enable fused linear grad add strategy, which will reduce elementwise add for grad accumulation in the backward of nn.Linear ."
        },
    )

    def __post_init__(self):
        self.bf16 = True
        if self.compute_type == "bf16":
            self.fp16 = False
            self.weight_quantize_algo = None
        elif self.compute_type == "fp16":
            self.bf16 = False
            self.fp16 = True
            self.weight_quantize_algo = None
        elif self.compute_type == "fp8":
            self.weight_quantize_algo = "fp8linear"
            self.apply_hadamard = True
            self.optim = "adamw_custom"
            self.use_lowprecision_moment = True
            self.tensorwise_offload_optimizer = True
            self.optim_shard_num = 8
            self.unified_checkpoint_config = "ignore_merge_optimizer"
        elif self.compute_type == "wint8":
            self.weight_quantize_algo = "weight_only_int8"
        elif self.compute_type == "wint4/8":
            self.weight_quantize_algo = "weight_only_mix"
        elif self.compute_type == "nf4":
            self.weight_quantize_algo = "nf4"
        else:
            raise ValueError(f"Unknown compute_type: {self.compute_type}")
        self.per_device_train_batch_size = self.batch_size
        self.per_device_eval_batch_size = self.batch_size
        self.server_tp_degree = self.tensor_parallel_degree

        super().__post_init__()

        # ERNIE VL model post init
        if in_auto_parallel_align_mode():
            self.adaptive_norm_clip = False

        self.global_batch_size = (
            self.per_device_train_batch_size * self.dataset_world_size * self.gradient_accumulation_steps
        )
        logger.info(f"reset finetuning arguments global_batch_size to {self.global_batch_size}")

        self.max_gradient_accumulation_steps = self.gradient_accumulation_steps

        if self.pipeline_parallel_degree > 1:
            self.per_device_eval_batch_size = self.per_device_train_batch_size * self.gradient_accumulation_steps
            logger.warning(f"eval_batch_size set to {self.per_device_eval_batch_size} in Pipeline Parallel!")
            user_defined_strategy = fleet.fleet._user_defined_strategy
            user_defined_strategy.strategy.pipeline_configs.accumulate_steps = self.gradient_accumulation_steps
            if self.pp_need_data and not self.pp_need_data_degree:
                self.pp_need_data_degree = self.pipeline_parallel_degree
            if self.pp_need_data_degree:
                assert self.gradient_accumulation_steps % self.pp_need_data_degree == 0, (
                    f"gradient_accumulation_steps[{self.gradient_accumulation_steps}] should be divisible by "
                    f"pp_need_data_degree[{self.pp_need_data_degree}]"
                )

                self.gradient_accumulation_steps = self.gradient_accumulation_steps // self.pp_need_data_degree
                logger.info(
                    f"pp-need-data hack args.gradient_accumulation_steps to - {self.gradient_accumulation_steps}"
                )
            self.max_gradient_accumulation_steps = self.gradient_accumulation_steps
            logger.info(f"fixing pp configs: {user_defined_strategy.pipeline_configs}")
        else:
            self.per_device_eval_batch_size = self.per_device_train_batch_size
            logger.warning(f"eval_batch_size set to {self.per_device_eval_batch_size}")

        if self.sharding_parallel_degree > 1:
            sharding_parallel_config = (
                set(self.sharding_parallel_config.split(" ")) if self.sharding_parallel_config else set()
            )
            sharding_comm_overlap_non_pp = (
                True
                if "shardingv1_comm_overlap" in sharding_parallel_config
                or "sharding_comm_overlap" in sharding_parallel_config
                else False
            )
            if sharding_comm_overlap_non_pp:
                assert hasattr(fleet.fleet, "_user_defined_strategy")
                user_defined_strategy = fleet.fleet._user_defined_strategy
                user_defined_strategy.hybrid_configs[
                    "sharding_configs"
                ].accumulate_steps = self.gradient_accumulation_steps

        if hasattr(fleet.fleet, "_user_defined_strategy"):
            user_defined_strategy = fleet.fleet._user_defined_strategy
            if (
                hasattr(user_defined_strategy, "hybrid_configs")
                and "sharding_configs" in user_defined_strategy.hybrid_configs
            ):
                sd_configs = user_defined_strategy.hybrid_configs["sharding_configs"]
                if sd_configs.comm_overlap:
                    assert self.global_batch_size % self.dataset_world_size == 0, (
                        f"global_batch_size[{self.global_batch_size}] should be divisible by "
                        f"dataset_world_size[{self.dataset_world_size}]"
                    )
                    lbs = self.global_batch_size // self.dataset_world_size
                    assert lbs % self.per_device_train_batch_size == 0, (
                        f"local_batch_size[{lbs}] should be divisible by "
                        f"per_device_train_batch_size[{self.per_device_train_batch_size}]"
                    )
                    assert lbs // self.per_device_train_batch_size == sd_configs.accumulate_steps, (
                        f"local_batch_size[{lbs}] should be equal to "
                        f"accumulate_steps[{sd_configs.accumulate_steps}] * "
                        f"per_device_train_batch_size[{self.per_device_train_batch_size}]"
                    )
        if self.vision_model_name_or_path is not None:
            self.multimodal = True
        if self.visual_ld and not self.vit_lr_ratio:
            self.vit_lr_ratio = self.visual_ld

        if ShardingOption.SHARD_GRAD_OP in self.sharding:
            logger.info("disabling `sp_callback` b/c using sharding stage2")
            self.use_sp_callback = False
