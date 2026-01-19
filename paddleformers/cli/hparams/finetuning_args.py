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
from paddleformers.transformers.configuration_utils import llmmetaclass
from paddleformers.utils.log import logger


@dataclass
class PreTrainingArguments(TrainingArguments):
    """pretraining arguments"""

    eval_iters: int = field(
        default=-1,
        metadata={"help": "eval iteration for every evaluation."},
    )
    use_async_save: Optional[bool] = field(default=False, metadata={"help": "whether enable async save"})
    pre_alloc_memory: float = field(
        default=0.0,
        metadata={
            "help": "Pre-allocate one specific-capacity empty tensor "
            "and release it for avoiding memory fragmentation"
        },
    )
    use_moe: Optional[bool] = field(default=False, metadata={"help": "whether enable moe"})
    enable_mtp_magic_send: Optional[bool] = field(default=False, metadata={"help": ""})
    lr_scheduler: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use. support linear, cosine, constant, constant_with_warmup"},
    )
    freeze_config: str = field(
        default="",
        metadata={
            "help": (
                "Some additional config for freeze params, we provide some option to config it."
                "following config is support: freeze_vision | freeze_llm | freeze_aligner"
            )
        },
    )
    pp_need_data_degree: int = field(
        default=0,
        metadata={"help": "pipline need data degree"},
    )
    pp_need_data: bool = field(default=False, metadata={"help": "pipline need fetch data"})
    balanced_image_preprocess: bool = field(default=False, metadata={"help": "balanced image preprocess"})
    decay_function: str = field(
        default="half_life",
        metadata={"help": "The decay function for WSD LR scheduler. support half_life(default), 1-sqrt"},
    )
    gc_interval: int = field(default=0, metadata={"help": "gc time"})
    global_batch_size: int = field(default=-1, metadata={"help": "global batch size"})
    global_logging_interval: int = field(
        default=1,
        metadata={"help": "the logging interval of global_training_logs"},
    )
    multi_token_pred_depth: Optional[int] = field(
        default=0,
        metadata={},
    )
    num_consecutive: int = field(
        default=1,
        metadata={"help": "H5 file consecutive num."},
    )
    same_data: Optional[bool] = field(
        default=None,
        metadata={"help": "when resume from checkpoint, keey data same with the ckpt"},
    )
    use_ortho_loss_callback: bool = field(default=False, metadata={"help": "Use orthogonal loss callback or not"})
    moe_with_send_router_loss: bool = field(default=True, metadata={"help": "Whether use send router loss"})
    log_global_grad_norm: Optional[bool] = field(
        default=False,
        metadata={"help": "print global grad-norm"},
    )
    moe_gate_lr_ratio: float = field(
        default=None,
        metadata={"help": ("special handle the lr for gate/router")},
    )
    log_global_grad_norm: Optional[bool] = field(
        default=False,
        metadata={"help": "print global grad-norm"},
    )
    shuffle_consecutive: Optional[bool] = field(
        default=False,
        metadata={"help": "shuffle num_consecutive or not"},
    )

    @property
    def need_data(self):
        """
        whether need load data
        return True
        """
        # only mp0、pp0 need data
        if self.pp_need_data_degree:
            assert self.pipeline_model_parallel_size > 1
            assert self.pp_need_data_degree >= 2 and self.pp_need_data_degree <= self.pipeline_model_parallel_size, (
                self.pp_need_data_degree,
                self.pipeline_model_parallel_size,
            )
            # shift by 1 to avoid last pp no nee data
            no_need_data_range = list(range(self.pp_need_data_degree - 1, self.pipeline_model_parallel_size - 1))
            return self.tensor_parallel_rank == 0 and (self.pipeline_parallel_rank not in no_need_data_range)
        return self.pipeline_parallel_rank == 0 and self.tensor_parallel_rank == 0

    @property
    def reeao_dataset_rank(self):
        """
        pp /sharding/ dp sum data stream rank
        """
        if not self.pp_need_data_degree:
            return super().dataset_rank
        no_need_data_range = list(range(self.pp_need_data_degree - 1, self.pipeline_model_parallel_size - 1))
        ranks = [i for i in range(self.pipeline_model_parallel_size) if i not in no_need_data_range]
        if self.pipeline_parallel_rank not in ranks:
            return None
        reeao_pp_rank = ranks.index(self.pipeline_parallel_rank)
        return (
            max(self.sharding_parallel_size, 1) * max(self.pp_need_data_degree, 1) * self.data_parallel_rank
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
        return max(self.sharding_parallel_size, 1) * max(self.pp_need_data_degree, 1) * max(self.data_parallel_size, 1)


@dataclass
class VLSFTTrainingArguments(PreTrainingArguments):
    factor: int = field(default=20, metadata={"help": "Pretrained model name or path to local model."})
    hidden_dropout_prob: float = field(default=0.0, metadata={"help": "hidden dropout rate"})
    moe_dropout_prob: float = field(default=0.0, metadata={"help": "moe dropout rate"})
    token_balance_loss: bool = field(default=False, metadata={"help": "use token_loss_equal_weight or not."})


@dataclass
class SFTTrainingArguments(TrainingArguments):
    """SFT Training Arguments"""

    max_estimate_samples: int = field(
        default=1e5,
        metadata={"help": "Maximum number of samples used in estimation."},
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
):
    """Finetuning Argument"""

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    # base
    decay_steps: int = field(
        default=None,
        metadata={
            "help": "The steps use to control the learing rate. If the step > decay_steps, "
            "will use the min_learning_rate."
        },
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
    multi_token_pred_lambda: float = field(default=0.3, metadata={"help": "multi token pred lambda"})
    use_recompute_mtp: bool = field(default=False, metadata={"help": "Whether to use recompute_mtp"})

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

        super().__post_init__()

        self.global_batch_size = (
            self.per_device_train_batch_size * self.dataset_world_size * self.gradient_accumulation_steps
        )
        logger.info(f"reset finetuning arguments global_batch_size to {self.global_batch_size}")

        self.max_gradient_accumulation_steps = self.gradient_accumulation_steps

        if self.pipeline_model_parallel_size > 1:
            # self.per_device_eval_batch_size = self.per_device_train_batch_size * self.gradient_accumulation_steps
            # logger.warning(f"eval_batch_size set to {self.per_device_eval_batch_size} in Pipeline Parallel!")
            user_defined_strategy = fleet.fleet._user_defined_strategy
            user_defined_strategy.strategy.pipeline_configs.accumulate_steps = self.gradient_accumulation_steps
            self.max_gradient_accumulation_steps = self.gradient_accumulation_steps
            logger.info(f"fixing pp configs: {user_defined_strategy.pipeline_configs}")
        # else:
        #     self.per_device_eval_batch_size = self.per_device_train_batch_size
        #     logger.warning(f"eval_batch_size set to {self.per_device_eval_batch_size}")

        if self.sharding_parallel_size > 1:
            sharding_comm_overlap_non_pp = (
                True if self.sd_shardingv1_comm_overlap or self.sd_sharding_comm_overlap else False
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
