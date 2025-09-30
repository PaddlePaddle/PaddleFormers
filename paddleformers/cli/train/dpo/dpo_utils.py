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

"""DPO utils"""
from dataclasses import dataclass, field


@dataclass
class DPOConfig:
    """DPOConfig"""

    beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    offset_alpha: float = field(default=0.0, metadata={"help": "the offset coefficient for score-based DPO loss"})
    simpo_gamma: float = field(default=0.5, metadata={"help": "the gamma parameter for SimPO loss"})
    normalize_logps: bool = field(
        default=True,
        metadata={"help": "Apply logprobs normalization."},
    )
    label_smoothing: float = field(default=0.0, metadata={"help": "label_smoothing ratio"})
    loss_type: str = field(default="sigmoid", metadata={"help": "DPO loss type"})
    pref_loss_ratio: float = field(default=1.0, metadata={"help": "DPO loss ratio"})
    sft_loss_ratio: float = field(default=0.0, metadata={"help": "SFT loss ratio"})
    dpop_lambda: float = field(default=50, metadata={"help": "dpop_lambda"})
    ref_model_update_steps: int = field(default=-1, metadata={"help": "Update ref model state dict "})
    reference_free: bool = field(default=False, metadata={"help": "No reference model."})
    lora: bool = field(default=False, metadata={"help": "Use LoRA model."})

    def __post_init__(self):
        if self.offset_alpha > 0.0:
            if self.loss_type != "sigmoid":
                raise ValueError(
                    "Only sigmoid loss_type supports score-based loss (offset_alpha > 0), "
                    "please set loss_type to sigmoid or set offset_alpha to 0."
                )


def calculate_effective_tokens(training_args, train_dataset, max_seq_len):
    """
    Caculate the effective tokens during training.

    Args:
        training_args (TrainingArguments): Configuration object containing:
            - data_parallel_degree (int): Number of data parallel partitions
            - sharding_parallel_degree (int): Number of sharding partitions
            - max_steps (int): Total training iterations
            - per_device_train_batch_size (int): Batch size per GPU/device
            - gradient_accumulation_steps (int): Grad accumulation steps
        train_dataset (IterableDataset): Training dataset with input_ids fields
        max_seq_len (int): Padded sequence length

    Returns:
        tuple: (effective_tokens, total_possible_tokens) where:
            - effective_tokens (int): Actual processed tokens (excludes padding)
            - total_possible_tokens (int): Theoretical maximum (batch_size * seq_len)
    """
    total_effective_tokens = 0
    try:
        data_parallel_degree = training_args.data_parallel_degree
    except:
        data_parallel_degree = 1
    if training_args.sharding_parallel_degree > 1:
        sharding_parallel_degree = training_args.sharding_parallel_degree
    else:
        sharding_parallel_degree = 1

    total_batch = (
        training_args.max_steps
        * training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * sharding_parallel_degree
        * data_parallel_degree
    )
    for i, data in enumerate(train_dataset):
        if i == total_batch:
            break
        for dd in data:
            total_effective_tokens += len(dd.input_ids)
    total_tokens = total_batch * max_seq_len

    return total_effective_tokens, total_tokens
