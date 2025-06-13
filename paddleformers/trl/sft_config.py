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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..trainer import TrainingArguments
from ..trainer.trainer_utils import IntervalStrategy
from ..trainer.utils.doc import add_start_docstrings
from ..transformers.configuration_utils import llmmetaclass

__all__ = ["SFTConfig"]


@dataclass
@llmmetaclass
@add_start_docstrings(TrainingArguments.__doc__)
class SFTConfig(TrainingArguments):
    benchmark: bool = field(default=False, metadata={"help": "Whether runs benchmark"})
    # NOTE(gongenlei): new add autotuner_benchmark
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Weather to run benchmark by autotuner. True for from_scratch and pad_max_length."},
    )
    decay_steps: int = field(
        default=0,
        metadata={"help": "The steps use to control the learing rate."},
    )
    tensor_parallel_output: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to output logits in distributed status"},
    )
    unified_checkpoint: bool = field(
        default=False,
        metadata={"help": "Unify hybrid parallel checkpoint."},
    )
    unified_checkpoint_config: Optional[str] = field(
        default="",
        metadata={"help": "Configs to unify hybrid parallel checkpoint.\n"},
    )
    dataset_text_field: str = "text"
    learning_rate: float = 2.0e-5
    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": "The maximum length that model input tokens can have. When Zero Padding is set to True, it's also the maximum length for Zero Padding data stream"
        },
    )
    dataset_num_proc: Optional[int] = None
    dataset_batch_size: int = 1000
    model_init_kwargs: Optional[dict[str, Any]] = None
    dataset_kwargs: Optional[dict[str, Any]] = None
    eval_packing: Optional[bool] = None
    use_ssa: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Shifted Sparse Attention (SSA), an efficient attention mechanism introduced in the LongLoRA paper."
        },
    )
    ssa_group_size_ratio: float = field(
        default=0.25,
        metadata={
            "help": "The ratio parameter for grouping in SSA, controlling the number of tokens considered in each group for sparse attention calculation."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # NOTE(gongenlei): new add autotuner_benchmark
        if self.autotuner_benchmark:
            self.max_steps = 5
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.report_to = []
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO
        if self.benchmark:
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.report_to = []
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO
