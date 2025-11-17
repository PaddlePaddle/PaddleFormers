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


from dataclasses import dataclass, field
from typing import Optional

from paddleformers.trainer import TrainingArguments
from paddleformers.trainer.trainer_utils import IntervalStrategy
from paddleformers.trainer.utils.doc import add_start_docstrings
from paddleformers.transformers.configuration_utils import llmmetaclass
from paddleformers.trl import DataConfig


@dataclass
@llmmetaclass
@add_start_docstrings(TrainingArguments.__doc__)
class DPOTrainingArguments(TrainingArguments):
    """DPOTrainingArguments"""

    num_of_gpus: int = field(
        default=-1,
        metadata={"help": "Number of gpus used in dpo estimate training."},
    )
    unified_checkpoint: bool = field(
        default=True,
        metadata={"help": "Enable fused linear grad add strategy."},
    )
    unified_checkpoint_config: Optional[str] = field(
        default="",
        metadata={"help": "Configs to unify hybrid parallel checkpoint.\n"},
    )
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to run benchmark by autotuner. True for from_scratch."},
    )
    use_intermediate_api: bool = field(
        default=False,
        metadata={"help": "Flag indicating whether to use the intermediate API for model."},
    )
    num_hidden_layers: int = field(default=2, metadata={"help": "The number of hidden layers in the network model."})

    def __post_init__(self):
        super().__post_init__()
        if self.autotuner_benchmark:
            self.num_train_epochs = 1
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
            if not self.disable_tqdm:
                self.logging_steps = 1
                self.logging_strategy = IntervalStrategy.STEPS
        if self.benchmark:
            self.do_train = True
            self.do_export = False
            self.do_predict = False
            self.do_eval = False
            self.overwrite_output_dir = True
            self.load_best_model_at_end = False
            self.save_strategy = IntervalStrategy.NO
            self.evaluation_strategy = IntervalStrategy.NO
            if not self.disable_tqdm:
                self.logging_steps = 1
                self.logging_strategy = IntervalStrategy.STEPS
        if self.max_steps > 0:
            self.num_train_epochs = 1


@dataclass
class DPOConfig:
    """DPOConfig"""

    beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    simpo_gamma: float = field(default=0.5, metadata={"help": "the gamma parameter for SimPO loss"})
    label_smoothing: float = field(default=0.0, metadata={"help": "label_smoothing ratio"})
    loss_type: str = field(default="sigmoid", metadata={"help": "DPO loss type"})
    pref_loss_ratio: float = field(default=1.0, metadata={"help": "DPO loss ratio"})
    sft_loss_ratio: float = field(default=0.0, metadata={"help": "SFT loss ratio"})
    dpop_lambda: float = field(default=50, metadata={"help": "dpop_lambda"})
    ref_model_update_steps: int = field(default=-1, metadata={"help": "Update ref model state dict "})
    reference_free: bool = field(default=False, metadata={"help": "No reference model."})
    lora: bool = field(default=False, metadata={"help": "Use LoRA model."})
    offset_alpha: float = field(default=0.0, metadata={"help": "offset alpha"})
    normalize_logps: bool = field(default=False, metadata={"help": "normalize logps"})
    ignore_eos_token: bool = field(default=False, metadata={"help": "ignore eos token"})


@dataclass
class DPODataArgument(DataConfig):
    """DataArgument"""

    max_seq_len: int = field(default=4096, metadata={"help": "Maximum sequence length."})
    max_prompt_len: int = field(default=2048, metadata={"help": "Maximum prompt length."})
    num_samples_each_epoch: int = field(default=6000000, metadata={"help": "Number of sample per training epoch."})
    buffer_size: int = field(default=1000, metadata={"help": "Preloading buffer capacity."})
    mask_out_eos_token: bool = field(default=True, metadata={"help": "EOS loss masking."})


@dataclass
class DPOModelArgument:
    """ModelArgument"""

    model_name_or_path: str = field(
        default=None, metadata={"help": "Pretrained model name or path to local directory."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    download_hub: str = field(
        default="aistudio",
        metadata={
            "help": "The source for model downloading, options include `huggingface`, `aistudio`, `modelscope`, default `aistudio`"
        },
    )
    flash_mask: bool = field(default=False, metadata={"help": "Whether to use flash mask in flash attention."})
    weight_quantize_algo: str = field(
        default=None,
        metadata={"help": "Model weight quantization algorithm including 'nf4'(qlora), 'weight_only_int8'."},
    )
    fuse_attention_qkv: bool = field(
        default=None,
        metadata={"help": "whether to fuse attention qkv"},
    )
    fuse_attention_ffn: bool = field(
        default=None,
        metadata={"help": "whether to fuse first up and gate proj in mlp block"},
    )
    use_attn_mask_startend_row_indices: bool = field(
        default=True,
        metadata={"help": "Sparse attention mode."},
    )

    # LoRA
    lora_rank: int = field(default=8, metadata={"help": "Lora rank."})
    lora_path: str = field(default=None, metadata={"help": "Initialize lora state dict."})
    rslora: bool = field(default=False, metadata={"help": "Whether to use RsLoRA"})
    lora_plus_scale: float = field(default=1.0, metadata={"help": "Lora B scale in LoRA+ technique"})
    lora_alpha: int = field(default=-1, metadata={"help": "lora_alpha"})
    rslora_plus: bool = field(default=False, metadata={"help": "Strengthen lora performance"})
    use_quick_lora: bool = field(default=True, metadata={"help": "quick lora"})

    # Attention
    attn_impl: str = field(default="flashmask", metadata={"help": "Attention implementation"})
