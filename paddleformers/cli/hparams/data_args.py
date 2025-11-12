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


@dataclass
class DataArguments:
    """Data Argument"""

    # data dir
    dataset_type: str = field(
        default="iterable",
        metadata={
            "help": (
                "Specify the type of dataset to use. Options are 'iterable' "
                "for 'IterableDataset' and 'map' for 'MapDataset'."
            )
        },
    )
    dataset_name: str = field(default="KnowledgeBasedSFTReader", metadata={"help": "."})
    input_dir: str = field(
        default=None,
        metadata={"help": "data path (only valid in offline pretrain dataset)"},
    )
    split: str = field(
        default="950,50",
        metadata={"help": "Train/Eval data split ratio (only valid in offline pretrain dataset)"},
    )
    train_dataset_type: str = field(
        default=None,
        metadata={
            "help": "type of training datasets. \
        Multi-source dataset is supported, e.g., erniekit,erniekit."
        },
    )
    train_dataset_path: str = field(
        default=None,
        metadata={
            "help": "path of training datasets. \
        Multi-source dataset is supported, e.g., ./sft-1.jsonl,./sft-2.jsonl."
        },
    )
    train_dataset_prob: str = field(
        default=None,
        metadata={
            "help": "probabilities of training datasets. \
        Multi-source dataset is supported, e.g., 0.8,0.2."
        },
    )
    text_dataset_path: str = field(default=None, metadata={"help": "sft txt data path"})
    text_dataset_prob: str = field(default=None, metadata={"help": "sft txt data prob"})
    eval_dataset_type: str = field(default="erniekit", metadata={"help": "type of eval datasets."})
    eval_dataset_path: str = field(
        default="examples/data/sft-eval.jsonl",
        metadata={"help": "path of eval datasets."},
    )
    eval_dataset_prob: str = field(
        default="1.0",
        metadata={"help": "probabilities of eval datasets."},
    )
    offline_dataset_path: str = field(
        default=None,
        metadata={
            "help": (
                "If 'dataset_type' is set to 'map', this field is required to "
                "specify the path to the offline dataset."
            )
        },
    )
    max_seq_len: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length."},
    )
    max_prompt_len: int = field(
        default=2048,
        metadata={"help": "Maximum prompt length."},
    )
    num_comparisons: int = field(
        default=6,
        metadata={"help": "Number of candidate responses."},
    )
    mask_out_eos_token: bool = field(
        default=True,
        metadata={"help": "Mask out eos token"},
    )
    random_shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to enable authorize code for privatization. Defaults to False."},
    )
    num_samples_each_epoch: int = field(
        default=6000000,
        metadata={"help": "Number of samples per epoch. Used for SFT."},
    )

    # strategy
    greedy_intokens: bool = field(
        default=True,
        metadata={"help": "Whether to use greedy_intokens packing method."},
    )
    buffer_size: int = field(
        default=500,
        metadata={"help": "Buffer size for greedy_intokens strategy."},
    )
    in_tokens_batching: bool = field(
        default=True,
        metadata={"help": "Whether to using in tokens batching strategy."},
    )
    use_cls: bool = field(
        default=True,
        metadata={"help": "Whether to use cls to predict RM score."},
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Enable sequences packing in training."},
    )
    mix_strategy: str = field(
        default="concat",
        metadata={
            "help": "Strategy to use in dataset mixing (random/concat/interleave) (undersampling/oversampling)."
        },
    )
    encode_one_turn: bool = field(
        default=True,
        metadata={"help": "Whether encode each round independently in a multi-round dialogue."},
    )
    use_template: bool = field(
        default=True,
        metadata={"help": "Whether to use template in data processing."},
    )
    eval_with_do_generation: bool = field(default=False, metadata={"help": "Whether to do generation for evaluation"})
    share_folder: bool = field(
        default=False,
        metadata={"help": "Use share folder for data dir and output dir on multi machine."},
    )

    data_impl: str = field(default="mmap", metadata={"help": "The format of the preprocessed data."})
    skip_warmup: bool = field(
        default=True,
        metadata={"help": "Whether to skip the warmup process of mmap files."},
    )
    data_cache: str = field(default=None, metadata={"help": "The path of the cached dataset."})
