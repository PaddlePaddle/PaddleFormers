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

__all__ = ["DataConfig"]


@dataclass
class DataConfig:

    dataset_name_or_path: str = field(default=None, metadata={"help": "Name or path for dataset"})
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
    eval_dataset_type: str = field(default="erniekit", metadata={"help": "type of eval datasets."})
    eval_dataset_path: str = field(
        default="examples/data/sft-eval.jsonl",
        metadata={"help": "path of eval datasets."},
    )
    eval_dataset_prob: str = field(
        default="1.0",
        metadata={"help": "probabilities of eval datasets."},
    )
    dataset_type: str = field(
        default="iterable",
        metadata={
            "help": (
                "Specify the type of dataset to use. Options are 'iterable' "
                "for 'IterableDataset' and 'map' for 'MapDataset'."
            )
        },
    )
    input_dir: str = field(
        default=None,
        metadata={"help": "data path (only valid in offline pretrain dataset)"},
    )
    split: str = field(
        default="950,50",
        metadata={"help": "Train/Eval data split ratio (only valid in offline pretrain dataset)"},
    )
    mix_strategy: str = field(
        default="concat",
        metadata={
            "help": "Strategy to use in dataset mixing (random/concat/interleave) (undersampling/oversampling)."
        },
    )
    use_template: bool = field(
        default=True,
        metadata={"help": "Whether to use template in data processing."},
    )
    encode_one_turn: bool = field(
        default=True,
        metadata={"help": "Whether encode each round independently in a multi-round dialogue."},
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Enable sequences packing in training."},
    )
    greedy_intokens: bool = field(
        default=True,
        metadata={"help": "Whether to use greedy_intokens packing method."},
    )
    random_shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to enable authorize code for privatization. Defaults to False."},
    )
    num_samples_each_epoch: int = field(
        default=6000000,
        metadata={"help": "Number of samples per epoch. Used for SFT."},
    )
    task_name: str = field(default=None, metadata={"help": "Additional name to select a more specific task."})
    pad_to_multiple_of: int = field(
        default=None, metadata={"help": "If set will pad the sequence to a multiple of the provided value."}
    )
    eval_with_do_generation: bool = field(default=False, metadata={"help": "Whether to do generation for evaluation"})
    save_generation_output: bool = field(
        default=False,
        metadata={"help": "Whether to save generated text to file when eval_with_do_generation set to True."},
    )
    lazy: bool = field(
        default=False,
        metadata={
            "help": "Weather to return `MapDataset` or an `IterDataset`.True for `IterDataset`. False for `MapDataset`."
        },
    )
    chat_template: str = field(
        default=None,
        metadata={
            "help": "the path of `chat_template.json` file to handle multi-rounds conversation. If is None, it will not use `chat_template.json`; If is equal with `model_name_or_path`, it will use the default loading; If is directory, it will find the `chat_template.json` under the directory; If is file, it will load it."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Pad the input sequence to `max_length`."},
    )
    autoregressive: bool = field(
        default=False,
        metadata={"help": "Whether to use autoregressive mode."},
    )
    # Pose related parameters
    use_pose_convert: bool = field(default=False, metadata={"help": "Whether to use PoSE data conversion function"})
