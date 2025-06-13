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

import os
from typing import Any

import numpy as np
import paddle
from datasets import load_dataset
from paddle.io import Dataset

from ...transformers import PretrainedTokenizer
from ...transformers.tokenizer_utils import PaddingStrategy


def left_padding(sequences, padding_value=0, max_length=None):
    arrs = [np.asarray(seq) for seq in sequences]
    max_length = max_length or max([len(seq) for seq in sequences])
    bs = len(sequences)
    data = np.full([bs, max_length], padding_value, dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        data[i, -len(arr) :] = arr
    return data


def padding_batch_data(
    samples: list[dict], pad_token_id: int, requires_label: bool, max_prompt_len: int
) -> list[dict]:
    input_dict = {}

    input_ids = [sample["input_ids"] for sample in samples]
    # TODO(drownfish19): confirm if this is correct
    # attention_mask = [np.ones(input_id.shape, dtype=bool) for input_id in input_ids]
    input_dict["input_ids"] = left_padding(input_ids, padding_value=pad_token_id, max_length=max_prompt_len)
    # input_dict["attention_mask"] = left_padding(attention_mask, padding_value=0)
    input_dict["raw_prompt_len"] = paddle.to_tensor([len(sample["input_ids"]) for sample in samples])

    if requires_label:
        label_ids = [sample["label_ids"] for sample in samples]
        input_dict["label_ids"] = left_padding(label_ids, padding_value=pad_token_id)
        input_dict["raw_label_ids_len"] = paddle.to_tensor([len(sample["label_ids"]) for sample in samples])

    return input_dict


def collate_fn(data_list: list[dict], pad_token_id: int, requires_label: bool, max_prompt_len: int) -> dict:
    input_dict = padding_batch_data(data_list, pad_token_id, requires_label, max_prompt_len)

    tensors = {}
    non_tensors = {}

    for key, val in input_dict.items():
        if isinstance(val, paddle.Tensor):
            tensors[key] = val
        if isinstance(val, np.ndarray):
            tensors[key] = paddle.to_tensor(val)
        else:
            non_tensors[key] = val

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    data: list[dict[str, paddle.Tensor]]
    _SENTINEL: Any = object()

    def __init__(
        self,
        dataset_name_or_path,
        tokenizer: PretrainedTokenizer,
        max_prompt_len=1024,
        filter_prompts=True,
        prompt_key="src",
        requires_label=False,
        response_key=None,
        chat_template_func=None,
        splits=None,
        filter_overlong_prompts=True,
        apply_chat_template=False,
    ):
        self.dataset_name_or_path = dataset_name_or_path
        self.tokenizer = tokenizer
        self.apply_chat_template = apply_chat_template

        self.max_prompt_len = max_prompt_len
        self.filter_prompts = filter_prompts

        self.prompt_key = prompt_key
        self.response_key = response_key
        self.chat_template_func = chat_template_func
        self.requires_label = requires_label
        self.splits = splits
        self.filter_overlong_prompts = filter_overlong_prompts
        # self.lazy = lazy

        # self._download()
        self._read_files()
        self.data = [self._SENTINEL for _ in range(len(self.rawdata))]

    def _read_files(self):
        if os.path.exists(self.dataset_name_or_path):
            # load file from local disk

            self.rawdata = load_dataset("json", data_files=self.dataset_name_or_path, split="train")
        else:
            # 先不管huggingface这个分支
            self.rawdata = load_dataset(self.dataset_name_or_path, splits=self.splits)[0]

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool = True,
        max_length: int | None = None,
    ) -> paddle.Tensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""

        if max_length is None:
            max_length = self.tokenizer.model_max_length

        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors="np",
        )["input_ids"][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, paddle.Tensor]:
        """Get a tokenized data sample by index."""
        data = self.data[index]
        if data is self._SENTINEL:
            data = {}
            raw_sample = self.rawdata[index]
            prompt = raw_sample[self.prompt_key]
            if self.apply_chat_template and self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)

            data["input_ids"] = self.tokenize(text=prompt, max_length=self.max_prompt_len, truncation=True)
            if self.requires_label:
                label = raw_sample[self.response_key]
                data["label_ids"] = self.tokenize(label)
            self.data[index] = data

        return data
