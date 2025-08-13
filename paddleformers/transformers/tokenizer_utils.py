# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
from typing import Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import (
    ADDED_TOKENS_FILE,
    CHAT_TEMPLATE_FILE,
    FULL_TOKENIZER_FILE,
    SPECIAL_TOKENS_MAP_FILE,
    TOKENIZER_CONFIG_FILE,
)
from transformers.utils import ExplicitEnum

from ..utils.download import resolve_file_path

DOWNDLOAD_SOURCE_MAPPING = {
    "Qwen": {"hf": "Qwen", "modelscope": "Qwen", "ai_studio": "PaddleNLP"},
    "DeepSeek": {"hf": "deepseek-ai", "modelscope": "deepseek-ai", "aistudio": "PaddleNLP"},
}


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PADDLE = "pd"
    NUMPY = "np"


class PaddleTokenizerMixin:
    # 复写hf的tokenizer的from_pretrained
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *args,
        **kwargs,
    ):
        from_hf_hub = kwargs.pop("from_hf_hub", False)
        from_aistudio = kwargs.pop("from_aistudio", False)
        from_modelscope = kwargs.pop("from_modelscope", False)
        local_files_only = kwargs.pop("local_files_only", False)

        if not from_hf_hub and not from_aistudio and not from_modelscope:
            from_hf_hub = True

        if not os.path.isdir(pretrained_model_name_or_path):
            download_source = None
            model_name = pretrained_model_name_or_path.split("/")[-1]
            for key in DOWNDLOAD_SOURCE_MAPPING.keys():
                if key in model_name:
                    download_source = DOWNDLOAD_SOURCE_MAPPING.get(key, None)
                    break
            if download_source is not None:
                if from_hf_hub:
                    pretrained_model_name_or_path = os.path.join(download_source["hf"], model_name)
                elif from_aistudio:
                    pretrained_model_name_or_path = os.path.join(download_source["ai_studio"], model_name)
                elif from_modelscope:
                    pretrained_model_name_or_path = os.path.join(download_source["modelscope"], model_name)
            else:
                print(
                    "this repo is not supported by paddleformers download source, please check the difference for repo id"
                )

        # 如果从hf下载，则使用原生的hf的from_pretrained
        if from_hf_hub:
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )

        cache_dir = kwargs.pop("cache_dir", None)
        subfolder = kwargs.pop("subfolder", "")

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        additional_files_names = {
            "added_tokens_file": ADDED_TOKENS_FILE,  # kept only for legacy
            "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,  # kept only for legacy
            "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
            # tokenizer_file used to initialize a slow from a fast. Properly copy the `addedTokens` instead of adding in random orders
            "tokenizer_file": FULL_TOKENIZER_FILE,
            "chat_template_file": CHAT_TEMPLATE_FILE,
        }
        # get hf的所有跟tokenizer相关的文件
        vocab_files = {**cls.vocab_files_names, **additional_files_names}

        # 返回所有文件的local file path
        if os.path.isdir(pretrained_model_name_or_path):
            for file_id, file_name in vocab_files.items():
                full_file_name = os.path.join(pretrained_model_name_or_path, subfolder, file_name)
                if os.path.isfile(full_file_name):
                    vocab_files[file_id] = full_file_name
                else:
                    vocab_files[file_id] = None

        resolved_vocab_files = {}
        for file_id, file_path in vocab_files.items():
            if file_path is None or os.path.isfile(file_path):
                resolved_vocab_files[file_id] = file_path
                continue
            try:
                resolved_vocab_files[file_id] = resolve_file_path(
                    pretrained_model_name_or_path,
                    [file_path],
                    subfolder,
                    cache_dir=cache_dir,
                    local_dir=cache_dir,
                    from_aistudio=from_aistudio,
                    from_modelscope=from_modelscope,
                    from_hf_hub=False,
                    local_files_only=local_files_only,
                )
            except Exception:
                pass
        # 获得cache_dir的目录
        for file_id, file_path in resolved_vocab_files.items():
            if resolved_vocab_files[file_id] is not None:
                cache_dir = os.path.dirname(resolved_vocab_files[file_id])
                break

        return super()._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            {},
            *args,
            cache_dir=cache_dir,
            local_files_only=True,
            **kwargs,
        )


def warp_tokenizer(hf_tokenizer_class: PreTrainedTokenizer):
    return type(hf_tokenizer_class.__name__, (PaddleTokenizerMixin, hf_tokenizer_class), {})
