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
import re
from functools import wraps
from typing import Any, Dict, List, Union

from transformers import BatchEncoding
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
    # only used for test
    pretrained_resource_files_map = {
        "vocab_file": {
            "__internal_testing__/micro-random-llama": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
            "__internal_testing__/tiny-random-llama": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
            "facebook/llama-7b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
            "facebook/llama-13b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
            "facebook/llama-30b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
            "facebook/llama-65b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
        },
        "tokenizer_file": {
            "__internal_testing__/micro-random-llama": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/tokenizer.json",
            "__internal_testing__/tiny-random-llama": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/tokenizer.json",
            "facebook/llama-7b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/tokenizer.json",
            "facebook/llama-13b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/tokenizer.json",
            "facebook/llama-30b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/tokenizer.json",
            "facebook/llama-65b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/tokenizer.json",
        },
    }

    pretrained_init_configuration = {
        "__internal_testing__/micro-random-llama": {},
        "__internal_testing__/tiny-random-llama": {},
        "facebook/llama-7b": {},
        "facebook/llama-13b": {},
        "facebook/llama-30b": {},
        "facebook/llama-65b": {},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wrap_return_tensor_methods()

    def _wrap_return_tensor_methods(self):
        methods_to_wrap = [
            "__call__",
            "pad",
            "encode_plus",
            "batch_encode_plus",
            "encode",
            "batch_encode",
            "apply_chat_template",
        ]

        for method_name in methods_to_wrap:
            if hasattr(self, method_name):
                self._wrap_single_method(method_name)

    def _wrap_single_method(self, method_name):
        original_method = getattr(self, method_name)

        def convert_to_paddle(inputs):
            import paddle

            if isinstance(inputs, list):
                if isinstance(inputs[0], int):
                    return paddle.to_tensor([inputs])
                else:
                    return paddle.to_tensor(inputs)
            elif isinstance(inputs, int):
                return paddle.to_tensor(inputs)
            elif isinstance(inputs, BatchEncoding):
                for key, value in inputs.items():
                    inputs[key] = convert_to_paddle(value)
                return inputs
            else:
                return inputs

        @wraps(original_method)
        def wrapper(*args, **kwargs):
            return_tensors = kwargs.get("return_tensors", None)
            if return_tensors == "pd":
                return_tensors = kwargs.pop("return_tensors", None)
            result = original_method(*args, **kwargs)
            if return_tensors == "pd":
                result = convert_to_paddle(result)
            return result

        setattr(self, method_name, wrapper)

    # 复写hf的tokenizer的from_pretrained
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *args,
        **kwargs,
    ):
        from_hf_hub = kwargs.get("from_hf_hub", False)
        from_aistudio = kwargs.get("from_aistudio", False)
        from_modelscope = kwargs.get("from_modelscope", False)
        local_files_only = kwargs.get("local_files_only", False)

        # if not from_hf_hub and not from_aistudio and not from_modelscope:
        #     from_aistudio = True

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
            # else:
            #     print(
            #         "this repo is not supported by paddleformers download source, please check the difference for repo id"
            #     )

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

        if pretrained_model_name_or_path in cls.pretrained_init_configuration:
            # From built-in pretrained models
            # just for test, should not be used in development
            vocab_files = {}
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                vocab_files[file_id] = map_list[pretrained_model_name_or_path]
            resolved_vocab_files = {}

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

    def _encode_chat_inputs_openai_format(
        self,
        conversations: Dict[str, Any],
        add_generation_prompt=True,
    ):
        conversation_dict = {} if "tools" not in conversations else {"tools": conversations["tools"]}
        conversation_dict["messages"] = (
            [conversations["messages"][0]] if conversations["messages"][0]["role"] == "system" else []
        )

        if conversations["messages"][0]["role"] == "system":
            conversations["messages"] = conversations["messages"][1:]

        cur_str = ""
        conversation_ids = []
        for idx in range(0, len(conversations["messages"]), 2):
            conversation_id = []
            conversation_dict["messages"].append(conversations["messages"][idx])
            round_str = self.apply_chat_template(
                conversation_dict["messages"], add_generation_prompt=True, tokenize=False
            )
            # query: user prefix + user content + assist prefix
            query = round_str[len(cur_str) :]
            input_ids = self.convert_tokens_to_ids(self.tokenize(query))
            conversation_id.append(input_ids)
            cur_str = round_str

            if idx + 1 < len(conversations["messages"]):
                conversation_dict["messages"].append(conversations["messages"][idx + 1])
                round_str = self.apply_chat_template(
                    conversation_dict["messages"], add_generation_prompt=False, tokenize=False
                )
                # answer: assistant content
                answer = round_str[len(cur_str) :]
                output_ids = self.convert_tokens_to_ids(self.tokenize(answer))
                conversation_id.append(output_ids)
                cur_str = round_str

            conversation_ids.append(conversation_id)

        return conversation_ids

    def _extract_non_learnable_parts(self, origin_msg: List[Dict[str, str]], split_s: List[str]):
        """Split the entire chat by specified words. Extract the non-learnable parts."""
        # distinguish and replace the special words in original string to an uncompiled form: Like | -> \|
        regex_pattern = "|".join(map(re.escape, split_s))
        # splited by replaced specified words
        non_learnable_parts = re.split(
            r"(?:%s)" % regex_pattern,
            self.apply_chat_template(conversation=origin_msg, add_generation_prompt=False, tokenize=False),
        )
        if non_learnable_parts[-1] == "":
            non_learnable_parts.pop()
        return non_learnable_parts

    def _encode_chat_inputs(
        self,
        conversations: List[List[str, str]],
        context_data: Dict[str, Any] = {},
        system: str = None,
        add_generation_prompt=True,
    ):
        result = {}

        # Some template do not support system msg, so we need to check it first.
        if system:
            try:
                self.apply_chat_template([{"role": "system", "content": system}], add_generation_prompt)
            except Exception as e:
                raise ValueError("System is not supported in this tokenizer.", e)

        # convert list msg to role dict msg
        conversation_dict = []
        origin_msg = []
        for round in conversations:
            round_role = [
                {"role": "user", "content": round[0]},
                {"role": "assistant", "content": round[1]},
            ]
            origin_msg.extend(round_role)
            conversation_dict.append(round_role)
        ans = []

        # get answer in single round, then compile the chat entirely and split by single round ans
        # attention: answer should include end token!
        for conv in conversation_dict:
            roundi = [{"role": "system", "content": system}] + conv if system else conv
            roundi_str = self.apply_chat_template(conversation=roundi, add_generation_prompt=False, tokenize=False)
            roundi_no_ans = [{"role": "system", "content": system}] + [conv[0]] if system else [conv[0]]
            roundi_no_ans_str = self.apply_chat_template(
                conversation=roundi_no_ans, add_generation_prompt=add_generation_prompt, tokenize=False
            )
            ans_roundi = roundi_str[len(roundi_no_ans_str) :]
            ans.append(ans_roundi)

        non_learnable_parts = self._extract_non_learnable_parts(origin_msg, ans)
        assert len(non_learnable_parts) == len(
            ans
        ), f"Get non_learnable_parts len: {len(non_learnable_parts)}, but ans len: {len(ans)}."

        conversation_ids = []
        for i in range(len(non_learnable_parts)):
            conversation_ids.append(
                self([non_learnable_parts[i], ans[i]], add_special_tokens=False, padding=False)["input_ids"]
            )

        result["conversations"] = conversation_ids
        return result

    def encode_chat_inputs(
        self, conversations: List[List[str, str]] | Dict[str, Any], context_data: Dict[str, Any] = {}, **kwargs
    ):
        """Encodes conversation to pairs of token ids.
        Turn 0: bos + system + sep + user     bot + eos
        Turn t: sep + bot + query             bot + eos

        Args:
            conversation (List[List[str, str]]): the conversation of data
            context_data (Dict[str, Any]): the context data of conversation

        Returns:
            List[list[int], list[int]]: the pair of input_ids and target_ids
        """
        if not self.chat_template:
            raise ValueError("chat_template is not set, please set chat_template first.")
        else:
            add_generation_prompt = kwargs.pop("add_generation_prompt", True)
            if not isinstance(conversations, dict):
                query = self._encode_chat_inputs(
                    conversations, context_data, add_generation_prompt=add_generation_prompt
                )
            else:
                conversations.update(add_generation_prompt=add_generation_prompt)
                query = self._encode_chat_inputs_openai_format(conversations)
        return query


def warp_tokenizer(hf_tokenizer_class: PreTrainedTokenizer):
    return type(hf_tokenizer_class.__name__, (PaddleTokenizerMixin, hf_tokenizer_class), {})
