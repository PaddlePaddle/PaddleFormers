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
from typing import Any, Dict, List, Optional, Tuple, Union

from tokenizers import AddedToken  # noqa: F401
from transformers import BatchEncoding
from transformers.tokenization_utils import (
    PreTrainedTokenizer as PreTrainedTokenizer_tf,
)
from transformers.tokenization_utils_base import (
    ADDED_TOKENS_FILE,
    CHAT_TEMPLATE_FILE,
    FULL_TOKENIZER_FILE,
    SPECIAL_TOKENS_MAP_FILE,
    TOKENIZER_CONFIG_FILE,
)
from transformers.utils.generic import ExplicitEnum

from ..utils import is_paddle_available
from ..utils.download import DownloadSource, resolve_file_path
from ..utils.log import logger

if is_paddle_available():
    from .legacy.tokenizer_utils import PretrainedTokenizer
else:

    class _MissingPaddleTokenizer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PretrainedTokenizer requires Paddle, but Paddle is not available. "
                "Please install Paddle to use this feature."
            )

    PretrainedTokenizer = _MissingPaddleTokenizer

# legacy PretrainedTokenizer, which is different from huggingface PreTrainedTokenizer


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PADDLE = "pd"
    NUMPY = "np"


class PaddleTokenizerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wrap_return_tensor_methods()

    def _wrap_return_tensor_methods(self):
        """Wrap all relevant methods of the class to support Paddle tensor return types.

        This method identifies and wraps several key methods that should support optional
        conversion of their return values to PaddlePaddle tensors when requested through
        the 'return_tensors="pd"' parameter.

        The methods being wrapped typically include:
        - Core calling functionality (__call__)
        - Padding operations
        - Various encoding methods
        - Batch processing methods
        - Chat template processing

        Only methods that actually exist in the class will be wrapped.
        """
        methods_to_wrap = [
            "__call__",
            "pad",
            "encode_plus",
            "batch_encode_plus",
            "encode",
            "apply_chat_template",
        ]

        for method_name in methods_to_wrap:
            if hasattr(self, method_name):
                self._wrap_single_method(method_name)

    def _wrap_single_method(self, method_name):
        """Wrap a single method of the class to convert its output to Paddle tensors when requested.

        This decorator modifies the specified method to optionally convert its return value to
        PaddlePaddle tensors when the 'return_tensors="pd"' parameter is provided.

        Args:
            method_name (str): The name of the method to be wrapped.

        Returns:
            None: This method modifies the class instance in-place by replacing the original method
            with the wrapped version.
        """
        original_method = getattr(self, method_name)

        def convert_to_paddle(inputs):
            """Convert various input types to Paddle tensors recursively.

            Handles conversion of:
            - Lists (both single and nested)
            - Integers
            - BatchEncoding objects (converts values recursively)
            - Other types (returns unchanged)

            Args:
                inputs: The input data to be converted

            Returns:
                The converted Paddle tensor or the original input if no conversion was needed
            """
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

    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]], dict[str, Any]],
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        """Applies chat template to conversation data (supports 3 formats):

        1. Standard chat format:
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"}
        ]

        2. Batch Conversation Format:
        [
            [{"role": "user", "content": "user messages"}, {"role": "assistant", "content": "assistant messages"}],
            [{"role": "user", "content": "user messages"}]
        ]

        3. Enhanced dictionary format (not natively supported by HuggingFace):
        {
            "messages": [
                {"role": "user", "content": "Query"},
                {"role": "assistant", "content": "Response"}
            ],
            "tools": [],    # Function call definitions
            "documents": [] # RAG context documents
        }

        Note: The legacy PaddleFormers `add_generation_prompt` default was True.
        For backward compatibility, we changed the default behavior of the HuggingFace
        `apply_chat_template` function from False to True.

        In future usage, explicitly pass the `add_generation_prompt` parameter
        to clearly specify the intended behavior.
        """
        if "add_generation_prompt" not in kwargs:
            logger.warning_once(
                "Warning: apply_chat_template() called without explicit `add_generation_prompt` "
                "parameter. Current default=True differs from Hugging Face's default=False. "
                "Always specify this parameter to ensure consistent behavior across versions."
            )

        # Changed the default behavior:
        # Original HF default was False, but we set it to True for compatibility
        add_generation_prompt = kwargs.pop("add_generation_prompt", True)

        if isinstance(conversation, dict):
            messages = conversation.get("messages", None)
            tools = conversation.get("tools", None)
            documents = conversation.get("documents", None)

            # Allow kwargs override for empty values
            if not tools and "tools" in kwargs:
                tools = kwargs.pop("tools")
            if not documents and "documents" in kwargs:
                documents = kwargs.pop("documents")

            return super().apply_chat_template(
                conversation=messages,
                chat_template=chat_template,
                tools=tools,
                documents=documents,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        else:
            return super().apply_chat_template(
                conversation=conversation,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )

    # Rewrite hf's tokenizer function from_pretrained
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *args,
        **kwargs,
    ):
        download_hub = kwargs.get("download_hub", None)
        local_files_only = kwargs.pop("local_files_only", False)

        if download_hub is None:
            download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")
        logger.info(f"Using download source: {download_hub}")

        # If downloaded from hf, use the native hf from pretrained
        if download_hub == DownloadSource.HUGGINGFACE:
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
        # get all tokenizer-related files
        vocab_files = {**cls.vocab_files_names, **additional_files_names}

        if "PF_HOME" in os.environ:
            home_path = os.environ["PF_HOME"]
            home_model_path = os.path.join(home_path, pretrained_model_name_or_path)
            if os.path.isfile(home_model_path) or os.path.isdir(home_model_path):
                pretrained_model_name_or_path = home_model_path

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
                    download_hub=download_hub,
                    local_files_only=local_files_only,
                )
            except (FileNotFoundError, EnvironmentError):
                pass
            except Exception as e:
                raise e
        # 获得cache_dir的目录
        for file_id, file_path in resolved_vocab_files.items():
            if resolved_vocab_files[file_id] is not None:
                cache_dir = os.path.dirname(resolved_vocab_files[file_id])
                break

        if not any(key in resolved_vocab_files for key in cls.vocab_files_names.keys()):
            hf_link = f"https://huggingface.co/{pretrained_model_name_or_path}"
            modelscope_link = f"https://modelscope.cn/models/{pretrained_model_name_or_path}"
            encoded_model_name = pretrained_model_name_or_path.replace("/", "%2F")
            aistudio_link = f"https://aistudio.baidu.com/modelsoverview?sortBy=weight&q={encoded_model_name}"

            raise ValueError(
                f"No vocabulary files found for model '{pretrained_model_name_or_path}'. "
                f"Please check:\n"
                f"1. The model repository ID is correct for your chosen source:\n"
                f"   - Hugging Face Hub: {hf_link}\n"
                f"   - ModelScope: {modelscope_link}\n"
                f"   - AI Studio: {aistudio_link}\n"
                f"2. You have permission to access this model repository\n"
                f"3. Network connection is working properly\n"
                f"4. Try clearing cache and downloading again\n"
                f"Expected vocabulary files: {list(cls.vocab_files_names.keys())}\n"
                f"Valid files found: {list(resolved_vocab_files.keys())}\n"
                f"Note: The repository ID may differ between ModelScope, AI Studio, and Hugging Face Hub.\n"
                f"You are currently using the download source: {download_hub}. Please check the repository ID on the official website."
            )

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

    def _encode_chat_inputs_oneturn(
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

            conversation_ids.append(conversation_id)
            conversation_dict["messages"] = []
            cur_str = ""

        return conversation_ids

    def _extract_non_learnable_parts(self, origin_msg: List[Dict[str, str]], split_s: List[str]):
        """Split the entire chat by specified words. Extract the non-learnable parts."""
        # TODO：We will upgrade this feature later
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
            encode_one_turn = kwargs.pop("encode_one_turn", True)
            add_generation_prompt = kwargs.pop("add_generation_prompt", True)
            if not isinstance(conversations, dict):
                query = self._encode_chat_inputs(
                    conversations, context_data, add_generation_prompt=add_generation_prompt
                )
            else:
                conversations.update(add_generation_prompt=add_generation_prompt)
                if encode_one_turn:
                    query = self._encode_chat_inputs_oneturn(conversations)
                else:
                    query = self._encode_chat_inputs_openai_format(conversations)
        return query

    def encode_chat_inputs_with_no_template(
        self, conversations: List[List[str, str]] | Dict[str, Any], context_data: Dict[str, Any] = {}, **kwargs
    ):
        """
        Args:
            conversation (List[List[str, str]]): the conversation of data
            context_data (Dict[str, Any]): the context data of conversation

        Returns:
            List[list[int], list[int]]: the pair of input_ids and target_ids
        """
        assert isinstance(conversations, dict)

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
            round_str = conversation_dict["messages"]
            # fake template
            tokenize_input = "".join(item["content"] for item in round_str)
            tokenize_input = tokenize_input[len(cur_str) :]
            input_ids = self.convert_tokens_to_ids(self.tokenize(tokenize_input))
            conversation_id.append(input_ids)
            cur_str = tokenize_input

            if idx + 1 < len(conversations["messages"]):
                conversation_dict["messages"].append(conversations["messages"][idx + 1])
                round_str = conversation_dict["messages"]
                # fake template
                tokenize_input = "".join(item["content"] for item in round_str)
                tokenize_input = tokenize_input[len(cur_str) :]
                output_ids = self.convert_tokens_to_ids(self.tokenize(tokenize_input))
                conversation_id.append(output_ids)

            conversation_ids.append(conversation_id)
            conversation_dict["messages"] = []
            cur_str = ""
        return conversation_ids

    def decode_token(
        self,
        all_input_ids: List[int],
        prefix_offset: int = 0,
        read_offset: int = 0,
        skip_special_tokens: bool = False,
    ) -> Tuple[str, int, int]:
        """tokenizer decoding for the streaming generation use case. This method can be overridden for tokenizer that doesn't follow this API"""
        # The prefix text is necessary only to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        prefix_text = self.decode(
            all_input_ids[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        new_text = self.decode(
            all_input_ids[prefix_offset:], skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False
        )

        if len(new_text) > len(prefix_text) and not new_text.endswith("�") and not new_text[:-1].endswith("�"):
            # utf-8 char at the end means it's a potential unfinished byte sequence
            # from byte fallback tokenization.
            # If it's in the middle, it's probably a real invalid id generated
            # by the model
            if new_text.startswith(prefix_text):
                prefix_index = new_text.index(prefix_text)
                new_text = new_text[prefix_index + len(prefix_text) :]
                return new_text, read_offset, len(all_input_ids)
            else:
                return "", prefix_offset, len(all_input_ids)
        else:
            return "", prefix_offset, read_offset


def warp_tokenizer(hf_tokenizer_class: PreTrainedTokenizer_tf):
    return type(hf_tokenizer_class.__name__, (PaddleTokenizerMixin, hf_tokenizer_class), {})


class PreTrainedTokenizer(PaddleTokenizerMixin, PreTrainedTokenizer_tf):
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
