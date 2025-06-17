# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import sys
import tempfile
import unittest
from typing import Optional

from parameterized import parameterized_class

from paddleformers.transformers import AutoTokenizer
from paddleformers.transformers.tokenizer_utils import ChatTemplate


def tokenize_rounds_example(tokenizer, example, data_args, **kwargs):
    """tokenize multi-rounds examples with chat_template.json

    Args:
        tokenizer (PretrainedTokenizer): the instance of tokenizer
        example (dict[str, str | list[str]]):
                the example instance, which can be: {"src": "src-sentence", "tgt": "tgt-sentence"}
                or {"src": ["src-sentence-1", ..., "src-sentence-N"], "tgt": ["tgt-sentence-1", ..., "tgt-sentence-N"]}
        data_args (DataArgument): the data_argument instance of data processing

    Returns:
        dict[str, list[int]]: return input_ids and labels fields
    """

    # 0. prepare data
    context_data = example.get("context", {})
    context_data["is_training"] = True

    example["src"] = example["src"] if isinstance(example["src"], list) else [example["src"]]
    example["tgt"] = example["tgt"] if isinstance(example["tgt"], list) else [example["tgt"]]

    assert len(example["src"]) == len(example["tgt"]), "the length of `src` and `tgt` field must be same."

    conversations = [[src, tgt] for src, tgt in zip(example["src"], example["tgt"])]

    # 1. only tokenize input_ids
    conversation_result: list[tuple[list[int], list[int]]] = tokenizer.encode_chat_inputs(
        conversations, context_data=context_data, **kwargs
    )
    system_ids = conversation_result.pop("system", []) or []

    # 2. truncate conversations based on conversation unit
    input_ids, labels = [], []
    conversations_ids = conversation_result.pop("conversations")

    assert (
        len(system_ids) < data_args.max_length
    ), f"the length of system_ids<{len(system_ids)}> should be smaller than max_length<{data_args.max_length}>."
    max_length = data_args.max_length - len(system_ids)

    should_break = False
    for index in range(len(conversations_ids) - 1, -1, -1):
        user_input_ids, bot_input_ids = conversations_ids[index][0], conversations_ids[index][1]

        # break when the length of current conversations is greater than max_length
        if len(input_ids) + len(user_input_ids) + len(bot_input_ids) > max_length:

            # when the length of last conversation is lager than max_length, we should not break: at least one round
            if index < len(conversations_ids) - 1:
                break

            user_input_ids = user_input_ids[: data_args.src_length - len(system_ids)]
            bot_input_ids = bot_input_ids[: max_length - len(user_input_ids)]

            should_break = True

        input_ids = user_input_ids + bot_input_ids + input_ids
        labels = len(user_input_ids) * [-100] + bot_input_ids + labels

        if should_break:
            break

    input_ids = system_ids + input_ids
    labels = [-100] * len(system_ids) + labels
    tokenized_source = {"input_ids": input_ids}
    sequence_length = len(input_ids)

    if "position_ids" in tokenizer.model_input_names:
        tokenized_source["position_ids"] = list(range(sequence_length))

    return tokenized_source, labels


class ChatTemplateTest(unittest.TestCase):
    chat_template_config_file = "./tests/fixtures/chat_template.json"

    @property
    def chat_template(self):
        return ChatTemplate.from_file(self.chat_template_config_file)

    def test_inference_template(self):
        query = "你好"
        final_query = self.chat_template(query)
        expected_query = f"你是一个人工智能助手\nHuman: {query}<sep> Bot:"
        self.assertEqual(final_query, expected_query)

    def test_inference_conversation_template(self):
        conversations = [["你好", "您好，我是个人人工智能助手，请问有什么可以帮您。"], ["今天的天气怎么样？"]]
        final_query = self.chat_template(conversations)
        expected_query = "你是一个人工智能助手\nHuman: 你好<sep> Bot:您好，我是个人人工智能助手，请问有什么可以帮您。\nHuman: 今天的天气怎么样？<sep> Bot:"
        self.assertEqual(final_query, expected_query)

    def test_inference_conversation_template_with_one_part(self):
        conversations = [["你好"], ["今天的天气怎么样？"]]
        with self.assertRaises(AssertionError):
            self.chat_template(conversations)

    def test_null_chat_template(self):
        chat_template = ChatTemplate()
        query = "今天吃啥"
        final_query = chat_template(query)
        assert final_query == query

    def test_system_query(self):
        system = "你是一个人工智能助手:"
        query_template = "Human: {{query}}"
        chat_template = ChatTemplate(system=system, query=query_template)
        query = "今天吃啥"
        final_query = chat_template(query)
        assert final_query == system + query_template.replace("{{query}}", query)

    def test_conversation(self):
        conversation = ["Human: {{user}}<sep>", "Bot: {{bot}}\n\n"]
        chat_template = ChatTemplate(conversation=conversation)

        query = "今天吃啥"
        final_query = chat_template(query)
        assert final_query == query

        second_query = [["你好", "您好，我是个人人工智能助手"], [query]]
        final_query = chat_template(second_query)
        assert final_query == "Human: 你好<sep>Bot: 您好，我是个人人工智能助手\n\n" + query


class ChatTemplateContextDataTest(unittest.TestCase):
    chat_template_config_file = "./tests/fixtures/chat_template_with_context.json"

    @property
    def chat_template(self):
        return ChatTemplate.from_file(self.chat_template_config_file)

    def test_inference_template(self):
        query = [["你好"]]
        context_data = {
            "system": "<<SYSTEM-MESSAGE>>",
            "instruction": "<<INSTRUCTION-MESSAGE>>",
        }
        final_query = self.chat_template(query, context_data=context_data)
        expected_query = "你是一个人工智能助手<<SYSTEM-MESSAGE>>-<<INSTRUCTION-MESSAGE>>\nHuman: 你好<sep> Bot:"
        self.assertEqual(final_query, expected_query)


class ChatTemplateIntegrationTest(unittest.TestCase):
    def test_linlyai_chinese_llama_2_chat_template(self):
        tokenizer = AutoTokenizer.from_pretrained("linly-ai/chinese-llama-2-7b")
        query = "你好"
        final_query = tokenizer.apply_chat_template(query, tokenize=False)
        expected_query = f"<s>### Instruction:{query}  ### Response:"
        self.assertEqual(final_query, expected_query)

        # test multi turns conversation
        query = [["你好", "您好，我是个人人工智能助手"], ["今天吃啥"]]
        final_query = tokenizer.apply_chat_template(query, tokenize=False)
        expected_query = "<s>### Instruction: 你好  ### Response:您好，我是个人人工智能助手 </s>### Instruction:今天吃啥  ### Response:"
        self.assertEqual(final_query, expected_query)

    def test_linlyai_chinese_llama_2_chat_template_with_none_saved(self):
        tokenizer = AutoTokenizer.from_pretrained("linly-ai/chinese-llama-2-7b")
        tokenizer.chat_template = None
        with tempfile.TemporaryDirectory() as tempdir:
            tokenizer.save_pretrained(tempdir)

            chat_template_file = os.path.join(tempdir, "chat_template.json")
            self.assertFalse(os.path.exists(chat_template_file))

    def test_qwen_14b_chat(self):
        # refer to: https://huggingface.co/Qwen/Qwen-14B-Chat/blob/main/qwen_generation_utils.py#L119

        # 1. test render base on query & conversation data
        tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-14b-chat")
        query = "你好"
        final_query = tokenizer.apply_chat_template(query, tokenize=False)

        expected_query = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(final_query, expected_query)

        query = [["你好", "您好，我是个人人工智能助手"], ["今天吃啥"]]
        final_query = tokenizer.apply_chat_template(query, tokenize=False)

        expected_query = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n"
            "<|im_start|>assistant\n您好，我是个人人工智能助手<|im_end|>\n"
            "<|im_start|>user\n今天吃啥<|im_end|>\n<|im_start|>assistant\n"
        )
        self.assertEqual(final_query, expected_query)


@parameterized_class(
    ["model_name"],
    [
        ["linly-ai/chinese-llama-2-7b"],
    ],
)
class TestChatTemplateSpecialTokens(unittest.TestCase):
    model_name: str

    def common_prefix(self, arr1, arr2):
        min_length = min(len(arr1), len(arr2))
        for i in range(min_length):
            if arr1[i] != arr2[i]:
                return arr1[:i]
        return arr1[:min_length]

    def get_common_prefix(self, tokenizer):
        first_ids = tokenizer("欢迎使用 PaddlePaddle")["input_ids"]
        second_ids = tokenizer("")["input_ids"]
        prefix_ids = self.common_prefix(first_ids, second_ids)
        return prefix_ids

    def test_prefix(self):
        prompt = "欢迎使用 PaddleFormers 大模型开发套件"
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        result = tokenizer.apply_chat_template(prompt, tokenize=False)

        result_ids = tokenizer(result, add_special_tokens=False)["input_ids"]
        special_token_prefix_ids = self.get_common_prefix(tokenizer)
        assert result_ids[: len(special_token_prefix_ids)] == special_token_prefix_ids


class TestChatTemplateTruncation(unittest.TestCase):
    class DataArg:
        def __init__(self, max_length, src_length: Optional[int] = None):
            self.max_length: int = max_length
            if src_length is None:
                src_length = self.max_length - 8

            self.src_length: int = src_length

    chat_template_config_file = "./tests/fixtures/chat_template.json"

    def setUp(self):
        sys.path.insert(0, "./llm")

    def tearDown(self):
        sys.path.remove("./llm")

    @property
    def chat_template(self):
        return ChatTemplate.from_file(self.chat_template_config_file)

    def test_must_have_system(self):
        tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-14b-chat")

        # get the length of system
        system = tokenizer.chat_template.render_system()
        system_ids = tokenizer.encode(system, add_special_tokens=False)["input_ids"]

        fake_data_args = self.DataArg(len(system_ids) + 5, src_length=len(system_ids) + 5)

        example = {"src": ["你好"], "tgt": ["您好，我是个人人工智能助手"]}
        result, _ = tokenize_rounds_example(tokenizer, example, fake_data_args)
        sentence_result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result["input_ids"]))
        expected_sentence = tokenizer.chat_template.system + "\n<|im_start|>user\n你好"
        self.assertEqual(expected_sentence, sentence_result)

    def test_at_least_one_turn(self):
        query = [["你好", "您好，我是个人人工智能助手"], ["今天吃啥", "你可以选择不同的菜系"]]
        tokenizer = AutoTokenizer.from_pretrained("linly-ai/chinese-llama-2-7b")
        # tokenizer.init_chat_template(self.chat_template_config_file)

        # get all query sentence
        all_sentence = tokenizer.chat_template.render_system()
        all_sentence += "".join(
            [
                " ".join(tokenizer.chat_template.render_conversation(one_turn, index=index))
                for index, one_turn in enumerate(query)
            ]
        )
        all_sentence_ids = tokenizer(all_sentence, add_special_tokens=False)["input_ids"]

        # get the max_length of conversation

        fake_data_args = self.DataArg(1024)
        example = {"src": ["你好", "今天吃啥"], "tgt": ["您好，我是个人人工智能助手", "你可以选择不同的菜系"]}
        tokenized_result, _ = tokenize_rounds_example(tokenizer, example, fake_data_args)

        assert len(all_sentence_ids) == len(tokenized_result["input_ids"])

        fake_data_args = self.DataArg(len(all_sentence_ids) - 4)
        expected_example = {"src": ["你好"], "tgt": ["您好，我是个人人工智能助手"]}
        expected_tokenized_result, _ = tokenize_rounds_example(tokenizer, expected_example, fake_data_args)

        sentence_result = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(expected_tokenized_result["input_ids"])
        )

        # https://github.com/PaddlePaddle/PaddleFormers/blob/v2.6.1/paddleformers/transformers/llama/tokenizer.py#L119
        # should use blank string to join
        expected_sentence = " ".join(tokenizer.chat_template.render_conversation(["你好", "您好，我是个人人工智能助手"]))
        expected_sentence = expected_sentence.replace("<s>", "<s> ")
        self.assertEqual(
            sentence_result,
            expected_sentence,
        )

    def test_inference_template_with_context_data(self):
        tokenizer = AutoTokenizer.from_pretrained("__internal_testing__/tiny-random-llama")
        chat_template_config_file = "./tests/fixtures/chat_template_with_context.json"
        tokenizer.init_chat_template(chat_template_config_file)

        query = "你好"
        context_data = {
            "system": "<<SYSTEM-MESSAGE>>",
            "instruction": "<<INSTRUCTION-MESSAGE>>",
        }
        final_query = tokenizer.apply_chat_template(query, context_data=context_data, tokenize=False)
        expected_query = "你是一个人工智能助手<<SYSTEM-MESSAGE>>-<<INSTRUCTION-MESSAGE>>\nHuman: 你好<sep> Bot:"
        self.assertEqual(final_query, expected_query)


class TemplateIntegrationTest(unittest.TestCase):
    class DataArg:
        def __init__(self, max_length, src_length: Optional[int] = None):
            self.max_length: int = max_length
            if src_length is None:
                src_length = self.max_length - 8

            self.src_length: int = src_length

    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-7b-chat")
        qwen_jinja = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        self.tokenizer.init_chat_template(qwen_jinja)
        sys.path.insert(0, "./llm")
        return super().setUp()

    def tearDown(self):
        sys.path.remove("./llm")

    def test_chat_template(self):
        # test single turn
        query = "你好"
        final_query = self.tokenizer.apply_chat_template(query, tokenize=False)
        expected_query = f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(final_query, expected_query)

        # test multi turns conversation
        query = [["你好", "您好，我是个人人工智能助手"], ["今天吃啥"]]
        final_query = self.tokenizer.apply_chat_template(query, tokenize=False)
        expected_query = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n您好，我是个人人工智能助手<|im_end|>\n<|im_start|>user\n今天吃啥<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(final_query, expected_query)

    def test_system_error(self):
        # test system messaage error
        error_jinja = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
        self.tokenizer.init_chat_template(error_jinja)
        from jinja2.exceptions import TemplateError

        with self.assertRaises(TemplateError):
            self.tokenizer.apply_chat_template([{"role": "system", "content": ""}])

    def test_round_error(self):
        # error round, 1 is not a valid role.
        query = [["你好", "您好，我是个人人工智能助手"], ["今天吃啥"], ["你好", "您好"]]
        with self.assertRaises(ValueError):
            self.tokenizer.apply_chat_template(query, tokenize=False)

    def test_jinja_syntax_error(self):
        # test system messaage error
        error_jinja = (
            "{ bos_token }{% if messages[0]['role'] == 'system' %}{ raise_exception('System role not supported')}"
        )
        from jinja2.exceptions import TemplateSyntaxError

        with self.assertRaises(TemplateSyntaxError):
            self.tokenizer.init_chat_template(error_jinja)

    def test_train_format(self):

        fake_data_args = self.DataArg(50, src_length=50)
        example = {"src": ["你好"], "tgt": ["您好，我是个人人工智能助手"]}
        result, tgt_id = tokenize_rounds_example(self.tokenizer, example, fake_data_args, add_generation_prompt=True)
        sentence_result = self.tokenizer.decode(result["input_ids"])
        expected_sentence = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n您好，我是个人人工智能助手<|im_end|>\n"
        self.assertEqual(expected_sentence, sentence_result)

        tgt_idx = len(
            self.tokenizer.encode(
                "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
            )["input_ids"]
        )
        self.assertEqual(tgt_id[tgt_idx - 1], -100)
        self.assertNotEqual(tgt_id[tgt_idx], -100)

    def test_train_format_multi(self):

        fake_data_args = self.DataArg(50, src_length=50)
        example = {"src": ["用户Round 1", "用户Round 2"], "tgt": ["回答Round 1", "回答Round 2"]}
        result, tgt_id = tokenize_rounds_example(self.tokenizer, example, fake_data_args, add_generation_prompt=True)

        tgt_idx_1 = len(
            self.tokenizer.encode(
                "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n用户Round 1<|im_end|>\n<|im_start|>assistant\n"
            )["input_ids"]
        )
        tgt_idx_2 = len(
            self.tokenizer.encode(
                "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n用户Round 1<|im_end|>\n<|im_start|>assistant\n"
                "回答Round 1<|im_end|>\n<|im_start|>user\n用户Round 2<|im_end|>\n<|im_start|>assistant\n"
            )["input_ids"]
        )

        self.assertEqual(tgt_id[tgt_idx_1 - 1], -100)
        self.assertNotEqual(tgt_id[tgt_idx_1], -100)
        self.assertEqual(tgt_id[tgt_idx_2 - 1], -100)
        self.assertNotEqual(tgt_id[tgt_idx_2], -100)

    def test_split_answer(self):
        original_msg = [
            {"role": "user", "content": "用户Round 1"},
            {"role": "assistant", "content": "|回答Round 1|"},
            {"role": "user", "content": "用户Round 2"},
            {"role": "assistant", "content": "_回答Round 2?"},
        ]
        answer = ["|回答Round 1|<|im_end|>\n", "_回答Round 2?<|im_end|>\n"]
        split_part = self.tokenizer._extract_non_learnable_parts(original_msg, answer)
        self.assertEqual(len(split_part), 2)
        self.assertEqual(
            split_part[0],
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n用户Round 1<|im_end|>\n<|im_start|>assistant\n",
        )
