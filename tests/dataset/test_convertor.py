# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from paddleformers.datasets.reader.convertor import (
    convert_dpo_txt_data,
    convert_mm_data,
    convert_pretraining_data,
    convert_txt_data,
    erniekit_convertor,
)


def make_dpo_txt_data():
    input_data = {
        "system": "系统设定",
        "src": [
            "第一轮输入",
            "第二轮输入",
        ],
        "tgt": [
            "output for first query",
        ],
        "response": [["<think>\n我要好好想一想\n</think>\n\n认真回答"], ["<think>\n这个问题我不会\n</think>\n\n错误示范"]],
        "sort": [1, 0],
        "label": [0, 1],
    }
    output_data = {
        "messages": [
            {"role": "system", "content": "系统设定"},
            {"role": "user", "content": "第一轮输入"},
            {"role": "assistant", "content": "output for first query"},
            {"role": "user", "content": "第二轮输入"},
        ],
        "chosen_response": [
            {"role": "assistant", "content": "<think>\n我要好好想一想\n</think>\n\n认真回答"},
        ],
        "rejected_response": [
            {"role": "assistant", "content": "<think>\n这个问题我不会\n</think>\n\n错误示范"},
        ],
    }
    return input_data, output_data


def make_txt_data():
    input_data = {
        "system": "系统设定",
        "src": [
            "第一轮输入",
            "第二轮输入",
        ],
        "tgt": [
            "output for first query",
            "<think>\n我要好好想一想\n</think>\n\n认真回答",
        ],
    }
    output_data = {
        "messages": [
            {"role": "system", "content": "系统设定"},
            {"role": "user", "content": "第一轮输入"},
            {"role": "assistant", "content": "output for first query"},
            {"role": "user", "content": "第二轮输入"},
            {"role": "assistant", "content": "<think>\n我要好好想一想\n</think>\n\n认真回答"},
        ],
    }
    return input_data, output_data


def make_mm_data():
    input_data = {
        "image_info": [{"matched_text_index": 0, "image_url": "PATH/TO/IMAGE"}],
        "text_info": [
            {"text": "第一轮输入", "tag": "mask", "tool_response": False},
            {
                "text": "<think>\n我要好好想一想\n</think>\n\n",
                "tool_calls": [{"name": "image_zoom_in_tool", "arguments": '{"bbox_2d": [0, 493, 518, 786]}'}],
                "tag": "no_mask",
            },
            {"text": "工具回复", "tag": "mask", "tool_response": True},
            {
                "text": "<think>\n我要好好想一想\n</think>\n\n",
                "tool_calls": [{"name": "image_zoom_in_tool", "arguments": '{"bbox_2d": [0, 493, 518, 786]}'}],
                "tag": "no_mask",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "image_zoom_in_tool",
                    "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bbox_2d": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner, and the values of x1, y1, x2, y2 are all normalized to the range 0–1000 based on the original image dimensions.",
                            },
                            "label": {
                                "type": "string",
                                "description": "The name or label of the object in the specified bounding box (optional).",
                            },
                        },
                        "required": ["bbox_2d"],
                    },
                    "strict": False,
                },
            }
        ],
    }
    output_data = {
        "messages": [
            {"role": "user", "content": "<image>第一轮输入"},
            {
                "role": "assistant",
                "content": "<think>\n我要好好想一想\n</think>\n\n",
                "tool_calls": '[{"name": "image_zoom_in_tool", "arguments": "{\\"bbox_2d\\": [0, 493, 518, 786]}"}]',
            },
            {"role": "observation", "content": "工具回复"},
            {
                "role": "assistant",
                "content": "<think>\n我要好好想一想\n</think>\n\n",
                "tool_calls": '[{"name": "image_zoom_in_tool", "arguments": "{\\"bbox_2d\\": [0, 493, 518, 786]}"}]',
            },
        ],
        "images": ["PATH/TO/IMAGE"],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "image_zoom_in_tool",
                    "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bbox_2d": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner, and the values of x1, y1, x2, y2 are all normalized to the range 0–1000 based on the original image dimensions.",
                            },
                            "label": {
                                "type": "string",
                                "description": "The name or label of the object in the specified bounding box (optional).",
                            },
                        },
                        "required": ["bbox_2d"],
                    },
                    "strict": False,
                },
            }
        ],
    }
    return input_data, output_data


def make_pretrain_txt_data():
    input_data = {"text": ["今天天气真好"]}
    output_data = {
        "messages": [
            {"role": "assistant", "content": "今天天气真好"},
        ],
    }
    return input_data, output_data


class TestErniekitConvertor(unittest.TestCase):
    def test_convert_dpo(self):
        input_data, output_data = make_dpo_txt_data()

        converted_data = convert_dpo_txt_data(input_data)

        for key in output_data.keys():
            self.assertEqual(output_data[key], converted_data[key])

    def test_erniekit_convertor_dpo(self):
        input_data, output_data = make_dpo_txt_data()

        converted_data = erniekit_convertor(input_data)

        for key in output_data.keys():
            self.assertEqual(output_data[key], converted_data[key])

    def test_convert_txt(self):
        input_data, output_data = make_txt_data()

        converted_data = convert_txt_data(input_data)

        self.assertEqual(output_data, converted_data)

    def test_erniekit_convertor_txt(self):
        input_data, output_data = make_txt_data()

        converted_data = erniekit_convertor(input_data)

        self.assertEqual(output_data, converted_data)

    def test_convert_mm(self):
        input_data, output_data = make_mm_data()

        converted_data = convert_mm_data(input_data)

        self.assertEqual(output_data, converted_data)

    def test_erniekit_convertor_mm(self):
        input_data, output_data = make_mm_data()

        converted_data = erniekit_convertor(input_data)

        self.assertEqual(output_data, converted_data)

    def test_convert_pretrain(self):
        input_data, output_data = make_pretrain_txt_data()

        converted_data = convert_pretraining_data(input_data)

        self.assertEqual(output_data, converted_data)

    def test_erniekit_convertor_pretrain(self):
        input_data, output_data = make_pretrain_txt_data()

        converted_data = erniekit_convertor(input_data)

        self.assertEqual(output_data, converted_data)
