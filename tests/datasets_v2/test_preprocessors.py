# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Basic tests for datasets_v2 preprocessors.

Run with: python -m pytest tests/datasets_v2/test_preprocessors.py -v
Or debug via VSCode launch configuration "Test Preprocessors".
"""

import importlib

# Workaround: broken torchcodec residual in env triggers datasets import error.
# Patch find_spec to return None for torchcodec so datasets thinks it's not installed.
_original_find_spec = importlib.util.find_spec


def _patched_find_spec(name, *args, **kwargs):
    if name == "torchcodec":
        return None
    return _original_find_spec(name, *args, **kwargs)


importlib.util.find_spec = _patched_find_spec

from datasets import Dataset

from paddleformers.datasets_v2.preprocessors import (
    AlpacaPreprocessor,
    AutoPreprocessor,
    MessagesPreprocessor,
    ResponsePreprocessor,
)


def make_dataset(data: dict) -> Dataset:
    """Helper: create a HF Dataset from a dict of columns."""
    return Dataset.from_dict(data)


# ============================================================
# ResponsePreprocessor
# ============================================================


def test_response_basic():
    """Basic query/response → messages conversion."""
    ds = make_dataset(
        {
            "query": ["What is 2+2?", "Hello"],
            "response": ["4", "Hi there"],
        }
    )
    preprocessor = ResponsePreprocessor()
    result = preprocessor(ds)

    assert "messages" in result.column_names
    row0 = result[0]
    assert row0["messages"] == [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]


def test_response_with_system_and_history():
    """Query/response with system prompt and history."""
    ds = make_dataset(
        {
            "query": ["Current question"],
            "response": ["Current answer"],
            "system": ["You are a helpful assistant."],
            "history": ["[['q1', 'r1']]"],
        }
    )
    preprocessor = ResponsePreprocessor()
    result = preprocessor(ds)

    row = result[0]
    assert row["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "r1"},
        {"role": "user", "content": "Current question"},
        {"role": "assistant", "content": "Current answer"},
    ]


def test_response_column_aliases():
    """Columns like 'instruction' or 'answer' should auto-map."""
    ds = make_dataset(
        {
            "instruction": ["Translate hello to French"],
            "answer": ["Bonjour"],
        }
    )
    preprocessor = ResponsePreprocessor()
    result = preprocessor(ds)

    row = result[0]
    assert row["messages"][0]["content"] == "Translate hello to French"
    assert row["messages"][1]["content"] == "Bonjour"


# ============================================================
# MessagesPreprocessor
# ============================================================


def test_messages_standard():
    """Standard messages format passthrough."""
    ds = make_dataset(
        {
            "messages": [
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            ],
        }
    )
    preprocessor = MessagesPreprocessor()
    result = preprocessor(ds)

    row = result[0]
    assert row["messages"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_messages_sharegpt_format():
    """ShareGPT paired format conversion."""
    ds = make_dataset(
        {
            "conversations": [
                [
                    {"human": "What is AI?", "gpt": "Artificial Intelligence."},
                    {"human": "Thanks", "gpt": "You are welcome."},
                ]
            ],
        }
    )
    preprocessor = MessagesPreprocessor()
    result = preprocessor(ds)

    row = result[0]
    assert row["messages"] == [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "Artificial Intelligence."},
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "You are welcome."},
    ]


def test_messages_from_value_keys():
    """Messages with 'from'/'value' keys instead of 'role'/'content'."""
    ds = make_dataset(
        {
            "conversations": [
                [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi!"},
                ]
            ],
        }
    )
    preprocessor = MessagesPreprocessor()
    result = preprocessor(ds)

    row = result[0]
    assert row["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]


# ============================================================
# AlpacaPreprocessor
# ============================================================


def test_alpaca_basic():
    """Alpaca format: instruction + input + output."""
    ds = make_dataset(
        {
            "instruction": ["Translate the following sentence."],
            "input": ["Hello, how are you?"],
            "output": ["Bonjour, comment allez-vous?"],
        }
    )
    preprocessor = AlpacaPreprocessor()
    result = preprocessor(ds)

    row = result[0]
    assert row["messages"][0]["content"] == "Translate the following sentence.\nHello, how are you?"
    assert row["messages"][1]["content"] == "Bonjour, comment allez-vous?"


def test_alpaca_no_input():
    """Alpaca format with empty input."""
    ds = make_dataset(
        {
            "instruction": ["Tell me a joke."],
            "input": [""],
            "output": ["Why did the chicken cross the road?"],
        }
    )
    preprocessor = AlpacaPreprocessor()
    result = preprocessor(ds)

    row = result[0]
    assert row["messages"][0]["content"] == "Tell me a joke."


# ============================================================
# AutoPreprocessor
# ============================================================


def test_auto_detects_messages():
    """Auto should pick MessagesPreprocessor when 'messages' column exists."""
    ds = make_dataset(
        {
            "messages": [
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            ],
        }
    )
    result = AutoPreprocessor()(ds)
    assert result[0]["messages"][0]["role"] == "user"


def test_auto_detects_alpaca():
    """Auto should pick AlpacaPreprocessor when instruction+input columns exist."""
    ds = make_dataset(
        {
            "instruction": ["Say hi"],
            "input": [""],
            "output": ["Hi!"],
        }
    )
    result = AutoPreprocessor()(ds)
    assert result[0]["messages"][1]["content"] == "Hi!"


def test_auto_detects_response():
    """Auto should pick ResponsePreprocessor as fallback."""
    ds = make_dataset(
        {
            "query": ["Hello"],
            "response": ["Hi"],
        }
    )
    result = AutoPreprocessor()(ds)
    assert result[0]["messages"][0] == {"role": "user", "content": "Hello"}


# ============================================================
# Multimodal (images/videos preserved alongside messages)
# ============================================================


def test_response_with_images():
    """Images column should be preserved and normalized."""
    ds = make_dataset(
        {
            "query": ["Describe this image"],
            "response": ["A cat sitting on a mat"],
            "images": [["path/to/cat.jpg"]],
        }
    )
    preprocessor = ResponsePreprocessor()
    result = preprocessor(ds)

    row = result[0]
    assert row["messages"][0]["content"] == "Describe this image"
    assert row["images"] == [{"bytes": None, "path": "path/to/cat.jpg"}]


def test_messages_with_videos():
    """Videos column should be preserved and normalized."""
    ds = make_dataset(
        {
            "messages": [
                [
                    {"role": "user", "content": "What happens in this video?"},
                    {"role": "assistant", "content": "A dog is running."},
                ]
            ],
            "videos": [["path/to/dog.mp4"]],
        }
    )
    preprocessor = MessagesPreprocessor()
    result = preprocessor(ds)

    row = result[0]
    assert row["videos"] == ["path/to/dog.mp4"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
