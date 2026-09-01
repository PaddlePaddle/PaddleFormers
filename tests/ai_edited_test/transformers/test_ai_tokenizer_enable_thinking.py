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

"""Formal YAML enable_thinking:false must reach apply_chat_template and glm5_2."""

import inspect

from paddleformers.cli.train.sft.workflow import run_sft
from paddleformers.transformers.tokenizer_utils import PreTrainedTokenizer


def test_encode_chat_inputs_pass_enable_thinking_false():
    methods = (
        PreTrainedTokenizer._encode_chat_inputs_openai_format,
        PreTrainedTokenizer._encode_chat_inputs_oneturn,
        PreTrainedTokenizer._extract_non_learnable_parts,
        PreTrainedTokenizer._encode_chat_inputs,
    )
    total = 0
    for method in methods:
        source = inspect.getsource(method)
        count = source.count("enable_thinking=False")
        assert count >= 1, method.__name__
        total += count
    assert total >= 8


def test_sft_dataset_config_copies_generating_enable_thinking():
    source = inspect.getsource(run_sft)
    assert '"enable_thinking": getattr(generating_args, "enable_thinking", None)' in source
