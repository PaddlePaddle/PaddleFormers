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

import os
import unittest

from paddleformers.cli.utils.process import add_new_special_tokens
from paddleformers.datasets.loader import create_dataset as create_dataset_sft
from paddleformers.datasets.template.template import get_template_and_fix_tokenizer
from paddleformers.transformers import (
    AutoProcessor,
    AutoTokenizer,
    Llama3Tokenizer,
    LlamaTokenizer,
)
from tests.testing_utils import get_tests_dir

EXPECTED_NO_TEMPLATE = {
    "input": "针对产品发布提出五种营销策略。1. 社交媒体活动。\n2. 电子邮件营销。\n3. 在线和离线广告。\n4. 推荐和评论。\n5. 合作名人推销。",
    "label": "1. 社交媒体活动。\n2. 电子邮件营销。\n3. 在线和离线广告。\n4. 推荐和评论。\n5. 合作名人推销。",
    "position_ids": list(range(0, 44)),
}

EXPECTED_WITH_TEMPLATE = {
    "input": "[gMASK]<sop>[gMASK]<sop><|user|>\n针对产品发布提出五种营销策略。<|assistant|>\n<think></think>\n1. 社交媒体活动。\n2. 电子邮件营销。\n3. 在线和离线广告。\n4. 推荐和评论。\n5. 合作名人推销。<|user|>",
    "label": "<think></think>\n1. 社交媒体活动。\n2. 电子邮件营销。\n3. 在线和离线广告。\n4. 推荐和评论。\n5. 合作名人推销。<|user|>",
    "position_ids": list(range(0, 56)),
}

MODEL_NAME_OR_PATH = "/home/models/PaddleFormers/tiny-random-glm4moe-bf16/"
MAX_SEQ_LEN = 8192
SEED = 23


class TestUseTemplate(unittest.TestCase):
    def _build_dataset(self, use_template):
        """Build a dataset with the given use_template flag and return the first sample."""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        add_new_special_tokens(tokenizer, None)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if isinstance(tokenizer, (LlamaTokenizer, Llama3Tokenizer)):
            tokenizer.pad_token_id = tokenizer.eos_token_id

        processor = AutoProcessor.from_pretrained(MODEL_NAME_OR_PATH, use_fast=None)

        dataset_config = {
            "tokenizer": tokenizer,
            "processor": processor,
            "max_seq_len": MAX_SEQ_LEN,
            "random_seed": SEED,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": 6000000,
            "random_shuffle": False,
            "greedy_intokens": False,
            "packing": False,
            "mix_strategy": "concat",
            "encode_one_turn": False,
            "use_template": use_template,
            "is_pretraining": False,
            "truncate_packing": False,
            "stage": "sft",
            "template_backend": "custom",
            "split_multi_turn": False,
            "dataset_type": "iterable",
            "truncation_strategy": "delete",
            "dtype": "bfloat16",
            "dataset_num_proc": 1,
            "binpacking": True,
            "packing_interval": 1000,
            "dataloader_num_workers": 0,
            "template": "glm4_moe",
            "tool_format": None,
            "default_system": None,
        }

        if dataset_config["template_backend"] == "custom":
            template_instance = get_template_and_fix_tokenizer(dataset_config)
        else:
            template_instance = None
        dataset_config["template_instance"] = template_instance

        dataset_dir = get_tests_dir(os.path.join("fixtures", "dummy"))
        dataset_path = os.path.join(dataset_dir, "sft", "train.jsonl")

        train_dataset = create_dataset_sft(
            task_group=dataset_path,
            task_group_prob="1.0",
            sub_dataset_type="erniekit",
            **dataset_config,
        )

        data_sample = next(iter(train_dataset))
        decoded_input = tokenizer.decode(data_sample[0].token_ids)
        decoded_label = tokenizer.decode([item for item in data_sample[0].labels if item != -100])
        position_ids = data_sample[0].position_ids
        return {"input": decoded_input, "label": decoded_label, "position_ids": position_ids}

    def test_use_no_template(self):
        result = self._build_dataset(use_template=False)
        self.assertEqual(result, EXPECTED_NO_TEMPLATE)

    def test_use_template(self):
        result = self._build_dataset(use_template=True)
        self.assertEqual(result, EXPECTED_WITH_TEMPLATE)
