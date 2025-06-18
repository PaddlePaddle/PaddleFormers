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

from paddleformers.transformers import AutoConfig, BertConfig
from paddleformers.utils.log import logger
from tests.testing_utils import slow


@unittest.skip("skipping due to connection error!")
class ConfigLoadTester(unittest.TestCase):
    @slow
    def test_bert_config_load(self):
        logger.info("Download Bert Config from PaddleFormers BOS")
        bert_config = BertConfig.from_pretrained("bert-base-uncased", from_hf_hub=False)
        bert_config = AutoConfig.from_pretrained("bert-base-uncased", from_hf_hub=False)

        logger.info("Download config from local")
        bert_config.save_pretrained("./paddleformers-test-config/bert-base-uncased")
        bert_config = BertConfig.from_pretrained("./paddleformers-test-config/bert-base-uncased")
        bert_config = AutoConfig.from_pretrained("./paddleformers-test-config/bert-base-uncased")
        logger.info("Download config from local with subfolder")
        bert_config = BertConfig.from_pretrained("./paddleformers-test-config", subfolder="bert-base-uncased")
        bert_config = AutoConfig.from_pretrained("./paddleformers-test-config", subfolder="bert-base-uncased")

        logger.info("Download Bert Config from PaddleFormers BOS with subfolder")
        bert_config = BertConfig.from_pretrained(
            "baicai/paddleformers-test-model", subfolder="bert-base-uncased", from_hf_hub=False
        )
        bert_config = AutoConfig.from_pretrained(
            "baicai/paddleformers-test-model", subfolder="bert-base-uncased", from_hf_hub=False
        )

        logger.info("Download Bert Config from aistudio")
        bert_config = BertConfig.from_pretrained("aistudio/bert-base-uncased", from_aistudio=True)
        bert_config = AutoConfig.from_pretrained("aistudio/bert-base-uncased", from_aistudio=True)
