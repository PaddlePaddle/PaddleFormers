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

from paddlenlp.transformers import AutoConfig, BertConfig, CLIPConfig, T5Config
from paddlenlp.utils.log import logger

from tests.testing_utils import slow


@unittest.skip("skipping due to connection error!")
class ConfigLoadTester(unittest.TestCase):
    @slow
    def test_bert_config_load(self):
        logger.info("Download Bert Config from PaddleFormers BOS")
        bert_config = BertConfig.from_pretrained("bert-base-uncased", from_hf_hub=False)
        bert_config = AutoConfig.from_pretrained("bert-base-uncased", from_hf_hub=False)

        logger.info("Download config from local")
        bert_config.save_pretrained("./paddlenlp-test-config/bert-base-uncased")
        bert_config = BertConfig.from_pretrained("./paddlenlp-test-config/bert-base-uncased")
        bert_config = AutoConfig.from_pretrained("./paddlenlp-test-config/bert-base-uncased")
        logger.info("Download config from local with subfolder")
        bert_config = BertConfig.from_pretrained("./paddlenlp-test-config", subfolder="bert-base-uncased")
        bert_config = AutoConfig.from_pretrained("./paddlenlp-test-config", subfolder="bert-base-uncased")

        logger.info("Download Bert Config from PaddleFormers BOS with subfolder")
        bert_config = BertConfig.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="bert-base-uncased", from_hf_hub=False
        )
        bert_config = AutoConfig.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="bert-base-uncased", from_hf_hub=False
        )

        logger.info("Download Bert Config from aistudio")
        bert_config = BertConfig.from_pretrained("aistudio/bert-base-uncased", from_aistudio=True)
        bert_config = AutoConfig.from_pretrained("aistudio/bert-base-uncased", from_aistudio=True)

    @slow
    def test_clip_config_load(self):
        logger.info("Download CLIP Config from PaddleFormers BOS")
        clip_config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=False)
        clip_config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32", from_hf_hub=False)

        logger.info("Download CLIP Config from local")
        clip_config.save_pretrained("./paddlenlp-test-config/clip-vit-base-patch32")
        clip_config = CLIPConfig.from_pretrained("./paddlenlp-test-config/clip-vit-base-patch32")
        clip_config = AutoConfig.from_pretrained("./paddlenlp-test-config/clip-vit-base-patch32")
        logger.info("Download CLIP Config from local with subfolder")
        clip_config = CLIPConfig.from_pretrained("./paddlenlp-test-config", subfolder="clip-vit-base-patch32")
        clip_config = AutoConfig.from_pretrained("./paddlenlp-test-config", subfolder="clip-vit-base-patch32")

        logger.info("Download CLIP Config from PaddleFormers BOS with subfolder")
        clip_config = CLIPConfig.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="clip-vit-base-patch32", from_hf_hub=False
        )
        clip_config = AutoConfig.from_pretrained(
            "baicai/paddlenlp-test-model", subfolder="clip-vit-base-patch32", from_hf_hub=False
        )

        logger.info("Download CLIP Config from aistudio")
        clip_config = CLIPConfig.from_pretrained("aistudio/clip-vit-base-patch32", from_aistudio=True)
        clip_config = AutoConfig.from_pretrained("aistudio/clip-vit-base-patch32", from_aistudio=True)

    @slow
    def test_t5_config_load(self):
        logger.info("Download T5 Config from PaddleFormers BOS")
        t5_config = T5Config.from_pretrained("t5-small", from_hf_hub=False)
        t5_config = AutoConfig.from_pretrained("t5-small", from_hf_hub=False)

        logger.info("Download T5 Config from PaddleFormers BOS with subfolder")
        t5_config = T5Config.from_pretrained("baicai/paddlenlp-test-model", subfolder="t5-small", from_hf_hub=False)
        t5_config = AutoConfig.from_pretrained("baicai/paddlenlp-test-model", subfolder="t5-small", from_hf_hub=False)
        logger.info("Download T5 Config from local")
        t5_config.save_pretrained("./paddlenlp-test-config/t5-small")
        t5_config = T5Config.from_pretrained("./paddlenlp-test-config/t5-small")
        t5_config = AutoConfig.from_pretrained("./paddlenlp-test-config/t5-small")

        logger.info("Download T5 Config from aistudio")
        t5_config = T5Config.from_pretrained("aistudio/t5-small", from_aistudio=True)
        t5_config = AutoConfig.from_pretrained("aistudio/t5-small", from_aistudio=True)
