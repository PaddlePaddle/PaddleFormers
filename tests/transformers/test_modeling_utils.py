# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from tempfile import TemporaryDirectory

from paddleformers.transformers import BertModel
from paddleformers.utils.env import CONFIG_NAME, PADDLE_WEIGHTS_NAME
from tests.testing_utils import slow


def download_bert_model(model_name: str):
    """set the global method: multiprocessing can not pickle local method

    Args:
        model_name (str): the model name
    """

    model = BertModel.from_pretrained(model_name)
    # free the model resource
    del model


class TestModeling(unittest.TestCase):
    """Test PretrainedModel single time, not in Transformer models"""

    def test_from_pretrained_cache_dir_community_model(self):
        model_name = "Paddleformers/tiny-random-bert"
        with TemporaryDirectory() as tempdir:
            BertModel.from_pretrained(model_name, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, CONFIG_NAME)))
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, PADDLE_WEIGHTS_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))

    @slow
    def test_from_pretrained_cache_dir_pretrained_init(self):
        model_name = "PaddleFormers/tiny-random-bert"
        with TemporaryDirectory() as tempdir:
            BertModel.from_pretrained(model_name, cache_dir=tempdir)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, CONFIG_NAME)))
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, PADDLE_WEIGHTS_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))
