# coding=utf-8
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2019 Hugging Face inc.
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
import shutil
import sys
import unittest

import paddle as paddle_shadow

sys.modules["paddle"] = None
from paddleformers.transformers import AutoImageProcessor

sys.modules["paddle"] = paddle_shadow


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dirs = [
            "./slow_image_processor",
        ]
        for test_dir in self.test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def tearDown(self):
        for test_dir in self.test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_slow_image_processor_from_pretrained(self):
        image_processor = AutoImageProcessor.from_pretrained("PaddleFormers/tiny-random-qwen2.5vl")
        if hasattr(image_processor, "use"):
            self.assertFalse(image_processor.is_fast)
        else:
            self.assertNotIn("Fast", image_processor.__class__.__name__)
        self.assertEqual(image_processor.__class__.__name__, "Qwen2VLImageProcessor")

    def test_slow_image_processor_save_pretrained(self):
        image_processor = AutoImageProcessor.from_pretrained("PaddleFormers/tiny-random-qwen2.5vl")
        image_processor.min_pixels = 2048
        image_processor.save_pretrained("./slow_image_processor")
        self.assertTrue(os.path.exists("./slow_image_processor/preprocessor_config.json"))
