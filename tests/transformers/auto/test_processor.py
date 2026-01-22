# coding=utf-8
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 the HuggingFace Inc. team.
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

import tempfile
import unittest

from paddleformers.transformers import AutoProcessor, Qwen2_5_VLProcessor


class AutoProcessorTest(unittest.TestCase):
    def test_video_processor_from_pretrained(self):
        processor = AutoProcessor.from_pretrained("PaddleFormers/tiny-random-qwen25vlv2")
        self.assertIsInstance(processor, Qwen2_5_VLProcessor)

    def test_auto_processor_load_tokenizer(self):
        processor = AutoProcessor.from_pretrained("PaddleFormers/tiny-random-qwen25vlv2")
        self.assertEqual(processor.tokenizer.__class__.__name__, "Qwen2TokenizerFast")

    def test_auto_processor_load_image_processor(self):
        processor = AutoProcessor.from_pretrained("PaddleFormers/tiny-random-qwen25vlv2")
        self.assertEqual(processor.image_processor.__class__.__name__, "Qwen2VLImageProcessorFast")

    def test_auto_processor_load_video_processor(self):
        processor = AutoProcessor.from_pretrained("PaddleFormers/tiny-random-qwen25vlv2")
        self.assertEqual(processor.video_processor.__class__.__name__, "Qwen2VLVideoProcessor")

    def test_auto_processor_save_load(self):
        processor = AutoProcessor.from_pretrained("PaddleFormers/tiny-random-qwen25vlv2")
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            second_processor = AutoProcessor.from_pretrained(tmp_dir)
            self.assertEqual(second_processor.__class__.__name__, processor.__class__.__name__)
            self.assertEqual(second_processor.tokenizer.__class__.__name__, processor.tokenizer.__class__.__name__)
            self.assertEqual(
                second_processor.image_processor.__class__.__name__, processor.image_processor.__class__.__name__
            )
            self.assertEqual(
                second_processor.video_processor.__class__.__name__, processor.video_processor.__class__.__name__
            )
