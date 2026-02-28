# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 Hugging Face inc.
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
import tempfile
import unittest

from paddleformers.transformers import AutoFeatureExtractor, WhisperFeatureExtractor


class AutoFeatureExtractorTest(unittest.TestCase):
    def test_feature_extraction_from_pretrained(self):
        processor = AutoFeatureExtractor.from_pretrained(
            "PaddleFormers/tiny-random-qwen3omni", download_hub="aistudio"
        )
        self.assertIsInstance(processor, WhisperFeatureExtractor)

    def test_feature_extraction_save_pretrained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "PaddleFormers/tiny-random-qwen3omni", download_hub="aistudio"
            )
            feature_extractor.save_pretrained(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "preprocessor_config.json")))
