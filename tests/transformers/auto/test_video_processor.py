# coding=utf-8
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import json
import tempfile
import unittest
from pathlib import Path

from paddleformers.transformers import AutoVideoProcessor, Qwen2VLVideoProcessor


class AutoVideoProcessorTest(unittest.TestCase):
    def test_video_processor_from_pretrained(self):
        processor = AutoVideoProcessor.from_pretrained("PaddleFormers/tiny-random-qwen25vlv2")  # noqa
        # self.assertIsInstance(processor, Qwen2VLVideoProcessor)
        # self.assertEqual(processor.__class__.__name__, 'Qwen2VLVideoProcessor')

    def test_video_processor_from_local_directory_from_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor_tmpfile = Path(tmpdir) / "video_preprocessor_config.json"
            config_tmpfile = Path(tmpdir) / "config.json"
            json.dump(
                {
                    "video_processor_type": "Qwen2VLVideoProcessor",
                    "processor_class": "Qwen2VLProcessor",
                },
                open(processor_tmpfile, "w"),
            )
            json.dump({"model_type": "qwen2_vl"}, open(config_tmpfile, "w"))

            config = AutoVideoProcessor.from_pretrained(tmpdir)
            # self.assertIsInstance(config, Qwen2VLVideoProcessor)
            self.assertEqual(config.__class__.__name__, "Qwen2VLVideoProcessor")

    def test_video_processor_from_local_directory_from_preprocessor_key(self):
        # Ensure we can load the image processor from the feature extractor config
        with tempfile.TemporaryDirectory() as tmpdir:
            processor_tmpfile = Path(tmpdir) / "preprocessor_config.json"
            config_tmpfile = Path(tmpdir) / "config.json"
            json.dump(
                {
                    "video_processor_type": "Qwen2VLVideoProcessor",
                    "processor_class": "Qwen2VLProcessor",
                },
                open(processor_tmpfile, "w"),
            )
            json.dump({"model_type": "qwen2_vl"}, open(config_tmpfile, "w"))

            config = AutoVideoProcessor.from_pretrained(tmpdir)
            # self.assertIsInstance(config, Qwen2VLVideoProcessor)
            self.assertEqual(config.__class__.__name__, "Qwen2VLVideoProcessor")

    def test_video_processor_from_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor_tmpfile = Path(tmpdir) / "video_preprocessor_config.json"
            json.dump(
                {
                    "video_processor_type": "Qwen2VLVideoProcessor",
                    "processor_class": "Qwen2VLProcessor",
                },
                open(processor_tmpfile, "w"),
            )

            config = AutoVideoProcessor.from_pretrained(processor_tmpfile)
            # self.assertIsInstance(config, Qwen2VLVideoProcessor)
            self.assertEqual(config.__class__.__name__, "Qwen2VLVideoProcessor")

    def test_video_processor_save_pretrained(self):
        config_dict = AutoVideoProcessor.from_pretrained("PaddleFormers/tiny-random-qwen25vlv2").to_dict()
        config_dict.pop("video_processor_type")
        config = Qwen2VLVideoProcessor(**config_dict)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            video_processor = AutoVideoProcessor.from_pretrained(tmpdir)
            dict_as_saved = json.loads(video_processor.to_json_string())
            self.assertTrue("_processor_class" not in dict_as_saved)
            self.assertEqual(video_processor.__class__.__name__, "Qwen2VLVideoProcessor")
