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

import shutil
import tempfile
import unittest

from paddleformers.transformers import AutoProcessor, Qwen3OmniMoeProcessor
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_processing_common import ProcessorTesterMixin


class Qwen3_Omni_ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen3OmniMoeProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()

        processor = Qwen3OmniMoeProcessor.from_pretrained(
            "PaddleFormers/tiny-random-qwen3omni", download_hub="aistudio"
        )

        processor.save_pretrained(cls.tmpdir)
        cls.image_token = processor.image_token
        # Use GPU 0 to prevent CUDA illegal memory access during resize

    @gpu_device_initializer(log_prefix="Qwen3_Omni_ProcessorTest", gpu_id=0)
    def setUp(self):
        pass

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs).image_processor

    def get_video_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs).video_processor

    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs).feature_extractor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()
        video_processor = self.get_video_processor()
        feature_extractor = self.get_feature_extractor()
        processor = Qwen3OmniMoeProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
        )
        processor.save_pretrained(self.tmpdir)
        processor = Qwen3OmniMoeProcessor.from_pretrained(self.tmpdir)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor.image_processor.__class__.__name__, "Qwen2VLImageProcessorFast")
        self.assertEqual(processor.feature_extractor.__class__.__name__, "WhisperFeatureExtractor")
        self.assertEqual(processor.video_processor.__class__.__name__, "Qwen2VLVideoProcessor")

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        video_processor = self.get_video_processor()
        feature_extractor = self.get_feature_extractor()
        processor = Qwen3OmniMoeProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
        )

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="pd")
        input_processor = processor(images=image_input, text="dummy", return_tensors="pd")

        for key in input_image_proc:
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        video_processor = self.get_video_processor()
        feature_extractor = self.get_feature_extractor()
        processor = Qwen3OmniMoeProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
        )

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input, return_tensors="pd")

        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"])

        # test if it raises when no input is passed
        with self.assertRaises(ValueError):
            processor()

        # test if it raises when no text is passed
        with self.assertRaises(ValueError):
            processor(images=image_input, return_tensors="pd")

    @unittest.skip("qwen3 omni do not support video input")
    def test_apply_chat_template_video_frame_sampling(self):
        pass

    @unittest.skip("qwen3 omni do not support image input")
    def test_overlapping_text_image_kwargs_handling(self):
        pass

    def test_structured_kwargs_nested_from_dict_video(self):
        pass

    def test_structured_kwargs_nested_video(self):
        pass

    def test_unstructured_kwargs_video(self):
        pass

    def test_kwargs_overrides_default_video_processor_kwargs(self):
        pass

    def test_tokenizer_defaults_preserved_by_kwargs_video(self):
        pass

    def test_video_processor_defaults_preserved_by_video_kwargs(self):
        pass

    def test_model_input_names(self):
        pass
