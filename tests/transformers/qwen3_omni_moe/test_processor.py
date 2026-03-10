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

import inspect
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

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image", "video", "audio"])
        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()
        audio_inputs = self.prepare_audio_inputs()
        inputs_dict = {"text": text, "images": image_input, "videos": video_inputs, "audio": audio_inputs}

        call_signature = inspect.signature(processor.__call__)
        input_args = [param.name for param in call_signature.parameters.values()]
        inputs_dict = {k: v for k, v in inputs_dict.items() if k in input_args}

        inputs = processor(**inputs_dict, return_tensors="pd")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_apply_chat_template_video_frame_sampling(self):
        pass
        # processor = self.get_processor()
        # if processor.chat_template is None:
        #     self.skipTest("Processor has no chat template")

        # signature = inspect.signature(processor.__call__)
        # if "videos" not in {*signature.parameters.keys()} or (
        #     signature.parameters.get("videos") is not None
        #     and signature.parameters["videos"].annotation == inspect._empty
        # ):
        #     self.skipTest("Processor doesn't accept videos at input")

        # messages = [
        #     [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": "What is shown in this video?"},
        #             ],
        #         },
        #     ]
        # ]

        # formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # self.assertEqual(len(formatted_prompt), 1)

        # formatted_prompt_tokenized = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        # expected_output = processor.tokenizer(formatted_prompt, return_tensors=None).input_ids
        # self.assertListEqual(expected_output, formatted_prompt_tokenized)

        # out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        # self.assertListEqual(list(out_dict.keys()), ["input_ids", "attention_mask"])

        # # Add video URL for return dict and load with `num_frames` arg
        # messages[0][0]["content"][0] = {
        #     "type": "video",
        #     "url": "http://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4",
        # }
        # num_frames = 3
        # out_dict_with_video = processor.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     tokenize=True,
        #     return_dict=True,
        #     num_frames=num_frames,
        # )
        # self.assertTrue(self.videos_input_name in out_dict_with_video)
        # self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 5760)

        # # Load with `fps` arg
        # fps = 1
        # out_dict_with_video = processor.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     tokenize=True,
        #     return_dict=True,
        #     fps=fps,
        # )
        # self.assertTrue(self.videos_input_name in out_dict_with_video)
        # self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 11520)

        # # Load with `fps` and `num_frames` args, should raise an error
        # with self.assertRaises(ValueError):
        #     out_dict_with_video = processor.apply_chat_template(
        #         messages,
        #         add_generation_prompt=True,
        #         tokenize=True,
        #         return_dict=True,
        #         fps=fps,
        #         num_frames=num_frames,
        #     )

        # # Load without any arg should load the whole video
        # out_dict_with_video = processor.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     tokenize=True,
        #     return_dict=True,
        # )
        # self.assertTrue(self.videos_input_name in out_dict_with_video)
        # self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 380160)

        # # Load video as a list of frames (i.e. images). NOTE: each frame should have same size
        # # because we assume they come from one video
        # messages[0][0]["content"][0] = {
        #     "type": "video",
        #     "url": [
        #         "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
        #         "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
        #     ],
        # }
        # out_dict_with_video = processor.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     tokenize=True,
        #     return_dict=True,
        # )
        # self.assertTrue(self.videos_input_name in out_dict_with_video)
        # self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 5808)

        # # When the inputs are frame URLs/paths we expect that those are already
        # # sampled and will raise an error is asked to sample again.
        # with self.assertRaises(ValueError):
        #     out_dict_with_video = processor.apply_chat_template(
        #         messages,
        #         add_generation_prompt=True,
        #         tokenize=True,
        #         return_dict=True,
        #         do_sample_frames=True,
        #         num_frames=num_frames,
        #     )
