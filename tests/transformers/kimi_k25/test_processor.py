# coding=utf-8
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

from __future__ import annotations

import inspect
import io
import shutil
import tempfile
import unittest

import paddle
import requests
from PIL import Image

from paddleformers.transformers import AutoProcessor, KimiK25Processor
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_processing_common import ProcessorTesterMixin


class KimiK25ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = KimiK25Processor

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        processor = KimiK25Processor.from_pretrained("PaddleFormers/tiny-random-kimi-k25")
        processor.save_pretrained(cls.tmpdir)

    # Use GPU 0 to prevent CUDA illegal memory access during resize
    @gpu_device_initializer(log_prefix="KimiK25ProcessorTest", gpu_id=0)
    def setUp(self):
        pass

    def prepare_image_inputs(self, batch_size: int | None = None):
        image_input = super().prepare_image_inputs()
        # Default batch size is 1
        if batch_size is None:
            batch_size = 1
        image_inputs = [{"type": "image", "image": image_input}] * batch_size
        return image_inputs

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs).image_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        vision_processor = self.get_image_processor()

        media = self.prepare_image_inputs()[0]
        output = vision_processor.get_resize_config(media)
        self.assertTrue("num_tokens" in output)
        self.assertEqual(output["num_tokens"], 30)

        self.assertTrue("new_width" in output)
        self.assertEqual(output["new_width"], 400)

        self.assertTrue("new_height" in output)
        self.assertEqual(output["new_height"], 30)

        self.assertTrue("pad_width" in output)
        self.assertEqual(output["pad_width"], 20)

        self.assertTrue("pad_height" in output)
        self.assertEqual(output["pad_height"], 26)

        self.assertTrue("sampled_nframes" in output)
        self.assertEqual(output["sampled_nframes"], 1)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()

        processor = KimiK25Processor(tokenizer=tokenizer, image_processor=image_processor)
        processor.save_pretrained(self.tmpdir)
        processor = KimiK25Processor.from_pretrained(self.tmpdir)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor.tokenizer.__class__.__name__, "TikTokenTokenizer")
        self.assertEqual(processor.image_processor.__class__.__name__, "KimiK25VisionProcessor")

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = KimiK25Processor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="pd")
        input_processor = processor(medias=image_input, text="dummy", return_tensors="pd")

        for key in input_image_proc:
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = KimiK25Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, medias=image_input, return_tensors="pd")

        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask", "pixel_values", "grid_thws"])

        # test if it raises when no input is passed
        with self.assertRaises(ValueError):
            processor()

        # test if it raises when no text or medias is passed
        with self.assertRaises(ValueError):
            processor(text=input_str, return_tensors="pd")
        with self.assertRaises(ValueError):
            processor(medias=image_input, return_tensors="pd")

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image"])
        image_input = self.prepare_image_inputs()
        inputs_dict = {"text": text, "medias": image_input}

        call_signature = inspect.signature(processor.__call__)
        input_args = [param.name for param in call_signature.parameters.values()]
        inputs_dict = {k: v for k, v in inputs_dict.items() if k in input_args}

        inputs = processor(**inputs_dict, return_tensors="pd")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_image_inputs(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                },
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        ).input_ids
        expected_output = processor.tokenizer(
            formatted_prompt, add_special_tokens=False, return_tensors=None
        ).input_ids
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertListEqual(list(out_dict.keys()), ["input_ids", "attention_mask"])

        url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
        image = Image.open(io.BytesIO(requests.get(url).content))

        # Add video URL for return dict and load with `num_frames` arg
        messages[0][0]["content"][0] = {
            "type": "image",
            "image_url": image,
        }
        output = processor(messages[0])

        EXPECTED_INPUT_IDS = paddle.to_tensor([163587, 2482, 163601, 163602, 4017, 163603, 163605, 163604, 198])
        EXPECTED_PIXEL_SLICE = paddle.to_tensor(
            [
                0.69411778,
                0.69411778,
                0.69411778,
                0.69411778,
                0.69411778,
                0.69411778,
                0.69411778,
                0.69411778,
                0.69411778,
                0.69411778,
            ]
        )
        EXPECTED_IMAGE_GRID_THW = [[1, 64, 94]]

        self.assertIsInstance(output["input_ids"], paddle.Tensor)
        self.assertEqual(output["input_ids"].shape, [1, 21])
        self.assertTrue(paddle.allclose(output["input_ids"][0, :9], EXPECTED_INPUT_IDS))
        self.assertIsInstance(output["pixel_values"], paddle.Tensor)
        self.assertEqual(output["pixel_values"].shape, [6016, 3, 14, 14])
        self.assertTrue(paddle.allclose(output["pixel_values"][0, 0, 0, 0:10], EXPECTED_PIXEL_SLICE))
        self.assertEqual(output["grid_thws"].shape, [1, 3])
        self.assertEqual(output["grid_thws"].tolist(), EXPECTED_IMAGE_GRID_THW)

    def test_video_frame_sampling(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_chunk"},
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        ).input_ids
        expected_output = processor.tokenizer(
            formatted_prompt, add_special_tokens=False, return_tensors=None
        ).input_ids
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertListEqual(list(out_dict.keys()), ["input_ids", "attention_mask"])

        # Add video URL for return dict and load with `num_frames` arg
        messages[0][0]["content"][0] = {
            "type": "video_url",
            "video_url": {"url": "http://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"},
        }
        output = processor(messages[0])

        EXPECTED_INPUT_IDS = paddle.to_tensor([163587, 2482, 163601, 465, 25, 465, 25, 465, 13])
        EXPECTED_PIXEL_SLICE = paddle.to_tensor(
            [
                -0.15294111,
                -0.15294111,
                -0.14509797,
                -0.14509797,
                -0.14509797,
                -0.13725483,
                -0.13725483,
                -0.13725483,
                -0.13725483,
                -0.13725483,
            ]
        )
        EXPECTED_IMAGE_GRID_THW = [[4, 48, 86]] * 4

        self.assertIsInstance(output["input_ids"], paddle.Tensor)
        self.assertEqual(output["input_ids"].shape, [1, 64])
        self.assertTrue(paddle.allclose(output["input_ids"][0, :9], EXPECTED_INPUT_IDS))
        self.assertIsInstance(output["pixel_values"], paddle.Tensor)
        self.assertEqual(output["pixel_values"].shape, [66048, 3, 14, 14])
        self.assertTrue(paddle.allclose(output["pixel_values"][0, 0, 0, 0:10], EXPECTED_PIXEL_SLICE))
        self.assertEqual(output["grid_thws"].shape, [4, 3])
        self.assertEqual(output["grid_thws"].tolist(), EXPECTED_IMAGE_GRID_THW)

        # set `num_frames_per_chunk` into different values
        processor = self.get_processor()
        processor.image_processor.media_proc_cfg["temporal_merge_kernel_size"] = 6
        output = processor(messages[0])

        EXPECTED_INPUT_IDS = paddle.to_tensor([163587, 2482, 163601, 465, 25, 465, 25, 465, 13])
        EXPECTED_PIXEL_SLICE = paddle.to_tensor(
            [
                -0.15294111,
                -0.15294111,
                -0.14509797,
                -0.14509797,
                -0.14509797,
                -0.13725483,
                -0.13725483,
                -0.13725483,
                -0.13725483,
                -0.13725483,
            ]
        )
        EXPECTED_IMAGE_GRID_THW = [[6, 48, 86]] * 3

        self.assertIsInstance(output["input_ids"], paddle.Tensor)
        self.assertEqual(output["input_ids"].shape, [1, 52])
        self.assertTrue(paddle.allclose(output["input_ids"][0, :9], EXPECTED_INPUT_IDS))
        self.assertIsInstance(output["pixel_values"], paddle.Tensor)
        self.assertEqual(output["pixel_values"].shape, [74304, 3, 14, 14])
        self.assertTrue(paddle.allclose(output["pixel_values"][0, 0, 0, 0:10], EXPECTED_PIXEL_SLICE))
        self.assertEqual(output["grid_thws"].shape, [3, 3])
        self.assertEqual(output["grid_thws"].tolist(), EXPECTED_IMAGE_GRID_THW)

    def test_special_mm_token_truncation(self):
        """Tests that special vision tokens do not get truncated when `truncation=True` is set."""

        processor = self.get_processor()

        input_str = self.prepare_text_inputs(batch_size=2, modalities="image")
        image_input = self.prepare_image_inputs(batch_size=2)

        _ = processor(
            text=input_str,
            medias=image_input,
            return_tensors="pd",
            truncation=True,
            padding=True,
            max_length=20,
        )

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                medias=image_input,
                return_tensors="pd",
            )

    def test_tokenizer_defaults_preserved_by_kwargs(self):
        pass

    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        pass

    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        pass

    def test_kwargs_overrides_default_image_processor_kwargs(self):
        pass

    def test_unstructured_kwargs(self):
        pass

    def test_unstructured_kwargs_batched(self):
        pass

    def test_doubly_passed_kwargs(self):
        pass

    def test_args_overlap_kwargs(self):
        pass

    def test_structured_kwargs_nested(self):
        pass

    def test_structured_kwargs_nested_from_dict(self):
        pass

    def test_tokenizer_defaults_preserved_by_kwargs_video(self):
        pass

    def test_video_processor_defaults_preserved_by_video_kwargs(self):
        pass

    def test_kwargs_overrides_default_tokenizer_kwargs_video(self):
        pass

    def test_kwargs_overrides_default_video_processor_kwargs(self):
        pass

    def test_unstructured_kwargs_video(self):
        pass

    def test_unstructured_kwargs_batched_video(self):
        pass

    def test_doubly_passed_kwargs_video(self):
        pass

    def test_structured_kwargs_nested_video(self):
        pass

    def test_structured_kwargs_nested_from_dict_video(self):
        pass

    def test_overlapping_text_image_kwargs_handling(self):
        pass

    def test_prepare_and_validate_optional_call_args(self):
        pass

    def test_chat_template_save_loading(self):
        pass

    def test_apply_chat_template_video_frame_sampling(self):
        pass

    def test_apply_chat_template_assistant_mask(self):
        pass
