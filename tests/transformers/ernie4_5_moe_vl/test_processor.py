# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import shutil
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import AutoProcessor, Ernie4_5_VLProcessor
from tests.transformers.test_processing_common import ProcessorTesterMixin


class Ernie4_5_VLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Ernie4_5_VLProcessor
    images_input_name = "images"
    videos_input_name = "images"
    image_token = "<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>"
    video_token = "<|VIDEO_START|><|video@placeholder|><|VIDEO_END|>"

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        processor = Ernie4_5_VLProcessor.from_pretrained(
            "PaddleFormers/tiny_random_ernie4_5_vl",
            download_hub="aistudio",
            patch_size=4,
            max_pixels=56 * 56,
            min_pixels=28 * 28,
        )
        processor.save_pretrained(cls.tmpdir)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs).image_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs)

    def prepare_image_inputs(self, batch_size: int | None = None):
        image = super().prepare_image_inputs()
        if batch_size is None:
            return [image]
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        return [image] * batch_size

    def prepare_video_inputs(self, batch_size: int | None = None):
        """This function prepares a list of numpy videos."""
        video_input = [self.prepare_image_inputs(batch_size=8)]
        if batch_size is None:
            return video_input
        return [video_input] * batch_size

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()

        processor = Ernie4_5_VLProcessor(tokenizer=tokenizer, image_processor=image_processor)
        processor.save_pretrained(self.tmpdir)
        processor = Ernie4_5_VLProcessor.from_pretrained(self.tmpdir)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor.tokenizer.__class__.__name__, "Ernie4_5_VLTokenizer")
        self.assertEqual(processor.image_processor.__class__.__name__, "Ernie4_5_VLImageProcessor")

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Ernie4_5_VLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(
            images=[image_input[0].convert("RGB")],
            do_normalize=False,
            do_rescale=False,
            predetermined_grid_thw=np.array([[8, 100]]),
            do_convert_rgb=True,
            return_tensors="pd",
        )
        input_processor = processor(text=self.image_token, images=image_input, return_tensors="pd")

        self.assertAlmostEqual(
            input_image_proc["pixel_values"].astype("float32").sum(),
            input_processor["images"].astype("float32").sum(),
            delta=1e-2,
        )
        self.assertAlmostEqual(input_image_proc["image_grid_thw"].sum(), input_processor["grid_thw"].sum(), delta=1e-2)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Ernie4_5_VLProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer" + self.image_token
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input, return_tensors="pd")

        self.assertListEqual(
            list(inputs.keys()),
            ["input_ids", "token_type_ids", "position_ids", "images", "grid_thw", "image_type_ids"],
        )

        # test if it raises when no input is passed
        with self.assertRaises(AttributeError):
            processor()

        # test if it raises when no text is passed
        with self.assertRaises(AttributeError):
            processor(images=image_input, return_tensors="pd")

    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list[str],
    ):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        if processor_name not in self.processor_class.attributes:
            self.skipTest(f"{processor_name} attribute not present in {self.processor_class}")

        batch_messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe this."}],
                },
            ]
        ] * batch_size

        # Test that jinja can be applied
        formatted_prompt = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), batch_size)

        # Test that tokenizing with template and directly with `self.tokenizer` gives same output
        formatted_prompt_tokenized = processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=True, return_tensors=return_tensors
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        tok_output = processor.tokenizer(
            formatted_prompt, return_tensors=return_tensors, add_special_tokens=add_special_tokens
        )
        expected_output = tok_output.input_ids
        self.assertListEqual(expected_output.tolist(), formatted_prompt_tokenized.tolist())

        # Test that kwargs passed to processor's `__call__` are actually used
        tokenized_prompt_100 = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
            max_length=100,
        )
        self.assertEqual(len(tokenized_prompt_100[0]), 100)

        # Test that `return_dict=True` returns text related inputs in the dict
        out_dict_text = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )
        self.assertTrue(all(key in out_dict_text for key in ["input_ids", "attention_mask"]))
        self.assertEqual(len(out_dict_text["input_ids"]), batch_size)
        self.assertEqual(len(out_dict_text["attention_mask"]), batch_size)

        # Test that with modality URLs and `return_dict=True`, we get modality inputs in the dict
        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx][0]["content"] = [batch_messages[idx][0]["content"][0], {"type": modality, "url": url}]

        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
            num_frames=2,  # by default no more than 2 frames, otherwise too slow
        )
        input_name = getattr(self, input_name)
        self.assertTrue(input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)
        if modality == "video":
            # qwen pixels don't scale with bs same way as other models, calculate expected video token count based on video_grid_thw
            expected_video_token_count = 0
            for thw in out_dict["video_grid_thw"]:
                expected_video_token_count += thw[0] * thw[1] * thw[2]
            mm_len = expected_video_token_count
        else:
            mm_len = batch_size * 192
        self.assertEqual(len(out_dict[input_name]), mm_len)

        return_tensor_to_type = {"pd": paddle.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])

    def test_apply_chat_template_video_frame_sampling(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest("Processor doesn't accept videos at input")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4",
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"
                        },
                    },
                    {"type": "text", "text": "What is shown in this video?"},
                ],
            },
        ]
        formatted_prompt_tokenized = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        expected_output = processor.tokenizer(formatted_prompt[0], return_tensors=None).input_ids
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = processor.process_vision_info(messages)
        out_dict = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )
        self.assertListEqual(
            list(out_dict.keys()),
            ["input_ids", "token_type_ids", "position_ids", "images", "grid_thw", "image_type_ids"],
        )

        # Add video URL for return dict and load with `num_frames` arg
        target_frames = 3
        messages[0]["content"][0] = {
            "type": "video_url",
            "video_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"},
            "target_frames": target_frames,
            "min_frames": target_frames,
        }
        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = processor.process_vision_info(messages)
        out_dict_with_video = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 231840)

        # Load with `fps` arg
        fps = 1
        messages[0]["content"][0] = {
            "type": "video_url",
            "video_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"},
            "fps": fps,
        }
        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = processor.process_vision_info(messages)
        out_dict_with_video = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 927360)

        # Load with `fps` and `num_frames` args, should raise an error
        with self.assertRaises(ValueError):
            messages[0]["content"][0] = {
                "type": "video_url",
                "video_url": {
                    "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"
                },
                "fps": fps,
                "target_frames": target_frames,
            }
            text = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = processor.process_vision_info(messages)
            out_dict_with_video = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pd",
            )

        # Load without any arg should load the whole video
        messages[0]["content"][0] = {
            "type": "video_url",
            "video_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"},
        }
        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = processor.process_vision_info(messages)
        out_dict_with_video = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1043280)

        # Load video as a list of frames (i.e. images)
        messages[0]["content"][0] = {
            "type": "video_url",
            "video_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"},
        }
        messages[0]["content"].append(
            {
                "type": "video_url",
                "video_url": {
                    "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4"
                },
            }
        )
        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = processor.process_vision_info(messages)
        out_dict_with_video = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 2086560)

    def test_kwargs_overrides_custom_image_processor_kwargs(self):
        processor = self.get_processor()
        # self.skip_processor_without_typed_kwargs(processor)

        input_str = self.image_token + self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input, return_tensors="pd")
        self.assertEqual(inputs[self.images_input_name].shape[0], 800)
        inputs = processor(text=input_str, images=image_input, max_pixels=56 * 56 * 4, return_tensors="pd")
        self.assertEqual(inputs[self.images_input_name].shape[0], 800)

    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pd",
        )

        self.assertEqual(inputs[self.images_input_name].dtype.name, "UINT8")

    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {"return_tensors": "pd"}

        inputs = processor(text=input_str, images=image_input, **all_kwargs)

        self.assertEqual(inputs[self.images_input_name].dtype.name, "UINT8")

    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {"return_tensors": "pd"}

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs[self.images_input_name].dtype.name, "UINT8")

    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_image_inputs()

        inputs = processor(
            text=input_str, images=image_input, do_rescale=True, rescale_factor=-1.0, return_tensors="pd"
        )
        self.assertEqual(inputs[self.images_input_name].dtype.name, "UINT8")

    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        """
        We use do_rescale=True, rescale_factor=-1.0 to ensure that image_processor kwargs are preserved in the processor.
        We then check that the mean of the pixel_values is less than or equal to 0 after processing.
        Since the original pixel_values are in [0, 255], this is a good indicator that the rescale_factor is indeed applied.
        """
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=-1.0
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pd")
        self.assertEqual(inputs[self.images_input_name].dtype.name, "UINT8")

    def test_unstructured_kwargs_batched(self):
        pass

    def test_tokenizer_defaults_preserved_by_kwargs(self):
        pass

    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        pass

    def test_overlapping_text_image_kwargs_handling(self):
        pass

    def test_doubly_passed_kwargs(self):
        pass

    def test_apply_chat_template_assistant_mask(self):
        pass
