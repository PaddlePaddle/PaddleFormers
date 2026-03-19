# coding=utf-8
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import paddle
from PIL import Image

from paddleformers.transformers.processing_utils import (
    MODALITY_TO_AUTOPROCESSOR_MAPPING,
)

from .test_utils import check_json_file_has_correct_format

MODALITY_INPUT_DATA = {
    "images": [
        "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
        "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
    ],
    "videos": [
        "http://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4",
    ],
}


def url_to_local_path(url, return_url_if_not_found=True):
    filename = url.split("/")[-1]

    if not os.path.exists(filename) and return_url_if_not_found:
        return url

    return filename


def prepare_image_inputs():
    """This function prepares a list of PIL images"""
    image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]
    image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
    return image_inputs


class ProcessorTesterMixin:
    processor_class = None
    text_input_name = "input_ids"
    images_input_name = "pixel_values"
    videos_input_name = "pixel_values_videos"

    @staticmethod
    def prepare_processor_dict():
        return {}

    def get_component(self, attribute, **kwargs):
        if attribute not in MODALITY_TO_AUTOPROCESSOR_MAPPING and "tokenizer" in attribute:
            auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING["tokenizer"]
            component = auto_processor_class.from_pretrained(self.tmpdir, subfolder=attribute, **kwargs)  # noqa
        else:
            auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[attribute]
            component = auto_processor_class.from_pretrained(self.tmpdir, **kwargs)  # noqa
        if "tokenizer" in attribute and not component.pad_token:
            component.pad_token = "[TEST_PAD]"
            if component.pad_token_id is None:
                component.pad_token_id = 0

        return component

    def prepare_components(self):
        components = {}
        for attribute in self.processor_class.get_attributes():
            component = self.get_component(attribute)
            components[attribute] = component

        return components

    def get_processor(self):
        components = self.prepare_components()
        processor = self.processor_class(**components, **self.prepare_processor_dict())
        return processor

    def prepare_text_inputs(self, batch_size: int | None = None, modalities: str | list | None = None):
        if isinstance(modalities, str):
            modalities = [modalities]

        special_token_to_add = ""
        if modalities is not None:
            for modality in modalities:
                special_token_to_add += getattr(self, f"{modality}_token", "")

        if batch_size is None:
            return f"lower newer {special_token_to_add}"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return [f"lower newer {special_token_to_add}"]
        return [f"lower newer {special_token_to_add}", f" {special_token_to_add} upper older longer string"] + [
            f"lower newer {special_token_to_add}"
        ] * (batch_size - 2)

    def prepare_image_inputs(self, batch_size: int | None = None):
        """This function prepares a list of PIL images for testing"""
        if batch_size is None:
            return prepare_image_inputs()[0]
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        return prepare_image_inputs() * batch_size

    def prepare_video_inputs(self, batch_size: int | None = None):
        """This function prepares a list of numpy videos."""
        video_input = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)] * 8
        video_input = np.array(video_input)
        if batch_size is None:
            return video_input
        return [video_input] * batch_size

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            # Chat template is saved as a separate file
            if key not in "chat_template":
                # json converts dict keys to str, but some processors force convert back to int when init
                if (
                    isinstance(obj[key], dict)
                    and isinstance(list(obj[key].keys())[0], str)
                    and isinstance(list(value.keys())[0], int)
                ):
                    obj[key] = {int(k): v for k, v in obj[key].items()}
                self.assertEqual(obj[key], value)
                self.assertEqual(getattr(processor, key, None), value)

    def test_processor_from_and_save_pretrained(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_files = processor_first.save_pretrained(tmpdir)
            if len(saved_files) > 0:
                check_json_file_has_correct_format(saved_files[0])
                processor_second = self.processor_class.from_pretrained(tmpdir)

                self.assertEqual(processor_second.to_dict(), processor_first.to_dict())

                for attribute in processor_first.attributes:
                    attribute_first = getattr(processor_first, attribute)
                    attribute_second = getattr(processor_second, attribute)

                    # tokenizer repr contains model-path from where we loaded
                    if "tokenizer" not in attribute:
                        self.assertEqual(repr(attribute_first), repr(attribute_second))

    def test_processor_from_and_save_pretrained_as_nested_dict(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_files = processor_first.save_pretrained(tmpdir, legacy_serialization=False)
            check_json_file_has_correct_format(saved_files[0])

            # Load it back and check if loaded correctly
            processor_second = self.processor_class.from_pretrained(tmpdir)

            self.assertEqual(processor_second.to_dict(), processor_first.to_dict())

            # Try to load each attribute separately from saved directory
            for attribute in processor_first.get_attributes():
                if attribute not in MODALITY_TO_AUTOPROCESSOR_MAPPING and "tokenizer" in attribute:
                    auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING["tokenizer"]
                    attribute_reloaded = auto_processor_class.from_pretrained(tmpdir, subfolder=attribute)
                else:
                    auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[attribute]
                    attribute_reloaded = auto_processor_class.from_pretrained(tmpdir)
                attribute_first = getattr(processor_first, attribute)

                # tokenizer repr contains model-path from where we loaded
                if "tokenizer" not in attribute:
                    self.assertEqual(repr(attribute_first), repr(attribute_reloaded))

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image", "video"])
        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()
        inputs_dict = {"text": text, "images": image_input, "videos": video_inputs}

        call_signature = inspect.signature(processor.__call__)
        input_args = [param.name for param in call_signature.parameters.values()]
        inputs_dict = {k: v for k, v in inputs_dict.items() if k in input_args}

        inputs = processor(**inputs_dict, return_tensors="pd")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input, return_tensors="pd")
        self.assertEqual(inputs[self.text_input_name].shape[-1], 117)

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
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding="longest")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str, images=image_input, return_tensors="pd", max_length=112, padding="max_length"
        )
        self.assertEqual(inputs[self.text_input_name].shape[-1], 112)

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
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

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
            do_rescale=True,
            rescale_factor=-1.0,
            padding="max_length",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(batch_size=2, modalities="image")
        image_input = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pd",
            do_rescale=True,
            rescale_factor=-1.0,
            padding="longest",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertTrue(
            len(inputs[self.text_input_name][0]) == len(inputs[self.text_input_name][1])
            and len(inputs[self.text_input_name][1]) < 76
        )

    def test_doubly_passed_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = [self.prepare_text_inputs(modalities="image")]
        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                images_kwargs={"do_rescale": True, "rescale_factor": -1.0},
                do_rescale=True,
                return_tensors="pd",
            )

    def test_args_overlap_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_first = self.get_processor()
        image_processor = processor_first.image_processor
        image_processor.is_override = True

        with tempfile.TemporaryDirectory() as tmpdir:
            processor_first.save_pretrained(tmpdir)
            processor_second = self.processor_class.from_pretrained(tmpdir, image_processor=image_processor)
            self.assertTrue(processor_second.image_processor.is_override)

    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pd"},
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1.0},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pd"},
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1.0},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_tokenizer_defaults_preserved_by_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=167, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()
        inputs = processor(text=input_str, videos=video_input, do_sample_frames=False, return_tensors="pd")
        self.assertEqual(inputs[self.text_input_name].shape[-1], 167)

    def test_video_processor_defaults_preserved_by_video_kwargs(self):
        """
        We use do_rescale=True, rescale_factor=-1.0 to ensure that image_processor kwargs are preserved in the processor.
        We then check that the mean of the pixel_values is less than or equal to 0 after processing.
        Since the original pixel_values are in [0, 255], this is a good indicator that the rescale_factor is indeed applied.
        """
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["video_processor"] = self.get_component(
            "video_processor", do_rescale=True, rescale_factor=-1.0
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=167, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()

        inputs = processor(text=input_str, videos=video_input, do_sample_frames=False, return_tensors="pd")
        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)

    def test_kwargs_overrides_default_tokenizer_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding="longest")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()
        inputs = processor(
            text=input_str,
            videos=video_input,
            do_sample_frames=False,
            return_tensors="pd",
            max_length=162,
            padding="max_length",
        )
        self.assertEqual(inputs[self.text_input_name].shape[-1], 162)

    def test_kwargs_overrides_default_video_processor_kwargs(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["video_processor"] = self.get_component(
            "video_processor", do_rescale=True, rescale_factor=1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=167, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()

        inputs = processor(
            text=input_str,
            videos=video_input,
            do_sample_frames=False,
            do_rescale=True,
            rescale_factor=-1.0,
            return_tensors="pd",
        )
        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)

    def test_unstructured_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()
        inputs = processor(
            text=input_str,
            videos=video_input,
            do_sample_frames=False,
            return_tensors="pd",
            do_rescale=True,
            rescale_factor=-1.0,
            padding="max_length",
            max_length=176,
        )

        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 176)

    def test_unstructured_kwargs_batched_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(batch_size=2, modalities="video")
        video_input = self.prepare_video_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            videos=video_input,
            do_sample_frames=False,
            return_tensors="pd",
            do_rescale=True,
            rescale_factor=-1.0,
            padding="longest",
            max_length=176,
        )

        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertTrue(
            len(inputs[self.text_input_name][0]) == len(inputs[self.text_input_name][1])
            and len(inputs[self.text_input_name][1]) < 176
        )

    def test_doubly_passed_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = [self.prepare_text_inputs(modalities="video")]
        video_input = self.prepare_video_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                videos=video_input,
                do_sample_frames=False,
                videos_kwargs={"do_rescale": True, "rescale_factor": -1.0},
                do_rescale=True,
                return_tensors="pd",
            )

    def test_structured_kwargs_nested_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pd"},
            "videos_kwargs": {"do_rescale": True, "rescale_factor": -1.0, "do_sample_frames": False},
            "text_kwargs": {"padding": "max_length", "max_length": 176},
        }

        inputs = processor(text=input_str, videos=video_input, **all_kwargs)

        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 176)

    def test_structured_kwargs_nested_from_dict_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pd"},
            "videos_kwargs": {"do_rescale": True, "rescale_factor": -1.0, "do_sample_frames": False},
            "text_kwargs": {"padding": "max_length", "max_length": 176},
        }

        inputs = processor(text=input_str, videos=video_input, **all_kwargs)
        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 176)

    def test_overlapping_text_image_kwargs_handling(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)

        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_image_inputs()

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pd",
                padding="max_length",
                text_kwargs={"padding": "do_not_pad"},
            )

    def test_prepare_and_validate_optional_call_args(self):
        processor = self.get_processor()
        optional_call_args_name = getattr(processor, "optional_call_args", [])
        num_optional_call_args = len(optional_call_args_name)
        if num_optional_call_args == 0:
            self.skipTest("No optional call args")
        # test all optional call args are given
        optional_call_args = processor.prepare_and_validate_optional_call_args(
            *(f"optional_{i}" for i in range(num_optional_call_args))
        )
        self.assertEqual(
            optional_call_args, {arg_name: f"optional_{i}" for i, arg_name in enumerate(optional_call_args_name)}
        )
        # test only one optional call arg is given
        optional_call_args = processor.prepare_and_validate_optional_call_args("optional_1")
        self.assertEqual(optional_call_args, {optional_call_args_name[0]: "optional_1"})
        # test no optional call arg is given
        optional_call_args = processor.prepare_and_validate_optional_call_args()
        self.assertEqual(optional_call_args, {})
        # test too many optional call args are given
        with self.assertRaises(ValueError):
            processor.prepare_and_validate_optional_call_args(
                *(f"optional_{i}" for i in range(num_optional_call_args + 1))
            )

    def test_chat_template_save_loading(self):
        processor = self.processor_class.from_pretrained(self.tmpdir)
        signature = inspect.signature(processor.__init__)
        if "chat_template" not in {*signature.parameters.keys()}:
            self.skipTest("Processor doesn't accept chat templates at input")

        existing_tokenizer_template = getattr(processor.tokenizer, "chat_template", None)
        processor.chat_template = "test template"
        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir, save_jinja_files=False)
            self.assertTrue(Path(tmpdir, "chat_template.json").is_file())
            self.assertFalse(Path(tmpdir, "chat_template.jinja").is_file())
            reloaded_processor = self.processor_class.from_pretrained(tmpdir)
            self.assertEqual(processor.chat_template, reloaded_processor.chat_template)
            # When we don't use single-file chat template saving, processor and tokenizer chat templates
            # should remain separate
            self.assertEqual(getattr(reloaded_processor.tokenizer, "chat_template", None), existing_tokenizer_template)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            self.assertTrue(Path(tmpdir, "chat_template.jinja").is_file())
            self.assertFalse(Path(tmpdir, "chat_template.json").is_file())
            reloaded_processor = self.processor_class.from_pretrained(tmpdir)
            self.assertEqual(processor.chat_template, reloaded_processor.chat_template)
            # When we save as single files, tokenizers and processors share a chat template, which means
            # the reloaded tokenizer should get the chat template as well
            self.assertEqual(reloaded_processor.chat_template, reloaded_processor.tokenizer.chat_template)

        with self.assertRaises(ValueError):
            # Saving multiple templates in the legacy format is not permitted
            with tempfile.TemporaryDirectory() as tmpdir:
                processor.chat_template = {"default": "a", "secondary": "b"}
                processor.save_pretrained(tmpdir, save_jinja_files=False)

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

        # some models have only Fast image processor
        if getattr(processor, processor_name).__class__.__name__.endswith("Fast"):
            return_tensors = "pd"

        batch_messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": "Describe this."}]},
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
            batch_messages[idx][1]["content"] = [batch_messages[idx][1]["content"][0], {"type": modality, "url": url}]

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
        self.assertEqual(len(out_dict[input_name]), batch_size)

        return_tensor_to_type = {"pd": paddle.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])

        # Test continue from final message
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "It is the sound of"}],
        }
        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx] = batch_messages[idx] + [assistant_message]
        continue_prompt = processor.apply_chat_template(batch_messages, continue_final_message=True, tokenize=False)
        for prompt in continue_prompt:
            self.assertTrue(prompt.endswith("It is the sound of"))  # no `eos` token at the end

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
                            "url": "http://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4",
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        num_frames = 3
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=num_frames,
            return_tensors="pd",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), num_frames)

        # Load with `fps` arg
        fps = 10
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            fps=fps,
            return_tensors="pd",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        # 3 frames are inferred from input video's length and FPS, so can be hardcoded
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 3)

        # When `do_sample_frames=False` no sampling is done and whole video is loaded, even if number of frames is passed
        fps = 10
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            do_sample_frames=False,
            fps=fps,
            return_tensors="pd",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 11)

        # Load with `fps` and `num_frames` args, should raise an error
        with self.assertRaises(ValueError):
            out_dict_with_video = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                fps=fps,
                num_frames=num_frames,
            )

        # Load without any arg should load the whole video
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 11)

        # Load video as a list of frames (i.e. images).
        # NOTE: each frame should have same size because we assume they come from one video
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": ["https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"] * 2,
        }
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 2)

        # When the inputs are frame URLs/paths we expect that those are already
        # sampled and will raise an error is asked to sample again.
        with self.assertRaisesRegex(
            ValueError, "Sampling frames from a list of images is not supported! Set `do_sample_frames=False`"
        ):
            out_dict_with_video = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                do_sample_frames=True,
            )

    def test_chat_template_jinja_kwargs(self):
        """Tests that users can pass any kwargs and they will be used in jinja templates."""
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Which of these animals is making the sound?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is a cow."}],
            },
        ]

        dummy_template = (
            "{% for message in messages %}"
            "{% if add_system_prompt %}"
            "{{'You are a helpful assistant.'}}"
            "{% endif %}"
            "{% if (message['role'] != 'assistant') %}"
            "{{'<|special_start|>' + message['role'] + '\n' + message['content'][0]['text'] + '<|special_end|>' + '\n'}}"
            "{% elif (message['role'] == 'assistant')%}"
            "{{'<|special_start|>' + message['role'] + '\n'}}"
            "{{message['content'][0]['text'] + '<|special_end|>' + '\n'}}"
            "{% endif %}"
            "{% endfor %}"
        )

        formatted_prompt = processor.apply_chat_template(
            messages, add_system_prompt=True, tokenize=False, chat_template=dummy_template
        )
        expected_prompt = "You are a helpful assistant.<|special_start|>user\nWhich of these animals is making the sound?<|special_end|>\nYou are a helpful assistant.<|special_start|>assistant\nIt is a cow.<|special_end|>\n"
        self.assertEqual(formatted_prompt, expected_prompt)

    def test_apply_chat_template_assistant_mask(self):
        processor = self.get_processor()

        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the capital of France?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The capital of France is Paris."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What about Italy?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The capital of Italy is Rome."},
                    ],
                },
            ]
        ]

        dummy_template = (
            "{% for message in messages %}"
            "{% if (message['role'] != 'assistant') %}"
            "{{'<|special_start|>' + message['role'] + '\n' + message['content'][0]['text'] + '<|special_end|>' + '\n'}}"
            "{% elif (message['role'] == 'assistant')%}"
            "{{'<|special_start|>' + message['role'] + '\n'}}"
            "{% generation %}"
            "{{message['content'][0]['text'] + '<|special_end|>' + '\n'}}"
            "{% endgeneration %}"
            "{% endif %}"
            "{% endfor %}"
        )

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pd",
            return_assistant_tokens_mask=True,
            chat_template=dummy_template,
        )
        self.assertTrue("assistant_masks" in inputs)
        self.assertEqual(len(inputs["assistant_masks"]), len(inputs["input_ids"]))

        mask = inputs["assistant_masks"].bool()
        assistant_ids = inputs["input_ids"][mask]

        assistant_text = (
            "The capital of France is Paris.<|special_end|>\nThe capital of Italy is Rome.<|special_end|>\n"
        )

        # Some tokenizers add extra spaces which aren't then removed when decoding, so we need to check token ids
        # if we can't get identical text outputs
        text_is_same = assistant_text == processor.decode(assistant_ids, clean_up_tokenization_spaces=True)
        ids_is_same = processor.tokenizer.encode(assistant_text, add_special_tokens=False), assistant_ids.tolist()
        self.assertTrue(text_is_same or ids_is_same)
