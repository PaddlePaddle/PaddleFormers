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
import shutil
import tempfile
import unittest

import numpy as np

from paddleformers.transformers import AutoProcessor, Glm46VProcessor
from tests.transformers.test_processing_common import ProcessorTesterMixin


class Glm46VProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Glm46VProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        processor = Glm46VProcessor.from_pretrained("PaddleFormers/tiny-random-glmocr")
        processor.save_pretrained(cls.tmpdir)
        cls.image_token = processor.image_token

        cls.maxDiff = None

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs).image_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdir, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image"])
        image_input = self.prepare_image_inputs()

        inputs_dict = {"text": text, "images": image_input}

        call_signature = inspect.signature(processor.__call__)
        input_args = [param.name for param in call_signature.parameters.values()]
        inputs_dict = {k: v for k, v in inputs_dict.items() if k in input_args}

        inputs = processor(**inputs_dict, return_tensors="pd")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()

        processor = Glm46VProcessor(tokenizer=tokenizer, image_processor=image_processor)
        processor.save_pretrained(self.tmpdir)
        processor = Glm46VProcessor.from_pretrained(self.tmpdir)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor.image_processor.__class__.__name__, "Glm46VImageProcessor")

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Glm46VProcessor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="pd")
        input_processor = processor(images=image_input, text="dummy", return_tensors="pd")

        for key in input_image_proc:
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Glm46VProcessor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input, return_tensors="pd")

        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"])

        # test if it raises when no input is passed
        with self.assertRaises(ValueError):
            processor()

        # test if it raises when no text is passed
        with self.assertRaises(TypeError):
            processor(images=image_input, return_tensors="pd")

    def test_image_token_expansion(self):
        """Verify that image token is correctly expanded to the corresponding number of patch tokens"""
        processor = self.get_processor()
        image_input = self.prepare_image_inputs()

        input_str = f"describe this image: {processor.image_token}"
        inputs = processor(text=input_str, images=image_input, return_tensors="pd")

        # The prod of image_grid_thw divided by merge_size^2 should equal the number of expanded image tokens
        merge_length = processor.image_processor.merge_size**2
        expected_num_image_tokens = inputs["image_grid_thw"][0].prod().item() // merge_length
        actual_num_image_tokens = (inputs["input_ids"] == processor.image_token_id).sum().item()

        self.assertEqual(actual_num_image_tokens, expected_num_image_tokens)

    def test_multiple_images(self):
        """Verify that each image token is independently expanded for multi-image input"""
        processor = self.get_processor()
        image_input = self.prepare_image_inputs()
        if not isinstance(image_input, list):
            image_input = [image_input]
        two_images = [image_input[0], image_input[0]]

        input_str = f"{processor.image_token} and {processor.image_token}"
        inputs = processor(text=input_str, images=two_images, return_tensors="pd")

        merge_length = processor.image_processor.merge_size**2
        expected_total = sum(thw.prod().item() // merge_length for thw in inputs["image_grid_thw"])
        actual_total = (inputs["input_ids"] == processor.image_token_id).sum().item()

        self.assertEqual(actual_total, expected_total)

    def test_mm_token_type_ids(self):
        """Verify that mm_token_type_ids is correctly returned when return_mm_token_type_ids=True"""
        processor = self.get_processor()
        image_input = self.prepare_image_inputs()

        input_str = f"describe: {processor.image_token}"
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="np",
            return_mm_token_type_ids=True,
        )

        self.assertIn("mm_token_type_ids", inputs)
        mm_ids = np.array(inputs["mm_token_type_ids"])
        input_ids = np.array(inputs["input_ids"])

        # Image token positions should be 1, others should be 0
        image_token_mask = input_ids == processor.image_token_id
        self.assertTrue(np.all(mm_ids[image_token_mask] == 1))
        self.assertTrue(np.all(mm_ids[~image_token_mask] == 0))

    def test_no_mm_token_type_ids_by_default(self):
        """Verify that mm_token_type_ids is not returned by default"""
        processor = self.get_processor()
        image_input = self.prepare_image_inputs()

        input_str = f"describe: {processor.image_token}"
        inputs = processor(text=input_str, images=image_input, return_tensors="pd")

        self.assertNotIn("mm_token_type_ids", inputs)

    def test_text_only_input(self):
        """Verify that text-only input (without images) works correctly"""
        processor = self.get_processor()

        input_str = "hello world"
        inputs = processor(text=input_str, return_tensors="pd")

        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertNotIn("pixel_values", inputs)
        self.assertNotIn("image_grid_thw", inputs)

    def test_post_process_image_text_to_text(self):
        """Verify that post_process_image_text_to_text correctly decodes token ids"""
        processor = self.get_processor()
        tokenizer = self.get_tokenizer()

        test_text = ["hello world", "paddle ocr"]
        encoded = tokenizer(test_text, return_tensors="pd", padding=True)
        decoded = processor.post_process_image_text_to_text(encoded["input_ids"], skip_special_tokens=True)

        self.assertEqual(len(decoded), 2)
        for original, result in zip(test_text, decoded):
            self.assertIn(original, result)

    def test_apply_chat_template_assistant_mask(self):
        pass

    def test_chat_template_jinja_kwargs(self):
        pass

    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data,
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
                    "content": [{"type": "text", "text": "OCR:"}],
                },
            ]
        ] * batch_size

        # Verify that jinja template can be applied
        formatted_prompt = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), batch_size)

        # Verify that template tokenization result matches direct tokenization
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

        # Verify max_length padding/truncation
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

        # Verify that return_dict=True includes text-related fields
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

    @unittest.skip("GlmOcr does not support video input")
    def test_apply_chat_template_video_frame_sampling(self):
        pass
