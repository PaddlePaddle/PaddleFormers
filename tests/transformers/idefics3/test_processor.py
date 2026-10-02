# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import shutil
import tempfile
import unittest
from types import SimpleNamespace

import paddle

from paddleformers.datasets.template.mm_plugin import Idefics3Plugin
from paddleformers.transformers import AutoProcessor, Idefics3Processor
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_processing_common import ProcessorTesterMixin


@unittest.skip("Idefics3 tiny checkpoint is not available yet.")
class Idefics3ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Idefics3Processor
    model_path = "PaddleFormers/tiny-random-idefics3"

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        processor = Idefics3Processor.from_pretrained(
            cls.model_path,
            size={"longest_edge": 8},
            max_image_size={"longest_edge": 8},
            do_image_splitting=False,
            image_seq_len=4,
        )
        processor.save_pretrained(cls.tmpdir)
        cls.image_token = processor.image_token

    @gpu_device_initializer(log_prefix="Idefics3ProcessorTest", gpu_id=0)
    def setUp(self):
        pass

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

        inputs = processor(text=text, images=image_input, return_tensors="pd")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_get_num_vision_tokens(self):
        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(8, 8), (16, 8)])

        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 2)
        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 2)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        image_processor = self.get_image_processor()

        processor = Idefics3Processor(tokenizer=tokenizer, image_processor=image_processor, image_seq_len=4)
        processor.save_pretrained(self.tmpdir)
        processor = Idefics3Processor.from_pretrained(self.tmpdir)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor.image_processor.__class__.__name__, "Idefics3ImageProcessor")

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        processor = Idefics3Processor(tokenizer=tokenizer, image_processor=image_processor, image_seq_len=4)

        image_input = self.prepare_image_inputs()
        input_image_proc = image_processor(image_input, return_tensors="pd")
        input_processor = processor(images=image_input, text="dummy <image>", return_tensors="pd")

        self.assertTrue(paddle.allclose(input_image_proc["pixel_values"], input_processor["pixel_values"]))

    def test_processor(self):
        processor = self.get_processor()
        image_input = self.prepare_image_inputs()

        inputs = processor(text="lower newer <image>", images=image_input, return_tensors="pd")

        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask", "pixel_values"])

        with self.assertRaises(ValueError):
            processor()

        with self.assertRaises(ValueError):
            processor(images=image_input, return_tensors="pd")

    def test_replace_image_token_without_split(self):
        processor = self.get_processor()
        image_inputs = {"rows": [[0]], "cols": [[0]]}

        replacement = processor.replace_image_token(image_inputs, image_idx=0, image_seq_len=3)

        self.assertEqual(
            replacement,
            "<fake_token_around_image><global-img><image><image><image><fake_token_around_image>",
        )

    def test_replace_image_token_with_split(self):
        processor = self.get_processor()
        image_inputs = {"rows": [[1]], "cols": [[2]]}

        replacement = processor.replace_image_token(image_inputs, image_idx=0, image_seq_len=2)

        self.assertIn("<row_1_col_1><image><image>", replacement)
        self.assertIn("<row_1_col_2><image><image>", replacement)
        self.assertTrue(replacement.endswith("<global-img><image><image><fake_token_around_image>"))

    def test_get_text_with_replacements(self):
        processor = self.get_processor()
        text = ["a <image> b", "c <image> d"]
        replacements = ["first", "second"]

        replaced = processor.get_text_with_replacements(text, replacements)

        self.assertEqual(replaced, ["a first b", "c second d"])

    def test_get_text_with_replacements_checks_replacement_count(self):
        processor = self.get_processor()

        with self.assertRaises(ValueError):
            processor.get_text_with_replacements(["a <image> b"], [])

        with self.assertRaises(ValueError):
            processor.get_text_with_replacements(["a b"], ["unused"])

    def test_validate_inputs_checks_image_count(self):
        processor = self.get_processor()
        with self.assertRaises(ValueError):
            processor.validate_inputs(images=[["img"]], text=["<image> <image>"])

    def test_create_mm_token_type_ids(self):
        processor = self.get_processor()
        image_token_id = processor.image_token_id
        token_type_ids = processor.create_mm_token_type_ids([[1, image_token_id, 2, image_token_id]])

        self.assertEqual(token_type_ids, [[0, 1, 0, 1]])

    @unittest.skip("Idefics3ImageProcessor does not support rescale_factor.")
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        pass

    @unittest.skip("Idefics3ImageProcessor does not support rescale_factor.")
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        pass

    @unittest.skip("Idefics3ImageProcessor does not support rescale_factor.")
    def test_unstructured_kwargs(self):
        pass

    @unittest.skip("Idefics3ImageProcessor does not support rescale_factor.")
    def test_unstructured_kwargs_batched(self):
        pass

    @unittest.skip("Idefics3ImageProcessor does not support rescale_factor.")
    def test_doubly_passed_kwargs(self):
        pass

    @unittest.skip("Idefics3ImageProcessor does not support rescale_factor.")
    def test_structured_kwargs_nested(self):
        pass

    @unittest.skip("Idefics3ImageProcessor does not support rescale_factor.")
    def test_structured_kwargs_nested_from_dict(self):
        pass


class Idefics3PluginTest(unittest.TestCase):
    def setUp(self):
        self.plugin = Idefics3Plugin(
            image_token="<image>",
            video_token=None,
            audio_token=None,
        )
        self.processor = SimpleNamespace(
            image_processor=object(),
            image_seq_len=4,
        )

    def process_messages(self, messages, images, mm_inputs):
        return self.plugin.process_messages(
            messages=messages,
            images=images,
            videos=[],
            audios=[],
            mm_inputs=mm_inputs,
            processor=self.processor,
        )

    def test_process_messages_expands_image_tokens_to_match_features(self):
        result = self.process_messages(
            messages=[{"content": "before <image> after"}],
            images=[object()],
            mm_inputs={"rows": [[2]], "cols": [[3]]},
        )

        content = result[0]["content"]
        self.assertEqual(content.count("<image>"), (2 * 3 + 1) * 4)
        self.assertIn("<row_1_col_1>", content)
        self.assertIn("<row_2_col_3>", content)
        self.assertIn("<global-img>", content)

    def test_process_messages_expands_unsplit_image_to_global_features(self):
        result = self.process_messages(
            messages=[{"content": "<image>"}],
            images=[object()],
            mm_inputs={"rows": [[0]], "cols": [[0]]},
        )

        content = result[0]["content"]
        self.assertEqual(content.count("<image>"), 4)
        self.assertNotIn("<row_1_col_1>", content)
        self.assertIn("<global-img>", content)

    def test_process_messages_rejects_missing_image_grid_metadata(self):
        with self.assertRaisesRegex(ValueError, "rows metadata"):
            self.process_messages(
                messages=[{"content": "<image>"}],
                images=[object()],
                mm_inputs={"rows": [], "cols": []},
            )


if __name__ == "__main__":
    unittest.main()
