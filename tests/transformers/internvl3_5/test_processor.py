# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from paddleformers.transformers import (
    AutoImageProcessor,
    AutoProcessor,
    InternVLImageProcessor,
    InternVLProcessor,
)

LOCAL_CHECKPOINT = "/sda/housaijie/code/InternVL3_5-1B"


class InternVLProcessorTest(unittest.TestCase):
    def test_dynamic_image_preprocess(self):
        image_processor = InternVLImageProcessor(size={"height": 28, "width": 28}, max_patches=2)
        image = Image.new("RGB", (40, 20), (127, 64, 32))

        outputs = image_processor(images=image, return_tensors="pd")

        self.assertEqual(outputs["num_patches_list"], [3])
        self.assertEqual(list(outputs["pixel_values"].shape), [3, 3, 28, 28])
        self.assertEqual(list(outputs["image_flags"].shape), [3, 1])

    def test_multi_image_call_kwargs_apply_to_every_image(self):
        image_processor = InternVLImageProcessor(size={"height": 28, "width": 28}, max_patches=12, use_thumbnail=True)
        images = [
            Image.new("RGB", (80, 20), (127, 64, 32)),
            Image.new("RGB", (20, 80), (32, 64, 127)),
        ]

        outputs = image_processor(images=images, return_tensors="pd", max_patches=1, use_thumbnail=False)

        self.assertEqual(outputs["num_patches_list"], [1, 1])
        self.assertEqual(list(outputs["pixel_values"].shape), [2, 3, 28, 28])

    def test_auto_image_processor_dispatches_internvl_compat_config(self):
        image = Image.new("RGB", (80, 20), (127, 64, 32))
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "preprocessor_config.json"), "w", encoding="utf-8") as config_file:
                json.dump(
                    {
                        "image_processor_type": "GotOcr2ImageProcessorFast",
                        "processor_class": "InternVLProcessor",
                        "size": {"height": 28, "width": 28},
                        "max_patches": 2,
                        "use_thumbnail": True,
                    },
                    config_file,
                )

            image_processor = AutoImageProcessor.from_pretrained(
                tmpdir, download_hub="huggingface", local_files_only=True
            )
            outputs = image_processor(images=image, return_tensors="pd")

        self.assertIsInstance(image_processor, InternVLImageProcessor)
        self.assertEqual(outputs["num_patches_list"], [3])
        self.assertEqual(list(outputs["pixel_values"].shape), [3, 3, 28, 28])

    def test_auto_processor_resolves_internvl_processor_from_preprocessor_config(self):
        processor = object()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "preprocessor_config.json"), "w", encoding="utf-8") as config_file:
                json.dump({"processor_class": "InternVLProcessor"}, config_file)

            with patch.object(InternVLProcessor, "from_pretrained", return_value=processor) as from_pretrained:
                actual = AutoProcessor.from_pretrained(tmpdir, download_hub="huggingface", local_files_only=True)

        self.assertIs(actual, processor)
        from_pretrained.assert_called_once_with(
            tmpdir, trust_remote_code=None, download_hub="huggingface", _from_auto=True, local_files_only=True
        )

    @unittest.skipUnless(os.path.isdir(LOCAL_CHECKPOINT), "requires local InternVL3_5-1B checkpoint")
    def test_auto_processor_real_checkpoint_chat_template(self):
        processor = AutoProcessor.from_pretrained(LOCAL_CHECKPOINT, local_files_only=True)
        messages = [{"role": "user", "content": "<image>\nDescribe the image shortly."}]

        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        self.assertIsInstance(processor, InternVLProcessor)
        self.assertIsNotNone(processor.chat_template)
        self.assertIn("Describe the image shortly.", rendered)

    def test_expand_image_tokens(self):
        processor = InternVLProcessor.__new__(InternVLProcessor)
        processor.image_seq_length = 4
        processor.image_token = "<image>"
        processor.img_start_token = "<img>"
        processor.img_end_token = "</img>"
        processor.img_context_token = "<IMG_CONTEXT>"

        expanded = processor._expand_image_tokens(["<image>\nDescribe."], [3])

        self.assertEqual(expanded[0].count("<img>"), 1)
        self.assertEqual(expanded[0].count("</img>"), 1)
        self.assertEqual(expanded[0].count("<IMG_CONTEXT>"), 12)


if __name__ == "__main__":
    unittest.main()
