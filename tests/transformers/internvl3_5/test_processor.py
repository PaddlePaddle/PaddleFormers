# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

from PIL import Image

from paddleformers.transformers import InternVLImageProcessor, InternVLProcessor


class InternVLProcessorTest(unittest.TestCase):
    def test_dynamic_image_preprocess(self):
        image_processor = InternVLImageProcessor(size={"height": 28, "width": 28}, max_patches=2)
        image = Image.new("RGB", (40, 20), (127, 64, 32))

        outputs = image_processor(images=image, return_tensors="pd")

        self.assertEqual(outputs["num_patches_list"], [3])
        self.assertEqual(list(outputs["pixel_values"].shape), [3, 3, 28, 28])
        self.assertEqual(list(outputs["image_flags"].shape), [3, 1])

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
