# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

from paddleformers.datasets.template.mm_plugin import get_mm_plugin
from paddleformers.transformers import Florence2Processor


class Florence2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = object.__new__(Florence2Processor)
        self.processor.task_prompts_without_inputs = {"<CAPTION>": "What does the image describe?"}
        self.processor.task_prompts_with_input = {"<REGION_TO_DESCRIPTION>": "What does the region {input} describe?"}

    def test_construct_prompts(self):
        prompts = self.processor._construct_prompts(["<CAPTION>", "<REGION_TO_DESCRIPTION><loc_1><loc_2>"])
        self.assertEqual(prompts[0], "What does the image describe?")
        self.assertIn("<loc_1><loc_2>", prompts[1])

    def test_post_process_detection(self):
        result = self.processor.post_process_generation("cat<loc_0><loc_1><loc_998><loc_999>", "<OD>", (100, 200))
        self.assertEqual(result["<OD>"]["labels"], ["cat"])
        self.assertEqual(len(result["<OD>"]["bboxes"][0]), 4)

    def test_post_process_segmentation(self):
        result = self.processor.post_process_generation(
            "cat<poly><loc_0><loc_1><loc_500><loc_501><loc_998><loc_999></poly>",
            "<REFERRING_EXPRESSION_SEGMENTATION>",
            (100, 200),
        )
        self.assertEqual(result["<REFERRING_EXPRESSION_SEGMENTATION>"]["labels"], ["cat"])
        self.assertEqual(len(result["<REFERRING_EXPRESSION_SEGMENTATION>"]["polygons"][0][0]), 6)

    def test_post_process_region_proposal(self):
        result = self.processor.post_process_generation(
            "<loc_0><loc_1><loc_998><loc_999>",
            "<REGION_PROPOSAL>",
            (100, 200),
        )
        self.assertEqual(len(result["<REGION_PROPOSAL>"]["bboxes"]), 1)

    def test_sft_message_format(self):
        self.processor.image_processor = object()
        plugin = get_mm_plugin("florence2", image_token="<image>")
        messages = [
            {"role": "user", "content": "<image><CAPTION>"},
            {"role": "assistant", "content": "A solid color image."},
        ]
        processed = plugin.process_messages(messages, ["image.jpg"], [], [], {}, self.processor)
        self.assertEqual(processed[0]["content"], "What does the image describe?")
        self.assertEqual(processed[1]["content"], "A solid color image.")
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            plugin.process_messages([], [], [], [], {}, self.processor)


if __name__ == "__main__":
    unittest.main()
