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

import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddleformers.transformers import ShieldGemma2Processor
from paddleformers.transformers.auto.processing import processor_class_from_name
from paddleformers.transformers.gemma3.processor import Gemma3Processor


class _DummyTokenizer:
    image_token_id = 1
    image_token = "<image>"
    boi_token = "<boi>"
    eoi_token = "<eoi>"
    init_kwargs = {}


class _DummyImageProcessor:
    pass


class ShieldGemma2ProcessorTest(unittest.TestCase):
    def setUp(self):
        with patch.object(ShieldGemma2Processor, "check_argument_for_proper_class"):
            self.processor = ShieldGemma2Processor(
                _DummyImageProcessor(), _DummyTokenizer(), chat_template="dummy", image_seq_length=2
            )

    def test_processor_class_registration(self):
        self.assertIs(processor_class_from_name("ShieldGemma2Processor"), ShieldGemma2Processor)

    def test_policy_expansion_and_defaults(self):
        rendered_messages = []

        def render(messages, tokenize=False):
            rendered_messages.extend(messages)
            return ["rendered"] * len(messages)

        with patch.object(self.processor, "apply_chat_template", side_effect=render), patch.object(
            Gemma3Processor, "__call__", return_value="encoded"
        ) as parent_call:
            output = self.processor(
                images=["image-a", "image-b"],
                policies=["dangerous", "custom"],
                custom_policies={"custom": "Custom policy"},
                images_kwargs={"do_pan_and_scan": True},
            )

        self.assertEqual(output, "encoded")
        self.assertEqual(len(rendered_messages), 4)
        self.assertEqual(rendered_messages[0][0]["content"][0], {"type": "image"})
        self.assertEqual(rendered_messages[1][0]["content"][-1]["text"], "Custom policy")
        call_kwargs = parent_call.call_args.kwargs
        self.assertEqual(call_kwargs["images"], [["image-a"], ["image-a"], ["image-b"], ["image-b"]])
        self.assertFalse(call_kwargs["images_kwargs"]["do_pan_and_scan"])
        self.assertTrue(call_kwargs["text_kwargs"]["padding"])

    def test_processor_rejects_missing_images_or_chat_template(self):
        with self.assertRaisesRegex(ValueError, "needs images"):
            self.processor(images=None)

        with self.assertRaisesRegex(ValueError, "needs images"):
            self.processor(images=[])

        self.processor.chat_template = None
        with self.assertRaisesRegex(ValueError, "requires the use"):
            self.processor(images=["image"])

    def test_processor_rejects_unknown_policy_and_multiple_images(self):
        with self.assertRaisesRegex(ValueError, "Unknown ShieldGemma2 policy"):
            self.processor(images=["image"], policies=["missing"])

        with self.assertRaisesRegex(ValueError, "at most one image"):
            self.processor(images=[["image-a", "image-b"]])

    def test_processor_accepts_single_array_or_tensor_image(self):
        images = [
            np.zeros((4, 4, 3), dtype=np.uint8),
            paddle.zeros([3, 4, 4]),
        ]

        with patch.object(self.processor, "apply_chat_template", return_value=["rendered"]), patch.object(
            Gemma3Processor, "__call__", return_value="encoded"
        ) as parent_call:
            for image in images:
                with self.subTest(image_type=type(image).__name__):
                    output = self.processor(images=image)

                    self.assertEqual(output, "encoded")
                    expanded_images = parent_call.call_args.kwargs["images"]
                    self.assertEqual(len(expanded_images), len(self.processor.policy_definitions))
                    self.assertTrue(all(batch[0] is image for batch in expanded_images))


if __name__ == "__main__":
    unittest.main()
