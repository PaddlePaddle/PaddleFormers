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
from types import SimpleNamespace

from paddleformers.datasets.template.mm_plugin import FastVLMPlugin
from paddleformers.datasets.template.template import TEMPLATES, FastVLMTemplate


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [len(text)] if text else []


class FastVLMTemplateTest(unittest.TestCase):
    def test_registered_template_injects_negative_image_token(self):
        template = TEMPLATES["fastvlm"]
        self.assertIsInstance(template, FastVLMTemplate)
        pairs = template.encode_multiturn(
            DummyTokenizer(),
            [
                {"role": "user", "content": "<image>Describe it."},
                {"role": "assistant", "content": "An image."},
            ],
        )
        prompt_ids, _ = pairs[0]
        self.assertEqual(prompt_ids.count(-200), 1)

    def test_plugin_validates_single_image_and_masks_placeholder(self):
        plugin = FastVLMPlugin(image_token="<image>", video_token=None, audio_token=None)
        processor = SimpleNamespace(image_processor=object(), tokenizer=object())
        messages = [{"role": "user", "content": "<image>Describe it."}]
        self.assertIs(plugin.process_messages(messages, [object()], [], [], {}, processor), messages)
        self.assertEqual(plugin.process_tokens([1, -200, 2], processor), [1, -100, 2])

        with self.assertRaisesRegex(ValueError, "exactly one image"):
            plugin.process_messages(
                [{"role": "user", "content": "<image><image>"}],
                [object(), object()],
                [],
                [],
                {},
                processor,
            )


if __name__ == "__main__":
    unittest.main()
