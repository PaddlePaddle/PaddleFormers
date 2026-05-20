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

import unittest

from paddleformers.datasets.template.grounding_plugin import (
    BaseGroundingPlugin,
    get_grounding_plugin,
    register_grounding_plugin,
)


class TestBaseGroundingPlugin(unittest.TestCase):
    """Tests for BaseGroundingPlugin."""

    def setUp(self):
        self.plugin = BaseGroundingPlugin()

    def test_normalize_bbox(self):
        bbox = [1.5, 2.7, 3.9, 4.1]
        result = self.plugin.normalize_bbox(bbox)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_format_ref_object(self):
        result = self.plugin.format_ref_object("cat")
        self.assertEqual(result, "<|object_ref_start|>cat<|object_ref_end|>")

    def test_format_bbox(self):
        result = self.plugin.format_bbox([10, 20, 30, 40])
        self.assertEqual(result, "<|box_start|>(10,20),(30,40)<|box_end|>")

    def test_process_messages(self):
        messages = [
            {"content": "Find the <ref-object> in <bbox>"},
            {"content": "Found <ref-object> at <bbox>"},
        ]
        objects = {
            "ref": ["cat", "dog"],
            "bbox": [[10, 20, 30, 40], [50, 60, 70, 80]],
        }
        result = self.plugin.process_messages(messages, objects)
        self.assertIn("<|object_ref_start|>cat<|object_ref_end|>", result[0]["content"])
        self.assertIn("<|box_start|>(10,20),(30,40)<|box_end|>", result[0]["content"])
        self.assertIn("<|object_ref_start|>dog<|object_ref_end|>", result[1]["content"])
        self.assertIn("<|box_start|>(50,60),(70,80)<|box_end|>", result[1]["content"])

    def test_process_messages_no_placeholders(self):
        messages = [{"content": "No objects here"}]
        objects = {"ref": [], "bbox": []}
        result = self.plugin.process_messages(messages, objects)
        self.assertEqual(result[0]["content"], "No objects here")


class TestGetGroundingPlugin(unittest.TestCase):
    """Tests for get_grounding_plugin and register_grounding_plugin."""

    def test_get_base_plugin(self):
        plugin = get_grounding_plugin(name="base")
        self.assertIsInstance(plugin, BaseGroundingPlugin)

    def test_get_nonexistent_plugin(self):
        with self.assertRaises(ValueError):
            get_grounding_plugin(name="nonexistent_grounding_plugin")

    def test_register_and_get_plugin(self):
        class CustomGroundingPlugin(BaseGroundingPlugin):
            pass

        register_grounding_plugin("custom_grounding", CustomGroundingPlugin)
        plugin = get_grounding_plugin(name="custom_grounding")
        self.assertIsInstance(plugin, CustomGroundingPlugin)

    def test_register_duplicate_plugin(self):
        with self.assertRaises(ValueError):
            register_grounding_plugin("base", BaseGroundingPlugin)


if __name__ == "__main__":
    unittest.main()
