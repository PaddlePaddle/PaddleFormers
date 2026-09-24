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


import tempfile
import unittest

from paddleformers.transformers.deepseek_v4.configuration import DeepseekV4Config


class TestIndexCacheConfigMigration(unittest.TestCase):
    def test_new_config_has_no_legacy_alias(self):
        config = DeepseekV4Config(indexcache_topk_pattern="FS", indexcache_multi_layer_distill=True)
        self.assertEqual(config.indexcache_topk_pattern, "FS")
        self.assertTrue(config.indexcache_multi_layer_distill)
        self.assertNotIn("index_topk_pattern", config.to_dict())

    def test_legacy_checkpoint_is_converted_without_mutating_input(self):
        original = {"index_topk_pattern": " fs "}
        config = DeepseekV4Config.from_dict(original)
        self.assertEqual(config.indexcache_topk_pattern, "FS")
        self.assertEqual(original, {"index_topk_pattern": " fs "})
        self.assertNotIn("index_topk_pattern", config.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            restored = DeepseekV4Config.from_pretrained(directory)
        self.assertEqual(restored.indexcache_topk_pattern, "FS")
        self.assertNotIn("index_topk_pattern", restored.to_dict())

    def test_checkpoint_conflict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            DeepseekV4Config.from_dict({"index_topk_pattern": "FS", "indexcache_topk_pattern": "FF"})

    def test_legacy_constructor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Use indexcache_topk_pattern"):
            DeepseekV4Config(index_topk_pattern="FS")

    def test_absent_and_null_pattern_stay_disabled(self):
        for data in ({}, {"index_topk_pattern": None}):
            config = DeepseekV4Config.from_dict(data)
            self.assertIsNone(config.indexcache_topk_pattern)
            self.assertNotIn("index_topk_pattern", config.to_dict())

    def test_explicit_override_and_unused_kwargs_contract(self):
        config, unused = DeepseekV4Config.from_dict(
            {"index_topk_pattern": "FS"},
            indexcache_topk_pattern="FF",
            return_unused_kwargs=True,
            unrelated_option=3,
        )
        self.assertEqual(config.indexcache_topk_pattern, "FF")
        self.assertEqual(unused, {"unrelated_option": 3})
