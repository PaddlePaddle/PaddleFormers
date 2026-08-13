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

"""Unit tests for DeepseekV4Config MoH (use_moh / num_activated_heads) plumbing."""

import unittest

from paddleformers.transformers.deepseek_v4.configuration import DeepseekV4Config


class TestDeepseekV4ConfigMoHDefaults(unittest.TestCase):
    """MoH fields default to off / None."""

    def test_use_moh_defaults_false(self):
        config = DeepseekV4Config()
        self.assertFalse(config.use_moh)

    def test_num_activated_heads_defaults_none(self):
        config = DeepseekV4Config()
        self.assertIsNone(config.num_activated_heads)


class TestDeepseekV4ConfigMoHAccepted(unittest.TestCase):
    """When both fields are given they are stored and accessible."""

    def test_fields_stored(self):
        config = DeepseekV4Config(use_moh=True, num_activated_heads=8)
        self.assertTrue(config.use_moh)
        self.assertEqual(config.num_activated_heads, 8)

    def test_num_activated_heads_equal_dsa_index_n_heads(self):
        config = DeepseekV4Config(use_moh=True, num_activated_heads=64, dsa_index_n_heads=64)
        self.assertEqual(config.num_activated_heads, 64)
        self.assertEqual(config.dsa_index_n_heads, 64)


class TestDeepseekV4ConfigMoHSerialization(unittest.TestCase):
    """Fields survive round-trip through to_dict."""

    def test_to_dict_includes_moh_fields(self):
        config = DeepseekV4Config(use_moh=True, num_activated_heads=16)
        d = config.to_dict()
        self.assertTrue(d["use_moh"])
        self.assertEqual(d["num_activated_heads"], 16)

    def test_default_to_dict_includes_moh_fields(self):
        config = DeepseekV4Config()
        d = config.to_dict()
        self.assertFalse(d["use_moh"])
        self.assertIsNone(d["num_activated_heads"])


class TestDeepseekV4ConfigMoHHFMapping(unittest.TestCase):
    """``_HF_TO_FLEET_FIELD_MAP`` does NOT remap these fields (same name)."""

    def test_no_hf_remapping_for_moh(self):
        # use_moh / num_activated_heads should NOT be in the remap dict since
        # the HF and Fleet names are identical. If they were remapped, their
        # default would be overwritten by the pop() in __init__.
        mapping = DeepseekV4Config._HF_TO_FLEET_FIELD_MAP
        self.assertNotIn("use_moh", mapping)
        self.assertNotIn("num_activated_heads", mapping)


if __name__ == "__main__":
    unittest.main()
