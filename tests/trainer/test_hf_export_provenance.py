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

"""Test the actual metadata helper without importing GPU training dependencies."""

import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("checkpoint_export", ROOT / "paddleformers/trainer/checkpoint_export.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class ExportProvenanceTests(unittest.TestCase):
    def test_live_provider_and_aoa_snapshot_are_serializable_and_independent(self):
        config = SimpleNamespace(
            multi_latent_attention=True,
            moe_expert_fusion=True,
            index_n_heads=32,
            indexer_types=["full", "shared"],
            num_nextn_predict_layers=1,
            n_routed_experts=16,
        )
        aoa = {"aoa_statements": ["internal.weight^T -> official.weight"]}
        record = EXPORT.hf_export_provenance(config, aoa, "results/checkpoint", 2)
        json.dumps(record)
        self.assertTrue(record["provider_config"]["multi_latent_attention"])
        self.assertEqual(record["provider_config"]["index_n_heads"], 32)
        self.assertEqual(record["global_step"], 2)
        self.assertEqual(record["output_dir"], os.path.abspath("results/checkpoint"))
        config.indexer_types.append("changed")
        aoa["aoa_statements"].clear()
        self.assertEqual(record["provider_config"]["indexer_types"], ["full", "shared"])
        self.assertEqual(record["aoa_config"]["aoa_statements"], ["internal.weight^T -> official.weight"])

    def test_missing_is_distinct_from_false_or_none_and_not_completion(self):
        record = EXPORT.hf_export_provenance(
            SimpleNamespace(moe_expert_fusion=False, params_dtype=None), {}, "checkpoint"
        )
        self.assertFalse(record["provider_config"]["moe_expert_fusion"])
        self.assertIsNone(record["provider_config"]["params_dtype"])
        self.assertIn("index_n_heads", record["missing_provider_fields"])
        self.assertNotIn("params_dtype", record["missing_provider_fields"])
        self.assertIsNone(record["global_step"])
        self.assertEqual(record["stage"], "prepared")


if __name__ == "__main__":
    unittest.main()
