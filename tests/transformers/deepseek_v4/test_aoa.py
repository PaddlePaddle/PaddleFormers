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

from paddleformers.transformers.deepseek_v4.configuration import DeepseekV4Config
from paddleformers.transformers.deepseek_v4.modeling import DeepseekV4PreTrainedModel


class DeepseekV4MhcAoaTest(unittest.TestCase):
    @staticmethod
    def _config():
        return DeepseekV4Config(
            num_hidden_layers=2,
            n_routed_experts=2,
            n_shared_experts=1,
            moe_n_hash_layers=0,
            csa_compress_ratios=[0, 0, 0],
            mtp_num_layers=1,
            num_nextn_predict_layers=1,
            num_empty_layers_add_in_head=1,
            moe_expert_fusion=False,
            moe_deep_gemm=False,
            fp8=None,
        )

    @staticmethod
    def _is_mhc(statement):
        return any(
            key in statement
            for key in (
                "hc_attn_",
                "hc_ffn_",
                "hc_head_",
                "hyper_connection",
                "mhc_contract",
            )
        )

    @classmethod
    def _mhc_statements(cls, inverse=False):
        generator = (
            DeepseekV4PreTrainedModel._gen_inv_aoa_config if inverse else DeepseekV4PreTrainedModel._gen_aoa_config
        )
        return [statement for statement in generator(cls._config())["aoa_statements"] if cls._is_mhc(statement)]

    @staticmethod
    def _canonical_keys():
        keys = {"hc_head_base", "hc_head_fn", "hc_head_scale"}
        for layer_idx in range(2):
            for module in ("attn", "ffn"):
                for suffix in ("scale", "base", "fn"):
                    keys.add(f"layers.{layer_idx}.hc_{module}_{suffix}")
        for module in ("attn", "ffn", "head"):
            for suffix in ("scale", "base", "fn"):
                keys.add(f"mtp.0.hc_{module}_{suffix}")
        return keys

    def test_mhc_load_aoa_uses_fp32_and_covers_all_keys(self):
        statements = self._mhc_statements()
        self.assertEqual(len(statements), 42)
        self.assertTrue(all(statement.count("->") == 1 for statement in statements))
        self.assertFalse(any("bfloat16" in statement for statement in statements))

        checkpoint_keys = self._canonical_keys()
        load_sources = set()
        for statement in statements:
            lhs = statement.split("->", 1)[0]
            for key in lhs.split(","):
                key = key.strip()
                if key.endswith("^T"):
                    key = key[:-2]
                if key in checkpoint_keys:
                    load_sources.add(key)
        self.assertEqual(load_sources, checkpoint_keys)

    def test_mhc_save_aoa_uses_fp32_and_covers_all_keys(self):
        statements = self._mhc_statements(inverse=True)
        self.assertEqual(len(statements), 30)
        self.assertTrue(all(statement.count("->") == 1 for statement in statements))
        self.assertTrue(all(statement == statement.strip() for statement in statements))
        self.assertFalse(any("bfloat16" in statement for statement in statements))

        checkpoint_keys = self._canonical_keys()
        save_targets = {statement.split("->", 1)[1].split(",", 1)[0].strip() for statement in statements}
        self.assertEqual(save_targets & checkpoint_keys, checkpoint_keys)


if __name__ == "__main__":
    unittest.main()
