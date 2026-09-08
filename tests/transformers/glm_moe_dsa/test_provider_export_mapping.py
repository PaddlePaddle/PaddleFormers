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

"""Provider and HF configuration must describe the same native export graph."""
import ast
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "paddleformers/transformers/aoa_config_base.py"
spec = importlib.util.spec_from_file_location("provider_export_aoa_base", BASE)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
SOURCE = ROOT / "paddleformers/transformers/glm_moe_dsa/modeling.py"
tree = ast.parse(SOURCE.read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_gen_inv_aoa_config")
method.decorator_list = []
namespace = {"copy": copy, "MoEAOAConfigGenerator": base.MoEAOAConfigGenerator, "GlmMoeDsaConfig": object}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(SOURCE), "exec"), namespace)
export_config = namespace["_gen_inv_aoa_config"]


class ProviderExportMappingTests(unittest.TestCase):
    def configuration(self, native):
        config = SimpleNamespace(
            num_hidden_layers=4,
            num_attention_heads=64,
            num_key_value_heads=64,
            n_routed_experts=16,
            n_shared_experts=1,
            first_k_dense_replace=3,
            num_nextn_predict_layers=1,
            multi_latent_attention=True,
            moe_expert_fusion=True,
            use_accuracy_compatible=True,
        )
        setattr(config, "dsa_index_n_heads" if native else "index_n_heads", 32)
        setattr(config, "dsa_indexer_types" if native else "indexer_types", ["full", "full", "full", "shared"])
        return config

    def test_native_provider_matches_hf_configuration_without_mutation(self):
        provider = self.configuration(True)
        before = copy.deepcopy(vars(provider))
        actual = export_config(None, provider)
        self.assertEqual(actual, export_config(None, self.configuration(False)))
        self.assertEqual(vars(provider), before)
        self.assertIn(
            "model.layers.0.self_attn.core_attention.indexer.wq_b.weight^T -> "
            "model.layers.0.self_attn.indexer.wq_b.weight",
            actual["aoa_statements"],
        )

    def test_grouped_expert_intermediates_have_concrete_rules(self):
        statements = export_config(None, self.configuration(True))["aoa_statements"]
        self.assertFalse(any("$EXPERT_ID" in statement for statement in statements))
        for expert in (0, 15):
            self.assertIn(
                f"model.layers.4.transformer_layer.mlp.experts.{expert}.gate_proj.weight^T -> "
                f"model.layers.4.mlp.experts.{expert}.gate_proj.weight",
                statements,
            )
            self.assertIn(
                f"model.layers.3.mlp.experts.{expert}.up_gate_proj.weight -> "
                f"model.layers.3.mlp.experts.{expert}.gate_proj.weight, "
                f"model.layers.3.mlp.experts.{expert}.up_proj.weight, axis=1",
                statements,
            )


if __name__ == "__main__":
    unittest.main()
