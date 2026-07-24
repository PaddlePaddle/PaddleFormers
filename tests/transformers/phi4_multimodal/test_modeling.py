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

from paddleformers.cli.utils.llm_utils import get_lora_target_modules
from paddleformers.transformers import Phi4MultimodalConfig, Phi4MultimodalForCausalLM
from paddleformers.transformers.phi4_multimodal.modeling import adaptive_enc_mask


class Phi4MultimodalModelingTest(unittest.TestCase):
    def test_lora_targets_only_language_projection_layers(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="phi4_multimodal"))

        self.assertEqual(
            get_lora_target_modules(model),
            [
                "model.layers.*.self_attn.qkv_proj",
                "model.layers.*.self_attn.o_proj",
                "model.layers.*.mlp.gate_up_proj",
                "model.layers.*.mlp.down_proj",
            ],
        )

    def test_adaptive_enc_mask_uses_chunk_windows(self):
        mask = adaptive_enc_mask(6, [2, 4])
        self.assertEqual(
            mask.tolist(),
            [
                [True, True, False, False, False, False],
                [True, True, False, False, False, False],
                [False, False, True, True, False, False],
                [False, False, True, True, False, False],
                [False, False, False, False, True, True],
                [False, False, False, False, True, True],
            ],
        )

        left_context_mask = adaptive_enc_mask(6, [2, 4], left_window=1)
        self.assertEqual(
            left_context_mask.tolist(),
            [
                [True, True, False, False, False, False],
                [True, True, False, False, False, False],
                [True, True, True, True, False, False],
                [True, True, True, True, False, False],
                [False, False, True, True, True, True],
                [False, False, True, True, True, True],
            ],
        )

    def test_aoa_lm_head_mapping_respects_tied_embeddings(self):
        untied_config = Phi4MultimodalConfig(num_hidden_layers=0, tie_word_embeddings=False)
        untied_statements = Phi4MultimodalForCausalLM._gen_aoa_config(untied_config)["aoa_statements"]
        self.assertIn("lm_head.weight -> lm_head.weight", untied_statements)
        self.assertNotIn("model.embed_tokens.weight -> lm_head.weight", untied_statements)

        tied_config = Phi4MultimodalConfig(num_hidden_layers=0, tie_word_embeddings=True)
        tied_statements = Phi4MultimodalForCausalLM._gen_aoa_config(tied_config)["aoa_statements"]
        self.assertIn("model.embed_tokens.weight -> lm_head.weight", tied_statements)
        self.assertNotIn("lm_head.weight -> lm_head.weight", tied_statements)

    def test_aoa_vision_head_splits_torch_in_projection(self):
        config = Phi4MultimodalConfig(num_hidden_layers=0)
        statements_text = "\n".join(Phi4MultimodalForCausalLM._gen_aoa_config(config)["aoa_statements"])

        self.assertIn(
            "head.attention.in_proj_weight -> "
            "model.embed_tokens_extend.image_embed.img_processor.head.attention.in_proj_weight.q, "
            "model.embed_tokens_extend.image_embed.img_processor.head.attention.in_proj_weight.k, "
            "model.embed_tokens_extend.image_embed.img_processor.head.attention.in_proj_weight.v, axis=0",
            statements_text,
        )
        self.assertIn(
            "head.attention.in_proj_weight.q^T -> "
            "model.embed_tokens_extend.image_embed.img_processor.head.attention.q_proj.weight",
            statements_text,
        )
        self.assertNotIn("head.attention.in_proj_weight_t", statements_text)
        self.assertNotIn("head.attention.in_proj_bias_t", statements_text)
