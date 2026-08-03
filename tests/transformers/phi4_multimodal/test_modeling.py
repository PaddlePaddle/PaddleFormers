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

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import paddle

from paddleformers.cli.utils.llm_utils import get_lora_target_modules
from paddleformers.transformers import (
    AutoConfig,
    Phi4MultimodalConfig,
    Phi4MultimodalForCausalLM,
)
from paddleformers.transformers.phi4_multimodal.modeling import (
    Phi4MultimodalAudioAttention,
    Phi4MultimodalAudioModel,
    _lora_adapter_from_input_mode,
    adaptive_enc_mask,
)


class Phi4MultimodalModelingTest(unittest.TestCase):
    def test_top_level_exports_pipe_and_phi4mm_aliases(self):
        from paddleformers.transformers import (
            Phi4MMForCausalLM,
            Phi4MMForCausalLMPipe,
            Phi4MMForConditionalGeneration,
            Phi4MultimodalForCausalLMPipe,
        )

        self.assertIs(Phi4MMForCausalLM, Phi4MultimodalForCausalLM)
        self.assertIs(Phi4MMForConditionalGeneration, Phi4MultimodalForCausalLM)
        self.assertIs(Phi4MMForCausalLMPipe, Phi4MultimodalForCausalLMPipe)

    def test_saved_config_uses_upstream_phi4mm_metadata(self):
        config = Phi4MultimodalConfig(
            vocab_size=101,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=0,
            num_attention_heads=4,
            num_key_value_heads=2,
            dtype="bfloat16",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            exported = json.loads((Path(tmpdir) / "config.json").read_text())

            self.assertEqual(exported["model_type"], "phi4mm")
            self.assertEqual(exported["architectures"], ["Phi4MMForCausalLM"])
            self.assertEqual(exported["auto_map"]["AutoConfig"], "configuration_phi4mm.Phi4MMConfig")
            self.assertEqual(exported["torch_dtype"], "bfloat16")
            self.assertTrue((Path(tmpdir) / "configuration_phi4mm.py").is_file())

            reloaded = AutoConfig.from_pretrained(tmpdir)
            self.assertIsInstance(reloaded, Phi4MultimodalConfig)
            self.assertEqual(reloaded.hidden_size, config.hidden_size)
            self.assertEqual(reloaded.dtype, config.dtype)

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

    def test_mixed_vision_and_speech_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mixing vision and speech"):
            _lora_adapter_from_input_mode(paddle.to_tensor([1, 2], dtype=paddle.int64))

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

    def test_audio_attention_mask_sets_forbidden_weights_to_zero(self):
        hs_mask = paddle.to_tensor([[[True, False], [True, True]]])
        relative_attention_bias = paddle.zeros([1, 1, 2, 2], dtype=paddle.float32)

        attention_mask = Phi4MultimodalAudioModel._prepare_attention_mask(hs_mask, relative_attention_bias)
        attention_weights = Phi4MultimodalAudioAttention._masked_softmax(attention_mask)

        self.assertEqual(attention_weights[0, 0, 0, 1].item(), 0.0)
        self.assertGreater(attention_weights[0, 0, 0, 0].item(), 0.0)

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
