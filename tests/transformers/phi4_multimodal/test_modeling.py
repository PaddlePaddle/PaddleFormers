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

import numpy as np
import paddle
from safetensors.numpy import save_file

from paddleformers.transformers import (
    AutoConfig,
    Phi4MultimodalAudioConfig,
    Phi4MultimodalConfig,
    Phi4MultimodalForCausalLM,
    Phi4MultimodalModel,
    Phi4MultimodalVisionConfig,
)
from paddleformers.transformers.model_utils import load_state_dict
from paddleformers.transformers.phi4_multimodal.modeling import (
    Phi4MultimodalAudioAttention,
    Phi4MultimodalAudioModel,
    Phi4MultimodalDecoderLayer,
    Phi4MultimodalPreTrainedModel,
    Phi4MultimodalRotaryEmbedding,
    Phi4MultimodalVisionEncoder,
    Phi4MultimodalVisionModel,
    _lora_adapter_from_input_mode,
    _merge_multimodal_embeddings,
    adaptive_enc_mask,
)


class Phi4MultimodalModelingTest(unittest.TestCase):
    def test_subpackage_exports_standalone_modality_models(self):
        from paddleformers.transformers.phi4_multimodal import (
            Phi4MultimodalAudioModel as ExportedAudioModel,
        )
        from paddleformers.transformers.phi4_multimodal import (
            Phi4MultimodalVisionModel as ExportedVisionModel,
        )

        self.assertIs(ExportedVisionModel, Phi4MultimodalVisionModel)
        self.assertIs(ExportedAudioModel, Phi4MultimodalAudioModel)

    def test_standalone_modality_models_save_and_reload(self):
        vision_config = Phi4MultimodalVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            crop_size=28,
            patch_size=14,
        )
        audio_config = Phi4MultimodalAudioConfig(
            hidden_size=8,
            intermediate_size=12,
            num_blocks=1,
            num_attention_heads=2,
            input_size=4,
            ext_pw_out_channel=8,
            depthwise_separable_out_channel=8,
            nemo_conv_channels=8,
        )

        for model_class, config in (
            (Phi4MultimodalVisionModel, vision_config),
            (Phi4MultimodalAudioModel, audio_config),
        ):
            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmpdir:
                model.save_pretrained(tmpdir, save_checkpoint_format="", save_safetensors=False)
                reloaded = model_class.from_pretrained(
                    tmpdir,
                    load_checkpoint_format="",
                    convert_from_hf=False,
                    use_safetensors=False,
                )

            with self.subTest(model=model_class.__name__):
                self.assertEqual(set(model.state_dict()), set(reloaded.state_dict()))
                self.assertTrue(model_class._no_split_modules)

    def test_general_rms_norm_keeps_unfused_phi4_path(self):
        config = Phi4MultimodalConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        layer = Phi4MultimodalDecoderLayer(config, layer_idx=0)

        self.assertEqual(layer.input_layernorm.__class__.__name__, "RMSNorm")
        self.assertFalse(config.fuse_rms_norm)

    def test_short_longrope_torch_rounding_is_narrowly_scoped(self):
        rope_parameters = {
            "rope_type": "longrope",
            "rope_theta": 10000.0,
            "short_factor": [1.0] * 48,
            "long_factor": [2.0] * 48,
            "original_max_position_embeddings": 4096,
        }
        config = Phi4MultimodalConfig(
            hidden_size=384,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=0,
            max_position_embeddings=131072,
            original_max_position_embeddings=4096,
            rope_parameters=rope_parameters,
        )
        raw_inv_freq, _ = Phi4MultimodalRotaryEmbedding.compute_default_rope_parameters(config)
        rotary = Phi4MultimodalRotaryEmbedding(config)
        self.assertFalse(np.array_equal(raw_inv_freq.numpy(), rotary.inv_freq.numpy()))

        non_upstream_config = Phi4MultimodalConfig(
            hidden_size=384,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_hidden_layers=0,
            max_position_embeddings=131072,
            original_max_position_embeddings=4096,
            rope_parameters={**rope_parameters, "rope_theta": 10001.0},
        )
        raw_non_upstream = Phi4MultimodalRotaryEmbedding.compute_default_rope_parameters(non_upstream_config)[0]
        non_upstream_rotary = Phi4MultimodalRotaryEmbedding(non_upstream_config)
        np.testing.assert_array_equal(raw_non_upstream.numpy(), non_upstream_rotary.inv_freq.numpy())

    def test_top_level_exports_phi4mm_aliases(self):
        from paddleformers.transformers import (
            Phi4MMForCausalLM,
            Phi4MMForConditionalGeneration,
        )

        self.assertIs(Phi4MMForCausalLM, Phi4MultimodalForCausalLM)
        self.assertIs(Phi4MMForConditionalGeneration, Phi4MultimodalForCausalLM)

    def test_saved_config_keeps_native_complete_vision_config(self):
        config = Phi4MultimodalConfig(
            vocab_size=101,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=0,
            num_attention_heads=4,
            num_key_value_heads=2,
            dtype="bfloat16",
            vision_config={
                "hidden_size": 48,
                "intermediate_size": 80,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 336,
                "patch_size": 14,
                "crop_size": 336,
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            saved = json.loads((Path(tmpdir) / "config.json").read_text())

            self.assertEqual(saved["model_type"], "phi4_multimodal")
            self.assertEqual(saved["vision_config"]["hidden_size"], 48)
            self.assertEqual(saved["vision_config"]["num_hidden_layers"], 2)
            self.assertEqual(saved["vision_config"]["image_size"], 336)
            self.assertEqual(saved["vision_config"]["patch_size"], 14)
            self.assertEqual(saved["vision_config"]["crop_size"], 336)
            self.assertFalse((Path(tmpdir) / "configuration_phi4mm.py").exists())

            reloaded = AutoConfig.from_pretrained(tmpdir)
            self.assertIsInstance(reloaded, Phi4MultimodalConfig)
            self.assertEqual(reloaded.hidden_size, config.hidden_size)
            self.assertEqual(reloaded.dtype, config.dtype)
            self.assertEqual(reloaded.vision_config.hidden_size, 48)
            self.assertEqual(reloaded.vision_config.num_hidden_layers, 2)
            self.assertEqual(reloaded.vision_config.image_size, 336)

    def test_upstream_config_export_is_explicit_and_preserves_vision_config(self):
        config = Phi4MultimodalConfig(
            num_hidden_layers=0,
            dtype="bfloat16",
            vision_config={
                "hidden_size": 48,
                "intermediate_size": 80,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 336,
                "patch_size": 14,
                "crop_size": 336,
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_upstream_config(tmpdir)
            exported = json.loads((Path(tmpdir) / "config.json").read_text())

            self.assertEqual(exported["model_type"], "phi4mm")
            self.assertEqual(exported["architectures"], ["Phi4MMForCausalLM"])
            self.assertEqual(exported["auto_map"]["AutoConfig"], "configuration_phi4mm.Phi4MMConfig")
            self.assertEqual(exported["torch_dtype"], "bfloat16")
            self.assertEqual(exported["vision_config"]["hidden_size"], 48)
            self.assertEqual(exported["vision_config"]["num_hidden_layers"], 2)
            self.assertEqual(exported["vision_config"]["image_size"], 336)
            self.assertTrue((Path(tmpdir) / "configuration_phi4mm.py").is_file())

            reloaded = Phi4MultimodalConfig.from_dict(exported)
            self.assertEqual(reloaded.vision_config.hidden_size, 48)
            self.assertEqual(reloaded.vision_config.num_hidden_layers, 2)
            self.assertEqual(reloaded.vision_config.image_size, 336)

    def test_hf_linear_weights_are_transposed_by_standard_loader(self):
        weights = {
            "model.layers.0.self_attn.qkv_proj.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
            "model.embed_tokens_extend.image_embed.img_processor.encoder.layers.0.mlp.fc1.weight": np.arange(
                12, dtype=np.float32
            ).reshape(3, 4),
            "model.embed_tokens_extend.audio_embed.encoder.embed.out.weight": np.arange(20, dtype=np.float32).reshape(
                4, 5
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = str(Path(tmpdir) / "model.safetensors")
            save_file(weights, checkpoint, metadata={"format": "np"})
            loaded = load_state_dict(
                checkpoint,
                return_numpy=True,
                convert_from_hf=True,
                transpose_weight_keys=Phi4MultimodalPreTrainedModel.transpose_weight_keys,
            )

        for key, weight in weights.items():
            np.testing.assert_array_equal(loaded[key], weight.T)
        self.assertTrue(
            all(not key.endswith(".weight") for key in Phi4MultimodalPreTrainedModel.transpose_weight_keys)
        )

    def test_unsupported_tensor_and_sequence_parallel_fail_early(self):
        for sequence_parallel in (False, True):
            config = Phi4MultimodalConfig(
                num_hidden_layers=0,
                tensor_model_parallel_size=2,
                sequence_parallel=sequence_parallel,
            )
            with self.subTest(sequence_parallel=sequence_parallel):
                with self.assertRaisesRegex(NotImplementedError, "tensor parallel or sequence parallel"):
                    Phi4MultimodalModel(config)

    def test_multimodal_embeddings_are_vectorized_and_validate_token_count(self):
        input_ids = paddle.to_tensor([[1, 99, 2], [99, 3, 99]], dtype="int64")
        inputs_embeds = paddle.zeros([2, 3, 2], dtype="float32")
        features = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        merged = _merge_multimodal_embeddings(input_ids, inputs_embeds, features, 99, "Image")
        expected = paddle.to_tensor([[[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]], [[3.0, 4.0], [0.0, 0.0], [5.0, 6.0]]])
        np.testing.assert_array_equal(merged.numpy(), expected.numpy())

        with self.assertRaisesRegex(ValueError, "Image features and Image tokens do not match"):
            _merge_multimodal_embeddings(input_ids, inputs_embeds, features[:2], 99, "Image")

    def test_lora_targets_only_language_projection_layers(self):
        from paddleformers.cli.utils.llm_utils import get_lora_target_modules

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

    def test_mixed_input_modes_are_rejected(self):
        for input_modes in ([1, 2], [0, 1]):
            with self.subTest(input_modes=input_modes):
                with self.assertRaisesRegex(ValueError, "mixing different input modes"):
                    _lora_adapter_from_input_mode(paddle.to_tensor(input_modes, dtype=paddle.int64))

    def test_recompute_preserves_lora_adapter_during_backward(self):
        config = SimpleNamespace(_active_lora_adapter="vision")

        class AdapterLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.vision_lora = self.create_parameter(
                    shape=[1], default_initializer=paddle.nn.initializer.Constant(2.0)
                )

            def forward(self, hidden_states):
                if config._active_lora_adapter == "vision":
                    return hidden_states * self.vision_lora
                return hidden_states

        layer = AdapterLayer()
        reference_input = paddle.to_tensor([3.0], stop_gradient=False)
        reference_output = layer(reference_input)
        reference_output.sum().backward()
        reference_input_grad = reference_input.grad.clone()
        reference_lora_grad = layer.vision_lora.grad.clone()
        layer.clear_gradients()

        recompute_input = paddle.to_tensor([3.0], stop_gradient=False)
        recompute_output = Phi4MultimodalModel.recompute_training_full(
            SimpleNamespace(config=config), layer, recompute_input
        )
        config._active_lora_adapter = None
        recompute_output.sum().backward()

        self.assertEqual(recompute_output.tolist(), reference_output.tolist())
        self.assertEqual(recompute_input.grad.tolist(), reference_input_grad.tolist())
        self.assertEqual(layer.vision_lora.grad.tolist(), reference_lora_grad.tolist())

    def test_recompute_config_is_propagated_to_modality_encoders(self):
        config = Phi4MultimodalConfig(
            num_hidden_layers=0,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )

        for modality_config in (config.vision_config, config.audio_config):
            self.assertEqual(modality_config.recompute_granularity, "full")
            self.assertEqual(modality_config.recompute_method, "uniform")
            self.assertEqual(modality_config.recompute_num_layers, 1)

    def test_vision_and_audio_encoder_recompute_preserve_forward_and_backward(self):
        class MaskedLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.scale = self.create_parameter(shape=[1], default_initializer=paddle.nn.initializer.Constant(2.0))

            def forward(self, hidden_states, attention_mask):
                return hidden_states * self.scale + attention_mask

        attention_mask = paddle.to_tensor([[[1.0], [2.0]]])
        for encoder_class in (Phi4MultimodalVisionEncoder, Phi4MultimodalAudioModel):
            layer = MaskedLayer()
            reference_input = paddle.to_tensor([[[3.0], [4.0]]], stop_gradient=False)
            reference_output = layer(reference_input, attention_mask)
            reference_output.sum().backward()
            reference_input_grad = reference_input.grad.clone()
            reference_scale_grad = layer.scale.grad.clone()
            layer.clear_gradients()

            recompute_input = paddle.to_tensor([[[3.0], [4.0]]], stop_gradient=False)
            recompute_output = encoder_class.recompute_training_full(
                SimpleNamespace(), layer, recompute_input, attention_mask
            )
            recompute_output.sum().backward()

            with self.subTest(encoder=encoder_class.__name__):
                np.testing.assert_array_equal(recompute_output.numpy(), reference_output.numpy())
                np.testing.assert_array_equal(recompute_input.grad.numpy(), reference_input_grad.numpy())
                np.testing.assert_array_equal(layer.scale.grad.numpy(), reference_scale_grad.numpy())

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
