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
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import paddle

from paddleformers.transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from paddleformers.transformers.janus.configuration import JanusConfig
from paddleformers.transformers.janus.conversion import (
    convert_janus_state_dict,
    convert_language_state_dict,
    expected_language_keys,
    expected_multimodal_keys,
    merge_shard_state_dicts,
)
from paddleformers.transformers.janus.modeling import (
    JanusForCausalLM,
    JanusLlamaAttention,
    JanusLlamaModel,
    JanusLlamaRotaryEmbedding,
    _is_raw_hf_checkpoint_reference,
    janus_apply_rotary_pos_emb,
)
from paddleformers.transformers.janus.vision import (
    JanusVisionTransformer,
    _effective_vision_layers,
)
from paddleformers.transformers.llama.configuration import LlamaConfig
from paddleformers.transformers.llama.modeling import (
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    rotate_half,
)
from tests.testing_utils import require_package


def tiny_janus_config(with_vision=False):
    vision_config = {}
    aligner_config = {}
    if with_vision:
        vision_config = {
            "cls": "CLIPVisionTower",
            "params": {
                "image_size": 16,
                "patch_size": 4,
                "width": 8,
                "layers": 1,
                "heads": 2,
                "mlp_ratio": 2.0,
                "global_pool": "map",
                "paddle_high_precision": True,
            },
        }
        aligner_config = {
            "cls": "MlpProjector",
            "params": {
                "depth": 2,
                "input_dim": 8,
                "n_embed": 32,
                "projector_type": "mlp_gelu",
            },
        }
    return JanusConfig(
        language_config={
            "vocab_size": 97,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "torch_dtype": "float32",
            "fuse_rms_norm": False,
        },
        vision_config=vision_config,
        aligner_config=aligner_config,
        architectures=["JanusForCausalLM"],
    )


class JanusContractTest(unittest.TestCase):
    def test_shared_llama_rotary_default_keeps_float32_accumulation(self):
        """The shared Llama helper must retain its historical default path."""
        query = (paddle.arange(1 * 2 * 4 * 8, dtype="float32").reshape([1, 2, 4, 8]) / 7).astype("float16")
        key = (query * 0.7).astype("float16")
        phases = paddle.arange(4 * 8, dtype="float32").reshape([1, 4, 8])
        cos = (phases / 11).cos().astype("float16")
        sin = (phases / 13).sin().astype("float16")

        actual_query, actual_key = apply_rotary_pos_emb(query, key, cos, sin)
        expected_query = (
            query.astype("float32") * cos.unsqueeze(1) + rotate_half(query).astype("float32") * sin.unsqueeze(1)
        ).astype("float16")
        expected_key = (
            key.astype("float32") * cos.unsqueeze(1) + rotate_half(key).astype("float32") * sin.unsqueeze(1)
        ).astype("float16")

        self.assertTrue(paddle.equal(actual_query, expected_query).all().item())
        self.assertTrue(paddle.equal(actual_key, expected_key).all().item())

    def test_janus_input_dtype_rotary_is_instance_scoped(self):
        config = tiny_janus_config()
        janus_model = JanusForCausalLM(config)
        self.assertTrue(
            all(isinstance(layer.self_attn, JanusLlamaAttention) for layer in janus_model.language_model.model.layers)
        )
        self.assertIsInstance(janus_model.language_model.model, JanusLlamaModel)

        llama_model = LlamaForCausalLM(config.language_config)
        self.assertFalse(any(isinstance(layer.self_attn, JanusLlamaAttention) for layer in llama_model.model.layers))

    def test_janus_vision_constructor_does_not_change_global_cublas_switch(self):
        params = {
            "image_size": 16,
            "patch_size": 4,
            "width": 8,
            "layers": 1,
            "heads": 2,
            "mlp_ratio": 2.0,
            "paddle_high_precision": True,
        }
        with (
            mock.patch("paddle.is_compiled_with_cuda", return_value=True),
            mock.patch("paddle.base.core.set_cublas_switch") as set_cublas_switch,
        ):
            JanusVisionTransformer(params)
        set_cublas_switch.assert_not_called()

    def test_bfloat16_rope_rounds_inverse_frequency_before_frequency_product(self):
        """RoPE must follow the dtype of the model buffer as in the Torch reference."""
        config = LlamaConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            max_position_embeddings=64,
            rope_theta=10000.0,
        )
        rotary = JanusLlamaRotaryEmbedding(config)
        hidden_states = paddle.zeros([1, 4, 32], dtype="bfloat16")
        position_ids = paddle.arange(4, dtype="int64").unsqueeze(0)

        actual_cos, actual_sin = rotary(hidden_states, position_ids)

        inverse_frequency = rotary.inv_freq.astype("bfloat16").astype("float32")
        frequencies = (inverse_frequency[None, :, None] @ position_ids[:, None, :].astype("float32")).transpose(
            [0, 2, 1]
        )
        embedding = paddle.concat((frequencies, frequencies), axis=-1)
        expected_cos = embedding.cos().astype("bfloat16")
        expected_sin = embedding.sin().astype("bfloat16")

        self.assertTrue(paddle.equal(actual_cos, expected_cos).all().item())
        self.assertTrue(paddle.equal(actual_sin, expected_sin).all().item())

    def test_float32_runtime_rope_uses_unrounded_original_frequency(self):
        """Promoting a BF16 checkpoint must not retain BF16-rounded RoPE values."""
        config = LlamaConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            max_position_embeddings=64,
            rope_theta=10000.0,
        )
        rotary = JanusLlamaRotaryEmbedding(config)
        original = rotary.original_inv_freq.clone()
        # Simulate the checkpoint/runtime sequence: model is first materialized
        # in BF16, then the language tower is promoted to FP32.
        rotary.register_buffer("inv_freq", rotary.inv_freq.astype("bfloat16"), persistable=False)

        hidden_states = paddle.zeros([1, 4, 32], dtype="float32")
        position_ids = paddle.arange(4, dtype="int64").unsqueeze(0)
        actual_cos, actual_sin = rotary(hidden_states, position_ids)

        frequencies = (original[None, :, None] @ position_ids[:, None, :].astype("float32")).transpose([0, 2, 1])
        embedding = paddle.concat((frequencies, frequencies), axis=-1)
        expected_cos = embedding.cos()
        expected_sin = embedding.sin()

        self.assertLess(float(paddle.max(paddle.abs(actual_cos - expected_cos))), 1e-7)
        self.assertLess(float(paddle.max(paddle.abs(actual_sin - expected_sin))), 1e-7)

    @unittest.skipUnless(
        paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
        "BF16 elementwise kernels require a CUDA device",
    )
    def test_bfloat16_rotary_apply_preserves_reference_rounding(self):
        paddle.set_device("gpu")
        query = (paddle.arange(1 * 2 * 4 * 8, dtype="float32").reshape([1, 2, 4, 8]) / 7).astype("bfloat16")
        key = (query * 0.7).astype("bfloat16")
        phases = paddle.arange(4 * 8, dtype="float32").reshape([1, 4, 8])
        cos = (phases / 11).cos().astype("bfloat16")
        sin = (phases / 13).sin().astype("bfloat16")

        actual_query, actual_key = janus_apply_rotary_pos_emb(query, key, cos, sin)

        def rotate_half(value):
            return paddle.concat((-value[..., 4:], value[..., :4]), axis=-1)

        expected_query = query * cos.unsqueeze(1) + rotate_half(query) * sin.unsqueeze(1)
        expected_key = key * cos.unsqueeze(1) + rotate_half(key) * sin.unsqueeze(1)
        self.assertTrue(paddle.equal(actual_query, expected_query).all().item())
        self.assertTrue(paddle.equal(actual_key, expected_key).all().item())

    def test_config_defaults_and_round_trip(self):
        config = JanusConfig(language_config={"vocab_size": 97})
        self.assertEqual(config.model_type, "multi_modality")
        self.assertEqual(config.language_config.hidden_size, 4096)
        self.assertEqual(config.language_config.num_hidden_layers, 30)
        self.assertEqual(config.language_config.num_key_value_heads, 32)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = JanusConfig.from_pretrained(tmpdir)
        self.assertIsInstance(loaded.language_config, type(config.language_config))
        self.assertEqual(loaded.language_config.vocab_size, 97)

    def test_auto_classes_resolve_local_config(self):
        config = tiny_janus_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            self.assertIsInstance(AutoConfig.from_pretrained(tmpdir), JanusConfig)
        self.assertIsInstance(AutoModel.from_config(config), JanusForCausalLM)
        self.assertIsInstance(AutoModelForCausalLM.from_config(config), JanusForCausalLM)

    def test_text_forward_and_generation_delegate(self):
        model = JanusForCausalLM(tiny_janus_config())
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        output = model(input_ids=input_ids, use_cache=False, return_dict=True)
        self.assertEqual(tuple(output.logits.shape), (1, 4, 97))
        self.assertIs(model.get_input_embeddings(), model.language_model.get_input_embeddings())
        prepared = model.prepare_inputs_for_generation(input_ids, use_cache=False)
        self.assertIn("input_ids", prepared)

    def test_causal_loss_shifts_labels_like_transformers(self):
        model = JanusForCausalLM(tiny_janus_config())
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 5]], dtype="int64")
        labels = paddle.to_tensor([[-100, 2, 3, 4, 5]], dtype="int64")

        with paddle.no_grad():
            output = model(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            expected = paddle.nn.functional.cross_entropy(
                output.logits[:, :-1, :].reshape([-1, output.logits.shape[-1]]),
                labels[:, 1:].reshape([-1, 1]),
                reduction="mean",
                ignore_index=-100,
            )

        self.assertAlmostEqual(float(output.loss), float(expected), places=6)

    def test_image_inputs_prepare_embeddings(self):
        model = JanusForCausalLM(tiny_janus_config(with_vision=True))
        self.assertTrue(model.vision_model.vision_tower.high_precision)
        input_ids = paddle.ones([1, 3], dtype="int64")
        pixel_values = paddle.zeros([1, 1, 3, 16, 16], dtype="float32")
        images_seq_mask = paddle.to_tensor([[False, True, True]], dtype="bool")
        images_emb_mask = paddle.to_tensor([[[True, True] + [False] * 14]], dtype="bool")
        with paddle.no_grad():
            embeds = model.prepare_inputs_embeds(
                input_ids,
                pixel_values,
                images_seq_mask,
                images_emb_mask,
            )
        self.assertEqual(tuple(embeds.shape), (1, 3, 32))

        with paddle.no_grad():
            output = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                images_seq_mask=images_seq_mask,
                images_emb_mask=images_emb_mask,
                use_cache=False,
                return_dict=True,
            )
        self.assertEqual(tuple(output.logits.shape), (1, 3, 97))

    def test_image_generation_with_cache_only_uses_images_during_prefill(self):
        model = JanusForCausalLM(tiny_janus_config(with_vision=True))
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        pixel_values = paddle.zeros([1, 1, 3, 16, 16], dtype="float32")
        images_seq_mask = paddle.to_tensor([[False, True, True]], dtype="bool")
        images_emb_mask = paddle.to_tensor([[[True, True] + [False] * 14]], dtype="bool")
        vision_outputs = []

        def capture_vision_output(layer, inputs, output):
            vision_outputs.append(output)

        hook = model.vision_model.register_forward_post_hook(capture_vision_output)
        try:
            with paddle.no_grad():
                output_ids, _ = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    images_seq_mask=images_seq_mask,
                    images_emb_mask=images_emb_mask,
                    decode_strategy="greedy_search",
                    max_new_tokens=2,
                    use_cache=True,
                )
        finally:
            hook.remove()

        self.assertEqual(tuple(output_ids.shape), (1, 2))
        self.assertEqual(len(vision_outputs), 1)

    def test_pixel_values_reject_cross_sample_placeholder_counts(self):
        model = JanusForCausalLM(tiny_janus_config(with_vision=True))
        input_ids = paddle.ones([2, 3], dtype="int64")
        pixel_values = paddle.zeros([2, 1, 3, 16, 16], dtype="float32")
        images_seq_mask = paddle.to_tensor(
            [[False, True, True], [False, False, False]],
            dtype="bool",
        )
        images_emb_mask = paddle.to_tensor(
            [
                [[True] + [False] * 15],
                [[True] + [False] * 15],
            ],
            dtype="bool",
        )

        with self.assertRaisesRegex(ValueError, "for each sample"):
            model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                images_seq_mask=images_seq_mask,
                images_emb_mask=images_emb_mask,
                use_cache=False,
            )

    def test_image_embeds_reject_cross_sample_placeholder_counts(self):
        model = JanusForCausalLM(tiny_janus_config(with_vision=True))
        input_ids = paddle.ones([2, 3], dtype="int64")
        image_embeds = paddle.zeros([2, 16, 32], dtype="float32")
        images_seq_mask = paddle.to_tensor(
            [[False, True, True], [False, False, False]],
            dtype="bool",
        )
        images_emb_mask = paddle.to_tensor(
            [
                [[True] + [False] * 15],
                [[True] + [False] * 15],
            ],
            dtype="bool",
        )

        with self.assertRaisesRegex(ValueError, "for each sample"):
            model(
                input_ids=input_ids,
                image_embeds=image_embeds,
                images_seq_mask=images_seq_mask,
                images_emb_mask=images_emb_mask,
                use_cache=False,
            )

    def test_vision_parity_precision_is_explicit_and_validated(self):
        config = tiny_janus_config(with_vision=True)
        config.vision_config["params"]["vision_parity_precision"] = "fp64_accumulate"
        model = JanusForCausalLM(config)
        tower = model.vision_model.vision_tower
        self.assertEqual(tower.vision_parity_precision, "fp64_accumulate")
        self.assertTrue(tower.high_precision)

        config.vision_config["params"]["vision_parity_precision"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "unsupported vision parity precision"):
            JanusForCausalLM(config)

    def test_bfloat16_parity_policy_promotes_runtime_compute_dtypes(self):
        """BF16 checkpoints use the auditable mixed-precision runtime policy."""
        config = tiny_janus_config(with_vision=True)
        config.language_compute_dtype = "float32"
        config.vision_compute_dtype = "float64"

        model = JanusForCausalLM.from_config(config, dtype="bfloat16")

        self.assertEqual(model.language_model.model.embed_tokens.weight.dtype, paddle.float32)
        self.assertEqual(model.language_model.lm_head.weight.dtype, paddle.float32)
        self.assertEqual(model.vision_model.vision_tower.patch_embed.proj.weight.dtype, paddle.float64)
        self.assertEqual(model.aligner.layers[0].weight.dtype, paddle.float64)

    def test_bfloat16_multimodal_defaults_to_parity_runtime_policy(self):
        """An unspecified BF16 multimodal policy must not silently use native kernels."""
        config = tiny_janus_config(with_vision=True)
        config.language_config.torch_dtype = "bfloat16"

        model = JanusForCausalLM.from_config(config, dtype="bfloat16")

        self.assertEqual(config.language_compute_dtype, "float32")
        self.assertEqual(config.vision_compute_dtype, "float64")
        self.assertEqual(model.language_model.model.embed_tokens.weight.dtype, paddle.float32)
        self.assertEqual(model.vision_model.vision_tower.patch_embed.proj.weight.dtype, paddle.float64)
        self.assertEqual(model.aligner.layers[0].weight.dtype, paddle.float64)

    def test_explicit_native_vision_policy_is_not_overridden(self):
        """A native diagnostic config remains native under BF16 construction."""
        config = tiny_janus_config(with_vision=True)
        config.language_config.torch_dtype = "bfloat16"
        config.vision_config["params"]["vision_parity_precision"] = "native"
        config.vision_config["params"]["paddle_high_precision"] = False

        model = JanusForCausalLM.from_config(config, dtype="bfloat16")

        self.assertIsNone(config.language_compute_dtype)
        self.assertIsNone(config.vision_compute_dtype)
        self.assertEqual(model.runtime_compute_dtypes, {"language": "checkpoint", "vision": "checkpoint"})
        self.assertFalse(model.vision_model.vision_tower.high_precision)


class JanusConversionTest(unittest.TestCase):
    def setUp(self):
        self.model = JanusForCausalLM(tiny_janus_config())
        self.target = {key: value.numpy() for key, value in self.model.state_dict().items()}

    def source_state_dict(self):
        source = {key: value.copy() for key, value in self.target.items()}
        for key in source:
            if any(key.endswith(f".{name}.weight") for name in self.model.transpose_weight_keys):
                source[key] = source[key].T
        source["vision_model.dummy.weight"] = np.ones([2, 3], dtype="float32")
        return source

    def test_language_conversion_transposes_only_linear_weights(self):
        converted, report = convert_language_state_dict(self.source_state_dict(), self.target)

        self.assertEqual(set(converted), expected_language_keys(self.target))
        self.assertEqual(report["accepted_language_keys"], len(self.target))
        self.assertEqual(report["skipped_non_language_keys"], 1)
        self.assertEqual(report["skipped_keys"], ["vision_model.dummy.weight"])
        for key, expected in self.target.items():
            np.testing.assert_array_equal(converted[key], expected)

    def test_language_conversion_reports_all_key_errors(self):
        source = self.source_state_dict()
        source.pop("language_model.model.norm.weight")
        source["language_model.unexpected.weight"] = np.ones([1], dtype="float32")

        with self.assertRaisesRegex(ValueError, "missing.*language_model.model.norm.weight") as context:
            convert_language_state_dict(source, self.target)

        self.assertIn("unexpected language keys: language_model.unexpected.weight", str(context.exception))

    def test_duplicate_keys_across_shards_are_rejected(self):
        duplicate_key = "language_model.model.embed_tokens.weight"
        shards = [
            {duplicate_key: np.ones([2, 2], dtype="float32")},
            {duplicate_key: np.zeros([2, 2], dtype="float32")},
        ]

        with self.assertRaisesRegex(ValueError, f"duplicate language keys: {duplicate_key}"):
            merge_shard_state_dicts(shards)

    def test_multimodal_conversion_accepts_vision_aligner_and_language(self):
        model = JanusForCausalLM(tiny_janus_config(with_vision=True))
        target = {key: value.numpy() for key, value in model.state_dict().items()}
        source = {}
        linear_suffixes = (
            ".qkv.weight",
            ".proj.weight",
            ".q.weight",
            ".kv.weight",
            ".fc1.weight",
            ".fc2.weight",
            "aligner.layers.0.weight",
            "aligner.layers.2.weight",
        )
        for key, value in target.items():
            if value.ndim == 2 and (
                key.endswith(linear_suffixes)
                or any(key.endswith(f".{name}.weight") for name in model.transpose_weight_keys)
            ):
                value = value.T
            source[key] = np.ascontiguousarray(value)
        source["gen_embed.weight"] = np.ones([2, 3], dtype="float32")

        converted, report = convert_janus_state_dict(source, target)

        self.assertEqual(set(converted), expected_multimodal_keys(target))
        self.assertGreater(report["accepted_vision_keys"], 0)
        self.assertEqual(report["accepted_aligner_keys"], 4)
        self.assertEqual(report["skipped_keys"], ["gen_embed.weight"])
        for key, expected in target.items():
            np.testing.assert_array_equal(converted[key], expected)


class JanusCheckpointTest(unittest.TestCase):
    def test_raw_hf_detection_is_scoped_to_huggingface_sources(self):
        self.assertTrue(_is_raw_hf_checkpoint_reference("deepseek-ai/Janus-Pro-7B", download_hub="huggingface"))
        self.assertFalse(_is_raw_hf_checkpoint_reference("paddle/Janus-Pro-7B", download_hub="aistudio"))
        self.assertFalse(_is_raw_hf_checkpoint_reference("paddle/Janus-Pro-7B", download_hub="modelscope"))

    def test_remote_detection_uses_format_probe_when_source_is_unspecified(self):
        with mock.patch(
            "paddleformers.transformers.janus.modeling._probe_remote_hf_checkpoint_reference",
            return_value=False,
        ) as probe:
            self.assertFalse(_is_raw_hf_checkpoint_reference("org/native-janus", download_hub=None))
            probe.assert_called_once_with("org/native-janus", None)

        with mock.patch(
            "paddleformers.transformers.janus.modeling._probe_remote_hf_checkpoint_reference",
            return_value=True,
        ):
            self.assertTrue(_is_raw_hf_checkpoint_reference("org/raw-janus", download_hub=None))

    def test_select_layer_matches_official_vision_depth_semantics(self):
        base = {
            "model_name": "siglip_large_patch16_384",
            "layers": 4,
            "image_size": 16,
            "patch_size": 4,
            "width": 8,
            "heads": 2,
            "mlp_ratio": 2.0,
        }
        self.assertEqual(_effective_vision_layers({**base, "select_layer": -1}), 4)
        self.assertEqual(_effective_vision_layers({**base, "select_layer": -2}), 3)
        self.assertEqual(_effective_vision_layers({**base, "select_layer": 2}), 2)
        self.assertEqual(_effective_vision_layers({**base, "select_layer": 0}), 4)
        with self.assertRaisesRegex(ValueError, "select_layer"):
            _effective_vision_layers({**base, "select_layer": -5})

        tower = JanusVisionTransformer({**base, "select_layer": -2})
        self.assertEqual(len(tower.blocks), 3)

    def test_select_layer_keeps_model_and_aoa_depth_in_sync(self):
        config = tiny_janus_config(with_vision=True)
        config.vision_config["params"].update({"layers": 3, "select_layer": -2})
        model = JanusForCausalLM(config)
        self.assertEqual(len(model.vision_model.vision_tower.blocks), 2)
        statements = JanusForCausalLM._gen_aoa_config(config)["aoa_statements"]
        block_ids = {
            int(match.group(1))
            for statement in statements
            for match in [re.search(r"vision_tower\.blocks\.(\d+)\.", statement)]
            if match
        }
        self.assertEqual(block_ids, {0, 1})

    def test_transpose_patterns_are_scoped_to_janus_modules(self):
        patterns = JanusForCausalLM.transpose_weight_keys
        self.assertIn(r"aligner\.layers\.\d+", patterns)
        self.assertNotIn(r"layers\.\d+", patterns)
        self.assertTrue(
            any(
                re.search(rf"\.{pattern}\.weight$", ".aligner.layers.0.weight")
                or re.fullmatch(rf"{pattern}\.weight", "aligner.layers.0.weight")
                for pattern in patterns
            )
        )
        self.assertFalse(
            any(
                re.search(rf"\.{pattern}\.weight$", ".future.layers.0.weight")
                or re.fullmatch(rf"{pattern}\.weight", "future.layers.0.weight")
                for pattern in patterns
            )
        )

    def test_unrelated_metadata_does_not_hide_local_hf_safetensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()
            Path(tmpdir, "README.md.metadata").touch()
            self.assertTrue(_is_raw_hf_checkpoint_reference(tmpdir))

    def test_raw_hf_detection_supports_variants_and_subfolders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir, "weights")
            weights_dir.mkdir()
            Path(weights_dir, "model.fp32.safetensors").touch()
            self.assertTrue(_is_raw_hf_checkpoint_reference(tmpdir, subfolder="weights"))

    def test_aoa_config_covers_every_tiny_multimodal_parameter(self):
        from paddle.distributed.flex_checkpoint.aoa.aoa_engine import AOAEngine
        from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
            ShardedWeightDesc,
        )

        model = JanusForCausalLM(tiny_janus_config(with_vision=True))
        source = {}
        target = {}
        language_projection_names = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        for key, value in model.state_dict().items():
            target_shape = tuple(value.shape)
            transpose = value.ndim == 2 and (
                (
                    key.startswith("language_model.")
                    and any(key.endswith(f".{name}.weight") for name in language_projection_names)
                )
                or (
                    key.startswith("vision_model.")
                    and key.endswith(
                        (".qkv.weight", ".q.weight", ".kv.weight", ".proj.weight", ".fc1.weight", ".fc2.weight")
                    )
                )
                or (key.startswith("aligner.") and key.endswith(".weight"))
            )
            source_shape = tuple(reversed(target_shape)) if transpose else target_shape
            source[key] = [ShardedWeightDesc(key, source_shape, source_shape, (0,) * len(source_shape))]
            target[key] = [ShardedWeightDesc(key, target_shape, target_shape, (0,) * len(target_shape))]

        engine = AOAEngine(JanusForCausalLM._gen_aoa_config(model.config), source, target)
        self.assertEqual(set(engine.output_vars), set(target))
        self.assertTrue(
            all(engine.output_vars[key].shape == tuple(value[0].global_shape) for key, value in target.items())
        )

    def test_hf_safetensors_are_loaded_directly_with_multimodal_transposes(self):
        """A raw HF safetensors directory must be loadable without an offline conversion."""
        from safetensors.numpy import save_file

        source_model = JanusForCausalLM(tiny_janus_config(with_vision=True))
        source_model.eval()
        source_state = {}
        for key, value in source_model.state_dict().items():
            array = value.numpy()
            needs_transpose = (
                (
                    key.startswith("language_model.")
                    and any(key.endswith(f".{name}.weight") for name in source_model.transpose_weight_keys)
                )
                or (
                    key.startswith("vision_model.")
                    and key.endswith(
                        (".qkv.weight", ".q.weight", ".kv.weight", ".proj.weight", ".fc1.weight", ".fc2.weight")
                    )
                )
                or (key.startswith("aligner.") and key.endswith(".weight") and array.ndim == 2)
            )
            if needs_transpose and array.ndim == 2:
                array = np.ascontiguousarray(array.T)
            source_state[key] = np.ascontiguousarray(array)
        # The official checkpoint also contains generator-only tensors.  They
        # must be ignored by the understanding model rather than treated as a
        # load error.
        source_state["gen_embed.weight"] = np.ones([2, 3], dtype="float32")

        input_ids = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        pixel_values = paddle.zeros([1, 1, 3, 16, 16], dtype="float32")
        images_seq_mask = paddle.to_tensor([[False, True, True]], dtype="bool")
        images_emb_mask = paddle.to_tensor([[[True, True] + [False] * 14]], dtype="bool")
        with paddle.no_grad():
            expected = source_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                images_seq_mask=images_seq_mask,
                images_emb_mask=images_emb_mask,
                use_cache=False,
                return_dict=True,
            ).logits.numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model.config.save_pretrained(tmpdir)
            save_file(source_state, str(Path(tmpdir) / "model.safetensors"))
            loaded = JanusForCausalLM.from_pretrained(tmpdir)
            loaded.eval()
            with paddle.no_grad():
                actual = loaded(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    images_seq_mask=images_seq_mask,
                    images_emb_mask=images_emb_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits.numpy()

        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)

    def test_sharded_hf_safetensors_are_loaded_directly(self):
        """The automatic path must also handle the standard HF shard index."""
        from safetensors.numpy import save_file

        source_model = JanusForCausalLM(tiny_janus_config())
        source_state = {}
        for key, value in source_model.state_dict().items():
            array = value.numpy()
            if array.ndim == 2 and any(
                key.endswith(f".{name}.weight") for name in source_model.transpose_weight_keys[:7]
            ):
                array = np.ascontiguousarray(array.T)
            source_state[key] = np.ascontiguousarray(array)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_model.config.save_pretrained(tmpdir)
            keys = sorted(source_state)
            midpoint = len(keys) // 2
            shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
            save_file({key: source_state[key] for key in keys[:midpoint]}, str(Path(tmpdir) / shard_names[0]))
            save_file({key: source_state[key] for key in keys[midpoint:]}, str(Path(tmpdir) / shard_names[1]))
            index = {
                "metadata": {"total_size": sum(array.nbytes for array in source_state.values())},
                "weight_map": {
                    key: shard_names[0] if index < midpoint else shard_names[1] for index, key in enumerate(keys)
                },
            }
            (Path(tmpdir) / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

            loaded = JanusForCausalLM.from_pretrained(tmpdir)

        for key, value in source_model.state_dict().items():
            np.testing.assert_array_equal(value.numpy(), loaded.state_dict()[key].numpy())

    def test_save_and_reload_unsharded_and_sharded(self):
        model = JanusForCausalLM(tiny_janus_config())
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        expected = model(input_ids=input_ids, use_cache=False, return_dict=True).logits

        for max_shard_size in (None, "10KB"):
            with tempfile.TemporaryDirectory() as tmpdir:
                save_kwargs = {"save_safetensors": False, "save_checkpoint_format": ""}
                if max_shard_size is not None:
                    save_kwargs["max_shard_size"] = max_shard_size
                model.save_pretrained(tmpdir, **save_kwargs)

                reloaded = JanusForCausalLM.from_pretrained(
                    tmpdir,
                    convert_from_hf=False,
                    load_checkpoint_format="",
                )
                actual = reloaded(input_ids=input_ids, use_cache=False, return_dict=True).logits
                self.assertTrue(paddle.allclose(expected, actual, atol=0.0, rtol=0.0))
                self.assertTrue(all(key.startswith("language_model.") for key in reloaded.state_dict()))

    @require_package("transformers", "torch")
    def test_tiny_torch_paddle_parity(self):
        import torch
        import transformers

        config = tiny_janus_config()
        paddle_model = JanusForCausalLM(config)
        paddle_model.eval()
        torch_config = transformers.LlamaConfig(
            vocab_size=config.language_config.vocab_size,
            hidden_size=config.language_config.hidden_size,
            intermediate_size=config.language_config.intermediate_size,
            num_hidden_layers=config.language_config.num_hidden_layers,
            num_attention_heads=config.language_config.num_attention_heads,
            num_key_value_heads=config.language_config.num_key_value_heads,
            head_dim=config.language_config.head_dim,
            max_position_embeddings=config.language_config.max_position_embeddings,
            rms_norm_eps=config.language_config.rms_norm_eps,
            rope_theta=config.language_config.rope_theta,
            tie_word_embeddings=False,
        )
        torch_model = transformers.LlamaForCausalLM(torch_config)
        torch_state = {}
        for paddle_key, paddle_value in paddle_model.state_dict().items():
            torch_key = paddle_key.removeprefix("language_model.")
            value = paddle_value.numpy()
            if value.ndim == 2 and any(
                torch_key.endswith(f".{name}.weight") for name in paddle_model.transpose_weight_keys
            ):
                value = value.T
            torch_state[torch_key] = torch.from_numpy(np.ascontiguousarray(value))
        torch_model.load_state_dict(torch_state, strict=True)
        torch_model.eval()

        input_ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6]], dtype="int64")
        with paddle.no_grad():
            paddle_logits = paddle_model(input_ids=input_ids, use_cache=False, return_dict=True).logits
        torch_input_ids = torch.tensor(input_ids.numpy(), dtype=torch.long)
        with torch.no_grad():
            torch_logits = torch_model(torch_input_ids, use_cache=False).logits

        paddle_logits_np = paddle_logits.numpy()
        torch_logits_np = torch_logits.detach().cpu().numpy()
        max_abs_diff = np.max(np.abs(paddle_logits_np - torch_logits_np))
        self.assertLessEqual(max_abs_diff, 1e-2)
        self.assertEqual(
            np.argmax(paddle_logits_np, axis=-1)[:, :10].tolist(),
            np.argmax(torch_logits_np, axis=-1)[:, :10].tolist(),
        )

        paddle_generated = input_ids.clone()
        torch_generated = torch_input_ids.clone()
        for _ in range(10):
            with paddle.no_grad():
                next_paddle = paddle.argmax(
                    paddle_model(input_ids=paddle_generated, use_cache=False, return_dict=True).logits[:, -1, :],
                    axis=-1,
                ).unsqueeze(-1)
            with torch.no_grad():
                next_torch = torch.argmax(torch_model(torch_generated, use_cache=False).logits[:, -1, :], dim=-1)
            paddle_generated = paddle.concat([paddle_generated, next_paddle], axis=-1)
            torch_generated = torch.cat([torch_generated, next_torch.unsqueeze(-1)], dim=-1)
        self.assertEqual(paddle_generated[:, -10:].numpy().tolist(), torch_generated[:, -10:].tolist())
