# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import tempfile
import unittest

import paddle
import paddle.nn.functional as F
from safetensors.numpy import save_file

from paddleformers.transformers import (
    InternVisionModel,
    InternVLChatConfig,
    InternVLChatModel,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLMDeprecated,
)
from tests.transformers.test_configuration_common import ConfigTester


class InternVLModelTest(unittest.TestCase):
    @staticmethod
    def _to_hf_layout_state_dict(model):
        hf_transpose_keys = set(model.transpose_weight_keys)
        hf_transpose_keys.add("gate")
        state_dict = {}
        for key, value in model.state_dict().items():
            array = value.numpy()
            if array.ndim == 2 and any(
                key.endswith(f".{transpose_key}.weight") or key == f"{transpose_key}.weight"
                for transpose_key in hf_transpose_keys
            ):
                array = array.T.copy()
            state_dict[key] = array
        return state_dict

    def get_config(self):
        return InternVLChatConfig(
            vision_config={
                "image_size": 28,
                "patch_size": 14,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "qkv_bias": True,
                "qk_normalization": False,
                "norm_type": "layer_norm",
                "drop_path_rate": 0.0,
            },
            llm_config={
                "architectures": ["Qwen3ForCausalLM"],
                "vocab_size": 200,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000,
                "attention_dropout": 0.0,
                "attention_bias": False,
                "hidden_act": "silu",
                "use_cache": False,
                "bos_token_id": 0,
                "eos_token_id": 2,
                "pad_token_id": 1,
            },
            force_image_size=28,
            downsample_ratio=0.5,
            ps_version="v2",
            img_context_token_id=151,
        )

    def get_inputs(self):
        return {
            "input_ids": paddle.to_tensor([[10, 151, 11, 12]], dtype="int64"),
            "pixel_values": paddle.randn([1, 3, 28, 28]),
        }

    def get_moe_config(self):
        config = self.get_config()
        llm_config = config.llm_config.to_dict()
        llm_config.update(
            {
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "decoder_sparse_step": 1,
                "moe_intermediate_size": 8,
                "num_experts_per_tok": 1,
                "num_experts": 2,
                "norm_topk_prob": False,
                "output_router_logits": False,
            }
        )
        return InternVLChatConfig(
            vision_config=config.vision_config.to_dict(),
            llm_config=llm_config,
            force_image_size=config.force_image_size,
            downsample_ratio=config.downsample_ratio,
            ps_version=config.ps_version,
            img_context_token_id=config.img_context_token_id,
        )

    def test_config(self):
        config = self.get_config()
        config_tester = ConfigTester(
            self,
            config_class=InternVLChatConfig,
            has_text_modality=True,
            common_properties=[],
            vision_config=config.vision_config.to_dict(),
            llm_config=config.llm_config.to_dict(),
            force_image_size=config.force_image_size,
            downsample_ratio=config.downsample_ratio,
            ps_version=config.ps_version,
            img_context_token_id=config.img_context_token_id,
        )
        config_tester.create_and_test_config_from_and_save_pretrained()

    def test_forward_and_loss(self):
        model = InternVLChatModel(self.get_config()).eval()
        inputs = self.get_inputs()
        labels = paddle.to_tensor([[-100, -100, 11, 12]], dtype="int64")
        with paddle.no_grad():
            outputs = model(**inputs, labels=labels, use_cache=False)
        self.assertEqual(list(outputs.logits.shape), [1, 4, 200])
        self.assertEqual(outputs.loss.ndim, 0)
        flat_logits = outputs.logits.reshape([-1, 200])
        flat_labels = labels.reshape([-1])
        valid_mask = flat_labels != -100
        safe_labels = paddle.where(valid_mask, flat_labels, paddle.zeros_like(flat_labels))
        token_loss = F.cross_entropy(flat_logits, safe_labels, reduction="none")
        expected_loss = (token_loss * valid_mask.astype(token_loss.dtype)).sum() / valid_mask.astype(
            token_loss.dtype
        ).sum()
        paddle.testing.assert_close(outputs.loss, expected_loss)

    def test_bfloat16_vision_model_accepts_float32_pixel_values(self):
        model = InternVisionModel(self.get_config().vision_config).eval()
        model.to(dtype="bfloat16")
        pixel_values = paddle.randn([1, 3, 28, 28], dtype="float32")

        with paddle.no_grad():
            outputs = model(pixel_values=pixel_values)

        self.assertEqual(outputs.last_hidden_state.dtype, paddle.bfloat16)

    def test_qwen3_moe_config_model_and_conversion_dispatch(self):
        config = self.get_moe_config()
        model = InternVLChatModel(config).eval()

        self.assertIsInstance(config.llm_config, Qwen3MoeConfig)
        self.assertIsInstance(model.language_model, Qwen3MoeForCausalLMDeprecated)

        mappings = InternVLChatModel._get_fuse_or_split_param_mappings(config)
        self.assertIn(
            (
                "language_model.model.layers.0.mlp.experts.0.gate_proj.weight",
                "language_model.model.layers.0.mlp.experts.0.up_proj.weight",
                "language_model.model.layers.0.mlp.experts.0.up_gate_proj.weight",
            ),
            mappings,
        )
        aoa_statements = InternVLChatModel._gen_aoa_config(config)["aoa_statements"]
        self.assertTrue(
            any(
                "language_model.model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight" in statement
                and "up_gate_proj.weight" in statement
                for statement in aoa_statements
            )
        )

        with paddle.no_grad():
            outputs = model(**self.get_inputs(), use_cache=False)

        self.assertEqual(list(outputs.logits.shape), [1, 4, 200])
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            reloaded_config = InternVLChatConfig.from_pretrained(tmpdir, local_files_only=True)
        self.assertIsInstance(reloaded_config.llm_config, Qwen3MoeConfig)

    def test_qwen3_moe_hf_checkpoint_transposes_router_gate(self):
        config = self.get_moe_config()
        model = InternVLChatModel(config).eval()
        gate_key = "language_model.model.layers.0.mlp.gate.weight"
        expected_gate = model.state_dict()[gate_key]

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            save_file(
                self._to_hf_layout_state_dict(model),
                f"{tmpdir}/model.safetensors",
                metadata={"format": "pt"},
            )
            reloaded = InternVLChatModel.from_pretrained(tmpdir, load_checkpoint_format="").eval()

        actual_gate = reloaded.state_dict()[gate_key]
        self.assertEqual(list(actual_gate.shape), list(expected_gate.shape))
        paddle.testing.assert_close(actual_gate, expected_gate, atol=0.0, rtol=0.0)

    def test_save_load(self):
        paddle.seed(42)
        model = InternVLChatModel(self.get_config()).eval()
        inputs = self.get_inputs()
        with paddle.no_grad():
            expected = model(**inputs, use_cache=False).logits
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir, save_checkpoint_format="")
            reloaded = InternVLChatModel.from_pretrained(tmpdir, load_checkpoint_format="").eval()
            with paddle.no_grad():
                actual = reloaded(**inputs, use_cache=False).logits
        paddle.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_determinism(self):
        paddle.seed(42)
        model = InternVLChatModel(self.get_config()).eval()
        inputs = self.get_inputs()
        with paddle.no_grad():
            first = model(**inputs, use_cache=False).logits
            second = model(**inputs, use_cache=False).logits
        paddle.testing.assert_close(second, first, atol=0.0, rtol=0.0)

    def test_cache_is_reused_for_generation_steps(self):
        model = InternVLChatModel(self.get_config()).eval()
        inputs = self.get_inputs()
        with paddle.no_grad():
            outputs = model(**inputs, use_cache=True)
        self.assertIsNotNone(outputs.past_key_values)
        self.assertEqual(outputs.past_key_values.get_seq_length(), inputs["input_ids"].shape[1])

        next_input_ids = paddle.concat([inputs["input_ids"], paddle.to_tensor([[13]], dtype="int64")], axis=-1)
        attention_mask = paddle.ones_like(next_input_ids)
        prepared = model.prepare_inputs_for_generation(
            next_input_ids,
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            visual_features=paddle.randn([1, 1, 16]),
            use_cache=True,
        )

        self.assertEqual(list(prepared["input_ids"].shape), [1, 1])
        self.assertIsNone(prepared["visual_features"])
        self.assertIs(prepared["past_key_values"], outputs.past_key_values)

    def test_chat_model_rejects_output_hidden_states(self):
        model = InternVLChatModel(self.get_config()).eval()

        with self.assertRaises(TypeError):
            model(**self.get_inputs(), use_cache=False, output_hidden_states=True)

    def test_vision_model_output_hidden_states(self):
        config = self.get_config()
        model = InternVisionModel(config.vision_config).eval()
        pixel_values = self.get_inputs()["pixel_values"]

        with paddle.no_grad():
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)

        self.assertIsNotNone(outputs.hidden_states)
        self.assertEqual(len(outputs.hidden_states), config.vision_config.num_hidden_layers + 1)
        self.assertEqual(list(outputs.hidden_states[0].shape), [1, 5, config.vision_config.hidden_size])
        self.assertEqual(list(outputs.hidden_states[-1].shape), [1, 5, config.vision_config.hidden_size])

    def test_resize_tokens_embeddings(self):
        model = InternVLChatModel(self.get_config()).eval()
        old_input_embeddings = model.get_input_embeddings().weight.detach().clone()
        old_output_embeddings = model.get_output_embeddings().weight.detach().clone()

        model.resize_token_embeddings(205)

        self.assertEqual(model.config.vocab_size, 205)
        self.assertEqual(model.config.llm_config.vocab_size, 205)
        self.assertEqual(list(model.get_input_embeddings().weight.shape), [205, 16])
        self.assertEqual(list(model.get_output_embeddings().weight.shape), [205, 16])
        paddle.testing.assert_close(model.get_input_embeddings().weight[:200], old_input_embeddings)
        paddle.testing.assert_close(model.get_output_embeddings().weight[:200], old_output_embeddings)

    def test_greedy_generate(self):
        model = InternVLChatModel(self.get_config()).eval()
        with paddle.no_grad():
            output_ids, _ = model.generate(**self.get_inputs(), max_new_tokens=2)
        self.assertEqual(list(output_ids.shape), [1, 2])

    def test_beam_search_generate(self):
        model = InternVLChatModel(self.get_config()).eval()
        with paddle.no_grad():
            output_ids, _ = model.generate(**self.get_inputs(), max_new_tokens=2, num_beams=2)
        self.assertEqual(list(output_ids.shape), [1, 2])

    def test_sample_generate(self):
        model = InternVLChatModel(self.get_config()).eval()
        with paddle.no_grad():
            output_ids, _ = model.generate(**self.get_inputs(), max_new_tokens=2, do_sample=True, top_k=10)
        self.assertEqual(list(output_ids.shape), [1, 2])

    def test_mismatching_image_tokens(self):
        model = InternVLChatModel(self.get_config()).eval()
        input_ids = paddle.to_tensor([[10, 151, 151, 12]], dtype="int64")
        pixel_values = paddle.randn([1, 3, 28, 28])
        with self.assertRaisesRegex(ValueError, "does not match"):
            model(input_ids=input_ids, pixel_values=pixel_values, use_cache=False)


if __name__ == "__main__":
    unittest.main()
