# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import paddle
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.transformers.fastvlm.configuration import FastVLMConfig
from paddleformers.transformers.fastvlm.image_processor import FastVLMImageProcessor
from paddleformers.transformers.fastvlm.modeling import FastVLMForConditionalGeneration
from paddleformers.transformers.fastvlm.processor import FastVLMProcessor


class FastVLMModelTest(unittest.TestCase):
    def setUp(self):
        self.config = FastVLMConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            mm_vision_tower="mobileclip_l_64",
            mm_hidden_size=128,
            vision_config={"layers": [1, 1, 1, 1], "embed_dims": [8, 16, 32, 64]},
            tie_word_embeddings=False,
            use_cache=False,
            architectures=["LlavaQwen2ForCausalLM"],
        )

    def get_inputs(self):
        input_ids = paddle.to_tensor([[1, -200, 2, 3]], dtype="int64")
        labels = paddle.to_tensor([[1, -100, 2, 3]], dtype="int64")
        pixel_values = paddle.randn([1, 3, 64, 64])
        return input_ids, labels, pixel_values

    def test_multimodal_forward_and_loss(self):
        model = FastVLMForConditionalGeneration(self.config)
        input_ids, labels, pixel_values = self.get_inputs()
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            return_dict=True,
        )
        self.assertEqual(outputs.logits.shape, [1, 7, self.config.vocab_size])
        self.assertIsNotNone(outputs.loss)

    def test_backward(self):
        model = FastVLMForConditionalGeneration(self.config)
        input_ids, labels, pixel_values = self.get_inputs()
        loss = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            return_dict=True,
        ).loss
        loss.backward()
        self.assertIsNotNone(model.model.mm_projector[0].weight.grad)

    def test_mixed_batch_uses_image_order_not_batch_index(self):
        model = FastVLMForConditionalGeneration(self.config).eval()
        image_ids, image_labels, pixels = self.get_inputs()
        text_ids = paddle.to_tensor([[1, 4, 2, 3]], dtype="int64")
        with paddle.no_grad():
            reference = model(input_ids=image_ids, pixel_values=pixels, return_dict=True).logits
            for image_first in (False, True):
                rows = [image_ids, text_ids] if image_first else [text_ids, image_ids]
                ids = paddle.concat(rows, axis=0)
                labels = paddle.concat([image_labels, text_ids] if image_first else [text_ids, image_labels], axis=0)
                prepared = model._prepare_multimodal_inputs(ids, None, None, labels, pixels)
                image_row = 0 if image_first else 1
                self.assertIsNotNone(prepared[1])
                self.assertEqual(int(prepared[1][1 - image_row].sum()), 4)
                outputs = model(input_ids=ids, labels=labels, pixel_values=pixels, return_dict=True)
                np.testing.assert_allclose(
                    outputs.logits[image_row].numpy(), reference[0].numpy(), atol=1e-5, rtol=1e-5
                )

    def test_image_count_mismatch_is_rejected(self):
        model = FastVLMForConditionalGeneration(self.config).eval()
        ids, labels, pixels = self.get_inputs()
        with self.assertRaisesRegex(ValueError, "more images"):
            model._prepare_multimodal_inputs(ids, None, None, labels, paddle.concat([pixels, pixels]))
        with self.assertRaisesRegex(ValueError, "fewer images"):
            model._prepare_multimodal_inputs(paddle.concat([ids, ids]), None, None, None, pixels)

    def test_plugin_preserves_existing_label_mask(self):
        from paddleformers.datasets.template.mm_plugin import FastVLMPlugin

        plugin = FastVLMPlugin(image_token="<image>", video_token=None, audio_token=None)
        labels = [-100, 99, 2, 3]
        actual = plugin.process_tokens([1, -200, 2, 3], None, labels=labels)
        self.assertEqual(actual, [-100, -100, 2, 3])
        self.assertEqual(labels, [-100, 99, 2, 3])

    @unittest.skipUnless(paddle.device.cuda.device_count() > 0, "CINN regression requires a CUDA device")
    def test_se_block_cinn_matches_dynamic(self):
        from paddleformers.transformers.fastvlm.modeling_vision import SEBlock

        self.addCleanup(paddle.set_device, paddle.get_device())
        paddle.set_device("gpu:0")
        dynamic = SEBlock(16)
        compiled = SEBlock(16)
        compiled.set_state_dict(dynamic.state_dict())
        compiled = paddle.jit.to_static(compiled, full_graph=True, backend="CINN")
        inputs = paddle.randn([2, 16, 4, 4])
        expected = dynamic(inputs)
        actual = compiled(inputs)
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for expected_param, actual_param in zip(dynamic.parameters(), compiled.parameters()):
            self.assertIsNotNone(actual_param.grad)
            np.testing.assert_allclose(actual_param.grad.numpy(), expected_param.grad.numpy(), atol=1e-5, rtol=1e-5)

    def test_auto_model_from_config(self):
        model = AutoModelForConditionalGeneration.from_config(self.config)
        self.assertIsInstance(model, FastVLMForConditionalGeneration)

    def test_image_processor(self):
        outputs = FastVLMImageProcessor(image_size=64)(Image.new("RGB", (80, 60), (20, 30, 40)), return_tensors="pd")
        self.assertEqual(list(outputs.pixel_values.shape), [1, 3, 64, 64])

    def test_official_model_id_uses_fastvlm_conversion(self):
        reference = FastVLMForConditionalGeneration(self.config)
        converted_state = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
        with (
            mock.patch(
                "paddleformers.transformers.fastvlm.modeling._resolve_hf_checkpoint_dir",
                return_value="/tmp/fastvlm-hf-checkpoint",
            ),
            mock.patch("paddleformers.transformers.fastvlm.modeling._is_hf_fastvlm_checkpoint", return_value=True),
            mock.patch(
                "paddleformers.transformers.fastvlm.modeling.FastVLMConfig.from_pretrained",
                return_value=(self.config, {}),
            ),
            mock.patch(
                "paddleformers.transformers.fastvlm.modeling._load_hf_fastvlm_state_dict",
                return_value=converted_state,
            ) as load_checkpoint,
        ):
            loaded = FastVLMForConditionalGeneration.from_pretrained("apple/FastVLM-0.5B")
        load_checkpoint.assert_called_once_with("/tmp/fastvlm-hf-checkpoint", self.config)
        for name, tensor in reference.state_dict().items():
            np.testing.assert_array_equal(tensor.numpy(), loaded.state_dict()[name].numpy())

    def test_processor_official_model_id_without_preprocessor_config(self):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1}, unk_token="<unk>")),
            unk_token="<unk>",
            eos_token="<eos>",
        )
        with (
            mock.patch("paddleformers.transformers.auto.processing.resolve_file_path", return_value=None),
            mock.patch(
                "paddleformers.transformers.auto.processing.AutoConfig.from_pretrained", return_value=self.config
            ),
            mock.patch.object(FastVLMProcessor, "get_processor_dict", return_value=({}, {})),
            mock.patch(
                "paddleformers.transformers.fastvlm.processor.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ),
            mock.patch(
                "paddleformers.transformers.fastvlm.processor.FastVLMImageProcessor.from_pretrained",
                side_effect=TypeError("preprocessor_config.json resolved to None"),
            ),
            mock.patch(
                "paddleformers.transformers.fastvlm.processor.FastVLMConfig.from_pretrained",
                return_value=self.config,
            ),
        ):
            processor = AutoProcessor.from_pretrained("apple/FastVLM-0.5B")
        self.assertIsInstance(processor, FastVLMProcessor)
        self.assertIs(processor.tokenizer, tokenizer)
        self.assertEqual(processor.image_processor.crop_size, {"height": 64, "width": 64})

    def test_default_save_pretrained_round_trip(self):
        model = FastVLMForConditionalGeneration(self.config)
        with tempfile.TemporaryDirectory() as save_dir:
            model.save_pretrained(save_dir)
            self.assertTrue(os.path.exists(os.path.join(save_dir, "model_state.pdparams")))
            loaded = FastVLMForConditionalGeneration.from_pretrained(save_dir)
        for name, tensor in model.state_dict().items():
            np.testing.assert_array_equal(tensor.numpy(), loaded.state_dict()[name].numpy())

    def test_tensor_parallel_fused_parameter_splits(self):
        config = FastVLMConfig(**self.config.to_dict())
        config.tensor_model_parallel_size = 2
        qkv = np.arange(self.config.hidden_size * 64, dtype="float32").reshape(self.config.hidden_size, 64)
        ffn = np.arange(self.config.hidden_size * 128, dtype="float32").reshape(self.config.hidden_size, 128)
        shards = []
        ffn_shards = []
        for rank in range(2):
            config.tensor_parallel_rank = rank
            actions = FastVLMForConditionalGeneration._get_tensor_parallel_mappings(config)
            shards.append(actions["layers.0.self_attn.qkv_proj.weight"](qkv))
            ffn_shards.append(actions["layers.0.mlp.up_gate_proj.weight"](ffn))
        np.testing.assert_array_equal(np.concatenate(shards, axis=-1), qkv)
        np.testing.assert_array_equal(
            np.concatenate([item[:, :32] for item in ffn_shards] + [item[:, 32:] for item in ffn_shards], axis=-1),
            ffn,
        )
        hidden_states = np.arange(3 * self.config.hidden_size, dtype="float32").reshape(3, self.config.hidden_size)
        single_card_output = hidden_states @ qkv
        tensor_parallel_output = np.concatenate([hidden_states @ shard for shard in shards], axis=-1)
        np.testing.assert_array_equal(tensor_parallel_output, single_card_output)


if __name__ == "__main__":
    unittest.main()
