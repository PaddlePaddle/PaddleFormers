# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The MiniCPM team and The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.nn.pp_model import (
    EmbeddingPipe,
    RMSNormPipe,
    make_decoder_layer_pipe,
)
from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    MiniCPMConfig,
    MiniCPMForCausalLM,
    MiniCPMForSequenceClassification,
    MiniCPMModel,
)
from paddleformers.transformers.minicpm.modeling import (
    MiniCPMForCausalLMPipe,
    MiniCPMLMHeadPipe,
    MiniCPMRotaryEmbedding,
    _create_scaled_embedding,
    apply_rotary_pos_emb,
)
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    GenerationD2STestMixin,
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class MiniCPMModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        scope=None,
    ):
        self.parent: MiniCPMModelTest = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self) -> MiniCPMConfig:
        return MiniCPMConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            attention_bias=False,
            scale_emb=12,
            dim_model_base=16,
            scale_depth=1.4,
        )

    def create_and_check_model(
        self, config: MiniCPMConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MiniCPMModel(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(self, config: MiniCPMConfig, input_ids):
        model = MiniCPMModel(config)
        model.eval()
        attn_mask_2d = random_attention_mask([self.batch_size, self.seq_length])
        result_2d = model(input_ids, attention_mask=attn_mask_2d)[0]
        batch, seq_length = input_ids.shape
        causal_mask = paddle.tril(paddle.ones((batch, seq_length, seq_length), dtype=attn_mask_2d.dtype))
        attn_mask_3d = causal_mask & attn_mask_2d.unsqueeze(-1)
        result_3d = model(input_ids, attention_mask=attn_mask_3d)[0]
        attn_mask_4d = attn_mask_3d.unsqueeze(1)
        result_4d = model(input_ids, attention_mask=attn_mask_4d)[0]
        result_no_attention_mask = model(input_ids, attention_mask=None)[0]
        # Assert non-padding tokens have the same logits with different attention_mask shape
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_3d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_4d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_no_attention_mask[attn_mask_2d]).all())

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.add_cross_attention = True
        model = MiniCPMModel(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = MiniCPMForCausalLM(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, *args):
        model = MiniCPMForCausalLM(config)
        model.eval()

        result = model(
            input_ids,
            use_cache=True,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def check_model_position_ids(self, config, input_ids, input_mask, *args):
        model = MiniCPMForCausalLM(config)
        model.eval()

        result_no_position_id = model(
            input_ids,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        batch_size, seq_len = input_ids.shape
        position_ids = paddle.arange(seq_len).expand((batch_size, seq_len))
        result_position_id = model(
            input_ids,
            position_ids=position_ids,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertTrue((result_position_id[1] == result_no_position_id[1]).all())
        else:
            self.parent.assertTrue((result_position_id[0] == result_no_position_id[0]).all())

    def create_and_check_cached_forward(self, config, input_ids, *args):
        model = MiniCPMForCausalLM(config)
        model.eval()

        full_logits = model(input_ids, use_cache=True, return_dict=True).logits[:, -1, :]
        prefix_outputs = model(input_ids[:, :-1], use_cache=True, return_dict=True)
        cached_logits = model(
            input_ids[:, -1:],
            past_key_values=prefix_outputs.past_key_values,
            use_cache=True,
            return_dict=True,
        ).logits[:, -1, :]

        self.parent.assertTrue(paddle.allclose(cached_logits, full_logits, atol=1e-5, rtol=1e-5))


class MiniCPMModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = MiniCPMModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (MiniCPMModel, MiniCPMForCausalLM)
    all_generative_model_classes = {MiniCPMForCausalLM: {MiniCPMModel, "minicpm"}}
    pipeline_model_mapping = {
        "feature-extraction": MiniCPMModel,
        "text-classification": MiniCPMForSequenceClassification,
        "text-generation": MiniCPMForCausalLM,
        "zero-shot": MiniCPMForSequenceClassification,
    }

    @gpu_device_initializer(log_prefix="MiniCPMModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = MiniCPMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MiniCPMConfig, hidden_size=37)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_attention_mask(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model_attention_mask(config, input_dict["input_ids"])

    def test_model_position_ids(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)

    def test_cached_forward_matches_full_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_cached_forward(*config_and_inputs)

    def test_model_decoder_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

    def test_model_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_model_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_MiniCPM_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = MiniCPMForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])

    def test_MiniCPM_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = MiniCPMForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])

    def test_MiniCPM_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(paddle.float32)
        model = MiniCPMForSequenceClassification(config)

        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])

    @unittest.skip("MiniCPM buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip("MiniCPM uses GQA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass

    def test_minicpm_scaling_config(self):
        config = self.model_tester.get_config()
        model = MiniCPMForCausalLM(config)
        self.assertEqual(model.model.config.scale_emb, 12)
        self.assertEqual(model.model.config.scale_depth, 1.4)
        self.assertEqual(model.model.config.dim_model_base, 16)


@unittest.skip("MiniCPM D2S generation uses remote tiny checkpoint that will be added with CE assets.")
class MiniCPMGenerationD2STest(GenerationD2STestMixin, unittest.TestCase):
    internal_testing_model = "PaddleFormers/tiny-random-minicpm"


class MiniCPMCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(paddle.set_device, paddle.get_device())
        if paddle.device.cuda.device_count():
            paddle.set_device("gpu:0")
        self.config = MiniCPMConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            scale_emb=3.0,
            dim_model_base=8,
            scale_depth=1.4,
            _attn_implementation="eager",
            fuse_rms_norm=False,
        )

    def test_pp_component_chain_matches_forward_and_gradients(self):
        self._check_pp_component_chain(use_attention_mask=True)

    def test_pp_without_mask_matches_forward_and_gradients(self):
        self._check_pp_component_chain(use_attention_mask=False)

    def _check_pp_component_chain(self, use_attention_mask):
        model = MiniCPMForCausalLM(self.config)
        model.eval()
        pipe_cls = MiniCPMForCausalLMPipe
        embedding = EmbeddingPipe(self.config, pipe_cls._embed_cls, pipe_cls._rotary_emb_cls)
        embedding.embed_tokens.set_state_dict(model.model.embed_tokens.state_dict())
        decoder_cls = make_decoder_layer_pipe(pipe_cls._decoder_layer_cls)
        layers = [decoder_cls(self.config, layer_idx=i) for i in range(self.config.num_hidden_layers)]
        for layer, reference in zip(layers, model.model.layers):
            layer.set_state_dict(reference.state_dict())
            layer.eval()
        norm = RMSNormPipe(self.config)
        norm.set_state_dict(model.model.norm.state_dict())
        head = pipe_cls._lmhead_pipe_cls(self.config)
        head.set_state_dict(model.lm_head.state_dict())
        if self.config.tie_word_embeddings:
            head.weight = embedding.embed_tokens.weight
        ids = paddle.to_tensor([[1, 2, 3]])
        expected = model(ids)[0]
        mask = paddle.tril(paddle.ones([1, 1, 3, 3], dtype="float32"))
        actual = embedding((ids, mask) if use_attention_mask else ids)
        for layer in layers:
            actual = layer(actual)
        actual = head(norm(actual))
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-7)
        actual.square().mean().backward()
        expected.square().mean().backward()
        np.testing.assert_allclose(
            embedding.embed_tokens.weight.grad.numpy(),
            model.model.embed_tokens.weight.grad.numpy(),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_safetensors_flex_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            model = MiniCPMForCausalLM(self.config)
            model.eval()
            ids = paddle.to_tensor([[1, 2, 3]])
            expected = model(ids)[0].numpy()
            model.save_pretrained(directory, safe_serialization=True)
            restored = MiniCPMForCausalLM.from_pretrained(directory, load_checkpoint_format="flex_checkpoint")
            restored.eval()
            self.assertEqual(set(model.state_dict()), set(restored.state_dict()))
            for name, value in model.state_dict().items():
                np.testing.assert_array_equal(value.numpy(), restored.state_dict()[name].numpy(), err_msg=name)
            np.testing.assert_array_equal(expected, restored(ids)[0].numpy())

    def test_hf_defaults(self):
        config = MiniCPMConfig()
        self.assertEqual((config.vocab_size, config.hidden_size, config.num_hidden_layers), (32000, 4096, 32))
        self.assertEqual((config.bos_token_id, config.eos_token_id, config.pad_token_id), (1, 2, None))
        self.assertEqual(config.num_key_value_heads, config.num_attention_heads)
        self.assertEqual((config.scale_emb, config.dim_model_base, config.scale_depth), (1, 1, 1))
        self.assertFalse(hasattr(config, "sliding_window"))

    def _get_longrope_config(self):
        config = self.config.to_dict()
        config.update(
            _name_or_path="openbmb/MiniCPM4.1-8B",
            architectures=["MiniCPMForCausalLM"],
            max_position_embeddings=65536,
            scale_emb=12,
            scale_depth=1.4,
            dim_model_base=8,
            tie_word_embeddings=False,
            rope_scaling={
                "rope_type": "longrope",
                "short_factor": [1.0, 2.0, 4.0, 8.0],
                "long_factor": [1.0, 2.0, 4.0, 8.0],
                "original_max_position_embeddings": 65536,
            },
        )
        config.pop("rope_parameters", None)
        return MiniCPMConfig.from_dict(config)

    def test_minicpm4_1_auto_loading_preserves_original_model_type(self):
        config = self._get_longrope_config()
        ids = paddle.to_tensor([[1, 2, 3]])
        model = AutoModelForCausalLM.from_config(config, dtype="float32")
        self.assertIs(type(model), MiniCPMForCausalLM)
        self.assertIs(type(AutoModel.from_config(config, dtype="float32")), MiniCPMModel)
        self.assertIs(
            AutoModelForCausalLMPipe._get_model_class_from_config(None, None, config), MiniCPMForCausalLMPipe
        )
        model.eval()
        expected = model(ids)[0].numpy()
        with tempfile.TemporaryDirectory() as root:
            directory = os.path.join(root, "MiniCPM4.1-8B")
            model.save_pretrained(directory, safe_serialization=True)
            loaded_config = AutoConfig.from_pretrained(directory)
            self.assertIs(type(loaded_config), MiniCPMConfig)
            self.assertEqual(loaded_config.model_type, "minicpm")
            restored = AutoModelForCausalLM.from_pretrained(directory, load_checkpoint_format="flex_checkpoint")
            restored.eval()
            self.assertIs(type(restored), MiniCPMForCausalLM)
            np.testing.assert_array_equal(expected, restored(ids)[0].numpy())
            with open(os.path.join(directory, "config.json")) as handle:
                saved = json.load(handle)
            self.assertEqual(saved["model_type"], "minicpm")
            self.assertEqual(saved["architectures"], ["MiniCPMForCausalLM"])
            self.assertEqual(saved["rope_scaling"]["short_factor"], config.rope_scaling["short_factor"])

    def test_longrope_cached_forward_matches_full_forward(self):
        model = MiniCPMForCausalLM(self._get_longrope_config())
        model.eval()
        ids = paddle.to_tensor([[1, 2, 3, 4]])
        positions = paddle.to_tensor([[65000, 65001, 65002, 65003]])
        full = model(ids, position_ids=positions, use_cache=False)[0]
        prefix = model(ids[:, :3], position_ids=positions[:, :3], use_cache=True, return_dict=True)
        cached = model(
            ids[:, 3:], position_ids=positions[:, 3:], past_key_values=prefix.past_key_values, use_cache=True
        )[0]
        np.testing.assert_allclose(cached.numpy(), full[:, 3:].numpy(), rtol=1e-5, atol=1e-6)
        generated = model.generate(ids, max_new_tokens=3, decode_strategy="greedy_search")[0]
        self.assertEqual(generated.shape[-1], 3)

    def test_longrope_keeps_bfloat16_rotation_in_float32(self):
        config = self._get_longrope_config()
        rotary = MiniCPMRotaryEmbedding(config)
        positions = paddle.to_tensor([[0, 65000]])
        query = paddle.randn([1, 2, 2, 8]).astype("bfloat16")
        key = paddle.randn([1, 1, 2, 8]).astype("bfloat16")
        cos, sin = rotary(query, positions)
        self.assertEqual(cos.dtype, paddle.float32)
        inv_freq = 1.0 / (
            paddle.to_tensor(config.rope_scaling["short_factor"])
            * config.rope_theta ** (paddle.arange(0, 8, 2).astype("float32") / 8)
        )
        freqs = positions.astype("float32").unsqueeze(-1) * inv_freq
        angles = paddle.concat([freqs, freqs], axis=-1)
        np.testing.assert_allclose(cos.numpy(), angles.cos().numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(sin.numpy(), angles.sin().numpy(), rtol=1e-6, atol=1e-6)
        actual = apply_rotary_pos_emb(query, key, cos, sin)
        for original, rotated in zip((query, key), actual):
            value = original.astype("float32")
            half = paddle.concat([-value[..., 4:], value[..., :4]], axis=-1)
            expected = (value * cos.unsqueeze(1) + half * sin.unsqueeze(1)).astype("bfloat16")
            self.assertEqual(rotated.dtype, paddle.bfloat16)
            np.testing.assert_array_equal(rotated.astype("float32").numpy(), expected.astype("float32").numpy())

    def test_longrope_selects_frequencies_without_mutating_buffers(self):
        config = self._get_longrope_config()
        config.rope_parameters["long_factor"] = [2.0, 4.0, 8.0, 16.0]
        rotary = MiniCPMRotaryEmbedding(config)
        static_rotary = paddle.jit.to_static(MiniCPMRotaryEmbedding(config), full_graph=True)
        original = rotary.inv_freq.numpy().copy()
        for position, factors in (
            (65536, config.rope_parameters["long_factor"]),
            (7, config.rope_parameters["short_factor"]),
            (65537, config.rope_parameters["long_factor"]),
        ):
            positions = paddle.to_tensor([[position]])
            cos, sin = rotary(paddle.zeros([1, 1, 16]), positions)
            inv_freq = 1.0 / (
                paddle.to_tensor(factors) * config.rope_theta ** (paddle.arange(0, 8, 2).astype("float32") / 8)
            )
            angles = paddle.concat([position * inv_freq, position * inv_freq])[None, None]
            np.testing.assert_allclose(cos.numpy(), angles.cos().numpy(), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(sin.numpy(), angles.sin().numpy(), rtol=1e-6, atol=1e-6)
            static_cos, static_sin = static_rotary(paddle.zeros([1, 1, 16]), positions)
            np.testing.assert_allclose(static_cos.numpy(), cos.numpy(), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(static_sin.numpy(), sin.numpy(), rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(rotary.inv_freq.numpy(), original)

    def test_longrope_config_normalization_preserves_caller_and_saved_parameters(self):
        config = self._get_longrope_config()
        self.assertNotIn("factor", config.rope_scaling)
        self.assertEqual(config.rope_parameters["factor"], 1.0)
        restored = MiniCPMConfig.from_dict(config.to_dict())
        self.assertEqual(restored.rope_scaling, config.rope_scaling)
        self.assertEqual(restored.rope_parameters, config.rope_parameters)
        self.assertEqual(restored.model_type, "minicpm")

    def test_pp_scaling_components(self):
        embedding = _create_scaled_embedding(self.config)
        ids = paddle.to_tensor([[1, 2, 3]])
        expected = paddle.nn.functional.embedding(ids, embedding.weight) * self.config.scale_emb
        np.testing.assert_allclose(embedding(ids).numpy(), expected.numpy())
        head = MiniCPMLMHeadPipe(self.config)
        hidden = paddle.randn([1, 3, self.config.hidden_size])
        expected = paddle.matmul(hidden / 2, head.weight, transpose_y=True)
        np.testing.assert_allclose(head(hidden).numpy(), expected.numpy())

    def test_forward_backward(self):
        model = MiniCPMForCausalLM(self.config)
        ids = paddle.to_tensor([[1, 2, 3]])
        result = model(ids, labels=ids, return_dict=True)
        self.assertEqual(list(result.logits.shape), [1, 3, 32])
        result.loss.backward()
        self.assertIsNotNone(model.model.embed_tokens.weight.grad)

    def test_loss_mask_excludes_unselected_tokens(self):
        model = MiniCPMForCausalLM(self.config)
        model.eval()
        ids = paddle.to_tensor([[1, 2, 3]])
        labels = paddle.to_tensor([[4, 5, 6]])
        mask = paddle.to_tensor([[0.0, 1.0, 0.0]])
        result = model(ids, labels=labels, loss_mask=mask, return_dict=True)
        expected = paddle.nn.functional.cross_entropy(result.logits[:, 1], labels[:, 1])
        np.testing.assert_allclose(result.loss.numpy(), expected.numpy(), rtol=1e-6)
        result.loss.backward()
        self.assertIsNotNone(model.model.embed_tokens.weight.grad)

    def test_classification_pools_last_token_without_padding_information(self):
        ids = paddle.to_tensor([[1, 2, 3]])
        for pad_token_id in (None, 0):
            self.config.pad_token_id = pad_token_id
            model = MiniCPMForSequenceClassification(self.config)
            model.eval()
            expected = model.score(model.model(ids)[0])[:, -1]
            with self.subTest(pad_token_id=pad_token_id, input="ids"):
                actual = model(input_ids=ids, return_dict=True).logits
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6)
            with self.subTest(pad_token_id=pad_token_id, input="embeddings"):
                embeddings = model.model.embed_tokens(ids) * self.config.scale_emb
                actual = model(inputs_embeds=embeddings, return_dict=True).logits
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6)
        self.config.pad_token_id = None
        model = MiniCPMForSequenceClassification(self.config)
        with self.assertRaisesRegex(ValueError, "batch sizes > 1"):
            model(ids.tile([2, 1]))

    def test_classification_pools_before_padding(self):
        self.config.pad_token_id = 0
        model = MiniCPMForSequenceClassification(self.config)
        model.eval()
        ids = paddle.to_tensor([[1, 2, 0], [1, 2, 3]])
        logits = model.score(model.model(ids)[0])
        expected = paddle.stack([logits[0, 1], logits[1, 2]])
        np.testing.assert_allclose(model(ids)[0].numpy(), expected.numpy(), rtol=1e-6)

    def test_invalid_attention_head_ratio_is_rejected(self):
        self.config.num_attention_heads = 4
        self.config.num_key_value_heads = 3
        with self.assertRaisesRegex(ValueError, "num_attention_heads.*divisible"):
            MiniCPMModel(self.config)
