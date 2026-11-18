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

import unittest

import paddle

from paddleformers.transformers import (
    MiniCPMConfig,
    MiniCPMForCausalLM,
    MiniCPMForSequenceClassification,
    MiniCPMModel,
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


class MiniCPMReviewRegressionTest(unittest.TestCase):
    def test_pp_component_chain_matches_forward_and_gradients(self):
        import numpy as np

        from paddleformers.nn.pp_model import (
            EmbeddingPipe,
            RMSNormPipe,
            make_decoder_layer_pipe,
        )
        from paddleformers.transformers.minicpm.modeling import MiniCPMForCausalLMPipe

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
        actual = embedding((ids, mask))
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
        import tempfile

        import numpy as np

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

    def setUp(self):
        paddle.set_device("cpu")
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

    def test_hf_defaults(self):
        config = MiniCPMConfig()
        self.assertEqual((config.vocab_size, config.hidden_size, config.num_hidden_layers), (32000, 4096, 32))
        self.assertEqual((config.bos_token_id, config.eos_token_id, config.pad_token_id), (1, 2, None))
        self.assertEqual(config.num_key_value_heads, config.num_attention_heads)
        self.assertEqual((config.scale_emb, config.dim_model_base, config.scale_depth), (1, 1, 1))
        self.assertFalse(hasattr(config, "sliding_window"))

    def test_pp_scaling_components(self):
        import numpy as np

        from paddleformers.transformers.minicpm.modeling import (
            MiniCPMLMHeadPipe,
            _create_scaled_embedding,
        )

        embedding = _create_scaled_embedding(self.config)
        ids = paddle.to_tensor([[1, 2, 3]])
        expected = paddle.nn.functional.embedding(ids, embedding.weight) * self.config.scale_emb
        np.testing.assert_allclose(embedding(ids).numpy(), expected.numpy())
        head = MiniCPMLMHeadPipe(self.config)
        hidden = paddle.randn([1, 3, self.config.hidden_size])
        expected = paddle.matmul(hidden / 2, head.weight, transpose_y=True)
        np.testing.assert_allclose(head(hidden).numpy(), expected.numpy())

    def test_cpu_forward_backward(self):
        model = MiniCPMForCausalLM(self.config)
        ids = paddle.to_tensor([[1, 2, 3]])
        result = model(ids, labels=ids, return_dict=True)
        self.assertEqual(list(result.logits.shape), [1, 3, 32])
        result.loss.backward()
        self.assertIsNotNone(model.model.embed_tokens.weight.grad)
