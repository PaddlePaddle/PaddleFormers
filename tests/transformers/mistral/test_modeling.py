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
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddleformers.nn.attention.eager_attention import eager_attention_forward
from paddleformers.nn.pp_model import EmbeddingPipe, parse_args
from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    MistralConfig,
    MistralForCausalLM,
    MistralForCausalLMPipe,
    MistralForSequenceClassification,
    MistralModel,
)
from paddleformers.transformers.mistral import modeling as mistral_modeling
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import ModelTesterMixin


def tiny_mistral_config(**kwargs):
    config_kwargs = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        sliding_window=4,
        use_sliding_window=True,
        fuse_rms_norm=False,
    )
    config_kwargs.update(kwargs)
    return MistralConfig(**config_kwargs)


class MistralModelTester:
    batch_size = 2
    seq_length = 8
    hidden_size = 16
    num_hidden_layers = 2
    num_attention_heads = 4
    is_training = False

    def get_config(self):
        return tiny_mistral_config(pad_token_id=0, use_cache=False, return_dict=True)

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        input_ids = paddle.randint(1, config.vocab_size, [self.batch_size, self.seq_length], dtype="int64")
        return config, {"input_ids": input_ids}


class MistralModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = MistralModel
    all_model_classes = (MistralModel, MistralForCausalLM)
    all_generative_model_classes = {MistralForCausalLM: (MistralModel, "mistral")}
    has_attentions = False

    def setUp(self):
        super().setUp()
        self.model_tester = MistralModelTester()
        self.config_tester = ConfigTester(self, config_class=MistralConfig, hidden_size=16, num_attention_heads=4)

    def _get_input_ids_and_config(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs["input_ids"][:, :4].clone()
        return config, input_ids, paddle.ones_like(input_ids), 3

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_sliding_attention_matches_manual_mask(self):
        settings = (
            {"use_sliding_window": False},
            {},
            {"layer_types": ["sliding_attention", "full_attention"]},
            {"layer_types": ["full_attention", "sliding_attention"]},
        )
        for overrides in settings:
            with self.subTest(overrides=overrides):
                config = tiny_mistral_config(**overrides)
                model = MistralForCausalLM(config)
                model.eval()
                input_ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int64")

                def manual_attention(module, query, key, value, **kwargs):
                    positions = paddle.arange(input_ids.shape[-1])
                    distance = positions[:, None] - positions[None, :]
                    allowed = distance >= 0
                    if config.layer_types[module.layer_idx] == "sliding_attention":
                        allowed = allowed & (distance < config.sliding_window)
                    mask = paddle.where(allowed, 0.0, paddle.finfo(query.dtype).min)[None, None, :, :]
                    return eager_attention_forward(
                        module, query, key, value, attention_mask=mask, scaling=module.scaling
                    )

                with paddle.no_grad():
                    actual = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
                    with patch.object(
                        mistral_modeling,
                        "ALL_ATTENTION_FUNCTIONS",
                        {"eager": manual_attention, "sdpa": manual_attention},
                    ):
                        expected = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-6, rtol=1e-6)

    def test_packed_row_indices_match_separate_sequences(self):
        settings = (
            {"use_sliding_window": False},
            {},
            {"layer_types": ["sliding_attention", "full_attention"]},
            {"layer_types": ["full_attention", "sliding_attention"]},
        )
        for backend in ("eager", "sdpa"):
            for overrides in settings:
                with self.subTest(backend=backend, overrides=overrides):
                    paddle.seed(23)
                    config = tiny_mistral_config(sliding_window=2, _attn_implementation=backend, **overrides)
                    model = MistralForCausalLM(config)
                    model.eval()
                    input_ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int64")
                    position_ids = paddle.to_tensor([[0, 1, 2, 0, 1, 2, 3, 4]], dtype="int64")
                    row_indices = paddle.to_tensor([3, 3, 3, 8, 8, 8, 8, 8], dtype="int32").reshape([1, 1, 8, 1])
                    with paddle.no_grad():
                        packed = model(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            attn_mask_startend_row_indices=row_indices,
                            use_cache=False,
                            return_dict=True,
                        ).logits
                        separate = [
                            model(input_ids=input_ids[:, start:end], use_cache=False, return_dict=True).logits
                            for start, end in ((0, 3), (3, 8))
                        ]
                    np.testing.assert_allclose(
                        packed.numpy(), paddle.concat(separate, axis=1).numpy(), atol=1e-5, rtol=1e-5
                    )

    def test_cached_decoding_matches_full_forward(self):
        settings = (
            {"use_sliding_window": False},
            {},
            {"layer_types": ["sliding_attention", "full_attention"]},
            {"layer_types": ["full_attention", "sliding_attention"]},
        )
        for backend in ("eager", "sdpa"):
            for overrides in settings:
                for prefix_length in (3, 4, 8):
                    for chunk_length in (1, 3):
                        for padding in (False, True):
                            with self.subTest(
                                backend=backend,
                                overrides=overrides,
                                prefix_length=prefix_length,
                                chunk_length=chunk_length,
                                padding=padding,
                            ):
                                paddle.seed(23)
                                config = tiny_mistral_config(_attn_implementation=backend, **overrides)
                                model = MistralForCausalLM(config)
                                model.eval()
                                input_ids = paddle.arange(1, prefix_length + chunk_length + 1, dtype="int64")[None, :]
                                attention_mask = paddle.ones_like(input_ids)
                                if padding:
                                    attention_mask[:, :2] = 0
                                with paddle.no_grad():
                                    full = model(
                                        input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        use_cache=False,
                                        return_dict=True,
                                    )
                                    prefix = model(
                                        input_ids=input_ids[:, :prefix_length],
                                        attention_mask=attention_mask[:, :prefix_length],
                                        use_cache=True,
                                        return_dict=True,
                                    )
                                    cached = model(
                                        input_ids=input_ids[:, prefix_length:],
                                        attention_mask=attention_mask,
                                        past_key_values=prefix.past_key_values,
                                        use_cache=True,
                                        return_dict=True,
                                    )
                                np.testing.assert_allclose(
                                    cached.logits.numpy(), full.logits[:, prefix_length:].numpy(), atol=2e-6, rtol=2e-6
                                )

    def test_lm_head_conversion_does_not_depend_on_class_name(self):
        class LanguageModel(MistralForCausalLM):
            pass

        for tied in (False, True):
            config = tiny_mistral_config(tie_word_embeddings=tied)
            for model_class in (MistralModel, MistralForSequenceClassification, MistralForCausalLM, LanguageModel):
                with self.subTest(tied=tied, model_class=model_class):
                    forward = model_class._gen_aoa_config(config)["aoa_statements"]
                    inverse = model_class._gen_inv_aoa_config(config)["aoa_statements"]
                    has_head = issubclass(model_class, MistralForCausalLM)
                    self.assertEqual(any("lm_head.weight" in item for item in forward), has_head)
                    self.assertEqual(any("lm_head.weight" in item for item in inverse), has_head and not tied)

    def test_pipeline_rotary_embeddings_match_base_model(self):
        for rope_scaling in (None, {"rope_type": "linear", "factor": 2.0}):
            with self.subTest(rope_scaling=rope_scaling):
                config = tiny_mistral_config(rope_scaling=rope_scaling)
                model = MistralModel(config)
                embedding_pipe = EmbeddingPipe(config, rotary_emb_cls=MistralForCausalLMPipe._rotary_emb_cls)
                input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
                position_ids = paddle.to_tensor([[0, 4, 63, 80]], dtype="int64")

                embeddings, _, _, position_embeddings, _ = parse_args(embedding_pipe((input_ids, None, position_ids)))
                expected_cos, expected_sin = model.rotary_emb(embeddings, position_ids)

                np.testing.assert_allclose(
                    position_embeddings[0].numpy(),
                    expected_cos.numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    position_embeddings[1].numpy(),
                    expected_sin.numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_causal_lm_forward(self):
        config = tiny_mistral_config()
        model = MistralForCausalLM(config)
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")

        outputs = model(input_ids=input_ids, return_dict=True)

        self.assertEqual(list(outputs.logits.shape), [2, 8, config.vocab_size])

    def test_causal_lm_forwards_loss_mask_to_criterion(self):
        class CaptureCriterion(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.loss_mask = None

            def forward(self, logits, labels, loss_mask=None):
                self.loss_mask = loss_mask
                return paddle.to_tensor(0.0), None

        config = tiny_mistral_config()
        model = MistralForCausalLM(config)
        criterion = CaptureCriterion()
        model.criterion = criterion
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")
        labels = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")
        loss_mask = paddle.ones([2, 8], dtype="float32")

        outputs = model(input_ids=input_ids, labels=labels, loss_mask=loss_mask, return_dict=True)

        self.assertIsNotNone(outputs.loss)
        self.assertIs(criterion.loss_mask, loss_mask)

    def test_base_model_tuple_output_includes_cache(self):
        config = tiny_mistral_config()
        model = MistralModel(config)
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")

        outputs = model(input_ids=input_ids, use_cache=True, return_dict=False)

        self.assertEqual(list(outputs[0].shape), [2, 8, config.hidden_size])
        self.assertIsNotNone(outputs[1])

    def test_sequence_classification_forward(self):
        config = tiny_mistral_config(num_labels=3, pad_token_id=0)
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")

        seq_model = MistralForSequenceClassification(config)
        seq_outputs = seq_model(input_ids=input_ids, return_dict=True)
        self.assertEqual(list(seq_outputs.logits.shape), [2, config.num_labels])

    def test_from_pretrained_and_auto_model(self):
        config = tiny_mistral_config()
        model = MistralForCausalLM(config)
        input_ids = paddle.randint(low=0, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            class_loaded = MistralForCausalLM.from_pretrained(tmpdir)
            class_outputs = class_loaded(input_ids=input_ids, return_dict=True)
            self.assertEqual(list(class_outputs.logits.shape), [2, 8, config.vocab_size])

            auto_loaded = AutoModelForCausalLM.from_pretrained(tmpdir)
            self.assertEqual(type(auto_loaded).__name__, "MistralForCausalLM")
            auto_outputs = auto_loaded(input_ids=input_ids, return_dict=True)
            self.assertEqual(list(auto_outputs.logits.shape), [2, 8, config.vocab_size])

    def test_auto_config_with_model_type(self):
        config = tiny_mistral_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            config_path = os.path.join(tmpdir, "config.json")

            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict.pop("architectures", None)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f)

            auto_config = AutoConfig.from_pretrained(tmpdir)

            self.assertEqual(type(auto_config).__name__, "MistralConfig")
            self.assertEqual(auto_config.model_type, "mistral")

    def test_auto_model_base_dispatch(self):
        config = tiny_mistral_config()
        model = MistralForCausalLM(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = AutoModel.from_pretrained(tmpdir)
            self.assertIsInstance(loaded, MistralModel)

    def test_auto_model_sequence_classification_dispatch(self):
        config = tiny_mistral_config(num_labels=3, pad_token_id=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            MistralForSequenceClassification(config).save_pretrained(tmpdir)
            seq_loaded = AutoModelForSequenceClassification.from_pretrained(tmpdir)
            self.assertIsInstance(seq_loaded, MistralForSequenceClassification)


if __name__ == "__main__":
    unittest.main()
