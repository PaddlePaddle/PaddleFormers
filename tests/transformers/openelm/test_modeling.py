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


import os
import shutil
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM, OpenELMModel
from tests.testing_utils import gpu_device_initializer, require_package, slow
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import ModelTesterMixin, ids_tensor

_MODEL_HF_ID = "apple/OpenELM-1_1B-Instruct"
_TOKENIZER_ID = "PaddleNLP/Llama-2-7b"
_PROMPT_DIFF = "Hello, how are you today?"
_PROMPT_QUESTION = "What is the capital of China?"
_SEED = 42
_NUM_GEN_TOKENS = 10
_DIFF_THRESHOLD = 1e-2
SMALL_CONFIG = {
    "vocab_size": 1000,
    "max_context_length": 128,
    "num_transformer_layers": 2,
    "model_dim": 64,
    "head_dim": 16,
    "qkv_multipliers": 1.0,
    "num_gqa_groups": 1,
    "ffn_multipliers": 2.0,
    "ffn_with_glu": True,
    "ffn_dim_divisor": 64,
    "activation_fn_name": "swish",
    "normalize_qk_projections": False,
    "share_input_output_layers": True,
    "rope_freq_constant": 10000,
    "rope_max_length": 256,
    "initializer_range": 0.02,
    "use_cache": True,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


class OpenELMModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.seq_length = 10
        self.hidden_size = SMALL_CONFIG["model_dim"]
        self.num_hidden_layers = SMALL_CONFIG["num_transformer_layers"]
        self.num_attention_heads = self.hidden_size // SMALL_CONFIG["head_dim"]
        self.is_training = False

    def get_config(self):
        return OpenELMConfig(**dict(SMALL_CONFIG, use_cache=False))

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size, dtype=paddle.int64)
        return config, input_ids

    def prepare_config_and_inputs_for_common(self):
        config, input_ids = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids}


class OpenELMModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = OpenELMModel
    all_model_classes = (OpenELMModel, OpenELMForCausalLM)
    all_generative_model_classes = {OpenELMForCausalLM: (OpenELMModel, "openelm")}
    use_test_inputs_embeds = True
    test_tie_weights = True
    BATCH = 2
    SEQ = 10

    @gpu_device_initializer(log_prefix="OpenELMModelTest")
    def setUp(self):
        self.model_tester = OpenELMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OpenELMConfig, common_properties=["vocab_size"])

        self.config = OpenELMConfig(**SMALL_CONFIG)
        self.model = OpenELMModel(self.config)
        self.model.eval()

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_training(self):
        config, ids = self.model_tester.prepare_config_and_inputs()
        model = OpenELMForCausalLM(config)
        model.train()
        optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
        before = model.model.layers[0].self_attn.q_proj.weight.clone()
        loss = model(ids, labels=ids, use_cache=False, return_dict=True).loss
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(paddle.isfinite(parameter.grad).all().item(), name)
        optimizer.step()
        self.assertFalse(paddle.equal_all(before, model.model.layers[0].self_attn.q_proj.weight).item())

    def test_training_gradient_checkpointing(self):
        from unittest.mock import patch

        from paddle.distributed.fleet.utils import recompute

        for frozen_embeddings in (False, True):
            for reentrant in (False, True):
                config, ids = self.model_tester.prepare_config_and_inputs()
                reference = OpenELMForCausalLM(config)
                reference.train()
                reference.model.embed_tokens.weight.stop_gradient = frozen_embeddings
                model = OpenELMForCausalLM(self.model_tester.get_config())
                model.set_state_dict(reference.state_dict())
                model.model.embed_tokens.weight.stop_gradient = frozen_embeddings
                model.train()
                model.config.recompute_granularity = "full"
                model.config.recompute_method = "uniform"
                model.config.recompute_num_layers = 1
                model.config.recompute_use_reentrant = reentrant
                calls = []
                hook = model.model.layers[0].register_forward_post_hook(lambda *args: calls.append(1))
                expected = reference(ids, labels=ids, use_cache=False, return_dict=True)
                expected.loss.backward()
                with patch("paddle.distributed.fleet.utils.recompute", wraps=recompute) as checkpoint:
                    actual = model(ids, labels=ids, use_cache=False, return_dict=True)
                    actual.loss.backward()
                    self.assertEqual(checkpoint.call_count, config.num_hidden_layers)
                hook.remove()
                self.assertGreaterEqual(len(calls), 2)
                np.testing.assert_allclose(actual.loss.numpy(), expected.loss.numpy(), atol=1e-5, rtol=1e-5)
                expected_parameters = dict(reference.named_parameters())
                for name, parameter in model.named_parameters():
                    if parameter.stop_gradient:
                        continue
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(paddle.isfinite(parameter.grad).all().item(), name)
                    np.testing.assert_allclose(
                        parameter.grad.numpy(), expected_parameters[name].grad.numpy(), atol=1e-5, rtol=1e-5
                    )

    def test_model_from_config(self):
        from paddleformers.transformers import OpenELMConfig, OpenELMModel

        config = OpenELMConfig(**SMALL_CONFIG)
        model = OpenELMModel(config)
        self.assertIsNotNone(model)

        self.assertEqual(config.vocab_size, SMALL_CONFIG["vocab_size"])
        self.assertEqual(config.model_dim, SMALL_CONFIG["model_dim"])
        self.assertEqual(config.num_transformer_layers, SMALL_CONFIG["num_transformer_layers"])
        self.assertEqual(config.head_dim, SMALL_CONFIG["head_dim"])

        self.assertEqual(config.num_hidden_layers, config.num_transformer_layers)

        self.assertIsInstance(config.num_query_heads, list)
        self.assertIsInstance(config.num_kv_heads, list)
        self.assertEqual(len(config.num_query_heads), SMALL_CONFIG["num_transformer_layers"])
        self.assertEqual(len(config.num_kv_heads), SMALL_CONFIG["num_transformer_layers"])

    def test_config_standard_aliases(self):
        from paddleformers.transformers import OpenELMConfig

        config = OpenELMConfig(**dict(SMALL_CONFIG, qkv_multipliers=(0.5, 1.0)))
        self.assertEqual(config.hidden_size, config.model_dim)
        self.assertEqual(config.num_hidden_layers, config.num_transformer_layers)
        self.assertEqual(config.num_attention_heads, config.num_query_heads)
        self.assertEqual(config.num_key_value_heads, config.num_kv_heads)
        self.assertEqual(config.max_position_embeddings, config.max_context_length)
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            loaded = OpenELMConfig.from_pretrained(directory)
        self.assertEqual(loaded.num_attention_heads, config.num_query_heads)
        self.assertEqual(loaded.num_key_value_heads, config.num_kv_heads)
        self.assertEqual(loaded.hidden_size, config.model_dim)
        standard = OpenELMConfig(hidden_size=64, num_hidden_layers=2, max_position_embeddings=128, head_dim=16)
        self.assertEqual(standard.model_dim, 64)
        self.assertEqual(standard.num_transformer_layers, 2)
        self.assertEqual(standard.max_context_length, 128)
        self.assertTrue(OpenELMConfig(tie_word_embeddings=True).share_input_output_layers)
        self.assertFalse(OpenELMConfig(tie_word_embeddings=True, share_input_output_layers=False).tie_word_embeddings)

    def test_invalid_tensor_parallel_heads(self):
        from paddleformers.transformers import OpenELMConfig, OpenELMModel

        config = OpenELMConfig(
            **dict(SMALL_CONFIG, model_dim=96, head_dim=8, num_gqa_groups=4, tensor_model_parallel_size=2)
        )
        with self.assertRaisesRegex(ValueError, "divisible by TP size"):
            OpenELMModel(config)

    def test_model_initialization(self):
        from unittest.mock import patch

        from paddleformers.transformers import (
            OpenELMConfig,
            OpenELMForCausalLM,
            OpenELMModel,
        )
        from paddleformers.transformers.model_utils import no_init_weights

        config = OpenELMConfig(**dict(SMALL_CONFIG, model_dim=128, vocab_size=256, share_input_output_layers=False))
        for model_class in (OpenELMModel, OpenELMForCausalLM):
            with self.subTest(model_class=model_class.__name__):
                paddle.seed(_SEED)
                model = model_class(config)
                base = model if model_class is OpenELMModel else model.model
                expected_std = config.model_dim**-0.5
                residual_std = expected_std * (2 * config.num_transformer_layers) ** -0.5
                projections = [(base.embed_tokens.weight, expected_std)]
                for layer in base.layers:
                    projections.extend(
                        [
                            (layer.self_attn.q_proj.weight, expected_std),
                            (layer.self_attn.k_proj.weight, expected_std),
                            (layer.self_attn.v_proj.weight, expected_std),
                            (layer.self_attn.o_proj.weight, residual_std),
                            (layer.mlp.up_proj.weight, expected_std),
                            (layer.mlp.gate_proj.weight, expected_std),
                            (layer.mlp.down_proj.weight, residual_std),
                        ]
                    )
                    np.testing.assert_array_equal(layer.input_layernorm.weight.numpy(), 1.0)
                    np.testing.assert_array_equal(layer.post_attention_layernorm.weight.numpy(), 1.0)
                if model_class is OpenELMForCausalLM:
                    projections.append((model.lm_head.weight, config.initializer_range))
                for weight, std in projections:
                    self.assertAlmostEqual(float(weight.numpy().std()), std, delta=std * 0.1)
                before = {key: value.numpy().copy() for key, value in model.state_dict().items()}
                model.init_weights()
                for key, value in model.state_dict().items():
                    np.testing.assert_array_equal(value.numpy(), before[key], err_msg=key)
                with no_init_weights(), patch.object(model_class, "_init_weights", side_effect=AssertionError):
                    model_class(config)

    def test_model_forward(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [self.BATCH, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, return_dict=True)

        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertEqual(
            list(output.last_hidden_state.shape),
            [self.BATCH, self.SEQ, SMALL_CONFIG["model_dim"]],
        )

    def test_forward_with_attention_mask(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [self.BATCH, self.SEQ])
        attn_mask = paddle.ones([self.BATCH, self.SEQ], dtype=paddle.int64)
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)

        self.assertEqual(
            list(output.last_hidden_state.shape),
            [self.BATCH, self.SEQ, SMALL_CONFIG["model_dim"]],
        )

    def test_forward_with_cache(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, use_cache=True, return_dict=True)

        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertIsNotNone(output.past_key_values)

    def test_forward_without_cache(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, use_cache=False, return_dict=True)

        self.assertIsNone(output.past_key_values)

    def test_deterministic_output(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, 6])
        with paddle.no_grad():
            out1 = self.model(input_ids=input_ids, return_dict=True).last_hidden_state
            out2 = self.model(input_ids=input_ids, return_dict=True).last_hidden_state
        self.assertTrue(paddle.allclose(out1, out2))

    def test_output_dtype_consistent(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, return_dict=True)
        self.assertTrue(output.last_hidden_state.dtype in [paddle.float32, paddle.float16, paddle.bfloat16])

    def test_forward_with_inputs_embeds(self):
        inputs_embeds = paddle.randn([self.BATCH, self.SEQ, SMALL_CONFIG["model_dim"]])
        with paddle.no_grad():
            output = self.model(inputs_embeds=inputs_embeds, return_dict=True)

        self.assertEqual(
            list(output.last_hidden_state.shape),
            [self.BATCH, self.SEQ, SMALL_CONFIG["model_dim"]],
        )

    def test_forward_output_hidden_states(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)

        self.assertIsNotNone(output.hidden_states)
        self.assertEqual(
            len(output.hidden_states),
            SMALL_CONFIG["num_transformer_layers"] + 1,
        )

    def test_forward_tuple_output(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, return_dict=False)

        self.assertIsInstance(output, tuple)
        self.assertEqual(list(output[0].shape), [1, self.SEQ, SMALL_CONFIG["model_dim"]])


class OpenELMForCausalLMTest(unittest.TestCase):
    BATCH = 2
    SEQ = 10

    def setUp(self):
        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        self.config = OpenELMConfig(**SMALL_CONFIG)
        self.model = OpenELMForCausalLM(self.config)
        self.model.eval()

    def test_causal_lm_from_config(self):
        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        config = OpenELMConfig(**SMALL_CONFIG)
        model = OpenELMForCausalLM(config)
        self.assertIsNotNone(model)

        if config.share_input_output_layers:
            self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_causal_lm_forward(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [self.BATCH, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, return_dict=True)

        self.assertTrue(hasattr(output, "logits"))
        self.assertEqual(
            list(output.logits.shape),
            [self.BATCH, self.SEQ, SMALL_CONFIG["vocab_size"]],
        )

    def test_causal_lm_forward_with_labels(self):
        original_device = paddle.get_device()
        paddle.set_device("cpu")
        try:
            from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

            config = OpenELMConfig(**SMALL_CONFIG)
            model = OpenELMForCausalLM(config)
            model.eval()

            input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
            labels = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, self.SEQ])
            with paddle.no_grad():
                output = model(input_ids=input_ids, labels=labels, return_dict=True)

            self.assertIsNotNone(output.loss)
            self.assertEqual(output.loss.shape, [])
            self.assertTrue(float(output.loss) > 0)
        finally:
            paddle.set_device(original_device)

    def test_causal_lm_generate_autoregressive(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, 5])
        gen_ids = []
        max_gen_tokens = 5

        cur_ids = input_ids.numpy()
        with paddle.no_grad():
            for step in range(max_gen_tokens):
                input_tensor = paddle.to_tensor(cur_ids, dtype="int64")
                out = self.model(input_ids=input_tensor, use_cache=False, return_dict=True)
                next_token = int(out.logits[0, -1].argmax().item())
                gen_ids.append(next_token)
                cur_ids = np.concatenate([cur_ids, [[next_token]]], axis=1)

        self.assertEqual(len(gen_ids), max_gen_tokens)
        for tid in gen_ids:
            self.assertGreaterEqual(tid, 0)
            self.assertLess(tid, SMALL_CONFIG["vocab_size"])

    def test_causal_lm_masked_loss(self):
        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        input_ids = paddle.to_tensor([[1, 7, 9, 2]], dtype="int64")
        labels = paddle.to_tensor([[7, 9, 2, -100]], dtype="int64")
        loss_mask = paddle.to_tensor([[1, 0, 1, 0]], dtype="float32")
        for fused_head in (False, True):
            with self.subTest(use_fused_head_and_loss_fn=fused_head):
                config = OpenELMConfig(**dict(SMALL_CONFIG, use_fused_head_and_loss_fn=fused_head))
                model = OpenELMForCausalLM(config)
                model.train()
                output = model(
                    input_ids=input_ids, labels=labels, loss_mask=loss_mask, use_cache=False, return_dict=True
                )
                unmasked = model(input_ids=input_ids, labels=labels, use_cache=False, return_dict=True)
                with paddle.no_grad():
                    hidden = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
                    logits = paddle.matmul(hidden, model.lm_head.weight, transpose_y=True)
                    token_loss = paddle.nn.functional.cross_entropy(
                        logits.astype("float32"), labels.unsqueeze(-1), reduction="none", ignore_index=-100
                    ).reshape(labels.shape)
                    expected = (token_loss * loss_mask).sum() / loss_mask.sum()
                    expected_unmasked = token_loss.sum() / (labels != -100).astype("float32").sum()
                np.testing.assert_allclose(output.loss.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(unmasked.loss.numpy(), expected_unmasked.numpy(), atol=1e-5, rtol=1e-5)
                output.loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(paddle.isfinite(parameter.grad).all().item(), name)

    def test_causal_lm_deterministic_output(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, 6])
        with paddle.no_grad():
            out1 = self.model(input_ids=input_ids, return_dict=True).logits
            out2 = self.model(input_ids=input_ids, return_dict=True).logits
        self.assertTrue(paddle.allclose(out1, out2))

    def test_causal_lm_prepare_inputs_for_generation(self):
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 5]], dtype="int64")
        attention_mask = paddle.ones([1, 5], dtype="int64")

        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            use_cache=True,
        )

        self.assertIn("input_ids", model_inputs)
        self.assertIn("attention_mask", model_inputs)
        self.assertIn("position_ids", model_inputs)
        self.assertIn("use_cache", model_inputs)

    def test_causal_lm_with_cache_autoregressive(self):
        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, 3])
        gen_ids = []
        max_gen_tokens = 3

        with paddle.no_grad():
            out = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
            next_token = int(out.logits[0, -1].argmax().item())
            gen_ids.append(next_token)
            past_kv = out.past_key_values

            for step in range(1, max_gen_tokens):
                next_input = paddle.to_tensor([[next_token]], dtype="int64")
                attention_mask = paddle.ones([1, input_ids.shape[1] + step], dtype="int64")
                out = self.model(
                    input_ids=next_input,
                    past_key_values=past_kv,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                next_token = int(out.logits[0, -1].argmax().item())
                gen_ids.append(next_token)
                past_kv = out.past_key_values

        self.assertEqual(len(gen_ids), max_gen_tokens)

    def test_causal_lm_not_share_input_output(self):
        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        config_dict = dict(SMALL_CONFIG)
        config_dict["share_input_output_layers"] = False
        config = OpenELMConfig(**config_dict)
        model = OpenELMForCausalLM(config)
        model.eval()

        self.assertIsNotNone(model.lm_head)
        self.assertFalse(hasattr(model.model, "classifier"))

        input_ids = paddle.randint(0, config.vocab_size, [1, 5])
        with paddle.no_grad():
            output = model(input_ids=input_ids, return_dict=True)

        self.assertEqual(list(output.logits.shape), [1, 5, config.vocab_size])
        model.train()
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False, return_dict=True)
        output.loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(paddle.isfinite(parameter.grad).all().item(), name)

    def test_causal_lm_chunked_cache_and_masks(self):
        from paddle.nn.functional.flash_attention import sdp_kernel

        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        ids = paddle.to_tensor([[1, 7, 8, 9, 2]], dtype="int64")
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend), sdp_kernel(
                enable_math=True, enable_flash=False, enable_mem_efficient=False
            ):
                config = OpenELMConfig(**dict(SMALL_CONFIG, num_gqa_groups=2, _attn_implementation=backend))
                model = OpenELMForCausalLM(config)
                model.eval()
                with paddle.no_grad():
                    cached = model(ids[:, :2], use_cache=True, return_dict=True)
                    cache = cached.past_key_values
                    for start, end in ((2, 4), (4, 5)):
                        cached = model(ids[:, start:end], past_key_values=cache, use_cache=True, return_dict=True)
                        full = model(ids[:, :end], use_cache=False, return_dict=True)
                        np.testing.assert_allclose(
                            cached.logits.numpy(), full.logits[:, start:end].numpy(), atol=1e-5, rtol=1e-5
                        )
                        self.assertIs(cached.past_key_values, cache)
                        self.assertEqual(cache.get_seq_length(), end)
                    padding = paddle.to_tensor([[0, 1, 1, 1, 1]], dtype="bool")
                    visible = paddle.tril(paddle.ones([5, 5], dtype="bool"))[None, None] & padding[:, None, None, :]
                    additive = paddle.where(visible, 0.0, paddle.finfo("float32").min)
                    a = model(ids, attention_mask=padding, use_cache=False, return_dict=True).logits
                    b = model(ids, attention_mask=additive, use_cache=False, return_dict=True).logits
                    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-5, rtol=1e-5)
                    row_indices = paddle.full([1, 1, 5, 1], 5, dtype="int32")
                    sparse = model(ids, attn_mask_startend_row_indices=row_indices, use_cache=False, return_dict=True)
                    dense = model(ids, use_cache=False, return_dict=True)
                    np.testing.assert_allclose(sparse.logits.numpy(), dense.logits.numpy(), atol=1e-5, rtol=1e-5)
                    prefill = model(ids[:, :2], use_cache=True, return_dict=True)
                    chunk = model(
                        ids[:, 2:],
                        past_key_values=prefill.past_key_values,
                        attn_mask_startend_row_indices=row_indices,
                        use_cache=True,
                        return_dict=True,
                    )
                    np.testing.assert_allclose(chunk.logits.numpy(), dense.logits[:, 2:].numpy(), atol=1e-5, rtol=1e-5)
                    model_inputs = model.prepare_inputs_for_generation(ids, use_cache=True)
                    self.assertEqual(list(model_inputs["position_ids"].shape), [1, 5])

    def test_pipeline_stage_equivalence(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from paddle.distributed.fleet.meta_parallel.pp_utils import (
            p2p_communication as p2p,
        )

        from paddleformers.nn.pp_model import (
            LMHeadPipe,
            RMSNormPipe,
            make_decoder_layer_pipe,
        )
        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM
        from paddleformers.transformers.openelm.modeling import (
            OpenELMCriterionPipe,
            OpenELMDecoderLayer,
            OpenELMEmbeddingPipe,
            OpenELMRotaryEmbedding,
        )

        def transport(tensors):
            messages = []
            sender, receiver = p2p.SendRecvMeta(), p2p.SendRecvMeta()
            group = SimpleNamespace(_get_p2p_next_rank=lambda: 1, _get_p2p_prev_rank=lambda: 0)
            with patch.object(p2p, "_hcg", group), patch.object(
                paddle.distributed, "send", side_effect=lambda tensor, **kwargs: messages.append(tensor.clone())
            ), patch.object(
                paddle.distributed,
                "recv",
                side_effect=lambda tensor, **kwargs: tensor.set_value(messages.pop(0).reshape(tensor.shape)),
            ):
                sender.send_meta(tensors, None)
                receiver.recv_meta(None)
            for tensor, shape, dtype, stopped in zip(
                tensors, receiver.recv_shape_message, receiver.recv_dtype_message, receiver.recv_stop_gradient
            ):
                self.assertEqual(list(tensor.shape), shape)
                self.assertEqual(tensor.dtype, getattr(paddle, p2p.number_2_dtype(dtype)))
                self.assertEqual(tensor.stop_gradient, bool(stopped))
            return tuple(tensor.clone() for tensor in tensors)

        ids = paddle.to_tensor([[1, 7, 8, 2]], dtype="int64")
        padding = paddle.to_tensor([[1, 1, 1, 0]], dtype="int64")
        labels = paddle.to_tensor([[7, 8, 2, -100]], dtype="int64")
        mask = paddle.to_tensor([[1, 0, 1, 0]], dtype="float32")
        for tied in (False, True):
            for use_recompute in (False, True):
                with self.subTest(tied=tied, recompute=use_recompute):
                    config = OpenELMConfig(
                        **dict(
                            SMALL_CONFIG,
                            share_input_output_layers=tied,
                            recompute_granularity="full" if use_recompute else None,
                            recompute_method="uniform",
                            recompute_num_layers=1,
                        )
                    )
                    reference = OpenELMForCausalLM(config)
                    reference.train()
                    embedding = OpenELMEmbeddingPipe(config, rotary_emb_cls=OpenELMRotaryEmbedding)
                    embedding.embed_tokens.set_state_dict(reference.model.embed_tokens.state_dict())
                    layers = [
                        make_decoder_layer_pipe(OpenELMDecoderLayer)(config, i)
                        for i in range(config.num_hidden_layers)
                    ]
                    for layer, source in zip(layers, reference.model.layers):
                        layer.set_state_dict(source.state_dict())
                    norm = RMSNormPipe(config)
                    norm.set_state_dict(reference.model.norm.state_dict())
                    head = LMHeadPipe(config)
                    head.set_state_dict(reference.lm_head.state_dict())
                    if tied:
                        head.weight = embedding.embed_tokens.weight
                    stage_output = transport(embedding((ids, padding, None)))
                    for layer in layers:
                        stage_output = transport(layer(stage_output))
                    logits = head(norm(stage_output))
                    expected = reference(
                        ids, attention_mask=padding, labels=labels, loss_mask=mask, use_cache=False, return_dict=True
                    )
                    np.testing.assert_allclose(logits.numpy(), expected.logits.numpy(), atol=1e-5, rtol=1e-5)
                    criterion = OpenELMCriterionPipe(config)
                    criterion.train()
                    loss = criterion(logits, (labels, mask))
                    np.testing.assert_allclose(loss.numpy(), expected.loss.numpy(), atol=1e-5, rtol=1e-5)
                    loss.backward()
                    expected.loss.backward()
                    for layer, source in zip(layers, reference.model.layers):
                        parameters = dict(source.named_parameters())
                        for name, parameter in layer.named_parameters():
                            np.testing.assert_allclose(
                                parameter.grad.numpy(),
                                parameters[name].grad.numpy(),
                                atol=1e-5,
                                rtol=1e-5,
                                err_msg=name,
                            )
                    np.testing.assert_allclose(
                        embedding.embed_tokens.weight.grad.numpy(),
                        reference.model.embed_tokens.weight.grad.numpy(),
                        atol=1e-5,
                        rtol=1e-5,
                    )


class OpenELMTokenizerIntegrationTest(unittest.TestCase):
    @slow
    @require_package("transformers")
    def test_tokenizer_with_model(self):
        from paddleformers.transformers import (
            AutoTokenizer,
            OpenELMConfig,
            OpenELMForCausalLM,
        )

        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)

        config_dict = dict(SMALL_CONFIG)
        config_dict["vocab_size"] = 32000
        config = OpenELMConfig(**config_dict)
        model = OpenELMForCausalLM(config)
        model.eval()

        text = "Hello"
        encoded = tokenizer(text, return_tensors=None)
        input_ids = encoded["input_ids"]

        for tid in input_ids:
            self.assertGreaterEqual(tid, 0)
            self.assertLess(tid, config.vocab_size)

        input_tensor = paddle.to_tensor([input_ids], dtype="int64")
        with paddle.no_grad():
            output = model(input_ids=input_tensor, return_dict=True)

        self.assertEqual(
            list(output.logits.shape),
            [1, len(input_ids), config.vocab_size],
        )


class OpenELMCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="openelm_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @require_package("safetensors")
    def test_hf_checkpoint_conversion(self):
        from safetensors.numpy import load_file, save_file

        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        for glu in (False, True):
            for tied in (False, True):
                with self.subTest(ffn_with_glu=glu, share_input_output_layers=tied):
                    config = OpenELMConfig(
                        **dict(
                            SMALL_CONFIG,
                            head_dim=8,
                            qkv_multipliers=(0.5, 1.0),
                            num_gqa_groups=2,
                            normalize_qk_projections=True,
                            ffn_with_glu=glu,
                            share_input_output_layers=tied,
                        )
                    )
                    reference = OpenELMForCausalLM(config)
                    reference.eval()
                    weights = {key: tensor.numpy() for key, tensor in reference.state_dict().items()}
                    external = {"transformer.token_embeddings.weight": weights["model.embed_tokens.weight"]}
                    external["transformer.norm.weight"] = weights["model.norm.weight"]
                    for layer in range(config.num_transformer_layers):
                        src = f"model.layers.{layer}"
                        dst = f"transformer.layers.{layer}"
                        for old, new in (("attn_norm", "input_layernorm"), ("ffn_norm", "post_attention_layernorm")):
                            external[f"{dst}.{old}.weight"] = weights[f"{src}.{new}.weight"]
                        for norm in ("q_norm", "k_norm"):
                            external[f"{dst}.attn.{norm}.weight"] = weights[f"{src}.self_attn.{norm}.weight"]
                        external[f"{dst}.attn.qkv_proj.weight"] = np.concatenate(
                            [weights[f"{src}.self_attn.{proj}_proj.weight"].T for proj in ("q", "k", "v")], axis=0
                        )
                        external[f"{dst}.attn.out_proj.weight"] = weights[f"{src}.self_attn.o_proj.weight"].T
                        up = weights[f"{src}.mlp.up_proj.weight"].T
                        external[f"{dst}.ffn.proj_1.weight"] = (
                            np.concatenate([weights[f"{src}.mlp.gate_proj.weight"].T, up], axis=0) if glu else up
                        )
                        external[f"{dst}.ffn.proj_2.weight"] = weights[f"{src}.mlp.down_proj.weight"].T
                    if not tied:
                        external["lm_head.weight"] = weights["lm_head.weight"]
                    source_weights = {key: np.ascontiguousarray(value) for key, value in external.items()}
                    if not tied:
                        source_weights["transformer.classifier.weight"] = np.ones(
                            [config.vocab_size, config.model_dim], dtype="float32"
                        )
                    source_dir = os.path.join(self.tmp_dir, "source")
                    os.makedirs(source_dir, exist_ok=True)
                    config.save_pretrained(source_dir)
                    save_file(
                        source_weights,
                        os.path.join(source_dir, "model.safetensors"),
                        metadata={"format": "pt"},
                    )
                    loaded = OpenELMForCausalLM.from_pretrained(
                        source_dir, dtype="float32", load_checkpoint_format="flex_checkpoint"
                    )
                    loaded.eval()
                    if tied:
                        self.assertIs(loaded.lm_head.weight, loaded.model.embed_tokens.weight)
                    self.assertEqual(set(loaded.state_dict()), set(weights))
                    for key, tensor in loaded.state_dict().items():
                        np.testing.assert_array_equal(tensor.numpy(), weights[key], err_msg=key)
                    low_precision = OpenELMForCausalLM.from_pretrained(
                        source_dir, dtype="bfloat16", load_checkpoint_format="flex_checkpoint"
                    )
                    for key, tensor in low_precision.state_dict().items():
                        self.assertEqual(tensor.dtype, paddle.bfloat16, key)
                        expected_weight = paddle.to_tensor(weights[key]).astype("bfloat16").astype("float32")
                        np.testing.assert_array_equal(
                            tensor.astype("float32").numpy(), expected_weight.numpy(), err_msg=key
                        )
                    input_ids = paddle.to_tensor([[1, 7, 9, 2]], dtype="int64")
                    with paddle.no_grad():
                        expected = reference(input_ids, use_cache=False, return_dict=True).logits
                        actual = loaded(input_ids, use_cache=False, return_dict=True).logits
                    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)
                    export_dir = os.path.join(self.tmp_dir, "export")
                    loaded.save_pretrained(
                        export_dir, save_checkpoint_format="flex_checkpoint", safe_serialization=True
                    )
                    exported = {}
                    for filename in os.listdir(export_dir):
                        if filename.endswith(".safetensors"):
                            exported.update(load_file(os.path.join(export_dir, filename)))
                    self.assertEqual(set(exported), set(external))
                    for key, value in external.items():
                        np.testing.assert_array_equal(exported[key], value, err_msg=key)

    def test_save_and_load(self):
        from paddleformers.transformers import OpenELMConfig, OpenELMForCausalLM

        config = OpenELMConfig(**SMALL_CONFIG)
        model = OpenELMForCausalLM(config)
        model.eval()

        input_ids = paddle.randint(0, SMALL_CONFIG["vocab_size"], [1, 5])
        with paddle.no_grad():
            out_before = model(input_ids=input_ids, return_dict=True)

        # Save state dict and config to tmp directory
        state_dict = model.state_dict()
        paddle.save(state_dict, os.path.join(self.tmp_dir, "model_state.pdparams"))
        config.save_pretrained(self.tmp_dir)

        # Load from tmp: create new model with same config, load saved state dict
        loaded = OpenELMForCausalLM(config)
        loaded.set_state_dict(paddle.load(os.path.join(self.tmp_dir, "model_state.pdparams")))
        loaded.eval()

        with paddle.no_grad():
            out_after = loaded(input_ids=input_ids, return_dict=True)

        self.assertTrue(
            paddle.allclose(out_before.logits, out_after.logits),
            "Model output inconsistent after save/load",
        )


class OpenELMIntegrationTest(unittest.TestCase):
    @slow
    @require_package("transformers")
    def test_paddle_inference_chinese(self):
        from paddleformers.transformers import AutoTokenizer, OpenELMForCausalLM

        paddle.seed(_SEED)
        paddle.set_device("gpu")

        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)

        model = OpenELMForCausalLM.from_pretrained(
            _MODEL_HF_ID,
            convert_from_hf=True,
            load_checkpoint_format="flex_checkpoint",
            download_hub="modelscope",
            dtype="float32",
        )
        model.eval()

        encoded = tokenizer(_PROMPT_QUESTION, return_tensors=None)
        input_ids_list = encoded["input_ids"]
        print(f"[Paddle Inference] prompt: {repr(_PROMPT_QUESTION)}")
        print(f"[Paddle Inference] input_ids: {input_ids_list}")

        cur_ids = np.array([input_ids_list], dtype=np.int64)
        gen_ids = []
        max_gen_tokens = 64
        with paddle.no_grad():
            for step in range(max_gen_tokens):
                input_tensor = paddle.to_tensor(cur_ids, dtype="int64")
                out = model(input_ids=input_tensor, use_cache=False, return_dict=True)
                next_token = int(out.logits[0, -1].argmax().item())
                gen_ids.append(next_token)
                cur_ids = np.concatenate([cur_ids, [[next_token]]], axis=1)
                if next_token == tokenizer.eos_token_id:
                    break

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(input_ids_list + gen_ids, skip_special_tokens=True)

        print("=" * 60)
        print(f"[Paddle Inference] Generated text: {repr(gen_text)}")
        print(f"[Paddle Inference] Full text: {repr(full_text)}")
        print("=" * 60)

        self.assertGreater(len(gen_ids), 0, "Model generated no tokens")
        self.assertGreater(len(gen_text.strip()), 0, "Model generated empty text")

        self.assertIn(
            "beijing",
            full_text.lower(),
            "Model output should contain beijing",
        )

        printable_chars = sum(1 for c in gen_text if c.isprintable() or c.isspace())
        printable_ratio = printable_chars / len(gen_text) if len(gen_text) > 0 else 0
        self.assertGreater(printable_ratio, 0.5, "Output contains too many unprintable characters")


class OpenELMHuggingFaceCompatibilityTest(unittest.TestCase):
    @slow
    @require_package("transformers", "torch")
    def test_diff_alignment(self):
        import json
        import sys

        import numpy as np
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from paddleformers.transformers import AutoTokenizer, OpenELMForCausalLM

        paddle.set_device("gpu")

        # Clear transformers modules to ensure fresh load
        for mod_name in [m for m in list(sys.modules) if m == "transformers" or m.startswith("transformers.")]:
            del sys.modules[mod_name]

        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        TorchOpenELMConfig = get_class_from_dynamic_module("configuration_openelm.OpenELMConfig", _MODEL_HF_ID)
        TorchOpenELMForCausalLM = get_class_from_dynamic_module("modeling_openelm.OpenELMForCausalLM", _MODEL_HF_ID)

        config_path = hf_hub_download(_MODEL_HF_ID, "config.json")
        safetensors_path = hf_hub_download(_MODEL_HF_ID, "model.safetensors")

        with open(config_path) as f:
            config_dict = json.load(f)

        torch_config = TorchOpenELMConfig(**config_dict)

        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model = TorchOpenELMForCausalLM(torch_config)
        state_dict = load_file(safetensors_path)
        loading = torch_model.load_state_dict(state_dict, strict=False)
        self.assertEqual(loading.missing_keys, [])
        torch_model = torch_model.float().to(torch_device)
        torch_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)
        inputs = tokenizer(_PROMPT_DIFF, return_tensors="np")
        input_ids_np = np.array(inputs["input_ids"], dtype=np.int64)
        input_ids_list = input_ids_np[0].tolist()
        input_ids_pt = torch.tensor(input_ids_np, dtype=torch.long)
        print(f"\n[Diff] prompt: {repr(_PROMPT_DIFF)}")
        print(f"[Diff] input_ids: {input_ids_list}, seq_len={len(input_ids_list)}")

        # --- Torch forward ---
        input_ids_dev = input_ids_pt.to(torch_device)
        with torch.inference_mode():
            torch_out = torch_model(input_ids=input_ids_dev, use_cache=False)
        torch_logits = torch_out.logits.float().cpu().numpy()

        # --- Torch generate ---
        torch_gen_ids = []
        cur_torch = input_ids_dev.clone()
        with torch.inference_mode():
            for _ in range(_NUM_GEN_TOKENS):
                out = torch_model(input_ids=cur_torch, use_cache=False)
                next_token = out.logits[0, -1].argmax().item()
                torch_gen_ids.append(next_token)
                cur_torch = torch.cat([cur_torch, torch.tensor([[next_token]], device=torch_device)], dim=1)
        print(f"[Diff] PyTorch generated tokens: {torch_gen_ids}")
        print(f"[Diff] PyTorch generated text: {repr(tokenizer.decode(torch_gen_ids, skip_special_tokens=True))}")

        del torch_model
        torch.cuda.empty_cache()

        # --- Paddle model ---
        paddle.seed(_SEED)
        paddle_model = OpenELMForCausalLM.from_pretrained(
            _MODEL_HF_ID,
            convert_from_hf=True,
            load_checkpoint_format="flex_checkpoint",
            download_hub="modelscope",
            dtype="float32",
        )
        paddle_model.eval()

        input_ids_pd = paddle.to_tensor(np.array([input_ids_list], dtype=np.int64), dtype="int64")

        with paddle.no_grad():
            paddle_out = paddle_model(input_ids=input_ids_pd, use_cache=False, return_dict=True)
        paddle_logits = paddle_out.logits.astype("float32").numpy()

        cur_ids = np.array([input_ids_list], dtype=np.int64)
        paddle_gen_ids = []
        with paddle.no_grad():
            for step in range(_NUM_GEN_TOKENS):
                input_tensor = paddle.to_tensor(cur_ids, dtype="int64")
                out = paddle_model(input_ids=input_tensor, use_cache=False, return_dict=True)
                next_token = int(out.logits[0, -1].argmax().item())
                paddle_gen_ids.append(next_token)
                cur_ids = np.concatenate([cur_ids, [[next_token]]], axis=1)

        print(f"[Diff] Paddle generated tokens: {paddle_gen_ids}")
        print(f"[Diff] Paddle generated text: {repr(tokenizer.decode(paddle_gen_ids, skip_special_tokens=True))}")

        self.assertEqual(
            list(torch_logits.shape),
            list(paddle_logits.shape),
            f"logits shape mismatch: torch={torch_logits.shape}, paddle={paddle_logits.shape}",
        )

        diff = np.abs(torch_logits - paddle_logits)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        print(f"[Diff] logits max_diff: {max_diff:.6e} (threshold: {_DIFF_THRESHOLD})")
        print(f"[Diff] logits mean_diff: {mean_diff:.6e}")

        self.assertLess(
            max_diff,
            _DIFF_THRESHOLD,
            f"logits max_diff={max_diff:.6e} >= threshold={_DIFF_THRESHOLD}",
        )
        self.assertLess(
            mean_diff,
            1e-2,
            f"logits mean_diff={mean_diff:.6e} >= threshold={1e-2}",
        )

        self.assertEqual(len(torch_gen_ids), _NUM_GEN_TOKENS)
        self.assertEqual(len(paddle_gen_ids), _NUM_GEN_TOKENS)
        n = _NUM_GEN_TOKENS
        print(f"\n[Diff] Token comparison (first {n} tokens):")
        print(f"  {'Step':>4}  {'Torch':>10}  {'Paddle':>10}  {'Status':>6}")
        print("  " + "-" * 42)
        for i in range(n):
            ok = "OK" if torch_gen_ids[i] == paddle_gen_ids[i] else "FAIL"
            print(f"  {i+1:4d}  {torch_gen_ids[i]:10d}  {paddle_gen_ids[i]:10d}  {ok:>6}")

        self.assertEqual(
            torch_gen_ids[:n],
            paddle_gen_ids[:n],
            f"First {n} generated tokens mismatch",
        )


if __name__ == "__main__":
    unittest.main()
