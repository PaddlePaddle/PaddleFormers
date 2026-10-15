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
from __future__ import annotations

import itertools
import tempfile
import unittest

import paddle
from parameterized import parameterized

from paddleformers.transformers import (
    StableLmConfig,
    StableLmForCausalLM,
    StableLmModel,
    StableLmTokenizer,
)
from paddleformers.transformers.stablelm.modeling import (
    StableLmAttention,
    StableLmDecoderLayer,
    StableLmLayerNormPerHead,
    StableLmRotaryEmbedding,
)
from tests.testing_utils import gpu_device_initializer, require_package, slow


class StableLmLayerNormPerHeadTest(unittest.TestCase):
    @parameterized.expand([(1, False), (2, False), (4, False), (4, True)])
    def test_per_head_normalization(self, num_heads, bias):
        layer = StableLmLayerNormPerHead(8, num_heads, bias=bias)
        hidden_states = paddle.randn([2, num_heads, 3, 8])
        expected_heads = []
        for head, norm in enumerate(layer.norms):
            norm.weight.set_value(paddle.full([8], float(head + 1)))
            if bias:
                norm.bias.set_value(paddle.full([8], float(head) / 10))
            states = hidden_states[:, head : head + 1]
            centered = states - states.mean(axis=-1, keepdim=True)
            expected = centered / paddle.sqrt((centered * centered).mean(axis=-1, keepdim=True) + 1e-5)
            expected = expected * norm.weight
            if bias:
                expected = expected + norm.bias
            expected_heads.append(expected)
        paddle.testing.assert_close(layer(hidden_states), paddle.concat(expected_heads, axis=1), rtol=1e-5, atol=1e-6)


class StableLmModelTester:
    def __init__(
        self,
        parent,
        vocab_size=32000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=256,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        is_training=True,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        dtype="bfloat16",
        batch_size: int = 2,
        seq_length: int = 10,
        use_input_mask: bool = False,
        use_labels: bool = False,
        return_dict: bool = False,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_training = is_training
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.dtype = dtype
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.return_dict = return_dict

    def get_config(self):
        return StableLmConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            initializer_range=self.initializer_range,
            use_qkv_bias=False,
            qk_layernorm=False,
            use_parallel_residual=False,
        )

    def prepare_inputs_for_common(self):
        input_ids = paddle.randint(0, self.vocab_size, [self.batch_size, self.seq_length], dtype="int64")
        return {"input_ids": input_ids}


class StableLmModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = StableLmModelTester(self)
        self.config = self.model_tester.get_config()

    def test_qk_layernorm_rejects_tensor_parallelism(self):
        self.config.qk_layernorm = True
        self.config.tensor_model_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "does not support tensor parallelism"):
            StableLmAttention(self.config)

    @parameterized.expand([("linear",), ("dynamic",)])
    def test_legacy_rope_scaling(self, rope_type):
        scaling = {"type": rope_type, "factor": 2.0}
        config = StableLmConfig(rope_scaling=scaling)
        self.assertEqual(config.rope_parameters["rope_type"], rope_type)
        self.assertEqual(config.rope_parameters["factor"], 2.0)
        self.assertEqual(scaling, {"type": rope_type, "factor": 2.0})
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = StableLmConfig.from_pretrained(tmpdir)
        self.assertEqual(loaded.rope_parameters, config.rope_parameters)

    def test_rope_parameters_take_precedence(self):
        config = StableLmConfig(
            rope_parameters={"rope_type": "linear", "factor": 3.0},
            rope_scaling={"type": "linear", "factor": 2.0},
        )
        self.assertEqual(config.rope_parameters["factor"], 3.0)

    @parameterized.expand([(False,), (True,)])
    @gpu_device_initializer(log_prefix="StableLmModelTest")
    def test_residual_formula(self, parallel):
        config = StableLmConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_parallel_residual=parallel,
            qk_layernorm=True,
        )
        layer = StableLmDecoderLayer(config, 0)
        layer.eval()
        hidden_states = paddle.randn([2, 4, 64])
        position_ids = paddle.arange(4).unsqueeze(0)
        with paddle.no_grad():
            position_embeddings = StableLmRotaryEmbedding(config)(hidden_states, position_ids)
            normalized = layer.input_layernorm(hidden_states)
            attention_output, _ = layer.self_attn(normalized, position_embeddings=position_embeddings)
            if parallel:
                expected = hidden_states + attention_output + layer.mlp(normalized)
            else:
                residual = hidden_states + attention_output
                expected = residual + layer.mlp(layer.post_attention_layernorm(residual))
            actual = layer(hidden_states, position_embeddings=position_embeddings)
        paddle.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    @gpu_device_initializer(log_prefix="StableLmModelTest")
    def test_loss_mask(self):
        config = StableLmConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model = StableLmForCausalLM(config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        labels = paddle.to_tensor([[2, 3, 4, 5]], dtype="int64")
        loss_mask = paddle.to_tensor([[1.0, 0.0, 0.5, 0.0]])
        with paddle.no_grad():
            outputs = model(input_ids, labels=labels, loss_mask=loss_mask, return_dict=True)
            token_loss = paddle.nn.functional.cross_entropy(outputs.logits, labels, reduction="none")
            expected = (token_loss * loss_mask).sum() / loss_mask.sum()
            changed_labels = paddle.to_tensor([[2, 8, 4, 9]], dtype="int64")
            changed = model(input_ids, labels=changed_labels, loss_mask=loss_mask, return_dict=True)
        paddle.testing.assert_close(outputs.loss, expected, rtol=1e-5, atol=1e-6)
        paddle.testing.assert_close(changed.loss, outputs.loss, rtol=0.0, atol=0.0)

    @gpu_device_initializer(log_prefix="StableLmModelTest")
    def test_config(self):
        config = StableLmConfig()
        self.assertEqual(config.model_type, "stablelm")
        self.assertEqual(config.hidden_size, 2560)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.vocab_size, 50304)
        self.assertIsInstance(config.rope_parameters, dict)
        self.assertEqual(config.rope_parameters.get("partial_rotary_factor"), 0.25)

    @gpu_device_initializer(log_prefix="StableLmModelTest")
    def test_model_creation(self):
        model = StableLmModel(self.config)
        self.assertIsInstance(model, StableLmModel)
        self.assertEqual(len(model.layers), self.config.num_hidden_layers)

    @gpu_device_initializer(log_prefix="StableLmModelTest")
    def test_model_forward(self):
        model = StableLmModel(self.config)
        model.eval()

        input_ids = paddle.randint(0, self.config.vocab_size, [1, 8], dtype="int64")
        with paddle.no_grad():
            outputs = model(input_ids, return_dict=True)
        self.assertEqual(outputs.last_hidden_state.shape, [1, 8, self.config.hidden_size])

    @gpu_device_initializer(log_prefix="StableLmModelTest")
    def test_causal_lm_model_forward(self):
        model = StableLmForCausalLM(self.config)
        model.eval()

        input_ids = paddle.randint(0, self.config.vocab_size, [1, 8], dtype="int64")
        with paddle.no_grad():
            outputs = model(input_ids, return_dict=True)
        self.assertEqual(outputs.logits.shape, [1, 8, self.config.vocab_size])

    @parameterized.expand(list(itertools.product([False, True], repeat=3)))
    @gpu_device_initializer(log_prefix="StableLmModelTest")
    def test_optional_branches_roundtrip(self, use_qkv_bias, qk_layernorm, use_parallel_residual):
        config = StableLmConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_qkv_bias=use_qkv_bias,
            qk_layernorm=qk_layernorm,
            use_parallel_residual=use_parallel_residual,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        model = StableLmForCausalLM(config)
        model.eval()
        with paddle.no_grad():
            for name, param in model.named_parameters():
                if "layernorm.norms." in name or name.endswith(".bias"):
                    param.set_value(paddle.randn(param.shape) * 0.1)
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        with paddle.no_grad():
            expected = model(input_ids, return_dict=True).logits
        self.assertTrue(paddle.isfinite(expected).all().item())
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = StableLmForCausalLM.from_pretrained(
                tmpdir, dtype="float32", load_checkpoint_format="flex_checkpoint"
            )
            loaded.eval()
            self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))
            for name, value in model.state_dict().items():
                paddle.testing.assert_close(value, loaded.state_dict()[name], rtol=0.0, atol=0.0)
            with paddle.no_grad():
                actual = loaded(input_ids, return_dict=True).logits
            paddle.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    @gpu_device_initializer(log_prefix="StableLmModelTest")
    def test_tiny_model_init_and_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StableLmConfig(
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                intermediate_size=512,
                vocab_size=1024,
            )
            model = StableLmForCausalLM(config)
            model.save_pretrained(tmpdir)

            loaded = StableLmForCausalLM.from_pretrained(
                tmpdir,
                dtype="float32",
                load_checkpoint_format="flex_checkpoint",
            )
            loaded.eval()

            input_ids = paddle.randint(0, 1024, [1, 4], dtype="int64")
            with paddle.no_grad():
                outputs = loaded(input_ids, return_dict=True)
            self.assertEqual(outputs.logits.shape, [1, 4, 1024])


class StableLmSlowTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="StableLmSlowTest")
    @require_package("transformers", "torch")
    @slow
    def test_generation_hello_world(self):
        tokenizer = StableLmTokenizer.from_pretrained(
            "stabilityai/stable-code-3b", trust_remote_code=False, download_hub="modelscope"
        )

        model = StableLmForCausalLM.from_pretrained(
            "stabilityai/stable-code-3b",
            dtype="bfloat16",
            load_via_cpu=True,
            download_hub="modelscope",
        )
        model.eval()

        prompt = "def hello_world():"
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = paddle.to_tensor(inputs["input_ids"])

        with paddle.no_grad():
            gen = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = gen[0][0].tolist()
        decoded = tokenizer.decode(generated_tokens)
        print("create code:" + decoded)
        self.assertIn("hello", decoded.lower())
        self.assertGreater(len(generated_tokens), 3)


class StableLmDiffTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="StableLmDiffTest")
    @require_package("transformers", "torch")
    @slow
    def test_diff_generated_tokens(self):
        """Compare first 10 generated tokens between Paddle and HF."""
        import torch
        from transformers import AutoModelForCausalLM

        tokenizer = StableLmTokenizer.from_pretrained(
            "stabilityai/stable-code-3b", trust_remote_code=False, download_hub="modelscope"
        )

        prompt = "def fib(n):"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids_np = inputs["input_ids"].cpu().numpy()

        hf_model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stable-code-3b",
            dtype=torch.float32,
            device_map="auto",
        )
        hf_model.eval()
        hf_inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            hf_gen = hf_model.generate(
                **hf_inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        hf_new_tokens = hf_gen[0, inputs["input_ids"].shape[1] :].tolist()

        pd_model = StableLmForCausalLM.from_pretrained(
            "stabilityai/stable-code-3b",
            dtype="float32",
            load_via_cpu=True,
            download_hub="modelscope",
        )
        pd_model.eval()
        pd_input_ids = paddle.to_tensor(input_ids_np)
        with paddle.no_grad():
            pd_gen = pd_model.generate(
                pd_input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pd_new_tokens = pd_gen[0][0].tolist()

        self.assertEqual(
            hf_new_tokens,
            pd_new_tokens,
            f"Generated tokens differ: HF={hf_new_tokens}, PD={pd_new_tokens}",
        )

    @gpu_device_initializer(log_prefix="StableLmDiffTest")
    @require_package("transformers", "torch")
    @slow
    def test_diff_first_token_logits(self):
        """Compare first token logits between Paddle and HF, diff should be within 1e-2."""
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM

        tokenizer = StableLmTokenizer.from_pretrained(
            "stabilityai/stable-code-3b", trust_remote_code=False, download_hub="modelscope"
        )

        prompt = "def sort_array(arr):"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids_np = inputs["input_ids"].cpu().numpy()

        hf_model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stable-code-3b",
            dtype=torch.float32,
            device_map="auto",
        )
        hf_model.eval()
        hf_inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            hf_outputs = hf_model(**hf_inputs)
            hf_logits = hf_outputs.logits[0, -1].cpu().float().numpy()

        pd_model = StableLmForCausalLM.from_pretrained(
            "stabilityai/stable-code-3b",
            dtype="float32",
            load_via_cpu=True,
            download_hub="modelscope",
        )
        pd_model.eval()
        pd_input_ids = paddle.to_tensor(input_ids_np)
        with paddle.no_grad():
            pd_outputs = pd_model(pd_input_ids, return_dict=True)
            pd_logits = pd_outputs.logits[0, -1].cpu().astype("float32").numpy()

        abs_diff = np.abs(hf_logits - pd_logits)
        mean_diff = np.mean(abs_diff)

        self.assertLess(
            mean_diff,
            1e-2,
            f"Logits mean diff {mean_diff:.8f} exceeds threshold 1e-2",
        )

        hf_top10 = np.argsort(hf_logits)[-10:][::-1]
        pd_top10 = np.argsort(pd_logits)[-10:][::-1]
        self.assertEqual(
            hf_top10.tolist(),
            pd_top10.tolist(),
            "Top 10 tokens differ between HF and Paddle",
        )


if __name__ == "__main__":
    unittest.main()
