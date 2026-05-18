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

import tempfile
import unittest

import paddle

from paddleformers.transformers import (
    StableLmConfig,
    StableLmForCausalLM,
    StableLmModel,
    StableLmTokenizer,
)
from tests.testing_utils import gpu_device_initializer, require_package, slow


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


class StableLmTokenizerTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="StableLmTokenizerTest")
    @require_package("transformers")
    def test_tokenizer_encode_decode(self):
        tokenizer = StableLmTokenizer.from_pretrained(
            "stabilityai/stable-code-3b", trust_remote_code=False, download_hub="modelscope"
        )
        self.assertEqual(tokenizer.bos_token_id, 0)
        self.assertEqual(tokenizer.eos_token_id, 0)

        text = "def hello_world():"
        tokens = tokenizer.encode(text)
        self.assertGreater(len(tokens), 0)
        decoded = tokenizer.decode(tokens)
        self.assertIsInstance(decoded, str)

    @gpu_device_initializer(log_prefix="StableLmTokenizerTest")
    @require_package("transformers")
    def test_tokenizer_special_tokens(self):
        tokenizer = StableLmTokenizer.from_pretrained(
            "stabilityai/stable-code-3b", trust_remote_code=False, download_hub="modelscope"
        )
        self.assertIsNotNone(tokenizer.bos_token)
        self.assertIsNotNone(tokenizer.eos_token)
        self.assertGreaterEqual(tokenizer.vocab_size, 50000)


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
