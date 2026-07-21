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

import os
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import (
    InternLM2Config,
    InternLM2ForCausalLM,
    InternLM2Tokenizer,
)
from tests.testing_utils import require_gpu, require_package, slow

MODEL_PATH = "Shanghai_AI_Laboratory/internlm2-chat-7b"


class TestInternLM2Config(unittest.TestCase):
    def test_config_initialization(self):
        config = InternLM2Config()
        self.assertEqual(config.vocab_size, 92550)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)

    def test_config_custom_values(self):
        config = InternLM2Config(
            vocab_size=92544,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=14336,
        )
        self.assertEqual(config.vocab_size, 92544)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 14336)

    def test_config_save_and_load(self):
        config = InternLM2Config(vocab_size=92544, hidden_size=4096)

        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_pretrained(temp_dir)
            loaded_config = InternLM2Config.from_pretrained(temp_dir)
            self.assertEqual(config.vocab_size, loaded_config.vocab_size)
            self.assertEqual(config.hidden_size, loaded_config.hidden_size)


class InternLM2ModelTest(unittest.TestCase):
    def setUp(self):
        self.config = InternLM2Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
            use_cache=False,
        )

    def test_model_initialization(self):
        model = InternLM2ForCausalLM(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config.vocab_size, 1000)
        self.assertEqual(model.config.hidden_size, 256)

    def test_flash_attn_helpers_roundtrip(self):
        from paddleformers.transformers.intern_lm2.modeling import (
            pad_input,
            unpad_input,
        )

        batch, seqlen, hidden = 2, 6, 8
        # mask out the last 2 tokens of sequence 0 -> padding present
        attention_mask = paddle.to_tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype="int32")
        hidden_states = paddle.randn([batch, seqlen, hidden])

        unpadded, indices, cu_seqlens, max_seqlen = unpad_input(hidden_states, attention_mask)
        # total non-padding tokens = 4 + 6 = 10
        self.assertEqual(unpadded.shape[0], 10)
        self.assertEqual(unpadded.shape[1], hidden)
        self.assertEqual(max_seqlen, 6)

        repadded = pad_input(unpadded, indices, batch, seqlen)
        self.assertEqual(repadded.shape, [batch, seqlen, hidden])
        # padded positions should be zero; non-padded positions should match the input
        valid = attention_mask.astype("bool").unsqueeze(-1)
        self.assertTrue(
            paddle.all(paddle.where(valid, repadded == hidden_states, paddle.ones_like(valid, dtype="bool")))
        )
        self.assertTrue(paddle.all(paddle.where(~valid, repadded == 0, paddle.ones_like(valid, dtype="bool"))))

    @require_gpu(min_gpus=1)
    def test_flash_attention_2_with_padding_mask(self):
        from paddleformers.transformers.intern_lm2.modeling import has_flash_attn

        if not has_flash_attn:
            self.skipTest("flash_attention is not available")

        config = InternLM2Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            max_position_embeddings=128,
            use_cache=False,
            attn_implementation="flash_attention_2",
        )
        model = InternLM2ForCausalLM(config)
        model.eval()

        # batch with padding: seq 0 shorter than seq 1
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 0, 0], [1, 2, 3, 4, 5, 6]], dtype="int64")
        attention_mask = paddle.to_tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype="int64")

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        logits = outputs.logits
        self.assertEqual(logits.shape, [2, 6, config.vocab_size])
        self.assertTrue(paddle.isfinite(logits).all().item())

    def test_model_forward(self):
        model = InternLM2ForCausalLM(self.config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits
        self.assertEqual(logits.shape, [batch_size, seq_length, self.config.vocab_size])

    def test_model_generation(self):
        model = InternLM2ForCausalLM(self.config)
        model.eval()

        input_ids = paddle.randint(0, self.config.vocab_size, [1, 5])

        with paddle.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_length=20,
                min_length=10,
                use_cache=False,
            )

        if isinstance(generated_ids, tuple):
            generated_ids = generated_ids[0]

        self.assertGreaterEqual(generated_ids.shape[1], 10)
        self.assertLessEqual(generated_ids.shape[1], 20)

    def test_model_save_and_load(self):
        model = InternLM2ForCausalLM(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, save_checkpoint_format="", save_to_hf=False)

            self.assertTrue(os.path.exists(os.path.join(temp_dir, "model_state.pdparams")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "config.json")))

            loaded_model = InternLM2ForCausalLM.from_pretrained(temp_dir, load_checkpoint_format="")

            self.assertEqual(model.config.vocab_size, loaded_model.config.vocab_size)
            self.assertEqual(model.config.hidden_size, loaded_model.config.hidden_size)

    @slow
    def test_auto_model_load(self):
        from paddleformers.transformers.auto.modeling import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_checkpoint_format="", download_hub="modelscope")
        self.assertIsNotNone(model)

    @slow
    def test_paddle_hello(self):
        model = InternLM2ForCausalLM.from_pretrained(MODEL_PATH, load_checkpoint_format="", download_hub="modelscope")
        tokenizer = InternLM2Tokenizer.from_pretrained(
            MODEL_PATH, load_checkpoint_format="", download_hub="modelscope"
        )

        model.eval()

        prompt = "What is the difference between cats and dogs?"
        chat_inputs = model.build_inputs(
            tokenizer, prompt, history=[], meta_instruction="You are a helpful assistant."
        )
        input_ids = chat_inputs["input_ids"].to("gpu")
        attention_mask = chat_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to("gpu")

        with paddle.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                use_cache=True,
                decode_strategy="greedy_search",
                attention_mask=attention_mask,
            )

        seq = out[0] if isinstance(out, (list, tuple)) else out
        token_ids = seq.numpy().tolist()[0]
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print("reply:", decoded)
        self.assertGreater(len(decoded.strip()), 0)

    @slow
    def test_inference_with_paddle_model(self):
        model = InternLM2ForCausalLM.from_pretrained(MODEL_PATH, load_checkpoint_format="", download_hub="modelscope")
        model.eval()

        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]

        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[2], model.config.vocab_size)
        self.assertTrue(paddle.isfinite(output).all().item())


class InternLM2CompatibilityTest(unittest.TestCase):
    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        import json

        import torch
        from modelscope import AutoConfig
        from transformers import AutoModelForCausalLM

        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        cls.torch_model_path = tempfile.mkdtemp()

        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

        config.hidden_size = 128
        config.intermediate_size = 384
        config.num_hidden_layers = 4
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.vocab_size = 10000
        config.max_position_embeddings = 128

        cls.torch_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        torch.save(cls.torch_model.state_dict(), f"{cls.torch_model_path}/pytorch_model.bin")

        config_dict = config.to_dict()
        for key in ["_commit_hash", "_name_or_path"]:
            config_dict.pop(key, None)

        with open(f"{cls.torch_model_path}/config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    @require_package("transformers", "torch")
    def test_intern_converter(self):
        import paddle
        import torch

        paddle.seed(42)
        np.random.seed(42)

        input_ids = np.random.randint(100, 200, [1, 20])

        self.torch_model.eval()
        torch_output = self.torch_model(torch.tensor(input_ids), use_cache=False)
        torch_logit = torch_output[0] if isinstance(torch_output, tuple) else torch_output.logits

        paddle_model = InternLM2ForCausalLM.from_pretrained(
            self.torch_model_path, convert_from_hf=True, load_checkpoint_format=""
        )
        paddle_model.eval()
        paddle_logit = paddle_model(paddle.to_tensor(input_ids), use_cache=False)[0]

        paddle_out = paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy()
        torch_out = torch_logit.detach().cpu().reshape([-1])[:9].float().numpy()
        max_diff = np.max(np.abs(paddle_out - torch_out))
        print(f"\nMax diff: {max_diff}")

        paddle_token_ids = paddle.argmax(paddle_logit, axis=-1).cpu().numpy()[0][:10]
        torch_token_ids = torch.argmax(torch_logit, dim=-1).cpu().numpy()[0][:10]
        print(f"Paddle token ids: {paddle_token_ids}")
        print(f"Torch token ids:  {torch_token_ids}")
        self.assertTrue(
            np.array_equal(paddle_token_ids, torch_token_ids),
            f"Token ids mismatch: paddle={paddle_token_ids}, torch={torch_token_ids}",
        )

        self.assertTrue(
            np.allclose(paddle_out, torch_out, atol=1e-2, rtol=1e-2), f"Max diff {max_diff} exceeds tolerance"
        )


if __name__ == "__main__":
    unittest.main()
