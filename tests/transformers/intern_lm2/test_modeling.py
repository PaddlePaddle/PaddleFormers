# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from paddleformers.transformers import InternLM2ForCausalLM, InternLM2Config, InternLM2Tokenizer
from tests.testing_utils import require_package, slow


class TestInternLM2Config(unittest.TestCase):

    def test_config_initialization(self):
        config = InternLM2Config()
        self.assertEqual(config.vocab_size, 103168)
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
            model.save_pretrained(temp_dir, save_checkpoint_format="", save_to_hf = False)

            self.assertTrue(os.path.exists(os.path.join(temp_dir, "model_state.pdparams")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "config.json")))

            loaded_model = InternLM2ForCausalLM.from_pretrained(temp_dir, load_checkpoint_format="")

            self.assertEqual(model.config.vocab_size, loaded_model.config.vocab_size)
            self.assertEqual(model.config.hidden_size, loaded_model.config.hidden_size)

    @slow
    def test_auto_model_load(self):
        model_repo_id = "learncat/internlm2_tiny_paddle"
        from paddleformers.transformers.auto.modeling import AutoModelForCausalLM
        tiny_torch_model = AutoModelForCausalLM.from_pretrained(model_repo_id, load_checkpoint_format="")
        self.assertIsNotNone(tiny_torch_model, "AutoModelForCausalLM load should not none")

    @slow
    def test_inference_with_torch_model(self):
        raw_model_repo_id = "learncat/internlm2_tiny_raw"

        from transformers import AutoModelForCausalLM
        tiny_torch_model = AutoModelForCausalLM.from_pretrained(raw_model_repo_id, trust_remote_code=True)
        tiny_torch_model.eval()



        import torch
        torch_input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        torch_attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with torch.no_grad():
            output = tiny_torch_model(torch_input_ids, attention_mask=torch_attention_mask)[0]


        expected_slice = torch.tensor(
        [[
            [-0.01625147, -0.10887568, -0.27600563],
            [-0.01467928, 0.58325601, -0.08758156],
            [-0.05791061, 0.50822973, -0.39710030]
            ]],
            dtype=output.dtype,
        )
        rs = torch.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4)
        self.assertTrue(rs)

    @slow
    def test_paddle_hello(self):
        model_path = "learncat/internlm2_7b_paddle"
        model = InternLM2ForCausalLM.from_pretrained(
            model_path,
            load_checkpoint_format ="",
            convert_from_hf = False
        )
        device = "gpu"
        tokenizer = InternLM2Tokenizer.from_pretrained(model_path, load_checkpoint_format="")

        model.eval()

        prompt = "猫和狗的区别是什么？"
        chat_inputs = model.build_inputs(tokenizer, prompt, history=[], meta_instruction="You are a helpful assistant.")
        input_ids = chat_inputs["input_ids"].to(device)
        attention_mask = chat_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

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
        model_repo_id = "learncat/internlm2_tiny_paddle"
        model = InternLM2ForCausalLM.from_pretrained(model_repo_id, load_checkpoint_format="", convert_from_hf = False)
        model.eval()

        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]

        expected_shape = [1, 11, 92544]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = paddle.to_tensor(
            [
                [
                    [-0.01625147, -0.10887568, -0.27600563],
                    [-0.01467928, 0.58325601, -0.08758156],
                    [-0.05791061, 0.50822973, -0.39710030]
                ]
            ],
            dtype=output.dtype,
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


class InternLM2CompatibilityTest(unittest.TestCase):
    tiny_torch_model_path = "learncat/internlm2_tiny_raw"
    tiny_torch_model = None

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import AutoModelForCausalLM
        cls.tiny_torch_model = AutoModelForCausalLM.from_pretrained(cls.tiny_torch_model_path, trust_remote_code=True)
        cls.tiny_torch_model.eval()

    @classmethod
    def tearDownClass(cls) -> None:
        pass


    @require_package("transformers", "torch")
    def test_intern_converter(self):
        input_ids = np.random.randint(100, 200, [1, 20])
        paddle_model = InternLM2ForCausalLM.from_pretrained(self.tiny_torch_model_path, convert_from_hf=True, load_checkpoint_format="")
        paddle_model.eval()
        paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

        import torch
        torch_logit = self.tiny_torch_model(torch.tensor(input_ids), return_dict=False)[0]

        self.assertTrue(
            np.allclose(
                paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                rtol=1e-2,
            )
        )

    @require_package("transformers", "torch")
    def test_llama_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            input_ids = np.random.randint(100, 200, [1, 20])
            import torch

            self.tiny_torch_model.save_pretrained(tempdir)
            torch_logit = self.tiny_torch_model(torch.tensor(input_ids), return_dict=False)[0]


            paddle_model = InternLM2ForCausalLM.from_pretrained(tempdir, convert_from_hf=True, load_checkpoint_format="")
            paddle_model.eval()
            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].numpy(),
                    rtol=1e-2,
                )
            )


if __name__ == "__main__":
    unittest.main()
