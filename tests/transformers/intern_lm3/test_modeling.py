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

import paddle

from paddleformers.transformers import (
    InternLM3Config,
    InternLM3ForCausalLM,
    InternLM3Tokenizer,
)
from tests.testing_utils import slow


class TestInternLM3Config(unittest.TestCase):
    def test_config_custom_values(self):
        config = InternLM3Config(
            vocab_size=10000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=11008,
        )
        self.assertEqual(config.vocab_size, 10000)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 11008)

    def test_config_save_and_load(self):
        config = InternLM3Config(vocab_size=10000, hidden_size=4096)

        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_pretrained(temp_dir)
            loaded_config = InternLM3Config.from_pretrained(temp_dir)
            self.assertEqual(config.vocab_size, loaded_config.vocab_size)
            self.assertEqual(config.hidden_size, loaded_config.hidden_size)


class InternLM3ModelTest(unittest.TestCase):
    def setUp(self):
        self.config = InternLM3Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
            use_cache=False,
        )

    def test_model_initialization(self):
        model = InternLM3ForCausalLM(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config.vocab_size, 1000)
        self.assertEqual(model.config.hidden_size, 256)

    def test_model_forward(self):
        model = InternLM3ForCausalLM(self.config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits
        self.assertEqual(logits.shape, [batch_size, seq_length, self.config.vocab_size])

    def test_model_generation(self):
        model = InternLM3ForCausalLM(self.config)
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

        self.assertIsNotNone(generated_ids)
        assert generated_ids is not None
        self.assertGreaterEqual(generated_ids.shape[1], 10)
        self.assertLessEqual(generated_ids.shape[1], 20)

    def test_model_save_and_load(self):
        model = InternLM3ForCausalLM(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, save_checkpoint_format="", save_to_hf=False)

            self.assertTrue(os.path.exists(os.path.join(temp_dir, "model_state.pdparams")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "config.json")))

            loaded_model = InternLM3ForCausalLM.from_pretrained(temp_dir, load_checkpoint_format="")

            self.assertEqual(model.config.vocab_size, loaded_model.config.vocab_size)
            self.assertEqual(model.config.hidden_size, loaded_model.config.hidden_size)

    def test_model_with_attention_mask(self):
        model = InternLM3ForCausalLM(self.config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])
        attention_mask = paddle.ones([batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        logits = outputs.logits
        self.assertEqual(logits.shape, [batch_size, seq_length, self.config.vocab_size])

    def test_model_with_past_key_values(self):
        config = InternLM3Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
            use_cache=True,
        )
        model = InternLM3ForCausalLM(config)
        model.eval()

        batch_size = 1
        seq_length = 5
        input_ids = paddle.randint(0, config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values
            next_input_ids = paddle.randint(0, config.vocab_size, [batch_size, 1])
            outputs = model(
                input_ids=next_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        self.assertIsNotNone(outputs.past_key_values)


class InternLM3ConvertedWeightTest(unittest.TestCase):
    def setUp(self):
        self._original_dtype: str = paddle.get_default_dtype()
        paddle.set_default_dtype("bfloat16")  # type: ignore[arg-type]

    def tearDown(self):
        paddle.set_default_dtype(self._original_dtype)  # type: ignore[arg-type]

    @slow
    def test_paddle_model_load_and_infer(self):
        hf_model_path = "Shanghai_AI_Laboratory/internlm3-8b-instruct"
        paddle.device.set_device("gpu")
        model = InternLM3ForCausalLM.from_pretrained(
            hf_model_path,
            download_hub="modelscope",
            convert_from_hf=True,
            dtype="bfloat16",
            low_cpu_mem_usage=True,
            load_checkpoint_format="",
        )
        model.eval()
        tokenizer = InternLM3Tokenizer.from_pretrained(hf_model_path, download_hub="modelscope")
        prompt = "What are the main differences between cats and dogs? List 3 points."
        meta_instruction = "You are a helpful AI assistant. Please answer in English."
        chat_inputs = model.build_inputs(tokenizer, prompt, history=[], meta_instruction=meta_instruction)
        print("\n" + "=" * 80)
        print(f"Prompt: {prompt}")
        print(f"Meta Instruction: {meta_instruction}")
        print(f"Input Length: {chat_inputs['input_ids'].shape[1]} tokens")
        self.assertIsNotNone(chat_inputs)
        self.assertIn("input_ids", chat_inputs)
        self.assertGreater(chat_inputs["input_ids"].shape[1], 0, "Input should not be empty")
        with paddle.no_grad():
            outputs = model(
                input_ids=chat_inputs["input_ids"],
                attention_mask=chat_inputs.get("attention_mask"),
                return_dict=True,
            )
            self.assertIsNotNone(outputs.logits)

        with paddle.no_grad():
            out = model.generate(
                input_ids=chat_inputs["input_ids"],
                attention_mask=chat_inputs.get("attention_mask"),
                max_new_tokens=128,
                use_cache=True,
                decode_strategy="sampling",
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.005,
            )
        if isinstance(out, (list, tuple)):
            out = out[0]
        input_length = chat_inputs["input_ids"].shape[1]
        output_ids = out[0][input_length:]
        output_text = tokenizer.decode(output_ids.squeeze().numpy().tolist(), skip_special_tokens=True)

        print(
            f"Output Length: {out.shape[1]} tokens (input: {input_length}, generated: {out.shape[1] - input_length})"
        )
        print("-" * 80)
        print(output_text if output_text else "(no output)")
        print("=" * 80 + "\n")

        self.assertIsNotNone(out)
        self.assertGreater(out.shape[1], 0, "Output should not be empty")
        self.assertGreater(len(output_text.strip()), 10, "Generated output should have meaningful content")


if __name__ == "__main__":
    unittest.main()
