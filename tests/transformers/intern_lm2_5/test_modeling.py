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
    InternLM25Config,
    InternLM25ForCausalLM,
    InternLM25Tokenizer,
)
from paddleformers.transformers.intern.configuration import InternLM2Config
from paddleformers.transformers.intern.modeling import (
    InternLM2ForCausalLM,
    InternLM2ForQuestionAnswering,
    InternLM2ForSequenceClassification,
    InternLM2ForTokenClassification,
    InternLM2Model,
)
from tests.testing_utils import require_package, slow

# https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat/summary
modelscope_lm25_model_location = "Shanghai_AI_Laboratory/internlm2_5-1_8b-chat"


class TestInternLM25Config(unittest.TestCase):
    def test_config_custom_values(self):
        config = InternLM25Config(
            vocab_size=10000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=14336,
        )
        self.assertEqual(config.vocab_size, 10000)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 14336)

    def test_config_save_and_load(self):
        config = InternLM25Config(vocab_size=10000, hidden_size=4096)

        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_pretrained(temp_dir)
            loaded_config = InternLM25Config.from_pretrained(temp_dir)
            self.assertEqual(config.vocab_size, loaded_config.vocab_size)
            self.assertEqual(config.hidden_size, loaded_config.hidden_size)


class InternLM25ModelTest(unittest.TestCase):
    def setUp(self):
        self.config = InternLM25Config(
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
        model = InternLM25ForCausalLM(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.config.vocab_size, 1000)
        self.assertEqual(model.config.hidden_size, 256)

    def test_model_forward(self):
        model = InternLM25ForCausalLM(self.config)
        model.eval()

        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits
        self.assertEqual(logits.shape, [batch_size, seq_length, self.config.vocab_size])

    def test_model_generation(self):
        model = InternLM25ForCausalLM(self.config)
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
        model = InternLM25ForCausalLM(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir, save_checkpoint_format="", save_to_hf=False)

            self.assertTrue(os.path.exists(os.path.join(temp_dir, "model_state.pdparams")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "config.json")))

            loaded_model = InternLM25ForCausalLM.from_pretrained(temp_dir, load_checkpoint_format="")

            self.assertEqual(model.config.vocab_size, loaded_model.config.vocab_size)
            self.assertEqual(model.config.hidden_size, loaded_model.config.hidden_size)

    def test_chat_method(self):
        model = InternLM25ForCausalLM(self.config)
        model.eval()
        self.assertTrue(hasattr(model, "chat"))
        self.assertTrue(hasattr(model, "build_inputs"))
        self.assertTrue(hasattr(model, "stream_chat"))

    def test_model_with_attention_mask(self):
        model = InternLM25ForCausalLM(self.config)
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
        config = InternLM25Config(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=128,
            use_cache=True,
        )
        model = InternLM25ForCausalLM(config)
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


class InternLM25ConvertedTest(unittest.TestCase):
    def setUp(self):
        self._original_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype("bfloat16")

    def tearDown(self):
        paddle.set_default_dtype(self._original_dtype)

    @slow
    def test_hf_direct_load_and_inference(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is required for this test")

        paddle.set_device("gpu")
        paddle.set_default_dtype("bfloat16")

        model = InternLM25ForCausalLM.from_pretrained(
            modelscope_lm25_model_location,
            convert_from_hf=True,
            dtype="bfloat16",
            low_cpu_mem_usage=True,
            load_checkpoint_format="",
            download_hub="modelscope",
        )
        model.eval()
        tokenizer = InternLM25Tokenizer.from_pretrained(
            modelscope_lm25_model_location, load_checkpoint_format="", download_hub="modelscope"
        )

        prompt = "What are the differences between cats and dogs? Here are the three main points"
        meta_instruction = "You are a helpful assistant. Please answer in plain text without markdown."
        inputs = model.build_inputs(tokenizer, prompt, history=[], meta_instruction=meta_instruction)
        with paddle.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=128,
                use_cache=True,
                decode_strategy="greedy_search",
            )

        generated_ids = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        decoded = tokenizer.decode(generated_ids[0].numpy().tolist(), skip_special_tokens=True)
        print("\n[HF Direct Load] prompt:", prompt)
        print("[HF Direct Load] response:", decoded)

        self.assertIsNotNone(decoded)
        self.assertGreater(len(decoded.strip()), 0)


class InternLM25CompatibilityTest(unittest.TestCase):
    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        import json

        import numpy as np
        import torch
        from modelscope import AutoConfig
        from transformers import AutoModelForCausalLM

        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        cls.torch_model_path = tempfile.mkdtemp()

        config = AutoConfig.from_pretrained(modelscope_lm25_model_location, trust_remote_code=True)

        # Override with small test parameters,  accelerate calc
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
        # Set seeds for reproducibility
        import paddle
        import torch

        paddle.seed(42)
        np.random.seed(42)

        input_ids = np.random.randint(100, 200, [1, 20])

        self.torch_model.eval()
        torch_output = self.torch_model(torch.tensor(input_ids), use_cache=False)
        torch_logit = torch_output[0] if isinstance(torch_output, tuple) else torch_output.logits

        paddle_model = InternLM25ForCausalLM.from_pretrained(
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


def _make_internlm2_config(**overrides):
    config = InternLM2Config(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        max_position_embeddings=64,
        use_cache=False,
        **overrides,
    )
    config.auto_map = {"AutoModelForSequenceClassification": "intern_lm2_5"}
    return config


class InternLM2ProxyEmbeddingTest(unittest.TestCase):
    def setUp(self):
        self.config = _make_internlm2_config()

    def _get_new_embedding(self, vocab_size=2000):
        return paddle.nn.Embedding(vocab_size, self.config.hidden_size)

    def test_model_get_input_embeddings(self):
        model = InternLM2Model(self.config)
        emb = model.get_input_embeddings()
        self.assertIsNotNone(emb)
        self.assertEqual(emb.weight.shape, [self.config.vocab_size, self.config.hidden_size])

    def test_model_set_input_embeddings(self):
        model = InternLM2Model(self.config)
        new_emb = self._get_new_embedding()
        model.set_input_embeddings(new_emb)
        emb = model.get_input_embeddings()
        self.assertIs(emb, new_emb)
        self.assertEqual(emb.weight.shape, [2000, self.config.hidden_size])

    def test_for_causal_lm_get_input_embeddings(self):
        model = InternLM2ForCausalLM(self.config)
        emb = model.get_input_embeddings()
        self.assertIsNotNone(emb)
        self.assertEqual(emb.weight.shape, [self.config.vocab_size, self.config.hidden_size])

    def test_for_causal_lm_set_input_embeddings(self):
        model = InternLM2ForCausalLM(self.config)
        new_emb = self._get_new_embedding()
        model.set_input_embeddings(new_emb)
        emb = model.get_input_embeddings()
        self.assertIs(emb, new_emb)

    def test_for_causal_lm_get_output_embeddings(self):
        model = InternLM2ForCausalLM(self.config)
        out = model.get_output_embeddings()
        self.assertIsNotNone(out)
        self.assertEqual(out.weight.shape, [self.config.hidden_size, self.config.vocab_size])

    def test_for_causal_lm_set_output_embeddings(self):
        model = InternLM2ForCausalLM(self.config)
        new_head = paddle.nn.Linear(self.config.hidden_size, 3000, bias_attr=False)
        model.set_output_embeddings(new_head)
        out = model.get_output_embeddings()
        self.assertIs(out, new_head)
        self.assertEqual(out.weight.shape, [self.config.hidden_size, 3000])

    def test_for_sequence_classification_get_input_embeddings(self):
        config = _make_internlm2_config(num_labels=2)
        model = InternLM2ForSequenceClassification(config)
        emb = model.get_input_embeddings()
        self.assertIsNotNone(emb)
        self.assertEqual(emb.weight.shape, [config.vocab_size, config.hidden_size])

    def test_for_sequence_classification_set_input_embeddings(self):
        config = _make_internlm2_config(num_labels=2)
        model = InternLM2ForSequenceClassification(config)
        new_emb = self._get_new_embedding()
        model.set_input_embeddings(new_emb)
        emb = model.get_input_embeddings()
        self.assertIs(emb, new_emb)

    def test_for_question_answering_get_input_embeddings(self):
        config = _make_internlm2_config(num_labels=2)
        model = InternLM2ForQuestionAnswering(config)
        emb = model.get_input_embeddings()
        self.assertIsNotNone(emb)
        self.assertEqual(emb.weight.shape, [config.vocab_size, config.hidden_size])

    def test_for_question_answering_set_input_embeddings(self):
        config = _make_internlm2_config(num_labels=2)
        model = InternLM2ForQuestionAnswering(config)
        new_emb = self._get_new_embedding()
        model.set_input_embeddings(new_emb)
        emb = model.get_input_embeddings()
        self.assertIs(emb, new_emb)

    def test_for_token_classification_get_input_embeddings(self):
        config = _make_internlm2_config(num_labels=5)
        model = InternLM2ForTokenClassification(config)
        emb = model.get_input_embeddings()
        self.assertIsNotNone(emb)
        self.assertEqual(emb.weight.shape, [config.vocab_size, config.hidden_size])

    def test_for_token_classification_set_input_embeddings(self):
        config = _make_internlm2_config(num_labels=5)
        model = InternLM2ForTokenClassification(config)
        new_emb = self._get_new_embedding()
        model.set_input_embeddings(new_emb)
        emb = model.get_input_embeddings()
        self.assertIs(emb, new_emb)

    def test_resize_token_embeddings_on_proxy(self):
        model = InternLM2ForCausalLM(self.config)
        new_vocab = 2000
        model.resize_token_embeddings(new_vocab)
        emb = model.get_input_embeddings()
        self.assertEqual(emb.weight.shape[0], new_vocab)


if __name__ == "__main__":
    unittest.main()
