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

import paddle

from paddleformers.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    MiniCPM3Config,
    MiniCPM3ForCausalLM,
)


def tiny_minicpm3_config(**kwargs):
    config_kwargs = dict(
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        q_lora_rank=16,
        kv_lora_rank=8,
        v_head_dim=8,
        max_position_embeddings=64,
        scale_emb=1.0,
        dim_model_base=32,
        scale_depth=1.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=True,
        use_flash_attention=False,
        fuse_rms_norm=False,
    )
    config_kwargs.update(kwargs)
    return MiniCPM3Config(**config_kwargs)


class MiniCPM3ModelingTest(unittest.TestCase):
    def test_tied_lm_head_aoa_config(self):
        config = tiny_minicpm3_config(tie_word_embeddings=True)

        aoa_statements = MiniCPM3ForCausalLM._gen_aoa_config(config)["aoa_statements"]
        inv_aoa_statements = MiniCPM3ForCausalLM._gen_inv_aoa_config(config)["aoa_statements"]

        self.assertIn("model.embed_tokens.weight -> lm_head.weight", aoa_statements)
        self.assertIn("lm_head.weight -> _", inv_aoa_statements)

    def test_causal_lm_forward_and_loss(self):
        config = tiny_minicpm3_config()
        model = MiniCPM3ForCausalLM(config)
        input_ids = paddle.randint(low=3, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")
        labels = input_ids.clone()
        labels[:, :3] = -100

        outputs = model(input_ids=input_ids, labels=labels, return_dict=True)

        self.assertEqual(list(outputs.logits.shape), [2, 8, config.vocab_size])
        self.assertIsNotNone(outputs.loss)

    def test_from_pretrained_and_auto_model(self):
        config = tiny_minicpm3_config()
        model = MiniCPM3ForCausalLM(config)
        input_ids = paddle.randint(low=3, high=config.vocab_size - 1, shape=[2, 8], dtype="int64")

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            class_loaded = MiniCPM3ForCausalLM.from_pretrained(tmpdir)
            class_outputs = class_loaded(input_ids=input_ids, return_dict=True)
            self.assertEqual(list(class_outputs.logits.shape), [2, 8, config.vocab_size])

            auto_loaded = AutoModelForCausalLM.from_pretrained(tmpdir)
            self.assertEqual(type(auto_loaded).__name__, "MiniCPM3ForCausalLM")
            auto_outputs = auto_loaded(input_ids=input_ids, return_dict=True)
            self.assertEqual(list(auto_outputs.logits.shape), [2, 8, config.vocab_size])

    def test_auto_config_with_model_type(self):
        config = tiny_minicpm3_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            config_path = os.path.join(tmpdir, "config.json")

            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict.pop("architectures", None)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f)

            auto_config = AutoConfig.from_pretrained(tmpdir)

            self.assertEqual(type(auto_config).__name__, "MiniCPM3Config")
            self.assertEqual(auto_config.model_type, "minicpm3")

    def test_chat_uses_paddle_tensors(self):
        class FakeTokenizer:
            def __init__(self):
                self.return_tensors = None

            def apply_chat_template(self, history, tokenize=False, add_generation_prompt=True):
                return "hello"

            def __call__(self, text, return_tensors=None):
                self.return_tensors = return_tensors
                return {"input_ids": paddle.to_tensor([[1, 2, 3]], dtype="int64")}

            def decode(self, outputs):
                return "ok"

        config = tiny_minicpm3_config()
        model = MiniCPM3ForCausalLM(config)
        tokenizer = FakeTokenizer()

        def fake_generate(**kwargs):
            self.assertIsInstance(kwargs["input_ids"], paddle.Tensor)
            self.assertEqual(str(kwargs["input_ids"].place), str(paddle.to_tensor([0]).place))
            return paddle.to_tensor([[1, 2, 3, 4, 2]], dtype="int64")

        model.generate = fake_generate
        response, history = model.chat(tokenizer, query="hi")

        self.assertEqual(tokenizer.return_tensors, "pd")
        self.assertEqual(response, "ok")
        self.assertEqual(history[-1], {"role": "assistant", "content": "ok"})


if __name__ == "__main__":
    unittest.main()
