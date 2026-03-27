# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddleformers.transformers import (
    InternLM3Config,
    InternLM3ForCausalLM,
    InternLM3Tokenizer,
)
from tests.testing_utils import slow


# config常规测试
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


# model层的常规测试
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


"""
！！！较为消耗时间，直接加载HF原生格式权重并推理，有slow标记

也支持直接从 转换过的格式进行推理，转换后的paddle权重地址 https://aistudio.baidu.com/modelsdetail/45407

仅供本地评估时使用
"""


class InternLM3ConvertedWeightTest(unittest.TestCase):
    def setUp(self):
        self._original_dtype: str = paddle.get_default_dtype()
        paddle.set_default_dtype("bfloat16")  # type: ignore[arg-type]

    def tearDown(self):
        paddle.set_default_dtype(self._original_dtype)  # type: ignore[arg-type]

    @slow
    def test_paddle_model_load_and_infer(self):
        hf_model_path = "internlm/internlm3-8b-instruct"
        paddle.device.set_device("gpu")
        model = InternLM3ForCausalLM.from_pretrained(
            hf_model_path,
            download_hub="huggingface",
            convert_from_hf=True,
            dtype="bfloat16",
            low_cpu_mem_usage=True,
            load_checkpoint_format="",
        )
        model.eval()
        tokenizer = InternLM3Tokenizer.from_pretrained(hf_model_path)
        prompt = "猫和狗的区别是什么，列出主要的3点"
        meta_instruction = "你是一个有用的AI助手，请用中文回答。"
        chat_inputs = model.build_inputs(tokenizer, prompt, history=[], meta_instruction=meta_instruction)
        print("\n" + "=" * 80)
        print("InternLM3 模型推理测试")
        print("=" * 80)
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
        print("模型生成内容:")
        print(output_text if output_text else "(无输出)")
        print("=" * 80 + "\n")

        self.assertIsNotNone(out)
        self.assertGreater(out.shape[1], 0, "Output should not be empty")
        self.assertGreater(len(output_text.strip()), 10, "Generated output should have meaningful content")


"""
测试 torch (safetensors) 和 paddle 的对齐;

因为lm3.5版本有点特殊，用到了tranformers的一些tranformers 4.53.0 版本之后的特性，但是在 5.x 版本又没有了

导致在 4.53.0-5.x 区间之外的tranformers 运行的时候会有各种报错，即使使用 monkey patch 等手段处理了 modeling_internlm3.py 但是原版推理出的仍旧不正确

所以，当前的对齐代码是分离的，先在 4.53.0 的环境里跑出 结果，然后再和paddle版本对比

"""


class InternLM3CompatibilityTest(unittest.TestCase):
    """测试 Paddle 模型推理与 Transformers 参考值对齐"""

    MINI_MODEL_PATH = "learncat/internlm3-8b-instruct-mini-raw"

    # 以下参考值由 tf4.53 环境的 transformers 生成，固定随机种子42和固定输入
    # 参考代码（不在当前环境执行）:
    # ------------------------------------------------------------
    # import torch
    # import numpy as np
    # from transformers import AutoModelForCausalLM
    # np.random.seed(42)
    # torch.manual_seed(42)
    # input_ids = np.array([[100, 200, 300, 400, 500, 600, 700, 800, 900, 950]])
    # model = AutoModelForCausalLM.from_pretrained(
    #     "/mnt/caoyuanye/llm/internlm/internlm3-8b-instruct-mini",
    #     trust_remote_code=True, torch_dtype=torch.float32
    # ).cpu().eval()
    # torch_input = torch.from_numpy(input_ids).long()
    # with torch.no_grad():
    #     logits = model(torch_input).logits[0, -1, :].cpu().numpy()
    # print(logits[:20])
    # print(np.argsort(logits)[-10:][::-1])
    # ------------------------------------------------------------
    REF_LOGITS_FIRST_20 = [
        -0.144880,
        -0.648041,
        0.386456,
        0.213346,
        0.771256,
        0.620751,
        0.073640,
        -0.021458,
        -0.764580,
        0.317350,
        -0.025523,
        -0.056741,
        -0.671094,
        -0.187901,
        0.286278,
        -0.182251,
        0.849138,
        0.340502,
        0.324327,
        0.609751,
    ]
    REF_TOP10_TOKEN_IDS = [72267, 94359, 95067, 121546, 19719, 125351, 74467, 115313, 87550, 24000]

    # @classmethod
    # def setUpClass(cls) -> None:
    #     if not os.path.exists(cls.MINI_MODEL_PATH):
    #         cls.skipTest(f"Mini model not found at {cls.MINI_MODEL_PATH}")

    def test_torch_paddle_model_alignment(self):
        """验证 Paddle 输出与 Transformers 参考值对齐"""
        np.random.seed(42)
        paddle.seed(42)

        input_ids = np.array([[100, 200, 300, 400, 500, 600, 700, 800, 900, 950]])
        paddle_input = paddle.to_tensor(input_ids, dtype="int64")

        paddle_model = InternLM3ForCausalLM.from_pretrained(self.MINI_MODEL_PATH, dtype=paddle.float32)
        paddle_model.eval()

        with paddle.no_grad():
            paddle_logits = paddle_model(paddle_input)[0][0, -1, :].cpu().numpy()

        # 指标1: 前20个logits值对齐 < 1e-2
        paddle_first20 = paddle_logits[:20]
        max_diff = np.max(np.abs(paddle_first20 - np.array(self.REF_LOGITS_FIRST_20)))
        print(f"paddle and transformer models differ: {max_diff}")
        self.assertLess(max_diff, 1e-2, f"First 20 logits diff={max_diff}")

        # 指标2: Top-10 token 相同
        paddle_top10 = set(np.argsort(paddle_logits)[-10:])
        ref_top10 = set(self.REF_TOP10_TOKEN_IDS)
        self.assertEqual(paddle_top10, ref_top10, "Top-10 tokens mismatch")
        print("paddle and transformer has same 10 tokens id")


if __name__ == "__main__":
    unittest.main()
