# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from tests.testing_utils import require_package, slow

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

_MODEL_HF_ID = "apple/OpenELM-1_1B-Instruct"
_TOKENIZER_ID = "hf-internal-testing/llama-tokenizer"
_PROMPT_DIFF = "Hello, how are you today?"
_PROMPT_QUESTION = "What is the capital of China?"
_SEED = 42
# 最大对比生成的前10个token
_NUM_GEN_TOKENS = 10
# logits最大diff 阈值
_DIFF_THRESHOLD = 1e-2
# 小型单元测试案例
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


class TestOpenELMModel(unittest.TestCase):
    BATCH = 2
    SEQ = 10

    def setUp(self):
        from paddleformers.transformers import OpenELMConfig, OpenELMModel

        self.config = OpenELMConfig(**SMALL_CONFIG)
        self.model = OpenELMModel(self.config)
        self.model.eval()

    def test_model_from_config(self):
        """测试从配置创建模型"""
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


class TestOpenELMForCausalLM(unittest.TestCase):
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
            self.assertIsNone(model.lm_head)

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
        # 强制cpu，方便处理数据
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

        input_ids = paddle.randint(0, config.vocab_size, [1, 5])
        with paddle.no_grad():
            output = model(input_ids=input_ids, return_dict=True)

        self.assertEqual(list(output.logits.shape), [1, 5, config.vocab_size])


class TestOpenELMToken(unittest.TestCase):
    @require_package("transformers")
    def test_tokenizer_load(self):
        from paddleformers.transformers import OpenELMTokenizer

        tokenizer = OpenELMTokenizer.from_pretrained(_TOKENIZER_ID, download_hub="huggingface")
        self.assertIsNotNone(tokenizer)

    @require_package("transformers")
    def test_tokenizer_encode_decode(self):
        from paddleformers.transformers import OpenELMTokenizer

        tokenizer = OpenELMTokenizer.from_pretrained(_TOKENIZER_ID, download_hub="huggingface")

        text = "Hello, how are you today?"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)

        self.assertIsInstance(encoded, list)
        self.assertGreater(len(encoded), 0)
        self.assertEqual(decoded.strip(), text)

    @require_package("transformers")
    def test_tokenizer_batch_encode(self):
        from paddleformers.transformers import OpenELMTokenizer

        tokenizer = OpenELMTokenizer.from_pretrained(_TOKENIZER_ID, download_hub="huggingface")

        tokenizer.pad_token = tokenizer.eos_token

        texts = [
            "Hello, how are you today?",
            "What is the difference between cats and dogs?",
        ]
        encoded = tokenizer(texts, padding=True, truncation=True)
        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)
        self.assertEqual(len(encoded["input_ids"]), len(texts))

    @require_package("transformers")
    def test_tokenizer_with_model(self):
        from paddleformers.transformers import (
            OpenELMConfig,
            OpenELMForCausalLM,
            OpenELMTokenizer,
        )

        tokenizer = OpenELMTokenizer.from_pretrained(_TOKENIZER_ID, download_hub="huggingface")

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


class TestOpenELMSaveLoad(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="openelm_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

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
            "保存后重新加载的模型输出不一致",
        )


class TestOpenELMPaddleInference(unittest.TestCase):
    @slow
    @require_package("transformers")
    def test_paddle_inference_chinese(self):
        from paddleformers.transformers import OpenELMForCausalLM, OpenELMTokenizer

        paddle.seed(_SEED)
        paddle.set_device("gpu")

        tokenizer = OpenELMTokenizer.from_pretrained(_TOKENIZER_ID, download_hub="huggingface")

        model = OpenELMForCausalLM.from_pretrained(
            _MODEL_HF_ID,
            convert_from_hf=True,
            load_checkpoint_format="",
            download_hub="huggingface",
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
        print(f"[Paddle Inference] 生成文本: {repr(gen_text)}")
        print(f"[Paddle Inference] 完整文本: {repr(full_text)}")
        print("=" * 60)

        self.assertGreater(len(gen_ids), 0, "模型没有生成任何 token")
        self.assertGreater(len(gen_text.strip()), 0, "模型生成的文本为空")

        self.assertIn(
            "beijing",
            full_text.lower(),
            "模型输出应包含 beijing 信息",
        )

        printable_chars = sum(1 for c in gen_text if c.isprintable() or c.isspace())
        printable_ratio = printable_chars / len(gen_text) if len(gen_text) > 0 else 0
        self.assertGreater(printable_ratio, 0.5, "输出包含过多不可打印字符，可能是乱码")


class TestOpenELMDiffAlignment(unittest.TestCase):
    @slow
    @require_package("transformers", "torch")
    def test_diff_alignment(self):
        import json
        import sys

        import numpy as np
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        from paddleformers.transformers import OpenELMForCausalLM, OpenELMTokenizer

        paddle.set_device("gpu")

        # openelm 没有完全收录在transformers 主线，加上测试框架对 transformers的特殊处理，这里手动加载了。
        for mod_name in [m for m in list(sys.modules) if m == "transformers" or m.startswith("transformers.")]:
            del sys.modules[mod_name]

        TorchOpenELMConfig = get_class_from_dynamic_module("configuration_openelm.OpenELMConfig", _MODEL_HF_ID)
        TorchOpenELMForCausalLM = get_class_from_dynamic_module("modeling_openelm.OpenELMForCausalLM", _MODEL_HF_ID)

        config_path = hf_hub_download(_MODEL_HF_ID, "config.json")
        safetensors_path = hf_hub_download(_MODEL_HF_ID, "model.safetensors")

        with open(config_path) as f:
            config_dict = json.load(f)

        config = TorchOpenELMConfig(**config_dict)
        if not hasattr(config, "use_cache"):
            config.use_cache = True

        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model = TorchOpenELMForCausalLM(config)
        state_dict = load_file(safetensors_path)
        torch_model.load_state_dict(state_dict, strict=False)
        torch_model = torch_model.float().to(torch_device)
        torch_model.eval()

        tokenizer = OpenELMTokenizer.from_pretrained(_TOKENIZER_ID, download_hub="huggingface")
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
        print(f"[Diff] PyTorch 生成 tokens: {torch_gen_ids}")
        print(f"[Diff] PyTorch 生成文本: {repr(tokenizer.decode(torch_gen_ids, skip_special_tokens=True))}")

        del torch_model
        torch.cuda.empty_cache()

        # --- Paddle model ---
        paddle.seed(_SEED)
        paddle_model = OpenELMForCausalLM.from_pretrained(
            _MODEL_HF_ID,
            convert_from_hf=True,
            load_checkpoint_format="",
            download_hub="huggingface",
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

        print(f"[Diff] Paddle 生成 tokens: {paddle_gen_ids}")
        print(f"[Diff] Paddle 生成文本: {repr(tokenizer.decode(paddle_gen_ids, skip_special_tokens=True))}")

        self.assertEqual(
            list(torch_logits.shape),
            list(paddle_logits.shape),
            f"logits shape 不一致: torch={torch_logits.shape}, paddle={paddle_logits.shape}",
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

        n = min(len(torch_gen_ids), len(paddle_gen_ids), _NUM_GEN_TOKENS)
        print(f"\n[Diff] Token 对比 (前 {n} 个):")
        print(f"  {'Step':>4}  {'Torch':>10}  {'Paddle':>10}  {'Status':>6}")
        print("  " + "-" * 42)
        for i in range(n):
            ok = "OK" if torch_gen_ids[i] == paddle_gen_ids[i] else "FAIL"
            print(f"  {i+1:4d}  {torch_gen_ids[i]:10d}  {paddle_gen_ids[i]:10d}  {ok:>6}")

        self.assertEqual(
            torch_gen_ids[:n],
            paddle_gen_ids[:n],
            f"前 {n} 个生成的 token 不一致",
        )


if __name__ == "__main__":
    unittest.main()
