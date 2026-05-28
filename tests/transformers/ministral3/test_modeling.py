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
import unittest

import numpy as np
import paddle

from tests.testing_utils import require_package, slow

_MODEL_3B_PADDLE_ID = "learncat/Ministral-3-3B-Instruct-2512-for-paddle"
_MODEL_3B_HF_ID = "mistralai/Ministral-3-3B-Instruct-2512"
_PROMPT_DIFF = "Hello, how are you today?"
_PROMPT_INFERENCE = "What is the difference between cats and dogs?"
_SEED = 42
_NUM_GEN_TOKENS = 10
_DIFF_THRESHOLD = 1e-2


SMALL_TEXT_CFG = {
    "attention_dropout": 0.0,
    "head_dim": 16,
    "hidden_act": "silu",
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 128,
    "max_position_embeddings": 128,
    "model_type": "ministral3",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
    },
    "sliding_window": None,
    "use_cache": True,
    "vocab_size": 1000,
}

SMALL_VISION_CFG = {
    "intermediate_size": 128,
    "hidden_size": 64,
    "patch_size": 14,
    "image_size": 56,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "vocab_size": 1000,
    "head_dim": 16,
    "hidden_act": "gelu",
}


class TestMinistral3TextDecoder(unittest.TestCase):
    BATCH = 2
    SEQ = 10

    def setUp(self):
        from paddleformers.transformers import (
            Ministral3TextConfig,
            Ministral3TextDecoder,
        )

        self.text_cfg = Ministral3TextConfig(SMALL_TEXT_CFG)
        self.model = Ministral3TextDecoder(self.text_cfg)
        self.model.eval()

    def test_forward_shape(self):
        input_ids = paddle.randint(0, SMALL_TEXT_CFG["vocab_size"], [self.BATCH, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids)
        self.assertEqual(
            list(output.last_hidden_state.shape),
            [self.BATCH, self.SEQ, SMALL_TEXT_CFG["hidden_size"]],
        )

    def test_forward_with_attention_mask(self):
        input_ids = paddle.randint(0, SMALL_TEXT_CFG["vocab_size"], [self.BATCH, self.SEQ])
        attn_mask = paddle.ones([self.BATCH, self.SEQ], dtype=paddle.int64)
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        self.assertEqual(
            list(output.last_hidden_state.shape),
            [self.BATCH, self.SEQ, SMALL_TEXT_CFG["hidden_size"]],
        )

    def test_deterministic_output(self):
        input_ids = paddle.randint(0, SMALL_TEXT_CFG["vocab_size"], [1, 6])
        with paddle.no_grad():
            out1 = self.model(input_ids=input_ids).last_hidden_state
            out2 = self.model(input_ids=input_ids).last_hidden_state
        self.assertTrue(paddle.allclose(out1, out2))


class TestMistral3Model(unittest.TestCase):
    BATCH = 2
    SEQ = 8

    def setUp(self):
        from paddleformers.transformers import Mistral3Config, Mistral3Model

        self.config = Mistral3Config(
            text_config=SMALL_TEXT_CFG,
            vision_config=SMALL_VISION_CFG,
        )
        self.model = Mistral3Model(self.config)
        self.model.eval()

    def test_forward_shape(self):
        input_ids = paddle.randint(0, SMALL_TEXT_CFG["vocab_size"], [self.BATCH, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids)
        self.assertEqual(
            list(output.last_hidden_state.shape),
            [self.BATCH, self.SEQ, SMALL_TEXT_CFG["hidden_size"]],
        )

    def test_forward_with_attention_mask(self):
        input_ids = paddle.randint(0, SMALL_TEXT_CFG["vocab_size"], [self.BATCH, self.SEQ])
        attn_mask = paddle.ones([self.BATCH, self.SEQ], dtype=paddle.int64)
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        self.assertEqual(
            list(output.last_hidden_state.shape),
            [self.BATCH, self.SEQ, SMALL_TEXT_CFG["hidden_size"]],
        )

    def test_output_dtype_consistent(self):
        input_ids = paddle.randint(0, SMALL_TEXT_CFG["vocab_size"], [1, self.SEQ])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids)
        self.assertTrue(output.last_hidden_state.dtype in [paddle.float32, paddle.float16, paddle.bfloat16])


def _load_paddle_model_3b(dtype="float32"):
    import paddle

    from paddleformers.transformers import Mistral3ForConditionalGeneration

    paddle.set_device("gpu")

    model = Mistral3ForConditionalGeneration.from_pretrained(
        _MODEL_3B_PADDLE_ID,
        download_hub="aistudio",
        load_checkpoint_format="legacy",
        use_safetensors=False,
        dtype=dtype,
    )
    model.eval()
    return model


def _dequant_fp8_torch_model(model, torch_dtype):
    import torch
    import torch.nn as nn

    fp8_count = 0
    for mod in model.modules():
        if hasattr(mod, "weight") and hasattr(mod, "weight_scale_inv"):
            if mod.weight.dtype == torch.float8_e4m3fn:
                w = (mod.weight.data.float() * mod.weight_scale_inv.data.float()).to(torch_dtype)
                mod.weight = nn.Parameter(w, requires_grad=False)
                fp8_count += 1
    return fp8_count


class TestMistral3DiffAlignment(unittest.TestCase):
    @slow
    @require_package("transformers", "torch")
    def test_diff_alignment(self):
        import paddle
        import torch
        from transformers import (
            Mistral3ForConditionalGeneration as TorchMistral3ForConditionalGeneration,
        )

        from paddleformers.transformers import Mistral3Tokenizer

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        hf_model_id = _MODEL_3B_HF_ID

        tokenizer = Mistral3Tokenizer.from_pretrained(hf_model_id)
        inputs = tokenizer(_PROMPT_DIFF, return_tensors="pt")
        input_ids_pt = inputs["input_ids"]
        input_ids_list = input_ids_pt[0].tolist()
        print(f"\n[Diff] prompt: {repr(_PROMPT_DIFF)}")
        print(f"[Diff] input_ids: {input_ids_list}, seq_len={len(input_ids_list)}")
        print(f"[Diff] hf_model_id: {hf_model_id}")

        torch.manual_seed(_SEED)
        print("[Diff] Loading PyTorch model (float32, GPU)...")
        torch_model = TorchMistral3ForConditionalGeneration.from_pretrained(
            hf_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        fp8_n = _dequant_fp8_torch_model(torch_model, torch.bfloat16)
        print(f"[Diff] Dequantized {fp8_n} FP8 weights")
        # bf16 logits max_diff ~0.015, upgrade to fp32 for comparison
        torch_model = torch_model.to(torch.float32)
        torch_model.eval()

        device = torch_model.device
        input_ids_dev = input_ids_pt.to(device)

        with torch.inference_mode():
            torch_out = torch_model(input_ids=input_ids_dev, use_cache=False)
        torch_logits = torch_out.logits.float().cpu().numpy()

        with torch.inference_mode():
            torch_gen = torch_model.generate(
                input_ids=input_ids_dev,
                max_new_tokens=_NUM_GEN_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
            )
        torch_gen_ids = torch_gen[0, input_ids_pt.shape[1] :].cpu().tolist()
        print(f"[Diff] PyTorch generated tokens: {torch_gen_ids}")
        print(f"[Diff] PyTorch generated text: {repr(tokenizer.decode(torch_gen_ids, skip_special_tokens=True))}")

        del torch_model
        torch.cuda.empty_cache()

        paddle.seed(_SEED)
        print("[Diff] Loading Paddle model...")
        paddle_model = _load_paddle_model_3b(dtype="float32")

        input_ids_pd = paddle.to_tensor(np.array([input_ids_list], dtype=np.int64), dtype="int64")

        with paddle.no_grad():
            paddle_out = paddle_model(input_ids=input_ids_pd, use_cache=False)
        paddle_logits = paddle_out.logits.astype("float32").numpy()

        cur_ids = np.array([input_ids_list], dtype=np.int64)
        paddle_gen_ids = []
        with paddle.no_grad():
            for step in range(_NUM_GEN_TOKENS):
                input_tensor = paddle.to_tensor(cur_ids, dtype="int64")
                out = paddle_model(input_ids=input_tensor, use_cache=False)
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

        n = min(len(torch_gen_ids), len(paddle_gen_ids), _NUM_GEN_TOKENS)
        print(f"\n[Diff] Token comparison (first {n}):")
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


class TestMistral3PaddleInference(unittest.TestCase):
    """Load paddle-format weights and run inference."""

    @slow
    @require_package("transformers", "torch")
    def test_paddle_inference(self):
        import paddle

        from paddleformers.transformers import Mistral3Tokenizer

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        paddle.seed(_SEED)
        tokenizer = Mistral3Tokenizer.from_pretrained(_MODEL_3B_PADDLE_ID, download_hub="aistudio")

        encoded = tokenizer(_PROMPT_INFERENCE, return_tensors=None)
        input_ids_list = encoded["input_ids"]
        print(f"[Inference] prompt: {repr(_PROMPT_INFERENCE)}")
        model = _load_paddle_model_3b()

        cur_ids = np.array([input_ids_list], dtype=np.int64)
        gen_ids = []
        max_gen_tokens = 64
        with paddle.no_grad():
            for step in range(max_gen_tokens):
                input_tensor = paddle.to_tensor(cur_ids, dtype="int64")
                out = model(input_ids=input_tensor, use_cache=False)
                next_token = int(out.logits[0, -1].argmax().item())
                gen_ids.append(next_token)
                cur_ids = np.concatenate([cur_ids, [[next_token]]], axis=1)
                if next_token == tokenizer.eos_token_id:
                    break

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(input_ids_list + gen_ids, skip_special_tokens=True)

        print("=" * 60)
        print(f"[Inference] Generated text: {gen_text}")
        print(f"[Inference] Full text: {full_text}")
        print("=" * 60)

        self.assertGreater(len(gen_ids), 0, "Model generated no tokens")
        self.assertGreater(len(gen_text.strip()), 0, "Model generated empty text")

    @slow
    @require_package("transformers", "torch")
    def test_02_load_hf_original_model(self):
        """Load original HF weights and run inference."""
        import paddle

        from paddleformers.transformers import (
            Mistral3ForConditionalGeneration,
            Mistral3Tokenizer,
        )

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        paddle.set_device("gpu")
        paddle.seed(_SEED)

        tokenizer = Mistral3Tokenizer.from_pretrained(_MODEL_3B_HF_ID)

        model = Mistral3ForConditionalGeneration.from_pretrained(
            _MODEL_3B_HF_ID,
            use_converted_weights=False,
        )
        model.eval()

        prompt = "Hello, how are you today?"
        encoded = tokenizer(prompt, return_tensors=None)
        input_ids_list = encoded["input_ids"]
        print(f"[HF Original] prompt: {repr(prompt)}, model: {_MODEL_3B_HF_ID}")

        cur_ids = np.array([input_ids_list], dtype=np.int64)
        gen_ids = []
        max_gen_tokens = 32
        with paddle.no_grad():
            for step in range(max_gen_tokens):
                input_tensor = paddle.to_tensor(cur_ids, dtype="int64")
                out = model(input_ids=input_tensor, use_cache=False)
                next_token = int(out.logits[0, -1].argmax().item())
                gen_ids.append(next_token)
                cur_ids = np.concatenate([cur_ids, [[next_token]]], axis=1)
                if next_token == tokenizer.eos_token_id:
                    break

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(input_ids_list + gen_ids, skip_special_tokens=True)

        print("=" * 60)
        print(f"[HF Original] Generated text: {gen_text}")
        print(f"[HF Original] Full text: {full_text}")
        print("=" * 60)

        self.assertIsNotNone(gen_text)
        self.assertIsInstance(gen_text, str)
        self.assertGreater(len(gen_text), 0, "Model generated no text")
        printable_chars = sum(1 for c in gen_text if c.isprintable() or c.isspace())
        printable_ratio = printable_chars / len(gen_text) if len(gen_text) > 0 else 0
        self.assertGreater(printable_ratio, 0.5, "Output contains too many non-printable characters, possibly garbled")


if __name__ == "__main__":
    unittest.main()
