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


# Small config with GQA (num_attention_heads != num_key_value_heads, groups > 1)
SMALL_GQA_CFG = dict(SMALL_TEXT_CFG)
SMALL_GQA_CFG = {
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
    "num_key_value_heads": 2,  # GQA: groups = 4 // 2 = 2
    "rms_norm_eps": 1e-5,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
    },
    "sliding_window": None,
    "use_cache": True,
    "vocab_size": 1000,
}

# Small config with yarn RoPE (the real Ministral-3 scaling type)
SMALL_YARN_CFG = {
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
    "rope_scaling": {
        "rope_type": "yarn",
        "rope_theta": 10000.0,
        "factor": 2.0,
        "original_max_position_embeddings": 64,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    },
    "sliding_window": None,
    "use_cache": True,
    "vocab_size": 1000,
}


class TestMinistral3KVCache(unittest.TestCase):
    """R3: use_cache=True path must match use_cache=False numerically.

    Exercises the past_key_values.update branch (modeling.py:179-185) and the
    prepare_inputs_for_generation input_ids[:, -1:] slicing (modeling.py via
    GenerationMixin base), which were previously uncovered.
    """

    BATCH = 2
    SEQ = 12

    def setUp(self):
        from paddleformers.transformers import (
            Ministral3TextConfig,
            Ministral3TextDecoder,
        )

        self.text_cfg = Ministral3TextConfig(SMALL_GQA_CFG)
        self.model = Ministral3TextDecoder(self.text_cfg)
        self.model.eval()

    def test_cache_matches_no_cache(self):
        """One-shot forward with use_cache=True must equal use_cache=False."""
        input_ids = paddle.randint(0, SMALL_GQA_CFG["vocab_size"], [self.BATCH, self.SEQ])
        with paddle.no_grad():
            out_no_cache = self.model(input_ids=input_ids, use_cache=False).last_hidden_state
            out_cache = self.model(input_ids=input_ids, use_cache=True).last_hidden_state
        self.assertEqual(out_cache.shape, out_no_cache.shape)
        self.assertTrue(
            paddle.allclose(out_cache, out_no_cache, atol=1e-5),
            "use_cache=True output diverges from use_cache=False",
        )

    def test_incremental_cache_stepwise(self):
        """Step-by-step decode with KV cache must equal full-sequence forward.

        This mirrors the generate loop: feed first token, then append one token
        at a time reusing past_key_values, and compare against a single forward
        over the whole sequence.
        """
        from paddleformers.transformers.cache_utils import DynamicCache

        seq_len = self.SEQ
        input_ids = paddle.randint(0, SMALL_GQA_CFG["vocab_size"], [1, seq_len])

        with paddle.no_grad():
            full_out = self.model(input_ids=input_ids, use_cache=False).last_hidden_state

            cache = DynamicCache()
            step_outs = []
            for i in range(seq_len):
                token = input_ids[:, i : i + 1]
                out = self.model(input_ids=token, past_key_values=cache, use_cache=True)
                step_outs.append(out.last_hidden_state)
                cache = out.past_key_values
            step_out = paddle.concat(step_outs, axis=1)

        self.assertEqual(list(step_out.shape), list(full_out.shape))
        self.assertTrue(
            paddle.allclose(step_out, full_out, atol=1e-5),
            "incremental cache decode diverges from full-sequence forward",
        )

    def test_cache_grows_correctly(self):
        """past_key_values length must equal the number of tokens processed."""
        from paddleformers.transformers.cache_utils import DynamicCache

        input_ids = paddle.randint(0, SMALL_GQA_CFG["vocab_size"], [1, 6])
        with paddle.no_grad():
            cache = DynamicCache()
            for i in range(6):
                out = self.model(
                    input_ids=input_ids[:, i : i + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                self.assertEqual(cache.get_seq_length(), i + 1)


class TestMinistral3YarnRoPE(unittest.TestCase):
    """R3: yarn RoPE path (the real Ministral-3 scaling) must be covered.

    Exercises ROPE_INIT_FUNCTIONS['yarn'] + dynamic_rope_update +
    _get_llama4_attn_scale, which the default-rope SMALL_TEXT_CFG does not.
    """

    def setUp(self):
        from paddleformers.transformers import (
            Ministral3TextConfig,
            Ministral3TextDecoder,
        )

        self.text_cfg = Ministral3TextConfig(SMALL_YARN_CFG)
        self.model = Ministral3TextDecoder(self.text_cfg)
        self.model.eval()

    def test_yarn_rope_type(self):
        self.assertEqual(self.text_cfg.rope_parameters["rope_type"], "yarn")
        self.assertEqual(self.model.rotary_emb.rope_type, "yarn")

    def test_yarn_forward_finite(self):
        input_ids = paddle.randint(0, SMALL_YARN_CFG["vocab_size"], [2, 10])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids)
        self.assertEqual(
            list(output.last_hidden_state.shape),
            [2, 10, SMALL_YARN_CFG["hidden_size"]],
        )
        self.assertTrue(paddle.isfinite(output.last_hidden_state).all())

    def test_yarn_longer_than_original_max(self):
        """yarn must handle seq_len > original_max_position_embeddings (scaling)."""
        seq_len = 80  # > original_max_position_embeddings=64 in SMALL_YARN_CFG
        input_ids = paddle.randint(0, SMALL_YARN_CFG["vocab_size"], [1, seq_len])
        with paddle.no_grad():
            output = self.model(input_ids=input_ids)
        self.assertEqual(
            list(output.last_hidden_state.shape),
            [1, seq_len, SMALL_YARN_CFG["hidden_size"]],
        )
        self.assertTrue(paddle.isfinite(output.last_hidden_state).all())

    def test_yarn_cache_matches_no_cache(self):
        """yarn path must also keep use_cache consistent."""
        input_ids = paddle.randint(0, SMALL_YARN_CFG["vocab_size"], [1, 8])
        with paddle.no_grad():
            out_no_cache = self.model(input_ids=input_ids, use_cache=False).last_hidden_state
            out_cache = self.model(input_ids=input_ids, use_cache=True).last_hidden_state
        self.assertTrue(paddle.allclose(out_cache, out_no_cache, atol=1e-5))


class TestMinistral3GQAPadding(unittest.TestCase):
    """R3: GQA + padding mask (attention_mask with 0s) must be covered.

    Exercises repeat_kv (groups=2) and the causal_mask + pad_mask branch
    (modeling.py:299-301), previously only hit with all-ones mask.
    """

    BATCH = 3
    SEQ = 10

    def setUp(self):
        from paddleformers.transformers import (
            Ministral3TextConfig,
            Ministral3TextDecoder,
        )

        self.text_cfg = Ministral3TextConfig(SMALL_GQA_CFG)
        self.model = Ministral3TextDecoder(self.text_cfg)
        self.model.eval()

    def test_gqa_groups(self):
        self.assertEqual(self.text_cfg.num_attention_heads, 4)
        self.assertEqual(self.text_cfg.num_key_value_heads, 2)
        self.assertEqual(self.model.layers[0].self_attn.num_kv_groups, 2)

    def test_padding_mask_different_lengths(self):
        """Batch with different sequence lengths; padded positions masked with 0."""
        input_ids = paddle.randint(0, SMALL_GQA_CFG["vocab_size"], [self.BATCH, self.SEQ])
        # row 0: full length, row 1: last 3 padded, row 2: last 6 padded
        attn_mask = paddle.ones([self.BATCH, self.SEQ], dtype="int64")
        attn_mask[1, 7:] = 0
        attn_mask[2, 4:] = 0
        with paddle.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attn_mask)
        self.assertEqual(
            list(output.last_hidden_state.shape),
            [self.BATCH, self.SEQ, SMALL_GQA_CFG["hidden_size"]],
        )
        self.assertTrue(paddle.isfinite(output.last_hidden_state).all())

    def test_padding_does_not_affect_valid_positions(self):
        """Output at non-padded positions must not depend on padded input tokens."""
        input_ids = paddle.randint(0, SMALL_GQA_CFG["vocab_size"], [1, 8])
        attn_mask = paddle.ones([1, 8], dtype="int64")
        attn_mask[0, 5:] = 0  # last 3 are padding

        input_ids_padded = input_ids.clone()
        input_ids_padded[0, 5:] = paddle.randint(0, SMALL_GQA_CFG["vocab_size"], [3])

        with paddle.no_grad():
            out_clean = self.model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            out_padded = self.model(input_ids=input_ids_padded, attention_mask=attn_mask).last_hidden_state
        # valid positions (0..4) must match despite different padding token ids
        self.assertTrue(
            paddle.allclose(out_clean[:, :5], out_padded[:, :5], atol=1e-5),
            "padded token values leaked into valid positions",
        )


def _load_paddle_model_3b(dtype="float32"):
    import paddle

    from paddleformers.transformers import Mistral3ForConditionalGeneration

    paddle.set_device("gpu")

    model = Mistral3ForConditionalGeneration.from_pretrained(
        _MODEL_3B_HF_ID,
        download_hub="modelscope",
        convert_from_hf=True,
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


def _run_torch_inference(result_path):
    import torch
    from transformers import (
        Mistral3ForConditionalGeneration as TorchMistral3ForConditionalGeneration,
    )

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from paddleformers.transformers import Mistral3Tokenizer

    hf_model_id = _MODEL_3B_HF_ID
    tokenizer = Mistral3Tokenizer.from_pretrained(hf_model_id)
    inputs = tokenizer(_PROMPT_DIFF, return_tensors="pt")
    input_ids_pt = inputs["input_ids"]
    input_ids_list = input_ids_pt[0].tolist()
    print(f"\n[Diff-Torch] prompt: {repr(_PROMPT_DIFF)}")
    print(f"[Diff-Torch] input_ids: {input_ids_list}, seq_len={len(input_ids_list)}")

    torch.manual_seed(_SEED)
    print("[Diff-Torch] Loading PyTorch model (bf16 -> fp32, GPU)...")
    torch_model = TorchMistral3ForConditionalGeneration.from_pretrained(
        hf_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    fp8_n = _dequant_fp8_torch_model(torch_model, torch.bfloat16)
    print(f"[Diff-Torch] Dequantized {fp8_n} FP8 weights")
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
    print(f"[Diff-Torch] generated tokens: {torch_gen_ids}")
    print(f"[Diff-Torch] generated text: {repr(tokenizer.decode(torch_gen_ids, skip_special_tokens=True))}")

    np.savez(
        result_path,
        logits=torch_logits,
        gen_ids=np.array(torch_gen_ids, dtype=np.int64),
        input_ids=np.array(input_ids_list, dtype=np.int64),
    )
    print(f"[Diff-Torch] Results saved to {result_path}")


def _run_paddle_inference(result_path):
    import paddle

    from paddleformers.transformers import Mistral3Tokenizer

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    hf_model_id = _MODEL_3B_HF_ID
    tokenizer = Mistral3Tokenizer.from_pretrained(hf_model_id)
    inputs = tokenizer(_PROMPT_DIFF, return_tensors=None)
    input_ids_list = inputs["input_ids"]
    print(f"\n[Diff-Paddle] prompt: {repr(_PROMPT_DIFF)}")
    print(f"[Diff-Paddle] input_ids: {input_ids_list}, seq_len={len(input_ids_list)}")

    paddle.seed(_SEED)
    print("[Diff-Paddle] Loading Paddle model...")
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
    print(f"[Diff-Paddle] generated tokens: {paddle_gen_ids}")
    print(f"[Diff-Paddle] generated text: {repr(tokenizer.decode(paddle_gen_ids, skip_special_tokens=True))}")

    np.savez(
        result_path,
        logits=paddle_logits,
        gen_ids=np.array(paddle_gen_ids, dtype=np.int64),
        input_ids=np.array(input_ids_list, dtype=np.int64),
    )
    print(f"[Diff-Paddle] Results saved to {result_path}")


class TestMistral3DiffAlignment(unittest.TestCase):
    @slow
    @require_package("transformers", "torch")
    def test_diff_alignment(self):
        import subprocess

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        tmp_dir = os.path.join(project_root, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        torch_result = os.path.join(tmp_dir, "torch_result.npz")
        paddle_result = os.path.join(tmp_dir, "paddle_result.npz")

        this_file = os.path.abspath(__file__)
        python = os.sys.executable

        # Phase 1: Torch inference in subprocess
        print("\n[Diff] === Phase 1: PyTorch inference (subprocess) ===")
        script = (
            "import sys, os\n"
            f"sys.path.insert(0, {os.path.abspath(os.path.join(os.path.dirname(this_file), '..', '..', '..'))!r})\n"
            f"sys.path.insert(0, {os.path.dirname(this_file)!r})\n"
            f"os.environ['PADDLEFORMERS_TESTING'] = '1'\n"
            "os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')\n"
            "import test_modeling as tm\n"
            f"tm._run_torch_inference({torch_result!r})\n"
        )
        result = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            self.fail(f"Torch inference subprocess failed (exit {result.returncode})")

        # Phase 2: Paddle inference in subprocess
        print("\n[Diff] === Phase 2: Paddle inference (subprocess) ===")
        script = (
            "import sys, os\n"
            f"sys.path.insert(0, {os.path.abspath(os.path.join(os.path.dirname(this_file), '..', '..', '..'))!r})\n"
            f"sys.path.insert(0, {os.path.dirname(this_file)!r})\n"
            f"os.environ['PADDLEFORMERS_TESTING'] = '1'\n"
            "os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')\n"
            "import test_modeling as tm\n"
            f"tm._run_paddle_inference({paddle_result!r})\n"
        )
        result = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            self.fail(f"Paddle inference subprocess failed (exit {result.returncode})")

        # Phase 3: Compare results
        print("\n[Diff] === Phase 3: Compare results ===")
        torch_data = np.load(torch_result)
        paddle_data = np.load(paddle_result)

        torch_logits = torch_data["logits"]
        paddle_logits = paddle_data["logits"]
        torch_gen_ids = torch_data["gen_ids"].tolist()
        paddle_gen_ids = paddle_data["gen_ids"].tolist()

        print(f"[Diff] torch_logits shape: {torch_logits.shape}")
        print(f"[Diff] paddle_logits shape: {paddle_logits.shape}")

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
        tokenizer = Mistral3Tokenizer.from_pretrained(_MODEL_3B_HF_ID, download_hub="modelscope")

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
