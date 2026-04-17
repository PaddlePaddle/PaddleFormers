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

"""Run an official-weight greedy generation alignment test for Olmo2.

This script is intentionally split by backend because the local Torch and Paddle
stacks live in separate virtual environments.

Example:
  source /root/venvs/transformers_cu126/bin/activate
  CUDA_VISIBLE_DEVICES=1 python tests/transformers/olmo2/run_generation_alignment.py --backend torch

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/root/PaddleFormers \
    python tests/transformers/olmo2/run_generation_alignment.py --backend paddle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "allenai/OLMo-2-0425-1B"
DEFAULT_WORK_DIR = Path("/tmp/olmo2_generation_alignment")
DEFAULT_PROMPT = "Question: What is 2 + 2?\nAnswer:"
LINEAR_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "paddle"], required=True)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    return parser.parse_args()


def resolve_model_dir(model_name_or_path):
    path = Path(model_name_or_path)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_name_or_path,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model*.safetensors",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
        )
    )


def run_torch(args):
    import torch
    from transformers import AutoTokenizer, Olmo2ForCausalLM

    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args.work_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolve_model_dir(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = Olmo2ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, device_map=None)
    model.eval().cuda()

    input_ids = tokenizer(args.prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
    prompt_len = input_ids.shape[-1]
    generated = input_ids
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            logits = model(input_ids=generated, use_cache=False).logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

    new_tokens = generated[0, prompt_len:].detach().cpu().numpy().astype("int64")
    np.savez(args.work_dir / "torch_generation_ref.npz", input_ids=input_ids.cpu().numpy(), new_tokens=new_tokens)
    (args.work_dir / "model_dir.txt").write_text(str(model_dir), encoding="utf-8")
    (args.work_dir / "prompt.txt").write_text(args.prompt, encoding="utf-8")

    print(f"model_dir: {model_dir}")
    print(f"prompt: {args.prompt!r}")
    print(f"torch_new_token_ids: {new_tokens.tolist()}")
    print(f"torch_new_text: {tokenizer.decode(new_tokens.tolist())!r}")


def build_olmo2_config_paddle(config_path):
    from paddleformers.transformers.olmo2 import Olmo2Config

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    return Olmo2Config(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        initializer_range=cfg["initializer_range"],
        rms_norm_eps=cfg["rms_norm_eps"],
        use_cache=False,
        attention_dropout=cfg["attention_dropout"],
        attention_bias=cfg["attention_bias"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        pad_token_id=cfg["pad_token_id"],
        eos_token_id=cfg["eos_token_id"],
        rope_theta=cfg["rope_theta"],
        _attn_implementation="eager",
        fuse_rms_norm=False,
    )


def load_hf_safetensors_into_paddle(model, model_dir):
    import paddle
    from safetensors import safe_open

    with open(model_dir / "model.safetensors.index.json", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    opened = {}

    def get_array(name):
        filename = weight_map[name]
        if filename not in opened:
            opened[filename] = safe_open(str(model_dir / filename), framework="np")
        return opened[filename].get_tensor(name)

    state = {}
    for name, tensor in model.state_dict().items():
        arr = get_array(name)
        if name.endswith(LINEAR_SUFFIXES):
            arr = arr.T
        state[name] = paddle.to_tensor(np.ascontiguousarray(arr), dtype=tensor.dtype)
    model.set_state_dict(state)


def run_paddle(args):
    import paddle
    from paddleformers.transformers import AutoTokenizer
    from paddleformers.transformers.olmo2 import Olmo2ForCausalLM

    paddle.seed(2026)
    paddle.set_default_dtype("float32")
    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")

    model_dir_file = args.work_dir / "model_dir.txt"
    if model_dir_file.exists():
        model_dir = Path(model_dir_file.read_text(encoding="utf-8").strip())
    else:
        model_dir = resolve_model_dir(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = Olmo2ForCausalLM(build_olmo2_config_paddle(model_dir / "config.json"))
    load_hf_safetensors_into_paddle(model, model_dir)
    model.eval()

    input_ids = np.asarray([tokenizer(args.prompt, add_special_tokens=False)["input_ids"]], dtype="int64")
    prompt_len = input_ids.shape[-1]
    generated = paddle.to_tensor(input_ids, dtype="int64")
    with paddle.no_grad():
        for _ in range(args.max_new_tokens):
            logits = model(input_ids=generated, use_cache=False, return_dict=True).logits
            next_token = paddle.argmax(logits[:, -1, :], axis=-1).unsqueeze(-1)
            generated = paddle.concat([generated, next_token], axis=-1)

    new_tokens = generated.numpy()[0, prompt_len:].astype("int64")
    print(f"prompt: {args.prompt!r}")
    print(f"paddle_new_token_ids: {new_tokens.tolist()}")
    print(f"paddle_new_text: {tokenizer.decode(new_tokens.tolist())!r}")

    ref_path = args.work_dir / "torch_generation_ref.npz"
    if ref_path.exists():
        ref = np.load(ref_path)["new_tokens"].astype("int64")
        token_ids_equal = bool(np.array_equal(ref, new_tokens))
        print(f"token_ids_equal: {token_ids_equal}")
        if not token_ids_equal:
            diff = np.nonzero(ref != new_tokens)[0].tolist()
            print(f"torch_new_token_ids: {ref.tolist()}")
            print(f"first_diff_index: {diff[0] if diff else None}")
            raise AssertionError("Paddle and Torch generated token ids are different.")


def main():
    args = parse_args()
    if args.backend == "torch":
        run_torch(args)
    else:
        run_paddle(args)


if __name__ == "__main__":
    main()
