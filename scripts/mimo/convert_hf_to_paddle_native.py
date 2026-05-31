#!/usr/bin/env python3
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
"""Convert official MiMo HF safetensors to a Paddle native checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import paddle
import torch
from paddle.utils.dlpack import from_dlpack
from safetensors.torch import load_file

TOKENIZER_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    return parser.parse_args()


def to_paddle(tensor: torch.Tensor) -> paddle.Tensor:
    tensor = tensor.contiguous().cpu()
    return from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def convert_dtype(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return tensor.to(dtype_map[dtype])


def fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, num_key_value_heads: int):
    q = q.transpose(0, 1).contiguous()
    k = k.transpose(0, 1).contiguous()
    v = v.transpose(0, 1).contiguous()
    hidden_size = q.shape[0]
    head_dim = q.shape[1] // num_heads
    num_key_value_groups = num_heads // num_key_value_heads
    q = q.reshape(hidden_size, num_key_value_heads, num_key_value_groups, head_dim)
    k = k.reshape(hidden_size, num_key_value_heads, 1, head_dim)
    v = v.reshape(hidden_size, num_key_value_heads, 1, head_dim)
    return torch.cat([q, k, v], dim=2).reshape(hidden_size, -1).contiguous()


def fuse_qkv_bias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, num_key_value_heads: int):
    head_dim = q.shape[0] // num_heads
    num_key_value_groups = num_heads // num_key_value_heads
    q = q.reshape(num_key_value_heads, num_key_value_groups, head_dim)
    k = k.reshape(num_key_value_heads, 1, head_dim)
    v = v.reshape(num_key_value_heads, 1, head_dim)
    return torch.cat([q, k, v], dim=1).reshape(-1).contiguous()


def load_hf_state(hf_dir: Path) -> dict[str, torch.Tensor]:
    index_path = hf_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            shard_names = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shard_names = sorted(path.name for path in hf_dir.glob("*.safetensors"))

    state = {}
    for shard_name in shard_names:
        state.update(load_file(str(hf_dir / shard_name), device="cpu"))
    return state


def main():
    args = parse_args()
    hf_dir = Path(args.hf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(hf_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    num_layers = int(config["num_hidden_layers"])
    num_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    num_mtp_layers = int(config.get("num_nextn_predict_layers", 0))

    hf_state = load_hf_state(hf_dir)
    paddle_state = {}
    consumed = set()

    def put(dst_key: str, tensor: torch.Tensor):
        paddle_state[dst_key] = to_paddle(convert_dtype(tensor, args.dtype))

    for key, tensor in hf_state.items():
        if any(part in key for part in [".q_proj.", ".k_proj.", ".v_proj.", ".gate_proj.", ".up_proj."]):
            continue
        consumed.add(key)
        if key.endswith((".o_proj.weight", ".down_proj.weight", ".input_proj.weight")):
            put(key, tensor.transpose(0, 1))
        else:
            put(key, tensor)

    def fuse_layer(prefix: str):
        q = hf_state[f"{prefix}.self_attn.q_proj.weight"]
        k = hf_state[f"{prefix}.self_attn.k_proj.weight"]
        v = hf_state[f"{prefix}.self_attn.v_proj.weight"]
        put(f"{prefix}.self_attn.qkv_proj.weight", fuse_qkv(q, k, v, num_heads, num_key_value_heads))
        consumed.update(
            [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
        )

        q_bias = hf_state.get(f"{prefix}.self_attn.q_proj.bias")
        k_bias = hf_state.get(f"{prefix}.self_attn.k_proj.bias")
        v_bias = hf_state.get(f"{prefix}.self_attn.v_proj.bias")
        if q_bias is not None and k_bias is not None and v_bias is not None:
            put(
                f"{prefix}.self_attn.qkv_proj.bias",
                fuse_qkv_bias(q_bias, k_bias, v_bias, num_heads, num_key_value_heads),
            )
            consumed.update(
                [
                    f"{prefix}.self_attn.q_proj.bias",
                    f"{prefix}.self_attn.k_proj.bias",
                    f"{prefix}.self_attn.v_proj.bias",
                ]
            )

        gate = hf_state[f"{prefix}.mlp.gate_proj.weight"].transpose(0, 1)
        up = hf_state[f"{prefix}.mlp.up_proj.weight"].transpose(0, 1)
        put(f"{prefix}.mlp.up_gate_proj.weight", torch.cat([gate, up], dim=1).contiguous())
        consumed.update([f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight"])

    for layer_id in range(num_layers):
        fuse_layer(f"model.layers.{layer_id}")
    for layer_id in range(num_mtp_layers):
        fuse_layer(f"model.mtp_layers.{layer_id}")

    leftovers = sorted(set(hf_state) - consumed)
    if leftovers:
        raise RuntimeError(f"Unconverted HF keys: {leftovers[:20]}")

    paddle.save(paddle_state, str(output_dir / "model_state.pdparams"))

    for name in TOKENIZER_FILES:
        src = hf_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    print(f"Saved Paddle native MiMo checkpoint to: {output_dir}")
    print(f"Converted tensors: {len(paddle_state)}")


if __name__ == "__main__":
    main()
