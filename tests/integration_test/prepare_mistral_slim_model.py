#!/usr/bin/env python

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

import argparse
import json
import os
import shutil
from pathlib import Path


def pick_existing(base: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"None of candidates found under {base}: {candidates}")


def symlink_or_copy(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare a slim Mistral model directory from a 7B checkpoint.")
    parser.add_argument("--src_dir", type=str, required=True, help="Source Mistral checkpoint directory.")
    parser.add_argument("--dst_dir", type=str, required=True, help="Destination slim model directory.")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Target number of decoder layers.")
    parser.add_argument("--force", action="store_true", help="Overwrite destination directory if exists.")
    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()

    if not src_dir.exists():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")

    if dst_dir.exists():
        if not args.force:
            raise FileExistsError(f"dst_dir already exists: {dst_dir}. Use --force to overwrite.")
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Normalize core files by symlink.
    config_src = pick_existing(src_dir, ["config.json", "config (1).json"])
    gen_src = pick_existing(src_dir, ["generation_config.json", "generation_config (1).json"])
    tok_cfg_src = pick_existing(src_dir, ["tokenizer_config.json", "tokenizer_config (1).json"])
    tok_json_src = pick_existing(src_dir, ["tokenizer.json", "tokenizer (1).json"])
    tok_model_src = pick_existing(src_dir, ["tokenizer.model", "tokenizer (1).model"])
    sp_map_src = pick_existing(src_dir, ["special_tokens_map.json", "special_tokens_map (1).json"])
    index_src = pick_existing(src_dir, ["model.safetensors.index.json"])

    # Read and rewrite config with fewer layers.
    with open(config_src, "r", encoding="utf-8") as f:
        config = json.load(f)
    origin_layers = int(config.get("num_hidden_layers", 32))
    config["num_hidden_layers"] = int(args.num_hidden_layers)

    with open(dst_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    symlink_or_copy(gen_src, dst_dir / "generation_config.json")
    symlink_or_copy(tok_cfg_src, dst_dir / "tokenizer_config.json")
    symlink_or_copy(tok_json_src, dst_dir / "tokenizer.json")
    symlink_or_copy(tok_model_src, dst_dir / "tokenizer.model")
    symlink_or_copy(sp_map_src, dst_dir / "special_tokens_map.json")
    symlink_or_copy(index_src, dst_dir / "model.safetensors.index.json")

    # Link shard files referenced by index.
    with open(index_src, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map = idx.get("weight_map", {})
    shard_names = sorted(set(weight_map.values()))
    for shard in shard_names:
        shard_src = src_dir / shard
        if not shard_src.exists():
            raise FileNotFoundError(f"Missing shard in source dir: {shard_src}")
        symlink_or_copy(shard_src, dst_dir / shard)

    est_ratio = float(args.num_hidden_layers) / float(origin_layers) if origin_layers > 0 else 0.0
    est_params_b = 7.0 * est_ratio
    print(f"Prepared slim mistral at: {dst_dir}")
    print(f"num_hidden_layers: {origin_layers} -> {args.num_hidden_layers}")
    print(f"Rough parameter scale estimate: ~{est_params_b:.3f}B (linear layer-ratio estimate)")


if __name__ == "__main__":
    main()
