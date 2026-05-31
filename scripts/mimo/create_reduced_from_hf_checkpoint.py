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

"""Create a reduced-depth full-width HF checkpoint from official MiMo weights."""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
MTP_LAYER_RE = re.compile(r"^model\.mtp_layers\.(\d+)\.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, help="HF MiMo checkpoint directory.")
    parser.add_argument("--output-dir", required=True, help="Output reduced HF checkpoint directory.")
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument(
        "--num-nextn-predict-layers",
        type=int,
        default=None,
        help="Optionally reduce MiMo MTP layers. Use 0 to match SFT configs that disable MTP.",
    )
    return parser.parse_args()


def keep_tensor(name: str, num_hidden_layers: int, num_nextn_predict_layers: int | None) -> bool:
    match = LAYER_RE.match(name)
    if match:
        return int(match.group(1)) < num_hidden_layers
    match = MTP_LAYER_RE.match(name)
    if match and num_nextn_predict_layers is not None:
        return int(match.group(1)) < num_nextn_predict_layers
    return True


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = source_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)

    tensors = {}
    shard_to_names = {}
    for name, shard in index["weight_map"].items():
        if keep_tensor(name, args.num_hidden_layers, args.num_nextn_predict_layers):
            shard_to_names.setdefault(shard, []).append(name)

    for shard, names in shard_to_names.items():
        with safe_open(source_dir / shard, framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name)

    save_file(tensors, output_dir / "model.safetensors", metadata={"format": "pt"})

    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    config["num_hidden_layers"] = args.num_hidden_layers
    if config.get("max_window_layers", args.num_hidden_layers) > args.num_hidden_layers:
        config["max_window_layers"] = args.num_hidden_layers
    if args.num_nextn_predict_layers is not None:
        config["num_nextn_predict_layers"] = args.num_nextn_predict_layers
    if isinstance(config.get("tokenizer_class"), list):
        config["tokenizer_class"] = config["tokenizer_class"][0]
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name in [
        ".gitattributes",
        "README.md",
        "configuration_mimo.py",
        "generation_config.json",
        "merges.txt",
        "modeling_mimo.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"Saved {len(tensors)} tensors to {output_dir}")
    print(f"Total tensor bytes: {total_size}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
