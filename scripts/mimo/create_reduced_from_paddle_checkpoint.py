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

"""Create a reduced-depth full-width MiMo checkpoint from a Paddle native checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import paddle

ASSET_FILES = [
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    return parser.parse_args()


def keep_key(key: str, num_hidden_layers: int) -> bool:
    match = re.match(r"model\.layers\.(\d+)\.", key)
    if match:
        return int(match.group(1)) < num_hidden_layers
    return True


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(source_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    config["num_hidden_layers"] = args.num_hidden_layers
    if "layer_types" in config:
        config["layer_types"] = config["layer_types"][: args.num_hidden_layers]

    state = paddle.load(str(source_dir / "model_state.pdparams"))
    reduced_state = {key: value for key, value in state.items() if keep_key(key, args.num_hidden_layers)}

    paddle.save(reduced_state, str(output_dir / "model_state.pdparams"))
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")

    for name in ASSET_FILES:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    print(f"Saved reduced MiMo checkpoint to: {output_dir}")
    print(f"Kept decoder layers: {args.num_hidden_layers}")
    print(f"Converted tensors: {len(reduced_state)}")


if __name__ == "__main__":
    main()
