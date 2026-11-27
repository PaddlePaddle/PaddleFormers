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

"""Maintainer-only export; acceptance runs MUST load, not regenerate, weights."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["FLAGS_use_cinn"] = "false"
os.environ["ENABLE_CINN_IN_DY2ST"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import paddle

from paddleformers.transformers import Olmo2Config, Olmo2ForCausalLM
from tests.cinn.olmo2.delivery_utils import (
    INITIAL_HASH,
    PARAMETERS,
    SEED,
    parameter_hash,
    sha256_file,
    validate_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(args.output.parent).free < PARAMETERS * 4 + 512 * 1024**2:
        raise RuntimeError("Insufficient disk space for the FP32 initialization")
    original = json.loads((args.source / "config.json").read_text())
    if original["num_hidden_layers"] != 16:
        raise ValueError("Expected original OLMo-2-0425-1B configuration")
    raw = dict(original, num_hidden_layers=2)
    validate_config(raw)
    paddle.set_device("gpu:0")
    paddle.seed(SEED)
    config = Olmo2Config(**raw)
    config._attn_implementation = "eager"
    model = Olmo2ForCausalLM(config)
    digest = parameter_hash(model)
    if digest != INITIAL_HASH:
        raise ValueError(f"Initialization does not reproduce the measured model: {digest}")
    assets = args.output / "assets"
    weights = args.output / "checkpoint"
    assets.mkdir(parents=True)
    weights.mkdir()
    (assets / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    (assets / "original_config.json").write_text(json.dumps(original, indent=2) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        shutil.copy2(args.source / name, assets / name)
    paddle.save(model.state_dict(), str(weights / "initial.pdparams"))
    manifest = []
    for root, directory in (("checkpoint", weights), ("assets", assets)):
        for path in sorted(directory.iterdir()):
            manifest.append(dict(root=root, file=path.name, bytes=path.stat().st_size, sha256=sha256_file(path)))
    data = Path(__file__).parent / "data/gsm8k_24.jsonl"
    manifest.append(dict(root="repo", file="data/gsm8k_24.jsonl", bytes=data.stat().st_size, sha256=sha256_file(data)))
    (args.output / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status="exported", parameter_sha256=digest, output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
