#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_MODEL_FILES = ["config.json", "model.safetensors.index.json"]
TOKENIZER_OR_PROCESSOR_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny-dir", default="./tiny-random-llavaonevision1_5")
    parser.add_argument("--reduced-dir", default="./.cache/llavaonevision1_5/reduced-random-llavaonevision1_5-4l-512h")
    parser.add_argument("--reduced-hf-dir", default="./.cache/llavaonevision1_5/reduced-random-llavaonevision1_5-4l-512h-hf")
    parser.add_argument("--reduced-reference-dir", default="./.cache/llavaonevision1_5/reduced_text_reference")
    parser.add_argument("--gsm8k-train", default="./data/gsm8k_erniekit/train.jsonl")
    parser.add_argument("--gsm8k-eval", default="./data/gsm8k_erniekit/test.jsonl")
    parser.add_argument("--sft-vl-dir", default="./tests/fixtures/dummy/sft-vl/DoclingMatix")
    return parser.parse_args()


def check_path(path: Path, label: str, required: bool = True) -> bool:
    exists = path.exists()
    status = "OK" if exists else ("MISSING" if required else "optional-missing")
    print(f"{status}: {label}: {path}")
    return exists or not required


def main():
    args = parse_args()
    ok = True
    tiny_dir = Path(args.tiny_dir)
    ok &= check_path(tiny_dir, "tiny checkpoint directory")
    for name in REQUIRED_MODEL_FILES:
        ok &= check_path(tiny_dir / name, f"tiny checkpoint {name}")

    tokenizer_assets = [name for name in TOKENIZER_OR_PROCESSOR_FILES if (tiny_dir / name).exists()]
    if tokenizer_assets:
        print(f"OK: tokenizer/processor assets present: {', '.join(tokenizer_assets)}")
    else:
        print("MISSING: tokenizer/processor assets for training-style CE.")
        ok = False

    reduced_dir = Path(args.reduced_dir)
    ok &= check_path(reduced_dir, "reduced checkpoint directory", required=False)
    if reduced_dir.exists():
        for name in REQUIRED_MODEL_FILES:
            ok &= check_path(reduced_dir / name, f"reduced checkpoint {name}")

    reduced_hf_dir = Path(args.reduced_hf_dir)
    ok &= check_path(reduced_hf_dir, "HF-compatible reduced checkpoint directory", required=False)
    if reduced_hf_dir.exists():
        for name in ["config.json", "model.safetensors.index.json", "configuration_llavaonevision1_5.py", "modeling_llavaonevision1_5.py"]:
            ok &= check_path(reduced_hf_dir / name, f"HF-compatible reduced checkpoint {name}")

    reduced_reference_dir = Path(args.reduced_reference_dir)
    ok &= check_path(reduced_reference_dir, "reduced text reference directory", required=False)
    if reduced_reference_dir.exists():
        for name in ["reference.npz", "metadata.json"]:
            ok &= check_path(reduced_reference_dir / name, f"reduced text reference {name}")

    ok &= check_path(Path(args.gsm8k_train), "GSM8K erniekit train data", required=False)
    ok &= check_path(Path(args.gsm8k_eval), "GSM8K erniekit eval data", required=False)
    ok &= check_path(Path(args.sft_vl_dir), "VL dummy image directory", required=False)

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
