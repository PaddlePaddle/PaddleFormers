#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_MODEL_FILES = ["config.json"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny-dir", default="./.cache/mimo/tiny-random-mimo")
    parser.add_argument("--reduced-dir", default="./.cache/mimo/reduced-depth-4l-fullwidth-random")
    parser.add_argument("--reference-dir", default="./.cache/mimo/reference")
    parser.add_argument("--gsm8k-train", default="./data/gsm8k_erniekit/train.jsonl")
    parser.add_argument("--gsm8k-eval", default="./data/gsm8k_erniekit/test.jsonl")
    return parser.parse_args()


def check_path(path: Path, label: str, required: bool = True) -> bool:
    exists = path.exists()
    status = "OK" if exists else ("MISSING" if required else "optional-missing")
    print(f"{status}: {label}: {path}")
    return exists or not required


def check_checkpoint(path: Path, label: str, required: bool = True) -> bool:
    ok = check_path(path, f"{label} checkpoint directory", required=required)
    if not path.exists():
        return ok

    has_weights = (
        any(path.glob("*.safetensors")) or any(path.glob("*.pdparams")) or (path / "model_state.pdparams").exists()
    )
    if has_weights:
        print(f"OK: {label} checkpoint weights")
    else:
        print(f"MISSING: {label} checkpoint weights")
        ok = False

    for name in REQUIRED_MODEL_FILES:
        ok &= check_path(path / name, f"{label} checkpoint {name}", required=True)

    tokenizer_assets = [name for name in TOKENIZER_FILES if (path / name).exists()]
    if tokenizer_assets:
        print(f"OK: {label} tokenizer assets present: {', '.join(tokenizer_assets)}")
    else:
        print(f"MISSING: {label} tokenizer assets for training/generation.")
        ok = False
    return ok


def main():
    args = parse_args()
    ok = True
    ok &= check_checkpoint(Path(args.tiny_dir), "tiny", required=False)
    ok &= check_checkpoint(Path(args.reduced_dir), "reduced", required=False)

    reference_dir = Path(args.reference_dir)
    ok &= check_path(reference_dir, "HF reference directory", required=False)
    if reference_dir.exists():
        for name in ["reference.npz", "metadata.json"]:
            ok &= check_path(reference_dir / name, f"HF reference {name}")

    ok &= check_path(Path(args.gsm8k_train), "GSM8K erniekit train data", required=False)
    ok &= check_path(Path(args.gsm8k_eval), "GSM8K erniekit eval data", required=False)

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
