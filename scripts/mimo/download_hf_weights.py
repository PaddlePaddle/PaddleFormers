#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Download official Xiaomi MiMo HF assets for Paddle alignment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.py",
    "tokenizer*",
    "vocab.json",
    "merges.txt",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="XiaomiMiMo/MiMo-7B-Base")
    parser.add_argument("--local-dir", default="./models/MiMo-7B-Base")
    parser.add_argument("--token", default=None, help="HF token. Defaults to HF_TOKEN/HUGGING_FACE_HUB_TOKEN env.")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
        help="Optional HF endpoint/mirror. Defaults to HF_ENDPOINT or https://hf-mirror.com.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--all-files", action="store_true", help="Download all repo files instead of model assets only."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
        print(f"Using HF_ENDPOINT={args.endpoint}")

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(local_dir),
        token=token,
        allow_patterns=None if args.all_files else DEFAULT_ALLOW_PATTERNS,
        max_workers=args.max_workers,
    )

    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    total_size = sum(p.stat().st_size for p in files)
    safetensors = [p for p in files if p.suffix == ".safetensors"]

    print(f"Downloaded snapshot to: {path}")
    print(f"Local dir: {local_dir}")
    print(f"Files: {len(files)}")
    print(f"Safetensors shards: {len(safetensors)}")
    print(f"Total size: {total_size / (1024 ** 3):.2f} GiB")
    for p in files:
        rel = p.relative_to(local_dir)
        print(f"{rel}\t{p.stat().st_size / (1024 ** 2):.2f} MiB")


if __name__ == "__main__":
    main()
