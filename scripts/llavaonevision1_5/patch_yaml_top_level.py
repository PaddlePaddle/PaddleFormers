#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Patch simple top-level YAML keys without requiring yq/PyYAML."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file")
    parser.add_argument("--set", action="append", default=[], help="Top-level KEY=VALUE assignment.")
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.yaml_file)
    assignments = {}
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid assignment {item!r}; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        assignments[key] = value

    lines = path.read_text(encoding="utf-8").splitlines()
    seen = set()
    patched = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if indent == "" and ":" in stripped and not stripped.startswith("#"):
            key = stripped.split(":", 1)[0].strip()
            if key in assignments:
                patched.append(f"{key}: {assignments[key]}")
                seen.add(key)
                continue
        patched.append(line)

    for key, value in assignments.items():
        if key not in seen:
            patched.append(f"{key}: {value}")

    path.write_text("\n".join(patched) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
