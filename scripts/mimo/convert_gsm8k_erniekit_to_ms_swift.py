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

"""Convert ERNIEKit-style GSM8K jsonl to ms-swift messages jsonl."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def first_text(value):
    if isinstance(value, list):
        return value[0]
    return value


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            query = first_text(row["src"])
            response = first_text(row["tgt"])
            out = {
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response},
                ]
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
