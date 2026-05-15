#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Convert GSM8K data to the erniekit src/tgt format used by SFT configs.

The input can be either a plain JSONL file with question/answer fields or a
HuggingFace datasets ``save_to_disk`` directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Input GSM8K jsonl file or HuggingFace datasets save_to_disk directory.",
    )
    parser.add_argument("--output", required=True, help="Output erniekit jsonl file.")
    parser.add_argument("--split", default=None, help="Dataset split to read when --input is a save_to_disk directory.")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument(
        "--prompt-template",
        default="{question}",
        help="Template used to build src[0]. It may contain {question}.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def iter_save_to_disk(path: Path, split: str | None) -> Iterable[tuple[int, dict]]:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise ImportError("Reading a save_to_disk dataset requires the `datasets` package.") from exc

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split is None:
            raise ValueError(f"{path} contains splits {list(dataset.keys())}; please pass --split.")
        if split not in dataset:
            raise KeyError(f"Split {split!r} not found in {path}; available splits: {list(dataset.keys())}.")
        dataset = dataset[split]
    elif split is not None:
        raise ValueError(f"{path} is a single dataset, but --split {split!r} was provided.")

    for idx, example in enumerate(dataset, start=1):
        yield idx, dict(example)


def iter_examples(path: Path, split: str | None) -> Iterable[tuple[int, dict]]:
    if path.is_dir():
        yield from iter_save_to_disk(path, split)
    else:
        if split is not None:
            raise ValueError("--split can only be used when --input is a save_to_disk directory.")
        yield from iter_jsonl(path)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for line_no, example in iter_examples(input_path, args.split):
            if args.question_key not in example or args.answer_key not in example:
                raise KeyError(
                    f"{input_path}:{line_no} must contain keys "
                    f"{args.question_key!r} and {args.answer_key!r}."
                )
            question = str(example[args.question_key])
            answer = str(example[args.answer_key])
            prompt = args.prompt_template.format(question=question)
            fout.write(json.dumps({"src": [prompt], "tgt": [answer]}, ensure_ascii=False) + "\n")
            count += 1

    split_suffix = f" split={args.split}" if args.split else ""
    print(f"Converted {count} examples{split_suffix}: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
