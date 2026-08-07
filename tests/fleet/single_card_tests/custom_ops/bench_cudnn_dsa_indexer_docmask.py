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

"""Benchmark cuDNN CSA indexer top-k document-mask overhead.

This script isolates the top-k stage used by the PaddleFleet cuDNN docmask
integration. It compares the legacy causal prefix path against the docmask path
that left-aligns each per-query valid range before calling the cuDNN radix
IndexerTopK kernel.

Example:
    PYTHONPATH=. python tests/fleet/single_card_tests/custom_ops/bench_cudnn_dsa_indexer_docmask.py \
        --sq 8192 --sk 2048 --topk 512 --ratio 4 --doc-lens 2048,2048,2048,2048
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle

from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
    cudnn_indexer_topk,
)
from paddleformers.fleet.cudnn_ops.indexer.docmask_utils import (
    shift_scores_to_local_window,
)
from paddleformers.fleet.transformer.csa_attention import get_valid_range

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Case:
    name: str
    sq: int
    sk: int
    topk: int
    ratio: int
    doc_lens: tuple[int, ...] | None


def _sync() -> None:
    paddle.device.synchronize()


def _time_ms(
    fn: Callable[[], object], warmup: int, repeat: int
) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    _sync()

    samples = []
    for _ in range(repeat):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1000.0)

    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return mean, median, stdev


def _make_startend(doc_lens: tuple[int, ...], sq: int) -> paddle.Tensor:
    if sum(doc_lens) != sq:
        raise ValueError(
            f"sum(doc_lens) must equal sq ({sq}), got {sum(doc_lens)}"
        )
    ends = []
    acc = 0
    for doc_len in doc_lens:
        if doc_len <= 0:
            raise ValueError(f"doc lengths must be positive, got {doc_len}")
        acc += int(doc_len)
        ends.extend([acc] * int(doc_len))
    return paddle.to_tensor(ends, dtype="int32").reshape([1, 1, sq, 1])


def _parse_doc_lens(value: str | None, sq: int) -> tuple[int, ...]:
    if value:
        return tuple(int(x) for x in value.split(",") if x)
    # Default to four equal docs; this stresses non-zero valid_start while still
    # matching the common ratio=4, topk=512 training shape family.
    if sq % 4 != 0:
        return (sq,)
    return (sq // 4, sq // 4, sq // 4, sq // 4)


def _run_case(
    case: Case, warmup: int, repeat: int
) -> dict[str, float | str | int]:
    paddle.seed(2026)
    scores = paddle.randn([1, case.sq, case.sk], dtype="float32")

    valid_range = None
    if case.doc_lens is not None:
        startend = _make_startend(case.doc_lens, case.sq)
        valid_range = get_valid_range(case.ratio, 1, case.sq, startend)

    def run_topk():
        return cudnn_indexer_topk(
            scores,
            case.sq,
            case.ratio,
            case.topk,
            valid_range=valid_range,
        )

    mean, median, stdev = _time_ms(run_topk, warmup, repeat)

    shift_mean = shift_median = shift_stdev = 0.0
    if valid_range is not None:

        def run_shift():
            return shift_scores_to_local_window(scores, valid_range)

        shift_mean, shift_median, shift_stdev = _time_ms(
            run_shift, warmup, repeat
        )

    return {
        "case": case.name,
        "sq": case.sq,
        "sk": case.sk,
        "topk": case.topk,
        "ratio": case.ratio,
        "docs": "none"
        if case.doc_lens is None
        else "+".join(str(x) for x in case.doc_lens),
        "topk_mean_ms": mean,
        "topk_median_ms": median,
        "topk_stdev_ms": stdev,
        "shift_mean_ms": shift_mean,
        "shift_median_ms": shift_median,
        "shift_stdev_ms": shift_stdev,
    }


def _print_row(row: dict[str, float | str | int]) -> None:
    print(
        "{case:>16} sq={sq:<6} sk={sk:<5} topk={topk:<4} ratio={ratio:<2} "
        "docs={docs:<24} topk={topk_median_ms:8.3f} ms "
        "shift={shift_median_ms:8.3f} ms".format(**row)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sq", type=int, default=8192)
    parser.add_argument("--sk", type=int, default=None)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--doc-lens", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--causal-only",
        action="store_true",
        help="Only benchmark the causal prefix path, without docmask.",
    )
    args = parser.parse_args()

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("This benchmark requires Paddle with CUDA support.")
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        raise RuntimeError(f"cuDNN IndexerTopK requires SM100, got SM{major}x")

    sk = (
        int(args.sk) if args.sk is not None else int(args.sq) // int(args.ratio)
    )
    cases = [Case("causal", args.sq, sk, args.topk, args.ratio, None)]
    if not args.causal_only:
        cases.append(
            Case(
                "docmask",
                args.sq,
                sk,
                args.topk,
                args.ratio,
                _parse_doc_lens(args.doc_lens, args.sq),
            )
        )

    print("cuDNN CSA indexer docmask top-k benchmark")
    print(f"warmup={args.warmup} repeat={args.repeat}")
    rows = [_run_case(case, args.warmup, args.repeat) for case in cases]
    for row in rows:
        _print_row(row)

    if len(rows) == 2:
        causal = float(rows[0]["topk_median_ms"])
        docmask = float(rows[1]["topk_median_ms"])
        if causal > 0:
            print(
                f"docmask / causal topk median ratio: {docmask / causal:.3f}x"
            )


if __name__ == "__main__":
    main()
