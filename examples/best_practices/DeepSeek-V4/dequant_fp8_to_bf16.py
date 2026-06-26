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

"""Dequantize DeepSeek-V4-Flash MXFP8/MXFP4 weights to BF16.

Supports two quantization formats found in DSV4-Flash checkpoints:
  - MXFP8 (float8_e4m3fn weight + float8_e8m0fnu scale, block 128x128)
    Used for attention projections (wq_a, wq_b, wkv, wo_a, wo_b).
  - MXFP4 (int8 packed weight + float8_e8m0fnu scale, group_size 32)
    Used for expert weights (w1, w2, w3).

Scale format: float8_e8m0fnu stores a per-block power-of-two exponent.
  Actual scale value = 2^(stored_byte - 127)
  PyTorch .float() conversion handles this automatically.

CPU-only. Processes shards sequentially with per-tensor progress tracking.

Usage:
    python dequant_fp8_to_bf16.py \
        --input_dir  /path/to/DeepSeek-V4-Flash \
        --output_dir /path/to/DeepSeek-V4-Flash-bf16 \
        --num_workers 4
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Disable CUDA entirely — pure CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# FP4 (e2m1) lookup table: maps 4-bit index to float value
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def fp8_weight_to_bf16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 (e4m3fn) weight to BF16."""
    assert weight.dtype == torch.float8_e4m3fn
    # assert scale.dtype == torch.float8_e8m0fnu
    assert scale.dtype == torch.float32
    out_blocks, in_blocks = scale.shape
    out = weight.float().view(out_blocks, 128, in_blocks, 128)
    out = out * scale.float()[:, None, :, None]
    return out.reshape(weight.shape).bfloat16()


def fp4_weight_to_bf16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 (e2m1, packed int8) weight to BF16."""
    assert weight.dtype == torch.int8
    assert scale.dtype == torch.float8_e8m0fnu
    packed = weight.view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    values = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(1)
    values = values.view(values.shape[0], scale.shape[1], 32)
    values = values * scale.float().unsqueeze(-1)
    return values.reshape(values.shape[0], -1).bfloat16()


def dequant_tensor(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Auto-dispatch dequantization based on weight dtype."""
    if weight.dtype == torch.float8_e4m3fn:
        return fp8_weight_to_bf16(weight, scale)
    elif weight.dtype == torch.int8:
        return fp4_weight_to_bf16(weight, scale)
    else:
        raise ValueError(f"Unexpected quantized weight dtype: {weight.dtype}")


def process_shard(
    shard_file: str,
    shard_keys: list,
    input_dir: str,
    output_dir: str,
    quant_weight_keys: set,
    scale_keys: set,
    shard_idx: int,
    total_shards: int,
) -> tuple:
    """Process a single shard: dequantize and write output."""
    shard_path = os.path.join(input_dir, shard_file)
    out_path = os.path.join(output_dir, shard_file)

    # Count work items (excluding scales)
    work_keys = [k for k in shard_keys if k not in scale_keys]
    n_quant = sum(1 for k in work_keys if k in quant_weight_keys)
    n_copy = len(work_keys) - n_quant

    print(f"  [{shard_idx + 1} / {total_shards}] {shard_file}: " f"{n_quant} dequant + {n_copy} copy ...", flush=True)

    t_start = time.time()
    new_tensors = {}
    dequanted = 0

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for i, key in enumerate(work_keys):
            if key in quant_weight_keys:
                scale_key = key.replace(".weight", ".scale")
                weight = f.get_tensor(key)
                scale = f.get_tensor(scale_key)
                new_tensors[key] = dequant_tensor(weight, scale)
                del weight, scale
                dequanted += 1
            else:
                new_tensors[key] = f.get_tensor(key)

            # Progress every 100 tensors
            if (i + 1) % 200 == 0:
                print(f"    ... {i + 1} / {len(work_keys)} tensors processed", flush=True)

    t_dequant = time.time()
    print(f"    dequant done in {t_dequant - t_start:.1f}s, writing {shard_file} ...", flush=True)

    # Write output
    save_file(new_tensors, out_path)
    del new_tensors
    gc.collect()

    t_write = time.time()
    out_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(
        f"    written {out_size_mb:.0f} MB in {t_write - t_dequant:.1f}s " f"(total {t_write - t_start:.1f}s)",
        flush=True,
    )

    return shard_file, dequanted


# ---------------------------------------------------------------------------
# Worker function for parallel mode
# ---------------------------------------------------------------------------

_g_input_dir: str = ""
_g_output_dir: str = ""
_g_quant_weight_keys: set = set()
_g_scale_keys: set = set()


def _worker_init(input_dir: str, output_dir: str, quant_weight_keys: set, scale_keys: set):
    global _g_input_dir, _g_output_dir, _g_quant_weight_keys, _g_scale_keys
    _g_input_dir = input_dir
    _g_output_dir = output_dir
    _g_quant_weight_keys = quant_weight_keys
    _g_scale_keys = scale_keys


def _process_shard_worker(task: tuple) -> tuple:
    """Worker wrapper with error reporting."""
    shard_file, shard_keys, shard_idx, total_shards = task
    try:
        return process_shard(
            shard_file,
            shard_keys,
            _g_input_dir,
            _g_output_dir,
            _g_quant_weight_keys,
            _g_scale_keys,
            shard_idx,
            total_shards,
        )
    except Exception as e:
        print(f"  ERROR in {shard_file}: {e}", flush=True)
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="Dequantize DeepSeek-V4-Flash MXFP8/MXFP4 weights to BF16")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Parallel workers (default 4; each needs ~12GB RAM)"
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Process shards sequentially (lowest memory, easiest to debug)"
    )
    parser.add_argument(
        "--start_shard", type=int, default=0, help="Resume from this shard index (0-based, skip already done)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    with open(input_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Group params by shard
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard_file in weight_map.items():
        shard_to_keys.setdefault(shard_file, []).append(key)

    # Identify quantized weights
    scale_keys = {k for k in weight_map if k.endswith(".scale")}
    quant_weight_keys = set()
    for sk in scale_keys:
        weight_key = sk.replace(".scale", ".weight")
        if weight_key in weight_map:
            quant_weight_keys.add(weight_key)

    num_fp4 = sum(1 for k in quant_weight_keys if "experts" in k)
    num_fp8 = len(quant_weight_keys) - num_fp4

    print(f"{'=' * 60}")
    print("DeepSeek-V4-Flash Dequantization: MXFP8 + MXFP4 -> BF16")
    print(f"{'=' * 60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Quantized weights:   {len(quant_weight_keys)} " f"({num_fp8} MXFP8 + {num_fp4} MXFP4)")
    print(f"Scale keys (drop):   {len(scale_keys)}")
    print(f"Shards:              {len(shard_to_keys)}")
    print(f"Mode:                {'sequential' if args.sequential else f'{args.num_workers} workers'}")
    if args.start_shard > 0:
        print(f"Resuming from shard: {args.start_shard}")
    print()

    sorted_shards = sorted(shard_to_keys.keys())
    total_shards = len(sorted_shards)
    total_dequanted = 0

    if args.sequential:
        # ---- Sequential mode: easiest to debug, lowest memory ----
        for idx, shard_file in enumerate(sorted_shards):
            if idx < args.start_shard:
                continue
            _, dequanted = process_shard(
                shard_file,
                shard_to_keys[shard_file],
                str(input_dir),
                str(output_dir),
                quant_weight_keys,
                scale_keys,
                idx,
                total_shards,
            )
            total_dequanted += dequanted
    else:
        # ---- Parallel mode ----
        tasks = [
            (shard_file, shard_to_keys[shard_file], idx, total_shards)
            for idx, shard_file in enumerate(sorted_shards)
            if idx >= args.start_shard
        ]

        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_worker_init,
            initargs=(str(input_dir), str(output_dir), quant_weight_keys, scale_keys),
        ) as executor:
            futures = {executor.submit(_process_shard_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                try:
                    shard_file, dequanted = future.result(timeout=600)
                    total_dequanted += dequanted
                except Exception as e:
                    shard_name = futures[future]
                    print(f"\nFATAL: Shard {shard_name} failed: {e}", file=sys.stderr)
                    raise

    print(f"\n{'=' * 60}")
    print(f"Dequantized {total_dequanted} weights to BF16")

    # ---- Update index (drop .scale entries) ----
    new_weight_map = {k: v for k, v in weight_map.items() if k not in scale_keys}
    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # ---- Update config.json ----
    config_path = input_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        config.pop("quantization_config", None)
        config.pop("expert_dtype", None)
        config["torch_dtype"] = "bfloat16"
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    # ---- Copy other files ----
    for fname in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json"]:
        src = input_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(output_dir / fname))

    encoding_dir = input_dir / "encoding"
    if encoding_dir.is_dir() and not (output_dir / "encoding").exists():
        shutil.copytree(str(encoding_dir), str(output_dir / "encoding"))

    print(f"Done! BF16 model saved to: {output_dir}")
    print(f"Weight map: {len(new_weight_map)} entries (dropped {len(scale_keys)} scales)")


if __name__ == "__main__":
    main()
