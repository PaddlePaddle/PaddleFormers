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

"""Dequantize Kimi-K3 MXFP4 routed-expert weights to BF16.

Kimi-K3 stores routed-expert matrices as ``*.weight_packed`` U8 tensors and
their E8M0 block scales as ``*.weight_scale`` U8 tensors. Two E2M1 FP4 values
are packed in each weight byte, low nibble first, and one scale is shared by
32 consecutive values.

The output replaces each packed/scale pair with one BF16 ``*.weight`` tensor.
Unquantized BF16/F32 tensors are copied unchanged. Shards are written through
per-process temporary files and atomically renamed, so ``--resume`` can safely
continue an interrupted conversion.

Example:
    python scripts/ckpt_utils/kimik3_dequant_mxfp4_to_bf16.py \
        --input_dir /path/to/Kimi-K3-MXFP4 \
        --output_dir /path/to/Kimi-K3-BF16 \
        --num_workers 4 \
        --resume
"""

import argparse
import gc
import json
import math
import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import paddle
from safetensors import safe_open
from safetensors.paddle import save_file

# This conversion is intentionally CPU-only.
paddle.set_device("cpu")


FP4_GROUP_SIZE = 32
FP4_TABLE = paddle.to_tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=paddle.float32,
)

_LOG_HANDLE = None


def emit(message: str = "") -> None:
    """Print one parent-process status line and persist it when requested."""

    print(message, flush=True)
    if _LOG_HANDLE is not None:
        _LOG_HANDLE.write(message + "\n")
        _LOG_HANDLE.flush()


def read_safetensors_header(path: Path) -> dict:
    """Read and validate a safetensors header without touching tensor payload."""

    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors prefix")
        (header_length,) = struct.unpack("<Q", prefix)
        header_raw = handle.read(header_length)
        if len(header_raw) != header_length:
            raise ValueError(f"{path}: truncated safetensors header")

    header = json.loads(header_raw)
    header.pop("__metadata__", None)
    payload_end = max(
        (spec["data_offsets"][1] for spec in header.values()),
        default=0,
    )
    expected_size = 8 + header_length + payload_end
    actual_size = path.stat().st_size
    if expected_size != actual_size:
        raise ValueError(f"{path}: invalid size, expected {expected_size}, got {actual_size}")
    return header


def e8m0_scale_to_float(scale: paddle.Tensor) -> paddle.Tensor:
    """Decode U8 E8M0 scale bytes as powers of two in FP32."""

    if scale.dtype != paddle.uint8:
        raise TypeError(f"E8M0 scale must be uint8, got {scale.dtype}")
    if bool(paddle.any(scale == 255).item()):
        raise ValueError("E8M0 scale contains reserved NaN encoding 255")
    exponent = scale.astype(paddle.int32) - 127
    return paddle.ldexp(paddle.ones(scale.shape, dtype=paddle.float32), exponent)


def fp4_weight_to_bf16(
    packed_weight: paddle.Tensor,
    scale: paddle.Tensor,
    group_size: int = FP4_GROUP_SIZE,
) -> paddle.Tensor:
    """Expand packed E2M1 FP4 weights and apply per-group E8M0 scales."""

    if packed_weight.dtype not in (paddle.uint8, paddle.int8):
        raise TypeError(f"packed MXFP4 weight must be uint8/int8, got {packed_weight.dtype}")
    if packed_weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "Kimi-K3 MXFP4 weight and scale must both be rank-2, got "
            f"{tuple(packed_weight.shape)} and {tuple(scale.shape)}"
        )

    packed = packed_weight.view(paddle.uint8)
    unpacked_columns = packed.shape[1] * 2
    expected_columns = scale.shape[1] * group_size
    if packed.shape[0] != scale.shape[0] or unpacked_columns != expected_columns:
        raise ValueError(
            "MXFP4 packed/scale shape mismatch: "
            f"packed={tuple(packed.shape)}, scale={tuple(scale.shape)}, "
            f"group_size={group_size}"
        )

    low = paddle.bitwise_and(
        packed,
        paddle.full([], 0x0F, dtype=paddle.uint8),
    )
    high = paddle.bitwise_right_shift(
        packed,
        paddle.full([], 4, dtype=paddle.uint8),
    )
    values = paddle.stack(
        (
            FP4_TABLE[low.astype(paddle.int64)],
            FP4_TABLE[high.astype(paddle.int64)],
        ),
        axis=-1,
    ).reshape(packed.shape[0], unpacked_columns)
    values = values.reshape(scale.shape[0], scale.shape[1], group_size)
    values = values * e8m0_scale_to_float(scale).unsqueeze(-1)
    return values.reshape(scale.shape[0], expected_columns).astype(paddle.bfloat16)


def output_key(input_key: str) -> str | None:
    """Map one source key to its BF16 output key, dropping scale tensors."""

    if input_key.endswith(".weight_scale"):
        return None
    if input_key.endswith(".weight_packed"):
        return input_key.removesuffix("_packed")
    return input_key


def expected_output_spec(
    input_key: str,
    input_spec: dict,
    source_header: dict,
) -> dict | None:
    """Build the expected output header spec for one source tensor."""

    key = output_key(input_key)
    if key is None:
        return None
    if not input_key.endswith(".weight_packed"):
        return {
            "dtype": input_spec["dtype"],
            "shape": input_spec["shape"],
        }

    scale_key = input_key.removesuffix(".weight_packed") + ".weight_scale"
    if scale_key not in source_header:
        raise KeyError(f"{input_key}: missing paired {scale_key}")
    scale_spec = source_header[scale_key]
    packed_shape = input_spec["shape"]
    scale_shape = scale_spec["shape"]
    if input_spec["dtype"] != "U8" or scale_spec["dtype"] != "U8":
        raise TypeError(
            f"{input_key}: expected U8 packed weight and scale, got "
            f"{input_spec['dtype']} and {scale_spec['dtype']}"
        )
    if (
        len(packed_shape) != 2
        or len(scale_shape) != 2
        or packed_shape[0] != scale_shape[0]
        or packed_shape[1] * 2 != scale_shape[1] * FP4_GROUP_SIZE
    ):
        raise ValueError(f"{input_key}: invalid packed={packed_shape}, scale={scale_shape}")
    return {
        "dtype": "BF16",
        "shape": [packed_shape[0], packed_shape[1] * 2],
    }


def validate_output_shard(
    source_path: Path,
    output_path: Path,
    shard_keys: list[str],
) -> tuple[int, int]:
    """Validate output keys, shapes, dtypes and return tensor/payload counts."""

    source_header = read_safetensors_header(source_path)
    output_header = read_safetensors_header(output_path)
    expected = {}
    for key in shard_keys:
        spec = expected_output_spec(key, source_header[key], source_header)
        mapped = output_key(key)
        if mapped is not None:
            expected[mapped] = spec

    if set(output_header) != set(expected):
        missing = sorted(set(expected) - set(output_header))[:5]
        extra = sorted(set(output_header) - set(expected))[:5]
        raise ValueError(f"{output_path}: key mismatch, missing={missing}, extra={extra}")

    payload_bytes = 0
    for key, expected_spec in expected.items():
        actual = output_header[key]
        actual_spec = {
            "dtype": actual["dtype"],
            "shape": actual["shape"],
        }
        if actual_spec != expected_spec:
            raise ValueError(f"{output_path}: {key} expected {expected_spec}, got {actual_spec}")
        payload_bytes += actual["data_offsets"][1] - actual["data_offsets"][0]
    return len(expected), payload_bytes


def process_shard(
    shard_file: str,
    shard_keys: list[str],
    input_dir: str,
    output_dir: str,
) -> dict:
    """Dequantize one shard and atomically publish it."""

    source_path = Path(input_dir) / shard_file
    output_path = Path(output_dir) / shard_file
    temp_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    if temp_path.exists():
        temp_path.unlink()

    started = time.time()
    output_tensors = {}
    dequantized = 0
    copied = 0
    with safe_open(source_path, framework="paddle", device="cpu") as source:
        for key in shard_keys:
            mapped = output_key(key)
            if mapped is None:
                continue
            if key.endswith(".weight_packed"):
                scale_key = key.removesuffix(".weight_packed") + ".weight_scale"
                packed = source.get_tensor(key)
                scale = source.get_tensor(scale_key)
                output_tensors[mapped] = fp4_weight_to_bf16(packed, scale)
                del packed, scale
                dequantized += 1
            else:
                output_tensors[mapped] = source.get_tensor(key)
                copied += 1

    save_file(output_tensors, temp_path, metadata={"format": "pt"})
    del output_tensors
    gc.collect()

    tensor_count, payload_bytes = validate_output_shard(
        source_path,
        temp_path,
        shard_keys,
    )
    os.replace(temp_path, output_path)
    return {
        "shard": shard_file,
        "dequantized": dequantized,
        "copied": copied,
        "tensors": tensor_count,
        "payload_bytes": payload_bytes,
        "file_bytes": output_path.stat().st_size,
        "seconds": time.time() - started,
    }


def worker_init(threads_per_worker: int) -> None:
    paddle.set_device("cpu")
    paddle.set_flags({"FLAGS_paddle_num_threads": threads_per_worker})


def write_json_atomic(path: Path, value: dict) -> None:
    temp_path = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp_path.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dequantize Kimi-K3 MXFP4 routed-expert weights to BF16")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threads_per_worker", type=int, default=8)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start_shard", type=int, default=0)
    parser.add_argument(
        "--end_shard",
        type=int,
        default=None,
        help="Exclusive 0-based shard boundary; partial runs do not finalize index/config",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--log_file", default=None)
    return parser.parse_args()


def main() -> None:
    global _LOG_HANDLE
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("input_dir and output_dir must be different")
    if args.num_workers < 1 or args.threads_per_worker < 1:
        raise ValueError("num_workers and threads_per_worker must be positive")
    if args.start_shard < 0:
        raise ValueError("start_shard must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _LOG_HANDLE = log_path.open("a", buffering=1)

    index_path = input_dir / "model.safetensors.index.json"
    with index_path.open() as handle:
        index = json.load(handle)
    weight_map = index["weight_map"]

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard_file in weight_map.items():
        shard_to_keys.setdefault(shard_file, []).append(key)
    for keys in shard_to_keys.values():
        keys.sort()

    sorted_shards = sorted(shard_to_keys)
    total_shards = len(sorted_shards)
    if args.end_shard is None:
        args.end_shard = total_shards
    if not 0 <= args.start_shard <= args.end_shard <= total_shards:
        raise ValueError(f"invalid shard range [{args.start_shard}, {args.end_shard}) " f"for {total_shards} shards")

    packed_keys = {key for key in weight_map if key.endswith(".weight_packed")}
    scale_keys = {key for key in weight_map if key.endswith(".weight_scale")}
    if len(packed_keys) != len(scale_keys):
        raise ValueError(f"packed/scale count mismatch: {len(packed_keys)} vs {len(scale_keys)}")

    expected_tensor_count = 0
    expected_payload_bytes = 0
    for shard_file in sorted_shards:
        source_path = input_dir / shard_file
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_header = read_safetensors_header(source_path)
        shard_keys = shard_to_keys[shard_file]
        if set(source_header) != set(shard_keys):
            raise ValueError(f"{shard_file}: index/header key mismatch")
        for key in shard_keys:
            spec = expected_output_spec(key, source_header[key], source_header)
            if spec is None:
                continue
            expected_tensor_count += 1
            numel = math.prod(spec["shape"])
            dtype_bytes = {"BF16": 2, "F32": 4}.get(spec["dtype"])
            if dtype_bytes is None:
                source_spec = source_header[key]
                expected_payload_bytes += source_spec["data_offsets"][1] - source_spec["data_offsets"][0]
            else:
                expected_payload_bytes += numel * dtype_bytes

    emit("=" * 72)
    emit("Kimi-K3 MXFP4 -> BF16 dequantization")
    emit("=" * 72)
    emit(f"Input:                    {input_dir}")
    emit(f"Output:                   {output_dir}")
    emit(f"Source shards:            {total_shards}")
    emit(f"Packed MXFP4 weights:     {len(packed_keys)}")
    emit(f"Scale tensors to drop:    {len(scale_keys)}")
    emit(f"Expected output tensors:  {expected_tensor_count}")
    emit(f"Expected payload bytes:   {expected_payload_bytes}")
    emit(f"Expected payload TiB:     {expected_payload_bytes / 2**40:.3f}")
    emit(f"Mode:                     " f"{'sequential' if args.sequential else f'{args.num_workers} workers'}")
    emit(f"Shard range:              [{args.start_shard}, {args.end_shard})")
    emit("")

    if args.dry_run:
        emit("Dry run completed; no weight files were written.")
        return

    valid_shards = set()
    for shard_file in sorted_shards:
        output_path = output_dir / shard_file
        if not output_path.exists():
            continue
        try:
            validate_output_shard(
                input_dir / shard_file,
                output_path,
                shard_to_keys[shard_file],
            )
        except Exception:
            if not args.overwrite:
                raise
            output_path.unlink()
        else:
            valid_shards.add(shard_file)

    emit(f"Already valid output shards: {len(valid_shards)} / {total_shards}")
    if args.verify_only:
        if len(valid_shards) != total_shards:
            raise RuntimeError(f"verification incomplete: {len(valid_shards)} / {total_shards}")
        emit("All output shards passed header verification.")
        return

    selected = sorted_shards[args.start_shard : args.end_shard]
    tasks = []
    for shard_file in selected:
        if shard_file in valid_shards:
            if args.resume:
                emit(f"SKIP {shard_file}: existing output is valid")
                continue
            raise FileExistsError(f"{output_dir / shard_file} already exists; use --resume")
        tasks.append((shard_file, shard_to_keys[shard_file]))

    completed_dequant = 0
    if args.sequential:
        worker_init(args.threads_per_worker)
        for task_index, (shard_file, shard_keys) in enumerate(tasks, 1):
            emit(f"START [{task_index}/{len(tasks)}] {shard_file}")
            result = process_shard(
                shard_file,
                shard_keys,
                str(input_dir),
                str(output_dir),
            )
            completed_dequant += result["dequantized"]
            emit(
                f"DONE  {shard_file}: {result['dequantized']} dequant, "
                f"{result['copied']} copy, {result['file_bytes'] / 2**30:.3f} GiB, "
                f"{result['seconds']:.1f}s"
            )
    elif tasks:
        emit(f"Submitting {len(tasks)} shards to {args.num_workers} workers")
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=worker_init,
            initargs=(args.threads_per_worker,),
        ) as executor:
            futures = {
                executor.submit(
                    process_shard,
                    shard_file,
                    shard_keys,
                    str(input_dir),
                    str(output_dir),
                ): shard_file
                for shard_file, shard_keys in tasks
            }
            done = 0
            for future in as_completed(futures):
                shard_file = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    emit(f"FAILED {shard_file}: {error!r}")
                    raise
                done += 1
                completed_dequant += result["dequantized"]
                emit(
                    f"DONE [{done}/{len(tasks)}] {shard_file}: "
                    f"{result['dequantized']} dequant, {result['copied']} copy, "
                    f"{result['file_bytes'] / 2**30:.3f} GiB, "
                    f"{result['seconds']:.1f}s"
                )

    emit(f"Newly dequantized weights: {completed_dequant}")

    final_tensor_count = 0
    final_payload_bytes = 0
    complete = True
    for shard_file in sorted_shards:
        output_path = output_dir / shard_file
        if not output_path.exists():
            complete = False
            continue
        count, payload = validate_output_shard(
            input_dir / shard_file,
            output_path,
            shard_to_keys[shard_file],
        )
        final_tensor_count += count
        final_payload_bytes += payload

    if not complete:
        emit("Partial conversion completed; index/config were not finalized.")
        return
    if final_tensor_count != expected_tensor_count or final_payload_bytes != expected_payload_bytes:
        raise RuntimeError(
            "final inventory mismatch: "
            f"tensors {final_tensor_count}/{expected_tensor_count}, "
            f"bytes {final_payload_bytes}/{expected_payload_bytes}"
        )

    new_weight_map = {}
    for key, shard_file in weight_map.items():
        mapped = output_key(key)
        if mapped is not None:
            if mapped in new_weight_map:
                raise KeyError(f"duplicate output key: {mapped}")
            new_weight_map[mapped] = shard_file
    new_index = {
        "metadata": {
            **index.get("metadata", {}),
            "total_size": final_payload_bytes,
        },
        "weight_map": new_weight_map,
    }
    write_json_atomic(output_dir / "model.safetensors.index.json", new_index)

    with (input_dir / "config.json").open() as handle:
        config = json.load(handle)
    config.pop("quantization_config", None)
    config.pop("expert_dtype", None)
    config["dtype"] = "bfloat16"
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config.pop("quantization_config", None)
        text_config.pop("expert_dtype", None)
        text_config["dtype"] = "bfloat16"
    write_json_atomic(output_dir / "config.json", config)

    emit("")
    emit("=" * 72)
    emit("Conversion completed and finalized")
    emit(f"Output shards:            {total_shards}")
    emit(f"Output tensors:           {final_tensor_count}")
    emit(f"Output payload bytes:     {final_payload_bytes}")
    emit(f"Output payload TiB:       {final_payload_bytes / 2**40:.3f}")
    emit(f"Weight-map entries:       {len(new_weight_map)}")
    emit(f"BF16 model directory:     {output_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if _LOG_HANDLE is not None:
            _LOG_HANDLE.close()
