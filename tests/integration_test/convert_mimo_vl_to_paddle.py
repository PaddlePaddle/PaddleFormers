#!/usr/bin/env python3
"""Convert MiMo-VL (HF/PyTorch checkpoint) into Paddle checkpoint.

Example:
  python tests/integration_test/convert_mimo_vl_to_paddle.py \
    --source-dir /path/to/MiMo-VL-7B-SFT \
    --output-dir /path/to/MiMo-VL-7B-SFT-paddle
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HF MiMo-VL/Qwen2.5-VL weights to Paddle format.")
    parser.add_argument("--source-dir", required=True, help="HF checkpoint directory (contains config.json + *.safetensors)")
    parser.add_argument("--output-dir", default=None, help="Output Paddle checkpoint directory")
    parser.add_argument(
        "--paddleformers-root",
        default=None,
        help="Path to local PaddleFormers repo. Defaults to this repository root.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Load dtype for conversion",
    )
    parser.add_argument("--max-shard-size", default="5GB", help="Max shard size for saving safetensors")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", help="Enable low CPU memory loading")
    parser.add_argument(
        "--load-checkpoint-format",
        default="legacy",
        choices=["legacy", "flex_checkpoint"],
        help="Checkpoint loading backend in PaddleFormers. Use legacy to avoid FlexCheckpoint OOM in conversion.",
    )
    parser.add_argument(
        "--load-via-cpu",
        action="store_true",
        help="Offload checkpoint loading to CPU when backend supports it (effective for flex_checkpoint).",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["cpu", "gpu"],
        help="Device to initialize/load model on during conversion.",
    )
    parser.add_argument("--no-safe-serialization", action="store_true", help="Save as .pdparams instead of safetensors")
    parser.add_argument(
        "--save-to-hf-format",
        action="store_true",
        help="Save checkpoint in HF-compatible tensor orientation (transpose-selected). Default saves Paddle-native orientation.",
    )
    parser.add_argument("--no-copy-aux-files", action="store_true", help="Do not copy tokenizer/processor files")
    parser.add_argument("--verify-paddle-load", action="store_true", help="Load converted model once for smoke check")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it already exists")
    return parser.parse_args()


def add_local_paddleformers(paddleformers_root: Path) -> None:
    if paddleformers_root.exists() and str(paddleformers_root) not in sys.path:
        sys.path.insert(0, str(paddleformers_root))


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_output_dir(source_dir: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()
    return source_dir.parent / f"{source_dir.name}-paddle"


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_auxiliary_files(source_dir: Path, output_dir: Path) -> list[str]:
    """Copy tokenizer/processor metadata files that model.save_pretrained does not always include."""
    candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.json",
        "preprocessor_config.json",
        "generation_config.json",
        "README.md",
    ]
    copied = []
    for name in candidates:
        src = source_dir / name
        dst = output_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def write_paddle_configuration_json(output_dir: Path) -> None:
    meta_path = output_dir / "configuration.json"
    meta = {
        "framework": "paddle",
        "task": "text-generation",
        "allow_remote": True,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def detect_output_checkpoint_style(output_dir: Path) -> str:
    """
    Detect key style of saved checkpoint.
    Returns:
        "hf_keys"      -> keys like model.layers.*, visual.blocks.*
        "paddle_keys"  -> keys like model.language_model.*, model.visual.*
        "unknown"      -> cannot decide
    """
    index_path = output_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return "unknown"
    try:
        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        keys = list(weight_map.keys())
        if not keys:
            return "unknown"
        sample_keys = keys[: min(200, len(keys))]
        has_hf_keys = any(k.startswith("model.layers.") or k.startswith("visual.blocks.") for k in sample_keys)
        has_paddle_keys = any(k.startswith("model.language_model.") or k.startswith("model.visual.") for k in sample_keys)
        if has_hf_keys and not has_paddle_keys:
            return "hf_keys"
        if has_paddle_keys and not has_hf_keys:
            return "paddle_keys"
    except Exception:
        return "unknown"
    return "unknown"


def convert_model(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not (source_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in source directory: {source_dir}")

    output_dir = resolve_output_dir(source_dir, args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)

    paddleformers_root = Path(args.paddleformers_root).resolve() if args.paddleformers_root else default_repo_root()
    add_local_paddleformers(paddleformers_root)

    import paddle
    from paddleformers.transformers import Qwen2_5_VLForConditionalGeneration

    if args.device == "cpu":
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu")

    load_kwargs = {
        "convert_from_hf": True,
        "use_safetensors": True,
        "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
        "load_checkpoint_format": args.load_checkpoint_format,
        "load_via_cpu": bool(args.load_via_cpu),
    }
    if args.dtype != "auto":
        load_kwargs["dtype"] = args.dtype

    print("[1/4] Loading HF checkpoint and converting to Paddle tensors ...")
    print(f"       source={source_dir}")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(source_dir), **load_kwargs)
    t1 = time.time()
    print(f"       done in {t1 - t0:.1f}s")

    print("[2/4] Saving Paddle checkpoint ...")
    save_kwargs = {
        "safe_serialization": not args.no_safe_serialization,
        "max_shard_size": args.max_shard_size,
        "save_to_hf": bool(args.save_to_hf_format),
    }
    model.save_pretrained(str(output_dir), **save_kwargs)
    t2 = time.time()
    print(f"       saved to {output_dir}")
    print(f"       done in {t2 - t1:.1f}s")

    print("[3/4] Writing Paddle metadata and copying auxiliary files ...")
    write_paddle_configuration_json(output_dir)
    copied = []
    if not args.no_copy_aux_files:
        copied = copy_auxiliary_files(source_dir, output_dir)
    print(f"       copied_aux_files={copied if copied else '[]'}")

    if args.verify_paddle_load:
        # Release memory from the first loaded model before verify load.
        del model
        gc.collect()
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()

        print("[4/4] Verifying converted Paddle checkpoint can be loaded ...")
        ckpt_style = detect_output_checkpoint_style(output_dir)
        # VLM save_pretrained in PaddleFormers commonly writes HF-style keys.
        if ckpt_style == "hf_keys":
            verify_convert_from_hf = True
        elif ckpt_style == "paddle_keys":
            verify_convert_from_hf = False
        else:
            verify_convert_from_hf = bool(args.save_to_hf_format)
        print(f"       detected_checkpoint_style={ckpt_style}, verify_convert_from_hf={verify_convert_from_hf}")

        verify_kwargs = {
            "convert_from_hf": verify_convert_from_hf,
            "use_safetensors": not args.no_safe_serialization,
            "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
            "load_checkpoint_format": args.load_checkpoint_format,
            "load_via_cpu": bool(args.load_via_cpu),
        }
        if args.dtype != "auto":
            verify_kwargs["dtype"] = args.dtype
        _ = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(output_dir), **verify_kwargs)
        print("       verify_load=PASS")
    else:
        print("[4/4] Skipped verify load (use --verify-paddle-load to enable)")

    print("\nConversion finished successfully.")
    print(f"Output: {output_dir}")


def main() -> None:
    args = parse_args()
    convert_model(args)


if __name__ == "__main__":
    main()
