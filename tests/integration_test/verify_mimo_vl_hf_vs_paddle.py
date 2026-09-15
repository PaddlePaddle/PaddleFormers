#!/usr/bin/env python3
"""Compare HF and Paddle forward logits for MiMo-VL/Qwen2.5-VL (text-only).

This script uses the same tokenized text input for both models and compares logits.

Example:
  python tests/integration_test/verify_mimo_vl_hf_vs_paddle.py \
    --hf-dir /path/to/MiMo-VL-7B-SFT \
    --paddle-dir /path/to/MiMo-VL-7B-SFT-paddle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify HF vs Paddle logits consistency (text-only).")
    parser.add_argument("--hf-dir", required=True, help="HF checkpoint directory")
    parser.add_argument("--paddle-dir", required=True, help="Converted Paddle checkpoint directory")
    parser.add_argument(
        "--paddleformers-root",
        default=None,
        help="Path to local PaddleFormers repo. Defaults to this repository root.",
    )
    parser.add_argument("--prompt", default="Please briefly introduce yourself in one sentence.", help="Input prompt")
    parser.add_argument("--max-length", type=int, default=64, help="Max token length for verification input")
    parser.add_argument(
        "--hf-dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="HF model load dtype",
    )
    parser.add_argument(
        "--paddle-dtype",
        default="float32",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Paddle model load dtype",
    )
    parser.add_argument(
        "--paddle-convert-from-hf",
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether Paddle from_pretrained should treat checkpoint as HF-style keys",
    )
    parser.add_argument(
        "--paddle-load-checkpoint-format",
        default="legacy",
        choices=["legacy", "flex_checkpoint"],
        help="Paddle checkpoint loader backend",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["cpu", "gpu"],
        help="Run forward on cpu or gpu",
    )
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for np.allclose")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for np.allclose")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", help="Enable low CPU memory loading for Paddle")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero code if allclose is False")
    return parser.parse_args()


def add_local_paddleformers(paddleformers_root: Path) -> None:
    if paddleformers_root.exists() and str(paddleformers_root) not in sys.path:
        sys.path.insert(0, str(paddleformers_root))


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def detect_output_checkpoint_style(model_dir: Path) -> str:
    """
    Returns:
        hf_keys: keys like model.layers.*, visual.blocks.*
        paddle_keys: keys like model.language_model.*, model.visual.*
        unknown: cannot determine
    """
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return "unknown"
    try:
        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        keys = list(index_data.get("weight_map", {}).keys())
        if not keys:
            return "unknown"
        sample = keys[: min(200, len(keys))]
        has_hf = any(k.startswith("model.layers.") or k.startswith("visual.blocks.") for k in sample)
        has_pd = any(k.startswith("model.language_model.") or k.startswith("model.visual.") for k in sample)
        if has_hf and not has_pd:
            return "hf_keys"
        if has_pd and not has_hf:
            return "paddle_keys"
    except Exception:
        return "unknown"
    return "unknown"


def load_hf_logits(args: argparse.Namespace, input_ids_np: np.ndarray, attention_mask_np: np.ndarray) -> np.ndarray:
    import torch
    import importlib.metadata as importlib_metadata

    # Some transformers versions unconditionally query torchcodec metadata when importing model classes.
    # If torchcodec is absent, patch version lookup to avoid import failure for text-only verification.
    _orig_version = importlib_metadata.version

    def _safe_version(name: str) -> str:
        if name == "torchcodec":
            try:
                return _orig_version(name)
            except importlib_metadata.PackageNotFoundError:
                return "0.0.0"
        return _orig_version(name)

    importlib_metadata.version = _safe_version
    from transformers import Qwen2_5_VLForConditionalGeneration

    hf_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    hf_dtype = hf_dtype_map[args.hf_dtype]

    print("[HF] Loading model ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.hf_dir),
        torch_dtype=hf_dtype,
    )
    model.eval()
    if args.device == "gpu" and torch.cuda.is_available():
        model = model.to("cuda:0")
        device = "cuda:0"
    else:
        device = "cpu"
    t1 = time.time()
    print(f"[HF] Loaded in {t1 - t0:.1f}s")

    input_ids_t = torch.from_numpy(input_ids_np).to(device)
    attention_mask_t = torch.from_numpy(attention_mask_np).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t, return_dict=True)
        logits = outputs.logits.float().cpu().numpy()
    return logits


def load_paddle_logits(args: argparse.Namespace, input_ids_np: np.ndarray, attention_mask_np: np.ndarray) -> np.ndarray:
    import paddle
    from paddleformers.transformers import Qwen2_5_VLForConditionalGeneration

    if args.device == "gpu":
        paddle.set_device("gpu:0")
    else:
        paddle.set_device("cpu")

    if args.paddle_convert_from_hf == "auto":
        style = detect_output_checkpoint_style(Path(args.paddle_dir))
        convert_from_hf = style == "hf_keys"
        print(f"[Paddle] detected_checkpoint_style={style}, convert_from_hf={convert_from_hf}")
    else:
        convert_from_hf = args.paddle_convert_from_hf == "true"

    load_kwargs = {
        "convert_from_hf": convert_from_hf,
        "use_safetensors": True,
        "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
        "load_checkpoint_format": args.paddle_load_checkpoint_format,
    }
    if args.paddle_dtype != "auto":
        load_kwargs["dtype"] = args.paddle_dtype

    print("[Paddle] Loading model ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.paddle_dir), **load_kwargs)
    model.eval()
    t1 = time.time()
    print(f"[Paddle] Loaded in {t1 - t0:.1f}s")

    input_ids_t = paddle.to_tensor(input_ids_np, dtype="int64")
    attention_mask_t = paddle.to_tensor(attention_mask_np, dtype="int64")

    with paddle.no_grad():
        outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t, return_dict=True)
        logits = outputs.logits.astype("float32").numpy()
    return logits


def summarize_diff(hf_logits: np.ndarray, pd_logits: np.ndarray, attention_mask_np: np.ndarray, atol: float, rtol: float) -> dict:
    if hf_logits.shape != pd_logits.shape:
        raise ValueError(f"Shape mismatch: hf={hf_logits.shape}, paddle={pd_logits.shape}")

    diff = np.abs(hf_logits - pd_logits)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())
    p99_abs_diff = float(np.quantile(diff, 0.99))
    allclose = bool(np.allclose(hf_logits, pd_logits, atol=atol, rtol=rtol))

    last_idx = int(attention_mask_np[0].sum() - 1)
    hf_top1 = int(np.argmax(hf_logits[0, last_idx]))
    pd_top1 = int(np.argmax(pd_logits[0, last_idx]))

    return {
        "shape": list(hf_logits.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "p99_abs_diff": p99_abs_diff,
        "allclose": allclose,
        "atol": atol,
        "rtol": rtol,
        "last_token_index": last_idx,
        "hf_top1_token_id": hf_top1,
        "paddle_top1_token_id": pd_top1,
        "top1_match": hf_top1 == pd_top1,
    }


def main() -> None:
    args = parse_args()
    hf_dir = Path(args.hf_dir).resolve()
    paddle_dir = Path(args.paddle_dir).resolve()
    paddleformers_root = Path(args.paddleformers_root).resolve() if args.paddleformers_root else default_repo_root()
    add_local_paddleformers(paddleformers_root)

    if not hf_dir.exists():
        raise FileNotFoundError(f"HF directory not found: {hf_dir}")
    if not paddle_dir.exists():
        raise FileNotFoundError(f"Paddle directory not found: {paddle_dir}")

    print("[Tokenizer] Loading HF tokenizer ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir), use_fast=False)
    batch = tokenizer(
        args.prompt,
        return_tensors="np",
        max_length=args.max_length,
        truncation=True,
    )

    input_ids_np = batch["input_ids"].astype("int64")
    attention_mask_np = batch["attention_mask"].astype("int64")

    hf_logits = load_hf_logits(args, input_ids_np, attention_mask_np)
    pd_logits = load_paddle_logits(args, input_ids_np, attention_mask_np)

    summary = summarize_diff(
        hf_logits=hf_logits,
        pd_logits=pd_logits,
        attention_mask_np=attention_mask_np,
        atol=args.atol,
        rtol=args.rtol,
    )

    print("\n===== Verification Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.strict and not summary["allclose"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
