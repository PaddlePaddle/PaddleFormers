#!/usr/bin/env python3
"""Export and compare deterministic Gemma3 multimodal precision traces.

Torch and Paddle normally live in separate environments, so framework runs
write self-contained NPZ/JSON artifacts and comparison is offline::

    python compare_gemma3_multimodal_precision.py torch --model MODEL \
        --batch /tmp/gemma3-mm-batch.npz --output /tmp/gemma3-torch.npz
    python compare_gemma3_multimodal_precision.py paddle --model MODEL \
        --batch /tmp/gemma3-mm-batch.npz --output /tmp/gemma3-paddle.npz
    python compare_gemma3_multimodal_precision.py compare \
        --torch-output /tmp/gemma3-torch.npz \
        --paddle-output /tmp/gemma3-paddle.npz

Both exporters use FP32, eval mode, eager attention, fixed inputs, and precise
FP32 GEMM.  Large intermediate attention matrices are sampled at fixed query
positions; every vision layer's residual output is sampled as well.  Full
image features, final hidden states, and logits are retained for acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


LINEAR_WEIGHT_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for framework in ("torch", "paddle"):
        subparser = subparsers.add_parser(framework, help=f"export a {framework} trace")
        subparser.add_argument("--model", required=True, type=Path)
        subparser.add_argument("--batch", required=True, type=Path)
        subparser.add_argument("--output", required=True, type=Path)
        subparser.add_argument("--device", default="cuda:0")
        subparser.add_argument("--seed", type=int, default=2026)
        subparser.add_argument("--trace-layer", type=int, default=0)
        subparser.add_argument("--sample-tokens", default="0,1,2048,4095")
        subparser.add_argument(
            "--reuse-batch",
            action="store_true",
            help="in torch mode, reuse an existing batch instead of replacing it",
        )

    compare = subparsers.add_parser("compare", help="compare exported artifacts")
    compare.add_argument("--torch-output", required=True, type=Path)
    compare.add_argument("--paddle-output", required=True, type=Path)
    compare.add_argument("--vision-tolerance", type=float, default=2e-4)
    compare.add_argument("--logits-tolerance", type=float, default=1e-2)
    return parser.parse_args()


def _sha256(array) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _array_summary(array) -> dict:
    import numpy as np

    value = np.asarray(array)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "mean": float(value.astype(np.float64).mean()),
        "std": float(value.astype(np.float64).std()),
        "sha256": _sha256(value),
    }


def _write_metadata(output: Path, metadata: dict) -> None:
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _make_batch(config, batch_path: Path, seed: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    image_tokens = int(config.mm_tokens_per_image)
    seq_length = image_tokens + 9
    vocab_size = int(config.text_config.vocab_size)
    image_token_id = getattr(config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = config.image_token_index
    image_token_id = int(image_token_id)
    input_ids = (np.arange(seq_length, dtype=np.int64)[None, :] + 10) % vocab_size
    input_ids[0, 0] = int(config.text_config.bos_token_id)
    input_ids[0, 1 : image_tokens + 1] = image_token_id
    token_type_ids = np.zeros_like(input_ids)
    token_type_ids[0, 1 : image_tokens + 1] = 1
    attention_mask = np.ones_like(input_ids)
    image_size = int(config.vision_config.image_size)
    pixel_values = rng.standard_normal((1, 3, image_size, image_size), dtype=np.float32)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        batch_path,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        pixel_values=pixel_values,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
    }


def _load_batch(batch_path: Path) -> dict:
    import numpy as np

    if not batch_path.exists():
        raise FileNotFoundError(f"Batch does not exist; run torch mode first: {batch_path}")
    with np.load(batch_path) as batch:
        return {name: batch[name] for name in batch.files}


def _sample_indices(spec: str, sequence_length: int) -> list[int]:
    indices = sorted({int(value) for value in spec.split(",")})
    if not indices or indices[0] < 0 or indices[-1] >= sequence_length:
        raise ValueError(f"sample tokens must be within [0, {sequence_length}), got {indices}")
    return indices


def _canonical_weight_name(name: str, framework: str) -> str | None:
    if framework == "torch":
        if name.startswith("model.vision_tower.vision_model.") or name.startswith("model.multi_modal_projector."):
            return name[len("model.") :]
        return None
    if name.startswith("model.vision_tower."):
        return "vision_tower.vision_model." + name[len("model.vision_tower.") :]
    if name.startswith("model.multi_modal_projector."):
        return name[len("model.") :]
    return None


def _weight_manifest(state_dict, framework: str) -> dict:
    manifest = {}
    for name, tensor in state_dict.items():
        canonical_name = _canonical_weight_name(name, framework)
        if canonical_name is None:
            continue
        if framework == "torch":
            array = tensor.detach().float().cpu().numpy()
        else:
            array = tensor.astype("float32").cpu().numpy()
            if canonical_name.endswith(LINEAR_WEIGHT_SUFFIXES):
                array = array.T
        manifest[canonical_name] = _array_summary(array)
    return manifest


def _metadata(framework: str, args: argparse.Namespace, batch: dict, versions: dict, weights: dict) -> dict:
    return {
        "framework": framework,
        "versions": versions,
        "python": platform.python_version(),
        "model": str(args.model.resolve()),
        "device": args.device,
        "dtype": "float32",
        "seed": args.seed,
        "trace_layer": args.trace_layer,
        "sample_tokens": args.sample_tokens,
        "precision": {
            "torch_float32_matmul_precision": "highest" if framework == "torch" else None,
            "paddle_allow_tf32_cublas": False if framework == "paddle" else None,
        },
        "inputs": {name: _array_summary(value) for name, value in batch.items()},
        "weights": weights,
    }


def _run_torch(args: argparse.Namespace) -> None:  # pragma: no cover - separate Torch environment
    import numpy as np
    import torch
    import transformers
    from transformers import Gemma3ForConditionalGeneration

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    config = transformers.Gemma3Config.from_pretrained(str(args.model))
    if not args.reuse_batch or not args.batch.exists():
        batch = _make_batch(config, args.batch, args.seed)
    else:
        batch = _load_batch(args.batch)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(args.model),
        config=config,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    vision = model.model.vision_tower.vision_model
    trace_layer = vision.encoder.layers[args.trace_layer]
    indices = _sample_indices(args.sample_tokens, vision.embeddings.num_patches)
    captured = {}
    handles = []

    def save_post(name, keep_full=False):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value if keep_full else value[:, indices]

        return hook

    def save_pre(name):
        def hook(_module, inputs):
            captured[name] = inputs[0][:, indices]

        return hook

    def save_mask(name):
        def hook(_module, _inputs, kwargs):
            captured[name] = kwargs["attention_mask"]

        return hook

    handles.append(vision.embeddings.register_forward_hook(save_post("vision.embedding")))
    for layer_index, layer in enumerate(vision.encoder.layers):
        handles.append(layer.register_forward_hook(save_post(f"vision.layer.{layer_index:02d}.output")))
    handles += [
        trace_layer.layer_norm1.register_forward_hook(save_post("trace.ln1")),
        trace_layer.self_attn.q_proj.register_forward_hook(save_post("trace.q", keep_full=True)),
        trace_layer.self_attn.k_proj.register_forward_hook(save_post("trace.k", keep_full=True)),
        trace_layer.self_attn.v_proj.register_forward_hook(save_post("trace.v", keep_full=True)),
        trace_layer.self_attn.out_proj.register_forward_pre_hook(save_pre("trace.context")),
        trace_layer.self_attn.out_proj.register_forward_hook(save_post("trace.attn_out")),
        trace_layer.layer_norm2.register_forward_hook(save_post("trace.ln2")),
        trace_layer.mlp.fc1.register_forward_hook(save_post("trace.fc1")),
        trace_layer.mlp.fc2.register_forward_hook(save_post("trace.mlp")),
        vision.post_layernorm.register_forward_hook(save_post("vision.final")),
    ]
    language_layers = model.model.language_model.layers
    handles += [
        language_layers[0].register_forward_pre_hook(save_mask("mask.sliding"), with_kwargs=True),
        language_layers[5].register_forward_pre_hook(save_mask("mask.full"), with_kwargs=True),
    ]

    torch_batch = {
        "input_ids": torch.from_numpy(batch["input_ids"]).to(args.device),
        "attention_mask": torch.from_numpy(batch["attention_mask"]).to(args.device),
        "token_type_ids": torch.from_numpy(batch["token_type_ids"]).to(args.device),
        "pixel_values": torch.from_numpy(batch["pixel_values"]).to(args.device),
    }
    with torch.no_grad():
        outputs = model(**torch_batch, use_cache=False, output_hidden_states=True, return_dict=True)
        num_heads = vision.config.num_attention_heads
        head_dim = vision.config.hidden_size // num_heads
        q = captured.pop("trace.q").view(1, -1, num_heads, head_dim).transpose(1, 2)
        k = captured.pop("trace.k").view(1, -1, num_heads, head_dim).transpose(1, 2)
        v = captured.pop("trace.v").view(1, -1, num_heads, head_dim).transpose(1, 2)
        query_indices = torch.tensor(indices, device=q.device)
        sampled_q = q.index_select(2, query_indices)
        logits = torch.matmul(sampled_q, k.transpose(-1, -2)) * trace_layer.self_attn.scale
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        context = torch.matmul(probabilities, v).transpose(1, 2).reshape(1, len(indices), -1)
        captured["trace.q"] = sampled_q.transpose(1, 2).reshape(1, len(indices), -1)
        sampled_k = torch.index_select(k, 2, query_indices)
        sampled_v = torch.index_select(v, 2, query_indices)
        captured["trace.k"] = sampled_k.transpose(1, 2).reshape(1, len(indices), -1)
        captured["trace.v"] = sampled_v.transpose(1, 2).reshape(1, len(indices), -1)
        captured["trace.attn_logits"] = logits
        captured["trace.softmax"] = probabilities
        captured["trace.context_recomputed"] = context
        captured["trace.gelu"] = torch.nn.functional.gelu(captured["trace.fc1"], approximate="tanh")
        captured["image_features"] = outputs.image_hidden_states
        captured["final_hidden"] = outputs.hidden_states[-1]
        captured["logits"] = outputs.logits

    for handle in handles:
        handle.remove()
    arrays = {name: value.detach().float().cpu().numpy() for name, value in captured.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    versions = {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__}
    metadata = _metadata("torch", args, batch, versions, _weight_manifest(model.state_dict(), "torch"))
    metadata["outputs"] = {name: _array_summary(value) for name, value in arrays.items()}
    _write_metadata(args.output, metadata)
    print(json.dumps({"framework": "torch", "output": str(args.output), "arrays": len(arrays)}))


def _run_paddle(args: argparse.Namespace) -> None:  # pragma: no cover - separate Paddle environment
    import numpy as np
    import paddle

    from paddle.base import core
    from paddleformers.transformers import Gemma3ForConditionalGeneration

    paddle.seed(args.seed)
    paddle.set_device(args.device.replace("cuda", "gpu"))
    # Start from Paddle's default to verify that the model selects precise
    # FP32 GEMM itself before entering either backbone.
    core.set_cublas_switch(True)
    batch = _load_batch(args.batch)
    model = Gemma3ForConditionalGeneration.from_pretrained(str(args.model), dtype="float32")
    model.eval()
    vision = model.model.vision_tower
    trace_layer = vision.encoder.layers[args.trace_layer]
    indices = _sample_indices(args.sample_tokens, vision.embeddings.num_patches)
    captured = {}
    handles = []

    def save_post(name, keep_full=False):
        def hook(_layer, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value if keep_full else value[:, indices]

        return hook

    def save_pre(name):
        def hook(_layer, inputs):
            captured[name] = inputs[0][:, indices]

        return hook

    def save_mask(name):
        def hook(_layer, _inputs, kwargs):
            captured[name] = kwargs["attention_mask"]

        return hook

    handles.append(vision.embeddings.register_forward_post_hook(save_post("vision.embedding")))
    for layer_index, layer in enumerate(vision.encoder.layers):
        handles.append(layer.register_forward_post_hook(save_post(f"vision.layer.{layer_index:02d}.output")))
    handles += [
        trace_layer.layer_norm1.register_forward_post_hook(save_post("trace.ln1")),
        trace_layer.self_attn.q_proj.register_forward_post_hook(save_post("trace.q", keep_full=True)),
        trace_layer.self_attn.k_proj.register_forward_post_hook(save_post("trace.k", keep_full=True)),
        trace_layer.self_attn.v_proj.register_forward_post_hook(save_post("trace.v", keep_full=True)),
        trace_layer.self_attn.out_proj.register_forward_pre_hook(save_pre("trace.context")),
        trace_layer.self_attn.out_proj.register_forward_post_hook(save_post("trace.attn_out")),
        trace_layer.layer_norm2.register_forward_post_hook(save_post("trace.ln2")),
        trace_layer.mlp.fc1.register_forward_post_hook(save_post("trace.fc1")),
        trace_layer.mlp.fc2.register_forward_post_hook(save_post("trace.mlp")),
        vision.post_layernorm.register_forward_post_hook(save_post("vision.final")),
    ]
    language_layers = model.model.language_model.layers
    handles += [
        language_layers[0].register_forward_pre_hook(save_mask("mask.sliding"), with_kwargs=True),
        language_layers[5].register_forward_pre_hook(save_mask("mask.full"), with_kwargs=True),
    ]

    paddle_batch = {
        "input_ids": paddle.to_tensor(batch["input_ids"], dtype="int64"),
        "attention_mask": paddle.to_tensor(batch["attention_mask"], dtype="int64"),
        "token_type_ids": paddle.to_tensor(batch["token_type_ids"], dtype="int64"),
        "pixel_values": paddle.to_tensor(batch["pixel_values"], dtype="float32"),
    }
    with paddle.no_grad():
        outputs = model(**paddle_batch, use_cache=False, output_hidden_states=True, return_dict=True)
        num_heads = vision.config.num_attention_heads
        head_dim = vision.config.hidden_size // num_heads
        q = captured.pop("trace.q").reshape([1, -1, num_heads, head_dim]).transpose([0, 2, 1, 3])
        k = captured.pop("trace.k").reshape([1, -1, num_heads, head_dim]).transpose([0, 2, 1, 3])
        v = captured.pop("trace.v").reshape([1, -1, num_heads, head_dim]).transpose([0, 2, 1, 3])
        sampled_q = paddle.index_select(q, paddle.to_tensor(indices), axis=2)
        logits = paddle.matmul(sampled_q, k.transpose([0, 1, 3, 2])) * trace_layer.self_attn.scale
        probabilities = paddle.nn.functional.softmax(logits, axis=-1, dtype="float32")
        context = paddle.matmul(probabilities, v).transpose([0, 2, 1, 3]).reshape([1, len(indices), -1])
        captured["trace.q"] = sampled_q.transpose([0, 2, 1, 3]).reshape([1, len(indices), -1])
        sampled_k = paddle.index_select(k, paddle.to_tensor(indices), axis=2)
        sampled_v = paddle.index_select(v, paddle.to_tensor(indices), axis=2)
        captured["trace.k"] = sampled_k.transpose([0, 2, 1, 3]).reshape([1, len(indices), -1])
        captured["trace.v"] = sampled_v.transpose([0, 2, 1, 3]).reshape([1, len(indices), -1])
        captured["trace.attn_logits"] = logits
        captured["trace.softmax"] = probabilities
        captured["trace.context_recomputed"] = context
        captured["trace.gelu"] = paddle.nn.functional.gelu(captured["trace.fc1"], approximate=True)
        captured["image_features"] = outputs.image_hidden_states
        captured["final_hidden"] = outputs.hidden_states[-1]
        captured["logits"] = outputs.logits

    for handle in handles:
        handle.remove()
    arrays = {name: value.astype("float32").cpu().numpy() for name, value in captured.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    versions = {"paddle": paddle.__version__, "numpy": np.__version__}
    metadata = _metadata("paddle", args, batch, versions, _weight_manifest(model.state_dict(), "paddle"))
    metadata["precision"]["paddle_allow_tf32_cublas"] = core.get_cublas_switch()
    metadata["outputs"] = {name: _array_summary(value) for name, value in arrays.items()}
    _write_metadata(args.output, metadata)
    print(json.dumps({"framework": "paddle", "output": str(args.output), "arrays": len(arrays)}))


def _run_compare(args: argparse.Namespace) -> None:
    import numpy as np

    torch_metadata = json.loads(args.torch_output.with_suffix(".json").read_text())
    paddle_metadata = json.loads(args.paddle_output.with_suffix(".json").read_text())
    if torch_metadata["inputs"] != paddle_metadata["inputs"]:
        raise RuntimeError("Torch and Paddle artifacts were not generated from identical inputs")

    torch_weights = torch_metadata["weights"]
    paddle_weights = paddle_metadata["weights"]
    if torch_weights.keys() != paddle_weights.keys():
        missing = sorted(torch_weights.keys() - paddle_weights.keys())
        extra = sorted(paddle_weights.keys() - torch_weights.keys())
        raise RuntimeError(f"Vision weight keys differ; missing={missing}, extra={extra}")
    mismatched_weights = [
        name for name in torch_weights if torch_weights[name]["sha256"] != paddle_weights[name]["sha256"]
    ]
    if mismatched_weights:
        raise RuntimeError(f"Vision weights differ after conversion: {mismatched_weights}")

    results = {}
    failures = []
    with np.load(args.torch_output) as torch_arrays, np.load(args.paddle_output) as paddle_arrays:
        if set(torch_arrays.files) != set(paddle_arrays.files):
            raise RuntimeError("Torch and Paddle artifacts contain different arrays")
        for name in torch_arrays.files:
            if name.startswith("mask."):
                mismatch_count = int(np.count_nonzero(torch_arrays[name] != paddle_arrays[name]))
                results[name] = {"mismatch_count": mismatch_count}
                if mismatch_count:
                    failures.append(f"{name} differs at {mismatch_count} elements")
                continue
            torch_value = torch_arrays[name].astype(np.float64)
            paddle_value = paddle_arrays[name].astype(np.float64)
            difference = np.abs(torch_value - paddle_value)
            results[name] = {"mae": float(difference.mean()), "max_abs": float(difference.max())}

    for name, tolerance in (
        ("image_features", args.vision_tolerance),
        ("logits", args.logits_tolerance),
    ):
        if results[name]["mae"] > tolerance:
            failures.append(f"{name} MAE {results[name]['mae']:.6g} exceeds {tolerance:.6g}")

    report = {
        "weights": {"count": len(torch_weights), "all_exact": True},
        "tolerances": {"image_features": args.vision_tolerance, "logits": args.logits_tolerance},
        "results": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError("; ".join(failures))


def main() -> None:
    args = _parse_args()
    if args.mode == "torch":
        _run_torch(args)
    elif args.mode == "paddle":
        _run_paddle(args)
    else:
        _run_compare(args)


if __name__ == "__main__":
    main()
