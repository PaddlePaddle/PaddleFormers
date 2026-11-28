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

"""Accept full forward logits within 1e-2 and ten matching generated tokens."""
import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


def metrics(reference, actual):
    if reference.shape != actual.shape:
        return {"passed": False, "reference_shape": list(reference.shape), "actual_shape": list(actual.shape)}
    if reference.size == 0:
        return {"passed": True, "max_abs": 0.0, "mean_abs": 0.0, "relative_l2": 0.0}
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    matching_nan = np.array_equal(np.isnan(a), np.isnan(b))
    finite = np.isfinite(a) & np.isfinite(b)
    if (
        not np.array_equal(np.isfinite(a), np.isfinite(b))
        or not matching_nan
        or not np.array_equal(np.isposinf(a), np.isposinf(b))
        or not np.array_equal(np.isneginf(a), np.isneginf(b))
    ):
        return {"passed": False, "nonfinite_mismatch": True}
    delta = a[finite] - b[finite]
    if not delta.size:
        return {"passed": True, "max_abs": 0.0, "mean_abs": 0.0, "relative_l2": 0.0}
    return {
        "max_abs": float(np.abs(delta).max()),
        "mean_abs": float(np.abs(delta).mean()),
        "relative_l2": float(np.linalg.norm(delta) / max(1e-12, np.linalg.norm(a[finite]))),
    }


def compare_predictions(reference, actual, min_iou, geometry_atol):
    if len(reference) != len(actual):
        return {"passed": False, "reference_count": len(reference), "actual_count": len(actual)}
    if not reference:
        return {"passed": True, "reference_count": 0, "actual_count": 0, "instances": []}

    def decode(prediction):
        rle = prediction["mask_rle"]
        return mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode()}).astype(bool)

    am, bm = [decode(x) for x in reference], [decode(x) for x in actual]
    if any(x.shape != am[0].shape for x in am + bm):
        return {"passed": False, "shape_mismatch": True}
    ious = np.zeros([len(am), len(bm)])
    for i, a in enumerate(am):
        for j, b in enumerate(bm):
            union = np.logical_or(a, b).sum()
            ious[i, j] = float(np.logical_and(a, b).sum() / union) if union else 1.0
    rows, cols = linear_sum_assignment(-ious)
    instances = []
    for i, j in zip(rows, cols):
        geometry = max(
            abs(reference[i][field][key] - actual[j][field][key])
            for field, keys in [("xy", ["x", "y"]), ("hw", ["h", "w"])]
            for key in keys
        )
        instances.append(
            {
                "reference": int(i),
                "actual": int(j),
                "mask_iou": float(ious[i, j]),
                "geometry_max_abs": float(geometry),
                "passed": bool(ious[i, j] >= min_iou and geometry <= geometry_atol),
            }
        )
    return {
        "passed": all(x["passed"] for x in instances),
        "reference_count": len(reference),
        "actual_count": len(actual),
        "instances": instances,
    }


def compare_forward_logits(reference, actual, atol):
    """Compare every prompt position and vocabulary entry, in bounded CPU chunks."""
    if not reference.exists() or not actual.exists():
        return {"passed": False, "missing": "prefill_logits.npy; re-export with --trace"}
    a, b = np.load(reference, mmap_mode="r"), np.load(actual, mmap_mode="r")
    if a.shape != b.shape or a.ndim != 3 or a.size == 0:
        return {"passed": False, "reference_shape": list(a.shape), "actual_shape": list(b.shape)}
    maximum, total, count = 0.0, 0.0, 0
    prefix = None
    for begin in range(0, a.shape[1], 64):
        x = a[:, begin : begin + 64].astype(np.float64)
        y = b[:, begin : begin + 64].astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return {"passed": False, "nonfinite_logits": True}
        delta = np.abs(x - y)
        if begin == 0:
            first = delta[:, :20]
            prefix = {
                "positions": int(first.shape[1]),
                "shape": list(first.shape),
                "mean_abs": float(first.mean()),
                "max_abs": float(first.max()),
                "per_position_mean_abs": first.mean(axis=(0, 2)).tolist(),
                "per_position_max_abs": first.max(axis=(0, 2)).tolist(),
            }
        maximum = max(maximum, float(delta.max()))
        total += float(delta.sum())
        count += delta.size
    return {
        "passed": maximum <= atol,
        "shape": list(a.shape),
        "max_abs": maximum,
        "mean_abs": total / count,
        "atol": atol,
        "first_20_positions": prefix,
    }


def read_generated_tokens(directory, manifest):
    # Decode inputs are the tokens actually selected by generate, including its
    # terminal token. Do not count the unused argmax after the terminal forward.
    collected = []
    for filename in manifest["trace_files"][1:]:
        with np.load(directory / filename) as values:
            tokens = values["input_tokens"]
            if tokens.shape != (1, 1):
                raise ValueError("Expected a single generated token in each decode step.")
            collected.append(int(tokens[0, 0]))
    if "generated_tokens" in manifest and manifest["generated_tokens"] != collected:
        raise ValueError("Manifest tokens differ from actual decode inputs.")
    return collected


def compare_generation(reference, actual, required=10):
    compared = min(len(reference), len(actual), required)
    mismatch = next((i for i in range(compared) if reference[i] != actual[i]), None)
    sufficient = len(reference) >= required and len(actual) >= required
    return {
        "passed": sufficient and mismatch is None,
        "required_tokens": required,
        "reference_length": len(reference),
        "actual_length": len(actual),
        "reference_prefix": reference[:required],
        "actual_prefix": actual[:required],
        "matching_prefix_length": mismatch if mismatch is not None else compared,
        "first_mismatch_index": mismatch,
        "sufficient_tokens": sufficient,
    }


def validate_outputs(predictions):
    try:
        if not isinstance(predictions, list):
            raise ValueError("Predictions must be a list.")
        for prediction in predictions:
            xy = [prediction["xy"][key] for key in ("x", "y")]
            hw = [prediction["hw"][key] for key in ("h", "w")]
            if not all(np.isfinite(x) and 0 <= x <= 1 for x in xy):
                raise ValueError("Invalid normalized center coordinates.")
            if not all(np.isfinite(x) and 0 < x <= 1 for x in hw):
                raise ValueError("Invalid normalized object size.")
            rle = prediction["mask_rle"]
            mask = mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode()})
            if mask.ndim != 2 or not mask.size or not mask.any():
                raise ValueError("Invalid or empty instance mask.")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return {"passed": False, "error": str(error)}
    return {"passed": True, "instances": len(predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--logits-atol", type=float, default=1e-2)
    parser.add_argument("--hidden-relative-l2", type=float, default=5e-5)
    parser.add_argument("--upsampler-relative-l2", type=float, default=2e-3)
    parser.add_argument("--geometry-atol", type=float, default=1e-6)
    parser.add_argument("--min-mask-iou", type=float, default=0.99)
    parser.add_argument("--generation-prefix-length", type=int, default=10)
    parser.add_argument("--strict-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.generation_prefix_length < 1 or args.logits_atol < 0:
        parser.error("Token prefix must be positive and logits tolerance nonnegative.")
    ref = json.loads((args.reference / "manifest.json").read_text())
    actual = json.loads((args.actual / "manifest.json").read_text())
    metadata_keys = ["dtype", "image", "query", "min_dimension", "max_dimension", "max_new_tokens"]
    metadata = {k: {"reference": ref[k], "actual": actual[k], "passed": ref[k] == actual[k]} for k in metadata_keys}
    result = {
        "metadata": metadata,
        "thresholds": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "steps": [],
    }
    result["trace_file_list_equal"] = bool(ref["trace_files"]) and ref["trace_files"] == actual["trace_files"]
    for filename in ref["trace_files"]:
        step = {"file": filename, "tensors": {}}
        if not (args.actual / filename).exists():
            step["passed"] = False
            result["steps"].append(step)
            continue
        with np.load(args.reference / filename) as aa, np.load(args.actual / filename) as bb:
            step["keys_equal"] = set(aa.files) == set(bb.files)
            for key in aa.files:
                if key not in bb:
                    step["tensors"][key] = {"passed": False, "missing": True}
                    continue
                m = metrics(aa[key], bb[key])
                if "passed" not in m:
                    if key.startswith("input_"):
                        # Floating pixel/position inputs should be bitwise identical;
                        # predicted geometry can use the specified tolerance.
                        atol = args.geometry_atol if key in ["input_coord_xy", "input_size_hw"] else 0.0
                        m["passed"] = m["max_abs"] <= atol
                    elif key == "logits":
                        m["argmax_equal"] = bool(np.array_equal(aa[key].argmax(-1), bb[key].argmax(-1)))
                        m["passed"] = m["max_abs"] <= args.logits_atol and m["argmax_equal"]
                    else:
                        m["passed"] = m["relative_l2"] <= args.hidden_relative_l2
                step["tensors"][key] = m
        step["passed"] = step.get("keys_equal", False) and all(m["passed"] for m in step["tensors"].values())
        result["steps"].append(step)
    result["features"] = {}
    for filename in ["upsampler_samples.npy", "mask_logits.npz"]:
        a, b = args.reference / filename, args.actual / filename
        if not a.exists() or not b.exists():
            result["features"][filename] = {"passed": False, "missing": True}
        elif a.exists():
            if filename.endswith(".npy"):
                pairs = [(filename, np.load(a), np.load(b))]
            else:
                with np.load(a) as aa, np.load(b) as bb:
                    if set(aa.files) != set(bb.files):
                        result["features"][filename] = {"passed": False, "keys_mismatch": True}
                    pairs = [(key, aa[key], bb[key]) for key in aa.files if key in bb]
            for key, x, y in pairs:
                m = metrics(x, y)
                m["passed"] = (
                    m.get("passed", True) and m.get("relative_l2", float("inf")) <= args.upsampler_relative_l2
                )
                result["features"][key] = m
    result["predictions"] = compare_predictions(
        ref["predictions"], actual["predictions"], args.min_mask_iou, args.geometry_atol
    )
    result["diagnostics_passed"] = (
        all(x["passed"] for x in metadata.values())
        and result["trace_file_list_equal"]
        and all(x["passed"] for x in result["steps"])
        and all(x["passed"] for x in result["features"].values())
        and result["predictions"]["passed"]
    )
    forward = compare_forward_logits(
        args.reference / "prefill_logits.npy", args.actual / "prefill_logits.npy", args.logits_atol
    )
    prefill_inputs = result["steps"][0]["tensors"] if result["steps"] else {}
    forward["inputs_equal"] = (
        "input_tokens" in prefill_inputs
        and result["steps"][0].get("keys_equal", False)
        and all(m["passed"] for key, m in prefill_inputs.items() if key.startswith("input_"))
    )
    forward["passed"] = forward["passed"] and forward["inputs_equal"]
    try:
        generation = compare_generation(
            read_generated_tokens(args.reference, ref),
            read_generated_tokens(args.actual, actual),
            args.generation_prefix_length,
        )
    except (OSError, KeyError, ValueError) as error:
        generation = {"passed": False, "error": str(error)}
    outputs = {"reference": validate_outputs(ref["predictions"]), "actual": validate_outputs(actual["predictions"])}
    outputs["passed"] = outputs["reference"]["passed"] and outputs["actual"]["passed"]
    result["acceptance"] = {"forward": forward, "generation": generation, "outputs": outputs}
    result["passed"] = (
        all(x["passed"] for x in metadata.values())
        and forward["passed"]
        and generation["passed"]
        and outputs["passed"]
    )
    if args.strict_diagnostics:
        result["passed"] = result["passed"] and result["diagnostics_passed"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        "passed": result["passed"],
        "steps": len(result["steps"]),
        "acceptance": result["acceptance"],
        "diagnostics_passed": result["diagnostics_passed"],
        "predictions": result["predictions"],
        "max_logits_abs": max(
            (s["tensors"].get("logits", {}).get("max_abs", 0) for s in result["steps"]), default=None
        ),
        "report": str(args.report),
    }
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
