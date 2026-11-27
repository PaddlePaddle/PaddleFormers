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
"""Fail-closed comparison of shared-checkpoint runs; optional preregistered convergence."""
import argparse
import json
from pathlib import Path

import numpy as np

IDENTITY_KEYS = [
    "initial_checkpoint_sha256",
    "initial_parameter_hash",
    "input_sha256",
    "data_sha256",
    "source_commit",
    "source_diff_sha256",
    "entry_sha256",
    "acceptance_sha256",
    "shell_sha256",
    "model_sha256",
    "paddle_commit",
    "precision",
    "config",
    "seed",
    "learning_rate",
    "weight_decay",
    "batch",
    "sequence_length",
    "steps",
    "warmup",
    "boundary",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(root, convergence=False):
    criteria = json.loads(Path(__file__).with_name("acceptance.json").read_text())
    report = dict(criteria=criteria, scope="convergence" if convergence else "short_entry")
    tasks = {}
    for task in ("train", "infer"):
        runs = {}
        for mode in ("dynamic", "cinn"):
            directory = root / f"{mode}-{task}"
            result = json.loads((directory / "result.json").read_text())
            require(result["status"] == "completed", f"Incomplete {mode}/{task}")
            rows = [json.loads(s) for s in (directory / "steps.jsonl").read_text().splitlines()]
            require([r["step"] for r in rows] == list(range(result["steps"])), "Invalid step sequence")
            text = (root / f"{mode}-{task}.log").read_text(errors="replace")
            require(("Compiling subgraph with CINN backend" in text) == (mode == "cinn"), "Backend evidence mismatch")
            runs[mode] = dict(result=result, rows=rows, directory=directory)
        for key in IDENTITY_KEYS:
            require(runs["dynamic"]["result"][key] == runs["cinn"]["result"][key], f"Metadata mismatch: {task}/{key}")
        arrays = [np.load(runs[m]["directory"] / "initial_logits.npy") for m in ("dynamic", "cinn")]
        require(all(np.isfinite(x).all() for x in arrays), "Nonfinite logits")
        np.testing.assert_allclose(arrays[0], arrays[1], atol=criteria["initial_logits_atol"], rtol=0)
        report[task] = dict(initial_logits_max_abs=float(np.max(np.abs(arrays[0] - arrays[1]))))
        times = [runs[m]["result"]["median_ms"] for m in ("dynamic", "cinn")]
        report[task].update(
            dynamic_ms=times[0],
            cinn_ms=times[1],
            throughput_gain_percent=(times[0] / times[1] - 1) * 100 if all(times) else None,
        )
        tasks[task] = runs
    train = tasks["train"]
    losses = [np.array([r["loss"] for r in train[m]["rows"]]) for m in ("dynamic", "cinn")]
    require(all(np.isfinite(x).all() for x in losses), "Nonfinite training loss")
    np.testing.assert_allclose(
        losses[0][:20], losses[1][:20], atol=criteria["first20_loss_atol"], rtol=0, err_msg="Training loss mismatch"
    )
    gradients = [np.load(train[m]["directory"] / "gradient_samples.npz") for m in ("dynamic", "cinn")]
    require(gradients[0].files == gradients[1].files and bool(gradients[0].files), "Gradient keys mismatch")
    maximum = 0.0
    for name in gradients[0].files:
        d, c = [g[name] for g in gradients]
        require(np.isfinite(d).all() and np.isfinite(c).all(), f"Nonfinite gradient: {name}")
        np.testing.assert_allclose(
            d, c, atol=criteria["gradient_sample_atol"], rtol=0, err_msg=f"Gradient mismatch: {name}"
        )
        maximum = max(maximum, float(np.max(np.abs(d - c))))
    report["gradient_sample_max_abs"] = maximum
    report[
        "gradient_scope"
    ] = "all tensors checked finite/present; equality compared at <=1024 sampled elements per tensor"
    report["train"]["steps"] = len(losses[0])
    if convergence:
        rule = criteria["convergence"]
        require(len(losses[0]) >= rule["minimum_steps"], "Too few steps for convergence")
        means = [float(x[-100:].mean()) for x in losses]
        drops = [float(1 - x[-100:].mean() / x[:100].mean()) for x in losses]
        tail_mae = float(np.abs(losses[0][-100:] - losses[1][-100:]).mean())
        tolerance = max(rule["tail_mae_absolute_floor"], rule["tail_mae_relative"] * abs(means[0]))
        require(all(v <= rule["tail_mean_max"] for v in means), "Tail mean has not converged")
        require(all(v >= rule["minimum_loss_reduction"] for v in drops), "Insufficient loss reduction")
        require(tail_mae <= tolerance, "Tail loss curves differ beyond frozen tolerance")
        report["convergence"] = dict(
            last100_mean=means,
            loss_reduction=drops,
            tail_mae=tail_mae,
            tail_mae_tolerance=tolerance,
            step400_to499_mean=[float(x[400:500].mean()) for x in losses],
            scope="training subset fit; NOT task generalization",
        )
    report["status"] = "passed"
    return report, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--convergence", action="store_true")
    args = parser.parse_args()
    report, losses = compare(args.results, args.convergence)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4))
    for values, label in zip(losses, ["Paddle dynamic", "Paddle CINN"]):
        axis.plot(np.arange(1, len(values) + 1), values, label=label, linewidth=1)
    axis.set(xlabel="Step", ylabel="Loss", title="OLMo2: original width, 2 layers, FP32")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.results / "loss_raw.png", dpi=160)
    plt.close(figure)
    (args.results / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
