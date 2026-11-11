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
"""Compare quick-entry results without claiming long-run convergence."""
import argparse
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
parser.add_argument("--logits-atol", type=float, default=1e-4)
parser.add_argument("--loss-atol", type=float, default=1e-5)
parser.add_argument("--grad-atol", type=float, default=1e-4)
args = parser.parse_args()
for name in ("logits_atol", "loss_atol", "grad_atol"):
    tolerance = getattr(args, name)
    if not np.isfinite(tolerance) or tolerance < 0:
        parser.error(f"{name} must be finite and nonnegative")
root = args.results
losses = []
grads = []
for mode in ["dynamic", "cinn"]:
    folder = root / f"{mode}-train"
    meta = json.loads((folder / "run.json").read_text())
    rows = [json.loads(line) for line in (folder / "loss.jsonl").read_text().splitlines()]
    require(
        [r["step"] for r in rows] == list(range(meta["start_step"], meta["steps"])), f"Invalid step sequence: {mode}"
    )
    losses.append(np.asarray([r["loss"] for r in rows]))
    grads.append(dict(np.load(folder / "gradients.npz")))
if losses[0].shape != losses[1].shape or not losses[0].size:
    raise ValueError("Training loss sequences must be nonempty and have identical shapes")
if not all(np.isfinite(x).all() for x in losses):
    raise ValueError("Non-finite training loss")
np.testing.assert_allclose(losses[0], losses[1], atol=args.loss_atol, rtol=0, err_msg="Training loss mismatch")
require(grads[0].keys() == grads[1].keys() and bool(grads[0]), "Missing or mismatched gradient keys")
gradient_max = 0.0
for name in grads[0]:
    d, c = [np.asarray(g[name]) for g in grads]
    if d.shape != c.shape or not np.isfinite(d).all() or not np.isfinite(c).all():
        raise ValueError(f"Invalid gradient shape or non-finite gradient: {name}")
    np.testing.assert_allclose(d, c, atol=args.grad_atol, rtol=0, err_msg=f"Gradient mismatch: {name}")
    require(not (np.any(d) and not np.any(c)), f"CINN gradient became all zero: {name}")
    gradient_max = max(gradient_max, float(np.max(np.abs(d - c))))
for task in ["train", "infer"]:
    folders = [root / f"{mode}-{task}" for mode in ["dynamic", "cinn"]]
    metas = [json.loads((p / "run.json").read_text()) for p in folders]
    for key in ["source_commit", "checkpoint_sha256", "dataset_sha256"]:
        require(metas[0][key] == metas[1][key], f"Metadata mismatch: {task}/{key}")
    inputs = [np.load(p / "inputs.npz") for p in folders]
    require(inputs[0].files == inputs[1].files, f"Input key mismatch: {task}")
    for key in inputs[0].files:
        np.testing.assert_array_equal(inputs[0][key], inputs[1][key])
logits = [np.load(root / f"{mode}-infer/logits.npy") for mode in ["dynamic", "cinn"]]
require(all(np.isfinite(x).all() for x in logits), "Non-finite inference logits")
np.testing.assert_allclose(logits[0], logits[1], atol=args.logits_atol, rtol=0)
print(
    json.dumps(
        {
            "loss_max_abs_diff": float(np.max(np.abs(losses[0] - losses[1]))),
            "gradient_count": len(grads[0]),
            "gradient_max_abs_diff": gradient_max,
            "logits_max_abs_diff": float(np.max(np.abs(logits[0] - logits[1]))),
            "scope": "short-entry regression; not convergence acceptance",
            "tolerances": {"loss_atol": args.loss_atol, "grad_atol": args.grad_atol, "logits_atol": args.logits_atol},
        },
        indent=2,
    )
)
