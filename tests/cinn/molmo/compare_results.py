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


"""Compare short entry regressions, not long-run convergence."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("results", type=Path)
args = parser.parse_args()
root = args.results
for task in ["train", "infer"]:
    folders = [root / f"{mode}-{task}" for mode in ["dynamic", "cinn"]]
    metas = [json.loads((p / "run.json").read_text()) for p in folders]
    for key in ["source_commit", "initial_parameter_hash"]:
        assert metas[0][key] == metas[1][key], key
    inputs = [np.load(p / "inputs.npz") for p in folders]
    assert inputs[0].files == inputs[1].files
    for key in inputs[0].files:
        np.testing.assert_array_equal(inputs[0][key], inputs[1][key])
losses = []
for mode in ["dynamic", "cinn"]:
    p = root / f"{mode}-train"
    rows = [json.loads(line) for line in (p / "loss.jsonl").read_text().splitlines()]
    meta = json.loads((p / "run.json").read_text())
    assert [r["step"] for r in rows] == list(range(meta["start_step"], meta["steps"]))
    losses.append(np.asarray([r["loss"] for r in rows]))
assert losses[0].shape == losses[1].shape
assert all(np.isfinite(x).all() for x in losses)
grads = [np.load(root / f"{mode}-train/grads.npz") for mode in ["dynamic", "cinn"]]
assert set(grads[0].files) == set(grads[1].files) and grads[0].files
gradient_max = 0.0
for name in grads[0].files:
    d, c = [g[name] for g in grads]
    assert d.shape == c.shape and np.isfinite(d).all() and np.isfinite(c).all(), name
    assert not (np.any(d) and not np.any(c)), name
    gradient_max = max(gradient_max, float(np.max(np.abs(d - c))))
logits = [np.load(root / f"{mode}-infer/logits.npy") for mode in ["dynamic", "cinn"]]
assert all(np.isfinite(x).all() for x in logits)
np.testing.assert_allclose(logits[0], logits[1], atol=1e-3, rtol=0)
print(
    json.dumps(
        {
            "loss_max_abs_diff": float(np.max(np.abs(losses[0] - losses[1]))),
            "gradient_count": len(grads[0].files),
            "gradient_max_abs_diff": gradient_max,
            "logits_max_abs_diff": float(np.max(np.abs(logits[0] - logits[1]))),
            "scope": "short-entry regression; not convergence acceptance",
        },
        indent=2,
    )
)
