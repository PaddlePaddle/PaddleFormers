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

"""Serial convergence + three-repeat timing using only the delivered shell entries."""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    entry = Path(__file__).resolve().parent
    repo = entry.parents[2]
    state = dict(status="running", jobs=[], criteria=json.loads((entry / "acceptance.json").read_text()))

    def save():
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.output / "status.json")

    save()
    try:
        for repeat in (1, 2, 3):
            folder = args.output / f"r{repeat}"
            modes = ("dynamic", "cinn") if repeat != 2 else ("cinn", "dynamic")
            for mode in modes:
                state["current"] = f"r{repeat}/{mode}"
                save()
                env = dict(
                    os.environ,
                    RESULTS_DIR=str(folder),
                    PYTHON_BIN=sys.executable,
                    STEPS="1000" if repeat == 1 else "150",
                    INFER_STEPS="150",
                    WARMUP="50",
                )
                script = "run_model.sh" if mode == "dynamic" else "run_model_cinn.sh"
                command = ["bash", str(entry / script)]
                print("START", state["current"], flush=True)
                with (args.output / f"r{repeat}-{mode}.log").open("x") as log:
                    subprocess.run(
                        command, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800
                    )
                state["jobs"].append(dict(repeat=repeat, mode=mode, status="completed"))
                save()
            command = [sys.executable, str(entry / "compare_results.py"), str(folder)]
            if repeat == 1:
                command.append("--convergence")
            with (args.output / f"r{repeat}-comparison.log").open("x") as log:
                subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        metrics = {}
        for task in ("infer", "train"):
            metrics[task] = {}
            identities = []
            for mode in ("dynamic", "cinn"):
                medians = []
                for repeat in (1, 2, 3):
                    folder = args.output / f"r{repeat}" / f"{mode}-{task}"
                    result = json.loads((folder / "result.json").read_text())
                    identities.append(
                        {
                            k: result[k]
                            for k in (
                                "initial_checkpoint_sha256",
                                "input_sha256",
                                "model_sha256",
                                "source_diff_sha256",
                                "source_commit",
                                "entry_sha256",
                                "paddle_commit",
                                "precision",
                                "acceptance_sha256",
                                "shell_sha256",
                            )
                        }
                    )
                    rows = [json.loads(s) for s in (folder / "steps.jsonl").read_text().splitlines()]
                    values = [r["ms"] for r in rows if 50 <= r["step"] < 150]
                    if len(values) != 100:
                        raise ValueError("Missing fixed timing window")
                    medians.append(statistics.median(values))
                metrics[task][mode] = dict(
                    medians_ms=medians,
                    mean_median_ms=statistics.mean(medians),
                    range_percent=(max(medians) / min(medians) - 1) * 100,
                )
            if any(x != identities[0] for x in identities):
                raise ValueError(f"Asset or source identity differs across repeats: {task}")
            d, c = [metrics[task][m]["mean_median_ms"] for m in ("dynamic", "cinn")]
            metrics[task]["throughput_gain_percent"] = (d / c - 1) * 100
            metrics[task]["dynamic_samples_per_second"] = 1000 / d
            metrics[task]["cinn_samples_per_second"] = 1000 / c
        state.update(
            status="local_suite_passed",
            current=None,
            performance=metrics,
            convergence=json.loads((args.output / "r1/comparison.json").read_text())["convergence"],
            timing="mean of 3 independent process medians; steps 50..149; batch 1",
            limitation="fixed-input subset benchmark; not a guarantee for other hardware or software environments",
        )
        save()
    except Exception as error:
        state.update(status="needs_investigation", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
