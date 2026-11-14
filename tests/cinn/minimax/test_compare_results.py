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

"""Synthetic negative tests for the comparison gate, not model acceptance data."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import paddle


class CompareResultsTest(unittest.TestCase):
    def test_training_regressions_fail_even_with_python_optimization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("dynamic", "cinn"):
                for task in ("train", "infer"):
                    folder = root / f"{mode}-{task}"
                    folder.mkdir()
                    metadata = dict(
                        start_step=0,
                        steps=1,
                        initial_checkpoint_sha256="same",
                        initial_parameter_hash="same",
                        source_commit="same",
                        data_sha256="same",
                    )
                    (folder / "run.json").write_text(json.dumps(metadata))
                    np.savez(folder / "inputs.npz", ids=np.array([1, 2]))
                    if task == "train":
                        (folder / "loss.jsonl").write_text(json.dumps(dict(step=0, loss=1.0)) + "\n")
                        paddle.save(
                            {"weight": np.array([1.0], dtype="float32")}, str(folder / "first_gradients.pdparams")
                        )
                    else:
                        np.save(folder / "logits.npy", np.array([1.0], dtype="float32"))
                        (folder / "inference.json").write_text(json.dumps(dict(median_ms=1.0)))
            command = [sys.executable, "-O", str(Path(__file__).with_name("compare_results.py")), str(root)]
            baseline = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            loss_path = root / "cinn-train/loss.jsonl"
            loss_path.write_text(json.dumps(dict(step=0, loss=1.1)) + "\n")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Training loss mismatch", result.stderr)
            loss_path.write_text(json.dumps(dict(step=0, loss=1.0)) + "\n")
            paddle.save(
                {"weight": np.array([1.1], dtype="float32")}, str(root / "cinn-train/first_gradients.pdparams")
            )
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Gradient mismatch: weight", result.stderr)
