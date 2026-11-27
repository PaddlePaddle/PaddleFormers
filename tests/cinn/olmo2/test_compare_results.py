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

"""Synthetic negative gates; never used as model acceptance data."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.cinn.olmo2.compare_results import IDENTITY_KEYS, compare
from tests.cinn.olmo2.delivery_utils import CONTRACT, validate_config


class CompareResultsTest(unittest.TestCase):
    def fixture(self, root, steps=20):
        for mode in ("dynamic", "cinn"):
            for task in ("train", "infer"):
                folder = root / f"{mode}-{task}"
                folder.mkdir()
                meta = {key: "same" for key in IDENTITY_KEYS}
                meta.update(status="completed", steps=steps, median_ms=1.0)
                (folder / "result.json").write_text(json.dumps(meta))
                rows = [dict(step=i, loss=10.0 if i < 100 else 0.001) for i in range(steps)]
                (folder / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                (root / f"{mode}-{task}.log").write_text(
                    "Compiling subgraph with CINN backend" if mode == "cinn" else "dynamic"
                )
                np.save(folder / "initial_logits.npy", np.array([1.0], dtype="float32"))
                np.savez(folder / "gradient_samples.npz", weight=np.array([1.0], dtype="float32"))

    def test_short_and_convergence_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, 1000)
            self.assertEqual(compare(root, True)[0]["status"], "passed")
            path = root / "cinn-train/steps.jsonl"
            original = path.read_text()
            path.write_text(original.replace('"loss": 0.001', '"loss": 1.0'))
            with self.assertRaisesRegex(ValueError, "converged"):
                compare(root, True)

    def test_reject_loss_gradients_identity_and_missing_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertEqual(compare(root)[0]["status"], "passed")
            loss_path = root / "cinn-train/steps.jsonl"
            original = loss_path.read_text()
            loss_path.write_text(original.replace('"loss": 10.0', '"loss": 11.0'))
            with self.assertRaisesRegex(AssertionError, "Training loss mismatch"):
                compare(root)
            loss_path.write_text(original)
            np.savez(root / "cinn-train/gradient_samples.npz", weight=np.array([1.1], dtype="float32"))
            with self.assertRaisesRegex(AssertionError, "Gradient mismatch"):
                compare(root)
            np.savez(root / "cinn-train/gradient_samples.npz", weight=np.array([1.0], dtype="float32"))
            meta_path = root / "cinn-infer/result.json"
            meta = json.loads(meta_path.read_text())
            meta["initial_parameter_hash"] = "different"
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Metadata mismatch"):
                compare(root)
            meta["initial_parameter_hash"] = "same"
            meta_path.write_text(json.dumps(meta))
            (root / "cinn-train.log").write_text("FLAGS are not sufficient evidence")
            with self.assertRaisesRegex(ValueError, "Backend evidence mismatch"):
                compare(root)

    def test_tiny_config_is_rejected(self):
        validate_config(CONTRACT)
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            validate_config(dict(CONTRACT, hidden_size=64))
