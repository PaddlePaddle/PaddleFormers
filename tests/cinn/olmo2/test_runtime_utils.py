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

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.cinn.olmo2.runtime_utils import (
    benchmark_forward,
    check_loss,
    validate_inference_checkpoint,
)


class RuntimeUtilsTest(unittest.TestCase):
    def test_inference_requires_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for checkpoint in (None, root, root / "missing"):
                with self.subTest(checkpoint=checkpoint), self.assertRaisesRegex(ValueError, "initial-checkpoint"):
                    validate_inference_checkpoint(True, checkpoint)
            path = root / "weights"
            path.touch()
            validate_inference_checkpoint(True, path)
            validate_inference_checkpoint(False, None)

    def test_loss_finite_check(self):
        self.assertEqual(check_loss(1.25, 4), 1.25)
        for loss in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(loss=loss), self.assertRaisesRegex(FloatingPointError, "step 4"):
                check_loss(loss, 4)

    def test_timing_paths_use_identical_warmup_and_measurement(self):
        for save_logits in (False, True):
            with self.subTest(save_logits=save_logits), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                net = mock.Mock()
                net.return_value.numpy.return_value = np.array([1.0], dtype="float32")
                # Ten warmup starts, then thirty pairs of start/end timestamps.
                timestamps = list(range(10))
                for index in range(30):
                    timestamps.extend([10 + index, 10 + index + 0.002])
                with mock.patch("paddle.device.synchronize") as synchronize, mock.patch(
                    "tests.cinn.olmo2.runtime_utils.time.perf_counter", side_effect=timestamps
                ):
                    benchmark_forward(net, "inputs", output, "test scope", save_logits=save_logits)
                self.assertEqual(net.call_count, 40)
                self.assertEqual(synchronize.call_count, 80)
                result = json.loads((output / "inference.json").read_text())
                self.assertEqual(result["warmup"], 10)
                self.assertEqual(result["iterations"], 30)
                self.assertEqual(len(result["times_ms"]), 30)
                self.assertAlmostEqual(result["median_ms"], 2.0)
                self.assertEqual((output / "logits.npy").exists(), save_logits)
