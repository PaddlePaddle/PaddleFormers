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

"""Exercise production callback lifecycle without loading GPU dependencies."""
import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paddleformers/cli/train/sft/workflow.py"
tree = ast.parse(SOURCE.read_text())
callback = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelReproObservationCallback")
selected = {
    "__init__",
    "_is_writer",
    "_sha256_file",
    "_write_json",
    "_machine_loss_payload",
    "on_train_begin",
    "on_train_end",
    "on_log",
}
callback.body = [n for n in callback.body if isinstance(n, ast.FunctionDef) and n.name in selected]
namespace = {"TrainerCallback": object, "os": os, "Path": Path, "hashlib": hashlib, "json": json}
exec(compile(ast.fix_missing_locations(ast.Module(body=[callback], type_ignores=[])), str(SOURCE), "exec"), namespace)
Callback = namespace["ModelReproObservationCallback"]


class MachineOutputTests(unittest.TestCase):
    def test_unrounded_main_loss_and_stale_window_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, loss = Path(directory) / "raw.jsonl", Path(directory) / "loss.json"
            raw.write_text("old invocation\n")
            loss.write_text("old invocation\n")
            with patch.dict(os.environ, {"MODEL_REPRO_LOSS_PATH": str(loss)}, clear=True):
                callback = Callback(str(raw))
                state = SimpleNamespace(is_world_process_zero=True, global_step=1)
                callback.on_train_begin(None, state, None)
                self.assertFalse(loss.exists())
                value = 0.123456789123
                callback.on_log(None, state, None, logs={"loss": round(value, 8), "mtp 1 loss": 0.5}, raw_loss=value)
                callback.on_train_end(None, state, None)
                result = json.loads(loss.read_text())
                self.assertEqual(result["losses"], [value])
                self.assertEqual(result["steps"], [1])
                self.assertEqual(result["source_sha256"], hashlib.sha256(raw.read_bytes()).hexdigest())
                self.assertEqual(result["events"][0]["mtp_1_loss"], 0.5)
                self.assertNotIn("owning_cli_exit_code", result)

    def test_loss_without_raw_path_and_non_writer_is_inert(self):
        with tempfile.TemporaryDirectory() as directory:
            loss = Path(directory) / "loss.json"
            with patch.dict(os.environ, {"MODEL_REPRO_LOSS_PATH": str(loss)}, clear=True):
                callback = Callback()
                state = SimpleNamespace(is_world_process_zero=False, global_step=1)
                callback.on_train_begin(None, state, None)
                callback.on_log(None, state, None, raw_loss=0.25)
                callback.on_train_end(None, state, None)
                self.assertFalse(loss.exists())
                state.is_world_process_zero = True
                callback.on_train_begin(None, state, None)
                callback.on_log(None, state, None, raw_loss=0.25)
                callback.on_train_end(None, state, None)
                self.assertEqual(json.loads(loss.read_text())["losses"], [0.25])

    def test_environment_written_only_after_train_begin(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / "env.json"
            with patch.dict(os.environ, {"MODEL_REPRO_ENV_PATH": str(env)}, clear=True):
                callback = Callback(model_source="loaded/model", weights_loaded=True)
                callback._environment_payload = lambda args: {
                    "weights_loaded": callback.weights_loaded,
                    "model_source": callback.model_source,
                }
                self.assertFalse(env.exists())
                callback.on_train_begin(None, SimpleNamespace(is_world_process_zero=True), None)
                self.assertEqual(json.loads(env.read_text()), {"weights_loaded": True, "model_source": "loaded/model"})


if __name__ == "__main__":
    unittest.main()
