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
"""Oracle uniqueness + default HF cadence layout (resume / rotation / latest).

Default training still nests ``hf_checkpoint-*`` under ``output_dir``.
``save_hf_output_dir`` is opt-in. Formal YAML sets ``save_to_hf: false`` so
the oracle directory stays a unique safetensors root without moving cadence
for unrelated jobs.
"""

import os
import tempfile
import types
import unittest

from paddleformers.trainer.checkpoint_export import (
    assert_unique_safetensors_names,
    resolve_hf_checkpoint_dir,
    write_tiny_safetensors,
)
from paddleformers.trainer.trainer import Trainer
from paddleformers.trainer.trainer_callback import DefaultFlowCallback, IntervalStrategy
from paddleformers.trainer.trainer_utils import get_last_checkpoint


class _FakeArgs:
    def __init__(self, output_dir, save_hf_output_dir=None, save_hf_total_limit=None, save_total_limit=None):
        self.output_dir = output_dir
        self.save_hf_output_dir = save_hf_output_dir
        self.save_hf_total_limit = save_hf_total_limit
        self.save_total_limit = save_total_limit


class _FakeState:
    def __init__(self, global_step=0, best_model_checkpoint=None):
        self.global_step = global_step
        self.best_model_checkpoint = best_model_checkpoint


def _bare_trainer(output_dir, **kwargs):
    trainer = Trainer.__new__(Trainer)
    trainer.args = _FakeArgs(output_dir, **kwargs)
    trainer.state = _FakeState()
    return trainer


class ResolveHfCheckpointDirTests(unittest.TestCase):
    def test_default_is_nested_under_output_dir(self):
        output_dir = "/tmp/results/paddle/checkpoint"
        path = resolve_hf_checkpoint_dir(output_dir, 5)
        self.assertEqual(path, os.path.join(output_dir, "hf_checkpoint-5"))
        self.assertTrue(path.startswith(os.path.abspath(output_dir) + os.sep) or path.startswith(output_dir + os.sep))

    def test_opt_in_override_leaves_output_dir(self):
        output_dir = "/tmp/results/paddle/checkpoint"
        other = "/tmp/results/paddle/hf_cadence"
        path = resolve_hf_checkpoint_dir(output_dir, 5, save_hf_output_dir=other)
        self.assertEqual(path, os.path.join(other, "hf_checkpoint-5"))
        self.assertFalse(path.startswith(os.path.abspath(output_dir) + os.sep))


class UniqueNameTests(unittest.TestCase):
    def test_unique_names_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tiny_safetensors(
                os.path.join(tmp, "model-00001-of-00002.safetensors"),
                {"model.layers.0.mlp.up_proj.weight": ("F32", [1], b"\x00\x00\x00\x00")},
            )
            write_tiny_safetensors(
                os.path.join(tmp, "model-00002-of-00002.safetensors"),
                {"model.layers.0.mlp.down_proj.weight": ("F32", [1], b"\x00\x00\x00\x00")},
            )
            assert_unique_safetensors_names(tmp)

    def test_duplicate_nested_names_fail_for_oracle_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"model.layers.3.mlp.gate.e_score_correction_bias": ("F32", [1], b"\x00\x00\x00\x00")}
            write_tiny_safetensors(os.path.join(tmp, "model-00001-of-00001.safetensors"), payload)
            write_tiny_safetensors(os.path.join(tmp, "hf_checkpoint-5", "model-00001-of-00001.safetensors"), payload)
            with self.assertRaisesRegex(ValueError, "invalid or duplicate tensor name"):
                assert_unique_safetensors_names(tmp)

    def test_empty_name_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tiny_safetensors(
                os.path.join(tmp, "model.safetensors"),
                {"": ("F32", [1], b"\x00\x00\x00\x00")},
            )
            with self.assertRaisesRegex(ValueError, "invalid or duplicate tensor name"):
                assert_unique_safetensors_names(tmp)


class CadenceLayoutAndRotationTests(unittest.TestCase):
    def test_trainer_default_cadence_nested_for_resume_and_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _bare_trainer(tmp)
            trainer.state.global_step = 5
            run_dir, ckpt_path = trainer._hf_cadence_paths()
            self.assertEqual(run_dir, tmp)
            self.assertEqual(ckpt_path, os.path.join(tmp, "hf_checkpoint-5"))

    def test_trainer_opt_in_cadence_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = os.path.join(tmp, "oracle_cadence")
            trainer = _bare_trainer(os.path.join(tmp, "checkpoint"), save_hf_output_dir=other)
            trainer.state.global_step = 9
            run_dir, ckpt_path = trainer._hf_cadence_paths()
            self.assertEqual(run_dir, other)
            self.assertEqual(ckpt_path, os.path.join(other, "hf_checkpoint-9"))

    def test_rotate_hf_checkpoints_deletes_oldest_under_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            for step in (1, 2, 3):
                os.makedirs(os.path.join(tmp, f"hf_checkpoint-{step}"), exist_ok=True)
            trainer = _bare_trainer(tmp, save_hf_total_limit=2)
            trainer._rotate_hf_checkpoints(output_dir=tmp)
            remaining = sorted(p for p in os.listdir(tmp) if p.startswith("hf_checkpoint-"))
            self.assertEqual(remaining, ["hf_checkpoint-2", "hf_checkpoint-3"])

    def test_get_last_checkpoint_ignores_hf_cadence_and_returns_flex(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "checkpoint-5"))
            os.makedirs(os.path.join(tmp, "hf_checkpoint-9"))
            os.makedirs(os.path.join(tmp, "checkpoint-3"))
            last = get_last_checkpoint(tmp)
            self.assertEqual(os.path.basename(last), "checkpoint-5")

    def test_save_to_hf_false_does_not_arm_cadence(self):
        cb = DefaultFlowCallback()
        args = types.SimpleNamespace(
            logging_strategy=IntervalStrategy.NO,
            logging_steps=1,
            evaluation_strategy=IntervalStrategy.NO,
            eval_steps=1,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=5,
            save_hf_steps=-1,
            save_to_hf=False,
            flash_device_save_steps=0,
            save_last_step=False,
        )
        state = types.SimpleNamespace(global_step=5, max_steps=100)
        control = types.SimpleNamespace(
            should_log=False,
            should_evaluate=False,
            should_save=False,
            should_save_hf=False,
            should_training_stop=False,
        )
        out = cb.on_step_end(args, state, control)
        self.assertTrue(out.should_save)
        self.assertFalse(out.should_save_hf)

    def test_save_to_hf_true_arms_cadence_on_save_steps(self):
        cb = DefaultFlowCallback()
        args = types.SimpleNamespace(
            logging_strategy=IntervalStrategy.NO,
            logging_steps=1,
            evaluation_strategy=IntervalStrategy.NO,
            eval_steps=1,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=5,
            save_hf_steps=-1,
            save_to_hf=True,
            flash_device_save_steps=0,
            save_last_step=False,
        )
        state = types.SimpleNamespace(global_step=5, max_steps=100)
        control = types.SimpleNamespace(
            should_log=False,
            should_evaluate=False,
            should_save=False,
            should_save_hf=False,
            should_training_stop=False,
        )
        out = cb.on_step_end(args, state, control)
        self.assertTrue(out.should_save)
        self.assertTrue(out.should_save_hf)


if __name__ == "__main__":
    unittest.main()
