# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import paddle
from safetensors.paddle import save_file

from paddleformers.trainer.repro_receipt import (
    ReproReceiptCallback,
    _first_paddle_tensor,
    _is_paddlefleet_column_parallel_linear,
    _paddle_input_array,
    _write_internal_boundary_receipt,
)


class TestReproReceiptCallback(unittest.TestCase):
    def test_raw_losses_and_checkpoint_are_written_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model_dir = root / "profile"
            output_dir = root / "output"
            model_dir.mkdir()
            output_dir.mkdir()
            (model_dir / "config.json").write_text("{}\n")
            (model_dir / "repro_profile.json").write_text(
                json.dumps(
                    {
                        "schema": "test-profile/v1",
                        "source_model_id": "example/model",
                        "source_revision": "revision",
                        "source_config_sha256": "0" * 64,
                        "tensor_count": 1,
                        "payload_bytes": 8,
                    }
                )
            )
            args = SimpleNamespace(
                output_dir=str(output_dir),
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                expert_tensor_model_parallel_size=1,
                sharding_parallel_size=1,
                sharding="",
                sequence_parallel=False,
                use_accuracy_compatible=True,
                deterministic_mode=True,
                max_steps=2,
                seed=1234,
                learning_rate=1e-6,
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_epsilon=1e-8,
                weight_decay=0.0,
                max_grad_norm=0.0,
            )
            state = SimpleNamespace(is_world_process_zero=True, global_step=0)
            model = paddle.nn.Linear(2, 2)
            model.config = SimpleNamespace(text_config=SimpleNamespace())
            callback = ReproReceiptCallback(model_dir)
            callback.on_train_begin(args, state, None, model=model, optimizer=SimpleNamespace())

            initial_dir = output_dir / "initial_checkpoint"
            initial_dir.mkdir()
            save_file({"weight": paddle.arange(4, dtype="bfloat16")}, initial_dir / "model.safetensors")
            (initial_dir / "paddle_grouped_checkpoint_conversion.json").write_text("{}\n")
            callback.record_initial_checkpoint(args, initial_dir)
            initial_manifest = json.loads((initial_dir / "initial_checkpoint_manifest.json").read_text())
            self.assertEqual(initial_manifest["global_step"], 0)
            self.assertEqual(initial_manifest["tensor_count"], 1)

            for step, loss in ((1, 1.25), (2, 1.0)):
                state.global_step = step
                callback.on_log(
                    args,
                    state,
                    None,
                    logs={
                        "global_step": step,
                        "loss": round(loss, 8),
                        "repro_raw_loss": loss,
                        "learning_rate": 1e-6,
                        "interval_runtime": 99.0,
                    },
                )

            save_file({"weight": paddle.arange(4, dtype="bfloat16")}, output_dir / "model.safetensors")
            (output_dir / "paddle_grouped_checkpoint_conversion.json").write_text("{}\n")
            callback.finalize_checkpoint(args, state)

            self.assertEqual(
                json.loads((output_dir / "loss.json").read_text()),
                {"framework": "paddle", "losses": [1.25, 1.0]},
            )
            metrics = [json.loads(line) for line in (output_dir / "repro_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(
                metrics[0],
                {"step": 1, "loss": 1.25, "repro_raw_loss": 1.25, "learning_rate": 1e-6},
            )
            self.assertFalse((output_dir / "model.safetensors").exists())
            manifest = json.loads((output_dir / "checkpoint" / "checkpoint_manifest.json").read_text())
            self.assertEqual(manifest["tensor_count"], 1)
            self.assertEqual(manifest["payload_bytes"], 8)
            self.assertEqual(manifest["nlink"], 1)

    def test_gdn_in_proj_selector_accepts_fleet_subclasses_only(self):
        fleet_linear = type(
            "ColumnParallelLinear",
            (object,),
            {"__module__": "paddlefleet.tensor_parallel.layers"},
        )
        candidate = type("AccuracyCompatibleGDNInputProjection", (fleet_linear,), {})
        unrelated = type("ColumnParallelLinear", (object,), {"__module__": "example"})
        self.assertTrue(_is_paddlefleet_column_parallel_linear(candidate()))
        self.assertFalse(_is_paddlefleet_column_parallel_linear(unrelated()))

    def test_internal_boundary_tensor_prefers_hidden_states(self):
        hidden_states = paddle.randn([1, 8, 16])
        attention_mask = paddle.ones([1, 1, 8, 8])
        actual = _first_paddle_tensor({
            "attention_mask": attention_mask,
            "hidden_states": hidden_states,
        })
        self.assertIs(actual, hidden_states)

    def test_internal_boundary_receipt_preserves_call_order_and_raw_bits(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            calls = []
            for offset in (0, 4):
                input_tensor = paddle.arange(offset, offset + 4, dtype="bfloat16").reshape([2, 2])
                output_tensor = input_tensor + 1
                calls.append({
                    "input": _paddle_input_array(input_tensor),
                    "output": _paddle_input_array(output_tensor),
                })

            _write_internal_boundary_receipt(
                root, 0, "vision_merger", "model.visual.merger", "example.Merger", calls
            )

            manifest = json.loads((root / "internal_boundaries" / "vision_merger_rank0.json").read_text())
            arrays = __import__("numpy").load(root / "internal_boundaries" / manifest["npz"])
            self.assertEqual(manifest["call_count"], 2)
            self.assertEqual([call["call_index"] for call in manifest["calls"]], [0, 1])
            self.assertEqual(manifest["calls"][0]["input"]["dtype"], "bfloat16")
            self.assertEqual(str(arrays["c0_input"].dtype), "uint16")

    def test_missing_step_is_rejected(self):
        callback = ReproReceiptCallback(".")
        callback.losses_by_step = {1: 1.0}
        with self.assertRaisesRegex(RuntimeError, "expected raw losses"):
            callback._validate_losses(2)


if __name__ == "__main__":
    unittest.main()
