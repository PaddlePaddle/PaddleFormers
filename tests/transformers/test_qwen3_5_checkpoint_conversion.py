# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from types import SimpleNamespace

import paddle
from safetensors import safe_open
from safetensors.paddle import save_file

from paddleformers.transformers.qwen3_5.checkpoint_conversion import (
    finalize_grouped_expert_checkpoint,
    prepare_nonfused_expert_checkpoint,
)
from paddleformers.transformers.qwen3_5.modeling import Qwen3_5ForConditionalGeneration


class TestQwen35CheckpointConversion(unittest.TestCase):
    def test_accuracy_compatible_router_identity_preserves_runtime_dtype(self):
        text = SimpleNamespace(
            use_accuracy_compatible=True,
            layer_types=["linear_attention"],
            num_hidden_layers=1,
            num_experts=2,
            mtp_num_hidden_layers=1,
            shared_expert_intermediate_size=0,
            num_attention_heads=2,
            num_key_value_heads=1,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            hidden_size=16,
            attn_output_gate=False,
            attention_bias=False,
        )
        config = SimpleNamespace(
            text_config=text,
            vision_config=SimpleNamespace(depth=1, num_heads=2),
            model_type="qwen3_5_moe",
            tie_word_embeddings=False,
            _checkpoint_source_layout="qwen3_5-per-expert-hf/v1",
            tensor_model_parallel_size=1,
        )
        load = Qwen3_5ForConditionalGeneration._gen_aoa_config(config)["aoa_statements"]
        save = Qwen3_5ForConditionalGeneration._gen_inv_aoa_config(config)["aoa_statements"]
        load_router = [statement for statement in load if ".mlp.gate.weight ->" in statement]
        save_router = [statement for statement in save if "mlp.gate.weight ->" in statement]
        self.assertEqual(len(load_router), 2)
        self.assertEqual(len(save_router), 2)
        self.assertTrue(all("dtype=" not in statement for statement in load_router))
        self.assertTrue(all("dtype=" not in statement for statement in save_router))

        text.use_accuracy_compatible = False
        load = Qwen3_5ForConditionalGeneration._gen_aoa_config(config)["aoa_statements"]
        save = Qwen3_5ForConditionalGeneration._gen_inv_aoa_config(config)["aoa_statements"]
        load_router = [statement for statement in load if ".mlp.gate.weight ->" in statement]
        save_router = [statement for statement in save if "mlp.gate.weight ->" in statement]
        self.assertTrue(all("dtype='float32'" in statement for statement in load_router))
        self.assertTrue(all("dtype='bfloat16'" in statement for statement in save_router))

    def test_grouped_experts_are_partitioned_reversibly(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "profile"
            source.mkdir()
            tensors = {"unchanged.weight": paddle.arange(6, dtype="float32").reshape([2, 3])}
            originals = {}
            for layer_id in range(2):
                prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
                gate_up = paddle.arange(48, dtype="float32").reshape([2, 6, 4]) + layer_id * 100
                down = paddle.arange(40, dtype="float32").reshape([2, 4, 5]) + layer_id * 100
                tensors[f"{prefix}.gate_up_proj"] = gate_up
                tensors[f"{prefix}.down_proj"] = down
                originals[layer_id] = (gate_up, down)

            shard_name = "model-00001-of-00001.safetensors"
            save_file(tensors, source / shard_name)
            total_size = sum(tensor.numel().item() * 4 for tensor in tensors.values())
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": {key: shard_name for key in tensors},
            }
            (source / "model.safetensors.index.json").write_text(json.dumps(index))
            config = SimpleNamespace(
                moe_expert_fusion=False,
                text_config=SimpleNamespace(num_experts=2),
            )

            converted_path, manifest = prepare_nonfused_expert_checkpoint(source, config)

            converted = Path(converted_path)
            self.assertEqual(manifest["source_tensor_count"], 5)
            self.assertEqual(manifest["output_tensor_count"], 13)
            self.assertEqual(manifest["source_payload_bytes"], manifest["output_payload_bytes"])
            self.assertEqual(config._checkpoint_source_layout, "qwen3_5-per-expert-hf/v1")
            with safe_open(converted / shard_name, framework="paddle", device="cpu") as handle:
                self.assertTrue(paddle.equal_all(handle.get_tensor("unchanged.weight"), tensors["unchanged.weight"]))
                for layer_id, (expected_gate_up, expected_down) in originals.items():
                    prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
                    gates = []
                    ups = []
                    downs = []
                    for expert_id in range(2):
                        gates.append(handle.get_tensor(f"{prefix}.{expert_id}.gate_proj.weight"))
                        ups.append(handle.get_tensor(f"{prefix}.{expert_id}.up_proj.weight"))
                        downs.append(handle.get_tensor(f"{prefix}.{expert_id}.down_proj.weight"))
                    reconstructed_gate_up = paddle.stack(
                        [paddle.concat([gate, up], axis=0) for gate, up in zip(gates, ups)], axis=0
                    )
                    reconstructed_down = paddle.stack(downs, axis=0)
                    self.assertTrue(paddle.equal_all(reconstructed_gate_up, expected_gate_up))
                    self.assertTrue(paddle.equal_all(reconstructed_down, expected_down))

            # A second call validates and reuses the content-addressed cache.
            second_path, second_manifest = prepare_nonfused_expert_checkpoint(source, config)
            self.assertEqual(second_path, converted_path)
            self.assertEqual(second_manifest["output_index_sha256"], manifest["output_index_sha256"])

            # Inverse AOA save emits the normalized per-expert view. The native
            # finalizer must restore the official grouped names and bytes.
            save_config = SimpleNamespace(text_config=SimpleNamespace(num_experts=2))
            receipt = finalize_grouped_expert_checkpoint(converted, save_config)
            self.assertEqual(receipt["tensor_count_before"], 13)
            self.assertEqual(receipt["tensor_count_after"], 5)
            self.assertEqual(receipt["payload_bytes"], total_size)
            self.assertTrue((converted / "model.safetensors").is_file())
            self.assertFalse((converted / "model.safetensors.index.json").exists())
            with safe_open(converted / "model.safetensors", framework="paddle", device="cpu") as handle:
                self.assertEqual(len(handle.keys()), 5)
                self.assertTrue(paddle.equal_all(handle.get_tensor("unchanged.weight"), tensors["unchanged.weight"]))
                for layer_id, (expected_gate_up, expected_down) in originals.items():
                    prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
                    self.assertTrue(
                        paddle.equal_all(handle.get_tensor(f"{prefix}.gate_up_proj"), expected_gate_up)
                    )
                    self.assertTrue(paddle.equal_all(handle.get_tensor(f"{prefix}.down_proj"), expected_down))


if __name__ == "__main__":
    unittest.main()
