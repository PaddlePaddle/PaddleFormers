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
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest

import paddle
import yaml

TRAIN_PATH = "./examples"
CONFIG_PATH = "./examples/config/dpo"
OUTPUT_DIR = tempfile.TemporaryDirectory().name
MODEL_NAME_OR_PATH = "./models/tiny-random-qwen3"

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["NCCL_ALGO"] = "Tree"
os.environ["FLAGS_embedding_deterministic"] = "1"
os.environ["FLAGS_cudnn_deterministic"] = "1"


class DPOTrainTester(unittest.TestCase):
    def update_training_args(self, yaml_path, tmp_dir, updates) -> str:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config.update(updates)

        os.makedirs(tmp_dir, exist_ok=True)
        updated_yaml_path = os.path.join(tmp_dir, f"updated_{os.path.basename(yaml_path)}")
        with open(updated_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, indent=4, allow_unicode=True, sort_keys=False)

        return updated_yaml_path

    def assert_loss(self, output, base_loss):
        """
        Calculate the average loss from the log file, and compare it with the expected value.
        """

        loss_pattern = re.compile(r"loss:\s*([0-9]+\.[0-9]+)")
        losses = [float(m.group(1)) for m in loss_pattern.finditer(output)]

        if losses:
            sum_loss = sum(losses) / len(losses)
            avg_loss = round(sum_loss, 6)
        else:
            avg_loss = 0
        self.assertTrue(abs(avg_loss - base_loss) <= 0.0001, f"loss: {avg_loss}, base_loss: {base_loss}, exist diff!")

    def assert_result(self, ret_code, log_output):
        """assert result"""
        if ret_code != 0:
            print("\n".join(log_output.strip().splitlines()[-30:]))
            raise AssertionError("Training Failed")

    def create_and_check_model_generate(
        self,
        model_path,
        excepted_result,
    ):
        from paddleformers.transformers import Qwen3ForCausalLM

        input_ids = paddle.to_tensor([[1, 306, 4658, 278, 6593, 310, 2834, 338]])
        attention_mask = paddle.ones_like(input_ids)
        model = Qwen3ForCausalLM.from_pretrained(model_path, dtype="bfloat16", convert_from_hf=True)
        with paddle.no_grad():
            result = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10)
        self.assertTrue(paddle.allclose(result[0], excepted_result))


class DPOTrainTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dpotrain_tester = DPOTrainTester()

    def tearDown(self):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        super().tearDown()

    def test_dpo_full(self):
        output_dir = os.path.join(OUTPUT_DIR, "dpo_full")
        update_args = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "train_dataset_path": "./tests/fixtures/dummy/ernie/dpo-train.jsonl",
            "eval_dataset_path": "./tests/fixtures/dummy/ernie/dpo-train.jsonl",
            "output_dir": output_dir,
            "packing": True,
            "max_seq_len": 2048,
            "max_steps": 5,
            "save_steps": 3,
            "tensor_parallel_degree": 2,
            "pipeline_parallel_degree": 2,
            "sequence_parallel": True,
            "pipeline_parallel_config": "enable_clear_every_step_cache disable_partial_send_recv",
            "sharding": "stage1",
        }
        config_path = os.path.join(CONFIG_PATH, "full.yaml")
        updated_config_path = self.dpotrain_tester.update_training_args(config_path, output_dir, update_args)
        train_path = os.path.join(TRAIN_PATH, "alignment/dpo/run_dpo.py")
        cmd = [
            "python",
            "-u",
            "-m",
            "paddle.distributed.launch",
            "--devices",
            "0,1,2,3",
            train_path,
            updated_config_path,
        ]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # test training result
        self.dpotrain_tester.assert_result(training_p.returncode, training_p.stdout)

        # test training loss
        EXCEPTED_LOSS = 0.474259
        self.dpotrain_tester.assert_loss(training_p.stdout, EXCEPTED_LOSS)

        # test model generate
        EXPECTED_RESULT = paddle.to_tensor([[22407, 90612, 90612, 90612, 90612, 90612, 90612, 90612, 90612, 90612]])
        self.dpotrain_tester.create_and_check_model_generate(output_dir, EXPECTED_RESULT)

    def test_dpo_lora(self):
        output_dir = os.path.join(OUTPUT_DIR, "dpo_lora")
        update_args = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "train_dataset_path": "./tests/fixtures/dummy/ernie/dpo-train.jsonl",
            "eval_dataset_path": "./tests/fixtures/dummy/ernie/dpo-train.jsonl",
            "output_dir": output_dir,
            "packing": True,
            "max_seq_len": 2048,
            "max_steps": 5,
            "save_steps": 3,
        }
        config_path = os.path.join(CONFIG_PATH, "lora.yaml")
        updated_config_path = self.dpotrain_tester.update_training_args(config_path, output_dir, update_args)
        train_path = os.path.join(TRAIN_PATH, "alignment/dpo/run_dpo.py")
        cmd = [
            "python",
            "-u",
            train_path,
            updated_config_path,
        ]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # test training result
        self.dpotrain_tester.assert_result(training_p.returncode, training_p.stdout)

        # test training loss
        EXCEPTED_LOSS = 0.474163
        self.dpotrain_tester.assert_loss(training_p.stdout, EXCEPTED_LOSS)

        # test lora  merge
        lora_merge_output_dir = os.path.join(output_dir, "lora_merge")
        lora_merge_path = os.path.join(TRAIN_PATH, "tools/mergekit.py")

        lora_merge_cmd = [
            "python",
            "-u",
            lora_merge_path,
            "--lora_model_path",
            output_dir,
            "--model_name_or_path",
            MODEL_NAME_OR_PATH,
            "--output_path",
            lora_merge_output_dir,
        ]
        lora_merge_p = subprocess.run(lora_merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.dpotrain_tester.assert_result(lora_merge_p.returncode, lora_merge_p.stdout)

        # test lora_merge_model generate
        EXPECTED_RESULT = paddle.to_tensor(
            [[22407, 120525, 77505, 113631, 47887, 134141, 122487, 61092, 40897, 40601]]
        )
        self.dpotrain_tester.create_and_check_model_generate(lora_merge_output_dir, EXPECTED_RESULT)
