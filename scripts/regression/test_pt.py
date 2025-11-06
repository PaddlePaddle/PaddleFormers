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
CONFIG_PATH = "./examples/config/pt"
LOG_PATH = "./model_unittest_logs"
OUTPUT_DIR = tempfile.TemporaryDirectory().name
MODEL_NAME_OR_PATH = "./models/tiny-random-qwen3"

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["NCCL_ALGO"] = "Tree"
os.environ["FLAGS_embedding_deterministic"] = "1"
os.environ["FLAGS_cudnn_deterministic"] = "0"


class PTTrainTester(unittest.TestCase):
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

        loss_pattern = re.compile(r"(?<![A-Za-z_])loss:\s*([0-9]+\.[0-9]+)")
        losses = [float(m.group(1)) for m in loss_pattern.finditer(output)]
        print(f"losses list : {losses}")
        if losses:
            sum_loss = sum(losses) / len(losses)
            avg_loss = round(sum_loss, 6)
        else:
            avg_loss = 0
        print(f"Current loss : {avg_loss}")
        print(f"Base loss : {base_loss}")
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
        print(f"result is : {result}")
        print(f"result[0] is : {result[0]}")
        self.assertTrue(paddle.allclose(result[0], excepted_result))


class PTTrainTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.pttrain_tester = PTTrainTester()

    def tearDown(self):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        super().tearDown()

    def test_pt_full(self):
        output_dir = os.path.join(OUTPUT_DIR, "pt_full")
        update_args = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "train_dataset_path": "./tests/fixtures/dummy/pt/train.jsonl",
            "eval_dataset_path": "./tests/fixtures/dummy/pt/eval.jsonl",
            "output_dir": output_dir,
        }
        config_path = os.path.join(CONFIG_PATH, "full.yaml")
        updated_config_path = self.pttrain_tester.update_training_args(config_path, output_dir, update_args)
        # cli mode
        cmd = [
            "paddleformers-cli",
            "train",
            updated_config_path,
        ]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"pt_full cmd is : {cmd}")
        print(training_p.stdout)
        pt_full_output = training_p.stdout
        pt_full_log_file = os.path.join(LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "pt_full.log")
        if pt_full_output and pt_full_output.strip():
            with open(pt_full_log_file, "w", encoding="utf-8") as pt_full_f:
                pt_full_f.write(pt_full_output)
        # test training result
        self.pttrain_tester.assert_result(training_p.returncode, training_p.stdout)

        # test training loss
        EXCEPTED_LOSS = 11.978933
        self.pttrain_tester.assert_loss(training_p.stdout, EXCEPTED_LOSS)

        # test model resume
        reusme_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"pt_full reusme cmd is : {cmd}")
        print(reusme_p.stdout)
        pt_full_reusme_output = reusme_p.stdout
        pt_full_reusme_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "pt_full_reusme.log"
        )
        if pt_full_reusme_output and pt_full_reusme_output.strip():
            with open(pt_full_reusme_log_file, "w", encoding="utf-8") as pt_full_reusme_f:
                pt_full_reusme_f.write(pt_full_reusme_output)
        self.pttrain_tester.assert_result(reusme_p.returncode, reusme_p.stdout)
        EXCEPTED_LOSS = 11.978933
        self.pttrain_tester.assert_loss(reusme_p.stdout, EXCEPTED_LOSS)

        # test model generate
        EXPECTED_RESULT = paddle.to_tensor([[22407, 90612, 90612, 90612, 90612, 90612, 90612, 90612, 90612, 90612]])
        self.pttrain_tester.create_and_check_model_generate(output_dir, EXPECTED_RESULT)

    def test_pt_lora(self):
        output_dir = os.path.join(OUTPUT_DIR, "pt_lora")
        update_args = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "train_dataset_path": "./tests/fixtures/dummy/pt/train.jsonl",
            "eval_dataset_path": "./tests/fixtures/dummy/pt/eval.jsonl",
            "output_dir": output_dir,
        }
        config_path = os.path.join(CONFIG_PATH, "lora.yaml")
        updated_config_path = self.pttrain_tester.update_training_args(config_path, output_dir, update_args)
        # cli mode
        cmd = [
            "paddleformers-cli",
            "train",
            updated_config_path,
        ]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"pt_lora cmd is : {cmd}")
        print(training_p.stdout)
        pt_lora_output = training_p.stdout
        pt_lora_log_file = os.path.join(LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "pt_lora.log")
        if pt_lora_output and pt_lora_output.strip():
            with open(pt_lora_log_file, "w", encoding="utf-8") as pt_lora_f:
                pt_lora_f.write(pt_lora_output)

        # test training result
        self.pttrain_tester.assert_result(training_p.returncode, training_p.stdout)

        # test training loss
        EXCEPTED_LOSS = 11.978933
        self.pttrain_tester.assert_loss(training_p.stdout, EXCEPTED_LOSS)

        # test model resume
        reusme_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"pt_lora reusme cmd is : {cmd}")
        print(reusme_p.stdout)
        pt_lora_reusme_output = reusme_p.stdout
        pt_lora_reusme_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "pt_lora_reusme.log"
        )
        if pt_lora_reusme_output and pt_lora_reusme_output.strip():
            with open(pt_lora_reusme_log_file, "w", encoding="utf-8") as pt_lora_reusme_f:
                pt_lora_reusme_f.write(pt_lora_reusme_output)
        self.pttrain_tester.assert_result(reusme_p.returncode, reusme_p.stdout)

        EXCEPTED_LOSS = 11.978933
        self.pttrain_tester.assert_loss(reusme_p.stdout, EXCEPTED_LOSS)

        # test lora merge
        lora_merge_output_dir = os.path.join(output_dir, "export")
        # cli mode
        lora_merge_cmd = ["paddleformers-cli", "export", updated_config_path]
        lora_merge_p = subprocess.run(lora_merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.pttrain_tester.assert_result(lora_merge_p.returncode, lora_merge_p.stdout)

        # test lora_merge_model generate
        EXPECTED_RESULT = paddle.to_tensor(
            [[22407, 120525, 77505, 113631, 47887, 134141, 122487, 61092, 40897, 40601]]
        )
        self.pttrain_tester.create_and_check_model_generate(lora_merge_output_dir, EXPECTED_RESULT)

    def test_pt_full_tp_pp(self):
        output_dir = os.path.join(OUTPUT_DIR, "pt_full_tp_pp")
        update_args = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "train_dataset_path": "./tests/fixtures/dummy/pt/train.jsonl",
            "eval_dataset_path": "./tests/fixtures/dummy/pt/eval.jsonl",
            "output_dir": output_dir,
        }
        config_path = os.path.join(CONFIG_PATH, "full_tp_pp.yaml")
        updated_config_path = self.pttrain_tester.update_training_args(config_path, output_dir, update_args)
        cmd = [
            "paddleformers-cli",
            "train",
            updated_config_path,
        ]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"pt_full_tp_pp cmd is : {cmd}")
        print(training_p.stdout)
        pt_full_tp_pp_output = training_p.stdout
        pt_full_tp_pp_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "pt_full_tp_pp.log"
        )
        if pt_full_tp_pp_output and pt_full_tp_pp_output.strip():
            with open(pt_full_tp_pp_log_file, "w", encoding="utf-8") as pt_full_tp_pp_f:
                pt_full_tp_pp_f.write(pt_full_tp_pp_output)
        # test training result
        self.pttrain_tester.assert_result(training_p.returncode, training_p.stdout)

        # test training loss
        EXCEPTED_LOSS = 11.980663
        self.pttrain_tester.assert_loss(training_p.stdout, EXCEPTED_LOSS)

        # test model resume
        reusme_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"pt_full_tp_pp reusme cmd is : {cmd}")
        print(reusme_p.stdout)
        pt_full_tp_pp_reusme_output = reusme_p.stdout
        pt_full_tp_pp_reusme_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "pt_full_tp_pp_reusme.log"
        )
        if pt_full_tp_pp_reusme_output and pt_full_tp_pp_reusme_output.strip():
            with open(pt_full_tp_pp_reusme_log_file, "w", encoding="utf-8") as pt_full_tp_pp_reusme_f:
                pt_full_tp_pp_reusme_f.write(pt_full_tp_pp_reusme_output)
        self.pttrain_tester.assert_result(reusme_p.returncode, reusme_p.stdout)

        EXCEPTED_LOSS = 11.980663
        self.pttrain_tester.assert_loss(reusme_p.stdout, EXCEPTED_LOSS)
        # test model generate
        EXPECTED_RESULT = paddle.to_tensor([[22407, 90612, 90612, 90612, 90612, 90612, 90612, 90612, 90612, 90612]])
        self.pttrain_tester.create_and_check_model_generate(output_dir, EXPECTED_RESULT)

    def test_pt_lora_tp_pp(self):
        output_dir = os.path.join(OUTPUT_DIR, "pt_lora_tp_pp")
        update_args = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "train_dataset_path": "./tests/fixtures/dummy/pt/train.jsonl",
            "eval_dataset_path": "./tests/fixtures/dummy/pt/eval.jsonl",
            "output_dir": output_dir,
        }
        config_path = os.path.join(CONFIG_PATH, "lora_tp_pp.yaml")
        updated_config_path = self.pttrain_tester.update_training_args(config_path, output_dir, update_args)
        # cli mode
        cmd = [
            "paddleformers-cli",
            "train",
            updated_config_path,
        ]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"pt_lora_tp_pp cmd is : {cmd}")
        print(training_p.stdout)
        pt_lora_tp_pp_output = training_p.stdout
        pt_lora_tp_pp_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "pt_lora_tp_pp.log"
        )
        if pt_lora_tp_pp_output and pt_lora_tp_pp_output.strip():
            with open(pt_lora_tp_pp_log_file, "w", encoding="utf-8") as pt_lora_tp_pp_f:
                pt_lora_tp_pp_f.write(pt_lora_tp_pp_output)
        # test training result
        self.pttrain_tester.assert_result(training_p.returncode, training_p.stdout)

        # test training loss
        EXCEPTED_LOSS = 11.980666
        self.pttrain_tester.assert_loss(training_p.stdout, EXCEPTED_LOSS)

        # test model resume
        reusme_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"pt_lora_tp_pp reusme cmd is : {cmd}")
        print(reusme_p.stdout)
        pt_lora_tp_pp_reusme_output = reusme_p.stdout
        pt_lora_tp_pp_reusme_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "pt_lora_tp_pp_reusme.log"
        )
        if pt_lora_tp_pp_reusme_output and pt_lora_tp_pp_reusme_output.strip():
            with open(pt_lora_tp_pp_reusme_log_file, "w", encoding="utf-8") as pt_lora_tp_pp_reusme_f:
                pt_lora_tp_pp_reusme_f.write(pt_lora_tp_pp_reusme_output)
        self.pttrain_tester.assert_result(reusme_p.returncode, reusme_p.stdout)

        EXCEPTED_LOSS = 11.980666
        self.pttrain_tester.assert_loss(reusme_p.stdout, EXCEPTED_LOSS)

        # test lora merge
        lora_merge_output_dir = os.path.join(output_dir, "export")
        # cli mode
        lora_merge_cmd = ["paddleformers-cli", "export", updated_config_path]
        lora_merge_p = subprocess.run(lora_merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.pttrain_tester.assert_result(lora_merge_p.returncode, lora_merge_p.stdout)

        # test lora_merge_model generate
        EXPECTED_RESULT = paddle.to_tensor(
            [[22407, 120525, 77505, 113631, 47887, 134141, 122487, 61092, 40897, 40601]]
        )
        self.pttrain_tester.create_and_check_model_generate(lora_merge_output_dir, EXPECTED_RESULT)
