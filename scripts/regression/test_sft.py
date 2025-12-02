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
import pytest
import yaml

TRAIN_PATH = "./examples"
CONFIG_PATH = "./examples/config/sft"
LOG_PATH = "./model_unittest_logs"
OUTPUT_DIR = tempfile.TemporaryDirectory().name
MODEL_NAME_OR_PATH = "./models/tiny-random-qwen3"
MAX_STEPS = 6
SAVE_STEPS = 4

SFT_FULL_EXCEPTED_LOSS = 11.944916
SFT_FULL_RESUME_EXCEPTED_LOSS = 11.946712
SFT_FULL_EXCEPTED_RESULT = [[22407, 120525, 77505, 113631, 47887, 134141, 122487, 61092, 40897, 40601]]

SFT_LORA_EXCEPTED_LOSS = 11.944835
SFT_LORA_RESUME_EXCEPTED_LOSS = 11.94655
SFT_LORA_EXCEPTED_RESULT = [[22407, 120525, 77505, 113631, 47887, 134141, 122487, 61092, 40897, 40601]]

SFT_FULL_TP_PP_EXCEPTED_LOSS = 11.956736
SFT_FULL_TP_PP_RESUME_EXCEPTED_LOSS = 11.956321
SFT_FULL_TP_PP_EXCEPTED_RESULT = [[22407, 120525, 77505, 113631, 47887, 134141, 122487, 61092, 40897, 40601]]

SFT_LORA_TP_PP_EXCEPTED_LOSS = 11.956645
SFT_LORA_TP_PP_RESUME_EXCEPTED_LOSS = 11.956057
SFT_LORA_TP_PP_EXCEPTED_RESULT = [[22407, 120525, 77505, 113631, 47887, 134141, 122487, 61092, 40897, 40601]]

SFT_FC_EXCEPTED_LOSS = 11.95681
SFT_FC_RESUME_EXCEPTED_LOSS = 11.960528
SFT_FC_EXCEPTED_RESULT = [[22407, 120525, 77505, 113631, 47887, 134141, 122487, 61092, 40897, 40601]]

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["NCCL_ALGO"] = "Tree"
os.environ["FLAGS_embedding_deterministic"] = "1"
os.environ["FLAGS_cudnn_deterministic"] = "1"


class SFTTrainTester(unittest.TestCase):
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
        print(f"excepted_result is : {excepted_result}")
        print(f"result[0] is : {result[0]}")
        self.assertTrue(paddle.allclose(result[0], excepted_result))


class SFTTrainTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.sfttrain_tester = SFTTrainTester()

    def tearDown(self):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        super().tearDown()

    @pytest.mark.skipif(True, reason="Skip for timeout")
    def test_sft_full(self):
        output_dir = os.path.join(OUTPUT_DIR, "sft_full")
        update_args = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "output_dir": output_dir,
            "max_steps": MAX_STEPS,
            "save_steps": SAVE_STEPS,
        }
        config_path = os.path.join(CONFIG_PATH, "full.yaml")
        updated_config_path = self.sfttrain_tester.update_training_args(config_path, output_dir, update_args)
        # cli mode
        cmd = [
            "paddleformers-cli",
            "train",
            updated_config_path,
        ]

        # sft_full_resume_log_file = os.path.join(
        #     LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "sft_full_resume.log"
        # )

        # print(f"sft_full resume cmd is : {cmd}")

        # with open(sft_full_resume_log_file, "w", encoding="utf-8") as log_file:
        #     resume_p = subprocess.Popen(
        #         cmd,
        #         stdout=subprocess.PIPE,
        #         stderr=subprocess.STDOUT,
        #         text=True,
        #         bufsize=1
        #     )
        #     # 逐行处理输出
        #     for line in resume_p.stdout:
        #         print(line, end='', flush=True)  # 实时输出到终端
        #         log_file.write(line)  # 实时写入文件
        #         log_file.flush()
        #     resume_p.wait()
        # print("Command execution completed and log saved.")

        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"sft_full cmd is : {cmd}")
        print(training_p.stdout)
        sft_full_output = training_p.stdout
        sft_full_log_file = os.path.join(LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "sft_full.log")
        if sft_full_output and sft_full_output.strip():
            with open(sft_full_log_file, "w", encoding="utf-8") as sft_full_f:
                sft_full_f.write(sft_full_output)
        # test training result
        self.sfttrain_tester.assert_result(training_p.returncode, training_p.stdout)

        # test training loss
        self.sfttrain_tester.assert_loss(training_p.stdout, SFT_FULL_EXCEPTED_LOSS)

        # test model resume
        resume_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"sft_full resume cmd is : {cmd}")
        print(resume_p.stdout)
        sft_full_resume_output = resume_p.stdout
        sft_full_resume_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "sft_full_resume.log"
        )
        if sft_full_resume_output and sft_full_resume_output.strip():
            with open(sft_full_resume_log_file, "w", encoding="utf-8") as sft_full_resume_f:
                sft_full_resume_f.write(sft_full_resume_output)

        self.sfttrain_tester.assert_result(resume_p.returncode, resume_p.stdout)
        self.sfttrain_tester.assert_loss(resume_p.stdout, SFT_FULL_RESUME_EXCEPTED_LOSS)

        # test model generate
        EXPECTED_RESULT = paddle.to_tensor(SFT_FULL_EXCEPTED_RESULT)
        self.sfttrain_tester.create_and_check_model_generate(output_dir, EXPECTED_RESULT)

    @pytest.mark.skipif(True, reason="Skip for timeout")
    def test_sft_lora(self):
        output_dir = os.path.join(OUTPUT_DIR, "sft_lora")
        update_args = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "output_dir": output_dir,
            "max_steps": MAX_STEPS,
            "save_steps": SAVE_STEPS,
        }
        config_path = os.path.join(CONFIG_PATH, "lora.yaml")
        updated_config_path = self.sfttrain_tester.update_training_args(config_path, output_dir, update_args)
        # cli mode
        cmd = [
            "paddleformers-cli",
            "train",
            updated_config_path,
        ]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"sft_lora cmd is : {cmd}")
        print(training_p.stdout)
        sft_lora_output = training_p.stdout
        sft_lora_log_file = os.path.join(LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "sft_lora.log")
        if sft_lora_output and sft_lora_output.strip():
            with open(sft_lora_log_file, "w", encoding="utf-8") as sft_lora_f:
                sft_lora_f.write(sft_lora_output)

        # test training result
        self.sfttrain_tester.assert_result(training_p.returncode, training_p.stdout)

        # test training loss
        self.sfttrain_tester.assert_loss(training_p.stdout, SFT_LORA_EXCEPTED_LOSS)

        # test model resume
        resume_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"sft_lora resume cmd is : {cmd}")
        print(resume_p.stdout)
        sft_lora_resume_output = resume_p.stdout
        sft_lora_resume_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "sft_lora_resume.log"
        )
        if sft_lora_resume_output and sft_lora_resume_output.strip():
            with open(sft_lora_resume_log_file, "w", encoding="utf-8") as sft_lora_resume_f:
                sft_lora_resume_f.write(sft_lora_resume_output)
        self.sfttrain_tester.assert_result(resume_p.returncode, resume_p.stdout)

        self.sfttrain_tester.assert_loss(resume_p.stdout, SFT_LORA_RESUME_EXCEPTED_LOSS)

        # test lora merge
        lora_merge_output_dir = os.path.join(output_dir, "export")
        # cli mode
        lora_merge_cmd = ["paddleformers-cli", "export", updated_config_path]
        lora_merge_p = subprocess.run(lora_merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.sfttrain_tester.assert_result(lora_merge_p.returncode, lora_merge_p.stdout)

        # test lora_merge_model generate
        EXPECTED_RESULT = paddle.to_tensor(SFT_LORA_EXCEPTED_RESULT)
        self.sfttrain_tester.create_and_check_model_generate(lora_merge_output_dir, EXPECTED_RESULT)

    @pytest.mark.skipif(True, reason="Skip for timeout")
    def test_sft_full_tp_pp(self):
        output_dir = os.path.join(OUTPUT_DIR, "sft_full_tp_pp")
        update_args = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "output_dir": output_dir,
            "max_steps": MAX_STEPS,
            "save_steps": SAVE_STEPS,
        }
        config_path = os.path.join(CONFIG_PATH, "full_tp_pp.yaml")
        updated_config_path = self.sfttrain_tester.update_training_args(config_path, output_dir, update_args)
        cmd = [
            "paddleformers-cli",
            "train",
            updated_config_path,
        ]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"sft_full_tp_pp cmd is : {cmd}")
        print(training_p.stdout)
        sft_full_tp_pp_output = training_p.stdout
        sft_full_tp_pp_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "sft_full_tp_pp.log"
        )
        if sft_full_tp_pp_output and sft_full_tp_pp_output.strip():
            with open(sft_full_tp_pp_log_file, "w", encoding="utf-8") as sft_full_tp_pp_f:
                sft_full_tp_pp_f.write(sft_full_tp_pp_output)
        # test training result
        self.sfttrain_tester.assert_result(training_p.returncode, training_p.stdout)

        # test training loss
        self.sfttrain_tester.assert_loss(training_p.stdout, SFT_FULL_TP_PP_EXCEPTED_LOSS)

        # test model resume
        resume_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"sft_full_tp_pp resume cmd is : {cmd}")
        print(resume_p.stdout)
        sft_full_tp_pp_resume_output = resume_p.stdout
        sft_full_tp_pp_resume_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "sft_full_tp_pp_resume.log"
        )
        if sft_full_tp_pp_resume_output and sft_full_tp_pp_resume_output.strip():
            with open(sft_full_tp_pp_resume_log_file, "w", encoding="utf-8") as sft_full_tp_pp_resume_f:
                sft_full_tp_pp_resume_f.write(sft_full_tp_pp_resume_output)
        self.sfttrain_tester.assert_result(resume_p.returncode, resume_p.stdout)

        self.sfttrain_tester.assert_loss(resume_p.stdout, SFT_FULL_TP_PP_RESUME_EXCEPTED_LOSS)
        # test model generate
        EXPECTED_RESULT = paddle.to_tensor(SFT_FULL_TP_PP_EXCEPTED_RESULT)
        self.sfttrain_tester.create_and_check_model_generate(output_dir, EXPECTED_RESULT)

    @pytest.mark.skipif(True, reason="Skip for timeout")
    def test_sft_lora_tp_pp(self):
        output_dir = os.path.join(OUTPUT_DIR, "sft_lora_tp_pp")
        update_args = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "output_dir": output_dir,
            "max_steps": MAX_STEPS,
            "save_steps": SAVE_STEPS,
        }
        config_path = os.path.join(CONFIG_PATH, "lora_tp_pp.yaml")
        updated_config_path = self.sfttrain_tester.update_training_args(config_path, output_dir, update_args)
        # cli mode
        cmd = [
            "paddleformers-cli",
            "train",
            updated_config_path,
        ]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"sft_lora_tp_pp cmd is : {cmd}")
        print(training_p.stdout)
        sft_lora_tp_pp_output = training_p.stdout
        sft_lora_tp_pp_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "sft_lora_tp_pp.log"
        )
        if sft_lora_tp_pp_output and sft_lora_tp_pp_output.strip():
            with open(sft_lora_tp_pp_log_file, "w", encoding="utf-8") as sft_lora_tp_pp_f:
                sft_lora_tp_pp_f.write(sft_lora_tp_pp_output)
        # test training result
        self.sfttrain_tester.assert_result(training_p.returncode, training_p.stdout)

        # test training loss
        self.sfttrain_tester.assert_loss(training_p.stdout, SFT_LORA_TP_PP_EXCEPTED_LOSS)

        # test model resume
        resume_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"sft_lora_tp_pp resume cmd is : {cmd}")
        print(resume_p.stdout)
        sft_lora_tp_pp_resume_output = resume_p.stdout
        sft_lora_tp_pp_resume_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "sft_lora_tp_pp_resume.log"
        )
        if sft_lora_tp_pp_resume_output and sft_lora_tp_pp_resume_output.strip():
            with open(sft_lora_tp_pp_resume_log_file, "w", encoding="utf-8") as sft_lora_tp_pp_resume_f:
                sft_lora_tp_pp_resume_f.write(sft_lora_tp_pp_resume_output)
        self.sfttrain_tester.assert_result(resume_p.returncode, resume_p.stdout)

        self.sfttrain_tester.assert_loss(resume_p.stdout, SFT_LORA_TP_PP_RESUME_EXCEPTED_LOSS)

        # test lora merge
        lora_merge_output_dir = os.path.join(output_dir, "export")
        # cli mode
        lora_merge_cmd = ["paddleformers-cli", "export", updated_config_path]
        lora_merge_p = subprocess.run(lora_merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.sfttrain_tester.assert_result(lora_merge_p.returncode, lora_merge_p.stdout)

        # test lora_merge_model generate
        EXPECTED_RESULT = paddle.to_tensor(SFT_LORA_TP_PP_EXCEPTED_RESULT)
        self.sfttrain_tester.create_and_check_model_generate(lora_merge_output_dir, EXPECTED_RESULT)

    @pytest.mark.skipif(True, reason="Skip for timeout")
    def test_sft_full_function_call(self):
        output_dir = os.path.join(OUTPUT_DIR, "sft_full_function_call")
        update_args = {
            "model_name_or_path": MODEL_NAME_OR_PATH,
            "output_dir": output_dir,
            "max_steps": MAX_STEPS,
            "save_steps": SAVE_STEPS,
        }
        config_path = os.path.join(CONFIG_PATH, "full_function_call.yaml")
        updated_config_path = self.sfttrain_tester.update_training_args(config_path, output_dir, update_args)
        # cli mode
        cmd = [
            "paddleformers-cli",
            "train",
            updated_config_path,
        ]
        print(f"cmd {cmd}")
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"sft_full_function_call cmd is : {cmd}")
        print(training_p.stdout)
        sft_full_function_call_output = training_p.stdout
        sft_full_function_call_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "sft_full_function_call.log"
        )
        if sft_full_function_call_output and sft_full_function_call_output.strip():
            with open(sft_full_function_call_log_file, "w", encoding="utf-8") as sft_full_function_call_f:
                sft_full_function_call_f.write(sft_full_function_call_output)
        # test training result
        self.sfttrain_tester.assert_result(training_p.returncode, training_p.stdout)

        # test training loss
        self.sfttrain_tester.assert_loss(training_p.stdout, SFT_FC_EXCEPTED_LOSS)

        # test model resume
        resume_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"sft_full_function_call resume cmd is : {cmd}")
        print(resume_p.stdout)
        sft_full_function_call_resume_output = resume_p.stdout
        sft_full_function_call_resume_log_file = os.path.join(
            LOG_PATH, str(os.path.basename(MODEL_NAME_OR_PATH)) + "sft_full_function_call_resume.log"
        )
        if sft_full_function_call_resume_output and sft_full_function_call_resume_output.strip():
            with open(
                sft_full_function_call_resume_log_file, "w", encoding="utf-8"
            ) as sft_full_function_call_resume_f:
                sft_full_function_call_resume_f.write(sft_full_function_call_resume_output)
        self.sfttrain_tester.assert_result(resume_p.returncode, resume_p.stdout)

        self.sfttrain_tester.assert_loss(resume_p.stdout, SFT_FC_RESUME_EXCEPTED_LOSS)

        # test model generate
        EXPECTED_RESULT = paddle.to_tensor(SFT_FC_EXCEPTED_RESULT)
        self.sfttrain_tester.create_and_check_model_generate(output_dir, EXPECTED_RESULT)
