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
import pytest
import yaml
from dataclasses import dataclass
from typing import Any, Dict

CONFIG_PATH = "./examples/config/"
LOG_PATH = "./model_unittest_logs"
OUTPUT_DIR = tempfile.TemporaryDirectory().name
MAX_STEPS = 3
SAVE_STEPS = 2

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["NCCL_ALGO"] = "Tree"
os.environ["FLAGS_embedding_deterministic"] = "1"
os.environ["FLAGS_cudnn_deterministic"] = "1"


class TrainTester:
    @dataclass
    class ModelConfig:
        name: str
        repo_id: str
        cli_args: Dict[str, Any]
        base_loss: Dict[str, float]

    def load_model_config(self, model_key: str) -> ModelConfig:
        model_config_path = "./scripts/regression/config.yaml"
        with open(model_config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        model_cfg = data[model_key]
        return self.ModelConfig(
            name=model_key,
            repo_id=model_cfg.get("repo_id"),
            cli_args=model_cfg.get("cli_args", {}),
            base_loss=model_cfg.get("base_loss", {})
        )

    def update_training_args(self, yaml_path, tmp_dir, updates) -> str:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config.update(updates)

        os.makedirs(tmp_dir, exist_ok=True)
        updated_yaml_path = os.path.join(tmp_dir, f"updated_{os.path.basename(yaml_path)}")
        with open(updated_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, indent=4, allow_unicode=True, sort_keys=False)

        return updated_yaml_path

    def assert_loss(self, output, base_loss, resume_flag):
        loss_pattern = re.compile(r"(?<![A-Za-z_])loss:\s*([0-9]+\.[0-9]+)")
        losses = [float(m.group(1)) for m in loss_pattern.finditer(output)]

        if losses:
            sum_loss = sum(losses) / len(losses)
            avg_loss = round(sum_loss, 6)
        else:
            avg_loss = 0

        print(f"{resume_flag} loss : {avg_loss} || base loss : {base_loss}")

        # 返回 None 或 错误信息
        if abs(avg_loss - base_loss) > 0.0001:
            return f"{resume_flag} loss: {avg_loss}, base_loss: {base_loss}, exist diff!"
        return None


    def assert_result(self, ret_code, log_output):
        if ret_code != 0:
            print("\n".join(log_output.strip().splitlines()[-30:]))
            raise AssertionError("Training Failed")


class TestTrain:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.train_tester = TrainTester()
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    @pytest.mark.parametrize("train_type", ["sft", "dpo", "pt"])
    def test_full(self, train_type, model_key):
        print(f"\n[INFO] Testing with model={model_key}, train_type={train_type}_full")
        model_cfg = self.train_tester.load_model_config(model_key)
        cli_args = model_cfg.cli_args
        model_name_or_path = model_cfg.repo_id

        full_loss = model_cfg.base_loss.get(f"{train_type}_full_loss", 0)
        full_resume_loss = model_cfg.base_loss.get(f"{train_type}_full_resume_loss", 0)

        output_dir = os.path.join(OUTPUT_DIR, f"{train_type}_{model_key}")
        update_args = {
            "model_name_or_path": model_name_or_path,
            "max_steps": MAX_STEPS,
            "save_steps": SAVE_STEPS,
            "output_dir": output_dir,
        }
        update_args.update(cli_args)

        config_path = os.path.join(CONFIG_PATH, train_type, "full.yaml")
        updated_config_path = self.train_tester.update_training_args(config_path, output_dir, update_args)

        cmd = [
            "paddleformers-cli",
            "train",
            updated_config_path,
        ]

        # train
        
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        full_log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_full.log")
        os.makedirs(LOG_PATH, exist_ok=True)
        if training_p.stdout and training_p.stdout.strip():
            with open(full_log_file, "w", encoding="utf-8") as f:
                f.write(training_p.stdout)

        
        self.train_tester.assert_result(training_p.returncode, training_p.stdout)
        
        # resume 
        resume_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        full_resume_log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_full_resume.log")
        if resume_p.stdout and resume_p.stdout.strip():
            with open(full_resume_log_file, "w", encoding="utf-8") as f:
                f.write(resume_p.stdout)

        self.train_tester.assert_result(resume_p.returncode, resume_p.stdout)

        # check loss diff
        errors = []
        msg = self.train_tester.assert_loss(training_p.stdout, full_loss, "Fisrt-Training")
        if msg:
            errors.append(AssertionError(msg))

        msg = self.train_tester.assert_loss(resume_p.stdout, full_resume_loss, "Resume-Training")
        if msg:
            errors.append(AssertionError(msg))

        if errors:
            raise AssertionError(errors)

    @pytest.mark.parametrize("train_type", ["sft", "dpo", "pt"])
    def test_lora(self, train_type, model_key):
        print(f"[INFO] Testing with model={model_key}, train_type={train_type}_lora")
        model_cfg = self.train_tester.load_model_config(model_key)
        cli_args = model_cfg.cli_args
        model_name_or_path = model_cfg.repo_id
        lora_loss = model_cfg.base_loss.get(f"{train_type}_lora_loss", 0)
        lora_resume_loss = model_cfg.base_loss.get(f"{train_type}_lora_resume_loss", 0)

        output_dir = os.path.join(OUTPUT_DIR, f"{train_type}_{model_key}_lora")
        update_args = {
            "model_name_or_path": model_name_or_path,
            "max_steps": MAX_STEPS,
            "save_steps": SAVE_STEPS,
            "output_dir": output_dir,
        }
        update_args.update(cli_args)

        config_path = os.path.join(CONFIG_PATH, train_type, "lora.yaml")
        updated_config_path = self.train_tester.update_training_args(config_path, output_dir, update_args)

        # 训练
        cmd = ["paddleformers-cli", "train", updated_config_path]
        
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # 保存日志
        lora_log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_lora.log")
        if training_p.stdout.strip():
            with open(lora_log_file, "w", encoding="utf-8") as f:
                f.write(training_p.stdout)

        self.train_tester.assert_result(training_p.returncode, training_p.stdout)
        

        # resume 测试
        
        resume_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lora_resume_log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_lora_resume.log")
        if resume_p.stdout.strip():
            with open(lora_resume_log_file, "w", encoding="utf-8") as f:
                f.write(resume_p.stdout)

        self.train_tester.assert_result(resume_p.returncode, resume_p.stdout)

        # check loss diff
        errors = []
        msg = self.train_tester.assert_loss(training_p.stdout, lora_loss, "Fisrt-Training")
        if msg:
            errors.append(AssertionError(msg))

        msg = self.train_tester.assert_loss(resume_p.stdout, lora_resume_loss, "Resume-Training")
        if msg:
            errors.append(AssertionError(msg))

        if errors:
            raise AssertionError(errors)


        # merge 测试
        lora_merge_cmd = ["paddleformers-cli", "export", updated_config_path]
        merge_p = subprocess.run(lora_merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.train_tester.assert_result(merge_p.returncode, merge_p.stdout)

    @pytest.mark.parametrize("train_type", ["sft", "dpo", "pt"])
    def test_full_tp_pp(self, train_type, model_key):
        print(f"[INFO] Testing with model={model_key}, train_type={train_type}_tp_pp")
        model_cfg = self.train_tester.load_model_config(model_key)
        cli_args = model_cfg.cli_args
        model_name_or_path = model_cfg.repo_id
        tp_pp_loss = model_cfg.base_loss.get(f"{train_type}_full_tp_pp_loss", 0)
        tp_pp_resume_loss = model_cfg.base_loss.get(f"{train_type}_full_tp_pp_resume_loss", 0)

        output_dir = os.path.join(OUTPUT_DIR, f"{train_type}_{model_key}_tp_pp")
        update_args = {
            "model_name_or_path": model_name_or_path,
            "max_steps": MAX_STEPS,
            "save_steps": SAVE_STEPS,
            "output_dir": output_dir,
        }
        update_args.update(cli_args)

        config_path = os.path.join(CONFIG_PATH, train_type, f"full_tp_pp.yaml")
        updated_config_path = self.train_tester.update_training_args(config_path, output_dir, update_args)

        # 训练
        
        cmd = ["paddleformers-cli", "train", updated_config_path]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        full_tp_pp_log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_full_tp_pp.log")
        if training_p.stdout.strip():
            with open(full_tp_pp_log_file, "w", encoding="utf-8") as f:
                f.write(training_p.stdout)

        self.train_tester.assert_result(training_p.returncode, training_p.stdout)

        # resume 测试
        
        resume_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        full_tp_pp_resume_log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_full_tp_pp_resume.log")
        if resume_p.stdout.strip():
            with open(full_tp_pp_resume_log_file, "w", encoding="utf-8") as f:
                f.write(resume_p.stdout)

        self.train_tester.assert_result(resume_p.returncode, resume_p.stdout)

        # check loss diff
        errors = []
        msg = self.train_tester.assert_loss(training_p.stdout, tp_pp_loss, "Fisrt-Training")
        if msg:
            errors.append(AssertionError(msg))

        msg = self.train_tester.assert_loss(resume_p.stdout, tp_pp_resume_loss, "Resume-Training")
        if msg:
            errors.append(AssertionError(msg))

        if errors:
            raise AssertionError(errors)

    @pytest.mark.parametrize("train_type", ["sft", "dpo", "pt"])
    def test_lora_tp_pp(self, train_type, model_key):
        print(f"[INFO] Testing with model={model_key}, train_type={train_type}_lora_tp_pp")
        model_cfg = self.train_tester.load_model_config(model_key)
        cli_args = model_cfg.cli_args
        model_name_or_path = model_cfg.repo_id

        lora_tp_pp_loss = model_cfg.base_loss.get(f"{train_type}_lora_tp_pp_loss", 0)
        lora_tp_pp_resume_loss = model_cfg.base_loss.get(f"{train_type}_lora_tp_pp_resume_loss", 0)

        output_dir = os.path.join(OUTPUT_DIR, f"{train_type}_{model_key}_lora_tp_pp")
        update_args = {
            "model_name_or_path": model_name_or_path,
            "max_steps": MAX_STEPS,
            "save_steps": SAVE_STEPS,
            "output_dir": output_dir,
        }
        update_args.update(cli_args)

        config_path = os.path.join(CONFIG_PATH, train_type, "lora_tp_pp.yaml")
        updated_config_path = self.train_tester.update_training_args(config_path, output_dir, update_args)

        # 训练
        
        cmd = ["paddleformers-cli", "train", updated_config_path]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_lora_tp_pp.log")
        if training_p.stdout.strip():
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(training_p.stdout)

        self.train_tester.assert_result(training_p.returncode, training_p.stdout)
        

        # resume 测试
        
        resume_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        resume_log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_lora_tp_pp_resume.log")
        if resume_p.stdout.strip():
            with open(resume_log_file, "w", encoding="utf-8") as f:
                f.write(resume_p.stdout)

        self.train_tester.assert_result(resume_p.returncode, resume_p.stdout)
        
        # check loss diff
        errors = []
        msg = self.train_tester.assert_loss(training_p.stdout, lora_tp_pp_loss, "Fisrt-Training")
        if msg:
            errors.append(AssertionError(msg))

        msg = self.train_tester.assert_loss(resume_p.stdout, lora_tp_pp_resume_loss, "Resume-Training")
        if msg:
            errors.append(AssertionError(msg))

        if errors:
            raise AssertionError(errors)

        # merge 测试
        merge_cmd = ["paddleformers-cli", "export", updated_config_path]
        merge_p = subprocess.run(merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.train_tester.assert_result(merge_p.returncode, merge_p.stdout)

    @pytest.mark.parametrize("train_type", ["sft"])
    def test_full_function_call(self, train_type, model_key):
        print(f"[INFO] Testing with model={model_key}, train_type={train_type}_full_function_call")
        model_cfg = self.train_tester.load_model_config(model_key)
        cli_args = model_cfg.cli_args
        model_name_or_path = model_cfg.repo_id

        fc_loss = model_cfg.base_loss.get(f"{train_type}_fc_loss", 0)
        fc_resume_loss = model_cfg.base_loss.get(f"{train_type}_fc_resume_loss", 0)

        output_dir = os.path.join(OUTPUT_DIR, f"{train_type}_{model_key}_full_function_call")
        update_args = {
            "model_name_or_path": model_name_or_path,
            "max_steps": MAX_STEPS,
            "save_steps": SAVE_STEPS,
            "output_dir": output_dir,
        }
        update_args.update(cli_args)

        config_path = os.path.join(CONFIG_PATH, train_type, "full_function_call.yaml")
        updated_config_path = self.train_tester.update_training_args(config_path, output_dir, update_args)

        # 训练
        
        cmd = ["paddleformers-cli", "train", updated_config_path]
        training_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_full_function_call.log")
        if training_p.stdout.strip():
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(training_p.stdout)

        self.train_tester.assert_result(training_p.returncode, training_p.stdout)
        

        # resume 测试
        
        resume_p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        resume_log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_full_function_call_resume.log")
        if resume_p.stdout.strip():
            with open(resume_log_file, "w", encoding="utf-8") as f:
                f.write(resume_p.stdout)

        self.train_tester.assert_result(resume_p.returncode, resume_p.stdout)

        # check loss diff
        errors = []
        msg = self.train_tester.assert_loss(training_p.stdout, fc_loss, "Fisrt-Training")
        if msg:
            errors.append(AssertionError(msg))

        msg = self.train_tester.assert_loss(resume_p.stdout, fc_resume_loss, "Resume-Training")
        if msg:
            errors.append(AssertionError(msg))

        if errors:
            raise AssertionError(errors)