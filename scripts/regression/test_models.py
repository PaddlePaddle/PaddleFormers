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

"""
Model Training Regression Test Suite.

This module provides comprehensive regression tests for model training workflows,
including full training, LoRA fine-tuning, tensor/pipeline parallelism, and
function call training modes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest
import yaml
from create_and_check_model_generate import create_and_check_model_generate
from prepare_datasets import prepare_all_datasets

CONFIG_PATH = "./examples/config/"
LOG_PATH = "./model_unittest_logs"
OUTPUT_DIR = tempfile.TemporaryDirectory().name

MAX_STEPS = 2
SAVE_STEPS = 1
MAX_RESUME_STEPS = 2
SAVE_RESUME_STEPS = 1000

LOSS_TOLERANCE = 1e-10

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["FLAGS_embedding_deterministic"] = "1"
os.environ["FLAGS_cudnn_deterministic"] = "1"


@dataclass
class ModelConfig:
    """Configuration container for model testing parameters.

    Attributes:
        name: Model identifier key.
        repo_id: Repository identifier for model weights.
        model_type: Model type ('text' or 'vl'), defaults to 'text'.
        cli_args: Additional CLI arguments for training.
        base_loss: Expected baseline loss values for different training modes.
        base_result: Expected generation results for different training modes.
    """

    name: str
    repo_id: str
    model_type: str = "text"
    cli_args: Dict[str, Any] = field(default_factory=dict)
    base_loss: Dict[str, float] = field(default_factory=dict)
    base_result: Dict[str, List[List[int]]] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Container for training execution results.

    Attributes:
        return_code: Process return code.
        stdout: Captured standard output.
        log_file: Path to the saved log file.
    """

    return_code: int
    stdout: str
    log_file: str


class CompactListDumper(yaml.SafeDumper):
    """Custom YAML dumper that uses flow style for numeric lists."""

    def represent_sequence(self, tag, sequence, flow_style=None):
        """Use flow style for lists containing only numbers."""
        if all(isinstance(item, (int, float)) for item in sequence):
            return super().represent_sequence(tag, sequence, flow_style=True)
        return super().represent_sequence(tag, sequence, flow_style)

    def ignore_aliases(self, data):
        """Disable YAML aliases for cleaner output."""
        return True


class TrainTester:
    """Helper class providing utilities for model training tests.

    This class encapsulates common operations for model training regression tests,
    including configuration loading, YAML manipulation, loss validation, and
    baseline updates.
    """

    CONFIG_FILE_PATH = "./scripts/regression/config.yaml"

    def load_model_config(self, model_key: str) -> ModelConfig:
        """Load model configuration from YAML file.

        Args:
            model_key: The model identifier to load configuration for.

        Returns:
            ModelConfig object containing the model's test configuration.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            KeyError: If the model_key is not found in the configuration.
        """
        with open(self.CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if model_key not in data:
            pytest.skip(f"Model '{model_key}' not found in config")
            
        model_cfg = data[model_key]
        return ModelConfig(
            name=model_key,
            repo_id=model_cfg.get("repo_id"),
            model_type=model_cfg.get("model_type", "text"),
            cli_args=model_cfg.get("cli_args", {}),
            base_loss=model_cfg.get("base_loss", {}),
            base_result=model_cfg.get("base_result", {}),
        )

    def update_training_args(self, yaml_path: str, tmp_dir: str, updates: Dict[str, Any]) -> str:
        """Update training arguments in a YAML configuration file.

        Creates a new YAML file with updated parameters while preserving
        the original file.

        Args:
            yaml_path: Path to the original YAML configuration.
            tmp_dir: Directory to store the updated configuration.
            updates: Dictionary of parameters to update.

        Returns:
            Path to the updated YAML configuration file.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config.update(updates)
        os.makedirs(tmp_dir, exist_ok=True)

        updated_yaml_path = os.path.join(tmp_dir, f"updated_{os.path.basename(yaml_path)}")
        with open(updated_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, indent=4, allow_unicode=True, sort_keys=False)

        return updated_yaml_path

    def update_baseline(
        self,
        model_key: str,
        train_type: str,
        test_type: str,
        new_loss: float,
        new_resume_loss: float,
        new_result: List[List[int]],
    ) -> None:
        """Update baseline values in the configuration file.

        Safely updates the baseline loss and result values for a specific
        model and training type combination.

        Args:
            model_key: The model identifier.
            train_type: Training type (e.g., 'sft', 'dpo', 'pt').
            test_type: Test type (e.g., 'full', 'lora', 'full_tp_pp').
            new_loss: New baseline loss value for first training.
            new_resume_loss: New baseline loss value for resume training.
            new_result: New baseline generation result.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the model_key is not found or config is invalid.
        """
        config_path = self.CONFIG_FILE_PATH

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        backup_path = config_path + ".bak"

        # Create and validate backup
        self._create_backup(config_path, backup_path)

        try:
            config = self._load_and_validate_config(config_path, model_key)
            self._update_config_values(config, model_key, train_type, test_type, new_loss, new_resume_loss, new_result)
            self._save_config_safely(config, config_path)

        except Exception as e:
            # Restore backup on failure
            print(f"[ERROR] Failed to update config: {e}, restoring backup")
            shutil.copyfile(backup_path, config_path)
            raise e
        finally:
            if os.path.exists(backup_path):
                os.remove(backup_path)

    def _create_backup(self, config_path: str, backup_path: str) -> None:
        """Create and validate a backup of the configuration file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                original_content = f.read()
                if not original_content.strip():
                    raise ValueError("Config file is empty")

            shutil.copyfile(config_path, backup_path)

            with open(backup_path, "r", encoding="utf-8") as f:
                if not f.read().strip():
                    raise ValueError("Backup file is empty, aborting update")

        except Exception as e:
            print(f"[ERROR] Failed to create backup: {e}")
            if os.path.exists(backup_path):
                os.remove(backup_path)
            raise e

    def _load_and_validate_config(self, config_path: str, model_key: str) -> Dict[str, Any]:
        """Load and validate the configuration file."""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if model_key not in config:
            raise ValueError(f"Model key {model_key} not found in config")

        return config

    def _update_config_values(
        self,
        config: Dict[str, Any],
        model_key: str,
        train_type: str,
        test_type: str,
        new_loss: float,
        new_resume_loss: float,
        new_result: List[List[int]],
    ) -> None:
        """Update the loss and result values in the configuration."""
        model_cfg = config[model_key]

        # Update loss value
        loss_key = f"{train_type}_{test_type}_loss"
        resume_loss_key = f"{train_type}_{test_type}_resume_loss"
        if "base_loss" not in model_cfg:
            model_cfg["base_loss"] = {}
        model_cfg["base_loss"][loss_key] = new_loss
        model_cfg["base_loss"][resume_loss_key] = new_resume_loss

        # Update result value (convert Paddle Tensor if necessary)
        result_key = f"{train_type}_{test_type}_excepted_result"
        if hasattr(new_result, "numpy"):
            new_result = new_result.numpy().tolist()

        if "base_result" not in model_cfg:
            model_cfg["base_result"] = {}
        model_cfg["base_result"][result_key] = new_result

        print(f"[UPDATE INFO] Updated {model_key} base_loss, resume_loss and base_result")

    def _save_config_safely(self, config: Dict[str, Any], config_path: str) -> None:
        """Safely save the configuration using atomic file operations."""
        temp_path = config_path + ".tmp"

        with open(temp_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, Dumper=CompactListDumper, indent=4, allow_unicode=True, sort_keys=False)

        # Atomic replace
        os.replace(temp_path, config_path)

    def assert_loss(
        self, output: str, base_loss: float, phase_name: str, update_flag: bool = False
    ) -> Tuple[float, Optional[str]]:
        """Validate training loss against baseline.

        Args:
            output: Training output containing loss values.
            base_loss: Expected baseline loss value.
            phase_name: Name of the training phase for logging.
            update_flag: If True, skip validation and return actual loss.

        Returns:
            Tuple of (actual_loss, error_message). Error message is None if valid.
        """
        loss_pattern = re.compile(r"(?<![A-Za-z_])loss:\s*([0-9]+\.[0-9]+)")
        losses = [float(m.group(1)) for m in loss_pattern.finditer(output)]

        avg_loss = round(sum(losses) / len(losses), 10) if losses else 0
        print(f"{phase_name} loss: {avg_loss} || base loss: {base_loss}")

        if update_flag:
            return avg_loss, None

        if abs(avg_loss - base_loss) > LOSS_TOLERANCE:
            return avg_loss, f"{phase_name} loss: {avg_loss}, base_loss: {base_loss}, difference detected!"

        return avg_loss, None

    def extract_loss(self, log_content: str) -> Optional[float]:
        """Extract the first loss value from log content.

        Args:
            log_content: Log content to search.

        Returns:
            First loss value found, or None if not found.
        """
        loss_pattern = re.compile(r"(?<![A-Za-z_])loss:\s*([0-9]+\.[0-9]+)")
        match = loss_pattern.search(log_content)
        return float(match.group(1)) if match else None

    def assert_loss_consistent(
        self, log_file: str, resume_log_file: str
    ) -> Tuple[Tuple[Optional[float], Optional[float]], Optional[str]]:
        """Verify loss consistency between training and resume phases.

        Compares the last loss from First training with the first loss
        from resumed training to ensure checkpoint integrity.

        Args:
            log_file: Path to First training log file.
            resume_log_file: Path to resume training log file.

        Returns:
            Tuple of ((last_training_loss, first_resume_loss), error_message).
            Error message is None if consistent.
        """
        with open(log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
        with open(resume_log_file, "r", encoding="utf-8") as f:
            resume_log_content = f.read()

        loss_pattern = re.compile(r"(?<![A-Za-z_])loss:\s*([0-9]+\.[0-9]+)")

        # Extract last loss from training log
        training_losses = loss_pattern.findall(log_content)
        if not training_losses:
            return (None, None), f"Failed to extract any loss from training log: {log_file}"
        last_training_loss = float(training_losses[-1])

        # Extract first loss from resume log
        resume_loss_matches = list(loss_pattern.finditer(resume_log_content))
        if not resume_loss_matches:
            return (last_training_loss, None), f"Failed to extract first loss from resume log: {resume_log_file}"
        first_resume_loss = float(resume_loss_matches[0].group(1))

        # Compare losses
        if abs(last_training_loss - first_resume_loss) > LOSS_TOLERANCE:
            return (
                last_training_loss,
                first_resume_loss,
            ), f"Loss mismatch! Training loss: {last_training_loss}, Resume loss: {first_resume_loss}"

        print(f"Loss transition consistent: {last_training_loss}")
        return (last_training_loss, first_resume_loss), None

    def assert_result(self, ret_code: int, log_output: str) -> None:
        """Assert training completed successfully.

        Args:
            ret_code: Process return code.
            log_output: Training log output.

        Raises:
            AssertionError: If training failed (non-zero return code).
        """
        if ret_code != 0:
            print("\n".join(log_output.strip().splitlines()[-30:]))
            raise AssertionError("Training Failed")

    def get_model_path(self, repo_id: str) -> str:
        """Get the local path for a model repository.

        Args:
            repo_id: Repository identifier.

        Returns:
            Local filesystem path to the model.
        """
        if "PF_HOME" in os.environ:
            return os.path.join(os.environ["PF_HOME"], repo_id)
        return f"./{repo_id}"

    def run_training(self, config_path: str, log_file: str, sleep_before: int = 0) -> TrainingResult:
        """Execute a training run and save logs.

        Args:
            config_path: Path to the training configuration.
            log_file: Path to save the training log.
            sleep_before: Seconds to sleep before starting.

        Returns:
            TrainingResult containing execution results.
        """
        if sleep_before > 0:
            time.sleep(sleep_before)

        cmd_str = f"paddleformers-cli train {config_path} > {log_file} 2>&1"
        return_code = os.system(cmd_str)
        return_code = return_code >> 8 if return_code > 0 else return_code

        stdout_content = ""
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                stdout_content = f.read()

        return TrainingResult(return_code=return_code, stdout=stdout_content, log_file=log_file)

    def run_export(
        self, model_name_or_path: str, output_dir: str, log_file: str, lora: bool = True
    ) -> subprocess.CompletedProcess:
        """Export and merge LoRA weights.

        Args:
            model_name_or_path: Path to the base model.
            output_dir: Directory containing checkpoints.
            log_file: Path to save the export log.
            lora: Whether this is a LoRA export.

        Returns:
            Completed process result.
        """
        export_update_args = {
            "model_name_or_path": model_name_or_path,
            "output_dir": output_dir,
        }
        export_config_path = "./examples/config/run_export.yaml"
        updated_config_path = self.update_training_args(export_config_path, CONFIG_PATH, export_update_args)

        cmd = ["paddleformers-cli", "export", updated_config_path]
        if lora:
            cmd.append("lora=True")

        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Save export log
        if process.stdout and process.stdout.strip():
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(process.stdout)

        return process

    def cleanup_checkpoint(self, output_dir: str, checkpoint_name: str = "checkpoint-2") -> None:
        """Remove a checkpoint directory for resume testing.

        Args:
            output_dir: Base output directory.
            checkpoint_name: Name of checkpoint to remove.
        """
        checkpoint_path = os.path.join(output_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)


class BaseTrainingTest:
    """Base class for training test workflows.

    Provides common test workflow patterns that can be customized
    by specific test implementations.
    """

    def __init__(self, train_tester: TrainTester):
        self.tester = train_tester

    def execute_training_workflow(
        self,
        model_key: str,
        train_type: str,
        test_type: str,
        config_subpath: str,
        model_cfg: ModelConfig,
        should_update: bool,
        requires_export: bool = False,
    ) -> None:
        """Execute a complete training workflow including train, resume, and validation.

        Args:
            model_key: Model identifier.
            train_type: Training type (sft, dpo, pt).
            test_type: Test type (full, lora, etc.).
            config_subpath: Subpath to config file under CONFIG_PATH.
            model_cfg: Model configuration.
            should_update: Whether to update baselines.
            requires_export: Whether LoRA export is required.
        """
        model_name_or_path = self.tester.get_model_path(model_cfg.repo_id)

        # Get baseline values
        loss_key = f"{train_type}_{test_type}_loss"
        resume_loss_key = f"{train_type}_{test_type}_resume_loss"
        result_key = f"{train_type}_{test_type}_excepted_result"
        base_loss = model_cfg.base_loss.get(loss_key, 0)
        base_resume_loss = model_cfg.base_loss.get(resume_loss_key, 0)
        expected_result = model_cfg.base_result.get(result_key, 0)

        # Setup output directory
        output_dir = os.path.join(OUTPUT_DIR, f"{train_type}_{model_key}_{test_type}")
        config_path = os.path.join(CONFIG_PATH, config_subpath)

        # Prepare training arguments
        update_args = {
            "model_name_or_path": model_name_or_path,
            "max_steps": MAX_STEPS,
            "eval_steps": MAX_STEPS,
            "save_steps": SAVE_STEPS,
            "output_dir": output_dir,
            **model_cfg.cli_args,
        }

        # Execute First training
        updated_config = self.tester.update_training_args(config_path, output_dir, update_args)
        log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_{test_type}.log")
        training_result = self.tester.run_training(updated_config, log_file)
        self.tester.assert_result(training_result.return_code, training_result.stdout)

        # Thorough cleanup between training runs to avoid state pollution
        cleanup_cmds = [
            # Kill launcher processes
            "pkill -9 -f 'paddleformers/cli/launcher.py' 2>/dev/null || true",
            # Kill any remaining paddle/python training processes
            "pkill -9 -f 'paddle.distributed.launch' 2>/dev/null || true",
            # Clean up NCCL shared memory files that might cause issues
            "rm -f /dev/shm/nccl-* 2>/dev/null || true",
            "rm -f /dev/shm/*nccl* 2>/dev/null || true",
        ]
        for cmd in cleanup_cmds:
            subprocess.run(cmd, shell=True)

        # Wait for processes to fully terminate and release GPU resources
        time.sleep(5)

        # Debug: Check for any remaining processes that might interfere
        remaining_procs = subprocess.run(
            "ps aux | grep -E 'paddleformers|python.*train' | grep -v grep || true",
            shell=True,
            capture_output=True,
            text=True,
        )
        if remaining_procs.stdout.strip():
            print(f"[DEBUG] Remaining processes before resume training:\n{remaining_procs.stdout}")
        else:
            print("[DEBUG] No remaining paddleformers/training processes found.")

        # Execute resume training - use sed-like in-place modification (matching shell script behavior)
        # This modifies the same config file used in first training, not regenerating from original
        resume_sed_cmds = [
            f"sed -i 's|^\\s*max_steps:.*|max_steps: {MAX_RESUME_STEPS}|' {updated_config}",
            f"sed -i 's|^\\s*eval_steps:.*|eval_steps: 1000|' {updated_config}",
            f"sed -i 's|^\\s*save_steps:.*|save_steps: {SAVE_RESUME_STEPS}|' {updated_config}",
        ]
        for cmd in resume_sed_cmds:
            subprocess.run(cmd, shell=True)
        self.tester.cleanup_checkpoint(output_dir)

        resume_log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_{test_type}_resume.log")
        resume_result = self.tester.run_training(updated_config, resume_log_file)
        self.tester.assert_result(resume_result.return_code, resume_result.stdout)

        errors = []

        actual_loss, msg = self.tester.assert_loss(training_result.stdout, base_loss, "First-Training", should_update)
        if msg:
            errors.append(AssertionError(msg))

        actual_resume_loss, msg = self.tester.assert_loss(
            resume_result.stdout, base_resume_loss, "Resume-Training", should_update
        )
        if msg:
            errors.append(AssertionError(msg))

        _, msg = self.tester.assert_loss_consistent(log_file, resume_log_file)
        if msg:
            errors.append(AssertionError(msg))

        generate_dir = output_dir
        if requires_export:
            export_log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_{test_type}_export.log")
            merge_result = self.tester.run_export(model_name_or_path, output_dir, export_log_file, lora=True)
            self.tester.assert_result(merge_result.returncode, merge_result.stdout)
            generate_dir = os.path.join(output_dir, "export")

        # Test model generation
        generate_log_file = os.path.join(LOG_PATH, f"{model_key}_{train_type}_{test_type}_generate.log")
        skip_generation = model_key in ["qwen2_moe", "deepseek_v3"]
        if skip_generation:
            result = None
        else:
            result = self._run_generation_test(
                model_key, generate_dir, expected_result, should_update, generate_log_file
            )
        # Update baseline if needed
        if should_update:
            new_result = result[0] if result else None
            if new_result is not None:
                self.tester.update_baseline(
                    model_key=model_key,
                    train_type=train_type,
                    test_type=test_type,
                    new_loss=actual_loss,
                    new_resume_loss=actual_resume_loss,
                    new_result=new_result,
                )
            else:
                print(f"[SKIP] Skipping baseline update for {model_key} (generation test skipped)")

        if errors:
            raise AssertionError(errors)

    def _run_generation_test(
        self, model_key: str, output_dir: str, expected_result: Any, should_update: bool, log_file: str = ""
    ) -> List[List[int]]:
        """Run model generation test with error handling and logging.

        Args:
            model_key: Model identifier.
            output_dir: Directory containing the model.
            expected_result: Expected generation result.
            should_update: Whether in update mode.
            log_file: Path to save the generation test log.

        Returns:
            Generation result or placeholder if update mode.
        """

        log_lines = []
        log_lines.append(f"Model: {model_key}")
        log_lines.append("-" * 50)
        try:
            result = create_and_check_model_generate(model_key, output_dir, expected_result)
            log_lines.append("=== Execution Output(Success) ===")
            return result
        except Exception as e:
            log_lines.append("=== Execution Output (Failed) ===")
            log_lines.append(f"Error: {e}")
            if should_update:
                print(f"[UPDATE MODE] Model generate failed but continuing: {e}")
                return [[]]
            raise e
        finally:
            # Save logs to file
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(log_lines))
                print(f"[INFO] Generation test log saved to: {log_file}")


@pytest.fixture(scope="session", autouse=True)
def prepare_datasets_once():
    """Download all datasets once at the start of the test session."""
    prepare_all_datasets()


class TestTrain:
    """Pytest test class for model training regression tests.

    This class contains parameterized tests for various training modes:
    - Full training (test_full)
    - LoRA fine-tuning (test_lora)
    - Full_TP_PP training with tensor/pipeline parallelism (test_full_tp_pp)
    - LoRA_TP_PP with tensor/pipeline parallelism (test_lora_tp_pp)
    - Function_Call training with function calling (test_full_function_call)
    """

    @pytest.fixture(autouse=True)
    def setup_class(self):
        """Initialize test environment before each test."""
        self.train_tester = TrainTester()
        self.workflow = BaseTrainingTest(self.train_tester)

        subprocess.run("pkill -9 -f 'paddleformers/cli/launcher.py' 2>/dev/null || true", shell=True)

        dist_log_dir = "./paddleformers_dist_log"
        if os.path.exists(dist_log_dir):
            shutil.rmtree(dist_log_dir)

        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(LOG_PATH, exist_ok=True)

    def _should_update_baseline(self, request, model_key: str) -> bool:
        """Check if baseline should be updated for this test.

        Args:
            request: Pytest request fixture.
            model_key: Current model identifier.

        Returns:
            True if baseline should be updated.
        """
        update_baseline = request.config.getoption("--update-baseline")
        return update_baseline in ["all", model_key]

    @pytest.mark.model_type("text")
    @pytest.mark.parametrize("train_type", ["sft", "dpo", "pt"])
    def test_full(self, train_type: str, model_key: str, request) -> None:
        """Test full model training workflow for text models.

        Args:
            train_type: Training type (sft, dpo, pt).
            model_key: Model identifier from pytest parametrization.
            request: Pytest request fixture.
        """
        model_cfg = self.train_tester.load_model_config(model_key)
        print(f"\n[INFO] Testing model={model_key}, train_type={train_type}_full")
        should_update = self._should_update_baseline(request, model_key)

        self.workflow.execute_training_workflow(
            model_key=model_key,
            train_type=train_type,
            test_type="full",
            config_subpath=f"{train_type}/full.yaml",
            model_cfg=model_cfg,
            should_update=should_update,
            requires_export=False,
        )

    @pytest.mark.model_type("text")
    @pytest.mark.parametrize("train_type", ["sft", "dpo", "pt"])
    def test_lora(self, train_type: str, model_key: str, request) -> None:
        """Test LoRA fine-tuning workflow for text models.

        Args:
            train_type: Training type (sft, dpo, pt).
            model_key: Model identifier from pytest parametrization.
            request: Pytest request fixture.
        """
        model_cfg = self.train_tester.load_model_config(model_key)
        print(f"\n[INFO] Testing model={model_key}, train_type={train_type}_lora")
        should_update = self._should_update_baseline(request, model_key)

        self.workflow.execute_training_workflow(
            model_key=model_key,
            train_type=train_type,
            test_type="lora",
            config_subpath=f"{train_type}/lora.yaml",
            model_cfg=model_cfg,
            should_update=should_update,
            requires_export=True,
        )

    @pytest.mark.model_type("text")
    @pytest.mark.parametrize("train_type", ["sft", "dpo", "pt"])
    def test_full_tp_pp(self, train_type: str, model_key: str, request) -> None:
        """Test full training with tensor and pipeline parallelism for text models.

        Args:
            train_type: Training type (sft, dpo, pt).
            model_key: Model identifier from pytest parametrization.
            request: Pytest request fixture.
        """
        model_cfg = self.train_tester.load_model_config(model_key)
        print(f"\n[INFO] Testing model={model_key}, train_type={train_type}_full_tp_pp")
        should_update = self._should_update_baseline(request, model_key)

        self.workflow.execute_training_workflow(
            model_key=model_key,
            train_type=train_type,
            test_type="full_tp_pp",
            config_subpath=f"{train_type}/full_tp_pp.yaml",
            model_cfg=model_cfg,
            should_update=should_update,
            requires_export=False,
        )

    @pytest.mark.model_type("text")
    @pytest.mark.parametrize("train_type", ["sft", "pt", "dpo"])
    def test_lora_tp_pp(self, train_type: str, model_key: str, request) -> None:
        """Test LoRA training with tensor and pipeline parallelism for text models.

        Args:
            train_type: Training type (sft, pt, dpo).
            model_key: Model identifier from pytest parametrization.
            request: Pytest request fixture.
        """
        model_cfg = self.train_tester.load_model_config(model_key)
        print(f"\n[INFO] Testing model={model_key}, train_type={train_type}_lora_tp_pp")
        should_update = self._should_update_baseline(request, model_key)

        self.workflow.execute_training_workflow(
            model_key=model_key,
            train_type=train_type,
            test_type="lora_tp_pp",
            config_subpath=f"{train_type}/lora_tp_pp.yaml",
            model_cfg=model_cfg,
            should_update=should_update,
            requires_export=True,
        )

    @pytest.mark.model_type("text")
    @pytest.mark.parametrize("train_type", ["sft", "dpo"])
    def test_full_function_call(self, train_type: str, model_key: str, request) -> None:
        """Test full training with function calling support for text models.

        Args:
            train_type: Training type (sft, dpo).
            model_key: Model identifier from pytest parametrization.
            request: Pytest request fixture.
        """
        model_cfg = self.train_tester.load_model_config(model_key)
        print(f"\n[INFO] Testing model={model_key}, train_type={train_type}_full_function_call")
        should_update = self._should_update_baseline(request, model_key)

        self.workflow.execute_training_workflow(
            model_key=model_key,
            train_type=train_type,
            test_type="full_function_call",
            config_subpath=f"{train_type}/full_function_call.yaml",
            model_cfg=model_cfg,
            should_update=should_update,
            requires_export=False,
        )

    @pytest.mark.model_type("vl")
    @pytest.mark.parametrize("train_type", ["sft-vl"])
    def test_full_vl(self, train_type: str, model_key: str, request) -> None:
        """Test full model training workflow for VL models.

        Args:
            train_type: Training type (sft-vl, dpo-vl).
            model_key: Model identifier from pytest parametrization.
            request: Pytest request fixture.
        """

        model_cfg = self.train_tester.load_model_config(model_key)
        print(f"\n[INFO] Testing model={model_key}, train_type={train_type}_full")
        should_update = self._should_update_baseline(request, model_key)

        self.workflow.execute_training_workflow(
            model_key=model_key,
            train_type=train_type,
            test_type="full",
            config_subpath=f"{train_type}/full.yaml",
            model_cfg=model_cfg,
            should_update=should_update,
            requires_export=False,
        )

    @pytest.mark.model_type("vl")
    @pytest.mark.parametrize("train_type", ["sft-vl"])
    def test_lora_vl(self, train_type: str, model_key: str, request) -> None:
        """Test LoRA fine-tuning workflow for VL models.

        Args:
            train_type: Training type (sft-vl, dpo-vl).
            model_key: Model identifier from pytest parametrization.
            request: Pytest request fixture.
        """

        if model_key == "qwen3_vl_moe":
            pytest.skip("Unsupported resume hang")

        model_cfg = self.train_tester.load_model_config(model_key)
        print(f"\n[INFO] Testing model={model_key}, train_type={train_type}_lora")
        should_update = self._should_update_baseline(request, model_key)

        self.workflow.execute_training_workflow(
            model_key=model_key,
            train_type=train_type,
            test_type="lora",
            config_subpath=f"{train_type}/lora.yaml",
            model_cfg=model_cfg,
            should_update=should_update,
            requires_export=False,
        )

    @pytest.mark.model_type("vl")
    @pytest.mark.parametrize("train_type", ["sft-vl"])
    def test_full_tp_vl(self, train_type: str, model_key: str, request) -> None:
        """Test full training with tensor parallelism for VL models.

        Args:
            train_type: Training type (sft-vl, dpo-vl).
            model_key: Model identifier from pytest parametrization.
            request: Pytest request fixture.
        """
        model_cfg = self.train_tester.load_model_config(model_key)
        print(f"\n[INFO] Testing model={model_key}, train_type={train_type}_full_tp")

        if model_key == "paddleocr_vl":
            pytest.skip("Unsupported")

        if model_key == "qwen3_vl":
            pytest.skip("Unsupported resume hang")

        should_update = self._should_update_baseline(request, model_key)

        self.workflow.execute_training_workflow(
            model_key=model_key,
            train_type=train_type,
            test_type="full_tp",
            config_subpath=f"{train_type}/full_tp.yaml",
            model_cfg=model_cfg,
            should_update=should_update,
            requires_export=False,
        )

    @pytest.mark.model_type("vl")
    @pytest.mark.parametrize("train_type", ["sft-vl"])
    def test_full_fsdp_vl(self, train_type: str, model_key: str, request) -> None:
        """Test full FSDP training for VL models.

        Args:
            train_type: Training type (sft-vl, dpo-vl).
            model_key: Model identifier from pytest parametrization.
            request: Pytest request fixture.
        """

        model_cfg = self.train_tester.load_model_config(model_key)
        print(f"\n[INFO] Testing model={model_key}, train_type={train_type}_full_fsdp")

        if model_key == "paddleocr_vl":
            pytest.skip("Unsupported")
        if model_key == "qwen3_vl_moe" or model_key == "qwen3_vl":
            pytest.skip("Unsupported resume hang")

        should_update = self._should_update_baseline(request, model_key)

        self.workflow.execute_training_workflow(
            model_key=model_key,
            train_type=train_type,
            test_type="full_fsdp",
            config_subpath=f"{train_type}/full_fsdp.yaml",
            model_cfg=model_cfg,
            should_update=should_update,
            requires_export=False,
        )
