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

import sys
import unittest
from unittest.mock import MagicMock, patch


def _setup_tuner_mocks():
    """Set up sys.modules mocks for the heavy import chain in tuner.py."""
    modules_to_mock = [
        "paddleformers.cli.train.auto_parallel",
        "paddleformers.cli.train.dpo",
        "paddleformers.cli.train.sft",
        "paddleformers.cli.train.deepseek_v3_pretrain",
        "paddleformers.cli.train.ernie_pretrain",
    ]
    saved = {}
    for mod_name in modules_to_mock:
        if mod_name in sys.modules:
            saved[mod_name] = sys.modules[mod_name]
        sys.modules[mod_name] = MagicMock()
    # Force re-import of tuner module
    if "paddleformers.cli.train.tuner" in sys.modules:
        del sys.modules["paddleformers.cli.train.tuner"]
    return saved


def _teardown_tuner_mocks(saved):
    """Restore sys.modules after mocking."""
    for mod_name in [
        "paddleformers.cli.train.auto_parallel",
        "paddleformers.cli.train.dpo",
        "paddleformers.cli.train.sft",
        "paddleformers.cli.train.deepseek_v3_pretrain",
        "paddleformers.cli.train.ernie_pretrain",
        "paddleformers.cli.train.tuner",
    ]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    for mod_name, mod in saved.items():
        sys.modules[mod_name] = mod


class TestCheckPath(unittest.TestCase):
    """Tests for check_path function."""

    def setUp(self):
        self._saved = _setup_tuner_mocks()

    def tearDown(self):
        _teardown_tuner_mocks(self._saved)

    def test_none_path_raises(self):
        """Test check_path raises ValueError when path is None."""
        from paddleformers.cli.train.tuner import check_path

        with self.assertRaises(ValueError) as ctx:
            check_path(None)
        self.assertIn("None", str(ctx.exception))

    def test_valid_path_passes(self):
        """Test check_path does not raise for a valid path."""
        from paddleformers.cli.train.tuner import check_path

        # Should not raise any exception
        check_path("/some/valid/path")


class TestTrainingFunction(unittest.TestCase):
    """Tests for _training_function."""

    def setUp(self):
        self._saved = _setup_tuner_mocks()
        from paddleformers.cli.train import tuner as tuner_mod

        self.tuner_mod = tuner_mod

    def tearDown(self):
        _teardown_tuner_mocks(self._saved)

    def test_sft_stage_calls_run_sft(self):
        """Test _training_function calls run_sft for SFT stage."""
        mock_model_args = MagicMock(stage="SFT")
        mock_data_args = MagicMock(dataset_type="sft", train_dataset_path="/path", eval_dataset_path="/path")
        mock_preprocess_args = MagicMock()
        mock_generating_args = MagicMock()
        mock_finetuning_args = MagicMock()

        with patch.object(
            self.tuner_mod,
            "get_train_args",
            return_value=(
                mock_model_args,
                mock_data_args,
                mock_preprocess_args,
                mock_generating_args,
                mock_finetuning_args,
            ),
        ):
            with patch.object(self.tuner_mod, "run_sft") as mock_run_sft:
                self.tuner_mod._training_function({"args": {}})
                mock_run_sft.assert_called_once()

    def test_dpo_stage_calls_run_dpo(self):
        """Test _training_function calls run_dpo for DPO stage."""
        mock_model_args = MagicMock(stage="DPO")
        mock_data_args = MagicMock(dataset_type="sft", train_dataset_path="/path", eval_dataset_path="/path")
        mock_preprocess_args = MagicMock()
        mock_generating_args = MagicMock()
        mock_finetuning_args = MagicMock()

        with patch.object(
            self.tuner_mod,
            "get_train_args",
            return_value=(
                mock_model_args,
                mock_data_args,
                mock_preprocess_args,
                mock_generating_args,
                mock_finetuning_args,
            ),
        ):
            with patch.object(self.tuner_mod, "run_dpo") as mock_run_dpo:
                self.tuner_mod._training_function({"args": {}})
                mock_run_dpo.assert_called_once()

    def test_unknown_stage_raises(self):
        """Test _training_function raises ValueError for unknown stage."""
        mock_model_args = MagicMock(stage="UNKNOWN")
        mock_data_args = MagicMock(dataset_type="sft", train_dataset_path="/path", eval_dataset_path="/path")
        mock_preprocess_args = MagicMock()
        mock_generating_args = MagicMock()
        mock_finetuning_args = MagicMock()

        with patch.object(
            self.tuner_mod,
            "get_train_args",
            return_value=(
                mock_model_args,
                mock_data_args,
                mock_preprocess_args,
                mock_generating_args,
                mock_finetuning_args,
            ),
        ):
            with self.assertRaises(ValueError) as ctx:
                self.tuner_mod._training_function({"args": {}})
            self.assertIn("Unknown task", str(ctx.exception))

    def test_vl_stage_skips_path_check(self):
        """Test _training_function skips path check for VL stage."""
        mock_model_args = MagicMock(stage="VL-SFT")
        mock_data_args = MagicMock(dataset_type="sft", train_dataset_path=None, eval_dataset_path=None)
        mock_preprocess_args = MagicMock()
        mock_generating_args = MagicMock()
        mock_finetuning_args = MagicMock()

        with patch.object(
            self.tuner_mod,
            "get_train_args",
            return_value=(
                mock_model_args,
                mock_data_args,
                mock_preprocess_args,
                mock_generating_args,
                mock_finetuning_args,
            ),
        ):
            with patch.object(self.tuner_mod, "run_sft") as mock_run_sft:
                # Should not raise even though paths are None
                self.tuner_mod._training_function({"args": {}})
                mock_run_sft.assert_called_once()

    def test_pretrain_stage_skips_path_check(self):
        """Test _training_function skips path check for pretrain dataset_type."""
        mock_model_args = MagicMock(stage="SFT")
        mock_data_args = MagicMock(dataset_type="pretrain", train_dataset_path=None, eval_dataset_path=None)
        mock_preprocess_args = MagicMock()
        mock_generating_args = MagicMock()
        mock_finetuning_args = MagicMock()

        with patch.object(
            self.tuner_mod,
            "get_train_args",
            return_value=(
                mock_model_args,
                mock_data_args,
                mock_preprocess_args,
                mock_generating_args,
                mock_finetuning_args,
            ),
        ):
            with patch.object(self.tuner_mod, "run_sft") as mock_run_sft:
                self.tuner_mod._training_function({"args": {}})
                mock_run_sft.assert_called_once()

    def test_non_pretrain_null_train_path_raises(self):
        """Test _training_function raises when train_dataset_path is None for non-pretrain."""
        mock_model_args = MagicMock(stage="SFT")
        mock_data_args = MagicMock(dataset_type="sft", train_dataset_path=None, eval_dataset_path="/path")
        mock_preprocess_args = MagicMock()
        mock_generating_args = MagicMock()
        mock_finetuning_args = MagicMock()

        with patch.object(
            self.tuner_mod,
            "get_train_args",
            return_value=(
                mock_model_args,
                mock_data_args,
                mock_preprocess_args,
                mock_generating_args,
                mock_finetuning_args,
            ),
        ):
            with self.assertRaises(ValueError):
                self.tuner_mod._training_function({"args": {}})


class TestRunTuner(unittest.TestCase):
    """Tests for run_tuner function."""

    def setUp(self):
        self._saved = _setup_tuner_mocks()
        from paddleformers.cli.train import tuner as tuner_mod

        self.tuner_mod = tuner_mod

    def tearDown(self):
        _teardown_tuner_mocks(self._saved)

    def test_run_tuner_delegates(self):
        """Test run_tuner reads args and calls _training_function."""
        with patch.object(self.tuner_mod, "read_args", return_value={"output_dir": "/tmp/out"}) as mock_read_args:
            with patch.object(self.tuner_mod, "_training_function") as mock_training_func:
                self.tuner_mod.run_tuner({"output_dir": "/tmp/out"})
                mock_read_args.assert_called_once_with({"output_dir": "/tmp/out"})
                mock_training_func.assert_called_once()

    def test_run_tuner_with_none_args(self):
        """Test run_tuner with None args delegates to read_args."""
        with patch.object(self.tuner_mod, "read_args", return_value={}) as mock_read_args:
            with patch.object(self.tuner_mod, "_training_function"):
                self.tuner_mod.run_tuner(None)
                mock_read_args.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
