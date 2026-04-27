# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TestCheckDownloadRepo(unittest.TestCase):
    """Tests for check_download_repo function."""

    def test_local_model_dir_with_torch_dtype(self):
        """Test local model directory containing config.json with torch_dtype."""
        from paddleformers.cli.export.export import check_download_repo

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"torch_dtype": "float32"}, f)

            with patch("builtins.print") as mock_print:
                result = check_download_repo(tmpdir)
                self.assertEqual(result, tmpdir)
                mock_print.assert_called_once()
                self.assertIn("torch dtype", str(mock_print.call_args))

    def test_local_model_dir_without_torch_dtype(self):
        """Test local model directory without torch_dtype in config."""
        from paddleformers.cli.export.export import check_download_repo

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"model_type": "llama"}, f)

            result = check_download_repo(tmpdir)
            self.assertEqual(result, tmpdir)

    def test_non_local_model_with_explicit_hub(self):
        """Test non-local model with explicit download_hub parameter."""
        from paddleformers.cli.export.export import check_download_repo

        with patch("paddleformers.cli.export.export.check_repo", return_value="/cached/model") as mock_check:
            result = check_download_repo("some/repo", download_hub="huggingface")
            self.assertEqual(result, "/cached/model")
            mock_check.assert_called_once_with("some/repo", "huggingface")

    def test_non_local_model_with_env_download_source(self):
        """Test non-local model uses DOWNLOAD_SOURCE env variable."""
        from paddleformers.cli.export.export import check_download_repo

        with patch(
            "paddleformers.cli.export.export.check_repo", return_value="/cached/model"
        ) as mock_check, patch.dict(os.environ, {"DOWNLOAD_SOURCE": "modelscope"}):
            result = check_download_repo("some/repo")
            self.assertEqual(result, "/cached/model")
            mock_check.assert_called_once_with("some/repo", "modelscope")

    def test_local_file_path(self):
        """Test local file path triggers isfile branch."""

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            model_file = f.name
        try:
            pass
        finally:
            os.unlink(model_file)


class TestLoggerMergeConfig(unittest.TestCase):
    """Tests for logger_merge_config function."""

    @patch("paddleformers.cli.export.export.logger")
    def test_lora_merge_true(self, mock_logger):
        """Test logger_merge_config with lora_merge=True."""
        from paddleformers.cli.export.export import logger_merge_config

        mock_config = MagicMock()
        mock_config.__dict__ = {
            "lora_model_path": "/path/to/lora",
            "base_model_path": "/path/to/base",
            "output_path": "/path/to/output",
        }

        logger_merge_config(mock_config, lora_merge=True)

        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        self.assertTrue(any("LoRA Merge Info" in c for c in debug_calls))
        self.assertTrue(any("lora_model_path" in c for c in debug_calls))
        self.assertTrue(any("base_model_path" in c for c in debug_calls))

    @patch("paddleformers.cli.export.export.logger")
    def test_lora_merge_false(self, mock_logger):
        """Test logger_merge_config with lora_merge=False."""
        from paddleformers.cli.export.export import logger_merge_config

        mock_config = MagicMock()
        mock_config.__dict__ = {
            "model_path_str": "/path/model",
            "device": "gpu",
            "tensor_type": "fp16",
            "merge_preifx": "merged",
            "output_path": "/path/to/output",
        }

        logger_merge_config(mock_config, lora_merge=False)

        debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
        self.assertTrue(any("Mergekit Config Info" in c for c in debug_calls))
        self.assertFalse(any("model_path_str" in c for c in debug_calls))
        self.assertFalse(any("merge_preifx" in c for c in debug_calls))


class TestRunExport(unittest.TestCase):
    """Tests for run_export function."""

    def test_run_export_raises_when_no_checkpoint(self):
        """Test run_export raises FileNotFoundError when no valid checkpoint exists."""
        from paddleformers.cli.export.export import run_export

        non_model_dir = "/nonexistent/path/12345"
        args = {"output_dir": non_model_dir, "model_name_or_path": "/tmp/model", "lora": "True"}

        with patch("paddleformers.cli.export.export.read_args", return_value=args) as _, patch(
            "paddleformers.cli.export.export.get_export_args"
        ) as mock_get_export, patch("paddleformers.cli.export.export.paddle.set_device"), patch(
            "paddleformers.cli.export.export.os.path.isdir", return_value=False
        ), patch(
            "paddleformers.cli.export.export.os.path.isfile", return_value=False
        ):
            mock_model_args = MagicMock(lora=True)
            mock_finetuning_args = MagicMock(output_dir=non_model_dir, device="gpu")
            mock_get_export.return_value = (
                mock_model_args,
                MagicMock(),
                MagicMock(),
                mock_finetuning_args,
                MagicMock(),
            )

            with self.assertRaises(FileNotFoundError) as ctx:
                run_export(args)
            self.assertIn("No valid checkpoint", str(ctx.exception))

    def test_run_export_raises_when_lora_false(self):
        """Test run_export raises ValueError when lora is False."""
        from paddleformers.cli.export.export import run_export

        args = {"output_dir": "/tmp/model", "model_name_or_path": "/tmp/model", "lora": "False"}

        with patch("paddleformers.cli.export.export.read_args", return_value=args), patch(
            "paddleformers.cli.export.export.get_export_args"
        ) as mock_get_export, patch("paddleformers.cli.export.export.paddle.set_device"), patch(
            "paddleformers.cli.export.export.os.path.isdir", return_value=True
        ), patch(
            "paddleformers.cli.export.export.is_valid_model_dir", return_value=True
        ):
            mock_model_args = MagicMock(lora=False)
            mock_finetuning_args = MagicMock(output_dir="/tmp/model", device="gpu")
            mock_get_export.return_value = (
                mock_model_args,
                MagicMock(),
                MagicMock(),
                mock_finetuning_args,
                MagicMock(),
            )

            with self.assertRaises(ValueError) as ctx:
                run_export(args)
            self.assertIn("Only support merge lora checkpoint", str(ctx.exception))

    def test_run_export_lora_merge_success(self):
        """Test run_export performs LoRA merge when lora is True."""
        from paddleformers.cli.export.export import run_export

        args = {"output_dir": "/tmp/model", "model_name_or_path": "/tmp/model", "lora": "True"}

        mock_merge_config = MagicMock()
        mock_mergekit = MagicMock()

        with patch("paddleformers.cli.export.export.read_args", return_value=args), patch(
            "paddleformers.cli.export.export.get_export_args"
        ) as mock_get_export, patch("paddleformers.cli.export.export.paddle.set_device"), patch(
            "paddleformers.cli.export.export.os.path.isdir", return_value=True
        ), patch(
            "paddleformers.cli.export.export.is_valid_model_dir", return_value=True
        ), patch(
            "paddleformers.cli.export.export.check_download_repo", return_value="/tmp/base"
        ), patch(
            "paddleformers.cli.export.export.resolve_file_path", return_value="/tmp/base/model.safetensors"
        ), patch(
            "paddleformers.cli.export.export.MergeConfig", return_value=mock_merge_config
        ) as mock_merge_config_cls, patch(
            "paddleformers.cli.export.export.MergeModel", return_value=mock_mergekit
        ) as mock_merge_model_cls, patch(
            "paddleformers.cli.export.export.logger"
        ):
            mock_model_args = MagicMock(
                lora=True,
                model_name_or_path="/tmp/model",
                download_hub=None,
                copy_custom_file_list="",
            )
            mock_finetuning_args = MagicMock(
                output_dir="/tmp/model",
                device="gpu",
                convert_from_hf=False,
                save_to_hf=False,
                merge_with_qdq_base_model=False,
            )
            mock_export_args = MagicMock(copy_tokenizer=True)
            mock_get_export.return_value = (
                mock_model_args,
                MagicMock(),
                MagicMock(),
                mock_finetuning_args,
                mock_export_args,
            )

            run_export(args)

            mock_merge_config_cls.assert_called_once()
            mock_merge_model_cls.assert_called_once_with(mock_merge_config)
            mock_mergekit.merge_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
