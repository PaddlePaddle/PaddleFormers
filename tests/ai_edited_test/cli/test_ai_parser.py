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


class TestReadArgs(unittest.TestCase):
    """Tests for read_args function."""

    def test_read_args_returns_dict_when_provided(self):
        """Test read_args returns the provided dict directly."""
        from paddleformers.cli.hparams.parser import read_args

        args = {"model_name_or_path": "/tmp/model"}
        result = read_args(args)
        self.assertEqual(result, args)

    def test_read_args_returns_list_when_provided(self):
        """Test read_args returns the provided list directly."""
        from paddleformers.cli.hparams.parser import read_args

        args = ["--model_name_or_path", "/tmp/model"]
        result = read_args(args)
        self.assertEqual(result, args)

    def test_read_args_missing_config_files(self):
        """Test read_args raises AssertionError when sys.argv too short."""
        from paddleformers.cli.hparams.parser import read_args

        with patch("sys.argv", ["prog", "train"]):
            with self.assertRaises(AssertionError):
                read_args()

    def test_read_args_yaml_file(self):
        """Test read_args reads YAML config file."""
        from paddleformers.cli.hparams.parser import read_args

        yaml_content = "model_name_or_path: /tmp/model\nstage: SFT\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with patch("sys.argv", ["prog", "train", yaml_path]):
                result = read_args()
                self.assertIsInstance(result, dict)
                self.assertEqual(result.get("model_name_or_path"), "/tmp/model")
                self.assertEqual(result.get("stage"), "SFT")
        finally:
            os.unlink(yaml_path)

    def test_read_args_yml_file(self):
        """Test read_args reads .yml config file."""
        from paddleformers.cli.hparams.parser import read_args

        yaml_content = "model_name_or_path: /tmp/model\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            yml_path = f.name

        try:
            with patch("sys.argv", ["prog", "train", yml_path]):
                result = read_args()
                self.assertIsInstance(result, dict)
                self.assertEqual(result.get("model_name_or_path"), "/tmp/model")
        finally:
            os.unlink(yml_path)

    def test_read_args_json_file(self):
        """Test read_args reads JSON config file."""
        from paddleformers.cli.hparams.parser import read_args

        config = {"model_name_or_path": "/tmp/model", "stage": "SFT"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            json_path = f.name

        try:
            with patch("sys.argv", ["prog", "train", json_path]):
                result = read_args()
                self.assertIsInstance(result, dict)
                self.assertEqual(result.get("model_name_or_path"), "/tmp/model")
        finally:
            os.unlink(json_path)

    def test_read_args_py_file_raises(self):
        """Test read_args raises ValueError for .py config files."""
        from paddleformers.cli.hparams.parser import read_args

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# config\n")
            py_path = f.name

        try:
            with patch("sys.argv", ["prog", "train", py_path]):
                with self.assertRaises(ValueError) as ctx:
                    read_args()
                self.assertIn("Yaml/Json/Arguments", str(ctx.exception))
        finally:
            os.unlink(py_path)

    def test_read_args_non_config_file_returns_list(self):
        """Test read_args returns remaining argv as list for non-config files."""
        from paddleformers.cli.hparams.parser import read_args

        with patch("sys.argv", ["prog", "train", "--model_name_or_path", "/tmp/model"]):
            result = read_args()
            self.assertIsInstance(result, list)
            self.assertEqual(result, ["--model_name_or_path", "/tmp/model"])

    def test_read_args_yaml_with_override(self):
        """Test read_args merges YAML config with CLI overrides."""
        from paddleformers.cli.hparams.parser import read_args

        yaml_content = "stage: SFT\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with patch("sys.argv", ["prog", "train", yaml_path, "stage=DPO"]):
                result = read_args()
                self.assertIsInstance(result, dict)
                self.assertEqual(result.get("stage"), "DPO")
        finally:
            os.unlink(yaml_path)


class TestLoadCustomTemplate(unittest.TestCase):
    """Tests for _load_custom_template function."""

    @patch("paddleformers.cli.hparams.parser.logger")
    def test_load_custom_template_success(self, mock_logger):
        """Test successful loading of custom template file."""
        from paddleformers.cli.hparams.parser import _load_custom_template

        template_code = "custom_value = 42\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(template_code)
            template_path = f.name

        try:
            _load_custom_template(template_path)
            mock_logger.info.assert_called_once()
            self.assertIn(template_path, str(mock_logger.info.call_args))
        finally:
            os.unlink(template_path)

    def test_load_custom_template_failure(self):
        """Test _load_custom_template raises RuntimeError on failure."""
        from paddleformers.cli.hparams.parser import _load_custom_template

        with self.assertRaises(RuntimeError) as ctx:
            _load_custom_template("/nonexistent/path/template.py")
        self.assertIn("Failed to load", str(ctx.exception))


class TestParseArgs(unittest.TestCase):
    """Tests for _parse_args function."""

    @patch("paddleformers.cli.hparams.parser.is_env_enabled", return_value=False)
    def test_parse_args_dict_with_unknown_keys_raises(self, mock_env):
        """Test _parse_args raises ValueError for unknown keys in dict."""
        from paddleformers.cli.hparams.parser import _parse_args

        mock_parser = MagicMock()
        mock_parser.parse_dict.return_value = ([MagicMock()], {"unknown_key": "value"})

        with self.assertRaises(ValueError) as ctx:
            _parse_args(mock_parser, {"model_name_or_path": "/tmp", "unknown_key": "value"})
        self.assertIn("not used by the PdArgumentParser", str(ctx.exception))

    @patch("paddleformers.cli.hparams.parser.is_env_enabled", return_value=False)
    def test_parse_args_list_with_unknown_keys_raises(self, mock_env):
        """Test _parse_args raises ValueError for unknown args in list."""
        from paddleformers.cli.hparams.parser import _parse_args

        mock_parser = MagicMock()
        mock_parser.parse_args_into_dataclasses.return_value = ([MagicMock()], ["--unknown_arg"])
        mock_parser.format_help.return_value = "help text"

        with self.assertRaises(ValueError) as ctx:
            _parse_args(mock_parser, ["--model_name_or_path", "/tmp", "--unknown_arg"])
        self.assertIn("not used by the PdArgumentParser", str(ctx.exception))

    @patch("paddleformers.cli.hparams.parser.is_env_enabled", return_value=False)
    def test_parse_args_list_allow_extra_keys(self, mock_env):
        """Test _parse_args with allow_extra_keys=True does not raise."""
        from paddleformers.cli.hparams.parser import _parse_args

        mock_parser = MagicMock()
        mock_parsed = MagicMock()
        mock_parser.parse_args_into_dataclasses.return_value = (mock_parsed, ["--unknown_arg"])

        result = _parse_args(mock_parser, ["--model_name_or_path", "/tmp", "--unknown_arg"], allow_extra_keys=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_parsed)


class TestParseTrainArgs(unittest.TestCase):
    """Tests for _parse_train_args function."""

    @patch("paddleformers.cli.hparams.parser._parse_args")
    @patch("paddleformers.cli.hparams.parser.is_env_enabled", return_value=False)
    def test_parse_train_args_calls_parse_args(self, mock_env, mock_parse):
        """Test _parse_train_args delegates to _parse_args."""
        from paddleformers.cli.hparams.parser import _parse_train_args

        _parse_train_args({"output_dir": "/tmp/out"})
        mock_parse.assert_called_once()


class TestParseEvalArgs(unittest.TestCase):
    """Tests for _parse_eval_args function."""

    @patch("paddleformers.cli.hparams.parser._parse_args")
    @patch("paddleformers.cli.hparams.parser.is_env_enabled", return_value=False)
    def test_parse_eval_args_calls_parse_args(self, mock_env, mock_parse):
        """Test _parse_eval_args delegates to _parse_args."""
        from paddleformers.cli.hparams.parser import _parse_eval_args

        _parse_eval_args({"output_dir": "/tmp/out"})
        mock_parse.assert_called_once()


class TestParseServerArgs(unittest.TestCase):
    """Tests for _parse_server_args function."""

    @patch("paddleformers.cli.hparams.parser._parse_args")
    @patch("paddleformers.cli.hparams.parser.is_env_enabled", return_value=False)
    def test_parse_server_args_calls_parse_args(self, mock_env, mock_parse):
        """Test _parse_server_args delegates to _parse_args."""
        from paddleformers.cli.hparams.parser import _parse_server_args

        _parse_server_args({"output_dir": "/tmp/out"})
        mock_parse.assert_called_once()


class TestParseExportArgs(unittest.TestCase):
    """Tests for _parse_export_args function."""

    @patch("paddleformers.cli.hparams.parser._parse_args")
    @patch("paddleformers.cli.hparams.parser.is_env_enabled", return_value=False)
    def test_parse_export_args_calls_parse_args(self, mock_env, mock_parse):
        """Test _parse_export_args delegates to _parse_args."""
        from paddleformers.cli.hparams.parser import _parse_export_args

        _parse_export_args({"output_dir": "/tmp/out"})
        mock_parse.assert_called_once()


if __name__ == "__main__":
    unittest.main()
