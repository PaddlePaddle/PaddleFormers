# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import unittest
from unittest.mock import patch


class TestEnvUtils(unittest.TestCase):
    """Tests for utils/env.py"""

    def test_get_user_home(self):
        from paddleformers.utils.env import _get_user_home

        result = _get_user_home()
        self.assertTrue(result.endswith("/") or result == os.path.expanduser("~"))

    def test_get_pf_home_env(self):
        from paddleformers.utils.env import _get_pf_home

        with patch.dict(os.environ, {"PF_HOME": "/tmp/test_pf_home"}):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.isdir", return_value=True):
                    result = _get_pf_home()
        self.assertEqual(result, "/tmp/test_pf_home")

    def test_get_pf_home_env_not_dir(self):
        from paddleformers.utils.env import _get_pf_home

        with patch.dict(os.environ, {"PF_HOME": "/tmp/not_a_dir"}):
            with patch("os.path.exists", return_value=True):
                with patch("os.path.isdir", return_value=False):
                    with self.assertRaises(RuntimeError):
                        _get_pf_home()

    def test_get_pf_home_no_env(self):
        from paddleformers.utils.env import _get_pf_home

        env_copy = os.environ.copy()
        if "PF_HOME" in env_copy:
            del env_copy["PF_HOME"]

        with patch.dict(os.environ, env_copy, clear=True):
            with patch("paddleformers.utils.env._get_user_home", return_value="/tmp/user"):
                result = _get_pf_home()
        self.assertEqual(result, "/tmp/user/.paddleformers")

    def test_get_bool_env_true(self):
        from paddleformers.utils.env import _get_bool_env

        with patch.dict(os.environ, {"TEST_VAR": "true"}):
            self.assertTrue(_get_bool_env("TEST_VAR", "false"))

    def test_get_bool_env_false(self):
        from paddleformers.utils.env import _get_bool_env

        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_get_bool_env("MISSING_VAR", "false"))

    def test_get_bool_env_1(self):
        from paddleformers.utils.env import _get_bool_env

        with patch.dict(os.environ, {"TEST_VAR": "1"}):
            self.assertTrue(_get_bool_env("TEST_VAR", "0"))

    def test_get_sub_home(self):
        from paddleformers.utils.env import _get_sub_home

        with patch("os.path.exists", return_value=True):
            with patch("os.makedirs"):
                result = _get_sub_home("models", parent_home="/tmp/test")
        self.assertEqual(result, "/tmp/test/models")

    def test_checkpoint_regex(self):
        from paddleformers.utils.env import _re_checkpoint

        self.assertIsNotNone(_re_checkpoint.match("checkpoint-1000"))
        self.assertIsNotNone(_re_checkpoint.match("checkpoint-0"))
        self.assertIsNone(_re_checkpoint.match("checkpoint"))
        self.assertIsNone(_re_checkpoint.match("other-100"))

    def test_constants_defined(self):
        from paddleformers.utils.env import (
            CONFIG_NAME,
            FAILED_STATUS,
            MAX_BSZ,
            PADDLE_WEIGHTS_NAME,
            SAFE_WEIGHTS_NAME,
            SUCCESS_STATUS,
        )

        self.assertEqual(CONFIG_NAME, "config.json")
        self.assertEqual(FAILED_STATUS, -1)
        self.assertEqual(SUCCESS_STATUS, 0)
        self.assertEqual(MAX_BSZ, 512)
        self.assertEqual(PADDLE_WEIGHTS_NAME, "model_state.pdparams")
        self.assertEqual(SAFE_WEIGHTS_NAME, "model.safetensors")

    def test_weight_name_constants(self):
        from paddleformers.utils.env import (
            SAFE_MASTER_WEIGHTS_NAME,
            SAFE_OPTIMIZER_NAME,
            SAFE_WEIGHTS_INDEX_NAME,
        )

        self.assertIn(".index.json", SAFE_WEIGHTS_INDEX_NAME)
        self.assertIn(".safetensors", SAFE_OPTIMIZER_NAME)
        self.assertIn(".safetensors", SAFE_MASTER_WEIGHTS_NAME)

    def test_get_bool_env_default_true(self):
        from paddleformers.utils.env import _get_bool_env

        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(_get_bool_env("MISSING_VAR", "true"))

    def test_get_bool_env_case_insensitive(self):
        from paddleformers.utils.env import _get_bool_env

        with patch.dict(os.environ, {"TEST_VAR": "True"}):
            self.assertTrue(_get_bool_env("TEST_VAR", "false"))

        with patch.dict(os.environ, {"TEST_VAR": "FALSE"}):
            self.assertFalse(_get_bool_env("TEST_VAR", "true"))
