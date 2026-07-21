# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Tests for cp_balance_mode propagation through PaddleFormers."""

import dataclasses
import types
import unittest
from unittest.mock import MagicMock, patch

from paddleformers.trainer import TrainingArguments
from paddleformers.trainer.argparser import PdArgumentParser
from paddleformers.transformers.configuration_utils import LlmMetaConfig


class TestCpBalanceModeTrainingArgs(unittest.TestCase):
    """Test cp_balance_mode field on TrainingArguments."""

    def test_field_exists_with_default(self):
        fields = {f.name: f for f in dataclasses.fields(TrainingArguments)}
        self.assertIn("cp_balance_mode", fields)
        self.assertEqual(fields["cp_balance_mode"].default, "dualchunk_allgather")

    def test_parseable_from_cmdline(self):
        parser = PdArgumentParser((TrainingArguments,))
        ns = parser.parse_args(["--output_dir", "/tmp/test", "--cp_balance_mode", "contiguous_allgather"])
        self.assertEqual(ns.cp_balance_mode, "contiguous_allgather")


class TestCpBalanceModeLlmMetaConfig(unittest.TestCase):
    """Test cp_balance_mode in LlmMetaConfig hybrid_parallel_attributes."""

    def test_in_defaults(self):
        defaults = LlmMetaConfig._get_defaults()
        self.assertIn("cp_balance_mode", defaults)
        self.assertEqual(defaults["cp_balance_mode"], "dualchunk_allgather")

    def test_set_llm_config_propagates(self):
        config = types.SimpleNamespace()
        args = types.SimpleNamespace(cp_balance_mode="contiguous_allgather")
        LlmMetaConfig.set_llm_config(config, args)
        self.assertEqual(config.cp_balance_mode, "contiguous_allgather")

    def test_set_llm_config_default_when_missing(self):
        config = types.SimpleNamespace()
        args = types.SimpleNamespace()  # no cp_balance_mode attr
        LlmMetaConfig.set_llm_config(config, args)
        self.assertEqual(config.cp_balance_mode, "dualchunk_allgather")


class TestGetInputsListCpBalanceMode(unittest.TestCase):
    """Test _get_inputs_list passes cp_balance_mode to get_batch_on_this_cp_rank."""

    def test_passes_contiguous_mode(self):
        mock_get_batch = MagicMock(side_effect=lambda inputs, **kw: inputs)

        with patch("paddleformers.trainer.trainer.is_paddlefleet_available", return_value=True), patch(
            "paddleformers.trainer.trainer.FleetGPTModel", str
        ), patch("paddleformers.trainer.trainer.get_batch_on_this_cp_rank", mock_get_batch):

            from paddleformers.trainer.trainer import Trainer

            args = MagicMock()
            args.enable_auto_parallel = False
            args.use_hybrid_parallel = True
            args.sep_parallel_size = 1
            args.context_parallel_size = 4
            args.split_inputs_sequence_dim = False
            args.ignore_data_skip = False
            args.cp_balance_mode = "contiguous_allgather"

            trainer = object.__new__(Trainer)
            trainer.args = args
            trainer.model = "fake_model"  # isinstance("fake_model", str) == True
            trainer.timers = None

            trainer._get_inputs_list({"input_ids": [1, 2, 3]})

        mock_get_batch.assert_called_once()
        _, kwargs = mock_get_batch.call_args
        self.assertEqual(kwargs["cp_balance_mode"], "contiguous_allgather")


if __name__ == "__main__":
    unittest.main()
