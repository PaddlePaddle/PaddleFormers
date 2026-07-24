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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest
from unittest.mock import MagicMock, patch


class TestParseArgs(unittest.TestCase):
    """Tests for parse_args in paddleformers.fleet.training.arguments."""

    def test_parse_args_no_config(self):
        from paddleformers.fleet.training.arguments import parse_args

        with patch.object(sys, "argv", ["prog"]):
            args = parse_args()
        self.assertIsNone(args.configs)

    def test_parse_args_ignore_unknown(self):
        from paddleformers.fleet.training.arguments import parse_args

        with patch.object(sys, "argv", ["prog", "--unknown-flag", "value"]):
            args = parse_args(ignore_unknown_args=True)
        # Should not raise, unknown flags ignored
        self.assertIsNotNone(args)

    def test_parse_args_with_extra_args_provider(self):
        from paddleformers.fleet.training.arguments import parse_args

        def extra_args(parser):
            parser.add_argument("--custom-arg", type=int, default=10)
            return parser

        with patch.object(sys, "argv", ["prog", "--custom-arg", "42"]):
            args = parse_args(extra_args_provider=extra_args)
        self.assertEqual(args.custom_arg, 42)

    def test_parse_args_with_extra_args_provider_default(self):
        from paddleformers.fleet.training.arguments import parse_args

        def extra_args(parser):
            parser.add_argument("--my-flag", type=str, default="hello")
            return parser

        with patch.object(sys, "argv", ["prog"]):
            args = parse_args(extra_args_provider=extra_args)
        self.assertEqual(args.my_flag, "hello")

    def test_parse_args_without_ignore_unknown_raises(self):
        from paddleformers.fleet.training.arguments import parse_args

        with (
            patch.object(sys, "argv", ["prog", "--unknown-flag", "value"]),
            self.assertRaises(SystemExit),
        ):
            parse_args(ignore_unknown_args=False)

    def test_parse_args_with_configs_triggers_yaml_load(self):
        from paddleformers.fleet.training.arguments import parse_args

        mock_config = MagicMock()
        mock_config.some_field = "value"

        with patch.object(sys, "argv", ["prog", "--configs", "dummy.yaml"]):
            with patch.dict(
                "sys.modules",
                {
                    "paddleformers.fleet.training.yaml_arguments": MagicMock(
                        load_yaml=MagicMock(return_value=mock_config)
                    )
                },
            ):
                args = parse_args()
        # Should have called load_yaml and returned the mock config
        self.assertEqual(args, mock_config)


class TestCoreTransformerConfigFromArgs(unittest.TestCase):
    """Tests for core_transformer_config_from_args in paddleformers.fleet.training.arguments."""

    def test_from_args_matching_fields(self):
        from paddleformers.fleet.training.arguments import (
            core_transformer_config_from_args,
        )

        args = MagicMock(spec=[])
        args.hidden_size = 256
        args.num_attention_heads = 4
        args.num_hidden_layers = 6

        config = core_transformer_config_from_args(args)
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(config.num_hidden_layers, 6)

    def test_from_args_partial_fields(self):
        from paddleformers.fleet.training.arguments import (
            core_transformer_config_from_args,
        )

        args = MagicMock(spec=[])
        args.hidden_size = 512
        # Missing num_attention_heads -> uses default

        config = core_transformer_config_from_args(args)
        self.assertEqual(config.hidden_size, 512)

    def test_from_args_custom_config_class(self):
        from dataclasses import dataclass

        from paddleformers.fleet.training.arguments import (
            core_transformer_config_from_args,
        )

        @dataclass
        class CustomConfig:
            hidden_size: int = 128
            num_layers: int = 2

        args = MagicMock(spec=[])
        args.hidden_size = 64
        args.num_layers = 8

        config = core_transformer_config_from_args(
            args, config_class=CustomConfig
        )
        self.assertEqual(config.hidden_size, 64)
        self.assertEqual(config.num_layers, 8)

    def test_from_args_missing_fields_use_defaults(self):
        from paddleformers.fleet.training.arguments import (
            core_transformer_config_from_args,
        )

        args = MagicMock(spec=[])
        # No matching attributes - uses all defaults
        config = core_transformer_config_from_args(args)
        self.assertIsNotNone(config)
