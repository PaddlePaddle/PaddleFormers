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


# Tests for src/paddleformers.fleet/training/initialize.py
# Test initialize_fleet, set_logging

import logging
import unittest
from unittest import mock


class TestSetLogging(unittest.TestCase):
    """Tests for set_logging function."""

    def tearDown(self):
        # Remove any mock handlers added by set_logging to avoid test pollution.
        logger = logging.getLogger("paddleformers.fleet")
        logger.handlers = [
            h for h in logger.handlers if isinstance(h, logging.Handler)
        ]

    def test_set_logging_default_level(self):
        """Test set_logging with default logging level."""
        from paddleformers.fleet.training.initialize import set_logging

        mock_args = mock.MagicMock(spec=["logging_level"])
        mock_args.logging_level = logging.INFO

        mock_handler = mock.MagicMock()
        with mock.patch("colorlog.StreamHandler", return_value=mock_handler):
            with mock.patch("colorlog.ColoredFormatter"):
                set_logging(mock_args)
                mock_handler.setFormatter.assert_called_once()

    def test_set_logging_debug_level(self):
        """Test set_logging with DEBUG level."""
        from paddleformers.fleet.training.initialize import set_logging

        mock_args = mock.MagicMock()
        mock_args.logging_level = logging.DEBUG

        mock_handler = mock.MagicMock()
        with mock.patch("colorlog.StreamHandler", return_value=mock_handler):
            with mock.patch("colorlog.ColoredFormatter"):
                set_logging(mock_args)
                mock_handler.setFormatter.assert_called_once()

    def test_set_logging_warning_level(self):
        """Test set_logging with WARNING level."""
        from paddleformers.fleet.training.initialize import set_logging

        mock_args = mock.MagicMock()
        mock_args.logging_level = logging.WARNING

        mock_handler = mock.MagicMock()
        with mock.patch("colorlog.StreamHandler", return_value=mock_handler):
            with mock.patch("colorlog.ColoredFormatter"):
                set_logging(mock_args)
                mock_handler.setFormatter.assert_called_once()

    def test_set_logging_no_logging_level_attr(self):
        """Test set_logging when args has no logging_level attribute."""
        from paddleformers.fleet.training.initialize import set_logging

        mock_args = mock.MagicMock()
        # Remove logging_level attribute
        del mock_args.logging_level

        mock_handler = mock.MagicMock()
        with mock.patch("colorlog.StreamHandler", return_value=mock_handler):
            with mock.patch("colorlog.ColoredFormatter"):
                set_logging(mock_args)
                # Should use default logging.INFO
                logger = logging.getLogger("paddleformers.fleet")
                self.assertEqual(logger.level, logging.INFO)

    def test_set_logging_adds_handler(self):
        """Test set_logging adds a handler to the logger."""
        from paddleformers.fleet.training.initialize import set_logging

        mock_args = mock.MagicMock()
        mock_args.logging_level = logging.INFO

        mock_handler = mock.MagicMock()
        with mock.patch("colorlog.StreamHandler", return_value=mock_handler):
            with mock.patch("colorlog.ColoredFormatter"):
                set_logging(mock_args)
                logger = logging.getLogger("paddleformers.fleet")
                self.assertIn(mock_handler, logger.handlers)


class TestInitializeFleet(unittest.TestCase):
    """Tests for initialize_fleet function."""

    def test_initialize_fleet_with_parsed_args(self):
        """Test initialize_fleet with pre-parsed args."""
        from paddleformers.fleet.training.initialize import initialize_fleet

        mock_args = mock.MagicMock()
        mock_strategy = mock.MagicMock()

        with mock.patch(
            "paddleformers.fleet.training.initialize.set_global_variables"
        ) as mock_sg:
            with mock.patch(
                "paddleformers.fleet.training.initialize.set_logging"
            ) as mock_sl:
                with mock.patch("paddle.distributed.fleet.init"):
                    with mock.patch(
                        "paddle.distributed.fleet.get_hybrid_communicate_group"
                    ) as mock_hcg:
                        with mock.patch(
                            "paddle.distributed.get_rank", return_value=0
                        ):
                            with mock.patch(
                                "paddle.distributed.get_world_size",
                                return_value=1,
                            ):
                                with mock.patch(
                                    "paddleformers.fleet.training.initialize.ps.initialize_model_parallel"
                                ):
                                    initialize_fleet(
                                        mock_strategy, parsed_args=mock_args
                                    )
                                    mock_sg.assert_called_once_with(mock_args)
                                    mock_sl.assert_called_once_with(mock_args)

    def test_initialize_fleet_without_parsed_args(self):
        """Test initialize_fleet parses args when None."""
        from paddleformers.fleet.training.initialize import initialize_fleet

        mock_args = mock.MagicMock()
        mock_strategy = mock.MagicMock()

        with mock.patch(
            "paddleformers.fleet.training.initialize.parse_args",
            return_value=mock_args,
        ) as mock_parse:
            with mock.patch(
                "paddleformers.fleet.training.initialize.set_global_variables"
            ):
                with mock.patch(
                    "paddleformers.fleet.training.initialize.set_logging"
                ):
                    with mock.patch("paddle.distributed.fleet.init"):
                        with mock.patch(
                            "paddle.distributed.fleet.get_hybrid_communicate_group"
                        ):
                            with mock.patch(
                                "paddle.distributed.get_rank", return_value=0
                            ):
                                with mock.patch(
                                    "paddle.distributed.get_world_size",
                                    return_value=1,
                                ):
                                    with mock.patch(
                                        "paddleformers.fleet.training.initialize.ps.initialize_model_parallel"
                                    ):
                                        initialize_fleet(mock_strategy)
                                        mock_parse.assert_called_once_with(
                                            ignore_unknown_args=True
                                        )

    def test_initialize_fleet_kwargs(self):
        """Test initialize_fleet passes kwargs to parse_args."""
        from paddleformers.fleet.training.initialize import initialize_fleet

        mock_args = mock.MagicMock()
        mock_strategy = mock.MagicMock()

        with mock.patch(
            "paddleformers.fleet.training.initialize.parse_args",
            return_value=mock_args,
        ) as mock_parse:
            with mock.patch(
                "paddleformers.fleet.training.initialize.set_global_variables"
            ):
                with mock.patch(
                    "paddleformers.fleet.training.initialize.set_logging"
                ):
                    with mock.patch("paddle.distributed.fleet.init"):
                        with mock.patch(
                            "paddle.distributed.fleet.get_hybrid_communicate_group"
                        ):
                            with mock.patch(
                                "paddle.distributed.get_rank", return_value=0
                            ):
                                with mock.patch(
                                    "paddle.distributed.get_world_size",
                                    return_value=1,
                                ):
                                    with mock.patch(
                                        "paddleformers.fleet.training.initialize.ps.initialize_model_parallel"
                                    ):
                                        initialize_fleet(
                                            mock_strategy, some_arg=42
                                        )
                                        mock_parse.assert_called_once()
                                        _, kwargs = mock_parse.call_args
                                        self.assertEqual(kwargs["some_arg"], 42)

    def test_initialize_fleet_calls_fleet_init(self):
        """Test initialize_fleet calls fleet.init with is_collective=True."""
        from paddleformers.fleet.training.initialize import initialize_fleet

        mock_args = mock.MagicMock()
        mock_strategy = mock.MagicMock()

        with mock.patch(
            "paddleformers.fleet.training.initialize.set_global_variables"
        ):
            with mock.patch(
                "paddleformers.fleet.training.initialize.set_logging"
            ):
                with mock.patch("paddle.distributed.fleet.init") as mock_init:
                    with mock.patch(
                        "paddle.distributed.fleet.get_hybrid_communicate_group"
                    ):
                        with mock.patch(
                            "paddle.distributed.get_rank", return_value=0
                        ):
                            with mock.patch(
                                "paddle.distributed.get_world_size",
                                return_value=1,
                            ):
                                with mock.patch(
                                    "paddleformers.fleet.training.initialize.ps.initialize_model_parallel"
                                ):
                                    initialize_fleet(
                                        mock_strategy, parsed_args=mock_args
                                    )
                                    mock_init.assert_called_once_with(
                                        is_collective=True,
                                        strategy=mock_strategy,
                                    )


class TestInitializeFleetParallelState(unittest.TestCase):
    """Tests for parallel state initialization in initialize_fleet."""

    def test_calls_initialize_model_parallel(self):
        """Test initialize_fleet calls ps.initialize_model_parallel."""
        from paddleformers.fleet.training.initialize import initialize_fleet

        mock_args = mock.MagicMock()
        mock_strategy = mock.MagicMock()
        mock_hcg = mock.MagicMock()

        with mock.patch(
            "paddleformers.fleet.training.initialize.set_global_variables"
        ):
            with mock.patch(
                "paddleformers.fleet.training.initialize.set_logging"
            ):
                with mock.patch("paddle.distributed.fleet.init"):
                    with mock.patch(
                        "paddle.distributed.fleet.get_hybrid_communicate_group",
                        return_value=mock_hcg,
                    ):
                        with mock.patch(
                            "paddle.distributed.get_rank", return_value=0
                        ):
                            with mock.patch(
                                "paddle.distributed.get_world_size",
                                return_value=2,
                            ):
                                with mock.patch(
                                    "paddleformers.fleet.training.initialize.ps.initialize_model_parallel"
                                ) as mock_imp:
                                    initialize_fleet(
                                        mock_strategy, parsed_args=mock_args
                                    )
                                    mock_imp.assert_called_once_with(mock_hcg)


class TestInitializeModuleStructure(unittest.TestCase):
    """Tests for module structure."""

    def test_module_exports(self):
        """Test that expected functions are exported."""
        import paddleformers.fleet.training.initialize as init_mod

        self.assertTrue(hasattr(init_mod, "initialize_fleet"))
        self.assertTrue(hasattr(init_mod, "set_logging"))
        self.assertTrue(callable(init_mod.initialize_fleet))
        self.assertTrue(callable(init_mod.set_logging))


if __name__ == "__main__":
    unittest.main()
