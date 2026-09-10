# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest.mock import MagicMock, patch


class TestAsyncHandler(unittest.TestCase):
    """Tests for trainer/unified_checkpoint/async_handler.py"""

    def test_AsyncCheckpointHandler_init_no_async(self):
        from paddleformers.trainer.unified_checkpoint.async_handler import (
            AsyncCheckpointHandler,
        )

        args = MagicMock()
        args.unified_checkpoint_config = {}

        with patch(
            "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_rank", return_value=0
        ):
            with patch(
                "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_world_size",
                return_value=1,
            ):
                handler = AsyncCheckpointHandler(args)

        self.assertEqual(handler.global_rank, -1)
        self.assertIsNone(handler._shm_model_weight)
        self.assertIsNone(handler._lock)

    def test_AsyncCheckpointHandler_init_with_async(self):
        from paddleformers.trainer.unified_checkpoint.async_handler import (
            AsyncCheckpointHandler,
        )

        args = MagicMock()
        args.unified_checkpoint_config = {"async_save": True}

        with patch(
            "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_rank", return_value=0
        ):
            with patch(
                "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_world_size",
                return_value=1,
            ):
                with patch("paddleformers.trainer.unified_checkpoint.async_handler.multiprocessing.Lock"):
                    with patch("paddleformers.trainer.unified_checkpoint.async_handler.multiprocessing.Array"):
                        handler = AsyncCheckpointHandler(args)

        self.assertIsNotNone(handler._lock)
        self.assertIsNotNone(handler._shared_save_model_flag)
        self.assertIsNotNone(handler._shared_save_master_weight_flag)
        self.assertIsNotNone(handler._shared_save_optimizer_flag)

    def test_AsyncCheckpointHandler_init_multi_rank(self):
        from paddleformers.trainer.unified_checkpoint.async_handler import (
            AsyncCheckpointHandler,
        )

        args = MagicMock()
        args.unified_checkpoint_config = {}

        with patch(
            "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_rank", return_value=3
        ):
            with patch(
                "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_world_size",
                return_value=4,
            ):
                handler = AsyncCheckpointHandler(args)

        self.assertEqual(handler.global_rank, 3)

    def test_AsyncCheckpointHandler_sync_save(self):
        from paddleformers.trainer.unified_checkpoint.async_handler import (
            AsyncCheckpointHandler,
        )

        args = MagicMock()
        args.unified_checkpoint_config = {}

        with patch(
            "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_rank", return_value=0
        ):
            with patch(
                "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_world_size",
                return_value=1,
            ):
                handler = AsyncCheckpointHandler(args)

        state_dict = {"weight": MagicMock()}
        mock_metadata = {"format": "np"}

        with patch(
            "paddleformers.trainer.unified_checkpoint.async_handler.prepare_safe_save_state_dict",
            return_value=(state_dict, mock_metadata),
        ):
            with patch("paddleformers.trainer.unified_checkpoint.async_handler.safe_save_file"):
                handler._file_save_async_or_sync(
                    state_dict, "/tmp/path.safetensors", is_sync=True, state_dict_type="model_weight"
                )

    def test_AsyncCheckpointHandler_sync_save_with_quant(self):
        from paddleformers.trainer.unified_checkpoint.async_handler import (
            AsyncCheckpointHandler,
        )

        args = MagicMock()
        args.unified_checkpoint_config = {}

        with patch(
            "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_rank", return_value=0
        ):
            with patch(
                "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_world_size",
                return_value=1,
            ):
                handler = AsyncCheckpointHandler(args)

        state_dict = {"weight": MagicMock()}
        mock_metadata = {"format": "np"}

        with patch(
            "paddleformers.trainer.unified_checkpoint.async_handler.prepare_safe_save_state_dict",
            return_value=(state_dict, mock_metadata),
        ):
            with patch(
                "paddleformers.trainer.unified_checkpoint.async_handler.quant_unified_optimizer",
                return_value=state_dict,
            ) as mock_quant:
                with patch("paddleformers.trainer.unified_checkpoint.async_handler.safe_save_file"):
                    handler._file_save_async_or_sync(
                        state_dict,
                        "/tmp/path.safetensors",
                        is_sync=True,
                        state_dict_type="optimizer_weight",
                        ckpt_quant_stage="O1",
                    )
        mock_quant.assert_called_once()

    def test_AsyncCheckpointHandler_unlink_no_async(self):
        from paddleformers.trainer.unified_checkpoint.async_handler import (
            AsyncCheckpointHandler,
        )

        args = MagicMock()
        args.unified_checkpoint_config = {}

        with patch(
            "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_rank", return_value=0
        ):
            with patch(
                "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_world_size",
                return_value=1,
            ):
                handler = AsyncCheckpointHandler(args)

        # Should return early when async_save not in config
        handler.unlink_shared_memory()
        self.assertIsNone(handler._shm_model_weight)

    def test_AsyncCheckpointHandler_empty_state_dict_async(self):
        from paddleformers.trainer.unified_checkpoint.async_handler import (
            AsyncCheckpointHandler,
        )

        args = MagicMock()
        args.unified_checkpoint_config = {"async_save": True}

        with patch(
            "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_rank", return_value=0
        ):
            with patch(
                "paddleformers.trainer.unified_checkpoint.async_handler.paddle.distributed.get_world_size",
                return_value=1,
            ):
                with patch("paddleformers.trainer.unified_checkpoint.async_handler.multiprocessing.Lock"):
                    with patch("paddleformers.trainer.unified_checkpoint.async_handler.multiprocessing.Array"):
                        with patch("paddleformers.trainer.unified_checkpoint.async_handler.paddle.save"):
                            handler = AsyncCheckpointHandler(args)

                            handler._file_save_async_or_sync(
                                {},
                                "/tmp/path",
                                signal_path="/tmp/signal",
                                is_sync=False,
                                state_dict_type="model_weight",
                            )
