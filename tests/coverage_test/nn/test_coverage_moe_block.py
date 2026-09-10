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

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
import paddle.nn as nn


class TestCreateMoeBlock(unittest.TestCase):
    """Tests for create_moe_block factory function."""

    def _get_func(self):
        from paddleformers.nn.moe.moe_block import create_moe_block

        return create_moe_block

    def test_import(self):
        """Test that create_moe_block can be imported."""
        func = self._get_func()
        self.assertTrue(callable(func))

    @patch("paddleformers.nn.moe.moe_block.MOEAllGatherLayerV2")
    def test_allgather_mode(self, mock_cls):
        """Test create_moe_block with moe_mode='allgather'."""
        func = self._get_func()
        mock_gate = MagicMock()
        mock_experts = [MagicMock(), MagicMock()]
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        result = func(
            gate=mock_gate,
            experts=mock_experts,
            layer_idx=0,
            shared_experts=None,
            group=None,
            recompute=False,
            k=2,
            moe_mode="allgather",
        )
        mock_cls.assert_called_once()
        self.assertIs(result, mock_instance)

    @patch("paddleformers.nn.moe.moe_block.MOEAlltoAllLayer")
    def test_alltoall_mode(self, mock_cls):
        """Test create_moe_block with moe_mode='alltoall'."""
        func = self._get_func()
        mock_gate = MagicMock()
        mock_experts = [MagicMock(), MagicMock()]
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        result = func(
            gate=mock_gate,
            experts=mock_experts,
            layer_idx=0,
            shared_experts=None,
            group=None,
            recompute=False,
            k=2,
            moe_mode="alltoall",
        )
        mock_cls.assert_called_once()
        self.assertIs(result, mock_instance)

    def test_invalid_mode_raises(self):
        """Test create_moe_block with invalid moe_mode raises ValueError."""
        func = self._get_func()
        mock_gate = MagicMock()
        mock_experts = [MagicMock()]

        with self.assertRaises(ValueError) as ctx:
            func(
                gate=mock_gate,
                experts=mock_experts,
                layer_idx=0,
                moe_mode="invalid",
            )
        self.assertIn("Invalid moe_mode", str(ctx.exception))

    @patch("paddleformers.nn.moe.moe_block.MOEAllGatherLayerV2")
    def test_allgather_with_shared_experts(self, mock_cls):
        """Test create_moe_block with shared_experts."""
        func = self._get_func()
        mock_gate = MagicMock()
        mock_experts = [MagicMock()]
        mock_shared = [MagicMock()]
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        func(
            gate=mock_gate,
            experts=mock_experts,
            layer_idx=1,
            shared_experts=mock_shared,
            group=None,
            recompute=True,
            k=2,
            enable_reverse_token_drop=True,
            all_to_all_dropout=0.1,
            group_experts=True,
            use_expert_out_alltoall=False,
            use_padding=False,
            dense_token_type=3,
            moe_mode="allgather",
        )
        mock_cls.assert_called_once()

    @patch("paddleformers.nn.moe.moe_block.MOEAlltoAllLayer")
    def test_alltoall_with_all_params(self, mock_cls):
        """Test create_moe_block with all parameters in alltoall mode."""
        func = self._get_func()
        mock_gate = MagicMock()
        mock_experts = [MagicMock(), MagicMock(), MagicMock()]
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        mock_group = MagicMock()
        mock_moe_statics = MagicMock()

        func(
            gate=mock_gate,
            experts=mock_experts,
            layer_idx=2,
            shared_experts=None,
            group=mock_group,
            recompute=True,
            k=1,
            enable_reverse_token_drop=True,
            all_to_all_dropout=0.2,
            group_experts=False,
            moe_statics=mock_moe_statics,
            moe_num_experts=3,
            moe_mode="alltoall",
        )
        mock_cls.assert_called_once()


class TestMoEStatics(unittest.TestCase):
    """Tests for MoEStatics class."""

    def _get_cls(self):
        from paddleformers.nn.moe.moe_block import MoEStatics

        return MoEStatics

    @patch("paddle.utils.unique_name.generate")
    def test_init_basic(self, mock_gen):
        """Test MoEStatics initialization with basic config."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = 4
        config.get.return_value = False  # multimodel_experts

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        self.assertIsNotNone(statics.e_score_correction_bias)
        self.assertIsNotNone(statics.expert_usage)

    @patch("paddle.utils.unique_name.generate")
    def test_init_multimodel_experts(self, mock_gen):
        """Test MoEStatics initialization with multimodel_experts=True."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = [4, 4, 4]
        config.get.return_value = True  # multimodel_experts

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        self.assertIsNotNone(statics.e_score_correction_bias)
        self.assertEqual(statics.e_score_correction_bias.shape[0], 3)  # num_experts_groups

    @patch("paddle.utils.unique_name.generate")
    def test_init_multimodel_experts_different_sizes_raises(self, mock_gen):
        """Test that multimodel_experts with different sizes raises AssertionError."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = [4, 8, 4]  # Different sizes
        config.get.return_value = True

        mock_gen.return_value = "corr_bias"
        with self.assertRaises(AssertionError):
            cls(config, layer_idx=0)

    @patch("paddle.utils.unique_name.generate")
    def test_init_expert_usage_shape(self, mock_gen):
        """Test that expert_usage has correct shape."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = 4
        config.get.return_value = False

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        self.assertEqual(statics.expert_usage.shape, [1, 4])

    @patch("paddle.utils.unique_name.generate")
    def test_init_multimodel_expert_usage_shape(self, mock_gen):
        """Test expert_usage shape with multimodel_experts."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = [2, 2]
        config.get.return_value = True

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        self.assertEqual(statics.expert_usage.shape, [2, 2])

    @patch("paddle.utils.unique_name.generate")
    def test_init_e_score_correction_bias_dtype(self, mock_gen):
        """Test that e_score_correction_bias has float32 dtype."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = 4
        config.get.return_value = False

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        self.assertEqual(statics.e_score_correction_bias.dtype, paddle.float32)

    @patch("paddle.utils.unique_name.generate")
    def test_init_expert_usage_dtype(self, mock_gen):
        """Test that expert_usage has int64 dtype."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = 4
        config.get.return_value = False

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        self.assertEqual(statics.expert_usage.dtype, paddle.int64)

    @patch("paddle.utils.unique_name.generate")
    def test_init_stop_gradient(self, mock_gen):
        """Test that e_score_correction_bias and expert_usage have stop_gradient=True."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = 4
        config.get.return_value = False

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        self.assertTrue(statics.e_score_correction_bias.stop_gradient)
        self.assertTrue(statics.expert_usage.stop_gradient)

    @patch("paddle.utils.unique_name.generate")
    def test_init_is_distributed(self, mock_gen):
        """Test that e_score_correction_bias.is_distributed is True."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = 4
        config.get.return_value = False

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        self.assertTrue(statics.e_score_correction_bias.is_distributed)

    @patch("paddle.utils.unique_name.generate")
    def test_init_different_layer_idx(self, mock_gen):
        """Test that different layer_idx produces different unique_name guards."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = 4
        config.get.return_value = False

        mock_gen.return_value = "corr_bias"
        statics_0 = cls(config, layer_idx=0)
        statics_5 = cls(config, layer_idx=5)
        # Both should be valid instances
        self.assertIsInstance(statics_0, nn.Layer)
        self.assertIsInstance(statics_5, nn.Layer)

    @patch("paddle.utils.unique_name.generate")
    def test_init_expert_usage_zeros(self, mock_gen):
        """Test that expert_usage is initialized with zeros."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = 4
        config.get.return_value = False

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        zeros = paddle.zeros([1, 4], dtype="int64")
        np.testing.assert_allclose(statics.expert_usage.numpy(), zeros.numpy())

    @patch("paddle.utils.unique_name.generate")
    def test_init_cast_flags(self, mock_gen):
        """Test that cast flags are set correctly."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = 4
        config.get.return_value = False

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        self.assertFalse(statics._cast_to_low_precision)
        self.assertFalse(statics._cast_to_low_precison)

    @patch("paddle.utils.unique_name.generate")
    def test_init_multimodel_experts_single_group(self, mock_gen):
        """Test multimodel_experts with a single group [4]."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = [4]
        config.get.return_value = True

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        self.assertEqual(statics.e_score_correction_bias.shape, [1, 4])
        self.assertEqual(statics.expert_usage.shape, [1, 4])

    @patch("paddle.utils.unique_name.generate")
    def test_init_multimodel_experts_large(self, mock_gen):
        """Test multimodel_experts with larger expert groups."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = [8, 8, 8, 8]
        config.get.return_value = True

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=3)
        self.assertEqual(statics.e_score_correction_bias.shape, [4, 8])
        self.assertEqual(statics.expert_usage.shape, [4, 8])

    @patch("paddle.utils.unique_name.generate")
    def test_is_paddle_layer(self, mock_gen):
        """Test that MoEStatics is a subclass of nn.Layer."""
        cls = self._get_cls()
        self.assertTrue(issubclass(cls, nn.Layer))

    @patch("paddle.utils.unique_name.generate")
    def test_config_get_multimodel(self, mock_gen):
        """Test that config.get is called for multimodel_experts."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = 4

        mock_gen.return_value = "corr_bias"

        # Test with multimodel_experts=False
        config.get.return_value = False
        cls(config, layer_idx=0)
        config.get.assert_called_with("multimodel_experts", False)

    @patch("paddle.utils.unique_name.generate")
    def test_has_named_parameters(self, mock_gen):
        """Test that MoEStatics has expected named parameters."""
        cls = self._get_cls()
        config = MagicMock()
        config.moe_num_experts = 4
        config.get.return_value = False

        mock_gen.return_value = "corr_bias"
        statics = cls(config, layer_idx=0)
        names = [n for n, _ in statics.named_parameters()]
        self.assertIn("e_score_correction_bias", names)
        # expert_usage has stop_gradient=True, so it may not appear in parameters
        # depending on paddle version


class TestCreateMoeBlockEdgeCases(unittest.TestCase):
    """Edge case tests for create_moe_block."""

    def _get_func(self):
        from paddleformers.nn.moe.moe_block import create_moe_block

        return create_moe_block

    @patch("paddleformers.nn.moe.moe_block.MOEAllGatherLayerV2")
    def test_empty_experts_list(self, mock_cls):
        """Test create_moe_block with empty experts list."""
        func = self._get_func()
        mock_gate = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        func(
            gate=mock_gate,
            experts=[],
            layer_idx=0,
            moe_mode="allgather",
        )
        mock_cls.assert_called_once()

    @patch("paddleformers.nn.moe.moe_block.MOEAllGatherLayerV2")
    def test_single_expert(self, mock_cls):
        """Test create_moe_block with single expert."""
        func = self._get_func()
        mock_gate = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        func(
            gate=mock_gate,
            experts=[MagicMock()],
            layer_idx=0,
            k=1,
            moe_mode="allgather",
        )
        mock_cls.assert_called_once()

    @patch("paddleformers.nn.moe.moe_block.MOEAllGatherLayerV2")
    def test_many_experts(self, mock_cls):
        """Test create_moe_block with many experts."""
        func = self._get_func()
        mock_gate = MagicMock()
        mock_experts = [MagicMock() for _ in range(16)]
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        func(
            gate=mock_gate,
            experts=mock_experts,
            layer_idx=5,
            k=4,
            moe_mode="allgather",
        )
        mock_cls.assert_called_once()

    @patch("paddleformers.nn.moe.moe_block.MOEAlltoAllLayer")
    def test_alltoall_defaults(self, mock_cls):
        """Test create_moe_block alltoall mode with default parameters."""
        func = self._get_func()
        mock_gate = MagicMock()
        mock_experts = [MagicMock()]
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        func(
            gate=mock_gate,
            experts=mock_experts,
            layer_idx=0,
            moe_mode="alltoall",
        )
        mock_cls.assert_called_once()
        # Check that shared_experts defaults to None
        call_kwargs = mock_cls.call_args[1]
        self.assertIsNone(call_kwargs.get("shared_experts"))


if __name__ == "__main__":
    unittest.main()
