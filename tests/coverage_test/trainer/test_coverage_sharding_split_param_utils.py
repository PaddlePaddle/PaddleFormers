# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest.mock import MagicMock, patch

import paddle


class TestShardingSplitParamUtils(unittest.TestCase):
    """Tests for trainer/unified_checkpoint/sharding_split_param_utils.py"""

    def test_get_params_info_empty(self):
        from paddleformers.trainer.unified_checkpoint.sharding_split_param_utils import (
            get_params_info,
        )

        result = get_params_info([])
        expected_keys, param_slice_info, param_shape_info = result
        self.assertEqual(expected_keys, [])
        self.assertEqual(param_slice_info, {})
        self.assertEqual(param_shape_info, {})

    def test_reshape_params_empty(self):
        pass

        from paddleformers.trainer.unified_checkpoint.sharding_split_param_utils import (
            reshape_params,
        )

        state_dict = {}
        struct2static = {}
        param_shape_info = {}
        param_slice_info = {}

        result = reshape_params(state_dict, struct2static, param_shape_info, param_slice_info)
        self.assertEqual(result, {})

    def test_merge_splited_param_no_partial(self):
        import paddle

        from paddleformers.trainer.unified_checkpoint.sharding_split_param_utils import (
            merge_splited_param,
        )

        state_dict = {
            "param0.moment1_0": paddle.randn([8]),
            "beta1_pow_acc_0": paddle.to_tensor([1.0]),
        }
        partial_tensor_list = []
        param_shape_info = {"param0": (paddle.to_tensor([8]), 8, 0, 8)}
        send_table = {}
        recv_table = {}

        with patch(
            "paddleformers.trainer.unified_checkpoint.sharding_split_param_utils.dist.get_rank", return_value=0
        ):
            result = merge_splited_param(state_dict, partial_tensor_list, param_shape_info, send_table, recv_table)
        self.assertIn("param0.moment1_0", result)
        self.assertEqual(result["param0.moment1_0"].shape, [8])

    def test_merge_splited_param_single_element(self):
        import paddle

        from paddleformers.trainer.unified_checkpoint.sharding_split_param_utils import (
            merge_splited_param,
        )

        state_dict = {
            "beta1_pow_acc_0": paddle.to_tensor([1.0]),
        }
        partial_tensor_list = []
        param_shape_info = {}
        send_table = {}
        recv_table = {}

        with patch(
            "paddleformers.trainer.unified_checkpoint.sharding_split_param_utils.dist.get_rank", return_value=0
        ):
            result = merge_splited_param(state_dict, partial_tensor_list, param_shape_info, send_table, recv_table)
        # beta1_pow_acc_0 has numel==1 so should be skipped
        self.assertIn("beta1_pow_acc_0", result)

    def test_get_params_info_with_buffers(self):
        from paddleformers.trainer.unified_checkpoint.sharding_split_param_utils import (
            get_params_info,
        )

        mock_buffer = MagicMock()
        mock_view = MagicMock()
        mock_view._param_begin = 0
        mock_view._param_end = 8
        mock_view._param = MagicMock()
        mock_view._param.shape = [8]
        mock_view._param.numel.return_value = paddle.to_tensor(8)
        mock_view._index = 0
        mock_view._padded_size = 8
        mock_buffer._sharding_param_grad_view = {"param0": mock_view}

        result = get_params_info([mock_buffer])
        expected_keys, param_slice_info, param_shape_info = result
        self.assertEqual(expected_keys, ["param0"])
        self.assertIn("param0", param_slice_info)
        self.assertIn("param0", param_shape_info)
