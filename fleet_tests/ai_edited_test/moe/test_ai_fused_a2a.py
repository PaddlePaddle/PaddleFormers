# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
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

import paddle


def _make_mock_buffer():
    """Create a mock deep_ep buffer for testing."""
    mock_buffer = MagicMock()
    mock_buffer.get_dispatch_layout.return_value = (
        paddle.ones([4], dtype=paddle.int64),
        paddle.zeros([4], dtype=paddle.int64),
        paddle.ones([2], dtype=paddle.int64),
        paddle.ones([4], dtype=paddle.bool),
        None,
    )
    mock_buffer.dispatch.return_value = (
        paddle.randn([4, 64], dtype=paddle.float32),
        paddle.randint(0, 4, [4, 2]),
        paddle.randn([4, 2], dtype=paddle.float32),
        paddle.to_tensor([2, 2], dtype=paddle.int64),
        MagicMock(),
        MagicMock(),
    )
    mock_buffer.combine.return_value = (
        paddle.randn([4, 64], dtype=paddle.float32),
        MagicMock(),
        MagicMock(),
    )
    return mock_buffer


class TestFusedA2A(unittest.TestCase):
    """Unit tests for fused_a2a module."""

    def test_fused_dispatch_none_when_no_deep_ep(self):
        """Test HAVE_DEEP_EP state reflects deep_ep availability."""
        from paddleformers.fleet.transformer.moe import fused_a2a

        # Just check the module-level variable - it's set at import time
        # and reflects the actual deep_ep availability
        self.assertIsInstance(fused_a2a.HAVE_DEEP_EP, bool)

    def test_get_hidden_bytes(self):
        """Test get_hidden_bytes calculation."""
        from paddleformers.fleet.transformer.moe.fused_a2a import get_hidden_bytes

        x = paddle.randn([4, 64], dtype=paddle.float32)
        # hidden = 64, max(element_size=4, 2) = 4, so 64 * 4 = 256
        result = get_hidden_bytes(x)
        self.assertEqual(result, 256)

    def test_get_hidden_bytes_fp16(self):
        """Test get_hidden_bytes with fp16 tensor."""
        from paddleformers.fleet.transformer.moe.fused_a2a import get_hidden_bytes

        x = paddle.randn([4, 64], dtype=paddle.float16)
        result = get_hidden_bytes(x)
        self.assertEqual(result, 128)

    def test_get_hidden_bytes_fp8(self):
        """Test get_hidden_bytes with fp8 tensor (element_size=1)."""
        from paddleformers.fleet.transformer.moe.fused_a2a import get_hidden_bytes

        x = paddle.zeros([4, 64], dtype=paddle.float8_e4m3fn)
        result = get_hidden_bytes(x)
        # max(element_size=1, 2) = 2, so 64 * 2 = 128
        self.assertEqual(result, 128)

    def test_normalize_fp8_scale_for_deepep_transposes_scale(self):
        """Test FP8 scale layout is transposed to token-major format."""
        from paddleformers.fleet.transformer.moe.fused_a2a import (
            _normalize_fp8_scale_for_deepep,
        )

        x_fp8 = paddle.zeros([4, 256], dtype=paddle.float8_e4m3fn)
        scale = paddle.ones([2, 4], dtype=paddle.float32)
        result = _normalize_fp8_scale_for_deepep(x_fp8, scale)
        self.assertEqual(result.shape, [4, 2])

    def test_normalize_fp8_scale_for_deepep_trims_padded_tokens(self):
        """Test FP8 scale token padding is trimmed for inference dispatch."""
        from paddleformers.fleet.transformer.moe.fused_a2a import (
            _normalize_fp8_scale_for_deepep,
        )

        x_fp8 = paddle.zeros([3, 256], dtype=paddle.float8_e4m3fn)
        scale = paddle.ones([4, 2], dtype=paddle.float32)
        result = _normalize_fp8_scale_for_deepep(x_fp8, scale)
        self.assertEqual(result.shape, [3, 2])

    def test_normalize_fp8_scale_for_deepep_supports_ue8m0(self):
        """Test UE8M0 packed scale width is accepted and transposed."""
        from paddleformers.fleet.transformer.moe.fused_a2a import (
            _normalize_fp8_scale_for_deepep,
        )

        x_fp8 = paddle.zeros([4, 512], dtype=paddle.float8_e4m3fn)
        scale = paddle.ones([1, 4], dtype=paddle.int32)
        result = _normalize_fp8_scale_for_deepep(x_fp8, scale, use_ue8m0=True)
        self.assertEqual(result.shape, [4, 1])

    def test_normalize_fp8_scale_for_deepep_rejects_invalid_shape(self):
        """Test invalid FP8 scale shape fails before DeepEP dispatch."""
        from paddleformers.fleet.transformer.moe.fused_a2a import (
            _normalize_fp8_scale_for_deepep,
        )

        x_fp8 = paddle.zeros([4, 256], dtype=paddle.float8_e4m3fn)
        scale = paddle.ones([4, 3], dtype=paddle.float32)
        with self.assertRaisesRegex(RuntimeError, "Invalid FP8 scale shape"):
            _normalize_fp8_scale_for_deepep(x_fp8, scale)

    def test_dispatch_node_reset_statue(self):
        """Test DispatchNode.reset_statue clears handle."""
        from paddleformers.fleet.transformer.moe.fused_a2a import DispatchNode

        node = DispatchNode()
        node.handle = MagicMock()
        node.reset_statue()
        self.assertIsNone(node.handle)

    def test_combine_node_reset_statue(self):
        """Test CombineNode.reset_statue clears handle."""
        from paddleformers.fleet.transformer.moe.fused_a2a import CombineNode

        node = CombineNode()
        node.handle = MagicMock()
        node.reset_statue()
        self.assertIsNone(node.handle)

    def test_barrier_ep(self):
        """Test barrier_ep calls paddle distributed barrier."""
        from paddleformers.fleet.transformer.moe.fused_a2a import barrier_ep

        mock_group = MagicMock()
        with patch("paddle.distributed.barrier") as mock_barrier:
            barrier_ep(mock_group)
            mock_barrier.assert_called_once_with(mock_group)

    def test_fused_dispatch_forward_func_with_tuple(self):
        """Test fused_dispatch_forward_func with tuple input."""
        from paddleformers.fleet.transformer.moe.fused_a2a import (
            fused_dispatch_forward_func,
        )

        mock_group = MagicMock()
        mock_buffer = _make_mock_buffer()

        with patch(
            "paddleformers.fleet.transformer.moe.fused_a2a.get_buffer",
            return_value=mock_buffer,
        ):
            x = (
                paddle.randn([4, 64], dtype=paddle.float32),
                paddle.ones([1, 1], dtype=paddle.float32),
            )
            token_indices = paddle.randint(0, 4, [4, 2])
            token_probs = paddle.randn([4, 2], dtype=paddle.float32)
            recv_x, recv_probs, states, event = fused_dispatch_forward_func(
                x,
                token_indices,
                token_probs,
                4,
                mock_group,
                moe_ep_barrier=False,
            )
            self.assertIsNotNone(recv_x)
            self.assertIn("dispatched_indices", states)
            self.assertIn("tokens_per_expert", states)
            self.assertIn("handle", states)

    def test_fused_dispatch_forward_func_with_tensor(self):
        """Test fused_dispatch_forward_func with tensor input."""
        from paddleformers.fleet.transformer.moe.fused_a2a import (
            fused_dispatch_forward_func,
        )

        mock_group = MagicMock()
        mock_buffer = _make_mock_buffer()

        with patch(
            "paddleformers.fleet.transformer.moe.fused_a2a.get_buffer",
            return_value=mock_buffer,
        ):
            x = paddle.randn([4, 64], dtype=paddle.float32)
            token_indices = paddle.randint(0, 4, [4, 2])
            token_probs = paddle.randn([4, 2], dtype=paddle.float32)
            recv_x, recv_probs, states, event = fused_dispatch_forward_func(
                x,
                token_indices,
                token_probs,
                4,
                mock_group,
                moe_ep_barrier=False,
            )
            self.assertIsNotNone(recv_x)

    def test_fused_combine_forward_func(self):
        """Test fused_combine_forward_func."""
        from paddleformers.fleet.transformer.moe.fused_a2a import (
            fused_combine_forward_func,
        )

        mock_group = MagicMock()
        mock_buffer = _make_mock_buffer()

        with patch(
            "paddleformers.fleet.transformer.moe.fused_a2a.get_buffer",
            return_value=mock_buffer,
        ):
            x = paddle.randn([4, 64], dtype=paddle.float32)
            states = {"handle": MagicMock()}
            result = fused_combine_forward_func(
                x, mock_group, states, moe_ep_barrier=False
            )
            self.assertIsNotNone(result)

    def test_fused_dispatch_backward_func(self):
        """Test fused_dispatch_backward_func."""
        from paddleformers.fleet.transformer.moe.fused_a2a import (
            fused_dispatch_backward_func,
        )

        mock_group = MagicMock()
        mock_buffer = _make_mock_buffer()

        with patch(
            "paddleformers.fleet.transformer.moe.fused_a2a.get_buffer",
            return_value=mock_buffer,
        ):
            grad_output = paddle.randn([4, 64], dtype=paddle.float32)
            grad_probs = paddle.randn([4, 2], dtype=paddle.float32)
            grad_x, _, grad_probs_out = fused_dispatch_backward_func(
                grad_output,
                grad_probs,
                mock_group,
                MagicMock(),
                moe_ep_barrier=False,
            )
            self.assertIsNotNone(grad_x)
            self.assertIsNotNone(grad_probs_out)

    def test_fused_combine_backward_func_tensor(self):
        """Test fused_combine_backward_func with tensor grad."""
        from paddleformers.fleet.transformer.moe.fused_a2a import (
            fused_combine_backward_func,
        )

        mock_group = MagicMock()
        mock_buffer = _make_mock_buffer()

        with patch(
            "paddleformers.fleet.transformer.moe.fused_a2a.get_buffer",
            return_value=mock_buffer,
        ):
            grad_output = paddle.randn([4, 64], dtype=paddle.float32)
            result = fused_combine_backward_func(
                grad_output,
                mock_group,
                MagicMock(),
                moe_ep_barrier=False,
            )
            self.assertIsNotNone(result)

    def test_fused_combine_backward_func_tuple(self):
        """Test fused_combine_backward_func with tuple grad."""
        from paddleformers.fleet.transformer.moe.fused_a2a import (
            fused_combine_backward_func,
        )

        mock_group = MagicMock()
        mock_buffer = _make_mock_buffer()

        with patch(
            "paddleformers.fleet.transformer.moe.fused_a2a.get_buffer",
            return_value=mock_buffer,
        ):
            grad_output = (
                paddle.randn([4, 64], dtype=paddle.float32),
                paddle.randn([4, 64], dtype=paddle.float32),
            )
            result = fused_combine_backward_func(
                grad_output,
                mock_group,
                MagicMock(),
                moe_ep_barrier=False,
            )
            self.assertIsNotNone(result)

    def test_dispatch_node_forward(self):
        """Test DispatchNode.forward."""
        from paddleformers.fleet.transformer.moe.fused_a2a import DispatchNode

        mock_group = MagicMock()
        mock_buffer = _make_mock_buffer()

        node = DispatchNode()
        with patch(
            "paddleformers.fleet.transformer.moe.fused_a2a.get_buffer",
            return_value=mock_buffer,
        ):
            x = paddle.randn([4, 64], dtype=paddle.float32)
            recv_x, recv_probs, states = node.forward(
                x,
                paddle.randint(0, 4, [4, 2]),
                paddle.randn([4, 2], dtype=paddle.float32),
                4,
                mock_group,
            )
            self.assertIsNotNone(recv_x)

    def test_combine_node_forward(self):
        """Test CombineNode.forward."""
        from paddleformers.fleet.transformer.moe.fused_a2a import CombineNode

        mock_group = MagicMock()
        mock_buffer = _make_mock_buffer()

        node = CombineNode()
        with patch(
            "paddleformers.fleet.transformer.moe.fused_a2a.get_buffer",
            return_value=mock_buffer,
        ):
            x = paddle.randn([4, 64], dtype=paddle.float32)
            result = node.forward(x, mock_group, MagicMock())
            self.assertIsNotNone(result)
