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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

try:
    from paddleformers.fleet.transformer.moe.fused_a2a import (
        CombineNode,
        DeepEPCombine,
        DeepEPCombineAsync,
        DeepEPDispatch,
        DispatchNode,
        barrier_ep,
        get_hidden_bytes,
    )

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet.transformer.moe.fused_a2a not available")
class TestBarrierEp(unittest.TestCase):
    """Test barrier_ep function."""

    @patch("paddleformers.fleet.transformer.moe.fused_a2a.paddle.distributed.barrier")
    def test_barrier_calls_paddle_barrier(self, mock_barrier):
        group = MagicMock()
        barrier_ep(group)
        mock_barrier.assert_called_once_with(group)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet.transformer.moe.fused_a2a not available")
class TestGetHiddenBytes(unittest.TestCase):
    """Test get_hidden_bytes function."""

    def test_float32_tensor(self):
        x = paddle.randn([4, 64], dtype="float32")
        # float32: 4 bytes * 64 = 256, max(4, 2) * 64 = 256
        self.assertEqual(get_hidden_bytes(x), 256)

    def test_float16_tensor(self):
        x = paddle.randn([4, 64], dtype="float16")
        # float16: 2 bytes * 64 = 128, max(2, 2) * 64 = 128
        self.assertEqual(get_hidden_bytes(x), 128)

    def test_bfloat16_tensor(self):
        x = paddle.randn([4, 64], dtype="bfloat16")
        # bfloat16: 2 bytes * 64 = 128, max(2, 2) * 64 = 128
        self.assertEqual(get_hidden_bytes(x), 128)

    def test_bool_tensor_min_2(self):
        x = paddle.zeros([4, 64], dtype="bool")
        # bool: 1 byte, but min is 2
        self.assertEqual(get_hidden_bytes(x), 128)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet.transformer.moe.fused_a2a not available")
class TestDeepEPDispatch(unittest.TestCase):
    """Test DeepEPDispatch PyLayer."""

    def test_has_forward_and_backward(self):
        self.assertTrue(hasattr(DeepEPDispatch, "forward"))
        self.assertTrue(hasattr(DeepEPDispatch, "backward"))

    def test_is_pylayer(self):
        self.assertTrue(issubclass(DeepEPDispatch, paddle.autograd.PyLayer))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet.transformer.moe.fused_a2a not available")
class TestDeepEPCombine(unittest.TestCase):
    """Test DeepEPCombine PyLayer."""

    def test_has_forward_and_backward(self):
        self.assertTrue(hasattr(DeepEPCombine, "forward"))
        self.assertTrue(hasattr(DeepEPCombine, "backward"))

    def test_is_pylayer(self):
        self.assertTrue(issubclass(DeepEPCombine, paddle.autograd.PyLayer))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet.transformer.moe.fused_a2a not available")
class TestDeepEPCombineAsync(unittest.TestCase):
    """Test DeepEPCombineAsync PyLayer."""

    def test_has_forward_and_backward(self):
        self.assertTrue(hasattr(DeepEPCombineAsync, "forward"))
        self.assertTrue(hasattr(DeepEPCombineAsync, "backward"))

    def test_is_pylayer(self):
        self.assertTrue(issubclass(DeepEPCombineAsync, paddle.autograd.PyLayer))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet.transformer.moe.fused_a2a not available")
class TestDispatchNode(unittest.TestCase):
    """Test DispatchNode."""

    @patch("paddleformers.fleet.transformer.moe.fused_a2a.fused_dispatch_forward_func")
    def test_construction(self, mock_fwd_func):
        node = DispatchNode()
        self.assertEqual(node.name, "dispatch")
        # DispatchNode does not have a 'handle' attribute by default
        self.assertFalse(hasattr(node, "handle") and node.handle is not None)

    def test_reset_state(self):
        node = DispatchNode()
        # The method name is 'reset_statue' (typo in source)
        node.reset_statue()

    @patch("paddleformers.fleet.transformer.moe.fused_a2a.fused_dispatch_forward_func")
    @patch("paddleformers.fleet.transformer.moe.fused_a2a.fused_dispatch_backward_func")
    def test_forward_sets_handle(self, mock_bwd, mock_fwd):
        mock_fwd.return_value = (
            paddle.randn([4, 64]),
            paddle.randn([4, 2]),
            {
                "handle": MagicMock(),
                "dispatched_indices": MagicMock(),
                "tokens_per_expert": [2, 2],
            },
            None,
        )
        node = DispatchNode()
        group = MagicMock()
        x = paddle.randn([4, 64], dtype="float32")
        indices = paddle.to_tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
        probs = paddle.ones([4, 2], dtype="float32")
        recv_x, recv_probs, states = node.forward(x, indices, probs, 4, group)
        self.assertIsNotNone(node.handle)
        self.assertIsNotNone(node.group)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet.transformer.moe.fused_a2a not available")
class TestCombineNode(unittest.TestCase):
    """Test CombineNode."""

    def test_construction(self):
        node = CombineNode()
        self.assertEqual(node.name, "combine")
        # CombineNode does not have a 'handle' attribute by default
        self.assertFalse(hasattr(node, "handle") and node.handle is not None)

    def test_reset_state(self):
        node = CombineNode()
        # The method name is 'reset_statue' (typo in source)
        node.reset_statue()


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet.transformer.moe.fused_a2a not available")
class TestFusedDispatchForwardFunc(unittest.TestCase):
    """Test fused_dispatch_forward_func."""

    @patch("paddleformers.fleet.transformer.moe.fused_a2a.barrier_ep")
    @patch("paddleformers.fleet.transformer.moe.fused_a2a.get_buffer")
    def test_barrier_called(self, mock_buffer, mock_barrier):
        mock_buf = MagicMock()
        mock_buf.get_dispatch_layout.return_value = (
            paddle.to_tensor([4]),
            paddle.to_tensor([4]),
            paddle.to_tensor([2, 2]),
            paddle.to_tensor([1, 1]),
            None,
        )
        mock_buf.dispatch.return_value = (
            paddle.randn([4, 64]),
            paddle.randn([4, 2]),
            [2, 2],
            MagicMock(),
            None,
            None,
        )
        mock_buffer.return_value = mock_buf

        group = MagicMock()
        x = paddle.randn([4, 64], dtype="float32")
        indices = paddle.to_tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
        probs = paddle.ones([4, 2], dtype="float32")

        from paddleformers.fleet.transformer.moe.fused_a2a import (
            fused_dispatch_forward_func,
        )

        recv_x, recv_probs, states, event = fused_dispatch_forward_func(
            x,
            indices,
            probs,
            2,
            group,
            moe_ep_barrier=True,
        )
        mock_barrier.assert_called_once_with(group)
        self.assertIn("handle", states)
        self.assertIn("dispatched_indices", states)
        self.assertIn("tokens_per_expert", states)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddleformers.fleet.transformer.moe.fused_a2a not available")
class TestFusedCombineForwardFunc(unittest.TestCase):
    """Test fused_combine_forward_func."""

    @patch("paddleformers.fleet.transformer.moe.fused_a2a.barrier_ep")
    @patch("paddleformers.fleet.transformer.moe.fused_a2a.get_buffer")
    def test_barrier_called(self, mock_buffer, mock_barrier):
        mock_buf = MagicMock()
        mock_buf.combine.return_value = (paddle.randn([4, 64]), None, None)
        mock_buffer.return_value = mock_buf

        group = MagicMock()
        states = {"handle": MagicMock()}
        x = paddle.randn([4, 64], dtype="float32")

        from paddleformers.fleet.transformer.moe.fused_a2a import (
            fused_combine_forward_func,
        )

        result = fused_combine_forward_func(x, group, states, moe_ep_barrier=True)
        mock_barrier.assert_called_once_with(group)
        self.assertEqual(result.shape, [4, 64])


if __name__ == "__main__":
    unittest.main()
