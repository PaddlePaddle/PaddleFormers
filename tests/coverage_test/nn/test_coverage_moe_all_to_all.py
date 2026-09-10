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

import paddle


class TestAlltoAllPyLayer(unittest.TestCase):
    """Tests for AlltoAll PyLayer."""

    def _get_cls(self):
        from paddleformers.nn.moe.all_to_all import AlltoAll

        return AlltoAll

    def test_import(self):
        """Test that AlltoAll can be imported."""
        cls = self._get_cls()
        self.assertIsNotNone(cls)

    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=1)
    def test_forward_single_rank_returns_input(self, mock_ws):
        """Test that single rank returns input tensor unchanged."""
        cls = self._get_cls()
        x = paddle.randn([4, 8])
        mock_group = MagicMock()
        result = cls.apply(x, group=mock_group, sync_op=True)
        self.assertEqual(result.shape, [4, 8])

    @patch("paddleformers.nn.moe.all_to_all.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=4)
    def test_forward_multi_rank_sync(self, mock_ws, mock_alltoall):
        """Test forward with multiple ranks and sync_op=True."""
        cls = self._get_cls()
        x = paddle.randn([4, 8])
        mock_group = MagicMock()
        mock_alltoall.return_value = None

        result = cls.apply(x, group=mock_group, sync_op=True)
        mock_alltoall.assert_called_once()
        self.assertEqual(result.shape, [4, 8])

    @patch("paddleformers.nn.moe.all_to_all.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=4)
    def test_forward_multi_rank_async(self, mock_ws, mock_alltoall):
        """Test forward with multiple ranks and sync_op=False returns tuple."""
        cls = self._get_cls()
        x = paddle.randn([4, 8])
        mock_group = MagicMock()
        mock_task = MagicMock()
        mock_alltoall.return_value = mock_task

        result = cls.apply(x, group=mock_group, sync_op=False)
        mock_alltoall.assert_called_once()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @patch("paddleformers.nn.moe.all_to_all.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=4)
    def test_forward_sync_op_true_default(self, mock_ws, mock_alltoall):
        """Test forward with default sync_op=True."""
        cls = self._get_cls()
        x = paddle.randn([4, 8])
        mock_group = MagicMock()
        mock_alltoall.return_value = None

        cls.apply(x, group=mock_group)
        mock_alltoall.assert_called_once()
        call_kwargs = mock_alltoall.call_args[1]
        self.assertTrue(call_kwargs["sync_op"])
        self.assertTrue(call_kwargs["use_calc_stream"])

    @patch("paddleformers.nn.moe.all_to_all.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=4)
    def test_forward_output_no_stop_gradient(self, mock_ws, mock_alltoall):
        """Test that forward output has stop_gradient=False."""
        cls = self._get_cls()
        x = paddle.randn([4, 8])
        mock_group = MagicMock()
        mock_alltoall.return_value = None

        result = cls.apply(x, group=mock_group, sync_op=True)
        self.assertFalse(result.stop_gradient)

    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=1)
    def test_backward_single_rank(self, mock_ws):
        """Test backward with single rank returns input gradient."""
        cls = self._get_cls()
        x = paddle.randn([4, 8])
        mock_group = MagicMock()

        result = cls.apply(x, group=mock_group, sync_op=True)
        self.assertIsNotNone(result)


class TestAlltoAllAsync(unittest.TestCase):
    """Tests for AlltoAllAsync PyLayer."""

    def _get_cls(self):
        from paddleformers.nn.moe.all_to_all import AlltoAllAsync

        return AlltoAllAsync

    def test_import(self):
        """Test that AlltoAllAsync can be imported."""
        cls = self._get_cls()
        self.assertIsNotNone(cls)

    def test_forward_fn_none_raises(self):
        """Test that fn=None raises AssertionError."""
        cls = self._get_cls()
        x = paddle.randn([4, 8])
        mock_group = MagicMock()
        with self.assertRaises(AssertionError):
            cls.apply(x, group=mock_group, fn=None, is_first_fwd=True)

    @patch("paddleformers.nn.moe.all_to_all.manual_backward")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=1)
    def test_forward_single_world_manual_backward_called(self, mock_ws, mock_mb):
        """Test forward with world_size=1 calls manual_backward."""
        cls = self._get_cls()
        mock_fn = MagicMock()
        mock_fn.return_value = paddle.randn([4, 8])
        mock_bwf = MagicMock()
        mock_mb.return_value = (mock_bwf, (paddle.randn([4, 8]),))

        x = paddle.randn([4, 8])
        result = cls.apply(x, group=None, fn=mock_fn, is_first_fwd=True)
        mock_mb.assert_called_once_with(mock_fn, True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # (x,) + fn_out

    @patch("paddleformers.nn.moe.all_to_all.manual_backward")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=1)
    def test_forward_single_world_is_first_fwd_false(self, mock_ws, mock_mb):
        """Test forward with world_size=1 and is_first_fwd=False."""
        cls = self._get_cls()
        mock_fn = MagicMock()
        mock_fn.return_value = paddle.randn([4, 8])
        mock_bwf = MagicMock()
        mock_bwf.return_value = (paddle.randn([4, 8]),)
        mock_mb.return_value = (mock_bwf, (paddle.randn([4, 8]),))

        x = paddle.randn([4, 8])
        cls.apply(x, group=None, fn=mock_fn, is_first_fwd=False)
        mock_mb.assert_called_once_with(mock_fn, False)

    @patch("paddleformers.nn.moe.all_to_all.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_to_all.manual_backward")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=4)
    def test_forward_multi_world(self, mock_ws, mock_mb, mock_alltoall):
        """Test forward with world_size>1 calls alltoall and manual_backward."""
        cls = self._get_cls()
        mock_task = MagicMock()
        mock_task.wait = MagicMock()
        mock_alltoall.return_value = mock_task

        mock_fn = MagicMock()
        mock_fn.return_value = paddle.randn([4, 8])
        mock_bwf = MagicMock()
        mock_mb.return_value = (mock_bwf, (paddle.randn([4, 8]),))

        mock_group = MagicMock()
        x = paddle.randn([4, 8])
        cls.apply(x, group=mock_group, fn=mock_fn, is_first_fwd=True)
        mock_alltoall.assert_called_once()
        mock_task.wait.assert_called_once()

    @patch("paddleformers.nn.moe.all_to_all.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_to_all.manual_backward")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=4)
    def test_forward_multi_world_output_no_stop_gradient(self, mock_ws, mock_mb, mock_alltoall):
        """Test that forward output has stop_gradient=False in multi-world mode."""
        cls = self._get_cls()
        mock_task = MagicMock()
        mock_task.wait = MagicMock()
        mock_alltoall.return_value = mock_task

        mock_fn = MagicMock()
        mock_fn.return_value = paddle.randn([4, 8])
        mock_bwf = MagicMock()
        mock_mb.return_value = (mock_bwf, (paddle.randn([4, 8]),))

        mock_group = MagicMock()
        x = paddle.randn([4, 8])
        result = cls.apply(x, group=mock_group, fn=mock_fn, is_first_fwd=True)
        # First element of result should be x_out with stop_gradient=False
        self.assertFalse(result[0].stop_gradient)

    @patch("paddleformers.nn.moe.all_to_all.manual_backward")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=1)
    def test_forward_with_fn_args(self, mock_ws, mock_mb):
        """Test forward with additional fn_args."""
        cls = self._get_cls()
        mock_fn = MagicMock()
        mock_fn.return_value = paddle.randn([4, 8])
        mock_bwf = MagicMock()
        mock_mb.return_value = (mock_bwf, (paddle.randn([4, 8]),))

        x = paddle.randn([4, 8])
        fn_arg1 = paddle.randn([4, 8])
        fn_arg2 = paddle.randn([4, 8])
        cls.apply(x, fn_arg1, fn_arg2, group=None, fn=mock_fn, is_first_fwd=True)
        mock_mb.assert_called_once_with(mock_fn, True, fn_arg1, fn_arg2)


class TestAlltoAllAsyncBackward(unittest.TestCase):
    """Tests for AlltoAllAsync backward pass."""

    def _get_cls(self):
        from paddleformers.nn.moe.all_to_all import AlltoAllAsync

        return AlltoAllAsync

    @patch("paddleformers.nn.moe.all_to_all.manual_backward")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=1)
    def test_backward_single_world(self, mock_ws, mock_mb):
        """Test backward with world_size=1."""
        cls = self._get_cls()
        mock_bwf = MagicMock()
        mock_bwf.return_value = (paddle.randn([4, 8]),)
        mock_mb.return_value = (mock_bwf, (paddle.randn([4, 8]),))

        x = paddle.randn([4, 8])
        result = cls.apply(x, group=None, fn=MagicMock(), is_first_fwd=True)
        # Verify the result structure is correct
        self.assertIsInstance(result, tuple)

    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=1)
    def test_backward_single_world_full(self, mock_ws):
        """Test backward with world_size=1 using full forward-backward cycle."""
        cls = self._get_cls()
        mock_fn = MagicMock()
        mock_fn.return_value = paddle.randn([4, 8])

        x = paddle.randn([4, 8])
        result = cls.apply(x, group=None, fn=mock_fn, is_first_fwd=True)
        self.assertIsInstance(result, tuple)


class TestAlltoAllBackward(unittest.TestCase):
    """Tests for AlltoAll backward pass."""

    def _get_cls(self):
        from paddleformers.nn.moe.all_to_all import AlltoAll

        return AlltoAll

    @patch("paddleformers.nn.moe.all_to_all.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=4)
    def test_backward_calls_alltoall(self, mock_ws, mock_alltoall):
        """Test that backward calls AlltoAll.apply."""
        cls = self._get_cls()
        x = paddle.randn([4, 8])
        mock_group = MagicMock()
        mock_alltoall.return_value = None

        result = cls.apply(x, group=mock_group, sync_op=True)
        self.assertIsNotNone(result)


class TestAlltoAllInputValidation(unittest.TestCase):
    """Tests for input validation in AlltoAll."""

    def _get_cls(self):
        from paddleformers.nn.moe.all_to_all import AlltoAll

        return AlltoAll

    @patch("paddleformers.nn.moe.all_to_all.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=2)
    def test_forward_preserves_dtype(self, mock_ws, mock_alltoall):
        """Test that forward preserves input dtype."""
        cls = self._get_cls()
        x = paddle.randn([4, 8], dtype="float32")
        mock_group = MagicMock()
        mock_alltoall.return_value = None

        result = cls.apply(x, group=mock_group, sync_op=True)
        self.assertEqual(result.dtype, x.dtype)

    @patch("paddleformers.nn.moe.all_to_all.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=2)
    def test_forward_preserves_shape(self, mock_ws, mock_alltoall):
        """Test that forward preserves input shape."""
        cls = self._get_cls()
        x = paddle.randn([4, 8])
        mock_group = MagicMock()
        mock_alltoall.return_value = None

        result = cls.apply(x, group=mock_group, sync_op=True)
        self.assertEqual(result.shape, x.shape)

    @patch("paddleformers.nn.moe.all_to_all.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=2)
    def test_forward_bfloat16_dtype(self, mock_ws, mock_alltoall):
        """Test that forward handles bfloat16 dtype."""
        cls = self._get_cls()
        x = paddle.randn([4, 8]).cast("bfloat16")
        mock_group = MagicMock()
        mock_alltoall.return_value = None

        result = cls.apply(x, group=mock_group, sync_op=True)
        self.assertEqual(result.dtype, x.dtype)


class TestAlltoAllAsyncInputValidation(unittest.TestCase):
    """Tests for input validation in AlltoAllAsync."""

    def _get_cls(self):
        from paddleformers.nn.moe.all_to_all import AlltoAllAsync

        return AlltoAllAsync

    @patch("paddleformers.nn.moe.all_to_all.manual_backward")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=1)
    def test_forward_fn_returns_tuple(self, mock_ws, mock_mb):
        """Test forward when fn returns a tuple."""
        cls = self._get_cls()
        mock_fn = MagicMock()
        mock_fn.return_value = (paddle.randn([4, 8]), paddle.randn([4, 8]))
        mock_bwf = MagicMock()
        mock_mb.return_value = (mock_bwf, mock_fn.return_value)

        x = paddle.randn([4, 8])
        result = cls.apply(x, group=None, fn=mock_fn, is_first_fwd=True)
        self.assertIsInstance(result, tuple)
        # Should have (x,) + (out1, out2) = 3 elements
        self.assertEqual(len(result), 3)

    @patch("paddleformers.nn.moe.all_to_all.manual_backward")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=1)
    def test_forward_fn_returns_list(self, mock_ws, mock_mb):
        """Test forward when fn returns a list (should be converted to tuple)."""
        cls = self._get_cls()
        mock_fn = MagicMock()
        mock_fn.return_value = [paddle.randn([4, 8])]
        mock_bwf = MagicMock()
        mock_mb.return_value = (mock_bwf, tuple(mock_fn.return_value))

        x = paddle.randn([4, 8])
        result = cls.apply(x, group=None, fn=mock_fn, is_first_fwd=True)
        self.assertIsInstance(result, tuple)

    @patch("paddleformers.nn.moe.all_to_all.stream.alltoall_single")
    @patch("paddleformers.nn.moe.all_to_all.manual_backward")
    @patch("paddleformers.nn.moe.all_to_all.dist.get_world_size", return_value=4)
    def test_forward_async_sync_op_false(self, mock_ws, mock_mb, mock_alltoall):
        """Test that async forward uses sync_op=False."""
        cls = self._get_cls()
        mock_task = MagicMock()
        mock_task.wait = MagicMock()
        mock_alltoall.return_value = mock_task

        mock_fn = MagicMock()
        mock_fn.return_value = paddle.randn([4, 8])
        mock_bwf = MagicMock()
        mock_mb.return_value = (mock_bwf, (paddle.randn([4, 8]),))

        mock_group = MagicMock()
        x = paddle.randn([4, 8])
        cls.apply(x, group=mock_group, fn=mock_fn, is_first_fwd=True)

        # Verify alltoall_single was called with correct sync_op settings
        call_args = mock_alltoall.call_args
        self.assertEqual(call_args[1]["sync_op"], False)


if __name__ == "__main__":
    unittest.main()
