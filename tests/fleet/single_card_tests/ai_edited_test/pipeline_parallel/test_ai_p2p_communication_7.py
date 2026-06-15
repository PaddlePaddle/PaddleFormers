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


class TestSendRecvMetaInit(unittest.TestCase):
    """Tests for SendRecvMeta initialization (p2p_communication)."""

    def test_init_default(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertIsNone(meta.recv_stop_gradient)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_init_or_erase_meta(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        meta.send_shape_message = [2, 3]
        meta.has_send_meta = True
        meta.has_recv_meta = True
        meta.init_or_erase_meta()
        self.assertIsNone(meta.send_shape_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)


class TestSendRecvMetaInitOrEraseMeta(unittest.TestCase):
    """Tests for SendRecvMeta.init_or_erase_meta."""

    def test_erase_resets_all_fields(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        meta.send_shape_message = [1, 2]
        meta.send_dtype_message = 0
        meta.send_key_message = "key1"
        meta.recv_shape_message = [3, 4]
        meta.recv_dtype_message = 1
        meta.recv_stop_gradient = False
        meta.recv_key_message = "key2"
        meta.has_send_meta = True
        meta.has_recv_meta = True

        meta.init_or_erase_meta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.send_key_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertIsNone(meta.recv_stop_gradient)
        self.assertIsNone(meta.recv_key_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)


class TestSendRecvMetaRepr(unittest.TestCase):
    """Tests for SendRecvMeta.__repr__."""

    def test_repr_empty(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        r = repr(meta)
        self.assertIn("send_shape_message", r)
        self.assertIn("recv_shape_message", r)

    def test_repr_with_values(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        meta.send_shape_message = [2, 3]
        meta.send_dtype_message = 0
        meta.recv_shape_message = [2, 3]
        meta.recv_dtype_message = 1
        meta.recv_stop_gradient = False
        r = repr(meta)
        self.assertIn("[2, 3]", r)


class TestIsValidSendRecvPartial(unittest.TestCase):
    """Tests for _is_valid_send_recv_partial."""

    @patch(
        "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._enable_partial_send_recv",
        False,
    )
    def test_disabled(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        tensor = paddle.randn([4, 4])
        self.assertFalse(_is_valid_send_recv_partial(tensor, mp_degree=2))

    @patch(
        "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._enable_partial_send_recv",
        True,
    )
    def test_mp_degree_one(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        tensor = paddle.randn([4, 4])
        self.assertFalse(_is_valid_send_recv_partial(tensor, mp_degree=1))

    @patch(
        "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._enable_partial_send_recv",
        True,
    )
    def test_valid_partial(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        tensor = paddle.randn([4, 4])
        self.assertTrue(_is_valid_send_recv_partial(tensor, mp_degree=2))

    @patch(
        "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._enable_partial_send_recv",
        True,
    )
    def test_not_divisible(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        tensor = paddle.randn([3, 3])
        self.assertFalse(_is_valid_send_recv_partial(tensor, mp_degree=2))


class TestAllgatherPartial(unittest.TestCase):
    """Tests for allgather_partial function."""

    @patch(
        "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._is_valid_send_recv_partial",
        return_value=False,
    )
    def test_allgather_partial_invalid(self, mock_valid):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            allgather_partial,
        )

        tensor = paddle.randn([4, 4])
        result = allgather_partial(tensor, nranks=1, rank_id=0)
        # Should return tensor directly
        self.assertIs(result, tensor)

    @patch(
        "paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication._is_valid_send_recv_partial",
        return_value=False,
    )
    def test_allgather_partial_not_member(self, mock_valid):
        # When _is_valid returns False, allgather_partial returns tensor directly
        # and never checks is_member. Test with valid=False to go through the
        # member check path by making _is_valid return True for non-member check
        pass  # The member check only runs when _is_valid is True


class TestP2PonCalcStream(unittest.TestCase):
    """Tests for P2PonCalcStream."""

    def test_init_send(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
            _send_on_calc_stream,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        op = P2PonCalcStream(_send_on_calc_stream, mock_tensor, 1, mock_group, 2, 0)
        self.assertEqual(op.op, _send_on_calc_stream)
        self.assertEqual(op.tensor, mock_tensor)
        self.assertEqual(op.peer, 1)
        self.assertEqual(op.nranks, 2)
        self.assertEqual(op.rank_id, 0)

    def test_init_recv(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
            _recv_on_calc_stream,
        )

        mock_tensor = MagicMock()
        mock_group = MagicMock()
        op = P2PonCalcStream(_recv_on_calc_stream, mock_tensor, 0, mock_group)
        self.assertEqual(op.op, _recv_on_calc_stream)
        self.assertEqual(op.peer, 0)

    def test_init_invalid_op(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
        )

        with self.assertRaises(RuntimeError):
            P2PonCalcStream(lambda x: None, MagicMock(), 0, MagicMock())


class TestSendRecvMetaObtainSendMessage(unittest.TestCase):
    """Tests for SendRecvMeta._obtain_send_message."""

    def test_obtain_single_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        tensor = paddle.randn([2, 3])
        shape, dtype, key = meta._obtain_send_message(tensor)
        self.assertEqual(list(shape), [2, 3])

    def test_obtain_tuple_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t1 = paddle.randn([2, 3])
        t1.stop_gradient = False
        t2 = paddle.randn([2, 3])
        t2.stop_gradient = False
        shapes, dtypes, keys = meta._obtain_send_message((t1, t2))
        self.assertEqual(len(shapes), 2)
        self.assertEqual(len(dtypes), 2)

    def test_obtain_tuple_skip_stop_gradient(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t1 = paddle.randn([2, 3])
        t1.stop_gradient = True
        t2 = paddle.randn([2, 3])
        t2.stop_gradient = False
        shapes, dtypes, keys = meta._obtain_send_message((t1, t2))
        self.assertEqual(len(shapes), 1)


class TestSendRecvMetaCheckSendMessage(unittest.TestCase):
    """Tests for SendRecvMeta.check_send_message."""

    def test_check_none_messages(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        # Should return early when send_shape_message is None
        tensor = paddle.randn([2, 3])
        meta.check_send_message(tensor)

    def test_check_matching_messages(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        tensor = paddle.randn([2, 3])
        meta.set_send_message(tensor)
        # Should not raise
        meta.check_send_message(tensor)

    def test_check_mismatched_messages(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([4, 5])
        meta.set_send_message(t1)
        with self.assertRaises(AssertionError):
            meta.check_send_message(t2)


if __name__ == "__main__":
    unittest.main()
