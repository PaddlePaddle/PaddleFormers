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
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
    P2PonCalcStream,
    SendRecvMeta,
    _is_valid_send_recv_partial,
    allgather_partial,
    batch_send_recv_on_calc_stream,
    initialize_p2p_groups,
)
from paddleformers.fleet.pipeline_parallel.pp_utils.utils import paddle_2_number
from paddleformers.fleet.training.initialize import initialize_fleet

PP_DEGREE = 2


def _init_pp():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_DEGREE,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    initialize_fleet(strategy)


def setUpModule():
    """Initialize fleet once for all tests in this module (PP=2)."""
    _init_pp()
    np.random.seed(42)
    paddle.seed(42)


class TestSendRecvMetaSendMetaSingleTensor(unittest.TestCase):
    """Test SendRecvMeta.send_meta sets internal state for a single tensor."""

    def test_send_recv_meta_send_meta_single_tensor(self):
        """Verify that _obtain_send_message returns correct shape/dtype for a single tensor."""
        meta = SendRecvMeta()
        tensor = paddle.randn([4, 8], dtype="float32")
        meta.set_send_message(tensor)
        self.assertEqual(list(meta.send_shape_message), [4, 8])
        self.assertEqual(meta.send_dtype_message, paddle_2_number(paddle.float32))
        self.assertIsNone(meta.send_key_message)

        # check_send_message with matching tensor should pass
        meta.check_send_message(tensor)


class TestSendRecvMetaSendMetaTupleTensor(unittest.TestCase):
    """Test SendRecvMeta with tuple of tensors."""

    def test_send_recv_meta_send_meta_tuple_tensor(self):
        """Verify set_send_message works with tuple input and extracts per-tensor info."""
        meta = SendRecvMeta()
        t1 = paddle.randn([2, 4], dtype="float32")
        t1.stop_gradient = False
        t2 = paddle.randn([2, 4], dtype="float16")
        t2.stop_gradient = False
        meta.set_send_message((t1, t2))
        self.assertIsInstance(meta.send_shape_message, tuple)
        self.assertEqual(len(meta.send_shape_message), 2)
        self.assertEqual(list(meta.send_shape_message[0]), [2, 4])
        self.assertEqual(list(meta.send_shape_message[1]), [2, 4])
        self.assertIsInstance(meta.send_dtype_message, tuple)
        self.assertEqual(meta.send_dtype_message[0], paddle_2_number(paddle.float32))
        self.assertEqual(meta.send_dtype_message[1], paddle_2_number(paddle.float16))


class TestSendRecvMetaSetSendMessage(unittest.TestCase):
    """Test SendRecvMeta.set_send_message extracts shape/dtype from tensor."""

    def test_send_recv_meta_set_send_message(self):
        """Verify set_send_message correctly populates shape and dtype."""
        meta = SendRecvMeta()
        tensor = paddle.randn([3, 6, 9], dtype="float32")
        meta.set_send_message(tensor)
        self.assertEqual(meta.send_shape_message, [3, 6, 9])
        self.assertEqual(meta.send_dtype_message, paddle_2_number(paddle.float32))
        self.assertIsNone(meta.send_key_message)


class TestSendRecvMetaCheckSendMessage(unittest.TestCase):
    """Test SendRecvMeta.check_send_message validates consistency."""

    def test_send_recv_meta_check_send_message_match(self):
        """check_send_message should pass when tensor matches previously set message."""
        meta = SendRecvMeta()
        tensor = paddle.randn([4, 8], dtype="float32")
        meta.set_send_message(tensor)
        # Should not raise
        meta.check_send_message(tensor)

    def test_send_recv_meta_check_send_message_mismatch(self):
        """check_send_message should raise AssertionError on shape mismatch."""
        meta = SendRecvMeta()
        t1 = paddle.randn([4, 8], dtype="float32")
        meta.set_send_message(t1)
        t2 = paddle.randn([4, 16], dtype="float32")
        with self.assertRaises(AssertionError):
            meta.check_send_message(t2)


class TestP2PonCalcStreamCreation(unittest.TestCase):
    """Test P2PonCalcStream validates op functions on creation."""

    def test_p2p_on_calc_stream_creation_valid(self):
        """P2PonCalcStream creation should succeed with valid op functions."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            _recv_on_calc_stream,
            _send_on_calc_stream,
        )

        tensor = paddle.randn([4], dtype="float32")
        group = None  # Group not needed for object creation
        # Test with _send_on_calc_stream
        p2p_send = P2PonCalcStream(_send_on_calc_stream, tensor, 0, None)
        self.assertIs(p2p_send.op, _send_on_calc_stream)
        # Test with _recv_on_calc_stream
        p2p_recv = P2PonCalcStream(_recv_on_calc_stream, tensor, 0, None)
        self.assertIs(p2p_recv.op, _recv_on_calc_stream)

    def test_p2p_on_calc_stream_creation_invalid(self):
        """P2PonCalcStream should raise RuntimeError for invalid op."""
        tensor = paddle.randn([4], dtype="float32")
        with self.assertRaises(RuntimeError):
            P2PonCalcStream(lambda x, y, z: None, tensor, 0, None)


class TestIsValidSendRecvPartial(unittest.TestCase):
    """Test _is_valid_send_recv_partial with various tensor shapes and mp_degree."""

    def test_is_valid_send_recv_partial_divisible(self):
        """Should return True when tensor numel is divisible by mp_degree > 1."""
        tensor = paddle.randn([4, 8], dtype="float32")  # numel=32, divisible by 4
        self.assertTrue(_is_valid_send_recv_partial(tensor, 4))

    def test_is_valid_send_recv_partial_not_divisible(self):
        """Should return False when tensor numel is not divisible by mp_degree."""
        tensor = paddle.randn([3, 7], dtype="float32")  # numel=21, not divisible by 4
        self.assertFalse(_is_valid_send_recv_partial(tensor, 4))

    def test_is_valid_send_recv_partial_mp_degree_one(self):
        """Should return False when mp_degree is 1."""
        tensor = paddle.randn([4, 8], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 1))

    def test_is_valid_send_recv_partial_large_mp(self):
        """Should return True when tensor is large enough and divisible."""
        tensor = paddle.randn([16, 16], dtype="float32")  # numel=256, divisible by 8
        self.assertTrue(_is_valid_send_recv_partial(tensor, 8))

    def test_is_valid_send_recv_partial_odd_numel(self):
        """Should return False for odd numel with even mp_degree."""
        tensor = paddle.randn([3], dtype="float32")  # numel=3, not divisible by 2
        self.assertFalse(_is_valid_send_recv_partial(tensor, 2))


class TestAllgatherPartialNoOp(unittest.TestCase):
    """Test allgather_partial returns tensor unchanged when not divisible."""

    def test_allgather_partial_no_op(self):
        """allgather_partial should return the tensor unchanged when not divisible by nranks."""
        tensor = paddle.randn([3, 7], dtype="float32")
        result = allgather_partial(tensor, nranks=4, rank_id=0)
        # When tensor is not divisible, _is_valid_send_recv_partial returns False
        # and allgather_partial returns the tensor unchanged
        self.assertIs(result, tensor)

    def test_allgather_partial_mp_degree_one(self):
        """allgather_partial should return the tensor unchanged when nranks=1."""
        tensor = paddle.randn([4, 8], dtype="float32")
        result = allgather_partial(tensor, nranks=1, rank_id=0)
        self.assertIs(result, tensor)


class TestBatchSendRecvOnCalcStreamEmpty(unittest.TestCase):
    """Test batch_send_recv_on_calc_stream edge cases."""

    def test_batch_send_recv_on_calc_stream_empty(self):
        """batch_send_recv_on_calc_stream should handle empty list gracefully.

        With an empty list it will try to access index 0 which will raise IndexError.
        This test documents the expected behavior for the edge case.
        """
        empty_list = []
        # Calling with empty list raises IndexError because it accesses p2p_op_list[0]
        with self.assertRaises((IndexError, TypeError)):
            batch_send_recv_on_calc_stream(empty_list)


class TestInitializeP2PGroups(unittest.TestCase):
    """Test initialize_p2p_groups with real PP group."""

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() and dist.is_initialized()),
        "Requires CUDA and distributed environment",
    )
    def test_initialize_p2p_groups(self):
        """initialize_p2p_groups should succeed with a valid HCG."""
        hcg = fleet.get_hybrid_communicate_group()
        # Should not raise
        initialize_p2p_groups(hcg, enable_partial_send_recv=True)
        initialize_p2p_groups(hcg, enable_partial_send_recv=False)


class TestSendRecvMetaInitOrErase(unittest.TestCase):
    """Test SendRecvMeta.init_or_erase_meta resets all fields."""

    def test_init_or_erase_meta_resets_fields(self):
        """After init_or_erase_meta, all fields should be None/False."""
        meta = SendRecvMeta()
        meta.send_shape_message = [2, 3]
        meta.send_dtype_message = 5
        meta.recv_shape_message = [4, 5]
        meta.recv_dtype_message = 3
        meta.has_send_meta = True
        meta.has_recv_meta = True
        meta.init_or_erase_meta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.send_dtype_message)
        self.assertIsNone(meta.recv_shape_message)
        self.assertIsNone(meta.recv_dtype_message)
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)


class TestSendRecvMetaRepr(unittest.TestCase):
    """Test SendRecvMeta.__repr__ output."""

    def test_repr_contains_expected_fields(self):
        """__repr__ should contain key field names."""
        meta = SendRecvMeta()
        repr_str = repr(meta)
        self.assertIn("send_shape_message", repr_str)
        self.assertIn("send_dtype_message", repr_str)
        self.assertIn("recv_shape_message", repr_str)
        self.assertIn("recv_dtype_message", repr_str)


if __name__ == "__main__":
    unittest.main()
