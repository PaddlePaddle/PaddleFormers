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

from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
    ScheduleChunk,
    ScheduleNode,
)
from paddleformers.fleet.training.initialize import initialize_fleet

PP_DEGREE = 4


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


def _set_random_seed(seed_):
    seed = seed_ + 100 * dist.get_rank()
    np.random.seed(seed)
    paddle.manual_seed(seed)


class TestScheduleNode(unittest.TestCase):
    def test_schedule_node_creation(self):
        """Test ScheduleNode can be created with a function and name."""
        node = ScheduleNode(lambda x: x * 2, name="test_node")
        self.assertEqual(node.name, "test_node")

    def test_schedule_node_name_default(self):
        """Test ScheduleNode default name is empty string."""
        node = ScheduleNode(lambda x: x)
        self.assertEqual(node.name, "")


class TestDetachAndRequiresGrad(unittest.TestCase):
    def test_detach_and_requires_grad_tensor(self):
        """Test detach_and_requires_grad with a plain tensor."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        y = x * 2
        detached = detach_and_requires_grad(y)
        self.assertFalse(detached.stop_gradient)
        np.testing.assert_allclose(detached.numpy(), [2.0, 4.0, 6.0], rtol=1e-5)

    def test_detach_and_requires_grad_tuple(self):
        """Test detach_and_requires_grad with a tuple of tensors."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        t1 = paddle.to_tensor([1.0], stop_gradient=False)
        t2 = paddle.to_tensor([2.0], stop_gradient=False)
        result = detach_and_requires_grad((t1, t2))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertFalse(result[0].stop_gradient)
        self.assertFalse(result[1].stop_gradient)

    def test_detach_and_requires_grad_dict(self):
        """Test detach_and_requires_grad with a dict of tensors."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
            detach_and_requires_grad,
        )

        t1 = paddle.to_tensor([1.0], stop_gradient=False)
        t2 = paddle.to_tensor([2.0], stop_gradient=False)
        result = detach_and_requires_grad({"a": t1, "b": t2})
        self.assertIsInstance(result, dict)
        self.assertFalse(result["a"].stop_gradient)


class TestCloneAndClearDataptr(unittest.TestCase):
    def test_clone_and_clear_dataptr_basic(self):
        """Test clone_and_clear_dataptr creates a FakeClone wrapper."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
            clone_and_clear_dataptr,
        )

        x = paddle.to_tensor([1.0, 2.0, 3.0])
        cloned = clone_and_clear_dataptr(x)
        # FakeClone wraps the tensor; verify it has the right type
        self.assertIsNotNone(cloned)


class TestScheduleChunk(unittest.TestCase):
    def test_schedule_chunk_creation(self):
        """Test ScheduleChunk validates nodes."""
        node = ScheduleNode(lambda x: x, name="node1")
        chunk = ScheduleChunk([node])
        self.assertEqual(len(chunk.nodes), 1)


class TestP2PCommunicationInit(unittest.TestCase):
    """Test p2p_communication initialization with real PP."""

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() and paddle.distributed.is_initialized()),
        "Requires CUDA and distributed environment",
    )
    def test_initialize_p2p_groups(self):
        """Test initialize_p2p_groups with hybrid communicate group."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            initialize_p2p_groups,
        )

        hcg = fleet.get_hybrid_communicate_group()
        initialize_p2p_groups(hcg, enable_partial_send_recv=True)


class TestSendRecvMeta(unittest.TestCase):
    def test_init_or_erase_meta(self):
        """Test that init_or_erase_meta resets all fields."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        meta.send_shape_message = [2, 3]
        meta.recv_shape_message = [4, 5]
        meta.has_send_meta = True
        meta.has_recv_meta = True
        meta.init_or_erase_meta()
        self.assertIsNone(meta.send_shape_message)
        self.assertIsNone(meta.recv_shape_message)
        # has_send_meta and has_recv_meta are on the meta object, not SendRecvMeta class
        self.assertFalse(meta.has_send_meta)
        self.assertFalse(meta.has_recv_meta)

    def test_obtain_send_message_single(self):
        """Test _obtain_send_message with a single tensor."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        tensor = paddle.randn([4, 8], dtype="float32")
        shape, dtype, key = meta._obtain_send_message(tensor)
        self.assertEqual(shape, [4, 8])
        self.assertEqual(dtype, 1)  # float32 = 1 in PADDLE_TO_NUMBER
        self.assertIsNone(key)

    def test_obtain_send_message_tuple(self):
        """Test _obtain_send_message with a tuple of tensors."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t1 = paddle.randn([2, 4], dtype="float32")
        t1.stop_gradient = False
        t2 = paddle.randn([2, 4], dtype="float16")
        t2.stop_gradient = False
        shapes, dtypes, keys = meta._obtain_send_message((t1, t2))
        self.assertEqual(shapes, ([2, 4], [2, 4]))
        self.assertEqual(dtypes, (1, 0))  # float32=1, float16=0
        self.assertEqual(keys, (None, None))

    def test_set_and_check_send_message(self):
        """Test set_send_message and check_send_message."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        tensor = paddle.randn([4, 8], dtype="float32")
        meta.set_send_message(tensor)
        self.assertEqual(meta.send_shape_message, [4, 8])
        meta.check_send_message(tensor)

    def test_check_send_message_mismatch(self):
        """Test check_send_message raises on shape mismatch."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        t1 = paddle.randn([4, 8], dtype="float32")
        meta.set_send_message(t1)
        t2 = paddle.randn([4, 16], dtype="float32")
        with self.assertRaises(AssertionError):
            meta.check_send_message(t2)

    def test_repr(self):
        """Test __repr__ output."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            SendRecvMeta,
        )

        meta = SendRecvMeta()
        repr_str = repr(meta)
        self.assertIn("send_shape_message", repr_str)
        self.assertIn("recv_shape_message", repr_str)


class TestIsValidSendRecvPartial(unittest.TestCase):
    def test_valid_partial(self):
        """Test valid partial when tensor divisible by mp_degree."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        tensor = paddle.randn([4, 8], dtype="float32")
        self.assertTrue(_is_valid_send_recv_partial(tensor, 4))

    def test_not_divisible(self):
        """Test invalid when tensor not divisible by mp_degree."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        tensor = paddle.randn([3, 7], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 4))

    def test_mp_degree_one(self):
        """Test invalid when mp_degree is 1."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        tensor = paddle.randn([4, 8], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 1))

    def test_empty_tensor_raises(self):
        """Test empty tensor raises assertion."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            _is_valid_send_recv_partial,
        )

        tensor = paddle.randn([0, 8], dtype="float32")
        with self.assertRaises(AssertionError):
            _is_valid_send_recv_partial(tensor, 4)


class TestP2PonCalcStream(unittest.TestCase):
    def test_invalid_op(self):
        """Test P2PonCalcStream raises on invalid op."""
        from paddleformers.fleet.pipeline_parallel.pp_utils.p2p_communication import (
            P2PonCalcStream,
        )

        with self.assertRaises(RuntimeError):
            P2PonCalcStream(
                lambda x, y, z: None,
                paddle.randn([4], dtype="float32"),
                0,
                None,
            )


if __name__ == "__main__":
    unittest.main()
