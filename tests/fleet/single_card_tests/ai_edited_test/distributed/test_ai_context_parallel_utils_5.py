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


# Tests for src/paddleformers.fleet/context_parallel_utils.py
# Test ContextParallelScatterOp, ContextParallelGatherOp,
# ContextParallelAllGatherOp PyLayers

import unittest
from unittest import mock

import paddle


class TestContextParallelScatterOp(unittest.TestCase):
    """Tests for ContextParallelScatterOp PyLayer."""

    def test_forward_saves_axis(self):
        """Test that forward saves axis to context."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelScatterOp,
        )

        mock_ctx = mock.MagicMock()
        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_hcg.get_context_parallel_world_size.return_value = 2
        mock_hcg.get_context_parallel_group.return_value = mock_group

        x = paddle.randn([8, 16])

        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=mock_hcg,
            ),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.scatter_balance",
                return_value=paddle.randn([4, 16]),
            ) as mock_scatter,
        ):
            result = ContextParallelScatterOp.forward(mock_ctx, x, axis=0)
            self.assertEqual(mock_ctx.axis, 0)
            self.assertEqual(mock_ctx.group, mock_group)
            mock_scatter.assert_called_once()

    def test_forward_assertion_cp_world_size(self):
        """Test forward asserts context parallel world size > 1."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelScatterOp,
        )

        mock_ctx = mock.MagicMock()
        mock_hcg = mock.MagicMock()
        mock_hcg.get_context_parallel_world_size.return_value = 1

        x = paddle.randn([8, 16])

        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            with self.assertRaises(AssertionError) as ctx:
                ContextParallelScatterOp.forward(mock_ctx, x, axis=0)
            self.assertIn("ScatterOpCP", str(ctx.exception))

    def test_backward_calls_all_gather_balance(self):
        """Test backward calls all_gather_balance."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelScatterOp,
        )

        mock_ctx = mock.MagicMock()
        mock_ctx.axis = 0
        mock_ctx.group = mock.MagicMock()

        grad = paddle.randn([4, 16])

        with mock.patch(
            "paddleformers.fleet.context_parallel_utils.all_gather_balance",
            return_value=paddle.randn([8, 16]),
        ) as mock_gather:
            result = ContextParallelScatterOp.backward(mock_ctx, grad)
            mock_gather.assert_called_once_with(
                grad, axis=0, group=mock_ctx.group
            )


class TestContextParallelGatherOp(unittest.TestCase):
    """Tests for ContextParallelGatherOp PyLayer."""

    def test_forward_saves_axis_and_group(self):
        """Test that forward saves axis and group."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelGatherOp,
        )

        mock_ctx = mock.MagicMock()
        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_hcg.get_context_parallel_world_size.return_value = 2
        mock_hcg.get_context_parallel_group.return_value = mock_group

        x = paddle.randn([4, 16])

        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=mock_hcg,
            ),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.all_gather_balance",
                return_value=paddle.randn([8, 16]),
            ),
        ):
            result = ContextParallelGatherOp.forward(mock_ctx, x, axis=0)
            self.assertEqual(mock_ctx.axis, 0)
            self.assertEqual(mock_ctx.group, mock_group)

    def test_forward_assertion_cp_world_size(self):
        """Test forward asserts context parallel world size > 1."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelGatherOp,
        )

        mock_ctx = mock.MagicMock()
        mock_hcg = mock.MagicMock()
        mock_hcg.get_context_parallel_world_size.return_value = 1

        x = paddle.randn([4, 16])

        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            with self.assertRaises(AssertionError) as ctx:
                ContextParallelGatherOp.forward(mock_ctx, x, axis=0)
            self.assertIn("GatherOpCP", str(ctx.exception))

    def test_backward_calls_scatter_balance(self):
        """Test backward calls scatter_balance."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelGatherOp,
        )

        mock_ctx = mock.MagicMock()
        mock_ctx.axis = 1
        mock_ctx.group = mock.MagicMock()

        grad = paddle.randn([8, 16])

        with mock.patch(
            "paddleformers.fleet.context_parallel_utils.scatter_balance",
            return_value=paddle.randn([4, 16]),
        ) as mock_scatter:
            result = ContextParallelGatherOp.backward(mock_ctx, grad)
            mock_scatter.assert_called_once_with(
                grad, axis=1, group=mock_ctx.group
            )


class TestContextParallelAllGatherOp(unittest.TestCase):
    """Tests for ContextParallelAllGatherOp PyLayer."""

    def test_forward_saves_context(self):
        """Test that forward saves axis and group."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        mock_ctx = mock.MagicMock()
        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_hcg.get_context_parallel_world_size.return_value = 2
        mock_hcg.get_context_parallel_group.return_value = mock_group

        x = paddle.randn([4, 16])

        with (
            mock.patch(
                "paddle.distributed.fleet.get_hybrid_communicate_group",
                return_value=mock_hcg,
            ),
            mock.patch(
                "paddleformers.fleet.context_parallel_utils.all_gather_balance",
                return_value=paddle.randn([8, 16]),
            ),
        ):
            result = ContextParallelAllGatherOp.forward(mock_ctx, x, axis=0)
            self.assertEqual(mock_ctx.axis, 0)
            self.assertEqual(mock_ctx.group, mock_group)

    def test_forward_assertion_cp_world_size(self):
        """Test forward asserts context parallel world size > 1."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        mock_ctx = mock.MagicMock()
        mock_hcg = mock.MagicMock()
        mock_hcg.get_context_parallel_world_size.return_value = 1

        x = paddle.randn([4, 16])

        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            with self.assertRaises(AssertionError) as ctx:
                ContextParallelAllGatherOp.forward(mock_ctx, x, axis=0)
            self.assertIn("AllGatherOpCP", str(ctx.exception))

    def test_backward_calls_reduce_scatter_balance(self):
        """Test backward calls reduce_scatter_any_axis_balance."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        mock_ctx = mock.MagicMock()
        mock_ctx.axis = 1
        mock_ctx.group = mock.MagicMock()

        grad = paddle.randn([8, 16])

        with mock.patch(
            "paddleformers.fleet.context_parallel_utils.reduce_scatter_any_axis_balance",
            return_value=paddle.randn([4, 16]),
        ) as mock_rs:
            result = ContextParallelAllGatherOp.backward(mock_ctx, grad)
            mock_rs.assert_called_once_with(grad, axis=1, group=mock_ctx.group)


class TestPyLayerStaticMethods(unittest.TestCase):
    """Tests for static method signatures."""

    def test_scatter_op_has_forward_backward(self):
        """Test ContextParallelScatterOp has forward and backward."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelScatterOp,
        )

        self.assertTrue(hasattr(ContextParallelScatterOp, "forward"))
        self.assertTrue(hasattr(ContextParallelScatterOp, "backward"))
        self.assertTrue(callable(ContextParallelScatterOp.forward))
        self.assertTrue(callable(ContextParallelScatterOp.backward))

    def test_gather_op_has_forward_backward(self):
        """Test ContextParallelGatherOp has forward and backward."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelGatherOp,
        )

        self.assertTrue(hasattr(ContextParallelGatherOp, "forward"))
        self.assertTrue(hasattr(ContextParallelGatherOp, "backward"))

    def test_allgather_op_has_forward_backward(self):
        """Test ContextParallelAllGatherOp has forward and backward."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        self.assertTrue(hasattr(ContextParallelAllGatherOp, "forward"))
        self.assertTrue(hasattr(ContextParallelAllGatherOp, "backward"))


if __name__ == "__main__":
    unittest.main()
