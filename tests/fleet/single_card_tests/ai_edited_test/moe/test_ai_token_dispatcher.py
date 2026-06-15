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

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest
from unittest.mock import MagicMock, patch

import paddle


class TestTokenDispatcher(unittest.TestCase):
    """Unit tests for token_dispatcher module."""

    def test_moe_token_dispatcher_init(self):
        """Test MoETokenDispatcher init."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            MoETokenDispatcher,
        )

        mock_group = MagicMock()
        mock_group.world_size = 2
        dispatcher = MoETokenDispatcher(mock_group)
        self.assertEqual(dispatcher.ep_group, mock_group)
        self.assertEqual(dispatcher.ep_size, 2)

    def test_dispatch_manager_abstract(self):
        """Test _DispatchManager is abstract and cannot be instantiated."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            _DispatchManager,
        )

        with self.assertRaises(TypeError):
            _DispatchManager()

    def test_moe_flex_token_dispatcher_requires_ep_gt_1(self):
        """Test MoEFlexTokenDispatcher asserts EP > 1."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            MoEFlexTokenDispatcher,
        )

        mock_group = MagicMock()
        mock_group.world_size = 1
        with self.assertRaises(AssertionError):
            MoEFlexTokenDispatcher(
                num_local_experts=2,
                num_experts_per_tok=2,
                n_routed_experts=4,
                ep_group=mock_group,
            )

    def test_moe_flex_token_dispatcher_init(self):
        """Test MoEFlexTokenDispatcher initialization."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            MoEFlexTokenDispatcher,
        )

        mock_group = MagicMock()
        mock_group.world_size = 2
        mock_group.id = 0

        with (
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
                MagicMock(),
            ),
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine",
                MagicMock(),
            ),
        ):
            dispatcher = MoEFlexTokenDispatcher(
                num_local_experts=2,
                num_experts_per_tok=2,
                n_routed_experts=4,
                ep_group=mock_group,
            )
            self.assertIsNotNone(dispatcher._comm_manager)
            self.assertEqual(dispatcher.num_local_experts, 2)

    def test_alltoall_token_dispatcher_init(self):
        """Test AllToAllTokenDispatcher initialization."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            AllToAllTokenDispatcher,
        )

        mock_group = MagicMock()
        mock_group.world_size = 2

        dispatcher = AllToAllTokenDispatcher(
            moe_group=mock_group,
            expert_model_parallel_size=2,
            num_experts_per_device=2,
            local_expert_indices=[0, 1],
        )
        self.assertEqual(dispatcher.num_local_experts, 2)
        self.assertEqual(dispatcher.expert_model_parallel_size, 2)

    def test_alltoall_token_dispatcher_dispatch_preprocess_3d(self):
        """Test AllToAllTokenDispatcher.dispatch_preprocess with 3D input."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            AllToAllTokenDispatcher,
        )

        mock_group = MagicMock()
        mock_group.world_size = 1

        dispatcher = AllToAllTokenDispatcher(
            moe_group=mock_group,
            expert_model_parallel_size=1,
            num_experts_per_device=2,
            local_expert_indices=[0, 1],
        )

        hidden = paddle.randn([2, 4, 64], dtype=paddle.float32)
        gates_masked = paddle.randn([8, 2], dtype=paddle.float32)
        mask = paddle.zeros([8, 2], dtype=paddle.float32)
        mask[0, 0] = 1.0
        mask[1, 1] = 1.0

        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher.AllGatherGroupOp.apply",
            return_value=paddle.zeros([1, 2], dtype=paddle.int64),
        ):
            result = dispatcher.dispatch_preprocess(hidden, gates_masked, mask)
            self.assertIsNotNone(result)

    def test_alltoall_token_dispatcher_dispatch_preprocess_2d(self):
        """Test AllToAllTokenDispatcher.dispatch_preprocess with 2D input."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            AllToAllTokenDispatcher,
        )

        mock_group = MagicMock()
        mock_group.world_size = 1

        dispatcher = AllToAllTokenDispatcher(
            moe_group=mock_group,
            expert_model_parallel_size=1,
            num_experts_per_device=2,
            local_expert_indices=[0, 1],
        )

        hidden = paddle.randn([8, 64], dtype=paddle.float32)
        gates_masked = paddle.randn([8, 2], dtype=paddle.float32)
        mask = paddle.zeros([8, 2], dtype=paddle.float32)
        mask[0, 0] = 1.0
        mask[1, 1] = 1.0

        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher.AllGatherGroupOp.apply",
            return_value=paddle.zeros([1, 2], dtype=paddle.int64),
        ):
            result = dispatcher.dispatch_preprocess(hidden, gates_masked, mask)
            self.assertIsNotNone(result)

    def test_alltoall_token_dispatcher_token_dispatch(self):
        """Test AllToAllTokenDispatcher.token_dispatch."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            AllToAllTokenDispatcher,
        )

        mock_group = MagicMock()
        mock_group.world_size = 1

        dispatcher = AllToAllTokenDispatcher(
            moe_group=mock_group,
            expert_model_parallel_size=1,
            num_experts_per_device=2,
            local_expert_indices=[0, 1],
        )

        dispatcher.output_shape_tokens = [4, 64]
        dispatcher.output_splits = [4]
        dispatcher.input_split_sizes = paddle.to_tensor([2, 2], dtype=paddle.int64)
        dispatcher.permutated_local_input_tokens_shape = [4, 64]

        tokens = paddle.randn([4, 64], dtype=paddle.float32)

        with patch(
            "paddleformers.fleet.transformer.moe.token_dispatcher._AllToAll.apply",
            return_value=paddle.randn([4, 64], dtype=paddle.float32),
        ):
            result, _ = dispatcher.token_dispatch(tokens)
            self.assertIsNotNone(result)

    def test_alltoall_token_dispatcher_combine_postprocess(self):
        """Test AllToAllTokenDispatcher.combine_postprocess."""
        from paddleformers.fleet.transformer.moe.moe_utils import permute
        from paddleformers.fleet.transformer.moe.token_dispatcher import (
            AllToAllTokenDispatcher,
        )

        mock_group = MagicMock()
        mock_group.world_size = 1

        dispatcher = AllToAllTokenDispatcher(
            moe_group=mock_group,
            expert_model_parallel_size=1,
            num_experts_per_device=2,
            local_expert_indices=[0, 1],
        )

        hidden = paddle.randn([4, 64], dtype=paddle.float32)
        routing_map = paddle.zeros([4, 2], dtype=paddle.float32)
        routing_map[0, 0] = 1.0
        routing_map[1, 0] = 1.0
        routing_map[2, 1] = 1.0
        routing_map[3, 1] = 1.0
        probs = routing_map.clone()
        dispatcher.reshaped_input_shape = hidden.shape
        dispatcher.routing_map = routing_map
        dispatcher.probs = probs

        _, dispatcher.reversed_local_input_permutation_mapping = permute(hidden, routing_map)

        result = dispatcher.combine_postprocess(hidden)
        self.assertIsNotNone(result)

    def test_deepep_manager_no_deep_ep_raises(self):
        """Test _DeepEPManager raises ImportError when DeepEP unavailable."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import _DeepEPManager

        mock_group = MagicMock()
        mock_group.world_size = 2

        with (
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
                None,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine",
                None,
            ),
            self.assertRaises(ImportError),
        ):
            _DeepEPManager(
                group=mock_group,
                router_topk=2,
                num_experts=4,
                num_local_experts=2,
            )

    def test_deepep_manager_init(self):
        """Test _DeepEPManager initialization."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import _DeepEPManager

        mock_group = MagicMock()
        mock_group.world_size = 2

        mock_fused_dispatch = MagicMock()
        mock_fused_combine = MagicMock()

        with (
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
                mock_fused_dispatch,
            ),
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine",
                mock_fused_combine,
            ),
        ):
            manager = _DeepEPManager(
                group=mock_group,
                router_topk=2,
                num_experts=4,
                num_local_experts=2,
            )
            self.assertIsNone(manager.token_indices)
            self.assertIsNone(manager.token_probs)
            self.assertIsNone(manager.handle)

    def test_deepep_manager_setup_metadata(self):
        """Test _DeepEPManager.setup_metadata."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import _DeepEPManager

        mock_group = MagicMock()
        mock_group.world_size = 2

        with (
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
                MagicMock(),
            ),
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine",
                MagicMock(),
            ),
        ):
            manager = _DeepEPManager(
                group=mock_group,
                router_topk=2,
                num_experts=4,
                num_local_experts=2,
            )
            routing_map = paddle.zeros([4, 4], dtype=paddle.float32)
            probs = paddle.randn([4, 4], dtype=paddle.float32)
            manager.setup_metadata(routing_map, probs)
            self.assertIsNotNone(manager.token_probs)
            self.assertIsNotNone(manager.token_indices)

    def test_deepep_manager_indices_to_multihot(self):
        """Test _DeepEPManager._indices_to_multihot."""
        from paddleformers.fleet.transformer.moe.token_dispatcher import _DeepEPManager

        mock_group = MagicMock()
        mock_group.world_size = 2

        with (
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
                MagicMock(),
            ),
            patch(
                "paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine",
                MagicMock(),
            ),
        ):
            manager = _DeepEPManager(
                group=mock_group,
                router_topk=2,
                num_experts=4,
                num_local_experts=2,
            )
            indices = paddle.to_tensor([[0, 1], [0, -1]], dtype=paddle.int64)
            probs = paddle.randn([2, 2], dtype=paddle.float32)
            routing_map, multihot_probs = manager._indices_to_multihot(indices, probs)
            self.assertEqual(routing_map.shape, [2, 2])
            self.assertEqual(multihot_probs.shape, [2, 2])


if __name__ == "__main__":
    unittest.main()
