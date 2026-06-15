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
    from paddleformers.fleet.transformer.moe.token_dispatcher import (
        AllToAllTokenDispatcher,
        MoEFlexTokenDispatcher,
        _DeepEPManager,
        _DispatchManager,
    )

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestDispatchManagerInterface(unittest.TestCase):
    """Test _DispatchManager is abstract and defines the interface."""

    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            _DispatchManager()

    def test_abstract_methods_exist(self):
        abstract_methods = _DispatchManager.__abstractmethods__
        self.assertIn("setup_metadata", abstract_methods)
        self.assertIn("dispatch", abstract_methods)
        self.assertIn("combine", abstract_methods)
        self.assertIn("get_dispatched_metadata", abstract_methods)
        self.assertIn("get_permuted_hidden_states_by_experts", abstract_methods)
        self.assertIn("get_restored_hidden_states_by_experts", abstract_methods)


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestDeepepManagerConstruction(unittest.TestCase):
    """Test _DeepEPManager construction."""

    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch", None)
    def test_no_deepep_raises(self):
        group = MagicMock()
        with self.assertRaises(ImportError):
            _DeepEPManager(group, router_topk=2)

    @patch(
        "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
        MagicMock(),
    )
    def test_basic_construction(self):
        group = MagicMock()
        manager = _DeepEPManager(group, router_topk=2, num_experts=8, num_local_experts=4)
        self.assertEqual(manager.router_topk, 2)
        self.assertEqual(manager.num_experts, 8)
        self.assertEqual(manager.num_local_experts, 4)
        self.assertIsNone(manager.token_indices)
        self.assertIsNone(manager.token_probs)
        self.assertIsNone(manager.handle)


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestDeepepManagerSetupMetadata(unittest.TestCase):
    """Test _DeepEPManager.setup_metadata."""

    @patch(
        "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
        MagicMock(),
    )
    def test_setup_metadata_topk(self):
        group = MagicMock()
        manager = _DeepEPManager(group, router_topk=2, num_experts=4)
        routing_map = paddle.randn([4, 4], dtype="float32")
        probs = paddle.randn([4, 4], dtype="float32")
        manager.setup_metadata(routing_map, probs)
        self.assertIsNotNone(manager.token_indices)
        self.assertIsNotNone(manager.token_probs)
        # topk=2, so shape should be [4, 2]
        self.assertEqual(manager.token_indices.shape, [4, 2])
        self.assertEqual(manager.token_probs.shape, [4, 2])

    @patch(
        "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
        MagicMock(),
    )
    def test_setup_metadata_reshapes(self):
        group = MagicMock()
        manager = _DeepEPManager(group, router_topk=2, num_experts=4)
        # routing_map shape should be [num_tokens, num_experts] = [4, 4]
        routing_map = paddle.randn([4, 4], dtype="float32")
        probs = paddle.randn([4, 4], dtype="float32")
        manager.setup_metadata(routing_map, probs)
        self.assertEqual(manager.token_indices.shape, [4, 2])
        self.assertEqual(manager.token_probs.shape, [4, 2])


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestDeepepManagerIndicesToMultihot(unittest.TestCase):
    """Test _DeepEPManager._indices_to_multihot."""

    @patch(
        "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
        MagicMock(),
    )
    def test_basic_conversion(self):
        group = MagicMock()
        manager = _DeepEPManager(group, router_topk=2, num_experts=4, num_local_experts=4)
        indices = paddle.to_tensor([[0, 1], [2, 3], [0, 2], [1, 3]], dtype="int64")
        probs = paddle.ones([4, 2], dtype="float32") * 0.5
        multihot = manager._indices_to_multihot(indices, probs)
        # Should produce a multihot representation
        self.assertIsNotNone(multihot)


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestAllToAllTokenDispatcher(unittest.TestCase):
    """Test AllToAllTokenDispatcher construction and methods."""

    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.permute")
    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.unpermute")
    @patch("paddleformers.fleet.transformer.moe.token_dispatcher._AllToAll")
    def test_construction(self, mock_a2a, mock_unpermute, mock_permute):
        group = MagicMock()
        group.nranks = 2
        mock_permute.return_value = (
            paddle.randn([4, 64]),
            paddle.to_tensor([0, 1, 2, 3]),
        )
        mock_unpermute.return_value = paddle.randn([4, 64])

        dispatcher = AllToAllTokenDispatcher(
            group,
            expert_model_parallel_size=2,
            num_experts_per_device=2,
            local_expert_indices=[0, 1],
        )
        self.assertEqual(dispatcher.expert_model_parallel_size, 2)
        self.assertEqual(dispatcher.num_experts_per_device, 2)
        self.assertEqual(dispatcher.local_expert_indices, [0, 1])


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestMoEFlexTokenDispatcher(unittest.TestCase):
    """Test MoEFlexTokenDispatcher."""

    @patch(
        "paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch",
        MagicMock(),
    )
    @patch(
        "paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine",
        MagicMock(),
    )
    def test_construction(self):
        group = MagicMock()
        group.id = 0
        group.world_size = 2
        dispatcher = MoEFlexTokenDispatcher(
            num_local_experts=2,
            num_experts_per_tok=2,
            n_routed_experts=8,
            ep_group=group,
            moe_ep_barrier=True,
        )
        self.assertEqual(dispatcher.num_local_experts, 2)
        self.assertIsNotNone(dispatcher._comm_manager)
        self.assertEqual(dispatcher._comm_manager.router_topk, 2)
        self.assertEqual(dispatcher._comm_manager.num_experts, 8)


if __name__ == "__main__":
    unittest.main()
