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

import unittest
from unittest.mock import MagicMock, patch

import paddle

try:
    from paddleformers.fleet.transformer.moe.token_dispatcher import (
        AllToAllTokenDispatcher,
        MoETokenDispatcher,
        _DeepepManager,
        _DispatchManager,
    )

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestDispatchManagerAbstract(unittest.TestCase):
    """Tests for _DispatchManager abstract class."""

    def test_abstract_methods(self):
        """Test that _DispatchManager cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            _DispatchManager()

    def test_concrete_subclass_must_implement(self):
        """Test that concrete subclass must implement all abstract methods."""

        class IncompleteManager(_DispatchManager):
            pass

        with self.assertRaises(TypeError):
            IncompleteManager()


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestDeepepManagerConstruction(unittest.TestCase):
    """Tests for _DeepepManager construction."""

    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch", True)
    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine", True)
    def test_basic_construction(self):
        """Test basic _DeepepManager construction."""
        mock_group = MagicMock()
        manager = _DeepepManager(
            group=mock_group,
            router_topk=2,
            num_experts=8,
        )
        self.assertEqual(manager.router_topk, 2)
        self.assertEqual(manager.num_experts, 8)

    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch", True)
    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine", True)
    def test_construction_with_local_experts(self):
        """Test _DeepepManager with num_local_experts."""
        mock_group = MagicMock()
        manager = _DeepepManager(
            group=mock_group,
            router_topk=2,
            num_local_experts=4,
        )
        self.assertEqual(manager.num_local_experts, 4)

    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch", True)
    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine", True)
    def test_default_moe_ep_barrier(self):
        """Test default moe_ep_barrier value."""
        mock_group = MagicMock()
        manager = _DeepepManager(
            group=mock_group,
            router_topk=2,
        )
        self.assertTrue(manager.moe_ep_barrier)

    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch", True)
    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine", True)
    def test_metadata_initially_none(self):
        """Test that metadata is None initially."""
        mock_group = MagicMock()
        manager = _DeepepManager(
            group=mock_group,
            router_topk=2,
        )
        self.assertIsNone(manager.token_indices)
        self.assertIsNone(manager.token_probs)
        self.assertIsNone(manager.handle)


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestMoETokenDispatcherProperties(unittest.TestCase):
    """Tests for MoETokenDispatcher properties."""

    def test_ep_group_property(self):
        """Test ep_group property."""
        mock_ep = MagicMock()
        dispatcher = MoETokenDispatcher(mock_ep)
        self.assertEqual(dispatcher.ep_group, mock_ep)

    def test_ep_size_property(self):
        """Test ep_size property."""
        mock_ep = MagicMock()
        mock_ep.world_size = 4
        dispatcher = MoETokenDispatcher(mock_ep)
        self.assertEqual(dispatcher.ep_size, 4)

    def test_token_permutation_not_implemented(self):
        """Test token_permutation raises NotImplementedError."""
        mock_ep = MagicMock()
        dispatcher = MoETokenDispatcher(mock_ep)
        with self.assertRaises(NotImplementedError):
            dispatcher.token_permutation(
                paddle.randn([4, 8]), paddle.randn([4, 2]), paddle.randn([4, 2])
            )

    def test_token_unpermutation_not_implemented(self):
        """Test token_unpermutation raises NotImplementedError."""
        mock_ep = MagicMock()
        dispatcher = MoETokenDispatcher(mock_ep)
        with self.assertRaises(NotImplementedError):
            dispatcher.token_unpermutation(paddle.randn([4, 8]))


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestAllToAllTokenDispatcher(unittest.TestCase):
    """Tests for AllToAllTokenDispatcher."""

    def test_construction(self):
        """Test basic construction."""
        mock_group = MagicMock()
        dispatcher = AllToAllTokenDispatcher(
            moe_group=mock_group,
            expert_model_parallel_size=2,
            num_experts_per_device=2,
            local_expert_indices=[0, 1],
        )
        self.assertEqual(dispatcher.num_local_experts, 2)
        self.assertEqual(dispatcher.expert_model_parallel_size, 2)


if __name__ == "__main__":
    unittest.main()
