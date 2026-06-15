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
    from paddleformers.fleet.transformer.moe.moe_utils import AddAuxiliaryLoss
    from paddleformers.fleet.transformer.moe.token_dispatcher import (
        MoETokenDispatcher,
        _DeepepManager,
    )

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestDeepepManagerSetupMetadata(unittest.TestCase):
    """Tests for _DeepepManager setup_metadata."""

    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch", True)
    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine", True)
    def test_setup_metadata_with_topk_weights(self):
        """Test setup_metadata with pre-computed topk_weights and topk_indices."""
        manager = _DeepepManager(
            group=MagicMock(),
            router_topk=2,
            num_experts=4,
        )

        routing_map = paddle.zeros([3, 4])
        probs = paddle.randn([3, 4])
        topk_weights = paddle.randn([3, 2])
        topk_indices = paddle.randint(0, 4, [3, 2])

        manager.setup_metadata(routing_map, probs, topk_weights, topk_indices)

        # Should use provided topk_weights/indices directly
        self.assertIsNotNone(manager.token_probs)
        self.assertIsNotNone(manager.token_indices)

    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch", True)
    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine", True)
    def test_setup_metadata_without_topk(self):
        """Test setup_metadata without pre-computed topk data."""
        manager = _DeepepManager(
            group=MagicMock(),
            router_topk=2,
            num_experts=4,
        )

        routing_map = paddle.zeros([3, 4])
        probs = paddle.rand([3, 4])

        manager.setup_metadata(routing_map, probs)
        self.assertIsNotNone(manager.token_probs)
        self.assertIsNotNone(manager.token_indices)
        self.assertEqual(manager.token_probs.shape, [3, 2])
        self.assertEqual(manager.token_indices.shape, [3, 2])


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestDeepepManagerGetNumberofTokens(unittest.TestCase):
    """Tests for _DeepepManager get_number_of_tokens_per_expert."""

    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch", True)
    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine", True)
    def test_get_number_of_tokens_per_expert_requires_dispatch(self):
        """Test get_number_of_tokens_per_expert needs dispatch to set tokens_per_expert."""
        manager = _DeepepManager(
            group=MagicMock(),
            router_topk=2,
            num_experts=4,
        )

        # tokens_per_expert is only set after dispatch(), not after setup_metadata
        self.assertFalse(hasattr(manager, "tokens_per_expert"))


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestDeepepManagerGetDispatchedMetadata(unittest.TestCase):
    """Tests for _DeepepManager get_dispatched_metadata."""

    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_dispatch", True)
    @patch("paddleformers.fleet.transformer.moe.token_dispatcher.fused_combine", True)
    def test_get_dispatched_metadata_requires_dispatch(self):
        """Test get_dispatched_metadata needs dispatch to set dispatched_indices."""
        manager = _DeepepManager(
            group=MagicMock(),
            router_topk=2,
            num_experts=4,
        )

        # dispatched_indices is only set after dispatch(), not after setup_metadata
        self.assertFalse(hasattr(manager, "dispatched_indices"))


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestMoETokenDispatcherProperties(unittest.TestCase):
    """Tests for MoETokenDispatcher properties."""

    def test_ep_group_and_size(self):
        """Test ep_group and ep_size properties."""
        mock_ep = MagicMock()
        mock_ep.world_size = 8
        dispatcher = MoETokenDispatcher(mock_ep)
        self.assertEqual(dispatcher.ep_group, mock_ep)
        self.assertEqual(dispatcher.ep_size, 8)


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddleformers.fleet.transformer.moe.token_dispatcher not available",
)
class TestAddAuxiliaryLossScalar(unittest.TestCase):
    """Tests for AddAuxiliaryLoss with scalar loss."""

    def test_scalar_loss_required(self):
        """Test that loss must be scalar (numel==1)."""
        x = paddle.randn([4, 8])
        loss = paddle.to_tensor([0.5])  # scalar
        loss.stop_gradient = False
        out = AddAuxiliaryLoss.apply(x, loss)
        self.assertEqual(out.shape, x.shape)

    def test_non_scalar_loss_raises(self):
        """Test that non-scalar loss raises assertion."""
        x = paddle.randn([4, 8])
        loss = paddle.to_tensor([0.5, 0.3])  # non-scalar
        loss.stop_gradient = False
        with self.assertRaises(AssertionError):
            AddAuxiliaryLoss.apply(x, loss)


if __name__ == "__main__":
    unittest.main()
