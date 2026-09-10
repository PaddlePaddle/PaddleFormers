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

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock

import paddle

# Direct import to avoid __init__.py triggering workflow.py which requires AutoTokenizer
_MODULE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "paddleformers", "cli", "train", "deepseek_v3_pretrain"
)
_MODULE_DIR = os.path.abspath(_MODULE_DIR)


def _load_module(name, path):
    """Load a module directly from file path without going through __init__.py."""
    full_name = f"paddleformers.cli.train.deepseek_v3_pretrain.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-populate the package in sys.modules to prevent __init__.py from loading
_pkg_name = "paddleformers.cli.train.deepseek_v3_pretrain"
if _pkg_name not in sys.modules:
    _pkg_mod = types.ModuleType(_pkg_name)
    _pkg_mod.__path__ = [_MODULE_DIR]
    _pkg_mod.__package__ = _pkg_name
    sys.modules[_pkg_name] = _pkg_mod

# Pre-load moe_utils (dependency of token_dispatcher)
_moe_utils_mod = _load_module("moe_utils", os.path.join(_MODULE_DIR, "moe_utils.py"))

# Load the token_dispatcher module directly
_td_mod = _load_module("token_dispatcher", os.path.join(_MODULE_DIR, "token_dispatcher.py"))
MoETokenDispatcher = _td_mod.MoETokenDispatcher
PreDispatchNode = _td_mod.PreDispatchNode


class TestMoETokenDispatcher(unittest.TestCase):
    """Test MoETokenDispatcher base class."""

    def _create_dispatcher(self):
        """Create a basic MoETokenDispatcher for testing."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        return MoETokenDispatcher(ep_group=mock_group)

    def test_ep_group_property(self):
        """Test ep_group property."""
        dispatcher = self._create_dispatcher()
        self.assertIsNotNone(dispatcher.ep_group)

    def test_ep_size_property(self):
        """Test ep_size property."""
        dispatcher = self._create_dispatcher()
        self.assertEqual(dispatcher.ep_size, 1)

    def test_token_permutation_not_implemented(self):
        """Test that token_permutation raises NotImplementedError."""
        dispatcher = self._create_dispatcher()
        with self.assertRaises(NotImplementedError):
            dispatcher.token_permutation(
                tokens=paddle.randn([4, 8]),
                probs=paddle.randn([4, 4]),
                routing_map=paddle.randn([4, 4]),
            )

    def test_token_unpermutation_not_implemented(self):
        """Test that token_unpermutation raises NotImplementedError."""
        dispatcher = self._create_dispatcher()
        with self.assertRaises(NotImplementedError):
            dispatcher.token_unpermutation(
                expert_output=paddle.randn([4, 8]),
                bias=None,
            )


class TestPreDispatchNode(unittest.TestCase):
    """Test PreDispatchNode class."""

    def _create_node(self, num_experts=8, router_topk=2):
        """Create a PreDispatchNode with mock token dispatcher."""
        mock_dispatcher = MagicMock()
        mock_dispatcher._comm_manager = MagicMock()
        mock_dispatcher._comm_manager.num_experts = num_experts
        mock_dispatcher._comm_manager.router_topk = router_topk
        return PreDispatchNode(mock_dispatcher)

    def test_init(self):
        """Test PreDispatchNode initialization."""
        node = self._create_node()
        self.assertIsNotNone(node.token_dispatcher)
        self.assertIsNone(node.probs_origin_shape)

    def test_reset_status(self):
        """Test PreDispatchNode reset_status method."""
        node = self._create_node()
        node.probs = paddle.randn([4, 8])
        node.reshaped_probs = paddle.randn([4, 8])
        node.token_indices = paddle.to_tensor([[0, 1]])

        node.reset_status()
        self.assertIsNone(node.probs)
        self.assertIsNone(node.reshaped_probs)
        self.assertIsNone(node.token_indices)

    def test_forward_basic(self):
        """Test PreDispatchNode forward with basic inputs."""
        node = self._create_node(num_experts=8, router_topk=2)

        routing_map = paddle.zeros([4, 8])
        probs = paddle.randn([4, 8]).abs()

        token_indices, token_probs = node.forward(routing_map, probs)
        self.assertEqual(token_indices.shape, [4, 2])
        self.assertEqual(token_probs.shape, [4, 2])

    def test_forward_stores_state(self):
        """Test that forward stores internal state."""
        node = self._create_node(num_experts=8, router_topk=2)

        routing_map = paddle.zeros([4, 8])
        probs = paddle.randn([4, 8]).abs()

        node.forward(routing_map, probs)
        self.assertIsNotNone(node.probs)
        self.assertIsNotNone(node.reshaped_probs)
        self.assertIsNotNone(node.token_indices)
        self.assertEqual(node.probs_origin_shape, [4, 8])

    def test_forward_token_probs_gradient_enabled(self):
        """Test that token_probs has gradient enabled after forward."""
        node = self._create_node(num_experts=8, router_topk=2)

        routing_map = paddle.zeros([4, 8])
        probs = paddle.randn([4, 8]).abs()

        _, token_probs = node.forward(routing_map, probs)
        self.assertFalse(token_probs.stop_gradient)


class TestDeepepManagerIndicesToMultihot(unittest.TestCase):
    """Test _DeepepManager._indices_to_multihot method."""

    def test_basic_conversion(self):
        """Test basic indices to multihot conversion."""
        try:
            from paddleformers.transformers.fused_a2a import fused_dispatch

            if fused_dispatch is None:
                self.skipTest("fused_dispatch is not available")
        except ImportError:
            self.skipTest("fused_a2a not available")

        _DeepepManager = _td_mod._DeepepManager
        manager = _DeepepManager.__new__(_DeepepManager)
        manager.num_experts = 8
        manager.num_local_experts = 4
        manager.router_topk = 2

        indices = paddle.to_tensor([[0, 1], [1, 2], [0, -1]])
        probs = paddle.to_tensor([[0.5, 0.3], [0.4, 0.6], [0.7, 0.0]])

        routing_map, multihot_probs = manager._indices_to_multihot(indices, probs)
        self.assertEqual(routing_map.shape, [3, 4])
        self.assertEqual(multihot_probs.shape, [3, 4])
        self.assertEqual(routing_map.dtype, paddle.bool)

    def test_all_masked_indices(self):
        """Test with all indices masked (all -1)."""
        try:
            from paddleformers.transformers.fused_a2a import fused_dispatch

            if fused_dispatch is None:
                self.skipTest("fused_dispatch is not available")
        except ImportError:
            self.skipTest("fused_a2a not available")

        _DeepepManager = _td_mod._DeepepManager
        manager = _DeepepManager.__new__(_DeepepManager)
        manager.num_experts = 8
        manager.num_local_experts = 4
        manager.router_topk = 2

        indices = paddle.to_tensor([[-1, -1]])
        probs = paddle.to_tensor([[0.0, 0.0]])

        routing_map, multihot_probs = manager._indices_to_multihot(indices, probs)
        self.assertEqual(routing_map.shape, [1, 4])
        self.assertFalse(routing_map.any())


if __name__ == "__main__":
    unittest.main()
