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

import paddle
import paddle.nn as nn


class TestMoECommunicationInterface(unittest.TestCase):
    """Tests for MoECommunicationInterface ABC."""

    def test_import(self):
        """Test that MoECommunicationInterface can be imported."""
        from paddleformers.nn.moe_deepep.moe_communication import (
            MoECommunicationInterface,
        )

        self.assertIsNotNone(MoECommunicationInterface)

    def test_is_abstract(self):
        """Test that MoECommunicationInterface is an ABC."""
        from abc import ABC

        from paddleformers.nn.moe_deepep.moe_communication import (
            MoECommunicationInterface,
        )

        self.assertTrue(issubclass(MoECommunicationInterface, ABC))

    def test_has_forward_method(self):
        """Test that MoECommunicationInterface defines a forward method."""
        from paddleformers.nn.moe_deepep.moe_communication import (
            MoECommunicationInterface,
        )

        self.assertTrue(hasattr(MoECommunicationInterface, "forward"))


class TestAllToAllMoECommunication(unittest.TestCase):
    """Tests for AllToAllMoECommunication class."""

    def _get_cls(self):
        from paddleformers.nn.moe_deepep.moe_communication import (
            AllToAllMoECommunication,
        )

        return AllToAllMoECommunication

    def test_import(self):
        """Test that AllToAllMoECommunication can be imported."""
        cls = self._get_cls()
        self.assertIsNotNone(cls)

    def test_is_nn_layer(self):
        """Test that AllToAllMoECommunication is a subclass of nn.Layer."""
        cls = self._get_cls()
        self.assertTrue(issubclass(cls, nn.Layer))

    def test_instantiation(self):
        """Test that AllToAllMoECommunication can be instantiated."""
        cls = self._get_cls()
        instance = cls()
        self.assertIsInstance(instance, nn.Layer)

    def test_forward_single_rank_returns_input(self):
        """Test forward returns input when expert_model_parallel_size <= 1."""
        cls = self._get_cls()
        comm = cls()
        x = paddle.randn([4, 8])
        result = comm.forward(
            hidden_states=x,
            topk_indices=paddle.to_tensor([[0, 1]]),
            topk_weights=paddle.randn([4, 2]),
            gates_masked=paddle.randn([4, 8]),
            mask=paddle.ones([4, 8]),
            priorities=paddle.ones([4, 2]),
            expert_model_parallel_size=1,
            moe_group=None,
            experts=nn.LayerList([]),
            moe_rank=0,
            num_experts_per_device=8,
            num_experts=8,
            topk=2,
            token_dispatcher=None,
        )
        # The returned value should be the same tensor
        self.assertTrue(paddle.allclose(result, x).item())


class TestDeepEPMoECommunication(unittest.TestCase):
    """Tests for DeepEPMoECommunication class."""

    def _get_cls(self):
        from paddleformers.nn.moe_deepep.moe_communication import DeepEPMoECommunication

        return DeepEPMoECommunication

    def test_import(self):
        """Test that DeepEPMoECommunication can be imported."""
        cls = self._get_cls()
        self.assertIsNotNone(cls)

    def test_is_nn_layer(self):
        """Test that DeepEPMoECommunication is a subclass of nn.Layer."""
        cls = self._get_cls()
        self.assertTrue(issubclass(cls, nn.Layer))

    def test_instantiation(self):
        """Test that DeepEPMoECommunication can be instantiated."""
        cls = self._get_cls()
        instance = cls()
        self.assertIsInstance(instance, nn.Layer)

    def test_forward_single_rank_returns_input(self):
        """Test forward returns input when expert_model_parallel_size <= 1."""
        cls = self._get_cls()
        comm = cls()
        x = paddle.randn([4, 8])
        result = comm.forward(
            hidden_states=x,
            topk_indices=paddle.to_tensor([[0, 1]]),
            topk_weights=paddle.randn([4, 2]),
            gates_masked=paddle.randn([4, 8]),
            mask=paddle.ones([4, 8]),
            priorities=paddle.ones([4, 2]),
            expert_model_parallel_size=1,
            moe_group=None,
            experts=nn.LayerList([]),
            moe_rank=0,
            num_experts_per_device=8,
            num_experts=8,
            topk=2,
            token_dispatcher=None,
        )
        self.assertTrue(paddle.allclose(result, x).item())

    def test_expert_forward_with_experts(self):
        """Test expert_forward method dispatches tokens to experts."""
        cls = self._get_cls()
        comm = cls()

        class DummyExpert(nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

            def forward(self, x):
                return self.linear(x)

        experts = nn.LayerList([DummyExpert(), DummyExpert()])
        dispatched_input = paddle.randn([4, 8])
        tokens_per_expert = [2, 2]

        result = comm.expert_forward(
            dispatched_input, tokens_per_expert, experts, moe_rank=0, num_experts_per_device=2
        )
        self.assertEqual(result.shape[1], 8)

    def test_expert_forward_empty_outputs(self):
        """Test expert_forward when all tokens_per_expert are zero returns input."""
        cls = self._get_cls()
        comm = cls()

        class DummyExpert(nn.Layer):
            def forward(self, x):
                return x

        experts = nn.LayerList([DummyExpert()])
        dispatched_input = paddle.randn([0, 8])
        tokens_per_expert = [0]

        result = comm.expert_forward(
            dispatched_input, tokens_per_expert, experts, moe_rank=0, num_experts_per_device=1
        )
        # When no expert has tokens, the outputs list is empty, so input is returned
        self.assertEqual(result.shape[0], 0)

    def test_expert_forward_with_list_tokens(self):
        """Test expert_forward when tokens_per_expert is a list."""
        cls = self._get_cls()
        comm = cls()

        class DummyExpert(nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

            def forward(self, x):
                return self.linear(x)

        experts = nn.LayerList([DummyExpert(), DummyExpert()])
        dispatched_input = paddle.randn([6, 8])
        tokens_per_expert = [3, 3]

        result = comm.expert_forward(
            dispatched_input, tokens_per_expert, experts, moe_rank=0, num_experts_per_device=2
        )
        self.assertEqual(result.shape[0], 6)
        self.assertEqual(result.shape[1], 8)


if __name__ == "__main__":
    unittest.main()
