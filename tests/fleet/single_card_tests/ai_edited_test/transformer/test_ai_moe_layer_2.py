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
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import paddle

from paddleformers.fleet.transformer.moe.moe_layer import MoELayer


class MinimalMoE:
    pass


class Expert:
    def __init__(self, offset):
        self.offset = offset
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x)
        return x + self.offset, None


class Gate:
    def __init__(self):
        self.calls = []
        self.layer_number = None

    def __call__(self, hidden_states, input_ids=None):
        self.calls.append((hidden_states, input_ids))
        return "gate-output"

    def set_layer_number(self, layer_number, is_mtp_layer=False):
        self.layer_number = layer_number


class Combiner:
    def __init__(self):
        self.calls = []

    def combine(
        self,
        hidden_states,
        handle,
        async_finish=False,
        use_rr_deepep_combine=False,
    ):
        self.calls.append(
            (hidden_states, handle, async_finish, use_rr_deepep_combine)
        )
        return hidden_states + 3


class Dispatcher:
    def __init__(self):
        self._comm_manager = Combiner()


class SharedExpert:
    def __call__(self, residuals):
        return residuals + 5, None


class TestMoELayerLightweightMethods(unittest.TestCase):
    def test_expert_forward_concatenates_non_empty_expert_outputs(self):
        model = MinimalMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 3
        model.experts = [Expert(1), Expert(10), Expert(100)]
        dispatched_input = paddle.arange(8, dtype="float32").reshape([4, 2])

        output = MoELayer.expert_forward(
            model, dispatched_input, paddle.to_tensor([1, 0, 3], dtype="int64")
        )

        self.assertEqual(
            output.numpy().tolist(),
            [[1.0, 2.0], [102.0, 103.0], [104.0, 105.0], [106.0, 107.0]],
        )
        self.assertEqual(model.experts[0].inputs[0].shape, [1, 2])
        self.assertEqual(model.experts[2].inputs[0].shape, [3, 2])
        self.assertEqual(model.experts[1].inputs, [])

    def test_expert_forward_returns_input_when_all_sections_are_empty(self):
        model = MinimalMoE()
        model.moe_rank = 0
        model.num_experts_per_device = 1
        model.experts = [Expert(1)]
        dispatched_input = paddle.empty([0, 2], dtype="float32")

        output = MoELayer.expert_forward(model, dispatched_input, [0])

        self.assertIs(output, dispatched_input)
        self.assertEqual(model.experts[0].inputs, [])

    def test_compute_gate_and_hybrid_fusion_predicate(self):
        model = MinimalMoE()
        model.expert_model_parallel_size = 2
        model.sequence_parallel = True
        model.gate = Gate()
        model.moe_use_fusion_node = True
        model.use_hybrid_ep_backend = False
        hidden_states = paddle.ones([2, 2], dtype="float32")
        input_ids = paddle.ones([2], dtype="int64")

        self.assertEqual(
            MoELayer.compute_gate(model, hidden_states, input_ids=input_ids),
            "gate-output",
        )
        self.assertIs(model.gate.calls[0][0], hidden_states)
        self.assertIs(model.gate.calls[0][1], input_ids)
        self.assertFalse(MoELayer._use_hybrid_ep_fusion(model))

        model.use_hybrid_ep_backend = True
        self.assertTrue(MoELayer._use_hybrid_ep_fusion(model))

    def test_compute_combine_uses_fusion_or_regular_path(self):
        model = MinimalMoE()
        hidden_states = paddle.ones([2], dtype="float32")
        model.moe_use_fusion_node = True
        model.token_dispatcher = Dispatcher()

        output = MoELayer.compute_combine(
            model, hidden_states, async_finish=True
        )

        self.assertEqual(output.numpy().tolist(), [4.0, 4.0])
        self.assertEqual(model.token_dispatcher._comm_manager.calls[0][2], True)

        model.moe_use_fusion_node = False
        model.combine = lambda value, *args, **kwargs: value + 7
        output = MoELayer.compute_combine(model, hidden_states)
        self.assertEqual(output.numpy().tolist(), [8.0, 8.0])

    def test_aux_loss_compute_reshapes_and_adds_shared_expert(self):
        model = MinimalMoE()
        model.use_latent_moe = False
        model.training = False
        model.router_aux_loss_coef = 0.0
        model.shared_experts = SharedExpert()
        model.expert_model_parallel_size = 2
        model.sequence_parallel = True
        hidden_states = paddle.ones([4, 2], dtype="float32")
        residuals = paddle.zeros([2, 2, 2], dtype="float32")

        output = MoELayer.aux_loss_compute(
            model, (hidden_states, paddle.to_tensor([1.0]), None, residuals)
        )

        self.assertEqual(output.shape, [2, 2, 2])
        self.assertEqual(output.numpy().tolist()[0][0], [6.0, 6.0])

    def test_use_fp8_and_set_layer_number(self):
        model = MinimalMoE()
        model.moe_use_fusion_node = False
        model.fp8 = True
        self.assertFalse(MoELayer.use_fp8(model))

        model.moe_use_fusion_node = True
        self.assertTrue(MoELayer.use_fp8(model))

        model.gate = Gate()
        MoELayer.set_layer_number(model, 11)
        self.assertEqual(model.layer_number, 11)
        self.assertEqual(model.gate.layer_number, 11)

        model.gate = object()
        with self.assertRaises(AssertionError):
            MoELayer.set_layer_number(model, 12)


if __name__ == "__main__":
    unittest.main()
