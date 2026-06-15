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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import paddle

from paddleformers.fleet.transformer import transformer_layer
from paddleformers.fleet.transformer.moe.moe_layer import MoELayer
from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayerNode,
    TransformerLayerOverlappedScheduleNode,
)


class TinyConfig:
    def __init__(self, nextn=None, weight_only=False):
        self.num_nextn_predict_layers = nextn
        self.mtp_load_weight_only = weight_only


class RealDenseLayer:
    full_recompute = False
    mlp = object()

    def __init__(self, with_context=False):
        self.with_context = with_context
        self.attn_flags = []
        self.mlp_flags = []

    def compute_attention(self, inputs, is_first_fwd=False):
        self.attn_flags.append(is_first_fwd)
        hidden_states = inputs["hidden_states"] * 2.0
        context = None
        if self.with_context:
            context = hidden_states + 1.0
            context.stop_gradient = True
        return hidden_states, context

    def compute_mlp(self, hidden_states, is_first_fwd=False):
        self.mlp_flags.append(is_first_fwd)
        return hidden_states * 3.0


class FakeScheduleNode:
    def __init__(self, fwd_func, name=""):
        self.fwd_func = fwd_func
        self.name = name
        self.forward_calls = []
        self.backward_calls = []

    def forward(self, inputs=(), **kwargs):
        self.forward_calls.append((inputs, kwargs))
        return self.fwd_func(inputs, **kwargs)

    def backward(self, output_grad=None, scaler=None):
        del scaler
        self.backward_calls.append(output_grad)
        grad = self._first_tensor(output_grad)
        if self.name == "post_process_compute":
            return grad, grad
        if self.name == "aux_loss_compute":
            return grad, grad, grad, grad
        if self.name in ("dispatch_compute", "dispatch_preprocess_compute"):
            return grad, grad, grad
        return grad

    def _first_tensor(self, value):
        if isinstance(value, (list, tuple)):
            return self._first_tensor(value[0])
        return value


class FakeEvent:
    def __init__(self, calls):
        self.calls = calls

    def calc_stream_wait(self, group_id):
        self.calls.append(("calc_stream_wait", group_id))


class FakeDeepEP:
    def __init__(self, calls):
        self.calls = calls

    def get_event_from_comm_stream(self, group_id):
        self.calls.append(("get_event", group_id))
        return FakeEvent(self.calls)


class FakeGroup:
    id = 23


class FakeCommManager:
    def __init__(self):
        self.group = FakeGroup()


class FakeTokenDispatcher:
    def __init__(self):
        self._comm_manager = FakeCommManager()


class FakeMoE(MoELayer):
    def __init__(self):
        self.token_dispatcher = FakeTokenDispatcher()

    def compute_gate(self, hidden_states):
        return (
            hidden_states,
            hidden_states + 1.0,
            hidden_states + 2.0,
            hidden_states + 3.0,
            hidden_states + 4.0,
            hidden_states + 5.0,
            hidden_states + 6.0,
            hidden_states + 7.0,
        )

    def compute_dispatch(self, args, async_finish=False):
        hidden_states, token_indices, token_weights = args
        del async_finish, token_indices, token_weights
        return hidden_states + 2.0

    def compute_experts(self, hidden_states, is_first_fwd=False):
        del is_first_fwd
        return hidden_states * 2.0

    def compute_combine(
        self,
        hidden_states,
        async_finish=False,
        use_rr_deepep_combine=False,
    ):
        del async_finish, use_rr_deepep_combine
        return hidden_states + 3.0

    def aux_loss_compute(self, args):
        hidden_states, aux_loss, z_loss, residuals = args
        del aux_loss, z_loss, residuals
        return hidden_states + 4.0


class SparseLayer:
    def __init__(self, full_recompute=False):
        self.full_recompute = full_recompute
        self.mlp = FakeMoE()

    def compute_attention(self, inputs, is_first_fwd=False):
        hidden_states = inputs["hidden_states"]
        if is_first_fwd:
            hidden_states = hidden_states + 1.0
        return hidden_states * 2.0, None

    def pre_process_compute(self, hidden_states):
        return (
            hidden_states + 1.0,
            hidden_states + 2.0,
            hidden_states + 3.0,
            hidden_states + 4.0,
            hidden_states + 5.0,
            hidden_states + 6.0,
            hidden_states + 7.0,
        )

    def dispatch_preprocess_compute(self, args):
        hidden_states, topk_weights, topk_indices = args
        return hidden_states + 1.0, topk_indices, topk_weights

    def post_process_compute(self, args, is_first_fwd=False):
        mlp_output, residual = args
        output = mlp_output + residual
        if is_first_fwd:
            output.stop_gradient = False
        return output


class TestTransformerLayerDenseSchedule(unittest.TestCase):
    def test_dense_forward_backward_uses_real_schedule_nodes(self):
        layer = RealDenseLayer()
        node = TransformerLayerNode(layer, TinyConfig(), name="dense")
        hidden_states = paddle.ones([2, 2], dtype="float32")
        hidden_states.stop_gradient = False

        result = node.forward({"hidden_states": hidden_states})
        grads = node.backward(paddle.ones([2, 2], dtype="float32"))

        self.assertEqual(result["hidden_states"].numpy().tolist(), [[6.0, 6.0], [6.0, 6.0]])
        self.assertEqual(grads[0].numpy().tolist(), [[6.0, 6.0], [6.0, 6.0]])
        self.assertEqual(layer.attn_flags, [False])
        self.assertEqual(layer.mlp_flags, [False])

    def test_dense_forward_preserves_optional_context(self):
        node = TransformerLayerNode(RealDenseLayer(with_context=True), TinyConfig())
        result = node.forward({"hidden_states": paddle.ones([1, 2], dtype="float32")})

        self.assertIn("context", result)
        self.assertEqual(result["context"].numpy().tolist(), [[3.0, 3.0]])

    def test_dense_full_recompute_runs_cached_forwards(self):
        original_schedule_node = transformer_layer.ScheduleNode
        transformer_layer.ScheduleNode = FakeScheduleNode
        try:
            layer = RealDenseLayer()
            layer.full_recompute = True
            node = TransformerLayerNode(layer, TinyConfig())
            result = node.forward({"hidden_states": paddle.ones([1, 1], dtype="float32")})
            grads = node.backward(paddle.ones([1, 1], dtype="float32"))
        finally:
            transformer_layer.ScheduleNode = original_schedule_node

        self.assertIn("hidden_states", result)
        self.assertEqual(grads.shape, [1, 1])
        self.assertEqual(len(node.attn_node.forward_calls), 2)
        self.assertEqual(len(node.mlp_node.forward_calls), 2)


class TestTransformerLayerSparseSchedule(unittest.TestCase):
    def setUp(self):
        self.original_schedule_node = transformer_layer.ScheduleNode
        self.original_deep_ep = getattr(transformer_layer, "deep_ep", None)
        self.calls = []
        transformer_layer.ScheduleNode = FakeScheduleNode
        transformer_layer.deep_ep = FakeDeepEP(self.calls)

    def tearDown(self):
        transformer_layer.ScheduleNode = self.original_schedule_node
        if self.original_deep_ep is None:
            delattr(transformer_layer, "deep_ep")
        else:
            transformer_layer.deep_ep = self.original_deep_ep

    def test_sparse_forward_backward_waits_for_dispatch_and_combine(self):
        node = TransformerLayerNode(SparseLayer(full_recompute=True), TinyConfig())
        hidden_states = paddle.ones([1, 2], dtype="float32")

        result = node.forward({"hidden_states": hidden_states})
        grads = node.backward(paddle.ones([1, 2], dtype="float32"))

        self.assertIn("hidden_states", result)
        self.assertEqual(grads.shape, [1, 2])
        self.assertGreaterEqual(
            len([call for call in self.calls if call[0] == "calc_stream_wait"]),
            2,
        )

    def test_sparse_overlapped_forward_backward_interleaves_events(self):
        forward_node = TransformerLayerNode(SparseLayer(), TinyConfig())
        backward_node = TransformerLayerNode(SparseLayer(full_recompute=True), TinyConfig())
        backward_node.attn_recompute_args = {"hidden_states": paddle.ones([1, 2], dtype="float32")}
        backward_node.mlp_recompute_args = paddle.ones([1, 2], dtype="float32")
        backward_node.post_process_recompute_args = (
            paddle.ones([1, 2], dtype="float32"),
            paddle.ones([1, 2], dtype="float32"),
        )
        overlapped = TransformerLayerOverlappedScheduleNode(forward_node, backward_node)

        result, grads = overlapped.forward_backward(
            {"hidden_states": paddle.ones([1, 2], dtype="float32")},
            paddle.ones([1, 2], dtype="float32"),
        )

        self.assertIn("hidden_states", result)
        self.assertEqual(grads.shape, [1, 2])
        self.assertGreaterEqual(len([call for call in self.calls if call[0] == "get_event"]), 4)

    def test_overlapped_fallback_restores_mtp_inputs_and_grads(self):
        forward_node = TransformerLayerNode(RealDenseLayer(), TinyConfig())
        backward_node = TransformerLayerNode(RealDenseLayer(), TinyConfig())
        config = TinyConfig(nextn=1, weight_only=False)
        forward_node.config = config
        backward_node.config = config

        def forward_func(inputs):
            return {"hidden_states": inputs["hidden_states"] + 1.0}

        def backward_func(output_grad):
            del output_grad
            return (paddle.ones([1], dtype="float32"),)

        forward_node.forward = forward_func
        backward_node.backward = backward_func
        overlapped = TransformerLayerOverlappedScheduleNode(forward_node, backward_node)
        mtp_grad = paddle.full([1], 5.0, dtype="float32")

        result, grads = overlapped.forward_backward(
            {
                "hidden_states": paddle.ones([1], dtype="float32"),
                "decoder_input_0": paddle.full([1], 3.0, dtype="float32"),
            },
            [paddle.ones([1], dtype="float32"), mtp_grad],
        )

        self.assertIn("decoder_input_0", result)
        self.assertEqual(len(grads), 2)
        self.assertIs(grads[1], mtp_grad)


if __name__ == "__main__":
    unittest.main()
