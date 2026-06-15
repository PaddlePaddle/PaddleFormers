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
import functools
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import paddle

from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.transformer.moe import moe_layer
from paddleformers.fleet.transformer.moe.moe_layer import MoELayer


class FakeStream:
    cuda_stream = 123


class FakeCuda:
    @staticmethod
    def current_stream():
        return FakeStream()


class FakeGather:
    calls = []

    @staticmethod
    def apply(x):
        FakeGather.calls.append(x)
        return x + 1.0


class FakeScatter:
    calls = []

    @staticmethod
    def apply(x):
        FakeScatter.calls.append(x)
        return x + 2.0


class FakeCommManager:
    def __init__(self):
        self.dispatched_probs = paddle.ones([2, 2], dtype="float32")
        self.dispatched_indices = paddle.to_tensor([[0, 1], [1, 0]], dtype="int64")
        self.tokens_per_expert = paddle.to_tensor([2, 2], dtype="int64")
        self.combine_calls = []

    def combine(
        self,
        hidden_states,
        handle,
        async_finish=False,
        use_rr_deepep_combine=False,
    ):
        self.combine_calls.append((hidden_states, handle, async_finish, use_rr_deepep_combine))
        return hidden_states + 3.0


class FakeTokenDispatcher:
    def __init__(self):
        self._comm_manager = FakeCommManager()

    def token_dispatch_overlap(
        self,
        hidden_states,
        token_indices,
        token_weights,
        fp8_dispatch,
        async_finish=False,
        use_ue8m0=False,
    ):
        del token_indices, token_weights, fp8_dispatch, async_finish, use_ue8m0
        return hidden_states + 1.0, "fp8-handle"


class FakeWeightBox:
    def __init__(self):
        self.weight1 = paddle.ones([2, 4, 3], dtype="float32")
        self.weight2 = paddle.ones([2, 3, 4], dtype="float32")


class FakeUpProjection:
    calls = []

    @staticmethod
    def apply(*args, **kwargs):
        FakeUpProjection.calls.append((args, kwargs))
        x = args[0]
        return x + 10.0, x + 20.0


class FakeDownProjection:
    calls = []

    @staticmethod
    def apply(*args, **kwargs):
        FakeDownProjection.calls.append((args, kwargs))
        y1 = args[0]
        return y1 + 30.0


class FakeFusionMoePyLayer:
    calls = []

    @staticmethod
    def apply(*args, **kwargs):
        FakeFusionMoePyLayer.calls.append((args, kwargs))
        return args[0] + 40.0


class FakeLatentProj:
    def __init__(self, delta):
        self.delta = delta
        self.calls = []

    def __call__(self, x):
        self.calls.append(x)
        return x + self.delta


def build_gpt_model_with_moe():
    config = GPTConfig(
        num_hidden_layers=1,
        hidden_size=8,
        vocab_size=16,
        max_sequence_length=8,
        num_attention_heads=2,
        intermediate_size=16,
        n_routed_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=16,
        moe_layer_freq=1,
        moe_token_dispatcher_type="alltoall",
        moe_expert_fusion=False,
        moe_deep_gemm=False,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        tie_word_embeddings=False,
        use_qk_norm=True,
    )
    return gpt_builder(config, num_stages=1)


def get_gpt_moe_layer():
    for layer in build_gpt_model_with_moe().run_function:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, MoELayer):
            return mlp
    raise AssertionError("GPTModel did not create a MoELayer")


class TestMoELayerExtraExecutableBranchesNoMock(unittest.TestCase):
    def setUp(self):
        self.old_gather = moe_layer.GatherOp
        self.old_scatter = moe_layer.ScatterOp
        self.old_cuda = paddle.device.cuda
        self.old_activation_type = getattr(moe_layer, "ActivationType", None)
        self.old_up = getattr(moe_layer, "_UpProjection", None)
        self.old_down = getattr(moe_layer, "_DownProjection", None)
        self.old_count_cumsum = moe_layer.count_cumsum
        self.old_filter_scores = moe_layer.filter_scores
        self.old_metadata = moe_layer.fused_expert_parallel_TC_topk_router_metadata
        self.old_fusion = moe_layer.FusionMoePyLayer

    def tearDown(self):
        moe_layer.GatherOp = self.old_gather
        moe_layer.ScatterOp = self.old_scatter
        paddle.device.cuda = self.old_cuda
        if self.old_activation_type is None:
            if hasattr(moe_layer, "ActivationType"):
                delattr(moe_layer, "ActivationType")
        else:
            moe_layer.ActivationType = self.old_activation_type
        if self.old_up is None:
            if hasattr(moe_layer, "_UpProjection"):
                delattr(moe_layer, "_UpProjection")
        else:
            moe_layer._UpProjection = self.old_up
        if self.old_down is None:
            if hasattr(moe_layer, "_DownProjection"):
                delattr(moe_layer, "_DownProjection")
        else:
            moe_layer._DownProjection = self.old_down
        moe_layer.count_cumsum = self.old_count_cumsum
        moe_layer.filter_scores = self.old_filter_scores
        moe_layer.fused_expert_parallel_TC_topk_router_metadata = self.old_metadata
        moe_layer.FusionMoePyLayer = self.old_fusion

    def install_sonic_stubs(self):
        paddle.device.cuda = FakeCuda
        moe_layer.ActivationType = lambda name: name
        moe_layer._UpProjection = FakeUpProjection
        moe_layer._DownProjection = FakeDownProjection

        def fake_filter_scores(probs, indices):
            return probs + indices.astype("float32")

        def fake_count_cumsum(indices, expert_count, do_cumsum=True):
            del indices, do_cumsum
            freq = paddle.ones([expert_count], dtype="int64")
            offset = paddle.arange(expert_count + 1, dtype="int64")
            return freq, offset

        def fake_metadata(indices, offsets, k):
            flat_count = indices.shape[0] * k
            gather = paddle.arange(indices.shape[0], dtype="int64")
            scatter = paddle.arange(flat_count, dtype="int64")
            reverse = paddle.arange(flat_count, dtype="int64")
            active = paddle.arange(indices.shape[0] + 1, dtype="int64")
            return offsets, gather, scatter, reverse, active

        moe_layer.filter_scores = fake_filter_scores
        moe_layer.count_cumsum = fake_count_cumsum
        moe_layer.fused_expert_parallel_TC_topk_router_metadata = fake_metadata

    def test_compute_gate_uses_gather_for_single_card_sequence_parallel(self):
        moe_layer.GatherOp = FakeGather
        FakeGather.calls = []
        model = get_gpt_moe_layer()
        model.sequence_parallel = True

        class Gate(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.hidden = None
                self.input_ids = None

            def forward(self, hidden_states, input_ids=None):
                self.hidden = hidden_states
                self.input_ids = input_ids
                return "gate-result"

        model.gate = Gate()
        hidden = paddle.ones([2, 3], dtype="float32")
        input_ids = paddle.to_tensor([7, 8], dtype="int64")

        result = MoELayer.compute_gate(model, hidden, input_ids=input_ids)

        self.assertEqual(result, "gate-result")
        self.assertEqual(FakeGather.calls[0].numpy().tolist(), hidden.numpy().tolist())
        self.assertEqual(model.gate.hidden.numpy().tolist(), (hidden + 1.0).numpy().tolist())
        self.assertIs(model.gate.input_ids, input_ids)

    def test_aux_loss_compute_scatter_shared_and_latent_paths(self):
        moe_layer.ScatterOp = FakeScatter
        FakeScatter.calls = []
        model = get_gpt_moe_layer()
        model.use_latent_moe = True
        model.fc2_latent_proj = FakeLatentProj(1.0)
        model.training = True
        model.router_aux_loss_coef = 0.5
        model.shared_experts = lambda residuals: (residuals + 4.0,)
        model.expert_model_parallel_size = 1
        model.sequence_parallel = True
        hidden = paddle.ones([4, 2], dtype="float32")
        residuals = paddle.zeros([2, 2, 2], dtype="float32")

        out = MoELayer.aux_loss_compute(
            model,
            (
                hidden,
                paddle.to_tensor([0.5]),
                paddle.to_tensor([0.25]),
                residuals,
            ),
        )

        self.assertEqual(out.shape, [2, 2, 2])
        self.assertEqual(len(FakeScatter.calls), 1)
        self.assertTrue(float(out.sum()) > 0.0)

    def test_compute_dispatch_experts_and_combine_paths(self):
        model = get_gpt_moe_layer()
        model.moe_use_fusion_node = True
        model.use_hybrid_ep_backend = False
        model._use_hybrid_ep_fusion = lambda: False
        model.token_dispatcher = FakeTokenDispatcher()
        model.fp8_wgrad = False
        moe_layer.FusionMoePyLayer = FakeFusionMoePyLayer
        FakeFusionMoePyLayer.calls = []
        hidden = paddle.ones([2, 3], dtype="float32")
        indices = paddle.to_tensor([[0, 1], [1, 0]], dtype="int64")
        weights = paddle.ones([2, 2], dtype="float32")

        dispatched = MoELayer.compute_dispatch(model, (hidden, indices, weights), async_finish=True)
        expert_out = MoELayer.compute_experts(model, dispatched, is_first_fwd=True)
        combined = MoELayer.compute_combine(model, expert_out, async_finish=True)

        self.assertEqual(dispatched[1].shape, [2, 2])
        self.assertEqual(expert_out.numpy().tolist(), (hidden + 41.0).numpy().tolist())
        self.assertFalse(expert_out.stop_gradient)
        self.assertEqual(combined.numpy().tolist(), (expert_out + 3.0).numpy().tolist())

        dense_model = get_gpt_moe_layer()
        dense_model.moe_use_fusion_node = False
        dense_model.routed_experts_compute = lambda x: x + 5.0
        dense_model.combine = lambda x, *args, **kwargs: x + 6.0
        dense_expert = MoELayer.compute_experts(dense_model, (hidden, None))
        dense_combined = MoELayer.compute_combine(dense_model, dense_expert)
        self.assertEqual(dense_combined.numpy().tolist(), (hidden + 11.0).numpy().tolist())

    def test_fusion_moe_forward_sonic_branch_uses_projection_stubs(self):
        self.install_sonic_stubs()
        FakeUpProjection.calls = []
        FakeDownProjection.calls = []
        model = get_gpt_moe_layer()
        model.use_latent_moe = True
        model.fc1_latent_proj = FakeLatentProj(1.0)
        model.fc2_latent_proj = FakeLatentProj(2.0)
        model.use_hybrid_ep_backend = False
        model.moe_use_fusion_node = True
        model.using_sonic_moe = True
        model.grouped_gemm_experts = FakeWeightBox()
        model.token_dispatcher = FakeTokenDispatcher()
        model._use_hybrid_ep_fusion = lambda: False
        model.dispatch = lambda hidden_states, probs, routing_map, topk_weights, topk_indices: (
            hidden_states + 1.0,
            "handle",
        )
        hidden = paddle.ones([2, 3], dtype="float32")
        probs = paddle.ones([2, 2], dtype="float32")
        routing = paddle.ones([2, 2], dtype="bool")

        out = MoELayer.fusion_moe_forward(model, hidden, probs, routing, combine_overlap_handle=None)

        self.assertEqual(out.numpy().tolist(), (hidden + 47.0).numpy().tolist())
        self.assertEqual(len(FakeUpProjection.calls), 1)
        self.assertEqual(len(FakeDownProjection.calls), 1)

    def test_single_card_grouped_gemm_sonic_branch_uses_projection_stubs(self):
        self.install_sonic_stubs()
        FakeUpProjection.calls = []
        FakeDownProjection.calls = []
        model = get_gpt_moe_layer()
        model.using_sonic_moe = True
        model.grouped_gemm_experts = FakeWeightBox()
        hidden = paddle.ones([2, 3], dtype="float32")
        routing = paddle.to_tensor([[1, 1], [1, 1]], dtype="bool")
        probs = paddle.to_tensor([[0.7, 0.3], [0.4, 0.6]], dtype="float32")

        out = MoELayer._forward_single_card_grouped_gemm_moe(model, hidden, routing, probs)

        self.assertEqual(out.numpy().tolist(), (hidden + 40.0).numpy().tolist())
        self.assertEqual(len(FakeUpProjection.calls), 1)
        self.assertEqual(len(FakeDownProjection.calls), 1)

    def test_fp8_quant_weight_grouped_individual_mode_raises(self):
        model = get_gpt_moe_layer()
        model.moe_use_fusion_node = True
        model.fp8 = True
        model.grouped_gemm_experts = FakeWeightBox()
        model.use_ue8m0 = False

        with self.assertRaises(NotImplementedError):
            MoELayer.fp8_quant_weight(model, batch_mode=False)


if __name__ == "__main__":
    unittest.main()
