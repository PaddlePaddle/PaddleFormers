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

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.transformer import transformer_layer
from paddleformers.fleet.transformer.moe.moe_layer import MoELayer
from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerNode,
    TransformerLayerSublayersSpec,
    TransformerLayerWithOverlap,
)


class PGCollection:
    tp = None


class NonIdentityLayer(paddle.nn.Layer):
    def forward(self, x):
        return x + 1.0


class UnknownMLP(paddle.nn.Layer):
    def set_layer_number(self, layer_number, is_mtp_layer=False):
        self.layer_number = layer_number

    def forward(self, x):
        return x + 1.0, None


class FakeBDA(paddle.nn.Layer):
    def forward(self, training, bias_dropout_fusion):
        del training, bias_dropout_fusion

        def apply(output_with_bias, residual, hidden_dropout_prob):
            del hidden_dropout_prob
            if isinstance(output_with_bias, dict):
                value = output_with_bias.get("hidden_states", residual)
                bias = output_with_bias.get("bias", None)
            elif isinstance(output_with_bias, tuple):
                value, bias = output_with_bias
            else:
                value, bias = output_with_bias, None
            if bias is not None:
                value = value + bias
            return value + residual

        return apply


class TupleAttention(paddle.nn.Layer):
    def forward(self, x, **kwargs):
        self.kwargs = kwargs
        return x + 2.0, paddle.ones_like(x)


class ContextCrossAttention(paddle.nn.Layer):
    def forward(self, x, **kwargs):
        self.kwargs = kwargs
        return {"hidden_states": x + 3.0, "context": x + 4.0}


class FakeMoEForMlp(MoELayer):
    def __init__(self):
        paddle.nn.Layer.__init__(self)
        self.calls = []

    def forward(self, x, input_ids=None):
        self.calls.append(input_ids)
        return x + 2.0, paddle.ones_like(x)


class FakeOverlapMLP(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.dispatch_calls = []

    def compute_gate(self, hidden_states):
        return (
            None,
            hidden_states + 1.0,
            hidden_states + 2.0,
            None,
            None,
            None,
            paddle.to_tensor([0.25], dtype="float32"),
            paddle.to_tensor([0.5], dtype="float32"),
        )

    def dispatch_preprocess(self, args):
        self.dispatch_calls.append(args)
        hidden_states, topk_weights, topk_indices = args
        return hidden_states + 3.0, topk_indices + 1.0, topk_weights + 1.0


class InitConfig:
    def __init__(
        self, cp_comm_type, recompute_modules, recompute_method="block"
    ):
        self.gpt_model_use_experimental_version = False
        self.hidden_dropout_prob = 0.0
        self.sequence_parallel = False
        self.tensor_model_parallel_size = 1
        self.hidden_size = 2
        self.rms_norm_eps = 1e-5
        self.context_parallel_size = 2
        self.cp_comm_type = cp_comm_type
        self.recompute_granularity = "selective"
        self.recompute_modules = recompute_modules
        self.recompute_num_layers = 1
        self.recompute_method = recompute_method
        self.block_attention_residuals = False
        self.attn_res_block_size = 4
        self.num_empty_layers_add_in_head = 0
        self.num_hidden_layers = 2
        self.num_empty_layers_add_in_tail = 0
        self.virtual_pipeline_model_parallel_size = None
        self.pipeline_model_parallel_size = 1
        self.bias_dropout_fusion = False


class TinyConfig:
    num_nextn_predict_layers = 1
    mtp_load_weight_only = False


class DenseLayer:
    full_recompute = False
    mlp = object()

    def compute_attention(self, inputs, is_first_fwd=False):
        del is_first_fwd
        return inputs["hidden_states"] + 1.0, None

    def compute_mlp(self, hidden_states, is_first_fwd=False):
        del is_first_fwd
        return hidden_states + 2.0


class FakeScheduleNode:
    def __init__(self, fwd_func, name=""):
        self.fwd_func = fwd_func
        self.name = name

    def forward(self, inputs=(), **kwargs):
        return self.fwd_func(inputs, **kwargs)

    def backward(self, output_grad=None, scaler=None):
        del scaler
        if isinstance(output_grad, (list, tuple)):
            return (output_grad[0],)
        return (output_grad,)


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
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
        tie_word_embeddings=False,
        use_qk_norm=True,
    )
    return gpt_builder(config, num_stages=1)


def get_gpt_transformer_layer():
    for layer in build_gpt_model_with_moe().run_function:
        if isinstance(layer, TransformerLayer):
            return layer
    raise AssertionError("GPTModel did not create a TransformerLayer")


class TestTransformerLayerConstructorAndHelpers(unittest.TestCase):
    def setUp(self):
        self.old_build_spec_layer = transformer_layer.build_spec_layer
        self.old_recompute = transformer_layer.recompute
        self.old_log_layer_md5 = TransformerLayer._LOG_LAYER_MD5
        self.old_experimental = (
            TransformerLayer._gpt_model_use_experimental_version
        )

    def tearDown(self):
        transformer_layer.build_spec_layer = self.old_build_spec_layer
        transformer_layer.recompute = self.old_recompute
        TransformerLayer._LOG_LAYER_MD5 = self.old_log_layer_md5
        TransformerLayer._gpt_model_use_experimental_version = (
            self.old_experimental
        )

    def _spec(self):
        return TransformerLayerSublayersSpec(
            input_layernorm=NonIdentityLayer,
            self_attn=TupleAttention,
            self_attn_bda=FakeBDA,
            pre_cross_attn_layernorm=NonIdentityLayer,
            cross_attention=ContextCrossAttention,
            cross_attn_bda=FakeBDA,
            post_attention_layernorm=NonIdentityLayer,
            mlp=LayerSpec(UnknownMLP),
            mlp_bda=FakeBDA,
            block_attn_res=NonIdentityLayer,
        )

    def _install_build_stub(self):
        def build(spec, *args, **kwargs):
            del args
            if isinstance(spec, LayerSpec):
                layer = spec.layer()
            else:
                layer = spec()
            layer.init_kwargs = kwargs
            return layer

        transformer_layer.build_spec_layer = build

    def test_init_covers_cp_list_unknown_mlp_and_selective_list(self):
        self._install_build_stub()
        config = InitConfig(["zero", "one"], ["norm", "mlp"])

        layer = TransformerLayer(
            config, self._spec(), layer_number=0, pg_collection=PGCollection()
        )

        self.assertTrue(layer.recompute_input_layernorm)
        self.assertTrue(layer.recompute_post_attention_layernorm)
        self.assertTrue(layer.recompute_mlp)
        self.assertEqual(layer.self_attn.init_kwargs["cp_comm_type"], "zero")

    def test_init_covers_cp_string_and_selective_dict(self):
        self._install_build_stub()
        config = InitConfig("ring", {"norm": 1, "mlp": 1}, "first_n")

        layer = TransformerLayer(
            config, self._spec(), layer_number=0, pg_collection=PGCollection()
        )

        self.assertTrue(layer.recompute_input_layernorm)
        self.assertTrue(layer.recompute_post_attention_layernorm)
        self.assertTrue(layer.recompute_mlp)
        self.assertEqual(layer.self_attn.init_kwargs["cp_comm_type"], "ring")

    def test_forward_attention_context_and_block_residual_branches(self):
        model = get_gpt_transformer_layer()
        model.recompute_input_layernorm = False
        model.input_layernorm = NonIdentityLayer()
        model.self_attn = TupleAttention()
        model.pre_cross_attn_layernorm = NonIdentityLayer()
        model.cross_attention = ContextCrossAttention()
        model.cross_attn_bda = FakeBDA()
        model.hidden_dropout_prob = 0.0
        model.training = False
        model.config.bias_dropout_fusion = False
        model.layer_number = 5
        model._log_md5 = lambda *args, **kwargs: None
        hidden_states = paddle.ones([2, 2], dtype="float32")
        hidden_states.stop_gradient = False

        output, context = TransformerLayer._forward_attention(
            model,
            hidden_states,
            block_attention_residuals=True,
            is_first_fwd=True,
        )

        self.assertEqual(output.shape, [2, 2])
        self.assertEqual(context.shape, [2, 2])
        self.assertFalse(output.stop_gradient)

    def test_forward_mlp_recompute_md5_block_and_first_forward_branches(self):
        transformer_layer.recompute = lambda func, *args, **kwargs: func(
            *args, **kwargs
        )
        TransformerLayer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        model = get_gpt_transformer_layer()
        model.recompute_post_attention_layernorm = False
        model.post_attention_layernorm = NonIdentityLayer()
        model.mlp = FakeMoEForMlp()
        model.mlp_bda = FakeBDA()
        model.hidden_dropout_prob = 0.0
        model.training = False
        model.config.bias_dropout_fusion = False
        model.layer_number = 7
        model._log_md5 = lambda *args, **kwargs: None
        hidden_states = paddle.ones([2, 2], dtype="float32")
        hidden_states.stop_gradient = False
        model.recompute_mlp = True

        recomputed = TransformerLayer._forward_mlp(
            model,
            hidden_states,
            is_first_fwd=True,
            input_ids=paddle.to_tensor([[1, 2]], dtype="int64"),
        )

        model.recompute_mlp = False
        block_output = TransformerLayer._forward_mlp(
            model,
            hidden_states,
            is_first_fwd=True,
            block_attention_residuals=True,
        )

        self.assertEqual(recomputed.shape, [2, 2])
        self.assertEqual(block_output.shape, [2, 2])
        self.assertFalse(block_output.stop_gradient)

    def test_forward_impl_tuple_context_and_block_cast_branches(self):
        model = get_gpt_transformer_layer()
        model.config.block_attention_residuals = False
        model.layer_number = 1
        model.mlp = paddle.nn.Identity()
        model.full_recompute = False
        model._log_md5 = lambda *args, **kwargs: None
        model._forward_attention = lambda **kwargs: (
            kwargs["hidden_states"] + 1.0,
            kwargs["hidden_states"] + 2.0,
        )
        model._forward_mlp = lambda hidden_states, **kwargs: hidden_states + 3.0

        output, context = TransformerLayer._forward_impl(
            model, paddle.ones([1, 2], dtype="float32")
        )
        self.assertEqual(context.shape, [1, 2])

        forward_model = get_gpt_transformer_layer()
        forward_model.config.num_nextn_predict_layers = None
        forward_model.config.block_attention_residuals = False
        forward_model.full_recompute = False
        forward_model._forward_impl = lambda **kwargs: (
            kwargs["hidden_states"] + 1.0,
            kwargs["hidden_states"] + 2.0,
        )
        result = TransformerLayer.forward(
            forward_model,
            {"hidden_states": paddle.ones([1, 2], dtype="float32")},
        )
        self.assertIn("context", result)

        block_model = get_gpt_transformer_layer()
        block_model.config.block_attention_residuals = True
        block_model.layer_number = 1
        block_model.attn_res_block_size = 4
        block_model.mlp = paddle.nn.Identity()
        block_model.block_attn_res_before_attention = (
            lambda partial, blocks: partial
        )
        block_model.block_attn_res_before_mlp = lambda partial, blocks: partial
        block_model._forward_attention = lambda **kwargs: (
            kwargs["hidden_states"].cast("bfloat16"),
            None,
        )
        block_model._forward_mlp = (
            lambda hidden_states, **kwargs: hidden_states.cast("float32")
        )
        block_output = TransformerLayer._forward_impl(
            block_model, paddle.ones([1, 2], dtype="float32"), blocks=[]
        )
        self.assertEqual(block_output.shape, [1, 2])

    def test_overlap_helper_methods_drive_moe_paths(self):
        model = get_gpt_transformer_layer()
        model.mlp = FakeOverlapMLP()
        model.post_attention_layernorm = NonIdentityLayer()
        model.mlp_bda = FakeBDA()
        model.training = False
        model.hidden_dropout_prob = 0.0
        model.config.bias_dropout_fusion = False
        model._forward_attention = lambda **kwargs: (
            kwargs["hidden_states"] + 1.0,
            None,
        )
        model._forward_mlp = lambda hidden_states, **kwargs: hidden_states + 2.0
        hidden_states = paddle.ones([1, 2], dtype="float32")

        attn_out, _ = TransformerLayerWithOverlap.compute_attention(
            model, {"hidden_states": hidden_states}, is_first_fwd=True
        )
        mlp_out = TransformerLayerWithOverlap.compute_mlp(
            model, attn_out, is_first_fwd=True
        )
        preprocessed = TransformerLayerWithOverlap.pre_process_compute(
            model, hidden_states
        )
        dispatched = TransformerLayerWithOverlap.dispatch_preprocess_compute(
            model, (preprocessed[1], preprocessed[3], preprocessed[4])
        )
        post = TransformerLayerWithOverlap.post_process_compute(
            model, (mlp_out, hidden_states), is_first_fwd=True
        )

        self.assertEqual(dispatched[0].shape, [1, 2])
        self.assertFalse(post.stop_gradient)

    def test_dense_node_backward_restores_mtp_grads(self):
        original_schedule_node = transformer_layer.ScheduleNode
        transformer_layer.ScheduleNode = FakeScheduleNode
        try:
            node = TransformerLayerNode(DenseLayer(), TinyConfig())
            mtp_grad = paddle.full([1, 2], 5.0, dtype="float32")
            grads = node.backward(
                [paddle.ones([1, 2], dtype="float32"), mtp_grad]
            )
        finally:
            transformer_layer.ScheduleNode = original_schedule_node

        self.assertEqual(len(grads), 2)
        self.assertIs(grads[1], mtp_grad)


if __name__ == "__main__":
    unittest.main()
