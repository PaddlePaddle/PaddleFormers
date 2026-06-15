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
from paddle import nn
from paddle.distributed.fleet.meta_parallel import (
    ScheduleChunk,
    ScheduleNode,
    SharedLayerDesc,
)

from paddleformers.fleet.models.gpt import gpt_model
from paddleformers.fleet.models.gpt.gpt_model import GPTModel
from paddleformers.fleet.transformer.transformer_layer import TransformerLayerNode


class Value:
    def __init__(self, name=""):
        self.key = name
        self.global_expert_id_offset = None
        self.layer_cnt = None


class Config:
    def __init__(self, model_type=""):
        self.model_type = model_type


class LightweightGPT(GPTModel):
    def __init__(self, keys, model_type=""):
        self.config = Config(model_type)
        self._keys = keys
        self._sequential_layers = []
        self._pipeline_name_mapping = None
        self.layers = []
        self._stage_id = 0
        self._stage_for_index = 0
        self.loaded_state = None

    def get_stage_from_index(self, idx):
        del idx
        return self._stage_for_index

    def _parent_state_dict(self, *args, **kwargs):
        del args, kwargs
        return {key: Value(key) for key in self._keys}

    def _parent_set_state_dict(self, state_dict, *args, **kwargs):
        del args, kwargs
        self.loaded_state = dict(state_dict)
        return "loaded"

    def _parent_sharded_state_dict(self, *args, **kwargs):
        del args, kwargs
        result = {key: Value(key) for key in self._keys}
        if "2.experts.1.weight" in result:
            result["2.experts.1.weight"].global_expert_id_offset = 3
        if "2.layer_norm.weight" in result:
            result["2.layer_norm.weight"].layer_cnt = 9
        return result


class ParentMethods:
    def __init__(self, state_dict_func, set_state_dict_func, sharded_func):
        self._state_dict_func = state_dict_func
        self._set_state_dict_func = set_state_dict_func
        self._sharded_func = sharded_func

    def state_dict(self, *args, **kwargs):
        return self._state_dict_func(*args, **kwargs)

    def set_state_dict(self, *args, **kwargs):
        return self._set_state_dict_func(*args, **kwargs)

    def sharded_state_dict(self, *args, **kwargs):
        return self._sharded_func(*args, **kwargs)


class DummyEmbedding(nn.Layer):
    def forward(self, *args, **kwargs):
        del args, kwargs


class TinyConfig:
    num_nextn_predict_layers = None
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


class LossNode:
    def __init__(self):
        self.forward_inputs = []
        self.backward_scalers = []

    def forward(self, inputs):
        self.forward_inputs.append(inputs)
        return inputs[0].sum() if isinstance(inputs, tuple) else inputs["hidden_states"].sum()

    def backward(self, scaler=None):
        self.backward_scalers.append(scaler)
        return (paddle.ones([1, 2], dtype="float32"),)


class P2PHandle:
    def __init__(self):
        self.calls = []

    def forward_handle_wait(self):
        self.calls.append("forward_wait")

    def backward_handle_wait(self):
        self.calls.append("backward_wait")

    def forward_async_comm(self, value):
        self.calls.append(("forward_async", isinstance(value, tuple)))

    def backward_async_comm(self, value):
        self.calls.append(("backward_async", value))


class QuantLayer(gpt_model.TransformerLayer):
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.quant_calls = []

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        self.quant_calls.append((batch_mode, quant_transpose))

    def use_fp8(self):
        return self.enabled


class MTPWrapper(gpt_model.MultiTokenPredictionLayer):
    def __init__(self, layer):
        object.__setattr__(self, "transformer_layer", layer)


class FakeScheduleNode:
    def __init__(self, fwd_func, name=""):
        self.fwd_func = fwd_func
        self.name = name
        self.outputs = None

    def forward(self, inputs=(), **kwargs):
        self.outputs = self.fwd_func(inputs, **kwargs)
        return self.outputs

    def backward(self, output_grad=None, scaler=None):
        del scaler
        if output_grad is None:
            return (paddle.ones([1, 2], dtype="float32"),)
        if isinstance(output_grad, (tuple, list)):
            return tuple(output_grad)
        return (output_grad,)


class TestGPTOverlapAndStateNoMock(unittest.TestCase):
    def setUp(self):
        self.original_super = getattr(gpt_model, "super", None)
        self.original_schedule_node = gpt_model.TransformerLayerNode.__init__.__globals__["ScheduleNode"]

    def tearDown(self):
        if self.original_super is None:
            if hasattr(gpt_model, "super"):
                delattr(gpt_model, "super")
        else:
            gpt_model.super = self.original_super
        gpt_model.TransformerLayerNode.__init__.__globals__["ScheduleNode"] = self.original_schedule_node

    def _install_parent(self, model):
        gpt_model.super = lambda: ParentMethods(
            model._parent_state_dict,
            model._parent_set_state_dict,
            model._parent_sharded_state_dict,
        )

    def test_build_overlapped_nodes_splits_asymmetric_chunks(self):
        pre = ScheduleNode(lambda inputs: inputs, name="pre")
        post = ScheduleNode(lambda inputs: inputs, name="post")
        first = TransformerLayerNode(DenseLayer(), TinyConfig())
        second = TransformerLayerNode(DenseLayer(), TinyConfig())
        backward = TransformerLayerNode(DenseLayer(), TinyConfig())

        parts = gpt_model.build_overlapped_nodes(
            ScheduleChunk([pre, first, second, post]),
            ScheduleChunk([post, backward, pre]),
        )

        self.assertEqual([len(part.nodes) for part in parts], [1, 1, 1, 2, 1])

    def test_overlapped_forward_backward_drives_loss_and_p2p_handle(self):
        model = LightweightGPT([])
        gpt_model.TransformerLayerNode.__init__.__globals__["ScheduleNode"] = FakeScheduleNode
        forward_chunk = ScheduleChunk([TransformerLayerNode(DenseLayer(), TinyConfig())])
        backward_chunk = ScheduleChunk([TransformerLayerNode(DenseLayer(), TinyConfig())])
        forward_loss = LossNode()
        backward_loss = LossNode()
        handle = P2PHandle()
        hidden_states = paddle.ones([1, 2], dtype="float32")
        hidden_states.stop_gradient = False

        forward_inputs, loss, backward_grads = model.overlapped_forward_backward(
            forward_chunk,
            {"hidden_states": hidden_states},
            forward_loss,
            backward_chunk,
            backward_loss,
            None,
            scaler="scale",
            p2p_async_handle=handle,
        )

        self.assertIsNotNone(loss)
        self.assertTrue(isinstance(forward_inputs, tuple))
        self.assertEqual(forward_inputs[0].shape, [1, 2])
        self.assertEqual(backward_grads[0].shape, [1, 2])
        self.assertEqual(backward_loss.backward_scalers, ["scale"])
        self.assertEqual(handle.calls[0:2], ["forward_wait", "backward_wait"])
        self.assertEqual(handle.calls[2][0], "forward_async")

    def test_overlapped_forward_backward_without_scaler_calls_plain_backward(
        self,
    ):
        model = LightweightGPT([])
        gpt_model.TransformerLayerNode.__init__.__globals__["ScheduleNode"] = FakeScheduleNode
        forward_chunk = ScheduleChunk([TransformerLayerNode(DenseLayer(), TinyConfig())])
        backward_chunk = ScheduleChunk([TransformerLayerNode(DenseLayer(), TinyConfig())])
        backward_loss = LossNode()
        hidden_states = paddle.ones([1, 2], dtype="float32")
        hidden_states.stop_gradient = False

        _, loss, backward_grads = model.overlapped_forward_backward(
            forward_chunk,
            {"hidden_states": hidden_states},
            None,
            backward_chunk,
            backward_loss,
            None,
            scaler=None,
            p2p_async_handle=None,
        )

        self.assertIsNone(loss)
        self.assertEqual(backward_loss.backward_scalers, [None])
        self.assertEqual(backward_grads[0].shape, [1, 2])

    def test_state_dict_set_state_dict_and_sharded_renaming(self):
        model = LightweightGPT(
            ["0.weight", "2.experts.1.weight", "2.layer_norm.weight"],
            model_type="qwen3_vl",
        )
        model._sequential_layers = [
            {"layer": object(), "name_prefix": "model.language_model.layers.0"},
            {"layer": object(), "name_prefix": "model.language_model.layers.1"},
            {"layer": object(), "name_prefix": "model.language_model.layers.2"},
        ]
        self._install_parent(model)

        state = model.state_dict()
        self.assertIn("model.language_model.layers.0.weight", state)
        result = model.set_state_dict({"model.language_model.layers.0.weight": Value("single")})
        sharded = model.sharded_state_dict()

        self.assertEqual(result, "loaded")
        self.assertEqual(list(model.loaded_state.keys()), ["0.weight"])
        self.assertIn("model.language_model.layers.2.experts.4.weight", sharded)
        self.assertIn("model.language_model.layers.2.layer_norm.weight_layer_9", sharded)

    def test_pipeline_mapping_handles_shared_and_virtual_names(self):
        shared = SharedLayerDesc("embed", DummyEmbedding, shared_weight_attr="embedding_weight")
        model = LightweightGPT(["0.0.weight", "0.tail.weight", "shared_layers.embed.weight"])
        model.layers = [shared]
        model._sequential_layers = [
            {"layer": shared, "name_prefix": "model.embed"},
            {"layer": object(), "name_prefix": "model.layers.1"},
        ]
        self._install_parent(model)

        mapping = model._set_pipeline_name_mapping()

        self.assertEqual(mapping["model.embed.weight"], "shared_layers.embed.weight")
        self.assertEqual(mapping["model.layers.1.weight"], "0.tail.weight")

    def test_shared_layer_prefix_requires_current_stage(self):
        shared = SharedLayerDesc("embed", DummyEmbedding, shared_weight_attr="embedding_weight")
        model = LightweightGPT([])
        model.layers = [shared]
        model._sequential_layers = [{"layer": shared, "name_prefix": "model.embed"}]
        model._stage_id = 0
        model._stage_for_index = 1

        with self.assertRaises(ValueError):
            model.get_shardlayer_prefix(["shared_layers", "embed", "weight"])

    def test_fp8_quant_weight_and_use_fp8_paths(self):
        model = LightweightGPT([])
        layer = QuantLayer(enabled=True)
        mtp_layer = MTPWrapper(QuantLayer(enabled=False))
        model._num_virtual_pipeline_stages = 1
        model.run_function = [object(), layer, mtp_layer]

        model.fp8_quant_weight(batch_mode=True, quant_transpose=False)
        self.assertEqual(layer.quant_calls, [(True, False)])
        self.assertEqual(mtp_layer.transformer_layer.quant_calls, [(True, False)])
        self.assertTrue(model.use_fp8())

        model._num_virtual_pipeline_stages = 2
        vpp_layer = QuantLayer(enabled=True)
        model._model_chunks = [[object()], [vpp_layer]]
        self.assertTrue(model.use_fp8())
        model.fp8_quant_weight(batch_mode=False, quant_transpose=True)
        self.assertEqual(vpp_layer.quant_calls, [(False, True)])


if __name__ == "__main__":
    unittest.main()
