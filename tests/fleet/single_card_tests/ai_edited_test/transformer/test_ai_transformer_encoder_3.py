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

import paddle

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from paddle import nn
from paddle.distributed.fleet.meta_parallel import (
    ScheduleChunk,
    ScheduleNode,
    SharedLayerDesc,
)

from paddleformers.fleet.transformer import transformer_encoder
from paddleformers.fleet.transformer.transformer_encoder import (
    TransformerEncoder,
)
from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayerNode,
)


class Value:
    def __init__(self, name=""):
        self.key = name
        self.global_expert_id_offset = None
        self.layer_cnt = None


class Config:
    def __init__(self, model_type=""):
        self.model_type = model_type


class LightweightEncoder(TransformerEncoder):
    def __init__(self, keys, modal=None, model_type=""):
        self.modal = modal
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
            result["2.experts.1.weight"].global_expert_id_offset = 4
        if "2.layer_norm.weight" in result:
            result["2.layer_norm.weight"].layer_cnt = 7
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


class ForwardNode(ScheduleNode):
    def __init__(self, delta):
        super().__init__(lambda inputs: inputs, name="forward_node")
        self.delta = delta

    def forward(self, inputs):
        return {"hidden_states": inputs["hidden_states"] + self.delta}


class BackwardNode(ScheduleNode):
    def __init__(self, delta):
        super().__init__(lambda inputs: inputs, name="backward_node")
        self.delta = delta

    def backward(self, grads):
        return grads[0] + self.delta


class TinyConfig:
    num_nextn_predict_layers = None
    mtp_load_weight_only = False


class OverlapNode(TransformerLayerNode):
    def __init__(self, delta):
        self.delta = delta
        self.config = TinyConfig()
        self._is_sparse = False
        self.full_recompute = False

    def forward(self, inputs):
        return {"hidden_states": inputs["hidden_states"] + self.delta}

    def backward(self, grads):
        return grads + self.delta


class LossNode:
    def __init__(self):
        self.backward_scalers = []
        self.forward_inputs = []

    def forward(self, inputs):
        self.forward_inputs.append(inputs)
        if isinstance(inputs, tuple):
            return inputs[0].sum()
        return inputs["hidden_states"].sum()

    def backward(self, scaler=None):
        self.backward_scalers.append(scaler)
        return (paddle.ones([1], dtype="float32"),)


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
        self.calls.append(("backward_async", value.shape))


class TestTransformerEncoderPipelineMappingNoMock(unittest.TestCase):
    def setUp(self):
        self.original_super = getattr(transformer_encoder, "super", None)

    def tearDown(self):
        if self.original_super is None:
            if hasattr(transformer_encoder, "super"):
                delattr(transformer_encoder, "super")
        else:
            transformer_encoder.super = self.original_super

    def _install_parent(self, model):
        transformer_encoder.super = lambda: ParentMethods(
            model._parent_state_dict,
            model._parent_set_state_dict,
            model._parent_sharded_state_dict,
        )

    def test_overlapped_forward_backward_with_loss_and_p2p_handle(self):
        model = LightweightEncoder([])
        forward_loss = LossNode()
        backward_loss = LossNode()
        handle = P2PHandle()

        forward_inputs, loss, grads = model.overlapped_forward_backward(
            ScheduleChunk(
                [ForwardNode(1.0), OverlapNode(2.0), ForwardNode(3.0)]
            ),
            {"hidden_states": paddle.ones([1], dtype="float32")},
            forward_loss,
            ScheduleChunk(
                [BackwardNode(5.0), OverlapNode(4.0), BackwardNode(6.0)]
            ),
            backward_loss,
            None,
            scaler="scale",
            p2p_async_handle=handle,
        )

        self.assertEqual(loss.numpy().item(), 7.0)
        self.assertTrue(isinstance(forward_inputs, tuple))
        self.assertEqual(forward_inputs[0].numpy().tolist(), [7.0])
        self.assertEqual(grads.numpy().item(), 16.0)
        self.assertEqual(backward_loss.backward_scalers, ["scale"])
        self.assertEqual(handle.calls[0:2], ["forward_wait", "backward_wait"])
        self.assertEqual(handle.calls[2][0], "forward_async")

    def test_set_pipeline_mapping_handles_virtual_and_shared_names(self):
        shared = SharedLayerDesc(
            "embed", DummyEmbedding, shared_weight_attr="embedding_weight"
        )
        model = LightweightEncoder(
            [
                "0.0.weight",
                "0.tail.weight",
                "shared_layers.embed.weight",
                "plain.bias",
            ]
        )
        model.layers = [shared]
        model._sequential_layers = [
            {"layer": shared, "name_prefix": "model.embed"},
            {"layer": object(), "name_prefix": "model.layers.1"},
        ]
        self._install_parent(model)

        mapping = model._set_pipeline_name_mapping()

        self.assertEqual(
            mapping["model.embed.weight"], "shared_layers.embed.weight"
        )
        self.assertEqual(mapping["model.layers.1.weight"], "0.tail.weight")
        self.assertEqual(mapping["plain.bias"], "plain.bias")
        self.assertEqual(
            model._pp_to_single_mapping["0.0.weight"], "model.embed.weight"
        )

    def test_shared_layer_prefix_requires_current_stage(self):
        shared = SharedLayerDesc(
            "embed", DummyEmbedding, shared_weight_attr="embedding_weight"
        )
        model = LightweightEncoder([])
        model.layers = [shared]
        model._sequential_layers = [
            {"layer": shared, "name_prefix": "model.embed"}
        ]
        model._stage_id = 0
        model._stage_for_index = 1

        with self.assertRaises(ValueError):
            model.get_shardlayer_prefix(["shared_layers", "embed", "weight"])

    def test_state_and_set_state_dict_apply_bidirectional_mapping(self):
        model = LightweightEncoder(
            ["0.weight", "extra.bias"],
            model_type="qwen3_5",
        )
        model._sequential_layers = [
            {"layer": object(), "name_prefix": "model.layers.0"}
        ]
        self._install_parent(model)

        state = model.state_dict()
        self.assertIn("model.layers.0.weight", state)
        self.assertIn("extra.bias", state)
        self.assertEqual(
            state["model.layers.0.weight"].key, "model.layers.0.weight"
        )

        result = model.set_state_dict(
            {"model.layers.0.weight": Value("single"), "ignored": Value("x")}
        )
        self.assertEqual(result, "loaded")
        self.assertEqual(list(model.loaded_state.keys()), ["0.weight"])

    def test_check_shared_model_state_detects_duplicate_tensor_identity(self):
        model = LightweightEncoder(["0.weight"])
        value = Value("shared")
        model._pipeline_name_mapping = {
            "model.layers.0.weight": "0.weight",
            "model.shared.weight": "0.weight",
        }
        model._pp_to_single_mapping = {
            "0.weight": "model.layers.0.weight",
            "shared_layers.embed.weight": "model.shared.weight",
        }
        transformer_encoder.super = lambda: ParentMethods(
            lambda: {"0.weight": value, "shared_layers.embed.weight": value},
            model._parent_set_state_dict,
            model._parent_sharded_state_dict,
        )

        missing = model._check_shared_model_state()

        self.assertEqual(missing, {"shared_layers.embed.weight": "0.weight"})

    def test_sharded_state_dict_remaps_and_renames_expert_and_layer_keys(self):
        model = LightweightEncoder(
            ["0.weight", "2.experts.1.weight", "2.layer_norm.weight"],
            model_type="qwen3_vl",
        )
        model._sequential_layers = [
            {"layer": object(), "name_prefix": "model.layers.0"},
            {"layer": object(), "name_prefix": "model.layers.1"},
            {"layer": object(), "name_prefix": "model.layers.2"},
        ]
        self._install_parent(model)

        sharded = model.sharded_state_dict()

        self.assertIn("model.layers.0.weight", sharded)
        self.assertIn("model.layers.2.experts.5.weight", sharded)
        self.assertIn("model.layers.2.layer_norm.weight_layer_7", sharded)
        self.assertFalse(
            hasattr(
                sharded["model.layers.2.experts.5.weight"],
                "global_expert_id_offset",
            )
        )
        self.assertFalse(
            hasattr(
                sharded["model.layers.2.layer_norm.weight_layer_7"], "layer_cnt"
            )
        )


if __name__ == "__main__":
    unittest.main()
