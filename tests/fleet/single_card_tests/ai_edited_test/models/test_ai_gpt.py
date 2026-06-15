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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from paddle import nn
from paddle.distributed.fleet.meta_parallel import SharedLayerDesc

from paddleformers.fleet.models.gpt import gpt_model


class Config:
    def __init__(
        self,
        model_type="",
        experimental=False,
        nextn_layers=0,
    ):
        self.model_type = model_type
        self.gpt_model_use_experimental_version = experimental
        self.num_nextn_predict_layers = nextn_layers
        self.enable_mtp_magic_send = False


class LightweightGPTModel(gpt_model.GPTModel):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        self._sequential_layers = []
        self._pipeline_name_mapping = None
        self.layers = []
        self._stage_id = 0
        self._stage_for_index = 0
        self._values = []

    def get_stage_from_index(self, idx):
        del idx
        return self._stage_for_index

    def state_dict(self):
        return {str(i): value for i, value in enumerate(self._values)}


class Place:
    def __init__(self, is_gpu):
        self._is_gpu = is_gpu

    def is_gpu_place(self):
        return self._is_gpu


class Param:
    def __init__(self, is_weight_only_mtp=False, is_gpu=True):
        self.is_weight_only_mtp = is_weight_only_mtp
        self.place = Place(is_gpu)
        self.pin_memory_called = False
        self.cuda_called = False
        self.shared_to = None

    def pin_memory(self):
        self.pin_memory_called = True
        return self

    def cuda(self):
        self.cuda_called = True
        return self

    def _share_buffer_to(self, other):
        self.shared_to = other


class DummyLayer(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        del args, kwargs


class DummyEmbedding(DummyLayer):
    pass


class DummyHeadEmpty(DummyLayer):
    pass


class DummyTransformer(DummyLayer):
    pass


class DummyTailEmpty(DummyLayer):
    pass


class DummyMTP(DummyLayer):
    pass


class DummyLayerNorm(DummyLayer):
    pass


class DummyLMHead(DummyLayer):
    pass


class DummyMTPLMHead(DummyLayer):
    pass


class DummyMTPLoss(DummyLayer):
    pass


class TestGPTModelDescriptorUtilities(unittest.TestCase):
    def _spec(self):
        return gpt_model.GPTSublayersSpec(
            embedding=DummyEmbedding,
            head_empty_layers=[DummyHeadEmpty],
            transformer_layers=[DummyTransformer],
            tail_empty_layers=[DummyTailEmpty],
            mtp=[],
            layer_norm=DummyLayerNorm,
            lm_head=DummyLMHead,
            mtp_lm_head=None,
            mtp_loss=None,
        )

    def test_sublayers_spec_defaults_to_none(self):
        spec = gpt_model.GPTSublayersSpec()

        self.assertIsNone(spec.embedding)
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.transformer_layers)
        self.assertIsNone(spec.tail_empty_layers)
        self.assertIsNone(spec.mtp)
        self.assertIsNone(spec.layer_norm)
        self.assertIsNone(spec.lm_head)
        self.assertIsNone(spec.mtp_lm_head)
        self.assertIsNone(spec.mtp_loss)

    def test_get_layer_desc_list_uses_default_prefixes_and_lm_head(self):
        model = LightweightGPTModel(Config())
        layers = model.get_layer_desc_list(self._spec(), tie_word_embeddings=False)

        self.assertEqual(
            [layer["name_prefix"] for layer in layers],
            [
                "model",
                "model.layers.0",
                "model.layers.1",
                "model",
                "model.layers.2",
                "model.lm_head",
            ],
        )
        self.assertFalse(isinstance(layers[0]["layer"], SharedLayerDesc))
        self.assertFalse(isinstance(layers[-1]["layer"], SharedLayerDesc))

    def test_get_layer_desc_list_handles_qwen_shared_and_mtp_layout(self):
        spec = self._spec()
        spec.mtp = [DummyMTP]
        spec.mtp_lm_head = DummyMTPLMHead
        spec.mtp_loss = DummyMTPLoss
        model = LightweightGPTModel(
            Config(
                model_type="qwen3_vl",
                experimental=True,
                nextn_layers=1,
            )
        )

        layers = model.get_layer_desc_list(spec, tie_word_embeddings=True)

        self.assertEqual(
            [layer["name_prefix"] for layer in layers],
            [
                "model.language_model",
                "model.language_model.layers.0",
                "model.language_model.layers.1",
                "model.language_model.layers.2",
                "model.language_model.shared_mtp_lm_head",
                "model.language_model.mtp_loss",
                "model.language_model.layers.3",
                "model.language_model",
                "model.language_model.shared_head",
            ],
        )
        self.assertTrue(isinstance(layers[0]["layer"], SharedLayerDesc))
        self.assertTrue(isinstance(layers[4]["layer"], SharedLayerDesc))
        self.assertTrue(isinstance(layers[-1]["layer"], SharedLayerDesc))

    def test_sequential_layer_helpers_and_explicit_mapping(self):
        model = LightweightGPTModel(Config())
        layers = []
        first = object()
        second = object()
        model.add_sequential_layer(layers, first, "first.prefix")
        model.add_sequential_layer(layers, second, "second.prefix")
        model._sequential_layers = layers

        mapping = {"single": "pipeline"}

        self.assertEqual(model.get_sequential_layers(), [first, second])
        self.assertEqual(
            model.get_sequential_name_prefixes(),
            {"0": "first.prefix", "1": "second.prefix"},
        )
        self.assertIs(model._set_pipeline_name_mapping(mapping), mapping)
        self.assertIs(model._pipeline_name_mapping, mapping)


class TestGPTModelWeightOnlyAndShardPrefix(unittest.TestCase):
    def test_weight_only_params_filter_and_offload_reload(self):
        gpu_param = Param(is_weight_only_mtp=True, is_gpu=True)
        cpu_param = Param(is_weight_only_mtp=True, is_gpu=False)
        ignored = Param(is_weight_only_mtp=False, is_gpu=True)
        model = LightweightGPTModel(Config())
        model._values = [gpu_param, cpu_param, ignored]

        self.assertEqual(model._get_weight_only_params(), [gpu_param, cpu_param])

        model.offload_weight_only_params()
        self.assertTrue(gpu_param.pin_memory_called)
        self.assertIs(gpu_param.shared_to, gpu_param)
        self.assertFalse(cpu_param.pin_memory_called)

        model.reload_weight_only_params()
        self.assertFalse(gpu_param.cuda_called)
        self.assertTrue(cpu_param.cuda_called)
        self.assertIs(cpu_param.shared_to, cpu_param)

    def test_get_shardlayer_prefix_success_assertion_and_stage_error(self):
        shared = SharedLayerDesc("embed", DummyEmbedding, shared_weight_attr="embedding_weight")
        model = LightweightGPTModel(Config())
        model.layers = [shared]
        model._sequential_layers = [{"layer": shared, "name_prefix": "model.embed"}]

        self.assertEqual(
            model.get_shardlayer_prefix(["shared_layers", "embed", "weight"]),
            "model.embed",
        )

        with self.assertRaises(AssertionError):
            model.get_shardlayer_prefix(["shared_layers", "missing", "weight"])

        model._stage_for_index = 1
        with self.assertRaises(ValueError):
            model.get_shardlayer_prefix(["shared_layers", "embed", "weight"])


if __name__ == "__main__":
    unittest.main()
