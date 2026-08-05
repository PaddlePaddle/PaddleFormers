# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import LayerDesc, SharedLayerDesc

from paddleformers.transformers import Qwen3ForCausalLM
from paddleformers.transformers.model_utils import (
    PipelinePretrainedModel,
    PretrainedModel,
)
from paddleformers.utils.env import CONFIG_NAME, PADDLE_WEIGHTS_NAME
from tests.testing_utils import slow


def download_qwen3_model(model_name: str):
    """set the global method: multiprocessing can not pickle local method

    Args:
        model_name (str): the model name
    """

    model = Qwen3ForCausalLM.from_pretrained(model_name)
    # free the model resource
    del model


class TestModeling(unittest.TestCase):
    """Test PretrainedModel single time, not in Transformer models"""

    @slow
    def test_from_pretrained_cache_dir_community_model(self):
        model_name = "PaddleFormers/tiny-random-qwen3"
        with TemporaryDirectory() as tempdir:
            Qwen3ForCausalLM.from_pretrained(model_name, cache_dir=tempdir, convert_from_hf=True)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, CONFIG_NAME)))
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, PADDLE_WEIGHTS_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))

    @slow
    def test_from_pretrained_cache_dir_pretrained_init(self):
        model_name = "PaddleFormers/tiny-random-qwen3"
        with TemporaryDirectory() as tempdir:
            Qwen3ForCausalLM.from_pretrained(model_name, cache_dir=tempdir, convert_from_hf=True)
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, CONFIG_NAME)))
            self.assertTrue(os.path.exists(os.path.join(tempdir, model_name, PADDLE_WEIGHTS_NAME)))
            # check against double appending model_name in cache_dir
            self.assertFalse(os.path.exists(os.path.join(tempdir, model_name, model_name)))


class TestVirtualPipelineNameMapping(unittest.TestCase):
    """Test _set_pipeline_name_mapping for virtual pipeline parallelism.

    Under VPP, layers directly added to the PipelineLayer (e.g. lm_head) are named
    `{global_idx}.rest` while layers inside a chunk are named `{chunk_start}.{local_idx}.rest`.
    Both forms must map back to their single card names without collision.
    """

    # 4 layers split into 2 chunks of 2 layers, lm_head is the last layer.
    PREFIXES = {
        "0": "model.embed_tokens",
        "1": "model.layers.0",
        "2": "model.layers.1",
        "3": "model.lm_head",
    }

    def _build_mapping(self, pp_keys, layers_desc=(), stage_id=0, index_to_stage=None):
        model = PipelinePretrainedModel.__new__(PipelinePretrainedModel)
        model._layers_desc = list(layers_desc)
        model._stage_id = stage_id
        model.get_stage_from_index = lambda idx: (index_to_stage or {}).get(idx, stage_id)
        with mock.patch.object(
            PretrainedModel, "state_dict", return_value={k: None for k in pp_keys}
        ), mock.patch.object(PipelinePretrainedModel, "get_sequential_name_prefixes", return_value=self.PREFIXES):
            model._set_pipeline_name_mapping()
        return model._pp_to_single_mapping

    def test_directly_added_layer_keys_do_not_collide(self):
        pp_keys = [
            # chunk 0: `{chunk_start}.{local_idx}.rest`
            "0.0.embed_tokens.weight",
            "0.1.self_attn.o_proj.weight",
            # chunk 1 and the directly added lm_head, whose composite params all
            # share the `3.` prefix.
            "2.0.self_attn.o_proj.weight",
            "3.weight",
            "3.norm.weight",
            "3.enorm.weight",
            "3.transformer_layer.self_attn.o_proj.weight",
        ]
        mapping = self._build_mapping(pp_keys)

        self.assertEqual(
            mapping,
            {
                "0.0.embed_tokens.weight": "model.embed_tokens.embed_tokens.weight",
                "0.1.self_attn.o_proj.weight": "model.layers.0.self_attn.o_proj.weight",
                "2.0.self_attn.o_proj.weight": "model.layers.1.self_attn.o_proj.weight",
                "3.weight": "model.lm_head.weight",
                "3.norm.weight": "model.lm_head.norm.weight",
                "3.enorm.weight": "model.lm_head.enorm.weight",
                "3.transformer_layer.self_attn.o_proj.weight": "model.lm_head.transformer_layer.self_attn.o_proj.weight",
            },
        )
        # every pipeline key keeps a unique single card name
        self.assertEqual(len(set(mapping.values())), len(pp_keys))

    def test_chunk_shared_layer_keys_follow_shared_layer_rule(self):
        # A SharedLayerDesc with `forward_func` is registered on the chunk itself under
        # VPP, so the same parameter shows up both as `shared_layers.{name}.rest` and as
        # `{chunk_start}.{name}.rest`. Both aliases must resolve to the same name.
        layers_desc = [
            SharedLayerDesc("embed_weight_share", nn.Linear, shared_weight_attr="weight"),
            LayerDesc(nn.Linear),
            LayerDesc(nn.Linear),
            SharedLayerDesc(
                "embed_weight_share",
                nn.Linear,
                forward_func=lambda layer, x: x,
                shared_weight_attr="weight",
            ),
        ]
        pp_keys = [
            "shared_layers.embed_weight_share.weight",
            "1.0.self_attn.o_proj.weight",
            "3.embed_weight_share.weight",
        ]
        # stage 1 of a pp=2, vpp=2 run owns virtual stages 1 and 3
        mapping = self._build_mapping(
            pp_keys, layers_desc=layers_desc, stage_id=1, index_to_stage={0: 0, 1: 1, 2: 0, 3: 1}
        )

        self.assertEqual(
            mapping,
            {
                "shared_layers.embed_weight_share.weight": "model.lm_head.weight",
                "1.0.self_attn.o_proj.weight": "model.layers.0.self_attn.o_proj.weight",
                "3.embed_weight_share.weight": "model.lm_head.weight",
            },
        )

    def test_vpp_detected_when_shared_key_comes_first(self):
        # If the first chunk registers the SharedLayerDesc with `forward_func`, the first
        # non `shared_layers` key is `{chunk_start}.{name}.rest`, whose second segment is
        # not a digit. VPP must still be detected, otherwise the
        # `{chunk_start}.{local_idx}.rest` keys of the ordinary chunks would be resolved
        # against `prefixes[chunk_start]`.
        shared = SharedLayerDesc(
            "embed_weight_share", nn.Linear, forward_func=lambda layer, x: x, shared_weight_attr="weight"
        )
        layers_desc = [shared, LayerDesc(nn.Linear), LayerDesc(nn.Linear), shared]
        pp_keys = [
            "shared_layers.embed_weight_share.weight",
            "0.embed_weight_share.weight",
            "2.0.self_attn.o_proj.weight",
        ]
        # stage 0 of a pp=2, vpp=2 run owns virtual stages 0 and 2
        mapping = self._build_mapping(
            pp_keys, layers_desc=layers_desc, stage_id=0, index_to_stage={0: 0, 1: 1, 2: 0, 3: 1}
        )

        self.assertEqual(
            mapping,
            {
                "shared_layers.embed_weight_share.weight": "model.embed_tokens.weight",
                "0.embed_weight_share.weight": "model.embed_tokens.weight",
                "2.0.self_attn.o_proj.weight": "model.layers.1.self_attn.o_proj.weight",
            },
        )
