# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Llama model"""
from __future__ import annotations

import os
from typing import Optional

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.pipelining.schedules import (
    Schedule1F1B,
    ScheduleFThenB,
    ScheduleVPP,
)
from paddle.distributed.auto_parallel.pipelining.stage import PipelineStage
from paddle.distributed.fleet.utils import recompute

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None

try:
    from paddle.incubate.nn.functional import swiglu
except ImportError:

    def swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, chunks=2, axis=-1)
        return F.silu(x) * y


from ...utils.tools import get_env_device
from . import fusion_ops
from .configuration import LlamaConfig
from .modeling import _expand_2d_mask, _make_causal_mask, build_alibi_tensor
from .modeling_auto import LlamaDecoderLayerAuto, LlamaPretrainedModelAuto

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

__all__ = [
    "get_llama_pp_schedule",
]


def enable_fuse_ffn_qkv_pass():
    if os.getenv("FLAGS_enable_fused_ffn_qkv_pass") in [
        "True",
        "true",
        "1",
    ]:
        return True
    else:
        return False


def is_pp_enable():
    mesh = fleet.auto.get_mesh()
    return "pp" in mesh.dim_names


def get_mesh(pp_idx=0):
    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp", pp_idx)
    return mesh


def global_mesh_starts_with_pp():
    mesh = fleet.auto.get_mesh()
    if is_pp_enable():
        return mesh.get_mesh_with_dim("pp")
    else:
        return mesh


def get_attr(layer, name):
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


def parse_args(args):
    attention_mask, position_ids, alibi = None, None, None
    if isinstance(args, tuple):
        if len(args) == 4:
            hidden_states, attention_mask, position_ids, alibi = args
        if len(args) == 3:
            hidden_states, attention_mask, position_ids = args

        elif len(args) == 2:
            hidden_states, attention_mask = args

        if len(args) == 1:
            hidden_states = args[0]
    else:
        hidden_states = args

    if position_ids is not None:
        position_ids.stop_gradient = True

    if attention_mask is not None:
        attention_mask.stop_gradient = True

    if alibi is not None:
        alibi.stop_gradient = True

    return hidden_states, attention_mask, position_ids, alibi


def return_args(hidden_states, attention_mask=None, position_ids=None, alibi=None):
    ret = (hidden_states,)

    if attention_mask is not None:
        ret += (attention_mask.clone(),)
    if position_ids is not None:
        ret += (position_ids.clone(),)
    if alibi is not None:
        ret += (alibi.clone(),)
    if len(ret) == 1:
        ret = ret[0]

    return ret


class LlamaChunk(nn.Layer):
    def __init__(self, layers=None, is_first=False, is_last=False):
        super(LlamaChunk, self).__init__()
        assert not (is_first and is_last)
        self.layers = layers
        self.is_first = is_first
        self.is_last = is_last

    def forward(self, *args, **kwargs):
        if self.is_first:
            input_ids = kwargs.get("input_ids")
            attention_mask = kwargs.get("attention_mask")
            position_ids = kwargs.get("position_ids")
            outputs = tuple([input_ids, attention_mask, position_ids])
            # decoder layers
            for idx, (decoder_layer) in enumerate(self.layers):
                outputs = decoder_layer(outputs)
            return outputs
        elif self.is_last:
            outputs = args
            # decoder layers
            for idx, (decoder_layer) in enumerate(self.layers):
                outputs = decoder_layer(outputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        else:
            outputs = args
            # decoder layers
            for idx, (decoder_layer) in enumerate(self.layers):
                outputs = decoder_layer(outputs)
        return outputs


def manual_model_split(model, stage_idx, group, mode, pp_degree, need_shared_params=False):

    num_hidden_layers = model.config.num_hidden_layers
    virtual_pp_degree = model.config.virtual_pp_degree if mode == "VPP" else 1
    chunk_size = num_hidden_layers // virtual_pp_degree // pp_degree
    chunk_num = virtual_pp_degree * pp_degree
    layer_lists = None

    layer_lists = model.layers

    # if need_shared_params:
    #     shared_params_names = [["embedding_0.w_0.dist", "ernie_lm_head_0.w_0.dist"]]
    # else:
    #     shared_params_names = []
    shared_params_names = []
    shared_mp = build_shared_param_map(model, shared_params_names)

    def _build_stage(model, stage_idx, group):
        new_model = None
        if stage_idx == 0:
            new_model = LlamaChunk(layer_lists[:chunk_size], is_first=True, is_last=False)
        elif stage_idx == chunk_num - 1:
            new_model = LlamaChunk(
                layer_lists[stage_idx * chunk_size : (stage_idx + 1) * chunk_size], is_first=False, is_last=True
            )
        else:
            new_model = LlamaChunk(
                layer_lists[stage_idx * chunk_size : (stage_idx + 1) * chunk_size], is_first=False, is_last=False
            )
        stage = PipelineStage(new_model, stage_idx, chunk_num, group=group, shared_parameters=shared_mp)
        return stage

    stages = []
    for i in range(virtual_pp_degree):
        stage = _build_stage(model, stage_idx + i * pp_degree, group)
        stages.append(stage)
    return stages


def build_shared_param_map(model, shared_params_names):
    shared_mp = []
    for pair in shared_params_names:
        assert len(pair) == 2, "Only exactly two parameters are supported for sharing."
        ori_name = pair[0]
        sync_name = pair[1]
        ori_param = get_param_from_name(ori_name, model)
        sync_param = get_param_from_name(sync_name, model)
        shared_mp.append({"params": [ori_param, sync_param]})
    return shared_mp


def get_param_from_name(param_name, model):
    for param in model.parameters():
        if param.name == param_name:
            return param
    raise ValueError(f"{param_name} not found in model parameters")


def get_llama_pp_schedule(model, n_microbatches, loss_fn, mode, pp_degree, group, need_shared_params=False):
    assert mode in ["VPP", "1F1B", "FThenB"]
    stages = manual_model_split(model, group.rank, group, mode, pp_degree, need_shared_params)
    if mode == "VPP":
        schedule = ScheduleVPP(stages, n_microbatches=n_microbatches, loss_fn=loss_fn)
    elif mode == "1F1B":
        schedule = Schedule1F1B(stages[0], n_microbatches=n_microbatches, loss_fn=loss_fn)
    else:
        schedule = ScheduleFThenB(stages[0], n_microbatches=n_microbatches, loss_fn=loss_fn)
    return schedule

