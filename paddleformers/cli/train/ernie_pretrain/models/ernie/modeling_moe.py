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

import contextlib
import logging
import math
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.communication.group
import paddle.nn.functional as F
from paddle import nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.communication.group import _get_global_group
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.layers.mpu.mp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.nn.functional import fused_rms_norm_ext
from paddle.incubate.tensor.manipulation import async_offload

from paddleformers.cli.train.ernie_pretrain.models.comm_utils import profile
from paddleformers.cli.train.ernie_pretrain.models.ernie import ErnieMoEConfig
from paddleformers.cli.train.ernie_pretrain.models.ernie.modeling import (
    ErnieAttention,
    ErnieLMHead,
    ErnieMLP,
)
from paddleformers.cli.train.ernie_pretrain.models.ernie.modeling import (
    ErniePretrainingCriterion as ErniePretrainingCriterionBase,
)
from paddleformers.cli.train.ernie_pretrain.models.ernie.modeling import (
    FusedDropoutImpl,
    RMSNorm,
    RotaryEmbedding,
    _expand_mask,
    _make_causal_mask,
    finfo,
)
from paddleformers.cli.train.ernie_pretrain.models.fp8_linear import (
    Fp8FusedMlpFunc,
    MemEfficientFp8FusedMlpFunc,
)
from paddleformers.cli.train.ernie_pretrain.models.moe.moe_layer import (
    MOELayer,
    MoEStatics,
)
from paddleformers.cli.train.ernie_pretrain.models.moe.top2_gate import (
    Top2Gate,
    TopKGateFused,
)
from paddleformers.cli.train.ernie_pretrain.models.sequence_parallel_utils import (
    ColumnSequenceParallelLinear,
    GatherOp,
    RowSequenceParallelLinear,
    ScatterOp,
    get_async_loader,
    hack_offload_wait,
    mark_as_sequence_parallel_parameter,
)
from paddleformers.cli.train.ernie_pretrain.models.utils import get_global_training_logs
from paddleformers.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)
from paddleformers.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions as _BaseModelOutput,
)
from paddleformers.transformers.model_outputs import (
    CausalLMOutputWithCrossAttentions as _CausalLMOutput,
)
from paddleformers.transformers.model_utils import PretrainedModel, register_base_model
from paddleformers.utils.tools import get_env_device

try:
    from paddle.nn.functional import swiglu as fused_swiglu
except (ImportError, ModuleNotFoundError):
    fused_swiglu = None

logger = logging.getLogger(__name__)
paddle.distributed.communication.group.Group.__deepcopy__ = lambda self, _: self
paddle.distributed.communication.group.Group.to_json = lambda self: repr(self)


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(_BaseModelOutput):
    router_loss: Optional[paddle.Tensor] = None
    gate_logits: Optional[Tuple[paddle.Tensor]] = None
    mtp_outputs: Optional[paddle.Tensor] = None


@dataclass
class CausalLMOutputWithCrossAttentions(_CausalLMOutput):
    router_loss: Optional[paddle.Tensor] = None


global_training_logs = get_global_training_logs()

ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST = []

__all__ = [
    "ErnieMoEForCausalLM",
    "ErniePretrainingCriterion",
    "CausalLMOutputWithCrossAttentions",
]

gate_class = dict(
    top2=Top2Gate,
    top2_fused=TopKGateFused,
)


def get_gate(
    config: ErnieMoEConfig,
    expert: Tuple[Tuple[int, nn.Layer]],
    layer_idx: int,
) -> Tuple[nn.Layer, nn.LayerList]:
    moe_num_experts = config.moe_num_experts
    assert (
        moe_num_experts >= config.moe_world_size
    ), f"expert moe_num_experts={moe_num_experts} >= moe_world_size={config.moe_world_size}"
    assert (
        moe_num_experts % config.moe_world_size == 0
    ), f"expert moe_num_experts={moe_num_experts} % moe_world_size={config.moe_world_size} == 0"
    moe_num_experts_per_device = moe_num_experts // config.moe_world_size
    if not config.moe_fuse_experts:
        experts = nn.LayerList([])
        for expert_id, (experts_num, fc) in enumerate(expert):
            assert experts_num % config.moe_world_size == 0
            num_experts_per_device = experts_num // config.moe_world_size
            experts_to_append = []
            if not hasattr(fc, "__len__"):
                experts_to_append.append(fc)
                if expert_id == 1:
                    with paddle.utils.unique_name.guard("_mm_deepcopy"):
                        for _ in range(num_experts_per_device - 1):
                            experts_to_append.append(deepcopy(fc))
                else:
                    for _ in range(num_experts_per_device - 1):
                        experts_to_append.append(deepcopy(fc))
            else:
                experts_to_append = fc
            for ex in experts_to_append:
                for p in ex.parameters():
                    p.expert_type = f"expert_type_{expert_id}"
            experts.extend(experts_to_append)
        assert (
            len(experts) == moe_num_experts_per_device
        ), f"experts.len={len(experts)} != moe_num_experts_per_device={moe_num_experts_per_device}"
    else:
        assert expert[0][0] == 1, "experts are fused and must be one"
        experts = deepcopy(expert[0][1])

    logger.info(f"using moe-world-size: {config.moe_world_size} " f"expert-per-device: {moe_num_experts_per_device} ")
    if moe_num_experts <= 2:
        gate = None
        logger.info("MOE-GATE:-hard-gate")
    else:
        logger.info(f"MOE-GATE:-{config.moe_gate}")
        gate = gate_class[config.moe_gate.lower()](config, layer_idx=layer_idx, group=config.moe_group)

    lm_gate, lm_experts = gate, experts
    logger.info(f"LM-experts-{lm_experts} -- experts-{experts}")
    return gate, experts, lm_gate, lm_experts


def build_mpdp_group():
    hcg = fleet.get_hybrid_communicate_group()
    mp_world_size = hcg.get_model_parallel_world_size()
    dp_world_size = hcg.get_data_parallel_world_size()
    sharding_world_size = hcg.get_sharding_parallel_world_size()
    pp_world_size = hcg.get_pipe_parallel_world_size()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    topo = np.arange(world_size).reshape([pp_world_size, sharding_world_size, dp_world_size, mp_world_size])
    this_group = None
    for i in range(pp_world_size):
        for j in range(sharding_world_size):
            ranks = topo[i, j, :, :].reshape([-1]).tolist()
            group = dist.new_group(ranks)
            if rank in ranks:
                logger.info(f"building mpdp group, this group has rank: {ranks}")
                this_group = group
    return this_group


def _parse_moe_group(
    moe_group: str,
) -> Union[str, paddle.distributed.communication.group.Group]:
    moe_group = moe_group.lower()
    assert moe_group in {
        "sharding",
        "data",
        "dp",
        "mp",
        "tp",
        "model",
        "dummy",
        "none",
        "world",
        "all",
        "mpdp",
        "ep",
    }, f"moe-group not supported, got: {moe_group}"
    logger.info(f"using moe-group: {moe_group}")
    if not hasattr(fleet.fleet, "_hcg"):
        assert moe_group in {
            "dummy",
            "none",
            "world",
            "data",
        }, "only support dummy gate in `single-model`"
    if moe_group == "sharding":
        moe_group = fleet.get_hybrid_communicate_group().get_sharding_parallel_group()
    elif moe_group == "ep":
        moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
    elif moe_group in {"data", "dp"}:
        if hasattr(fleet.fleet, "_hcg"):
            moe_group = fleet.get_hybrid_communicate_group().get_data_parallel_group()
        else:
            moe_group = _get_global_group()
    elif moe_group in {"mp", "model", "tp"}:
        moe_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
    elif moe_group in {"dummy"}:
        dummy_group = paddle.distributed.communication.group.Group(0, None, [0])
        moe_group = dummy_group
    elif moe_group in {"mpdp"}:
        moe_group = build_mpdp_group()
    else:
        moe_group = _get_global_group()
    return moe_group


def moe_ep2mp(state_dict: Dict[str, paddle.Tensor], config: ErnieMoEConfig, split_actions):
    if config.tensor_model_parallel_size <= 1 or dist.get_world_size(config.moe_group) > 1:
        return state_dict
    if isinstance(config.moe_num_experts, (list, tuple)):
        num_lm_experts, num_mm_experts = config.moe_num_experts
        num_experts = sum(config.moe_num_experts)
    else:
        num_lm_experts, num_mm_experts = config.moe_num_experts, 0
        num_experts = config.moe_num_experts
    expert_ids = [int(re.search(r"mlp\.experts\.(\d+)", k).group(1)) for k in state_dict.keys() if "mlp.experts" in k]
    if expert_ids and max(expert_ids) == num_experts - 1:
        return state_dict

    logger.info("auto ep2mp")
    hcg = fleet.get_hybrid_communicate_group()
    mp_group = hcg.get_model_parallel_group()
    world_size = dist.get_world_size(mp_group)
    num_lm_local_experts = num_lm_experts // world_size
    num_mm_local_experts = num_mm_experts // world_size

    new_sd = {}

    actual_keys = []
    for k in state_dict.keys():
        actual_keys.append(k)
    actual_keys_sorted = sorted(actual_keys)

    for k in actual_keys_sorted:
        if "mlp.experts" in k:
            expert_id = int(re.search(r"mlp\.experts\.(\d+)", k).group(1))
            gathered_experts = []
            tensor = paddle.to_tensor(state_dict[k])
            dist.all_gather(gathered_experts, tensor, group=mp_group)
            for rank in range(len(gathered_experts)):
                if expert_id < num_lm_local_experts:
                    real_id = expert_id + rank * num_lm_local_experts
                else:
                    if num_mm_experts > 0:
                        real_id = num_lm_experts + (expert_id - num_lm_local_experts) + rank * num_mm_local_experts
                    else:
                        continue
                new_k = k.replace(f"mlp.experts.{expert_id}", f"mlp.experts.{real_id}")
                logger.info(f"auto ep2mp: {k}->{new_k}, expert_id: {expert_id}, real_id: {real_id}")
                new_sd[new_k] = split_actions[new_k.replace("ernie.", "")](gathered_experts[rank])
        else:
            new_sd[k] = state_dict[k]
    return new_sd


def moe_statedict_cherry_pick(state_dict: Dict[str, paddle.Tensor], config: ErnieMoEConfig):
    moe_num_experts = (
        sum(config.moe_num_experts) if isinstance(config.moe_num_experts, (list, tuple)) else config.moe_num_experts
    )
    if moe_num_experts <= 1:
        return state_dict
    moe_world_size = config.moe_world_size
    if moe_world_size <= 1:
        moe_world_size = 1
    moe_world_size_per_device = moe_num_experts // moe_world_size
    for key in list(state_dict.keys()):
        if "mlp.experts" in key:
            imoe = int(re.search(r"mlp\.experts\.(\d+)", key).group(1))
            if imoe >= moe_world_size_per_device:
                continue
            maybe_moe_name = key.replace(
                f"mlp.experts.{imoe}",
                f"mlp.experts.{config.moe_rank * moe_world_size_per_device + imoe}",
            )
            if maybe_moe_name != key and maybe_moe_name in state_dict:
                logger.info(f"moe auto changed state-dict using {maybe_moe_name} as {key}")
                state_dict[key] = state_dict.pop(maybe_moe_name)
    return state_dict


def moe_statedict_upcycle(
    state_dict: Dict[str, paddle.Tensor],
    config: ErnieMoEConfig,
    dtype,
    merge_actions,
    split_actions,
    layer_idxs=None,
):
    if not isinstance(config.moe_intermediate_size, int):
        logger.warning("moe upcycle only supports single modality expand !")
        return state_dict

    moe_layer_start_index = (
        min(config.moe_layer_start_index)
        if isinstance(config.moe_layer_start_index, (tuple, list))
        else config.moe_layer_start_index
    )
    moe_layer_end_index = (
        max(config.moe_layer_end_index)
        if isinstance(config.moe_layer_end_index, (tuple, list))
        else config.moe_layer_end_index
    )

    if config.moe_num_experts > 0:
        moe_world_size = config.moe_world_size
        if moe_world_size <= 1:
            moe_world_size = 1
        moe_world_size_per_device = config.moe_num_experts // moe_world_size

        granularity = (
            1 if config.moe_intermediate_size == 0 else config.intermediate_size // config.moe_intermediate_size
        )

        def slice_granularity(w, global_expert_id, column=True, shuffle=False, group_experts=False):
            if group_experts:
                part_id = global_expert_id // (config.moe_num_experts // config.moe_k)
            else:
                part_id = global_expert_id % config.moe_k
            part_id = part_id % granularity
            if shuffle:
                rng = random.Random(global_expert_id // config.moe_k)
                if column:
                    idx = np.arange(w.shape[-1])
                    rng.shuffle(idx)
                    w = w.index_select(paddle.to_tensor(idx), axis=-1)
                else:
                    idx = np.arange(w.shape[0])
                    rng.shuffle(idx)
                    w = w.index_select(paddle.to_tensor(idx), axis=0)
            if granularity == 1:
                return w
            if column:
                per_expert = w.shape[-1] // granularity
                return w[..., part_id * per_expert : (part_id + 1) * per_expert]
            per_expert = w.shape[0] // granularity
            w *= config.moe_k
            return w[part_id * per_expert : (part_id + 1) * per_expert, ...]

        def slice_granularity_shared(w, column=True):
            if column:
                per_expert = w.shape[-1] // granularity
                return w[..., -(per_expert * config.moe_num_shared_experts) :]
            per_expert = w.shape[0] // granularity
            return w[-(per_expert * config.moe_num_shared_experts) :, ...]

        def _chunk(t):
            return t.chunk(2, axis=-1) if isinstance(w, paddle.Tensor) else np.split(w, 2, axis=-1)

        def _cat(t):
            return paddle.concat(t, -1) if isinstance(t[0], paddle.Tensor) else np.concatenate(t, -1)

        granularity = (
            1 if config.moe_intermediate_size == 0 else config.intermediate_size // config.moe_intermediate_size
        )
        is_mp_moe = (
            hasattr(fleet.fleet, "_hcg")
            and config.moe_group is fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        logger.info(f"UPCYCLE-IS_MP_MOE: {is_mp_moe}")
        if is_mp_moe and fleet.get_hybrid_communicate_group().get_model_parallel_world_size() > 1:
            mp_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
        else:
            mp_group = None

        for ilayer in range(config.num_hidden_layers):
            if layer_idxs and ilayer not in layer_idxs:
                continue
            if ilayer < moe_layer_start_index or ilayer > moe_layer_end_index:
                continue
            if (ilayer + 1) % config.moe_layer_interval == 0:
                for k in ["up_proj", "gate_proj", "down_proj", "up_gate_proj"]:
                    for tail in ["weight", "bias"]:
                        non_moe_key = f"ernie.layers.{ilayer}.mlp.{k}.{tail}"
                        if non_moe_key in state_dict:
                            w = state_dict[non_moe_key]
                            if mp_group is not None and not (k == "down_proj" and tail == "bias"):
                                w = paddle.to_tensor(w).to(get_env_device())
                                gathered_w = []
                                logger.info(f"all_gather {non_moe_key} for moe upcycling")
                                dist.all_gather(gathered_w, w, group=mp_group)
                                w = w.cpu()
                                gathered_w = [v.cpu() for v in gathered_w]
                                gathered_w = merge_actions[non_moe_key.replace("ernie.", "")](gathered_w)
                                logger.info(f"gathered w is {gathered_w.shape}, type {gathered_w.dtype}")
                                w = gathered_w
                        for imoe in range(moe_world_size_per_device):
                            moe_name = f"ernie.layers.{ilayer}.mlp.experts.{imoe}.{k}.{tail}"
                            if moe_name not in state_dict and non_moe_key in state_dict:
                                if k == "up_gate_proj":
                                    w_ = _cat(
                                        [
                                            slice_granularity(
                                                ww,
                                                config.moe_rank * moe_world_size_per_device + imoe,
                                                column=True,
                                                group_experts=config.moe_group_experts,
                                            )
                                            for ww in _chunk(w)
                                        ]
                                    )
                                elif k == "down_proj" and tail == "bias":
                                    w_ = deepcopy(w)
                                else:
                                    w_ = slice_granularity(
                                        w,
                                        config.moe_rank * moe_world_size_per_device + imoe,
                                        column=k in {"up_proj", "gate_proj", "up_gate_proj"},
                                        group_experts=config.moe_group_experts,
                                    )
                                    logger.info(f"before slice: {w.shape} -> {w_.shape}")
                                logger.info(
                                    f"moe auto expand state-dict, ffn name G={granularity}: "
                                    f"{moe_name} {w_.shape} {w_.dtype} {dtype}"
                                )
                                if isinstance(w_, np.ndarray):
                                    w_ = paddle.to_tensor(w_)
                                if w_.dtype == dtype:
                                    state_dict[moe_name] = w_
                                else:
                                    state_dict[moe_name] = w_.cast(dtype)

                        if config.moe_num_shared_experts > 0:
                            moe_name = f"ernie.layers.{ilayer}.mlp.shared_experts.{k}.{tail}"
                            if moe_name not in state_dict and non_moe_key in state_dict:
                                if k == "up_gate_proj":
                                    w_ = _cat([slice_granularity_shared(ww, column=True) for ww in _chunk(w)])
                                    if mp_group is not None:
                                        w_ = split_actions[non_moe_key.replace("ernie.", "")](w_)
                                elif k == "down_proj" and tail == "bias":
                                    w_ = deepcopy(w)
                                else:
                                    w_ = slice_granularity_shared(
                                        w,
                                        column=k in {"up_proj", "gate_proj", "up_gate_proj"},
                                    )
                                    logger.info(f"W_ {k}-{w.shape}--shape-{w_.shape}")
                                    if mp_group is not None:
                                        w_ = split_actions[non_moe_key.replace("ernie.", "")](w_)
                                logger.info(
                                    f"moe auto expand state-dict, shared experts, ffn name G={granularity}: "
                                    f"{moe_name} {w_.shape} {w_.dtype}"
                                )
                                if isinstance(w_, np.ndarray):
                                    w_ = paddle.to_tensor(w_)
                                if w_.dtype == dtype:
                                    state_dict[moe_name] = w_
                                else:
                                    state_dict[moe_name] = w_.cast(dtype)

    return state_dict


class ErnieMoeMLP(ErnieMLP):
    def __init__(self, config, is_shared_expert=False):
        if getattr(config, "disable_ffn_model_parallel", False):
            config = deepcopy(config)
            config.tensor_model_parallel_size = 1
        super().__init__(config)
        self.moe_dropout_prob = config.moe_dropout_prob
        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."
        self.is_shared_expert = is_shared_expert
        self.shared_expert_mem_efficient = self.config.fp8_mem_configs["shared_expert"]

    def forward(self, x, use_comm=True):
        if (
            self.config.tensor_model_parallel_size <= 1
            and self.fuse_ffn
            and self.config.use_fp8_mlp
            and not self.config.use_bias
        ):
            if self.is_shared_expert and self.shared_expert_mem_efficient:
                return MemEfficientFp8FusedMlpFunc.apply(x, self.up_gate_proj.weight, self.down_proj.weight)
            return Fp8FusedMlpFunc.apply(x, self.up_gate_proj.weight, self.down_proj.weight)

        if self.fuse_ffn:
            up_gate_proj = (
                partial(self.up_gate_proj, use_comm=use_comm)
                if (isinstance(self.up_gate_proj, ColumnSequenceParallelLinear))
                else self.up_gate_proj
            )
        else:
            gate_proj = (
                partial(self.gate_proj, use_comm=use_comm)
                if (isinstance(self.gate_proj, ColumnSequenceParallelLinear))
                else self.gate_proj
            )
            up_proj = (
                partial(self.up_proj, use_comm=use_comm)
                if (isinstance(self.up_proj, ColumnSequenceParallelLinear))
                else self.up_proj
            )

        if self.fuse_swiglu:
            if self.fuse_ffn:
                if self.config.use_fp8 and self.config.fp8_configs["smooth_swiglu"]:
                    x, gate = up_gate_proj(x).chunk(2, axis=-1)

                    with paddle.no_grad():
                        scale = paddle.clip(gate.abs().max(axis=-1, keepdim=True), 1e-8)

                    gate = gate / scale
                    if self.config.sequence_parallel:
                        scale = ScatterOp.apply(scale)

                    x = paddle.concat([x, gate], axis=-1)
                else:
                    x = up_gate_proj(x)
                x = fused_swiglu(x)
            else:
                x = fused_swiglu(gate_proj(x), up_proj(x))
        else:
            if self.fuse_ffn:
                x, gate = up_gate_proj(x).chunk(2, axis=-1)
                x = F.silu(x) * gate
            else:
                x = F.silu(gate_proj(x)) * up_proj(x)
        if self.moe_dropout_prob > 0:
            with get_rng_state_tracker().rng_state("local_seed"):
                x = F.dropout(x=x, p=self.moe_dropout_prob)
        if self.config.use_fp8 and self.config.fp8_configs["smooth_swiglu"]:
            return self.down_proj(x) * scale
        ret = self.down_proj(x)
        return ret


class ErnieMoeDenseExpert(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else nn.Linear
        mp_degree = max(1, config.tensor_model_parallel_size)
        self.is_mp = mp_degree > 1
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fuse_ffn = config.fuse_attn_ffn

        if config.fuse_attn_ffn:
            self.up_gate_proj = LinearFN(
                self.hidden_size,
                self.intermediate_size * 2 // mp_degree,
                bias_attr=config.use_bias,
            )
            self.up_gate_proj.weight.is_distributed = self.is_mp
            if config.use_bias:
                self.up_gate_proj.bias.is_distributed = self.is_mp
        else:
            self.gate_proj = LinearFN(
                self.hidden_size,
                self.intermediate_size // mp_degree,
                bias_attr=config.use_bias,
            )
            self.up_proj = LinearFN(
                self.hidden_size,
                self.intermediate_size // mp_degree,
                bias_attr=config.use_bias,
            )
            self.gate_proj.weight.is_distributed = self.is_mp
            self.up_proj.weight.is_distributed = self.is_mp
            if config.use_bias:
                self.gate_proj.bias.is_distributed = self.is_mp
                self.up_proj.bias.is_distributed = self.is_mp
        self.down_proj = LinearFN(
            self.intermediate_size // mp_degree,
            self.hidden_size,
            bias_attr=config.use_bias,
        )
        self.down_proj.weight.is_distributed = self.is_mp

        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."
        if self.is_mp:
            self.mp_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()

    def forward(self, x):
        if self.fuse_swiglu:
            if self.fuse_ffn:
                x = fused_swiglu(self.up_gate_proj(x))
            else:
                x = fused_swiglu(self.gate_proj(x), self.up_proj(x))
        else:
            if self.fuse_ffn:
                x, gate = self.up_gate_proj(x).chunk(2, axis=-1)
                x = F.silu(x) * gate
            else:
                x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.is_mp:
            x = F.linear(x, self.down_proj.weight)
            output_ = mp_ops._mp_allreduce(
                x,
                group=self.mp_group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
            output = output_ + self.down_proj.bias if self.config.use_bias else output_
        else:
            output = self.down_proj(x)

        return output


class BMMLinear(nn.Layer):
    def __init__(self, experts, d_in, d_out, use_bias=False):
        super().__init__()
        self.weight = self.create_parameter([experts, d_in, d_out], dtype=paddle.get_default_dtype())
        if use_bias:
            self.bias = self.create_parameter([experts, d_out], dtype=paddle.get_default_dtype(), is_bias=True)
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return paddle.bmm(x, self.weight) + self.bias
        return paddle.bmm(x, self.weight)


class ErnieMoeMLPFused(nn.Layer):
    def __init__(self, config):
        assert (
            hasattr(config, "disable_ffn_model_parallel") or config.tensor_model_parallel_size == 1
        ), f"fused mlp only support mp-moe, mp={config.tensor_model_parallel_size}"
        assert config.fuse_attn_ffn, "fused mlp only support fuse_attn_ffn"
        super().__init__()
        self.moe_dropout_prob = config.moe_dropout_prob
        self.num_local_experts = config.moe_num_experts // config.moe_world_size
        logger.info(
            f"fused-expert-weight-shape: {[self.num_local_experts, config.hidden_size, config.intermediate_size]}"
        )

        self.up_gate_proj = BMMLinear(self.num_local_experts, config.hidden_size, config.intermediate_size * 2)
        self.down_proj = BMMLinear(self.num_local_experts, config.intermediate_size, config.hidden_size)
        self.fuse_swiglu = config.fuse_swiglu
        if self.fuse_swiglu:
            assert fused_swiglu is not None, "fused_swiglu operator is not found."

    def __len__(self):
        return self.num_local_experts

    def __iter__(self):
        return (self for _ in range(1))

    def forward(self, x):
        if self.fuse_swiglu:
            x = fused_swiglu(self.up_gate_proj(x))
        else:
            gate, x = self.up_gate_proj(x).chunk(2, axis=-1)
            x = F.silu(gate) * x
        x = self.down_proj(x)
        return x


class FusedLinearAddNormFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, residual, linear_weight, rms_norm_weight, eps):
        linear_out = paddle.matmul(x, linear_weight)
        add_out = linear_out + residual
        norm_out, invar = fused_rms_norm_ext(add_out, rms_norm_weight, eps)

        ctx.save_for_backward(x, residual, linear_weight, rms_norm_weight, eps)

        return norm_out, add_out

    @staticmethod
    def backward(ctx, d_rms_norm_out, d_residual_out):
        x, residual, linear_weight, rms_norm_weight, eps = ctx.saved_tensor()

        linear_out = paddle.matmul(x, linear_weight)
        add_out = linear_out + residual

        rms_out, invar = fused_rms_norm_ext(add_out, rms_norm_weight, eps)

        d_add_out, d_rms_norm_weight = paddle._C_ops.fused_rms_norm_ext_grad(
            add_out, rms_norm_weight, invar, d_rms_norm_out, eps
        )

        d_residual = d_add_out + d_residual_out
        d_linear_out = d_residual
        dx, d_linear_weight = paddle._C_ops.matmul_grad(x, linear_weight, d_linear_out, False, False)

        return dx, d_residual, d_linear_weight, d_rms_norm_weight


class FusedLinearAddNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps=1e-6) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()

        self.linear_weight = self.create_parameter(
            shape=[hidden_size, hidden_size],
            dtype=self._dtype,
            is_bias=False,
        )

        self.rms_norm_weight = self.create_parameter(
            shape=[hidden_size],
            dtype=self._dtype,
            default_initializer=nn.initializer.Constant(1.0),
        )

        self.eps = eps

    def forward(self, x, residual):
        return FusedLinearAddNormFunc.apply(x, residual, self.linear_weight, self.rms_norm_weight, self.eps)


class FusedRMSLinearFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, rms_norm_weight, linear_weight, eps):
        hidden_states, invar = fused_rms_norm_ext(x, rms_norm_weight, eps)
        q = paddle.matmul(hidden_states, linear_weight)

        ctx.save_for_backward(x, rms_norm_weight, linear_weight, eps)
        return q

    @staticmethod
    def backward(ctx, d_qkv):
        x, rms_norm_weight, linear_weight, eps = ctx.saved_tensor()
        hidden_states, invar = fused_rms_norm_ext(x, rms_norm_weight, eps)
        h_grad, d_linear_weight = paddle._C_ops.matmul_grad(hidden_states, linear_weight, d_qkv, False, False)

        dx, d_rms_norm_weight = paddle._C_ops.fused_rms_norm_ext_grad(x, rms_norm_weight, invar, h_grad, eps)

        return dx, d_rms_norm_weight, d_linear_weight


class FusedRMSLinear(paddle.nn.Layer):
    def __init__(self, hidden_size, eps=1e-6, num_heads=1, num_key_value_heads=1) -> None:
        super().__init__()
        self._dtype = self._helper.get_default_dtype()

        self.rms_norm_weight = self.create_parameter(
            shape=[hidden_size],
            dtype=self._dtype,
            default_initializer=nn.initializer.Constant(1.0),
        )
        kv_hidden_size = hidden_size // num_heads * num_key_value_heads
        qkv_out = hidden_size + kv_hidden_size * 2

        self.linear_weight = self.create_parameter(
            shape=[hidden_size, qkv_out],
            dtype=self._dtype,
            is_bias=False,
        )
        self.eps = eps

    def forward(self, x):
        return FusedRMSLinearFunc.apply(x, self.rms_norm_weight, self.linear_weight, self.eps)


class ErnieMoEAttention(ErnieAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config)

        self.use_linear_residual_norm_recompute = config.use_linear_residual_norm_recompute
        self.use_rms_qkv_recompute = config.use_rms_qkv_recompute
        if config.use_rms_qkv_recompute is True:

            assert config.use_rmsnorm is True and config.fuse_rms_norm is True
            assert config.fuse_linear is True and config.use_bias is False

            assert self.fuse_attn is True

            if self.is_gqa:
                self.fused_rms_norm_linear = FusedRMSLinear(
                    self.hidden_size,
                    config.rms_norm_eps,
                    self.num_heads,
                    self.num_key_value_heads,
                )
            else:
                self.fused_rms_norm_linear = FusedRMSLinear(self.hidden_size, config.rms_norm_eps)
            del self.qkv_proj

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        inbatch_pack_offset: Optional[Tuple[paddle.Tensor]] = None,
        token_type_ids: Optional[Tuple[paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :-1]
        if self.config.sequence_parallel:
            if token_type_ids is not None:
                token_type_ids = token_type_ids.reshape([-1])
                token_type_ids = ScatterOp.apply(token_type_ids)
                token_type_ids.stop_gradient = True
            q_len = self.config.seqlen
        else:
            q_len = hidden_states.shape[-2]

        query_states = key_states = value_states = mix_layer = None
        if self.use_rms_qkv_recompute:
            mix_layer = self.fused_rms_norm_linear(hidden_states)
        else:
            if self.fuse_attn:
                mix_layer = self.qkv_proj(
                    hidden_states,
                )
            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

        if self.fuse_attn:
            if self.is_gqa:
                query_states, key_states, value_states = paddle.split(
                    mix_layer.reshape(
                        [
                            -1,
                            q_len,
                            self.num_heads + 2 * self.num_key_value_heads,
                            self.head_dim,
                        ]
                    ),
                    [
                        self.num_heads,
                        self.num_key_value_heads,
                        self.num_key_value_heads,
                    ],
                    axis=2,
                )
                mix_layer = None
            else:
                mix_layer = mix_layer.reshape([-1, q_len, self.num_heads, 3 * self.head_dim])

        else:
            query_states = query_states.reshape(shape=[-1, q_len, self.num_heads, self.head_dim])
            key_states = key_states.reshape(
                shape=[
                    -1,
                    q_len,
                    self.num_key_value_heads if self.is_gqa else self.num_heads,
                    self.head_dim,
                ]
            )
            value_states = value_states.reshape(
                shape=[
                    -1,
                    q_len,
                    self.num_key_value_heads if self.is_gqa else self.num_heads,
                    self.head_dim,
                ]
            )
        if self.use_recompute_attn:
            assert past_key_value is None, "do not use kv cache in recompute"
            assert not use_cache
            attn_output, attn_weights, past_key_value = recompute(
                self.rope_attn,
                mix_layer,
                query_states,
                key_states,
                value_states,
                attention_mask,
                position_ids,
                output_attentions,
                past_key_value,
                use_cache,
                inbatch_pack_offset,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                use_reentrant=False,
            )
        else:
            attn_output, attn_weights, past_key_value = self.rope_attn(
                mix_layer=mix_layer,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                inbatch_pack_offset=inbatch_pack_offset,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            )
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])

        if self.use_linear_residual_norm_recompute is False:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class FakeMoERouterLoss(PyLayer):
    @staticmethod
    def forward(ctx, x, router_loss, num_acc_steps, enable_delay_scale_loss):
        ctx.num_acc_steps = num_acc_steps
        ctx.loss_shape = router_loss.shape
        ctx.loss_dtype = router_loss.dtype
        ctx.enable_delay_scale_loss = enable_delay_scale_loss
        return x

    @staticmethod
    def backward(ctx, out_grad):
        if ctx.enable_delay_scale_loss:
            router_loss_grad_value = 1.0
        else:
            router_loss_grad_value = 1.0 / ctx.num_acc_steps

        return out_grad, paddle.full(ctx.loss_shape, router_loss_grad_value, dtype=ctx.loss_dtype)


class ErnieDecoderLayer(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self._training = True
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe_infer = config.get("is_moe_infer", False)
        self.config = config
        self.use_moe = config.use_moe
        self.self_attn = ErnieMoEAttention(config, layer_idx)
        self.use_linear_residual_norm_recompute = config.use_linear_residual_norm_recompute
        self.use_rms_qkv_recompute = config.use_rms_qkv_recompute

        moe_layer_start_index = (
            min(config.moe_layer_start_index)
            if isinstance(config.moe_layer_start_index, (tuple, list))
            else config.moe_layer_start_index
        )
        moe_layer_end_index = (
            max(config.moe_layer_end_index)
            if isinstance(config.moe_layer_end_index, (tuple, list))
            else config.moe_layer_end_index
        )

        if (
            self.use_moe
            and ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index
            and layer_idx <= moe_layer_end_index
        ):
            gate, experts, lm_gate, lm_experts, moe_statics = self._init_gate_and_experts(layer_idx)
            shared_experts = self._init_shared_experts()
            dense_experts = self._init_dense_experts(layer_idx)
            moe_cls = MOELayer
            logger.info(f"moe_cls={moe_cls}")
            assert dense_experts is None
            self.mlp = moe_cls(
                gate,
                experts,
                layer_idx=layer_idx,
                shared_experts=shared_experts,
                group=config.moe_group,
                recompute=config.use_recompute_moe,
                k=config.moe_k,
                all_to_all_dropout=config.moe_all_to_all_dropout,
                group_experts=config.moe_group_experts,
                moe_statics=moe_statics,
            )
            if config.sequence_parallel:
                for p in gate.parameters():
                    mark_as_sequence_parallel_parameter(p)
        else:
            self.mlp = ErnieMLP(config)

        Norm = RMSNorm

        if self.use_rms_qkv_recompute is False:
            self.input_layernorm = Norm(config)

        if self.use_linear_residual_norm_recompute is True:
            assert config.hidden_dropout_prob == 0.0
            assert config.fuse_linear is True and config.use_bias is False
            assert config.use_rmsnorm is True and config.fuse_rms_norm is True
            self.fused_linear_add_norm = FusedLinearAddNorm(self.hidden_size, config.rms_norm_eps)
            del self.self_attn.o_proj
        else:
            self.residual_add1 = FusedDropoutImpl(config.hidden_dropout_prob, mode="upscale_in_train")
            self.post_attention_layernorm = Norm(config)

        self.residual_add2 = FusedDropoutImpl(config.hidden_dropout_prob, mode="upscale_in_train")

        if config.sequence_parallel:
            if self.use_linear_residual_norm_recompute is True:
                mark_as_sequence_parallel_parameter(self.fused_linear_add_norm.rms_norm_weight)
            else:
                mark_as_sequence_parallel_parameter(self.post_attention_layernorm.weight)
            if not hasattr(config, "disable_ffn_model_parallel"):
                if self.use_rms_qkv_recompute is True:
                    mark_as_sequence_parallel_parameter(self.self_attn.fused_rms_norm_linear.rms_norm_weight)
                else:
                    mark_as_sequence_parallel_parameter(self.input_layernorm.weight)

            if not config.use_rmsnorm:
                mark_as_sequence_parallel_parameter(self.post_attention_layernorm.bias)
                mark_as_sequence_parallel_parameter(self.input_layernorm.bias)

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, new):
        if hasattr(self, "mlp_text"):
            for c in self.mlp_text().sublayers():
                c.training = new
        self._training = new

    def fp8_quant_weight(self):
        if isinstance(self.mlp, MOELayer):
            logger.info(f"fp8 quant weight for mlp {type(self.mlp)}")
            self.mlp.fp8_quant_weight()

    def _init_gate_and_experts(self, layer_idx):
        cfg = deepcopy(self.config)
        fc_cls = ErnieMoeMLPFused if cfg.moe_fuse_experts and not cfg.use_fp8_mlp else ErnieMoeMLP
        if self.config.expert_mlp_use_bias is not None:
            cfg.use_bias = self.config.expert_mlp_use_bias

        if cfg.moe_intermediate_size:
            if isinstance(cfg.moe_intermediate_size, (tuple, list)):
                assert isinstance(cfg.moe_num_experts, (tuple, list)) and len(cfg.moe_num_experts) == len(
                    cfg.moe_intermediate_size
                )
                fc = []
                for _i, (num_experts, intermediate_size) in enumerate(
                    zip(cfg.moe_num_experts, cfg.moe_intermediate_size)
                ):
                    ex_cfg = deepcopy(cfg)
                    ex_cfg.intermediate_size = intermediate_size
                    cur_modality_start_layer_idx = (
                        cfg.moe_layer_start_index[_i]
                        if isinstance(cfg.moe_layer_start_index, (tuple, list))
                        else cfg.moe_layer_start_index
                    )
                    cur_modality_end_layer_idx = (
                        cfg.moe_layer_end_index[_i]
                        if isinstance(cfg.moe_layer_end_index, (tuple, list))
                        else cfg.moe_layer_end_index
                    )
                    if layer_idx >= cur_modality_start_layer_idx and layer_idx <= cur_modality_end_layer_idx:
                        if _i == 1:
                            with paddle.utils.unique_name.guard(f"mm_expert_{layer_idx}_"):
                                fc.append((num_experts, fc_cls(ex_cfg)))
                        else:
                            fc.append((num_experts, fc_cls(ex_cfg)))
                    else:
                        logger.info(f"moe experts use Identity layer_idx: {layer_idx}")
                        fc.append((num_experts, nn.Identity()))
            else:
                cfg.intermediate_size = cfg.moe_intermediate_size
                if cfg.moe_fuse_experts:
                    fc = [(1, fc_cls(cfg))]
                else:
                    fc = [(cfg.moe_num_experts, fc_cls(cfg))]
        else:
            fc = [(cfg.moe_num_experts, fc_cls(cfg))]
        gate, experts, lm_gate, lm_experts = get_gate(self.config, fc, layer_idx)
        if cfg.moe_use_aux_free:
            moe_statics = MoEStatics(cfg, layer_idx)
        else:
            moe_statics = None
        return gate, experts, lm_gate, lm_experts, moe_statics

    def _init_shared_experts(self):
        cfg = deepcopy(self.config)
        if cfg.moe_num_shared_experts > 0:
            if cfg.moe_intermediate_size:
                inter_size = (
                    cfg.moe_intermediate_size[0]
                    if isinstance(cfg.moe_intermediate_size, (tuple, list))
                    else cfg.moe_intermediate_size
                )
                cfg.intermediate_size = inter_size * cfg.moe_num_shared_experts
            else:
                cfg.intermediate_size = cfg.intermediate_size * cfg.moe_num_shared_experts
            cfg.disable_ffn_model_parallel = False
            shared_experts = ErnieMoeMLP(cfg, True)
        else:
            shared_experts = None
        return shared_experts

    def _init_dense_experts(self, layer_idx):
        cfg = deepcopy(self.config)
        cfg.sequence_parallel = False
        if cfg.moe_num_dense_experts > 0:
            logger.info("using dense experts")
            if cfg.moe_intermediate_size:
                inter_size = (
                    cfg.moe_intermediate_size[0]
                    if isinstance(cfg.moe_intermediate_size, (tuple, list))
                    else cfg.moe_intermediate_size
                )
                cfg.intermediate_size = inter_size * cfg.moe_num_dense_experts
            else:
                cfg.intermediate_size = cfg.intermediate_size * cfg.moe_num_shared_experts
            cfg.disable_ffn_model_parallel = False
            with paddle.utils.unique_name.guard(f"audio_expert_{layer_idx}_"):
                dense_experts = ErnieMoeDenseExpert(cfg)
            for p in dense_experts.parameters():
                p.expert_type = "expert_type_3"
        else:
            dense_experts = None
        return dense_experts

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        use_cache: Optional[bool] = False,
        inbatch_pack_offset: Optional[paddle.Tensor] = None,
        output_gate_logits=True,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        residual = hidden_states
        if token_type_ids is not None:
            is_multimodel_token = token_type_ids.any()
            has_dense_experts_token = (token_type_ids == self.config.moe_dense_experts_token_type_id).any()
            async_loader = get_async_loader()
            is_multimodel_token_cpu, is_multimodel_token_task = async_offload(is_multimodel_token, async_loader)
            has_dense_experts_token_cpu, has_dense_experts_token_task = async_offload(
                has_dense_experts_token, async_loader
            )
        else:
            is_multimodel_token_task = None
            has_dense_experts_token_task = None

        if self.use_rms_qkv_recompute is False:
            hidden_states = self.input_layernorm(hidden_states)

        (hidden_states, self_attn_weights, present_key_value) = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            inbatch_pack_offset=inbatch_pack_offset,
            token_type_ids=token_type_ids,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        if self.use_linear_residual_norm_recompute is True:
            hidden_states, residual = self.fused_linear_add_norm(hidden_states, residual)
        else:
            with self.model_parallel_dropout():
                hidden_states = self.residual_add1(hidden_states, residual)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        if isinstance(
            self.mlp,
            (MOELayer,),
        ):
            if is_multimodel_token_task is not None:
                hack_offload_wait(is_multimodel_token_task)
            if has_dense_experts_token_task is not None:
                hack_offload_wait(has_dense_experts_token_task)

            with profile("moe-mlp"):
                hidden_states, _, router_loss, gate_logits = self.mlp(hidden_states, token_type_ids)
        else:
            hidden_states = self.mlp(hidden_states)
            gate_logits = None

        with self.model_parallel_dropout():
            hidden_states = self.residual_add2(hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if self.use_moe:
            if output_gate_logits:
                outputs += (gate_logits,)

        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def model_parallel_dropout(self):
        if self.config.tensor_model_parallel_size > 1 and self.config.hidden_dropout_prob > 0.0:
            current_seed = "local_seed" if self.config.sequence_parallel else "global_seed"
            return get_rng_state_tracker().rng_state(current_seed)
        return contextlib.nullcontext()


class ErniePretrainedModel(PretrainedModel):
    config_class = ErnieMoEConfig
    base_model_prefix = "ernie"

    @classmethod
    def _get_name_mappings(cls, config: ErnieMoEConfig) -> StateDictNameMapping:
        model_mappings = [
            ["embed_tokens.weight"],
            ["norm.weight"],
        ]
        for layer_index in range(config.num_hidden_layers):
            if config.fuse_attn_ffn:
                layer_mappings = [
                    [
                        f"layers.{layer_index}.self_attn.qkv_proj.weight",
                        None,
                        "transpose",
                    ],
                    [
                        f"layers.{layer_index}.self_attn.o_proj.weight",
                        None,
                        "transpose",
                    ],
                    [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                    [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                    [
                        f"layers.{layer_index}.mlp.up_gate_proj.weight",
                        None,
                        "transpose",
                    ],
                    [f"layers.{layer_index}.input_layernorm.weight"],
                    [f"layers.{layer_index}.post_attention_layernorm.weight"],
                ]
            else:
                layer_mappings = [
                    [
                        f"layers.{layer_index}.self_attn.q_proj.weight",
                        None,
                        "transpose",
                    ],
                    [
                        f"layers.{layer_index}.self_attn.k_proj.weight",
                        None,
                        "transpose",
                    ],
                    [
                        f"layers.{layer_index}.self_attn.v_proj.weight",
                        None,
                        "transpose",
                    ],
                    [
                        f"layers.{layer_index}.self_attn.o_proj.weight",
                        None,
                        "transpose",
                    ],
                    [f"layers.{layer_index}.self_attn.rotary_emb.inv_freq"],
                    [f"layers.{layer_index}.mlp.gate_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.down_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.mlp.up_proj.weight", None, "transpose"],
                    [f"layers.{layer_index}.input_layernorm.weight"],
                    [f"layers.{layer_index}.post_attention_layernorm.weight"],
                ]
            model_mappings.extend(layer_mappings)

        init_name_mappings(mappings=model_mappings)
        if "ErnieModel" not in config.architectures:
            for mapping in model_mappings:
                mapping[0] = "model." + mapping[0]
                mapping[1] = "ernie." + mapping[1]
            model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        from paddleformers.cli.train.ernie_pretrain.models.ernie.modeling import (
            gqa_qkv_merge_func,
            gqa_qkv_split_func,
        )
        from paddleformers.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        if config.num_key_value_heads is not None and config.num_key_value_heads != config.num_attention_heads:
            if is_split:
                qkv_fn = partial(
                    gqa_qkv_split_func,
                    tensor_model_parallel_size=config.tensor_model_parallel_size,
                    tensor_parallel_rank=config.tensor_parallel_rank,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                )
            else:
                qkv_fn = partial(
                    gqa_qkv_merge_func,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                )
        else:
            qkv_fn = partial(fn, is_column=True)

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}
            if config.fuse_attn_ffn:
                base_actions = {
                    "layers.0.self_attn.qkv_proj.weight": qkv_fn,
                    "layers.0.mlp.up_gate_proj.weight": partial(fn, is_column=True, is_naive_2fuse=True),
                    "lm_head.weight": partial(fn, is_column=not config.tie_word_embeddings),
                    "embed_tokens.weight": partial(fn, is_column=False),
                    "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                    "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
                }
                if config.use_bias:
                    base_actions.update(
                        {
                            "layers.0.self_attn.qkv_proj.bias": qkv_fn,
                            "layers.0.mlp.up_gate_proj.bias": partial(fn, is_column=True, is_naive_2fuse=True),
                            "layers.0.mlp.down_proj.bias": lambda x: x,
                            "lm_head.bias": partial(fn, is_column=True),
                        }
                    )
            else:
                base_actions = {
                    "layers.0.self_attn.q_proj.weight": partial(fn, is_column=True),
                    "layers.0.self_attn.k_proj.weight": partial(fn, is_column=True),
                    "layers.0.self_attn.v_proj.weight": partial(fn, is_column=True),
                    "layers.0.mlp.gate_proj.weight": partial(fn, is_column=True),
                    "layers.0.mlp.up_proj.weight": partial(fn, is_column=True),
                    "embed_tokens.weight": partial(fn, is_column=False),
                    "layers.0.self_attn.o_proj.weight": partial(fn, is_column=False),
                    "layers.0.mlp.down_proj.weight": partial(fn, is_column=False),
                }
                if config.use_bias:
                    base_actions.update(
                        {
                            "layers.0.self_attn.q_proj.bias": partial(fn, is_column=True),
                            "layers.0.self_attn.k_proj.bias": partial(fn, is_column=True),
                            "layers.0.self_attn.v_proj.bias": partial(fn, is_column=True),
                            "layers.0.mlp.gate_proj.bias": partial(fn, is_column=True),
                            "layers.0.mlp.up_proj.bias": partial(fn, is_column=True),
                            "layers.0.mlp.down_proj.bias": lambda x: x,
                            "lm_head.bias": partial(fn, is_column=True),
                        }
                    )
            moe_in_mp = config.moe_group in {"mp", "model", "tp", "mpdp"}
            for key, action in base_actions.items():
                if "layers.0." in key:
                    for i in range(num_layers):
                        newkey = key.replace("layers.0.", f"layers.{i}.")
                        if config.moe_group in {"mpdp"}:
                            final_actions[newkey] = lambda x: x
                        else:
                            final_actions[newkey] = action
                        if "mlp" in key and (i + 1) % config.moe_layer_interval == 0:
                            moe_num_experts = config.moe_num_experts
                            if moe_num_experts > 0:
                                for expert_id in range(moe_num_experts):
                                    _key = key.replace(
                                        "layers.0.mlp",
                                        f"layers.{i}.mlp.experts.{expert_id}",
                                    )
                                    if moe_in_mp:
                                        final_actions[_key] = lambda x: x
                                    else:
                                        final_actions[_key] = action
                                for _ in range(config.moe_num_shared_experts):
                                    _key = key.replace("layers.0.mlp", f"layers.{i}.mlp.shared_experts")
                                    final_actions[_key] = action
                                for _ in range(config.moe_num_dense_experts):
                                    _key = key.replace("layers.0.mlp", f"layers.{i}.mlp.dense_experts")
                                    final_actions[_key] = action
                            else:
                                final_actions[key.replace("layers.0.", f"layers.{i}.")] = action

                        elif "self_attn" in key and (
                            "qkv_proj" in key or "q_proj" in key or "k_proj" in key or "v_proj" in key
                        ):
                            final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                        else:
                            final_actions[key.replace("layers.0.", f"layers.{i}.")] = action
                else:
                    final_actions[key] = action
            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)
        return mappings

    def _init_weights(self, layer):
        if get_rng_state_tracker().states_:
            rng_tracker = get_rng_state_tracker().rng_state
        else:
            rng_tracker = contextlib.nullcontext

        if isinstance(
            layer,
            (
                ColumnParallelLinear,
                RowParallelLinear,
                ColumnSequenceParallelLinear,
                RowSequenceParallelLinear,
                VocabParallelEmbedding,
                ErnieLMHead,
                nn.Embedding,
                BMMLinear,
                nn.Linear,
                paddle.incubate.nn.FusedLinear,
            ),
        ):
            if not hasattr(layer, "weight"):
                return

            is_moe = getattr(layer.weight, "no_sync", False)
            with rng_tracker("local_seed" if is_moe else "model_parallel_rng"):
                dtype = paddle.get_default_dtype()
                paddle.set_default_dtype("float32")
                layer.weight.set_value(
                    paddle.randn(layer.weight.shape, dtype=dtype).scale(self.config.initializer_range)
                )
                paddle.set_default_dtype(dtype)
                logger.info(
                    f"dist-init-fc: shape={layer.weight.shape}, dtype={layer.weight.dtype} "
                    f"range={self.config.initializer_range},type={type(layer)}, "
                    f'norm={layer.weight.astype("float32").norm().item()},is_moe={is_moe}'
                )
        elif isinstance(layer, (Top2Gate, TopKGateFused)):
            if not hasattr(layer, "weight"):
                return
            with rng_tracker("model_parallel_rng"):
                dtype = paddle.get_default_dtype()
                paddle.set_default_dtype("float32")
                layer.weight.set_value(
                    paddle.randn(layer.weight.shape, dtype=layer.weight.dtype).scale(self.config.initializer_range)
                )
                logger.info(
                    f"dist-init-moe_gate: shape={layer.weight.shape}, dtype={layer.weight.dtype} "
                    f"range={self.config.initializer_range},type={type(layer)}, "
                    f'norm={layer.weight.astype("float32").norm().item()}'
                )
                if isinstance(self.config.moe_num_experts, (tuple, list)):
                    for i in range(1, len(self.config.moe_num_experts)):
                        layer_weight = getattr(layer, f"weight_{i}")
                        layer_weight.set_value(
                            paddle.randn(layer_weight.shape, dtype=layer_weight.dtype).scale(
                                self.config.initializer_range
                            )
                        )
                        logger.info(
                            f"dist-init-moe_gate: shape={layer_weight.shape}, dtype={layer_weight.dtype} "
                            f"range={self.config.initializer_range},type={type(layer)}, "
                            f'norm={layer_weight.astype("float32").norm().item()}'
                        )
                paddle.set_default_dtype(dtype)

        elif isinstance(layer, RotaryEmbedding):
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            inv_freq = 1.0 / (layer.base ** (np.arange(0, head_dim, 2).astype("float32") / head_dim))
            t = np.arange(layer.max_position_embeddings, dtype="float32")
            freqs = np.einsum("i,j->ij", t, inv_freq)
            emb = np.concatenate([freqs, freqs], axis=-1)
            cos_cached = np.cos(emb)[:, :]
            sin_cached = np.sin(emb)[:, :]
            layer.cos_cached.set_value(cos_cached)
            layer.sin_cached.set_value(sin_cached)


@register_base_model
class ErnieModel(ErniePretrainedModel):
    def __init__(self, config: ErnieMoEConfig):
        if config.moe_group in {"mp", "model", "tp", "mpdp"}:
            logger.info(f"disable FFN tensor model parallel, moe-group={config.moe_group}")
            config.disable_ffn_model_parallel = True

        config.moe_group = _parse_moe_group(config.moe_group)

        config.moe_world_size = dist.get_world_size(config.moe_group)
        if config.moe_world_size < 0:
            config.moe_world_size = 1
        config.moe_rank = dist.get_rank(config.moe_group)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        if config.tensor_model_parallel_size > 1:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
            )

        self.layers = nn.LayerList([ErnieDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        Norm = RMSNorm

        self.norm = Norm(config)

        self.gradient_checkpointing = False

        if self.config.multi_token_pred_depth > 0:
            self.mtp_block = paddle.nn.LayerList(
                [ErnieDecoderLayer(config, layer_idx) for layer_idx in range(self.config.multi_token_pred_depth)]
            )
            Norm = RMSNorm

            self.mtp_hidden_norm = paddle.nn.LayerList(
                [Norm(config) for _ in range(self.config.multi_token_pred_depth)]
            )
            self.mtp_emb_norm = paddle.nn.LayerList([Norm(config) for _ in range(self.config.multi_token_pred_depth)])

            LinearFN = paddle.incubate.nn.FusedLinear if config.fuse_linear else paddle.nn.Linear
            self.mtp_linear_proj = paddle.nn.LayerList(
                [
                    LinearFN(
                        self.config.hidden_size * 2,
                        self.config.hidden_size,
                        bias_attr=config.use_bias,
                    )
                    for _ in range(self.config.multi_token_pred_depth)
                ]
            )
            if config.sequence_parallel:
                for mtp_linear in self.mtp_linear_proj:
                    mark_as_sequence_parallel_parameter(mtp_linear.weight)
                    if config.use_bias:
                        mark_as_sequence_parallel_parameter(mtp_linear.bias)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @classmethod
    def _prepare_decoder_attention_mask(cls, attention_mask, input_shape, past_key_values_length, dtype):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, past_key_values_length=past_key_values_length, dtype=dtype
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_length=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        combined_attention_mask = paddle.maximum(
            combined_attention_mask.astype(dtype),
            paddle.to_tensor(float(finfo(dtype).min), dtype=dtype),
        )
        return combined_attention_mask

    @paddle.jit.not_to_static
    def recompute_training(
        self,
        layer_module,
        hidden_states,
        attention_mask,
        position_ids,
        token_type_ids,
        output_attentions,
        past_key_value,
        use_cache,
        inbatch_pack_offset,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, output_gate_logits=False)

            return custom_forward

        decoderlayer_act_offload_settings = self.config.get(
            "decoderlayer_act_offload_settings", {"type": "", "value": ""}
        )

        setting_type = decoderlayer_act_offload_settings["type"]
        offload_value = decoderlayer_act_offload_settings["value"]

        def get_offload_kwargs(layer_idx, setting_type, offload_value):
            offload_kwargs = {}
            if "mod" == setting_type:
                assert isinstance(offload_value, (list, tuple))
                v1, v2 = offload_value
                offload_kwargs["offload_indices"] = [0] if layer_idx % v1 == v2 else []
            elif "layer_idxs" == setting_type:
                offload_kwargs["offload_indices"] = [0] if layer_idx in offload_value else []
            return offload_kwargs

        layer_idx = layer_module.layer_idx
        if layer_idx == 0:
            offload_kwargs = {}
        else:
            offload_kwargs = get_offload_kwargs(layer_idx, setting_type, offload_value)

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            attention_mask,
            position_ids,
            token_type_ids,
            output_attentions,
            past_key_value,
            use_cache,
            inbatch_pack_offset,
            **offload_kwargs,
        )
        return hidden_states

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
        inbatch_pack_offset=None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        seq_length -= self.config.multi_token_pred_depth
        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = paddle.shape(past_key_values[0][0])[1]
            seq_length_with_past += cache_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.astype(self.embed_tokens.weight.dtype)

        if self.config.multi_token_pred_depth > 0:
            inputs_embeds_extra = inputs_embeds[:, -self.config.multi_token_pred_depth :, :]
            inputs_embeds = inputs_embeds[:, : -self.config.multi_token_pred_depth, :]
            inputs_embeds_ori = inputs_embeds

        if self.config.sequence_parallel:
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
            inputs_embeds = ScatterOp.apply(inputs_embeds)

        can_use_fa = self.config.use_flash_attn
        can_mem_eff_attn = self.config.use_mem_eff_attn and inbatch_pack_offset is not None
        if can_use_fa or can_mem_eff_attn:
            if attention_mask is not None:
                attention_mask = None

        elif attention_mask is None:
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)

        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                cache_length,
                inputs_embeds.dtype,
            )
        hidden_states = inputs_embeds

        attn_mask_startend_row_indices_ori = attn_mask_startend_row_indices
        if attn_mask_startend_row_indices is not None:
            attn_mask_startend_row_indices = attn_mask_startend_row_indices[
                :, :, : -self.config.multi_token_pred_depth
            ]

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_router_loss = 0.0 if self.config.use_moe else None
        all_gate_logits = ()
        mtp_outputs = []

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            has_gradient = not hidden_states.stop_gradient
            if self.config.use_recompute and has_gradient:
                layer_outputs = self.recompute_training(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    inbatch_pack_offset,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    inbatch_pack_offset,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if self.config.use_moe:
                if not (self.config.use_recompute and has_gradient):
                    layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
                    all_gate_logits = all_gate_logits + (gate_logits,)

        if self.config.multi_token_pred_depth > 0:
            mtp_outputs.append(hidden_states)

            for depth in range(self.config.multi_token_pred_depth):
                if self.config.sequence_parallel:
                    hidden_states = GatherOp.apply(hidden_states)
                    hidden_states = hidden_states.reshape([-1, seq_length, hidden_states.shape[-1]])

                inputs_embeds_cur_depth = paddle.concat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )
                attn_mask_startend_row_indices_cur_depth = None
                if attn_mask_startend_row_indices is not None:
                    attn_mask_startend_row_indices_cur_depth = attn_mask_startend_row_indices_ori[
                        :, :, (depth + 1) : inputs_embeds_ori.shape[1] + (depth + 1)
                    ] - (depth + 1)

                inputs_embeds_cur_depth_norm = self.mtp_emb_norm[depth](inputs_embeds_cur_depth)
                hidden_states_norm = self.mtp_hidden_norm[depth](hidden_states)

                inputs_embeds_cur_depth = self.mtp_linear_proj[depth](
                    paddle.concat([inputs_embeds_cur_depth_norm, hidden_states_norm], axis=-1)
                )

                if self.config.sequence_parallel:
                    inputs_embeds_cur_depth = inputs_embeds_cur_depth.reshape([-1, inputs_embeds_cur_depth.shape[-1]])
                    inputs_embeds_cur_depth = ScatterOp.apply(inputs_embeds_cur_depth)

                decoder_layer = self.mtp_block[depth]
                past_key_value = None
                layer_outputs = decoder_layer(
                    inputs_embeds_cur_depth,
                    attention_mask,
                    position_ids,
                    token_type_ids,
                    output_attentions,
                    past_key_value,
                    use_cache,
                    inbatch_pack_offset,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices_cur_depth,
                )

                if isinstance(layer_outputs, (tuple, list)):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

                if self.config.use_moe:
                    if not (self.config.use_recompute and has_gradient):
                        layer_outputs, gate_logits = (
                            layer_outputs[:-1],
                            layer_outputs[-1],
                        )
                        all_gate_logits = all_gate_logits + (gate_logits,)

                mtp_outputs.append(hidden_states)
            mtp_outputs = [self.norm(hidden_states) for hidden_states in mtp_outputs]
            hidden_states, mtp_outputs = mtp_outputs[0], mtp_outputs[1:]
        else:
            hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_loss,
                    all_gate_logits,
                    mtp_outputs,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
            router_loss=all_router_loss,
            gate_logits=all_gate_logits,
            mtp_outputs=mtp_outputs,
        )


ErnieMoELMHead = ErnieLMHead


class ErniePretrainingCriterion(ErniePretrainingCriterionBase):
    def __init__(self, config, return_tuple=True):
        super(ErniePretrainingCriterion, self).__init__(config, return_tuple=return_tuple)
        self.ignored_index = getattr(config, "ignored_index", -100)
        self.config = config
        self.return_tuple = return_tuple
        self.enable_parallel_cross_entropy = config.tensor_model_parallel_size > 1 and config.tensor_parallel_output

        if self.enable_parallel_cross_entropy:
            logger.info("using parallel cross entropy, take care")
            self.loss_func = fleet.meta_parallel.ParallelCrossEntropy()
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none",
            )

    def forward(self, prediction_scores, masked_lm_labels, router_loss=None, mtp_logits=None):
        if self.config.multi_token_pred_depth > 0:
            masked_lm_labels_ori = masked_lm_labels
            masked_lm_labels = masked_lm_labels[:, : -self.config.multi_token_pred_depth]
            seq_length = masked_lm_labels.shape[1]
        res = super().forward(
            prediction_scores,
            masked_lm_labels,
        )
        global_training_logs = get_global_training_logs()

        if self.config.multi_token_pred_depth > 0:
            global_training_logs.update(mtp_depth_0_loss=res[0].clone().detach())
            mtp_loss_res = []
            for depth in range(self.config.multi_token_pred_depth):
                prediction_scores_cur_depth = mtp_logits[depth]
                masked_lm_labels_cur_depth = masked_lm_labels_ori[:, (depth + 1) : (depth + 1 + seq_length)]
                res_cur_depth = super().forward(
                    prediction_scores_cur_depth,
                    masked_lm_labels_cur_depth,
                )
                mtp_loss_res.append(res_cur_depth)
                global_training_logs.update(**{f"mtp_depth_{depth + 1}_loss": res_cur_depth[0].clone().detach()})

        def add_loss(main_loss, loss):
            return main_loss + loss - loss.detach()

        if self.return_tuple:
            loss, loss_sum = res
            if self.config.multi_token_pred_depth > 0:
                loss = add_loss(
                    loss,
                    self.config.multi_token_pred_lambda * sum([x[0] for x in mtp_loss_res]) / len(mtp_loss_res),
                )
                loss_sum = loss_sum + self.config.multi_token_pred_lambda * sum(
                    [x[1].detach() for x in mtp_loss_res]
                ) / len(mtp_loss_res)
        else:
            loss, loss_sum = res, None
            if self.config.multi_token_pred_depth > 0:
                loss = add_loss(
                    loss,
                    self.config.multi_token_pred_lambda * sum([x[0] for x in mtp_loss_res]) / len(mtp_loss_res),
                )

        global_training_logs.update(lm_loss=loss.clone().detach())
        if router_loss is not None and isinstance(router_loss, paddle.Tensor):
            loss = loss + router_loss - router_loss.detach()
            if isinstance(router_loss, paddle.Tensor):
                global_training_logs.update(router_loss=router_loss.detach())
        return loss, loss_sum


class ErnieMoEForCausalLM(ErniePretrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.sequence_parallel:
            logger.info(f"using sequence_parallel, input seqlen={config.seqlen}")
            assert config.seqlen is not None
            assert (
                config.tensor_model_parallel_size > 1
            ), f"sequence-parallel needs mp>1, got mp={config.tensor_model_parallel_size}"

        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(f"change initializer-range from {config.initializer_range} to {new_initializer_range}")
        config.initializer_range = new_initializer_range
        self.config = config
        self.ernie = ErnieModel(config)
        self.lm_head = ErnieMoELMHead(config)
        self.criterion = ErniePretrainingCriterion(config)

        self.tie_weights()

        if self.config.fuse_rms_norm:
            logger.info("Use fusedRMSNorm")
        else:
            logger.info("Use normal RMSNorm")

    def _post_init(self, original_init, *args, **kwargs):
        super()._post_init(self, original_init, *args, **kwargs)
        factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
        logger.info(f"using post init div: factor:{factor}")
        with paddle.no_grad():
            for layer in self.ernie.layers:
                if self.config.use_linear_residual_norm_recompute is True:
                    layer.fused_linear_add_norm.linear_weight.scale_(factor)
                else:
                    if isinstance(
                        layer.self_attn.o_proj,
                        (MOELayer,),
                    ):
                        for e in layer.self_attn.o_proj.experts:
                            e.weight.scale_(factor)
                        if hasattr(layer.self_attn.o_proj, "dense_experts"):
                            layer.self_attn.o_proj.dense_experts.down_proj.weight.scale_(factor)
                    else:
                        layer.self_attn.o_proj.weight.scale_(factor)

                if isinstance(
                    layer.mlp,
                    (MOELayer,),
                ):
                    for e in layer.mlp.experts:
                        if isinstance(e, ErnieMLP):
                            e.down_proj.weight.scale_(factor)
                    if getattr(layer.mlp, "dense_experts", None) and isinstance(layer.mlp.dense_experts, ErnieMLP):
                        layer.mlp.dense_experts.down_proj.weight.scale_(factor)
                else:
                    layer.mlp.down_proj.weight.scale_(factor)

    def set_state_dict(self, state_dict, *args, **kwargs):
        state_dict = moe_statedict_upcycle(
            state_dict,
            self.config,
            self.lm_head.weight.dtype,
            self._get_tensor_parallel_mappings(self.config, is_split=False),
            self._get_tensor_parallel_mappings(self.config, is_split=True),
        )
        state_dict = moe_statedict_cherry_pick(state_dict, self.config)
        state_dict = moe_ep2mp(
            state_dict,
            self.config,
            self._get_tensor_parallel_mappings(self.config, is_split=True),
        )
        ret = super().set_state_dict(state_dict, *args, **kwargs)
        logger.info(f"set_state_dict: {ret}")
        return ret

    def get_input_embeddings(self):
        return self.ernie.embed_tokens

    def set_input_embeddings(self, value):
        self.ernie.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.ernie = decoder

    def get_decoder(self):
        return self.ernie

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id
        ).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).astype("int64")
        else:
            attention_mask = paddle.ones_like(input_ids, dtype="int64")
        return attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache=False,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        attention_mask = kwargs.get("attention_mask", None)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
        )

        if self.config.rope_3d:
            model_inputs.update({"position_ids": kwargs["position_ids"]})

        return model_inputs

    def update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithCrossAttentions) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat([token_type_ids, token_type_ids[:, -1:]], axis=-1)

        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = paddle.concat(
                    [
                        attention_mask,
                        paddle.ones([attention_mask.shape[0], 1], dtype="int64"),
                    ],
                    axis=-1,
                )
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat([role_ids, role_ids[:, -1:]], axis=-1)

        if self.config.rope_3d:
            assert "position_ids" in model_kwargs, "position_ids must be provided if rope_3d is on"
            position_ids = model_kwargs["position_ids"]

            model_kwargs["position_ids"] = paddle.concat(
                [
                    position_ids,
                    position_ids.max(axis=(1, 2), keepdim=True).tile([1, 1, 3]) + 1,
                ],
                axis=1,
            )

        return model_kwargs

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        ignored_index=0,
        data_id=None,
        src_id=None,
        inbatch_pack_offset=None,
        attn_mask_startend_row_indices=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ernie(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            inbatch_pack_offset=inbatch_pack_offset,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        hidden_states = outputs.last_hidden_state
        mtp_outputs = outputs.mtp_outputs

        logits = self.lm_head(hidden_states)
        mtp_logits = []
        if len(mtp_outputs) > 0:
            mtp_logits = [self.lm_head(_hidden_states) for _hidden_states in mtp_outputs]

        if return_dict:
            if labels is not None:
                loss, _ = self.criterion(logits, labels)
            else:
                loss = None
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_loss=outputs.router_loss if self.config.use_moe else None,
            )
        if self.config.use_moe:
            router_loss = outputs.router_loss
        else:
            router_loss = None
        assert labels is not None
        return self.criterion(logits, labels, router_loss, mtp_logits)

    def sharded_state_dict(self, *args, **kwargs):
        sharded_state_dict = super().sharded_state_dict(*args, **kwargs)

        import re

        def increment_expert_number(s, increment):
            def replace(match):
                original_number = int(match.group(0))
                new_number = original_number + increment
                return str(new_number)

            return re.sub(r"(?<=experts\.)\d+", replace, s)

        renamed_sharded_state_dict = {}
        for k, v in sharded_state_dict.items():
            global_expert_id_offset = getattr(v, "global_expert_id_offset", None)
            if global_expert_id_offset is not None:
                new_key = increment_expert_number(k, global_expert_id_offset)
                v.key = new_key
                delattr(v, "global_expert_id_offset")
                renamed_sharded_state_dict[new_key] = v
            else:
                renamed_sharded_state_dict[k] = v

        return renamed_sharded_state_dict
