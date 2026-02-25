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
import copy
import logging
import math
from collections import deque

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu.mp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.distributed.fleet.utils import recompute

from paddleformers.cli.train.ernie_pretrain.models.ernie import ErnieMoEConfig
from paddleformers.cli.train.ernie_pretrain.models.ernie.modeling_moe import (
    ErnieDecoderLayer,
    ErnieMLP,
    ErnieModel,
    ErnieMoELMHead,
    ErniePretrainedModel,
    ErniePretrainingCriterion,
    RMSNorm,
    RotaryEmbedding,
    _parse_moe_group,
    moe_ep2mp,
    moe_statedict_upcycle,
)
from paddleformers.cli.train.ernie_pretrain.models.moe.moe_layer import MOELayer
from paddleformers.cli.train.ernie_pretrain.models.moe.top2_gate import (
    Top2Gate,
    TopKGateFused,
)
from paddleformers.cli.train.ernie_pretrain.models.sequence_parallel_utils import (
    ColumnSequenceParallelLinear,
    RowSequenceParallelLinear,
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)
from paddleformers.cli.train.ernie_pretrain.models.utils import inplace_offload
from paddleformers.transformers import PretrainedModel

try:
    from paddleformers.cli.train.ernie_pretrain.src.utils.misc import (
        global_training_logs,
    )
except ModuleNotFoundError:
    global_training_logs = {}


try:
    from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
        pipeline_bubble_hooks_,
    )
except ImportError:
    pipeline_bubble_hooks_ = None

try:
    from paddle.framework.recall_error import AADIFF_ERROR
except ImportError:
    AADIFF_ERROR = "CUDA error(1001)"


input_ids_for_mtp = deque()
attn_mask_startend_row_indices_for_mtp = deque()
NativeLinear = nn.Linear

logger = logging.getLogger(__name__)


class ErnieEmbeddingPipe(nn.Layer):
    def __init__(self, config):
        self.sequence_parallel = config.sequence_parallel
        self.use_mem_eff_attn = config.use_mem_eff_attn
        self.config = config

        super(ErnieEmbeddingPipe, self).__init__()
        self.use_moe = config.use_moe
        if config.tensor_model_parallel_size > 1:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    @property
    def embedding_weight(self):
        return self.embed_tokens.weight

    def forward(self, args):
        if isinstance(args, tuple):
            if len(args) == 4:
                input_ids, _, _, attn_mask_startend_row_indices = args
                attention_mask, position_ids, inbatch_pack_offset = None, None, None
            elif len(args) == 3:
                input_ids, attention_mask, position_ids = args
                inbatch_pack_offset, attn_mask_startend_row_indices = None, None
            elif len(args) == 2:
                if self.use_mem_eff_attn:
                    input_ids, inbatch_pack_offset = args
                    position_ids, attention_mask = None, None
                    inbatch_pack_offset.stop_gradient = True
                else:
                    input_ids, attention_mask = args
                    position_ids = None
                    inbatch_pack_offset = None
                attn_mask_startend_row_indices = None

        else:
            input_ids = args
            attention_mask, position_ids, inbatch_pack_offset = None, None, None
            attn_mask_startend_row_indices = None

        if position_ids is not None:
            position_ids.stop_gradient = True

        emb = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)

        if self.config.multi_token_pred_depth > 0:
            if self.config.enable_mtp_magic_send:
                emb = emb[:, : -self.config.multi_token_pred_depth, :]
                if self.sequence_parallel:
                    emb = emb.reshape([-1, emb.shape[-1]])
                    emb = ScatterOp.apply(emb)
            else:
                inputs_embeds_extra = emb[:, -self.config.multi_token_pred_depth :, :]
                inputs_embeds = emb[:, : -self.config.multi_token_pred_depth, :]
                inputs_embeds_ori = inputs_embeds
                batch_size, seq_length, _ = inputs_embeds.shape

                if self.sequence_parallel:
                    inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
                    inputs_embeds = ScatterOp.apply(inputs_embeds)
                mtp_emb_res = [inputs_embeds]
                for depth in range(self.config.multi_token_pred_depth):
                    inputs_embeds_mtp = paddle.concat(
                        [
                            inputs_embeds_ori[:, (depth + 1) :, :],
                            inputs_embeds_extra[:, : (depth + 1), :],
                        ],
                        axis=1,
                    )
                    if self.sequence_parallel:
                        inputs_embeds_mtp = inputs_embeds_mtp.reshape([-1, inputs_embeds_mtp.shape[-1]])
                        inputs_embeds_mtp = ScatterOp.apply(inputs_embeds_mtp)
                    mtp_emb_res.append(inputs_embeds_mtp)
                res = paddle.concat(mtp_emb_res)
                return [res]
        else:
            if self.sequence_parallel:
                emb = emb.reshape([-1, emb.shape[-1]])
                emb = ScatterOp.apply(emb)

        if attention_mask is not None:
            batch_size, seq_length = input_ids.shape
            attention_mask = ErnieModel._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), 0, emb.dtype
            )
            attention_mask.stop_gradient = True

        if attn_mask_startend_row_indices is not None:
            if self.config.multi_token_pred_depth > 0:
                attn_mask_startend_row_indices = attn_mask_startend_row_indices[
                    :, :, : -self.config.multi_token_pred_depth
                ].contiguous()

        ret = (emb,)
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset.clone(),)
        if attn_mask_startend_row_indices is not None:
            ret += (paddle.empty(0), paddle.empty(0), attn_mask_startend_row_indices)
        if self.config.multi_token_pred_depth > 0 and not self.config.enable_mtp_magic_send:
            ret += (input_ids,)
            assert len(ret) == 2, "mtp only support one input which is input_ids"
        if len(ret) == 1:
            ret = ret[0]
        return ret


class MTPEmbeddingPipe(ErnieEmbeddingPipe):
    def __init__(self, config):
        super(MTPEmbeddingPipe, self).__init__(config)

    @property
    def embedding_weight(self):
        return self.embed_tokens.weight

    def forward(self, args):
        assert (
            self.config.enable_mtp_magic_send
        ), "MTPEmbedding can only be added into model only support enable_mtp_magic_send=True"

        global input_ids_for_mtp, attn_mask_startend_row_indices_for_mtp
        assert len(input_ids_for_mtp) > 0, "input_ids for mtp is empty"
        hidden_states = args[0]
        input_ids = input_ids_for_mtp.popleft()
        input_embeds = self.embed_tokens(input_ids).astype(self.embed_tokens.weight.dtype)
        ret = (hidden_states, input_embeds)

        if len(args) == 4:
            assert len(attn_mask_startend_row_indices_for_mtp) > 0, "attn_mask_startend_row_indices for mtp is empty"
            attn_mask_startend_row_indices = attn_mask_startend_row_indices_for_mtp.popleft()
            ret += (attn_mask_startend_row_indices,)
        return ret


class EmptyLayer(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ErnieDecoderLayerPipe(ErnieDecoderLayer):
    def __init__(self, config, layer_idx, use_full_recompute=False):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.use_full_recompute = use_full_recompute
        logger.info(f"using pp full recompute={use_full_recompute}")
        self.use_mem_eff_attn = config.use_mem_eff_attn

    def forward(self, args):
        if self.config.multi_token_pred_depth > 0 and not self.config.enable_mtp_magic_send:
            res = args[0]
            tensor_list = paddle.split(res, self.config.multi_token_pred_depth + 1)
            inputs_embeds = tensor_list[-self.config.multi_token_pred_depth :]
            args = tuple(tensor_list[: -self.config.multi_token_pred_depth])
        else:
            res = None

        if isinstance(args, tuple):
            if len(args) == 4:
                hidden_states, _, _, attn_mask_startend_row_indices = args
                attention_mask, position_ids, inbatch_pack_offset = None, None, None
            elif len(args) == 3:
                hidden_states, attention_mask, position_ids = args
                attn_mask_startend_row_indices = None
            elif len(args) == 2:
                if self.use_mem_eff_attn:
                    hidden_states, inbatch_pack_offset = args
                    position_ids, attention_mask = None, None
                    inbatch_pack_offset.stop_gradient = True
                else:
                    hidden_states, attention_mask = args
                    position_ids, inbatch_pack_offset = None, None
                attn_mask_startend_row_indices = None
            elif len(args) == 1:
                (hidden_states,) = args
                attention_mask, position_ids, inbatch_pack_offset = None, None, None
                attn_mask_startend_row_indices = None
        else:
            hidden_states = args
            attention_mask, position_ids, inbatch_pack_offset = None, None, None
            attn_mask_startend_row_indices = None

        if position_ids is not None:
            position_ids.stop_gradient = True

        if attention_mask is not None:
            attention_mask.stop_gradient = True

        if self.training and self.use_full_recompute:
            decoderlayer_act_offload_settings = self.config.get(
                "decoderlayer_act_offload_settings", {"type": "", "value": ""}
            )
            setting_type = decoderlayer_act_offload_settings["type"]
            offload_value = decoderlayer_act_offload_settings["value"]
            offload_kwargs = {}
            if "mod" == setting_type:
                assert isinstance(offload_value, (list, tuple))
                v1, v2 = offload_value
                offload_kwargs["offload_indices"] = [0] if self.layer_idx % v1 == v2 else []
            elif "layer_idxs" == setting_type:
                offload_kwargs["offload_indices"] = [0] if self.layer_idx in offload_value else []

            if offload_kwargs.get("offload_indices", []) and res is not None:
                inplace_offload(res)

            ret = recompute(
                super().forward,
                hidden_states,
                attention_mask,
                position_ids,
                None,
                False,
                None,
                False,
                inbatch_pack_offset,
                False,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                **offload_kwargs,
            )
        else:
            ret = super().forward(
                hidden_states,
                attention_mask,
                position_ids,
                None,
                False,
                None,
                False,
                inbatch_pack_offset,
                False,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            )
        if isinstance(ret, paddle.Tensor):
            ret = (ret,)
        if attention_mask is not None:
            ret += (attention_mask.clone(),)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset.clone(),)
        if attn_mask_startend_row_indices is not None:
            ret += (paddle.empty(0), paddle.empty(0), attn_mask_startend_row_indices)
        if len(ret) == 1:
            (ret,) = ret
        if self.config.multi_token_pred_depth > 0:
            if self.config.enable_mtp_magic_send:
                ret = (ret,) if isinstance(ret, paddle.Tensor) else ret
            else:
                ret = (paddle.concat([ret, *inputs_embeds]),)
        return ret


class RMSNormPipe(RMSNorm):
    def __init__(self, config):
        super().__init__(config)
        self.use_moe = config.use_moe
        mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, args):
        if self.config.multi_token_pred_depth > 0:
            if self.config.enable_mtp_magic_send:
                assert len(args) == self.config.multi_token_pred_depth + 1, "the length is not valid in mtp"
                mtp_outputs = []
                for hidden_states in args:
                    mtp_outputs.append(super().forward(hidden_states))
                return mtp_outputs
            else:
                tensor_list = paddle.split(args[0], self.config.multi_token_pred_depth + 1)
                mtp_outputs = []
                for hidden_states in tensor_list:
                    mtp_outputs.append(super().forward(hidden_states))
                return mtp_outputs
        else:
            if self.use_moe:
                hidden_states = args[:1]
            if isinstance(args, tuple):
                if len(args) == 3:
                    hidden_states, attention_mask, position_ids = args
                elif len(args) == 2:
                    hidden_states, attention_mask = args
            else:
                hidden_states = args
            hidden_states = super().forward(hidden_states)
            return hidden_states


class ErnieMoELMHeadPipe(ErnieMoELMHead):
    def forward(self, args):
        if self.config.multi_token_pred_depth > 0:
            logits = list()
            for _hidden_states in args:
                logits.append(super().forward(_hidden_states))
            return logits
        hidden_states = args
        logits = super().forward(hidden_states)
        return logits


class MTPLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        config = copy.deepcopy(config)
        self.config = config
        if self.config.use_recompute_mtp:
            self.config.use_recompute = False
        assert self.config.multi_token_pred_depth > 0, "Adding MTPLayer must assign value to multi_token_pred_depth"

        self.mtp_block = paddle.nn.LayerList(
            [ErnieDecoderLayer(config, layer_idx) for layer_idx in range(self.config.multi_token_pred_depth)]
        )
        Norm = RMSNorm
        self.mtp_hidden_norm = paddle.nn.LayerList([Norm(config) for _ in range(self.config.multi_token_pred_depth)])
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

    def forward(self, args):
        def custom_forward(*inputs):
            return self.forward_impl(*inputs)

        if self.config.use_recompute_mtp:
            return recompute(custom_forward, *args)
        else:
            return custom_forward(*args)

    def forward_impl(self, *args):
        if self.config.enable_mtp_magic_send:
            assert isinstance(args, tuple), "Input for MTPLayer must be tuple"
            if len(args) == 3:
                hidden_states, inputs_embeds, attn_mask_startend_row_indices = args
            else:
                hidden_states, inputs_embeds = args
                attn_mask_startend_row_indices = None
            inputs_embeds_extra = inputs_embeds[:, -self.config.multi_token_pred_depth :, :]
            inputs_embeds = inputs_embeds[:, : -self.config.multi_token_pred_depth, :]
            inputs_embeds_ori = inputs_embeds
        else:
            res = args[0]
            tensor_list = paddle.split(res, self.config.multi_token_pred_depth + 1)
            hidden_states = tensor_list[0]
            inputs_embeds_cur_depth_list = tensor_list[1:]
            attn_mask_startend_row_indices = None

        output_list = [hidden_states]
        for depth in range(self.config.multi_token_pred_depth):
            if self.config.enable_mtp_magic_send:
                inputs_embeds_cur_depth = paddle.concat(
                    [
                        inputs_embeds_ori[:, (depth + 1) :, :],
                        inputs_embeds_extra[:, : (depth + 1), :],
                    ],
                    axis=1,
                )

                if self.config.sequence_parallel:
                    inputs_embeds_cur_depth = inputs_embeds_cur_depth.reshape([-1, inputs_embeds_cur_depth.shape[-1]])
                    inputs_embeds_cur_depth = ScatterOp.apply(inputs_embeds_cur_depth)
            else:
                inputs_embeds_cur_depth = inputs_embeds_cur_depth_list[depth]

            inputs_embeds_cur_depth_norm = self.mtp_emb_norm[depth](inputs_embeds_cur_depth)
            hidden_states_norm = self.mtp_hidden_norm[depth](hidden_states)

            inputs_embeds_cur_depth = self.mtp_linear_proj[depth](
                paddle.concat([inputs_embeds_cur_depth_norm, hidden_states_norm], axis=-1)
            )

            attn_mask_startend_row_indices_cur_depth = None
            if attn_mask_startend_row_indices is not None:
                attn_mask_startend_row_indices_cur_depth = attn_mask_startend_row_indices[
                    :, :, (depth + 1) : inputs_embeds_ori.shape[1] + (depth + 1)
                ] - (depth + 1)

            decoder_layer = self.mtp_block[depth]

            layer_outputs = decoder_layer(
                inputs_embeds_cur_depth,
                None,
                None,
                None,
                False,
                None,
                False,
                None,
                False,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices_cur_depth,
            )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            output_list.append(hidden_states)

        if self.config.enable_mtp_magic_send:
            return tuple(output_list)
        else:
            res = paddle.concat(output_list)
            return (res,)


class ErniePretrainingCriterionPipe(ErniePretrainingCriterion):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, logits, labels):
        if self.config.multi_token_pred_depth > 0:
            mtp_logits = logits[1:]
            logits = logits[0]
            loss, loss_sum = super().forward(logits, labels, mtp_logits=mtp_logits)
            if not self.training:
                return loss_sum
            return loss
        loss, loss_sum = super().forward(logits, labels)
        if not self.training:
            return loss_sum
        return loss


class PipelinePretrainedModel(PretrainedModel):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        super().__init__(config, *args, **kwargs)

    def init(self, config, *args, **kwargs):
        self._sequential_layers = []
        self._pipeline_name_mapping = None
        self._pp_to_single_mapping = None

    def add_sequential_layer(self, layer_desc, name_prefix=""):
        self._sequential_layers.append({"layer": layer_desc, "name_prefix": name_prefix})

    def get_sequential_layers(self):
        return [x["layer"] for x in self._sequential_layers]

    def get_sequential_name_prefixs(self):
        return {str(index): x["name_prefix"] for index, x in enumerate(self._sequential_layers)}

    def get_shardlayer_prefix(self, name_splited):
        shared_layer_names = {s.layer_name for s in self._layers_desc if isinstance(s, SharedLayerDesc)}
        assert name_splited[1] in shared_layer_names, f"The shared layer name {name_splited[1]} must be in prefixes!"
        shared_layer_key = name_splited[1]
        for idx, layer in enumerate(self._layers_desc):
            if isinstance(layer, SharedLayerDesc) and layer.layer_name == shared_layer_key:
                if self.get_stage_from_index(idx) == self._stage_id:
                    return self.get_sequential_name_prefixs()[str(idx)]

        raise ValueError(f"The shared layer {shared_layer_key} must be in the current stage!")

    def _set_pipeline_name_mapping(self, mappings=None):
        if mappings is not None:
            self._pipeline_name_mapping = mappings
        else:
            single_to_pp_mapping = {}
            pp_to_single_mapping = {}

            state_dict_keys = list(super().state_dict().keys())
            first_key = ""
            for k in state_dict_keys:
                if "shared_layers" not in k:
                    first_key = k
                    break
            first_key = first_key.split(".")
            use_virtual_pipeline_model_parallel_size = first_key[0].isdigit() and first_key[1].isdigit()

            prefixes = self.get_sequential_name_prefixs()
            for k in state_dict_keys:
                name_splited = k.split(".")
                if use_virtual_pipeline_model_parallel_size:
                    if name_splited[0].isdigit():
                        if name_splited[1].isdigit():
                            idx = str(int(name_splited[0]) + int(name_splited[1]))
                            single_name = [prefixes[idx]]
                            single_name.extend(name_splited[2:])
                        else:
                            single_name = [prefixes[str(len(prefixes) - 1)]]
                            single_name.extend(name_splited[2:])
                            logger.warning(
                                f"Please check! we treat this key as last layer, get {k}, \
                                        set origin name as {'.'.join(single_name)}"
                            )
                    elif name_splited[0] == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue
                else:
                    idx = name_splited[0]
                    if idx.isdigit():
                        single_name = [] if prefixes[idx] == "" else [prefixes[idx]]
                        single_name.extend(name_splited[1:])
                    elif idx == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue

                single_to_pp_mapping[".".join(single_name)] = k
                pp_to_single_mapping[k] = ".".join(single_name)

            self._pipeline_name_mapping = single_to_pp_mapping
            self._pp_to_single_mapping = pp_to_single_mapping

        return self._pipeline_name_mapping

    def _check_shared_model_state(self):
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        super_state_dict = super().state_dict()
        structure_name_to_tensor = {}
        for k, v in super_state_dict.items():
            k = self._pp_to_single_mapping[k]
            if k not in structure_name_to_tensor:
                structure_name_to_tensor[k] = v
            else:
                old_v = structure_name_to_tensor[k]
                assert old_v is v, f"Shared tensor with different structure name: {k}"

        missing_shared_keys = {}
        for k, v in self._pp_to_single_mapping.items():
            mapped_k = self._pipeline_name_mapping[v]
            if k != mapped_k:
                missing_shared_keys[k] = mapped_k
        return missing_shared_keys

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict[self._pp_to_single_mapping[k]] = v

        return state_dict

    def _init_weights(self, layer):
        if self.config.tensor_model_parallel_size > 1:
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
                ErnieMoELMHead,
                nn.Embedding,
                NativeLinear,
                paddle.incubate.nn.FusedLinear,
            ),
        ):
            is_moe = getattr(layer.weight, "no_sync", False)
            with rng_tracker("local_seed" if is_moe else "model_parallel_rng"):
                dtype = paddle.get_default_dtype()
                paddle.set_default_dtype("float32")
                layer.weight.set_value(
                    paddle.randn(layer.weight.shape, dtype=dtype).scale(self.config.initializer_range)
                )
                paddle.set_default_dtype(dtype)

        elif isinstance(layer, (Top2Gate, TopKGateFused)):
            if not hasattr(layer, "weight"):
                return
            with rng_tracker("model_parallel_rng"):
                dtype = paddle.get_default_dtype()
                paddle.set_default_dtype("float32")
                moe_num_experts = self.config.moe_num_experts
                if isinstance(moe_num_experts, (list, tuple)):
                    moe_num_experts = moe_num_experts[0]
                if self.config.moe_group_experts:
                    layer.weight.set_value(
                        paddle.randn(layer.weight.shape, dtype=layer.weight.dtype).scale(self.config.initializer_range)
                    )
                else:
                    layer.weight.set_value(
                        paddle.randn(
                            [self.config.hidden_size, moe_num_experts],
                            dtype="float32",
                        ).scale(self.config.initializer_range)
                    )
                if isinstance(self.config.moe_num_experts, (tuple, list)):
                    for i in range(1, len(self.config.moe_num_experts)):
                        layer_weight = getattr(layer, f"weight_{i}")
                        layer_weight.set_value(
                            paddle.randn(layer_weight.shape, dtype=layer_weight.dtype).scale(
                                self.config.initializer_range
                            )
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

    def sharded_state_dict(self, *args, **kwargs):
        sharded_state_dict = super().sharded_state_dict(*args, **kwargs)
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        for k in list(sharded_state_dict.keys()):
            v = sharded_state_dict.pop(k)
            v.key = self._pp_to_single_mapping[k]
            sharded_state_dict[self._pp_to_single_mapping[k]] = v

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


def get_pp_vp_split_layers(config):
    hcg = fleet.get_hybrid_communicate_group()
    pp_size = max(hcg.get_pipe_parallel_world_size(), 1)
    vp_size = max(config.virtual_pipeline_model_parallel_size, 1)
    layer_num = config.num_hidden_layers
    selective_no_recompute_num = config.selective_no_recompute_num

    no_recompute_layer_num = []
    if selective_no_recompute_num == 0:
        return set(no_recompute_layer_num)

    assert layer_num % (pp_size * vp_size) == 0, (
        "layer_num must be divisible by pp_size * vp_size,"
        f" but got layer_num: {layer_num}, pp_size: {pp_size}, vp_size: {vp_size}"
    )

    chunk_size = layer_num // (pp_size * vp_size)
    chunk_list = [list(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(pp_size * vp_size)]

    stage_chunk_list = [[] for _ in range(pp_size)]
    for i in range(pp_size * vp_size):
        stage_chunk_list[i % pp_size].append(chunk_list[i])

    if config.use_recompute_attn:
        logger.error("selective recompute only support full recompute now, please set use_recompute_attn to False")

    for i in range(pp_size):
        no_recompute_layer_num.extend(stage_chunk_list[i][-selective_no_recompute_num:])

    return set(sum(no_recompute_layer_num, []))


class ErnieMoEForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):

    config_class = ErnieMoEConfig
    _get_tensor_parallel_mappings = ErniePretrainedModel._get_tensor_parallel_mappings

    ErnieEmbeddingPipeClass = ErnieEmbeddingPipe
    ErnieDecoderLayerPipeClass = ErnieDecoderLayerPipe
    MTPEmbeddingPipeClass = MTPEmbeddingPipe
    MTPLayerClass = MTPLayer
    RMSNormPipeClass = RMSNormPipe
    ErnieMoELMHeadPipeClass = ErnieMoELMHeadPipe

    @classmethod
    def _prepare_pipeline_inputs_func(cls, data):
        global input_ids_for_mtp, attn_mask_startend_row_indices_for_mtp
        input_ids_for_mtp.clear()
        attn_mask_startend_row_indices_for_mtp.clear()
        for d in data:
            assert "input_ids" in d
            input_ids_for_mtp.append(d["input_ids"])
            if "attn_mask_startend_row_indices" in d:
                attn_mask_startend_row_indices_for_mtp.append(d["attn_mask_startend_row_indices"])

        if "attn_mask_startend_row_indices" in data[0]:
            inputs = (
                [d["input_ids"] for d in data],
                [paddle.empty(0) for _ in data],  # placeholder
                [paddle.empty(0) for _ in data],  # placeholder
                [d["attn_mask_startend_row_indices"] for d in data],
            )
        else:
            inputs = tuple(
                [d[k] for d in data]
                for k in [
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "inbatch_pack_offset",
                ]
                if k in data[0]
            )

        if len(inputs) == 1:
            inputs = inputs[0]
        labels = [d["labels"] for d in data]
        return inputs, labels

    def __init__(
        self,
        config,
    ):
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        config.initializer_range = new_initializer_range

        if config.moe_group == "mp":
            assert config.sequence_parallel

        if config.moe_group in {"mp", "model", "tp", "mpdp"}:
            assert config.sequence_parallel
            logger.info(f"disable FFN tensor model parallel, moe-group={config.moe_group}")
            config.disable_ffn_model_parallel = True

        config.moe_group = _parse_moe_group(config.moe_group)
        config.moe_world_size = dist.get_world_size(config.moe_group)
        if config.moe_world_size < 0:
            config.moe_world_size = 1
        config.moe_rank = dist.get_rank(config.moe_group)

        self.config = config

        hcg = fleet.get_hybrid_communicate_group()
        tensor_model_parallel_size = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)
        logger.info(f"using vpp={config.virtual_pipeline_model_parallel_size}")
        if config.sequence_parallel:
            logger.info(f"using sequence_parallel, input seqlen={config.seqlen}")
            assert config.seqlen is not None
            assert (
                config.tensor_model_parallel_size > 1
            ), f"sequence-parallel needs mp>1, got mp={config.tensor_model_parallel_size}"

        config.tensor_model_parallel_size = tensor_model_parallel_size
        config.tensor_parallel_rank = tensor_parallel_rank
        PipelinePretrainedModel.init(self, config=config)

        if config.pp_no_recompute_layer is not None:
            no_recompute_layers = config.pp_no_recompute_layer
        else:
            no_recompute_layers = get_pp_vp_split_layers(config)
        logger.info(f"use no_recompute_layers: {no_recompute_layers}")

        def _need_full_recompute(layer_idx):
            return layer_idx not in no_recompute_layers and config.use_recompute

        insert_empty_layer = config.insert_empty_layer
        if len(insert_empty_layer) > 0:
            assert min(insert_empty_layer) >= 0, "cannot insert empty layer as first layer of the model"
            assert max(insert_empty_layer) < config.num_hidden_layers, "empty layers location exceed the num layers"
        logger.info(f"use insert_empty_layer: {insert_empty_layer}")

        if config.multi_token_pred_depth == 0:
            self.add_sequential_layer(LayerDesc(self.ErnieEmbeddingPipeClass, config=config), "ernie")
        else:
            if config.enable_mtp_magic_send:
                self.add_sequential_layer(
                    SharedLayerDesc(
                        key="embed_weight_share",
                        layer_func=self.ErnieEmbeddingPipeClass,
                        shared_weight_attr="embedding_weight",
                        config=config,
                    ),
                    "ernie.embed",
                )
            else:
                self.add_sequential_layer(LayerDesc(self.ErnieEmbeddingPipeClass, config=config), "ernie")

        num_empty_layers = (
            config.num_empty_layers_add_in_tail if isinstance(config.num_empty_layers_add_in_tail, int) else 1
        )
        for i in range(config.num_hidden_layers):
            self.add_sequential_layer(
                LayerDesc(
                    self.ErnieDecoderLayerPipeClass,
                    config=config,
                    layer_idx=i,
                    use_full_recompute=_need_full_recompute(i),
                ),
                f"ernie.layers.{i}",
            )
            if i in insert_empty_layer:
                self.add_sequential_layer(
                    LayerDesc(
                        EmptyLayer,
                    ),
                    f"empty.layers.{i}",
                )

        if config.multi_token_pred_depth > 0:
            if config.enable_mtp_magic_send:
                self.add_sequential_layer(
                    SharedLayerDesc(
                        key="embed_weight_share",
                        layer_func=self.MTPEmbeddingPipeClass,
                        shared_weight_attr="embedding_weight",
                        config=config,
                    ),
                    "embed_share",
                )
            self.add_sequential_layer(LayerDesc(self.MTPLayerClass, config=config), "ernie")
            num_empty_layers = num_empty_layers - config.multi_token_pred_depth

        if config.num_empty_layers_add_in_tail:
            for n in range(num_empty_layers):
                self.add_sequential_layer(
                    LayerDesc(
                        EmptyLayer,
                    ),
                    f"empty.layers.{n}",
                )

        i = config.num_hidden_layers
        if i in insert_empty_layer:
            self.add_sequential_layer(
                LayerDesc(
                    EmptyLayer,
                ),
                f"empty.layers.{i}",
            )

        self.add_sequential_layer(
            LayerDesc(self.RMSNormPipeClass, config=config),
            "ernie.norm",
        )

        self.add_sequential_layer(LayerDesc(self.ErnieMoELMHeadPipeClass, config=config), "lm_head")

        recompute_interval = 0

        seg_method = "layer:ErnieDecoderLayer|EmptyLayer|MTPLayer"
        if config.num_hidden_layers % fleet.get_hybrid_communicate_group().topology().get_dim_size("pipe") != 0:
            seg_method = "uniform"
        logger.info(f"using recompute_interval={recompute_interval}, seg_method={seg_method}")

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=self.get_loss_fn(config),
            topology=fleet.get_hybrid_communicate_group().topology(),
            seg_method=seg_method,
            recompute_interval=recompute_interval,
            recompute_ctx={
                "mp_group": fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                "offload": False,
                "partition": False,
            },
            num_virtual_pipeline_stages=config.virtual_pipeline_model_parallel_size,
        )

    def get_loss_fn(self, config):
        return ErniePretrainingCriterionPipe(config)

    def rename_model_params(self, func):
        if self.config.virtual_pipeline_model_parallel_size == 1:
            _layers = iter(self.run_function)
        else:
            _layers = (cc for c in self._model_chunks for cc in c.run_function)
        func(self.config, _layers)

    def fp8_quant_weight(self):
        with paddle.no_grad():
            for i, layer in self._sub_layers.items():
                if isinstance(layer, ErnieDecoderLayer) and hasattr(layer, "fp8_quant_weight"):
                    layer.fp8_quant_weight()

    def _post_init(self, original_init, *args, **kwargs):
        super()._post_init(self, original_init, *args, **kwargs)
        with paddle.no_grad():
            for i, layer in self._sub_layers.items():
                if isinstance(layer, ErnieDecoderLayer):
                    factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                    if self.config.use_linear_residual_norm_recompute:
                        layer.fused_linear_add_norm.linear_weight.scale_(factor)
                    else:
                        layer.self_attn.o_proj.weight.scale_(factor)
                    if isinstance(layer.mlp, (MOELayer)):
                        for e in layer.mlp.experts:
                            if isinstance(e, ErnieMLP):
                                e.down_proj.weight.scale_(factor)
                    else:
                        layer.mlp.down_proj.weight.scale_(factor)

    def set_state_dict(self, state_dict, *args, **kwargs):
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()

        layer_idxs = []
        if self.config.virtual_pipeline_model_parallel_size == 1:
            _layers = iter(self.run_function)
        else:
            _layers = (cc for c in self._model_chunks for cc in c.run_function)

        for layer in _layers:
            if isinstance(layer, self.ErnieDecoderLayerPipeClass):
                layer_idxs.append(layer.layer_idx)
        logger.info(f"this pipeline stage has ErnieDecoderLayers: {layer_idxs}")
        if not self.parameters():
            logger.info("this pipe not need param, skip set state-dict")
            return {}, {}
        state_dict = moe_statedict_upcycle(
            state_dict,
            self.config,
            next(iter(self.parameters())).dtype,
            self._get_tensor_parallel_mappings(self.config, is_split=False),
            self._get_tensor_parallel_mappings(self.config, is_split=True),
            layer_idxs,
        )
        state_dict = moe_ep2mp(
            state_dict,
            self.config,
            self._get_tensor_parallel_mappings(self.config, is_split=True),
        )

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if k not in self._pipeline_name_mapping:
                continue
            state_dict[self._pipeline_name_mapping[k]] = v
        missing_keys, mismatch_keys = super().set_state_dict(state_dict, *args, **kwargs)

        missing_shared_keys = self._check_shared_model_state()
        tmp_missing_keys = []
        for key in missing_keys:
            if key in missing_shared_keys and missing_shared_keys[key] not in missing_keys:
                continue
            tmp_missing_keys.append(key)
        missing_keys = tmp_missing_keys

        logger.info(f"moe_set_state_dict: {missing_keys}, {mismatch_keys}")
        return missing_keys, mismatch_keys
