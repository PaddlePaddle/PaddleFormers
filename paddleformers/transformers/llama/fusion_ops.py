# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F

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

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None
try:
    if get_env_device() in ["npu", "mlu", "gcu"]:
        from paddle.base import core

        for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
            if lib.endswith(".so"):
                paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(lib)
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

from ..refined_recompute import no_recompute
from ..ring_flash_attention import RingFlashAttention


def fusion_rope(
    query_states,
    key_states,
    value_states,
    hidden_states,
    position_ids,
    past_key_value,
    rotary_emb,
    context_parallel_degree=-1,
):
    if get_env_device() not in ["gcu", "intel_hpu"]:
        assert past_key_value is None, "fuse rotary not support cache kv for now"
    batch_size, seq_length, num_heads, head_dim = query_states.shape
    _, kv_seq_len, num_key_value_heads, _ = key_states.shape
    if context_parallel_degree > 1:
        assert get_env_device() == "gpu", "context parallel only support cuda device for now"
        kv_seq_len *= context_parallel_degree
    if get_env_device() not in ["gcu", "intel_hpu"]:
        cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
    if get_env_device() == "npu":
        query_states = core.eager._run_custom_op("fused_rope", query_states, cos, sin)[0]
        key_states = core.eager._run_custom_op("fused_rope", key_states, cos, sin)[0]
    elif get_env_device() == "intel_hpu":
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]
        cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
        cos = cos.squeeze().unsqueeze(0).unsqueeze(0)
        sin = sin.squeeze().unsqueeze(0).unsqueeze(0)
        query_states, _, _ = paddle.incubate.nn.functional.fused_rotary_position_embedding(
            paddle.transpose(query_states, [0, 2, 1, 3]), None, None, sin=sin, cos=cos, position_ids=position_ids
        )
        key_states, _, _ = paddle.incubate.nn.functional.fused_rotary_position_embedding(
            paddle.transpose(key_states, [0, 2, 1, 3]), None, None, sin=sin, cos=cos, position_ids=position_ids
        )
        query_states = paddle.transpose(query_states, [0, 2, 1, 3])
        key_states = paddle.transpose(key_states, [0, 2, 1, 3])
    elif get_env_device() == "gcu":
        cos_sin = rotary_emb.get_fused_cos_sin(value_states, seq_len=kv_seq_len)
        query_states, key_states = core.eager._run_custom_op(
            "fused_rotary_embedding_gcu", query_states, key_states, cos_sin, position_ids, True
        )
    else:
        # paddle version > 2.6 or develop support q and k/v with different num_heads
        paddle_version = float(paddle.__version__[:3])
        if ((paddle_version != 0.0) and (paddle_version <= 2.6)) and (num_heads != num_key_value_heads):
            query_states, _, _ = fused_rotary_position_embedding(
                query_states,
                None,
                None,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
                use_neox_rotary_style=False,
            )
            key_states, _, _ = fused_rotary_position_embedding(
                key_states,
                None,
                None,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
                use_neox_rotary_style=False,
            )
        else:
            query_states, key_states, _ = fused_rotary_position_embedding(
                query_states,
                key_states,
                v=None,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
                use_neox_rotary_style=False,
            )
    return query_states, key_states


def fusion_rms_norm(hidden_states, weight, variance_epsilon):
    if get_env_device() == "npu":
        return core.eager._run_custom_op("rms_norm_npu", hidden_states, weight, variance_epsilon)[0]
    if get_env_device() == "mlu":
        return core.eager._run_custom_op("rms_norm_mlu", hidden_states, weight, variance_epsilon)[0]
    elif get_env_device() == "gcu":
        return core.eager._run_custom_op("rms_norm_gcu", hidden_states, weight, variance_epsilon)[0]
    elif get_env_device() == "intel_hpu":
        return paddle.incubate.nn.functional.fused_rms_norm(
            hidden_states, weight, None, variance_epsilon, hidden_states.dim() - 1
        )[0]
    elif get_env_device() == "xpu":
        try:
            import paddle_xpu_nn  # noqa: F821

            return paddle_xpu_nn.xpu_rms_norm(hidden_states, weight, variance_epsilon)[0]
        except ImportError:
            raise NotImplementedError(
                f"Implementation of fused_rms_norm is not available on {get_env_device()}. Please install paddle_xpu to use this feature"
            )
    return paddle.incubate.nn.functional.fused_rms_norm_ext(hidden_states, weight, variance_epsilon)[0].astype(
        weight.dtype
    )


def fusion_flash_attention(
    query_states,
    config,
    key_states,
    value_states,
    attention_mask,
    output_attentions,
    alibi=None,
    attn_mask_startend_row_indices=None,
    sequence_parallel=False,
    reshard_layer=None,
    npu_is_casual=False,
    skip_recompute=False,
):
    # Note:
    # 1. The head_dim of query_states and key_states should be the same. And the head_dim of value_states should be used for reshape.
    bsz, q_len, num_heads, _ = query_states.shape
    _, kv_seq_len, _, head_dim = value_states.shape
    version = paddle.version.full_version
    if version != "0.0.0" and version <= "2.5.2":
        if alibi is not None:
            raise ValueError("Flash Attention doesn't support alibi")
        if config.context_parallel_degree > 1:
            raise ValueError(f"Context parallel is not implemented in version {version}")
        attn_output, attn_weights = flash_attention(
            query_states,
            key_states,
            value_states,
            causal=True,
            return_softmax=output_attentions,
        )
    else:
        if alibi is not None:
            alibi = alibi.reshape([bsz, num_heads, 1, -1])
            attention_mask = attention_mask.cast(alibi.dtype) + alibi
        if get_env_device() == "npu":
            if config.context_parallel_degree > 1:
                raise ValueError("Context parallel is not implemented for npu")
            attn_output = core.eager._run_custom_op(
                "flash_attention_npu",
                query_states,
                key_states,
                value_states,
                None,
                attention_mask,
                None,
                None,
                0.0,
                attention_mask is None,
                True,
                False,
                npu_is_casual,
                False,
            )[0]
        elif get_env_device() == "gcu":
            if config.context_parallel_degree > 1:
                raise ValueError("Context parallel is not implemented for gcu")
            attn_output = core.eager._run_custom_op(
                "fused_sdp_flash_attention_gcu",
                query_states,
                key_states,
                value_states,
                attention_mask,
                0.0,
                attention_mask is None,
                True,
            )[0]
        elif get_env_device() == "intel_hpu":
            if config.context_parallel_degree > 1:
                raise ValueError("Context parallel is not implemented for intel_hpu")
            scaling_factor = query_states.shape[3] ** -0.5
            attention_mask = attention_mask.astype(query_states.dtype)
            attn_output = paddle.incubate.nn.functional.fused_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                0.0,
                attention_mask is None,
                scaling_factor,
                False,
            )
        else:
            if config.context_parallel_degree > 1:
                attn_output = RingFlashAttention.apply(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=None,
                    is_causal=True,
                )
            else:
                if attn_mask_startend_row_indices is not None:
                    assert alibi is None, "flashmask_attention or flash_attention_with_sparse_mask not support alibi"
                    if len(attn_mask_startend_row_indices.shape) == 2:
                        attn_mask_startend_row_indices = paddle.unsqueeze(attn_mask_startend_row_indices, axis=1)

                    if hasattr(F, "flashmask_attention"):
                        attn_output = no_recompute(
                            F.flashmask_attention,
                            query_states,
                            key_states,
                            value_states,
                            startend_row_indices=attn_mask_startend_row_indices.unsqueeze(-1),
                            causal=True,
                            enable=skip_recompute,
                        )
                    else:
                        attn_output = no_recompute(
                            F.flash_attention_with_sparse_mask,
                            query_states,
                            key_states,
                            value_states,
                            attn_mask_start_row_indices=attn_mask_startend_row_indices,
                            is_causal=True,
                            enable=skip_recompute,
                        )
                else:
                    attn_output = no_recompute(
                        F.scaled_dot_product_attention,
                        query_states,
                        key_states,
                        value_states,
                        attn_mask=attention_mask,
                        is_causal=query_states.shape[1] != 1,
                        enable=skip_recompute,
                    )
        attn_weights = None

    if reshard_layer is not None:
        # attn_output shape: [bs, seqlen, num_head/sep, head_dim]
        attn_output = reshard_layer(
            attn_output,
            split_axis=1,
            concat_axis=2,
        )
        # attn_output shape: [bs, seqlen/sep, num_head, head_dim]
        assert (
            config.sep_parallel_degree > 1 and q_len % config.sep_parallel_degree == 0
        ), f"q_len:{q_len}, config.sep_parallel_degree:{config.sep_parallel_degree}"
        q_len = q_len // config.sep_parallel_degree
        num_heads = num_heads * config.sep_parallel_degree

    if sequence_parallel:
        attn_output = attn_output.reshape([bsz * q_len, head_dim * num_heads])
    else:
        attn_output = attn_output.reshape([bsz, q_len, head_dim * num_heads])
    return (attn_output, attn_weights) if output_attentions else attn_output
