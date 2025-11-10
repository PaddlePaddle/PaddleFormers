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
import math

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
from paddle.nn.functional.flash_attention import flashmask_attention

from paddleformers.transformers.model_utils import PretrainedModel
from paddleformers.utils.log import logger

from ..distributed import get_hcg
from ..sequence_parallel_utils import AllGatherVarlenOpV2, SliceVarlenOp
from .activation import ACT2FN
from .configuration import DFNRopeVisionTransformerConfig


class _AllToAll(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        input,
        group,
        output_split_sizes=None,
        input_split_sizes=None,
    ):
        """
        All-to-all communication in the group.

        Args:
            ctx (Any): Context object.
            input (Tensor): Input tensor.
            group (Group): The group object.

        Returns:
            Tensor: Output tensor.
        """

        ctx.group = group
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        # return input
        if dist.get_world_size(group) <= 1:
            return input
        if input_split_sizes is None and output_split_sizes is None:
            output = paddle.empty_like(input)
            task = dist.stream.alltoall_single(output, input, None, None, group, True, True)
            task.wait()
        else:
            out_sizes = [sum(output_split_sizes)]
            out_sizes.extend(input.shape[1:])
            output = paddle.empty(out_sizes, dtype=input.dtype)
            task = dist.stream.alltoall_single(
                output,
                input,
                output_split_sizes,
                input_split_sizes,
                group,
                sync_op=False,
            )
            task.wait()
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """
        all-to-all backward

        """
        # return grad_output
        if ctx.input_split_sizes is None and ctx.output_split_sizes is None:
            return _AllToAll.apply(*grad_output, ctx.group)
        else:
            return _AllToAll.apply(*grad_output, ctx.group, ctx.input_split_sizes, ctx.output_split_sizes)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb_vision(tensor: paddle.Tensor, freqs: paddle.Tensor) -> paddle.Tensor:
    """Applies Rotary Position Embedding to the input tensors.

    Args:
        tensor (paddle.Tensor): The input tensor.
        freqs (paddle.Tensor): The frequencies used for the rotation.
    Returns:
        output (paddle.Tensor): the tensor rotated using the Rotary Position Embedding.
    """
    orig_dtype = tensor.dtype

    with paddle.amp.auto_cast(False):
        tensor = tensor.astype(dtype="float32")
        cos = freqs.cos()
        sin = freqs.sin()
        cos = cos.unsqueeze(1).tile(repeat_times=[1, 1, 2]).unsqueeze(0).astype(dtype="float32")
        sin = sin.unsqueeze(1).tile(repeat_times=[1, 1, 2]).unsqueeze(0).astype(dtype="float32")
        output = tensor * cos + rotate_half(tensor) * sin
    output = paddle.cast(output, orig_dtype)
    return output


def qkv_reshard_head(tensor, group):
    """
    After concatenating qkv in the seq dimension, perform the split dimension conversion together
    """
    parallelism = group.nranks
    qkv_seqlen, head_num, head_dim = tensor.shape
    tensor = tensor.transpose(perm=[1, 0, 2]).contiguous()
    out = _AllToAll.apply(tensor, group)
    out = paddle.split(out, parallelism, axis=0)
    output_q = []
    output_k = []
    output_v = []
    for output_i in out:
        outout = output_i.transpose(perm=[1, 0, 2]).contiguous()
        output = paddle.split(outout, 3, axis=0)
        output_q.append(output[0])
        output_k.append(output[1])
        output_v.append(output[2])
    q = paddle.concat(output_q, axis=0)
    k = paddle.concat(output_k, axis=0)
    v = paddle.concat(output_v, axis=0)
    return q, k, v


class VisionFlashAttention2(nn.Layer):
    """VisionFlashAttention2"""

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        """
        Args:
            dim (int): the dimension of each token.
            num_heads (int, optional): number of heads. Default: 16
        """
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=True)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // num_heads  # must added

    def forward(
        self,
        hidden_states: paddle.Tensor,
        startend_row_indices: paddle.Tensor,
        rotary_pos_emb: paddle.Tensor = None,
        attn_sep=False,
    ) -> paddle.Tensor:
        """
        Args:
            hidden_states (paddle.Tensor): hidden states
            cu_seqlens (paddle.Tensor): cumulative sequence lengths, with shape [batch_size + 1]
            rotary_pos_emb (paddle.Tensor, optional): rotary position embedding. Default: None
        Returns:
            paddle.Tensor: output tensor
        """
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape([seq_length, 3, self.num_heads, -1]).transpose(perm=[1, 0, 2, 3])
        q, k, v = qkv.unbind(axis=0)

        if attn_sep:
            hcg = get_hcg()
            mp_group = hcg.get_model_parallel_group()
            qkv = paddle.concat([q, k, v], axis=0)
            q, k, v = qkv_reshard_head(qkv, mp_group)
            seq_length = q.shape[0]

        q = apply_rotary_pos_emb_vision(q.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)

        attn_output = flashmask_attention(
            q.astype("bfloat16").unsqueeze(0),
            k.astype("bfloat16").unsqueeze(0),
            v.astype("bfloat16").unsqueeze(0),
            startend_row_indices=startend_row_indices,
            causal=False,
        )
        attn_output = attn_output.reshape([seq_length, -1])

        if attn_sep:
            out = _AllToAll.apply(attn_output, mp_group)
            out = paddle.split(out, mp_group.nranks, axis=0)
            attn_output = paddle.concat(out, axis=1)
        # attn_output = attn_output.astype(paddle.float32) # TODO: check (liaojincheng)
        attn_output = self.proj(attn_output)
        return attn_output


class PatchEmbed(nn.Layer):
    """PatchEmbed"""

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        """
        Args:
            patch_size (int, optional): patch size. Defaults to 14.
            in_channels (int, optional): number of channels. Defaults to 3.
            embed_dim (int, optional): embedding dimension. Defaults to 1152.
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias_attr=False)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states (paddle.Tensor): hidden states

        Returns:
            paddle.Tensor: output tensor
        """
        target_dtype = self.proj.weight.dtype

        hidden_states = self.proj(paddle.cast(hidden_states, dtype=target_dtype))

        return hidden_states


class VisionMlp(nn.Layer):
    """VisionMLP"""

    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> paddle.Tensor:
        """
        Args:
            x (paddle.Tensor): input tensor

        Returns:
            paddle.Tensor: VisionMLP output tensor
        """
        return self.fc2(self.act(self.fc1(x)))


class VisionRotaryEmbedding(nn.Layer):
    """VisionRotaryEmbedding"""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """
        Args:
            dim (int): the dimension of each token.
            theta (float, optional): the frequency factor. Defaults to 10000.0.
        """
        super().__init__()
        self.inv_freq = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=2, dtype="float32") / dim)

    def forward(self, seqlen: int) -> paddle.Tensor:
        """
        Args:
            seqlen (int): length of sequence.

        Returns:
            paddle.Tensor: rotary position embedding
        """
        seq = paddle.arange(seqlen).cast(self.inv_freq.dtype)
        freqs = paddle.outer(x=seq, y=self.inv_freq)
        return freqs


class DFNRopeVisionBlock(nn.Layer):
    """DFNRopeVisionBlock"""

    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        """
        Args:
            config (dict): model configuration.
            attn_implementation (str, optional): attention implementation. Defaults to "sdpa".
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, epsilon=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, epsilon=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionFlashAttention2(config.hidden_size, num_heads=config.num_heads)
        self.mlp = VisionMlp(
            dim=config.hidden_size,
            hidden_dim=mlp_hidden_dim,
            hidden_act=config.hidden_act,
        )
        self.config = config

    def forward(self, hidden_states, startend_row_indices, rotary_pos_emb, attn_sep=False) -> paddle.Tensor:
        """
        Args:
            hidden_states(paddle.Tensor): hidden states
            cu_seqlens (paddle.Tensor): cumulative sequence lengths
            rotary_pos_emb: rotary position embedding

        Returns:
            paddle.Tensor: output tensor
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            startend_row_indices=startend_row_indices,
            rotary_pos_emb=rotary_pos_emb,
            attn_sep=attn_sep,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class DFNRopeVisionTransformerPretrainedModel(PretrainedModel):
    """DFNRopeVisionTransformerPretrainedModel"""

    config_class = DFNRopeVisionTransformerConfig

    def __init__(self, config) -> None:
        """
        Args:
            config (dict): model configuration
        """
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        self.attn_sep = getattr(config, "attn_sep", False) and config.tensor_parallel_degree > 1

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.LayerList([DFNRopeVisionBlock(config) for _ in range(config.depth)])

        assert (
            config.hidden_size == config.hidden_size
        ), "in DFNRope, vit's config.hidden must be equal to config.hidden_size"
        self.ln = nn.LayerNorm(config.hidden_size, epsilon=1e-6)

    def get_dtype(self) -> paddle.dtype:
        """
        Returns:
            paddle.dtype: data type
        """
        return self.blocks[0].mlp.fc2.weight.dtype

    def rot_pos_emb(self, grid_thw, num_pad=0):
        """rot_pos_emb

        Args:
            grid_thw (paddle.Tensor): grid thw of input

        Returns:
            paddle.Tensor: rotary position embedding
        """
        pos_ids = []
        grid_hw_array = np.array(grid_thw, dtype=np.int64)
        for t, h, w in grid_hw_array:
            hpos_ids = np.arange(h).reshape(-1, 1)
            hpos_ids = np.tile(hpos_ids, (1, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = np.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = np.arange(w).reshape(1, -1)
            wpos_ids = np.tile(wpos_ids, (h, 1))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = np.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked_ids = np.stack([hpos_ids, wpos_ids], axis=-1)
            tiled_ids = np.tile(stacked_ids, (t, 1))
            pos_ids.append(tiled_ids)

        pos_ids = np.concatenate(pos_ids, axis=0)
        if num_pad > 0:
            pos_ids = np.concatenate([pos_ids, np.zeros((num_pad, 2), dtype=pos_ids.dtype)])
        max_grid_size = np.amax(grid_hw_array[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_axis=1)
        return rotary_pos_emb

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor, num_pad=0) -> paddle.Tensor:
        """
        Args:
            hidden_states (paddle.Tensor): input tensor
            grid_thw (paddle.Tensor): grid thw of input
            num_pad (int): number of padding tokens

        Returns:
            paddle.Tensor: output tensor
        """
        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw, num_pad=num_pad)

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype="int32"
        )

        if num_pad > 0:
            cu_seqlens = F.pad(cu_seqlens, (1, 1), value=0)
            cu_seqlens[-1] = cu_seqlens[-2] + num_pad
        else:
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # FlashAttentionVarlen cu_seqlens to FlashMask mask
        cu_seqlens_rm_first = cu_seqlens[1:]
        cu_seqlens_rm_last = cu_seqlens[:-1]
        repeats = cu_seqlens_rm_first - cu_seqlens_rm_last

        startend_row_indices_lts = paddle.repeat_interleave(cu_seqlens_rm_first, repeats).reshape([1, 1, -1, 1])
        startend_row_indices_ute = paddle.repeat_interleave(cu_seqlens_rm_last, repeats).reshape([1, 1, -1, 1])
        startend_row_indices = paddle.concat([startend_row_indices_lts, startend_row_indices_ute], axis=-1)

        attn_sep = getattr(self.config, "attn_sep", False)
        vit_num_recompute_layers = getattr(self.config, "vit_num_recompute_layers", self.config.depth)

        for idx, blk in enumerate(self.blocks):
            if self.config.recompute and self.training and idx < vit_num_recompute_layers:
                hidden_states = recompute(blk, hidden_states, startend_row_indices, rotary_pos_emb, attn_sep)
            else:
                hidden_states = blk(
                    hidden_states,
                    startend_row_indices=startend_row_indices,
                    rotary_pos_emb=rotary_pos_emb,
                    attn_sep=attn_sep,
                )

        ret = self.ln(hidden_states)  # add norm
        return ret

    def extract_feature(self, images, grid_thw):
        """extract feature"""
        if self.config.tensor_parallel_degree <= 1:
            return self._extract_feature(images, grid_thw)
        else:
            grid_thw = grid_thw.clone()
            # logger.info("use sp extract feature")
            images_indices = []
            hcg = fleet.get_hybrid_communicate_group()
            group = hcg.get_model_parallel_group()
            parallelism = group.nranks
            image_size_per_rank = paddle.zeros([parallelism], dtype="int64")
            images_indices = image_size_per_rank

            num_pad = 0
            if self.attn_sep:
                seqlen = images.shape[0]
                num_pad = math.ceil(seqlen / parallelism) * parallelism - seqlen
                images = paddle.nn.functional.pad(images, [0, num_pad, 0, 0], value=0)
                images_indices = [images.shape[0] // parallelism for _ in range(parallelism)]
                images = SliceVarlenOp.apply(images, images_indices)
            else:
                images = SliceVarlenOp.apply(images, images_indices)
                images = images.detach()

            if len(images):
                image_features = self._extract_feature(images, grid_thw, num_pad=num_pad)
            else:
                image_features = paddle.empty(
                    [0, self.config.hidden_size],
                    dtype=self.patch_embed.proj.weight.dtype,
                )
                image_features.stop_gradient = self.patch_embed.proj.weight.stop_gradient

            image_features = AllGatherVarlenOpV2.apply(image_features, images_indices)
            if self.attn_sep:
                image_features = image_features[:seqlen, :]

            return image_features

    def _extract_feature(self, images, grid_thw, num_pad=0):
        """extract feature"""
        ctx = paddle.no_grad if getattr(self.config, "freeze_vision", False) else contextlib.nullcontext
        with ctx():
            image_features = self.forward(images, grid_thw, num_pad)
        return image_features

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        """
        dummy
        """
        return {}

    def set_state_dict(self, state_dict, *args, **kwargs):
        """
        Args:
            state_dict (Mapping[str, Any]): state_dict
        """
        ret = super().set_state_dict(state_dict, *args, **kwargs)
        logger.info(f"dfn rope set_state_dict: {ret}")
