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

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

    from paddleformers.fleet.transformer.transformer_config import TransformerConfig
import paddle
from paddle import Tensor
from paddle.incubate.nn.functional import (
    fused_rotary_position_embedding as fused_rope,
)

if paddle.is_compiled_with_cuda():
    try:
        from paddlefleet_ops import fused_apply_rotary_pos_emb_vision
    except ImportError:
        logging.getLogger(__name__).warning(
            "Failed to import optional CUDA op "
            "'fused_apply_rotary_pos_emb_vision'; falling back to the "
            "non-fused rotary position embedding path."
        )
        fused_apply_rotary_pos_emb_vision = None
else:
    fused_apply_rotary_pos_emb_vision = None

from paddleformers.fleet.utils import get_pg_rank, get_pg_size

logger = logging.getLogger(__name__)

__all__ = [
    "apply_rotary_pos_emb",
    "get_pos_emb_on_this_cp_rank",
]


def get_pos_emb_on_this_cp_rank(
    pos_emb: Tensor, seq_dim: int, cp_group: Group
) -> Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        pos_emb (Tensor): Positional embedding tensor
        seq_dim (int): Sequence dimension
        cp_group (Group): The context parallel group
    """
    if cp_group is None:
        raise ValueError(
            "cp_group must be provided to get positional embedding per CP rank"
        )
    cp_size = get_pg_size(cp_group)
    cp_rank = get_pg_rank(cp_group)
    cp_idx = paddle.to_tensor([cp_rank, (2 * cp_size - cp_rank - 1)])
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim],
        2 * cp_size,
        -1,
        *pos_emb.shape[(seq_dim + 1) :],
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :]
    )
    return pos_emb


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = paddle.chunk(x, 2, axis=-1)
        return paddle.cat((-x2, x1), axis=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_new = paddle.stack((-x2, x1), axis=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def get_unsqueeze_dim(t, freqs):
    # x: [b,seq,head_nums,head_dim] or [b,head_nums,seq,head_dim]
    # freqs: [b,seq,head_dim]
    seq_len = freqs.shape[1]
    return 2 if t.shape[1] == seq_len else 1


def _apply_rotary_pos_emb_bshd(
    t: Tensor | tuple[Tensor, ...],
    freqs: Tensor | None,
    cos: Tensor | None = None,
    sin: Tensor | None = None,
    mscale: float = 1.0,
    position_ids: Tensor | None = None,
    apply_rope_fusion: bool = False,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    high_precision_rope: bool = False,
    rope_theta: float = 10000.0,
    time_major: bool = False,
    sp_group: Group = None,
    inverse: bool = False,
    mla_output_remove_interleaving: bool = False,
) -> Tensor | tuple[Tensor, ...]:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor | tuple[Tensor, ...]): Input tensor T is of shape [seq_length, ... , dim],
            or a tuple of tensors (e.g. (query, key)) for the fused path.
        freqs (Tensor | None): Rotary Positional embedding tensor freq is of shape
            [seq_length, ..., dim]. Can be None when using fused path.
        cos (Tensor | None): Pre-computed cosine values (for fused path).
        sin (Tensor | None): Pre-computed sine values (for fused path).
        mscale (float): Scaling factor for rotary embedding.
        position_ids (Tensor | None): Position indices (for fused path).
        apply_rope_fusion (bool): Whether to use fused RoPE kernel.
        rotary_interleaved (bool): Whether to use interleaved rotary embedding.
        multi_latent_attention (bool): Whether to use multi-latent attention
            interleave reordering.
        high_precision_rope (bool): Whether to use float32 precision for RoPE
            computation.
        rope_theta (float): Base frequency for rotary embedding (used in fused path).
        time_major (bool): Whether the input is time-major (used in fused path).

    Returns:
        Tensor | tuple[Tensor, ...]: The input tensor(s) after applying RoPE.
    """
    # Normalize mscale to avoid TypeError when None is passed
    mscale = mscale if mscale is not None else 1.0

    if apply_rope_fusion:
        if high_precision_rope:
            # Fused vision RoPE CUDA kernel path:
            # - internally computes cos/sin in fp32, equivalent to high_precision_rope
            # - only supports single Tensor input with explicit freqs
            # - only supports non-interleaved mode and mscale=1.0
            # - freqs must be reshaped to [s, dim//2]
            if (
                fused_apply_rotary_pos_emb_vision is not None
                and not rotary_interleaved
                and mscale == 1.0
                and freqs is not None
                and not isinstance(t, tuple)
            ):
                rot_dim = freqs.shape[-1]
                t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
                if freqs.ndim == 3:
                    freqs_2d = freqs.reshape([-1, freqs.shape[-1]])
                else:
                    freqs_2d = freqs
                freqs_half = freqs_2d[..., : freqs_2d.shape[-1] // 2]
                t = fused_apply_rotary_pos_emb_vision(t, freqs_half)
                return paddle.cat((t, t_pass), axis=-1)
            # Fall through to unfused path for unsupported cases
        else:
            # Fused path: delegate to Paddle's fused_rope kernel
            assert isinstance(t, tuple), (
                "The input for fused_rope should be a tuple of tensors"
            )
            return fused_rope(
                *t,
                sin=sin,
                cos=cos,
                rotary_emb_base=rope_theta,
                position_ids=position_ids,
                use_neox_rotary_style=rotary_interleaved,
                time_major=time_major,
            )

    rot_dim = freqs.shape[-1]

    # For Sequence Parallel: slice freqs to match sharded sequence length
    # When SP is enabled, each rank processes only a subset of the full sequence.
    # We need to slice freqs to corresponding positions for this rank.
    # NOTE: Determine unsqueeze_dim BEFORE slicing freqs, as slicing changes the shape.
    # But we need to handle M-RoPE case where freqs may be [S, B, D] vs t's [B, S, H, D].
    # In that case, we should transpose first before calculating unsqueeze_dim.

    # Check if freqs needs transpose (M-RoPE with swapped dims)
    needs_transpose = False
    if freqs.ndim == 3:
        t_d0, t_d1 = t.shape[0], t.shape[1]
        f_d0, f_d1 = freqs.shape[0], freqs.shape[1]
        # freqs is [S, B, D] but t is [B, S, H, D] -> need transpose
        if t_d0 == f_d1 and t_d1 == f_d0 and t_d0 != f_d0:
            needs_transpose = True

    if len(freqs.shape) < len(t.shape):
        if needs_transpose:
            # For M-RoPE [S, B, D] -> [B, S, D], unsqueeze should be at dim 2
            unsqueeze_dim = 2
        else:
            unsqueeze_dim = get_unsqueeze_dim(t, freqs)
    else:
        unsqueeze_dim = None

    if sp_group is not None and sp_group.nranks > 1 and not apply_rope_fusion:
        sp_rank = sp_group.rank
        sp_size = sp_group.nranks
        # Determine sequence dimension based on time_major flag
        if freqs.ndim == 2:
            # freqs: [S, D] -> slice to [S_sp, D]
            seq_len = freqs.shape[0]
            seq_per_rank = seq_len // sp_size
            freqs = freqs[sp_rank * seq_per_rank : (sp_rank + 1) * seq_per_rank]
        elif freqs.ndim == 3:
            # freqs: [B, S, D] or [S, B, D] based on time_major
            if time_major:
                # freqs: [S, B, D] -> slice to [S_sp, B, D]
                seq_len = freqs.shape[0]
                seq_per_rank = seq_len // sp_size
                freqs = freqs[
                    sp_rank * seq_per_rank : (sp_rank + 1) * seq_per_rank, :, :
                ]
            else:
                # freqs: [B, S, D] -> slice to [B, S_sp, D]
                seq_len = freqs.shape[1]
                seq_per_rank = seq_len // sp_size
                freqs = freqs[
                    :, sp_rank * seq_per_rank : (sp_rank + 1) * seq_per_rank, :
                ]

    # For M-RoPE with sequence parallel, freqs may be [S, B, D] while t is [B, S, H, D].
    # When the first two dims are swapped (same product but different order), transpose
    # freqs to align with t's [batch, seq] layout.  A plain reshape would silently
    # reinterpret the memory without reordering data, giving wrong results for B > 1.
    if freqs.ndim == 3:
        t_d0, t_d1 = t.shape[0], t.shape[1]
        f_d0, f_d1 = freqs.shape[0], freqs.shape[1]
        if (t_d0 != f_d0 or t_d1 != f_d1) and t_d0 * t_d1 == f_d0 * f_d1:
            freqs = freqs.transpose([1, 0, 2]).contiguous()
            # After transpose, need to recalculate unsqueeze_dim
            if len(freqs.shape) < len(t.shape):
                unsqueeze_dim = get_unsqueeze_dim(t, freqs)

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = paddle.cat((x1, x2), axis=-1)

    # For high_precision_rope, cast to float32 and disable auto_cast to ensure
    # numerical stability in the rotary computation.
    orig_t_dtype = t.dtype
    ctx = paddle.amp.auto_cast(False) if high_precision_rope else nullcontext()
    with ctx:
        if high_precision_rope:
            t = t.astype(dtype="float32")
            t_pass = t_pass.astype(dtype="float32")

        # first part is cosine component
        # second part is sine component, need to change signs with _rotate_half method
        cos_ = (paddle.cos(freqs) * mscale).to(t.dtype)
        sin_ = (paddle.sin(freqs) * mscale).to(t.dtype)
        if inverse:
            sin_ = -sin_
        if len(cos_.shape) < len(t.shape):
            # [b,s,h]->[b,s,1,h]
            # Use pre-computed unsqueeze_dim if available (for SP case),
            # otherwise compute it now
            if unsqueeze_dim is None:
                unsqueeze_dim = get_unsqueeze_dim(t, cos_)
            cos_.unsqueeze_(unsqueeze_dim)
            sin_.unsqueeze_(unsqueeze_dim)

        rotate_t = _rotate_half(t, rotary_interleaved)
        if len(rotate_t.shape) < len(t.shape):
            rotate_t.reshape_(t.shape)

        t = (t * cos_) + (rotate_t * sin_)

        if multi_latent_attention and mla_output_remove_interleaving:
            half = t.shape[-1] // 2
            x1 = t[..., :half]
            x2 = t[..., half:]
            t = paddle.stack([x1, x2], axis=-1).flatten(start_axis=-2)

        result = paddle.cat((t, t_pass), axis=-1)

    if high_precision_rope:
        result = result.astype(orig_t_dtype)
    return result


def _get_thd_freqs_on_this_cp_rank(
    cp_rank: int, cp_size: int, x: Tensor, freqs: Tensor, offset: int = 0
) -> Tensor:
    """Get the correct frequency slice for this context parallel rank with optional sequence offset.

    Args:
        cp_rank: Current context parallel rank
        cp_size: Total context parallel size
        x: Input tensor for current sequence
        freqs: Frequency tensor - either full batch positions or max sequence length
        offset: Starting position offset for this sequence in the original batch (default: 0)

    Returns:
        Tensor: Frequency slice corresponding to this CP rank's portion of the sequence

    Note:
        This function supports two modes based on the offset parameter:
        1. offset > 0: Exact mapping mode - freqs contains all positions across all sequences.
           The offset ensures each sequence gets frequencies from its actual position within
           the overall batch. Critical for non-1D RoPE in VLMs where spatial positions matter.
        2. offset = 0: Traditional mode - freqs contains only max sequence length positions.
           All sequences use frequencies starting from position 0, preserving backward
           compatibility.
    """
    if cp_size > 1:
        cp_seg = x.size(1) // 2
        full_seqlen = cp_size * x.size(1)
        # Apply offset to both forward and backward segments for context parallelism
        # offset=0: traditional behavior, freqs[0:cp_seg] and freqs[...]
        # offset>0: exact mapping, freqs[offset+0:offset+cp_seg] and freqs[offset+...]
        return paddle.cat(
            [
                freqs[
                    :,
                    offset + cp_rank * cp_seg : offset + (cp_rank + 1) * cp_seg,
                ],
                freqs[
                    :,
                    offset + full_seqlen - (cp_rank + 1) * cp_seg : offset
                    + full_seqlen
                    - cp_rank * cp_seg,
                ],
            ]
        )
    else:
        # For single context parallel rank:
        # offset=0: use freqs[0:x.size(0)] (traditional)
        # offset>0: use freqs[offset:offset+x.size(0)] (exact mapping)
        return freqs[:, offset : offset + x.size(1)]


def _apply_rotary_pos_emb_thd(
    t: Tensor | tuple[Tensor, ...],
    cu_seqlens: Tensor,
    total_seq_len: int | None,
    freqs: Tensor | None,
    cos: Tensor | None = None,
    sin: Tensor | None = None,
    mscale: float = 1.0,
    cp_group: Group = None,
    sp_group: Group = None,
    position_ids: Tensor | None = None,
    apply_rope_fusion: bool = False,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    high_precision_rope: bool = False,
    rope_theta: float = 10000.0,
    time_major: bool = False,
    inverse: bool = False,
    mla_output_remove_interleaving: bool = False,
) -> Tensor | tuple[Tensor, ...]:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor | tuple[Tensor, ...]): Input tensor T is of shape [t, h, d],
            or a tuple of tensors (e.g. (query, key)) for the fused path.
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
            with shape [b + 1] and dtype paddle.int32.
        total_seq_len (int | None): The actual total sequence length before padding.
            When cu_seqlens uses a padded version, this provides the true total length
            for correct frequency tensor selection. If None, falls back to cu_seqlens[-1].
        freqs (Tensor | None): Rotary Positional embedding tensor freq is of shape
            [max_s, 1, 1, d]. Can be None when using fused path.
        cos (Tensor | None): Pre-computed cosine values (for fused path).
        sin (Tensor | None): Pre-computed sine values (for fused path).
        mscale (float): Scaling factor for rotary embedding.
        cp_group (Group): The context parallel group.
        position_ids (Tensor | None): Position indices (for fused path).
        apply_rope_fusion (bool): Whether to use fused RoPE kernel.
        rotary_interleaved (bool): Whether to use interleaved rotary embedding.
        multi_latent_attention (bool): Whether to use multi-latent attention
            interleave reordering.
        high_precision_rope (bool): Whether to use float32 precision for RoPE
            computation.
        rope_theta (float): Base frequency for rotary embedding (used in fused path).
        time_major (bool): Whether the input is time-major (used in fused path).

    Returns:
        Tensor | tuple[Tensor, ...]: Shape [t, h, d]. The input tensor(s) after
            applying RoPE.
    """
    cp_size = get_pg_size(cp_group)
    cp_rank = get_pg_rank(cp_group)

    total_seq_len = (
        total_seq_len if total_seq_len is not None else cu_seqlens[-1]
    )

    # Handle two different frequency tensor formats:
    # 1. If freqs.size(1) == total_seq_len: freqs contains all positions across all sequences
    #    -> Use offset-based mapping for exact positional correspondence
    # 2. Otherwise: freqs contains only max sequence length positions
    #    -> Use traditional mapping without offsets (map first :seqlen part)
    if freqs.dim() >= 1 and freqs.size(1) == total_seq_len:
        # CASE 1: Exact mapping with offsets
        # When cp_size==1, every per-segment slice concatenates back to the original freqs.
        # Skip the split+cat and call bshd directly with the original freqs.
        if cp_size == 1:
            return _apply_rotary_pos_emb_bshd(
                t,
                freqs,
                cos=cos,
                sin=sin,
                mscale=mscale,
                position_ids=position_ids,
                apply_rope_fusion=apply_rope_fusion,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                high_precision_rope=high_precision_rope,
                rope_theta=rope_theta,
                time_major=time_major,
            )
        seqlens = ((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()
        # Build packed freqs in one pass, then apply once to the whole packed tensor
        cu_seqlens_list = cu_seqlens.tolist()
        sequence_splits = paddle.split(t, seqlens, axis=1 if t.ndim == 4 else 0)
        freq_slices = []
        for i, x in enumerate(sequence_splits):
            # cu_seqlens[i] is the starting offset of this sequence in the original batch
            seq_start_offset = cu_seqlens_list[i]
            freq_slices.append(
                _get_thd_freqs_on_this_cp_rank(
                    cp_rank, cp_size, x, freqs, seq_start_offset
                )
            )

        freqs_packed = paddle.cat(freq_slices, axis=1)
        # [b,seq,num_heads,head_dim]
        return _apply_rotary_pos_emb_bshd(
            t,
            freqs_packed,
            cos=cos,
            sin=sin,
            mscale=mscale,
            position_ids=position_ids,
            apply_rope_fusion=apply_rope_fusion,
            rotary_interleaved=rotary_interleaved,
            multi_latent_attention=multi_latent_attention,
            high_precision_rope=high_precision_rope,
            rope_theta=rope_theta,
            time_major=time_major,
        )
    else:
        # CASE 2: Traditional mapping without offsets
        # Build packed freqs for all sequences using the standard mapping, then apply once
        seqlens = ((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()
        sequence_splits = paddle.split(t, seqlens, axis=1 if t.ndim == 4 else 0)
        freqs_packed = paddle.cat(
            [
                _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs)
                for x in sequence_splits
            ],
            axis=1,
        )

        return _apply_rotary_pos_emb_bshd(
            t,
            freqs_packed,
            cos=cos,
            sin=sin,
            mscale=mscale,
            position_ids=position_ids,
            apply_rope_fusion=apply_rope_fusion,
            rotary_interleaved=rotary_interleaved,
            multi_latent_attention=multi_latent_attention,
            high_precision_rope=high_precision_rope,
            rope_theta=rope_theta,
            time_major=time_major,
        )


def apply_rotary_pos_emb(
    t: Tensor | tuple[Tensor, ...],
    freqs: Tensor | None,
    cos: Tensor | None,
    sin: Tensor | None,
    config: TransformerConfig,
    cu_seqlens: Tensor | None = None,
    total_seq_len: int | None = None,
    mscale: float = 1.0,
    cp_group: Group = None,
    sp_group: Group = None,
    position_ids: Tensor | None = None,
    inverse: bool = False,
    mla_output_remove_interleaving: bool = False,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    bshd (conventional) / thd (packed seq) format.

    The fused/unfused decision is handled internally by each format-specific
    function based on config.apply_rope_fusion.

    Args:
        t (Tensor | tuple[Tensor, ...]): Input tensor, or a tuple of tensors
            (e.g. (query, key)) for the fused path.
        freqs (Tensor | None): Rotary positional embedding frequencies.
            Can be None when using fused path.
        cos (Tensor | None): Pre-computed cosine values of freqs (used for
            fused implementation).
        sin (Tensor | None): Pre-computed sine values of freqs (used for
            fused implementation).
        config (TransformerConfig): Transformer configuration.
        cu_seqlens (Tensor | None): Cumulative sequence lengths.
        total_seq_len (int | None): The actual total sequence length before padding.
            Used in thd format to correctly select frequency tensor when cu_seqlens
            is padded. If None, falls back to cu_seqlens[-1].
        mscale (float): Scaling factor.
        cp_group (Group): Context parallel group.
        position_ids (Tensor | None): Position indices.
        inverse (bool): If True, negate sin to apply inverse RoPE rotation.
        mla_output_remove_interleaving (bool): If True, un-interleave the output
            after MLA-style RoPE (used by DSv4 Hybrid for inverse RoPE on output).
    """
    rope_kwargs = {
        "apply_rope_fusion": config.apply_rope_fusion if not inverse else False,
        "rotary_interleaved": config.rotary_interleaved,
        "multi_latent_attention": config.multi_latent_attention,
        "high_precision_rope": config.high_precision_rope,
        "rope_theta": config.rope_theta,
        "time_major": config.sequence_parallel,
        "sp_group": sp_group,
        "inverse": inverse,
        "mla_output_remove_interleaving": mla_output_remove_interleaving,
    }
    if cu_seqlens is None:
        return _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            cos=cos,
            sin=sin,
            mscale=mscale,
            position_ids=position_ids,
            **rope_kwargs,
        )
    else:
        return _apply_rotary_pos_emb_thd(
            t,
            cu_seqlens,
            total_seq_len,
            freqs,
            cos=cos,
            sin=sin,
            mscale=mscale,
            cp_group=cp_group,
            position_ids=position_ids,
            **rope_kwargs,
        )
