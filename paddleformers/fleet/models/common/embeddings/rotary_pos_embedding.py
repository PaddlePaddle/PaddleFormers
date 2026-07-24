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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddleformers.fleet.packed_seq_params import PackedSeqParams
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig

import math

import paddle
from paddle import Tensor, nn

from paddleformers.fleet import parallel_state

logger = logging.getLogger(__name__)


__all__ = [
    "RotaryEmbedding",
    "MultimodalRotaryEmbedding",
    "Rope2DPosEmbRepeated",
]


class RotaryEmbedding(nn.Layer):
    """Rotary Embedding for language model.

    Args:
        head_dim (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position
            embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE
            for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
        rope_scaling (bool, optional): Apply rope scaling as used in llama 3.x.
        rope_scaling_factor (float, optional): rope scaling factor in llama 3.x. Defaults to 8.
        cp_group (paddle.distributed.communication.group.Group, optional): Process group for context parallel.
            Defaults to None.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float | None = None,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        cp_group: paddle.distributed.communication.group.Group | None = None,
        use_accuracy_compatible: bool = False,
    ) -> None:
        super().__init__()

        dim = head_dim
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor

        if use_accuracy_compatible:
            _exp_cpu = (
                paddle.arange(0, dim, 2, dtype=paddle.int64)
                .cpu()
                .astype(paddle.float32)
                / dim
            )
            _inv_freq_cpu = 1.0 / (rotary_base**_exp_cpu)
            self.inv_freq = (
                _inv_freq_cpu.cuda()
                if paddle.is_compiled_with_cuda()
                else _inv_freq_cpu
            )
        else:
            self.inv_freq = 1.0 / (
                rotary_base
                ** (
                    paddle.arange(0, dim, 2, dtype=paddle.int64).astype(
                        dtype=paddle.float32
                    )
                    / dim
                )
            )

        if rope_scaling:
            self.inv_freq = self._apply_scaling(
                self.inv_freq, factor=rope_scaling_factor
            )

        self.cp_group = (
            cp_group
            if cp_group is not None
            else parallel_state.get_context_parallel_group(
                check_initialized=False
            )
        )

        self._cast_to_low_precision = False

    def _apply_scaling(
        self,
        freqs,
        factor=8,
        low_freq_factor=1,
        high_freq_factor=4,
        original_max_position_embeddings=8192,
    ):
        # This implementation is adapted from:
        # https://github.com/huggingface/transformers/blob/2a5a6ad18aa22e98429bb5ecb880660328030ea0/src/transformers/modeling_rope_utils.py#L303-L343

        factor = factor  # `8` in the original implementation
        low_freq_factor = low_freq_factor  # `1` in the original implementation
        high_freq_factor = (
            high_freq_factor  # `4` in the original implementation
        )
        old_context_len = original_max_position_embeddings  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / freqs
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = paddle.where(
            wavelen > low_freq_wavelen, freqs / factor, freqs
        )
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(
            wavelen > low_freq_wavelen
        )
        inv_freq_llama = paddle.where(
            is_medium_freq, smoothed_inv_freq, inv_freq_llama
        )

        return inv_freq_llama

    def get_freqs_non_repeated(
        self, max_seq_len: int, offset: int = 0, position_ids: Tensor = None
    ) -> Tensor:
        """Generates matrix of frequencies based on positions in the sequence,
        used to create positional encodings"""
        if position_ids is not None:
            # Handle different position_ids shapes:
            # - 1D [S]: use directly (also covers fastdeploy decode mode)
            # - 2D [B, S]: use first batch (assume all batches have same positions)
            # - 3D: not supported by RotaryEmbedding, use MultimodalRotaryEmbedding instead
            if position_ids.ndim == 1:
                seq = position_ids.astype(self.inv_freq.dtype)
            elif position_ids.ndim == 2:
                # Take first batch, assuming all batches have same position_ids
                seq = position_ids[0].astype(self.inv_freq.dtype)
            else:
                # For 3D position_ids (M-RoPE), this function should not be called
                # Fall back to max_seq_len to avoid cryptic errors
                seq = (
                    paddle.arange(max_seq_len).astype(self.inv_freq.dtype)
                    + offset
                )
        else:
            seq = (
                paddle.arange(max_seq_len).astype(self.inv_freq.dtype) + offset
            )

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = paddle.outer(seq, self.inv_freq)  # [seq len, dim]

        return freqs

    def get_cos_sin(
        self, max_seq_len: int, offset: int = 0
    ) -> (Tensor, Tensor):
        """Cosine and sine values for RoPE are precomputed for all positions up to the maximum
        sequence length"""
        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        cos = paddle.cos(freqs)
        sin = paddle.sin(freqs)
        return cos, sin

    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        position_ids: Tensor = None,
    ) -> Tensor:
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.
            packed_seq (bool, optional): Whether to use packed sequence. Defaults to False.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        freqs = self.get_freqs_non_repeated(
            max_seq_len, offset, position_ids=position_ids
        )
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = paddle.cat((freqs, freqs), axis=-1)
        else:
            emb = paddle.stack(
                (freqs.reshape((-1, 1)), freqs.reshape((-1, 1))), axis=-1
            ).reshape((freqs.shape[0], -1))
        # emb [1, seq_len, 1, dim]
        emb = emb[None, :, None, :]
        return emb

    def get_rotary_seq_len(
        self,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> int:
        """Function to get the rotary sequence length.

        Args:
            transformer_input (Tensor): Input tensor to the transformer
            transformer_config (TransformerConfig): Transformer config used by the model
            packed_seq_params (PackedSeqParams): Packed sequence params

        Returns:
            int: The rotary sequence length
        """

        if packed_seq_params is not None:
            # max_seqlen are the max sequence length in the packed sequence before being divived
            # by the tp and cp size.
            return max(
                packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv
            )
        else:
            if transformer_config.sequence_parallel:
                seq_axis = 0
            else:
                seq_axis = 1
            rotary_seq_len = transformer_input.shape[seq_axis]

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

        # TODO: self.cp_group.world_size --> transformer_config.context_parallel_size
        # rotary_seq_len *= transformer_config.context_parallel_size
        if self.cp_group is not None and self.cp_group.world_size > 1:
            rotary_seq_len *= self.cp_group.world_size

        return rotary_seq_len


class MultimodalRotaryEmbedding(nn.Layer):
    """Multimodal Rotary Embedding for language model.
    Based on https://github.com/alibaba/Pai-Megatron-Patch/blob/
    efa5a752e845267936db9ae7df1b6aba92e9ff9a/megatron_patch/model/qwen2_vl/rotary_pos_embedding.py
    Copyright (c) 2025 alibaba/Pai-Megatron-Patch. Apache 2.0 license.

    Args:
        head_dim (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position
            embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE
            for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float | None = None,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        cp_group: paddle.distributed.communication.group.Group | None = None,
        use_accuracy_compatible: bool = False,
    ) -> None:
        super().__init__()

        if rotary_percent < 1.0:
            head_dim = int(head_dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base
            ** (paddle.arange(0, head_dim, 2, dtype=paddle.float32) / head_dim)
        )

        self.cp_group = (
            cp_group
            if cp_group is not None
            else parallel_state.get_context_parallel_group(
                check_initialized=False
            )
        )

    def forward(
        self, position_ids: paddle.Tensor, mrope_section: list[int]
    ) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (Paddle.Tensor): A position_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal,
                height and width in rope calculation.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        assert mrope_section is not None, "Please provide mrope_section"

        seqlens = position_ids.shape[2]
        position_ids = position_ids.reshape([3, -1, seqlens])

        with paddle.amp.auto_cast(False):
            inv_freq_expanded = (
                self.inv_freq.unsqueeze(0)
                .unsqueeze(-1)
                .cast(paddle.float32)
                .expand([3, position_ids.shape[1], -1, 1])
            )
            position_ids_expanded = position_ids.unsqueeze(2).cast(
                paddle.float32
            )

            freqs = paddle.matmul(
                inv_freq_expanded, position_ids_expanded
            ).transpose([0, 1, 3, 2])

            freqs = self.apply_interleaved_mrope(freqs, mrope_section)
            emb = paddle.cat((freqs, freqs), axis=-1)

        return emb

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class Rope2DPosEmbRepeated(nn.Layer):
    """2D rotary position embedding with multi-resolution support.
    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.
    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py
    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        rotary_base (float): the base of the theta
    """

    def __init__(
        self,
        head_dim: int,
        max_height: int = 512,
        max_width: int = 512,
        rotary_base=10000,
        cp_group: paddle.distributed.communication.group.Group | None = None,
    ):
        super().__init__()
        self.dim = head_dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.rotary_base = rotary_base
        self.rotary_pos_cos = None
        self.rotary_pos_sin = None

    def _precompute_freqs_cis(self) -> paddle.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * rotary_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * rotary_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = paddle.arange(0, N).float()
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = paddle.arange(0, self.dim, 4)[
            : (self.dim // 4)
        ].float()  # C/4
        freqs = 1.0 / (self.rotary_base ** (dim_range / self.dim))
        x_freqs = paddle.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = paddle.outer(y_pos, freqs).float()  # N, C/4
        x_cis = paddle.polar(paddle.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = paddle.polar(paddle.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = paddle.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_thws: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            grid_thws (paddle.Tensor): grid time, height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis", self._precompute_freqs_cis(), persistent=False
            )

        shapes = grid_thws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width
            for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = paddle.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )
        return freqs_cis

    def forward(self, grid_thws: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            grid_thws (paddle.Tensor): grid time, height and width
         Returns:
            Tensor: Embeddings after applying RoPE. shape (sum(t * height * width), dim//2)
        """
        freqs_cis = self.get_freqs_cis(grid_thws)
        rotary_pos_emb = paddle.angle(freqs_cis)
        self.rotary_pos_cos = paddle.real(freqs_cis)
        self.rotary_pos_sin = paddle.imag(freqs_cis)
        return rotary_pos_emb

    def get_cos_sin(self, grid_thws: int, offset: int = 0) -> (Tensor, Tensor):
        """Cosine and sine values for RoPE are precomputed for all positions up to the maximum
        sequence length"""
        if self.rotary_pos_cos is None or self.rotary_pos_sin is None:
            self.forward(grid_thws)

        return self.rotary_pos_cos, self.rotary_pos_sin
