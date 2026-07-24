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

"""
Compressed Sparse Attention (CSA) for DeepSeekV4 Hybrid Attention.

Ported from Megatron-LM experimental_attention_variant/csa.py (commit bf4e1db).

Components:
  - Compressor: Gated pooling compressor with overlap (ratio=4) or non-overlap (ratio=128)
  - CSAIndexer: Learned top-k retrieval over compressed positions
  - CompressedSparseAttention: Core attention combining sliding window + compressed KV
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddleformers.fleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddleformers.fleet.transformer import FleetLayer

_ACCURACY_COMPATIBLE_KERNEL: bool = (
    os.environ.get("FLAGS_use_accuracy_compatible_kernel", "0") == "1"
)
from paddleformers.fleet.context_parallel_utils import ContextParallelGatherOp
from paddleformers.fleet.parallel_state import get_context_parallel_world_size
from paddleformers.fleet.transformer.dsa_attention import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    FusedDSAIndexerLoss,
    fused_qk_topk_naive,
    rotate_activation,
)

if TYPE_CHECKING:
    from paddleformers.fleet.process_groups_config import ProcessGroupCollection
    from paddleformers.fleet.transformer.enums import AttnMaskType
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig

# CP utilities are imported lazily inside _forward_cp to avoid circular imports
# at module load time. The public symbols are re-exported here for convenience.
from paddleformers.fleet.fp8.qat import fp8_simulate_qat
from paddleformers.fleet.transformer.cp_utils import (
    all_gather_cp,
    build_causal_mask_cp,
    get_compress_topk_idxs_cp,
    get_window_topk_idxs_cp,
    map_compressed_topk_to_kv_full_cp,
)


def _normalize_csa_docmask_args(
    ratio: int,
    batch_size: int,
    seqlen: int,
    n_compressed: int | None = None,
    *,
    require_batch_one: bool = True,
) -> tuple[int, int, int, int]:
    ratio = int(ratio)
    batch_size = int(batch_size)
    seqlen = int(seqlen)
    if n_compressed is not None:
        n_compressed = int(n_compressed)
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got ratio: {ratio}")
    if seqlen <= 0:
        raise ValueError(f"seqlen must be positive, got seqlen: {seqlen}")
    if require_batch_one and batch_size != 1:
        raise ValueError(
            f"only support batch_size = 1, got batch_size: {batch_size}"
        )
    if n_compressed is None:
        n_compressed = seqlen // ratio
    if n_compressed < 0:
        raise ValueError(
            f"n_compressed must be non-negative, got n_compressed: {n_compressed}"
        )
    return ratio, batch_size, seqlen, n_compressed


def _validate_csa_docmask_shape(
    startend_row_indices: Tensor,
    batch_size: int,
    seqlen: int,
) -> None:
    shape = list(startend_row_indices.shape)
    expected = [batch_size, 1, seqlen, 1]
    if shape != expected:
        raise ValueError(
            "startend_row_indices must have shape "
            f"{expected}, got shape: {shape}"
        )


def _derive_csa_doc_boundaries(
    startend_row_indices: Tensor,
    seqlen: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    mask = startend_row_indices.flatten().cast("int64")
    positions = paddle.arange(seqlen, dtype="int64")

    is_boundary = paddle.zeros([seqlen], dtype="bool")
    is_boundary[0] = True
    is_boundary[1:] = (positions[1:] == mask[:-1]) & (mask[1:] != mask[:-1])

    doc_start_per_pos = paddle.cummax(
        is_boundary.cast("int64") * positions, axis=0
    ).values
    pos_in_doc = positions - doc_start_per_pos
    doc_len_per_pos = mask - doc_start_per_pos
    is_valid = pos_in_doc < doc_len_per_pos

    doc_starts_i64 = paddle.nonzero(is_boundary).flatten()
    doc_lens = (mask[doc_starts_i64] - doc_starts_i64).cast("int32")
    doc_starts = doc_starts_i64

    return doc_start_per_pos, doc_len_per_pos, is_valid, doc_lens, doc_starts


def _build_window_topk_idxs_from_doc_bounds(
    batch_size: int,
    seqlen: int,
    window_size: int,
    doc_start_per_pos: Tensor,
    is_valid: Tensor,
) -> Tensor:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    positions = paddle.arange(seqlen, dtype="int64")
    win_start = paddle.maximum(doc_start_per_pos, positions - window_size + 1)
    offsets = paddle.arange(window_size, dtype="int64").unsqueeze(0)
    indices = win_start.unsqueeze(1) + offsets
    invalid = (
        (indices > positions.unsqueeze(1))
        | (indices < doc_start_per_pos.unsqueeze(1))
        | (~is_valid).unsqueeze(1).expand_as(indices)
    )
    result = paddle.where(invalid, paddle.full_like(indices, -1), indices)
    return result.unsqueeze(0).expand([batch_size, -1, -1])


def _build_compress_topk_idxs_from_valid_range(
    batch_size: int,
    seqlen: int,
    n_compressed: int,
    offset: int,
    valid_range: Tensor,
) -> Tensor:
    c_grid = paddle.arange(n_compressed, dtype="int64").unsqueeze(0)
    valid_range = valid_range[0].cast("int64")
    range_start = valid_range[:, 0]
    range_end = valid_range[:, 1]
    active = (c_grid >= range_start.unsqueeze(1)) & (
        c_grid < range_end.unsqueeze(1)
    )
    result = paddle.where(
        active,
        (c_grid + offset).cast("int32"),
        paddle.full([seqlen, n_compressed], -1, dtype="int32"),
    )
    return result.unsqueeze(0).expand([batch_size, -1, -1])


def _build_compressed_causal_mask_from_valid_range(
    batch_size: int,
    seqlen: int,
    n_compressed: int,
    valid_range: Tensor,
) -> Tensor:
    c_grid = paddle.arange(n_compressed, dtype="int64").unsqueeze(0)
    valid_range = valid_range[0].cast("int64")
    range_start = valid_range[:, 0]
    range_end = valid_range[:, 1]
    valid_mask = (c_grid >= range_start.unsqueeze(1)) & (
        c_grid < range_end.unsqueeze(1)
    )
    invalid = (
        (~valid_mask).unsqueeze(0).expand([batch_size, seqlen, n_compressed])
    )
    return paddle.where(
        invalid,
        paddle.full([1], float("-inf"), dtype="float32"),
        paddle.zeros([1], dtype="float32"),
    )


def _build_valid_range_from_doc_bounds(
    ratio: int,
    seqlen: int,
    doc_start_per_pos: Tensor,
    doc_len_per_pos: Tensor,
    is_valid: Tensor,
) -> Tensor:
    positions = paddle.arange(seqlen, dtype="int64")
    pos_in_doc = positions - doc_start_per_pos
    num_compressed_per_pos = doc_len_per_pos // ratio
    boundary_marker = (positions == doc_start_per_pos).cast("int64")
    boundary_compressed = boundary_marker * num_compressed_per_pos
    cum_compressed = paddle.cumsum(boundary_compressed, axis=0)
    doc_col_start = cum_compressed - num_compressed_per_pos

    causal_avail = (pos_in_doc + 1) // ratio
    num_available = paddle.minimum(causal_avail, num_compressed_per_pos)
    range_start = doc_col_start
    range_end = doc_col_start + num_available
    zero_mask = (num_available == 0) | (~is_valid)
    range_start = paddle.where(
        zero_mask, paddle.zeros_like(range_start), range_start
    )
    range_end = paddle.where(zero_mask, paddle.zeros_like(range_end), range_end)
    return paddle.stack([range_start, range_end], axis=-1).cast("int32")


class LinearBF16FP32Func(paddle.autograd.PyLayer):
    """BF16 activation x BF16 weight -> FP32 output autograd function.

    Forward matches SGLang's default DeepSeek-V4 compressor path
    (`sglang.jit_kernel.deepseek_v4.linear_bf16_fp32`, cublas backend).
    BF16 activation x BF16 weight -> FP32 output. This keeps Megatron's
    compressor log-prob computation aligned with SGLang rollout. Backward is
    only needed for training, so keep its gradient matmuls in FP32.
    """

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor) -> Tensor:
        """Forward pass: BF16 matmul with FP32 output."""
        x_bf16 = x.cast(paddle.bfloat16)
        weight_bf16 = weight.cast(paddle.bfloat16)
        ctx.save_for_backward(x_bf16, weight_bf16)
        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        # Paddle PyLayer requires None for stop_gradient inputs; record here.
        ctx.x_needs_grad = not x.stop_gradient
        ctx.weight_needs_grad = not weight.stop_gradient

        x_2d = x_bf16.reshape([-1, x_bf16.shape[-1]])
        out = paddle.mm(x_2d, weight_bf16, out_dtype=paddle.float32)
        return out.view(*x.shape[:-1], weight_bf16.shape[1])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Backward pass: compute gradients for x and weight."""
        x_bf16, weight_bf16 = ctx.saved_tensor()
        grad_output_2d = grad_output.reshape([-1, grad_output.shape[-1]]).cast(
            paddle.float32
        )

        grad_x = None
        if ctx.x_needs_grad:
            grad_x = grad_output_2d.matmul(weight_bf16.cast(paddle.float32).t())
            grad_x = grad_x.view(ctx.input_shape).cast(ctx.input_dtype)

        grad_weight = None
        if ctx.weight_needs_grad:
            x_2d = x_bf16.reshape([-1, x_bf16.shape[-1]])
            grad_weight = (
                x_2d.cast(paddle.float32)
                .t()
                .matmul(grad_output_2d)
                .cast(ctx.weight_dtype)
            )

        return grad_x, grad_weight


def linear_bf16_fp32(x: Tensor, weight: Tensor) -> Tensor:
    """BF16 matmul with FP32 output wrapper function."""
    return LinearBF16FP32Func.apply(x, weight)


# ---------------------------------------------------------------------------
# Helper functions for index computation
# ---------------------------------------------------------------------------


def get_cutoff_doc_lens(doc_lens: Tensor, ratio: int) -> Tensor:
    """Round each document length down to the nearest multiple of ratio.

    Args:
        doc_lens: [n_docs] tensor of document lengths.
        ratio: compression ratio.

    Returns:
        cutoff_doc_lens: [n_docs] int32 tensor.
    """
    return ((doc_lens // ratio) * ratio).cast("int32")


def get_cutoff_doc_starts(cutoff_doc_lens: Tensor) -> Tensor:
    """Compute cumulative start positions from cutoff document lengths.

    Args:
        cutoff_doc_lens: [n_docs] tensor of cutoff document lengths.

    Returns:
        cutoff_doc_starts: [n_docs] int32 tensor.
    """
    lens = cutoff_doc_lens.flatten().cast("int32")
    cum = paddle.cumsum(lens, axis=0)
    starts = paddle.zeros_like(cum)
    if cum.shape[0] > 1:
        starts[1:] = cum[:-1]
    return starts


@dataclass
class CSADocMaskMetadata:
    """
    Reusable CSA metadata derived from `startend_row_indices`.

    假设输入 seqlen=32，其中每个文本的 doc_lens=[5, 14, 3, 8]，最后还有 padding=2

    则非压缩相关的信息如下，注意所有列表的长度都是32：
    start_end_row_indices: [5 ... 5, 19  ... 19, 22 ... 22, 30 ... 30, 30, 30 ]
    doc_start_per_pos:     [0 ... 0,  5  ...  5, 19 ... 19, 22 ... 22, 22, 22 ]
    doc_len_per_pos:       [5 ... 5, 14  ... 14,  3 ...  3,  8 ...  8,  8,  8 ]
    pos_in_doc:            [0 ... 4,  0  ... 13,  0 ...  2,  0 ...  7,  8,  9 ]
    is_valid:              [T ... T,  T  ...  T,  T ...  T,  T ...  T,  F,  F ]
                            {- 5 -}  {-- 14 --}  {-- 3 --}  {-- 8 --}  {- 2 -}
    说明：
    * padding 在 start_end_row_indices 中用前一个 doc 的末尾数字或对角线的值表示
    * 其他变量中 padding 部分相当于自然接续前一个 doc，但实际上对梯度无贡献

    假设压缩 ratio=4，则压缩前的相关信息如下，注意最后填充-1使得列表长度依然为32：
    n_compressed:          seqlen // ratio = 8
    actual_n_compressed:   sum(n // ratio for n in doc_lens) = 6
    cutoff_gather_indices: [0 ... 3,  5  ... 16,   ...  , 22 ... 29, -1 ... -1]
    cutoff_pos_in_doc:     [0 ... 3,  0  ... 11,   ...  ,  0 ...  7, -1 ... -1]
                            {- 4 -}  {-- 12 --}  {- 0 -}  {-- 8 --}  {-- 8 --}

    针对压缩变量的信息，长度为 n_compressed=8，最后无效部分填充 False 或-1：
    compressed_is_first:   [ T,  T,  F,  F,  T,  F,  F,  F]
    compressed_pos_in_doc: [ 0,  0,  4,  8,  0,  4, -1, -1]

    其他变量：
    每个位置所属 doc 的首个压缩 kv 槽的全局下标，最后 padding 自然接续，总长度32：
    compressed_doc_start_per_pos: [0 ... 0,  1 ... 1, 4 ... 4, 4 ... 4,  4,  4 ]
                                   {- 5 -}  {- 14 -}  {- 3 -}  {- 8 -}  {- 2 -}

    window_topk_idxs: shape=[1, seqlen, window_size]
    * idxs[0, i] 表示每个 q[i] 可以看见的 kv 的下标，升序排列，但-1在最后
    * “看见”当且仅当两者属于同一个 doc、且满足因果律、且在 window_size 范围内、且 q[i] 有效

    compress_topk_idxs: shape=[1, seqlen, n_compressed]
    * idxs[0, i, j] = j 当且仅当 q[i] 可以看见 compressed_kv[j]，否则为-1
    * “看见”当且仅当同 doc、且 compressed_kv[j] 包含的 token 均不晚于 q[i]、且 q[i] 有效
      (也就是说，q[i] 要么完全在 compressed_kv[j] 之后，要么恰好是其最后一个 token)
    * 另外，支持 offset 参数，即给 idxs 中所有非-1的数统一加上 offset

    TODO(liangshuhao): 当前为了和原有流程兼容，压缩相关变量均截去了padding，后续会统一保留
    """

    startend_row_indices: Tensor
    ratio: int
    batch_size: int
    seqlen: int
    n_compressed: int
    doc_lens: Tensor
    doc_starts: Tensor
    doc_start_per_pos: Tensor
    doc_len_per_pos: Tensor
    is_valid: Tensor
    pos_in_doc: Tensor
    cutoff_gather_indices: Tensor
    cutoff_pos_in_doc: Tensor
    compressed_is_first: Tensor
    compressed_pos_in_doc: Tensor
    _doc_lens_cutoff: Tensor | None = None
    _doc_lens_list: list[int] | None = None
    _doc_starts_cutoff: Tensor | None = None
    _actual_n_compressed: int | None = None
    _valid_range: Tensor | None = None
    _window_topk_idxs: Tensor | None = None
    _window_size: int | None = None
    _compress_topk_idxs: Tensor | None = None
    _compress_offset: int | None = None
    _compressed_causal_mask: Tensor | None = None
    _is_first_compressed_group: Tensor | None = None

    @classmethod
    def build(
        cls,
        ratio: int,
        batch_size: int,
        seqlen: int,
        startend_row_indices: Tensor | None,
        n_compressed: int | None = None,
        dense_mode: bool = False,
    ) -> CSADocMaskMetadata | None:
        """Build metadata from ``startend_row_indices``.

        Args:
            ratio: compression ratio (e.g. 4 or 128).
            batch_size: batch size; must be 1 (varlen packs docs along seq).
            seqlen: sequence length; must equal
                ``startend_row_indices.shape[2]``.
            startend_row_indices: ``[batch_size, 1, seqlen, 1]`` document
                boundary tensor where entry ``t`` holds the (exclusive) end
                row index of ``t``'s document, or ``None`` for causal-only
                mode (returns ``None``).
            n_compressed: optional override for ``seqlen // ratio``; used when
                the caller already knows the compressed slot count.
            dense_mode: whether it is in dense mode (no indexer).

        Returns:
            A populated :class:`CSADocMaskMetadata`, or ``None`` when
            ``startend_row_indices is None``.
        """
        if startend_row_indices is None:
            return None

        from paddleformers.fleet.triton_ops.document_mask_fusion import (
            cutoff_compact_triton,
            document_mask_triton,
        )

        ratio, batch_size, seqlen, n_compressed = _normalize_csa_docmask_args(
            ratio, batch_size, seqlen, n_compressed
        )
        _validate_csa_docmask_shape(startend_row_indices, batch_size, seqlen)

        # Generate general metadata
        doc_start_per_pos, doc_len_per_pos, pos_in_doc = document_mask_triton(
            startend_row_indices.flatten()
        )
        (
            cutoff_gather_indices,
            cutoff_pos_in_doc,
            n_cutoff,
            compressed_is_first,
            compressed_pos_in_doc,
        ) = cutoff_compact_triton(pos_in_doc, doc_len_per_pos, ratio)

        n_cutoff = n_cutoff.item()
        assert n_cutoff % ratio == 0, (
            "seqlen after cutoff should be divisible by ratio, "
            f"got {n_cutoff} and {ratio}."
        )
        actual_n_compressed = n_cutoff // ratio
        cutoff_gather_indices = cutoff_gather_indices[:n_cutoff]
        cutoff_pos_in_doc = cutoff_pos_in_doc[:n_cutoff]
        compressed_is_first = compressed_is_first[:actual_n_compressed]
        compressed_pos_in_doc = compressed_pos_in_doc[:actual_n_compressed]

        # Generate metadata for indexer only
        if ratio > 1 and not dense_mode:
            (
                doc_start_per_pos,
                doc_len_per_pos,
                is_valid,
                doc_lens,
                doc_starts,
            ) = _derive_csa_doc_boundaries(startend_row_indices, seqlen)
        else:
            is_valid = doc_lens = doc_starts = None

        return cls(
            startend_row_indices=startend_row_indices,
            ratio=ratio,
            batch_size=batch_size,
            seqlen=seqlen,
            n_compressed=n_compressed,
            doc_lens=doc_lens,
            doc_starts=doc_starts,
            doc_start_per_pos=doc_start_per_pos,
            doc_len_per_pos=doc_len_per_pos,
            is_valid=is_valid,
            pos_in_doc=pos_in_doc,
            cutoff_gather_indices=cutoff_gather_indices,
            cutoff_pos_in_doc=cutoff_pos_in_doc,
            compressed_is_first=compressed_is_first,
            compressed_pos_in_doc=compressed_pos_in_doc,
            _actual_n_compressed=actual_n_compressed,
        )

    @property
    def doc_lens_cutoff(self) -> Tensor:
        if self._doc_lens_cutoff is None:
            self._doc_lens_cutoff = get_cutoff_doc_lens(
                self.doc_lens, self.ratio
            )
        return self._doc_lens_cutoff

    @property
    def doc_lens_list(self) -> list[int]:
        """Return cached Python document lengths for cuDNN THD glue."""
        if self._doc_lens_list is None:
            self._doc_lens_list = [
                int(length) for length in self.doc_lens.numpy().tolist()
            ]
        return self._doc_lens_list

    @property
    def doc_starts_cutoff(self) -> Tensor:
        if self._doc_starts_cutoff is None:
            self._doc_starts_cutoff = get_cutoff_doc_starts(
                self.doc_lens_cutoff
            )
        return self._doc_starts_cutoff

    @property
    def actual_n_compressed(self) -> int:
        if self._actual_n_compressed is None:
            total_cutoff = int(self.doc_lens_cutoff.sum().item())
            actual_n_compressed = total_cutoff // self.ratio
            if actual_n_compressed > self.n_compressed:
                raise ValueError(
                    "n_compressed must cover all packed document compressed "
                    "groups, got n_compressed="
                    f"{self.n_compressed}, required={actual_n_compressed}"
                )
            self._actual_n_compressed = actual_n_compressed
        return self._actual_n_compressed

    @property
    def valid_range(self) -> Tensor:
        if self._valid_range is None:
            self._valid_range = _build_valid_range_from_doc_bounds(
                self.ratio,
                self.seqlen,
                self.doc_start_per_pos,
                self.doc_len_per_pos,
                self.is_valid,
            ).unsqueeze(0)
        return self._valid_range

    def get_window_topk_idxs(self, window_size: int) -> Tensor:
        """Return ``[batch_size, seqlen, window_size]`` sliding-window indices.

        Indices reset at each document boundary; invalid slots (beyond the
        query position, before the document start, or on padding) are set to
        ``-1``. Cached and keyed by ``window_size``.
        """
        window_size = int(window_size)
        cache_hit = (
            self._window_topk_idxs is not None
            and self._window_size == window_size
        )
        if not cache_hit:
            from paddleformers.fleet.triton_ops.document_mask_fusion import (
                window_topk_idxs_triton,
            )

            self._window_topk_idxs = window_topk_idxs_triton(
                self.doc_start_per_pos, self.doc_len_per_pos, window_size
            )
            self._window_size = window_size
        return self._window_topk_idxs

    def get_compress_topk_idxs(self, offset: int) -> Tensor:
        """Return ``[batch_size, seqlen, n_compressed]`` compressed indices.

        For each query position, valid compressed positions (within the
        document's causal valid range) carry ``compressed_id + offset``;
        invalid slots are ``-1``. Cached and keyed by ``offset``.
        """
        offset = int(offset)
        cache_hit = (
            self._compress_topk_idxs is not None
            and self._compress_offset == offset
        )
        if not cache_hit:
            from paddleformers.fleet.triton_ops.document_mask_fusion import (
                compressed_doc_start_triton,
                compressed_topk_idxs_triton,
            )

            compressed_doc_start_per_pos = compressed_doc_start_triton(
                self.startend_row_indices.flatten(),
                self.doc_start_per_pos,
                self.ratio,
            )
            self._compress_topk_idxs = compressed_topk_idxs_triton(
                compressed_doc_start_per_pos,
                self.pos_in_doc,
                self.doc_len_per_pos,
                self.ratio,
                offset,
            )
            self._compress_offset = offset
        return self._compress_topk_idxs

    def get_compressed_causal_mask(self) -> Tensor:
        """Return ``[batch_size, seqlen, n_compressed]`` float32 causal mask.

        ``0`` for valid (within the document's compressed range), ``-inf``
        otherwise. Cached on first access.
        """
        cache_hit = self._compressed_causal_mask is not None
        if not cache_hit:
            self._compressed_causal_mask = (
                _build_compressed_causal_mask_from_valid_range(
                    self.batch_size,
                    self.seqlen,
                    self.n_compressed,
                    self.valid_range,
                )
            )
        return self._compressed_causal_mask

    def get_is_first_compressed_group(self) -> Tensor:
        """Return ``[actual_n_compressed]`` bool flags marking each document's
        first compressed group (used by overlap transforms to avoid reusing
        the previous document's data). Cached on first access.
        """
        cache_hit = self._is_first_compressed_group is not None
        if not cache_hit:
            is_first = paddle.zeros([self.actual_n_compressed], dtype="bool")
            first_indices = self.doc_starts_cutoff // self.ratio
            valid_indices = first_indices < self.actual_n_compressed
            is_first[first_indices[valid_indices]] = True
            self._is_first_compressed_group = is_first
        return self._is_first_compressed_group


def get_compress_topk_idxs(
    ratio: int,
    batch_size: int,
    seqlen: int,
    offset: int,
    startend_row_indices: Tensor | None = None,
    docmask_meta: CSADocMaskMetadata | None = None,
) -> Tensor:
    """Get compressed indices: [b, seqlen, seqlen // ratio].

    When startend_row_indices is provided, uses varlen-aware logic where
    documents' compressed KVs are packed contiguously. Each doc contributes
    cutoff_doc_len // ratio compressed positions. Padding positions (beyond
    doc end) output all -1.

    When startend_row_indices is None, uses simple causal logic where
    valid compressed range for query t is [0, (t+1) // ratio).

    Args:
        ratio: compression ratio.
        batch_size: batch size.
        seqlen: sequence length.
        offset: offset added to column indices to produce KV indices.
        startend_row_indices: [batch_size, h, seqlen, 1] tensor, or None.
        docmask_meta: optional reusable metadata for ``startend_row_indices``.

    Returns:
        result: [b, seqlen, seqlen // ratio] int32 tensor.
    """
    n_compressed = seqlen // ratio

    if docmask_meta is not None:
        return docmask_meta.get_compress_topk_idxs(offset)

    if startend_row_indices is None:
        # Original simple causal logic
        k_indices = paddle.arange(n_compressed)
        matrix = k_indices.unsqueeze(0).expand([seqlen, -1])
        causal_bound = paddle.arange(1, seqlen + 1).unsqueeze(1) // ratio
        causal_invalid = matrix >= causal_bound
        matrix = paddle.where(
            causal_invalid, paddle.full_like(matrix, -1), matrix + offset
        )
        return matrix.unsqueeze(0).expand([batch_size, -1, -1])

    docmask_meta = CSADocMaskMetadata.build(
        ratio, batch_size, seqlen, startend_row_indices, n_compressed
    )
    return docmask_meta.get_compress_topk_idxs(offset)


def get_window_topk_idxs(
    window_size: int,
    batch_size: int,
    seqlen: int,
    startend_row_indices: Tensor | None = None,
    docmask_meta: CSADocMaskMetadata | None = None,
) -> Tensor:
    """Get sliding window indices: [b, seqlen, window_size].

    When startend_row_indices is provided, the sliding window resets at
    document boundaries and padding positions output all -1.

    When startend_row_indices is None, uses simple causal sliding window.
    """
    if docmask_meta is not None:
        return docmask_meta.get_window_topk_idxs(window_size)

    if startend_row_indices is None:
        # Original simple sliding-window logic
        base = paddle.arange(seqlen).unsqueeze(1)  # [seqlen, 1]
        offsets = paddle.arange(window_size)  # [window_size]
        matrix = paddle.clip(base - window_size + 1, min=0) + offsets
        matrix = paddle.where(
            matrix > base, paddle.full_like(matrix, -1), matrix
        )
        return matrix.unsqueeze(0).expand([batch_size, -1, -1])

    docmask_meta = CSADocMaskMetadata.build(
        1, batch_size, seqlen, startend_row_indices, seqlen
    )
    return docmask_meta.get_window_topk_idxs(window_size)


def get_valid_range(
    ratio: int,
    batch_size: int,
    seqlen: int,
    startend_row_indices: Tensor | None = None,
    docmask_meta: CSADocMaskMetadata | None = None,
) -> Tensor | None:
    """Get valid compressed KV range [start, end) for each position.

    Returns shape [batch_size, seqlen, 2] with dtype int32, or None when
    startend_row_indices is not provided (causal-only mode, let the
    downstream kernel build its own valid range).
    """
    if docmask_meta is not None:
        return docmask_meta.valid_range
    if startend_row_indices is None:
        return None
    docmask_meta = CSADocMaskMetadata.build(
        ratio, batch_size, seqlen, startend_row_indices
    )
    return docmask_meta.valid_range


def _build_compressed_causal_mask(
    ratio: int,
    batch_size: int,
    seqlen: int,
    n_compressed: int,
    startend_row_indices: Tensor | None = None,
    docmask_meta: CSADocMaskMetadata | None = None,
) -> Tensor:
    """Build causal mask for compressed attention: [b, seqlen, n_compressed].

    When startend_row_indices is provided, the mask respects document
    boundaries so that queries only attend to compressed positions belonging
    to the same document.

    Returns:
        mask: [b, seqlen, n_compressed] float32, 0 for valid, -inf for invalid.
    """
    if docmask_meta is not None:
        return docmask_meta.get_compressed_causal_mask()

    if startend_row_indices is None:
        # Simple causal-only mask
        compressed_ids = paddle.arange(n_compressed).unsqueeze(0)
        positions = paddle.arange(1, seqlen + 1).unsqueeze(1)
        invalid = compressed_ids >= (positions // ratio)
        invalid = invalid.unsqueeze(0).expand(
            [batch_size, seqlen, n_compressed]
        )
        return paddle.where(
            invalid,
            paddle.full([1], float("-inf"), dtype="float32"),
            paddle.zeros([1], dtype="float32"),
        )

    docmask_meta = CSADocMaskMetadata.build(
        ratio, batch_size, seqlen, startend_row_indices, n_compressed
    )
    return docmask_meta.get_compressed_causal_mask()


def compact_kv_score_cutoff(
    doc_starts: Tensor,
    doc_lens_cutoff: Tensor,
    doc_starts_cutoff: Tensor,
    total_cutoff: int,
    kv: Tensor,
    score: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Compact `kv` and `score` by gathering valid rows into a dense buffer.

    Equivalent to the following loop implementation:
        kv_cutoff = paddle.zeros([b, total_cutoff, coff_head_dim], kv.dtype)
        score_cutoff = paddle.full([b, total_cutoff, coff_head_dim], float("-inf"), score.dtype)
        for s0, n, s1 in zip(doc_starts, doc_lens_cutoff, doc_starts_cutoff):
            kv_cutoff[:, s1 : (s1 + n), :] = kv[:, s0 : (s0 + n), :]
            score_cutoff[:, s1 : (s1 + n), :] = score[:, s0 : (s0 + n), :]
    """
    # seg_id[j] = which input segment the j-th output position came from
    seg_id = paddle.repeat_interleave(
        paddle.arange(len(doc_lens_cutoff), dtype="int64"),
        doc_lens_cutoff.astype("int64"),
    )
    # within-segment offset
    seg_offset = (
        paddle.arange(total_cutoff, dtype="int64") - doc_starts_cutoff[seg_id]
    )
    # source position in the (b, sq, head_dim) tensor
    src_idx = doc_starts[seg_id] + seg_offset

    kv_cutoff = paddle.gather(kv, src_idx, axis=1)  # [B, S, H]
    score_cutoff = paddle.gather(score, src_idx, axis=1)  # [B, S, H]

    return kv_cutoff, score_cutoff


# ---------------------------------------------------------------------------
# RoPE helper for CSA
# ---------------------------------------------------------------------------


def _apply_rope(
    x: Tensor,
    nope_dim: int,
    pos_dim: int,
    rotary_pos_emb_module,
    config: TransformerConfig,
    rotary_seq_len: int,
    ratio: int = 1,
    doc_lens_cutoff: Tensor | None = None,  # for token compressed, such as kv
    doc_lens: Tensor | None = None,  # for token not compressed, such as q
    compressed_pos_in_doc: Tensor | None = None,  # for token compressed
    position_offset: int = 0,
    high_precision_rope: bool = False,
) -> Tensor:
    """Apply RoPE to the last pos_dim dims, leaving first nope_dim unchanged.

    For compressed positions (ratio > 1), subsamples the RoPE frequencies
    by taking every ratio-th position.

    Args:
        x: [b, seq, ...dim...] where last dim = nope_dim + pos_dim
        nope_dim: dimensions that don't get RoPE
        pos_dim: dimensions that get RoPE
        rotary_pos_emb_module: RotaryEmbedding instance
        config: transformer config
        rotary_seq_len: sequence length for this tensor
        ratio: compression ratio for position subsampling
        doc_lens_cutoff: per-doc cutoff lengths for compressed RoPE (ratio > 1)
        position_offset: global position offset for CP (cp_rank * sq_local)
    """
    assert not (doc_lens_cutoff is not None and doc_lens is not None), (
        "Both doc_lens_cutoff and doc_lens are set, but only one is needed, or both of them are none."
    )
    if doc_lens_cutoff is not None:  # KV token + document mask
        compressed_doc_lens = (doc_lens_cutoff // ratio).cast("int32")
        max_compressed_doc_len = int(compressed_doc_lens.max().item())
        max_cutoff_doc_len = max_compressed_doc_len * ratio
        result = rotary_pos_emb_module(max_cutoff_doc_len, packed_seq=False)
    elif doc_lens is not None:  # Q token + document mask
        max_doc_len = int(doc_lens.max().item())
        result = rotary_pos_emb_module(max_doc_len, packed_seq=False)
    else:
        total_seq_len = (
            (rotary_seq_len + position_offset) * ratio
            if ratio > 1
            else (rotary_seq_len + position_offset)
        )
        result = rotary_pos_emb_module(total_seq_len, packed_seq=False)
    if isinstance(result, tuple):
        freqs, mscale = result
    else:
        freqs, mscale = result, 1.0
    # DSv4 reference RoPE is norm-preserving. Yarn's concentration scale is not
    # applied in the Megatron DSv4 CSA path, so keep Paddle CSA identical here.
    mscale = 1.0
    # freqs: [1, total_seq_len, pos_dim]
    if doc_lens_cutoff is not None:  # KV token + document mask
        freqs = freqs[:, :max_cutoff_doc_len:ratio, :, :]
        doc_freqs = [
            freqs[:, :doc_len, :, :]
            for doc_len in compressed_doc_lens.tolist()
            if doc_len > 0
        ]
        if doc_freqs:
            freqs = paddle.concat(doc_freqs, axis=1)
        else:
            freqs = freqs[:, :0, :, :]
        if freqs.shape[1] < rotary_seq_len:
            pad_len = rotary_seq_len - freqs.shape[1]
            freqs = paddle.concat(
                [
                    freqs,
                    paddle.zeros(
                        [1, pad_len, 1, freqs.shape[-1]], dtype=freqs.dtype
                    ),
                ],
                axis=1,
            )
        freqs = freqs[:, :rotary_seq_len, :, :]
    elif doc_lens is not None:  # Q token + document mask
        freqs = freqs[:, :max_doc_len, :, :]
        doc_freqs = [
            freqs[:, :doc_len, :, :]
            for doc_len in doc_lens.tolist()
            if doc_len > 0
        ]

        if doc_freqs:
            freqs = paddle.concat(doc_freqs, axis=1)
        else:
            freqs = freqs[:, :0, :, :]
        needed_len = position_offset + rotary_seq_len
        if freqs.shape[1] < needed_len:
            pad_len = needed_len - freqs.shape[1]
            freqs = paddle.concat(
                [
                    freqs,
                    paddle.zeros(
                        [1, pad_len, 1, freqs.shape[-1]], dtype=freqs.dtype
                    ),
                ],
                axis=1,
            )
        freqs = freqs[:, position_offset:needed_len, :, :]
    elif compressed_pos_in_doc is not None:
        freqs = paddle.gather(freqs, compressed_pos_in_doc, axis=1)
        if freqs.shape[1] < rotary_seq_len:
            freqs = paddle.nn.functional.pad(
                freqs,
                pad=[0, 0, 0, 0, 0, rotary_seq_len - freqs.shape[1]],
                mode="constant",
                value=0.0,
            )
        freqs = freqs[:, :rotary_seq_len]
    elif ratio > 1:  # CP without document mask -> KV
        freqs = freqs[:, position_offset * ratio : total_seq_len : ratio, :][
            :, :rotary_seq_len, :
        ]
    else:
        freqs = freqs[:, position_offset : position_offset + rotary_seq_len, :]

    squeeze_head = x.ndim == 3
    if squeeze_head:
        x = x.unsqueeze(2)  # [b, s, 1, dim]

    if getattr(config, "apply_rope_fusion", False) and not high_precision_rope:
        from paddleformers.fleet.triton_ops import fused_apply_mla_rope_inplace

        out = fused_apply_mla_rope_inplace(x, freqs, nope_dim, mscale)
    else:
        x_nope = x[..., :nope_dim]
        x_pe = x[..., nope_dim:]
        x_pe = _apply_rotary_pos_emb_bshd(
            x_pe,
            freqs,
            mscale=mscale,
            rotary_interleaved=False,
            multi_latent_attention=True,
            mla_output_remove_interleaving=True,
            high_precision_rope=high_precision_rope,
        )
        out = paddle.concat([x_nope, x_pe], axis=-1)

    if squeeze_head:
        out = out.squeeze(2)
    return out


# ---------------------------------------------------------------------------
# Unfused compressed sparse attention
# ---------------------------------------------------------------------------


def _resolve_csa_indexer_loss_topk_effective(
    config, index_topk: int, n_compressed: int
) -> int:
    """Return the TileLang CSA indexer top-k width used by indexer loss.

    Phase semantics (driven by ``dsa_indexer_use_sparse_loss``, **not** by the
    new TileLang switches):

    * Phase 3 (``dsa_indexer_use_sparse_loss=True``): ``min(index_topk,
      n_compressed)`` — selected-topk semantics, same as the existing
      ``FusedDSAIndexerLoss`` / ``CSAIndexer.forward`` choice.
    * Phase 2 (``dsa_indexer_use_sparse_loss=False``): ``n_compressed`` — the
      selected set covers the full compressed candidate range and is later
      consumed as full-range KL by the indexer loss path.
    """
    use_sparse_loss = bool(getattr(config, "dsa_indexer_use_sparse_loss", True))
    if use_sparse_loss:
        return min(int(index_topk), int(n_compressed))
    return int(n_compressed)


def _resolve_csa_indexer_attn_topk_effective(
    index_topk: int, n_compressed: int
) -> int:
    """Return the compressed top-k width consumed by main CSA attention."""
    return min(int(index_topk), int(n_compressed))


def _map_compressed_topk_to_kv_full(
    topk_indices_compressed: Tensor,
    sq: int,
    ratio: int,
    offset: int,
) -> Tensor:
    """Map compressed block ids to ``kv_full`` indices.

    For each query position ``t``, only ``(t + 1) // ratio`` compressed blocks
    are causally valid. Slots whose compressed id is out of that range are
    written back as ``-1``; valid slots are shifted by ``offset`` (which is
    the original sequence length so that compressed entries follow the raw
    KV positions inside ``kv_full``).
    """
    n_valid_per_pos = (
        paddle.arange(1, sq + 1, dtype=topk_indices_compressed.dtype).unsqueeze(
            1
        )
        // ratio
    ).unsqueeze(0)  # [1, sq, 1]
    valid = (topk_indices_compressed >= 0) & (
        topk_indices_compressed < n_valid_per_pos
    )
    return paddle.where(
        valid,
        topk_indices_compressed + offset,
        paddle.full_like(topk_indices_compressed, -1),
    )


def _compute_attn_target_on_selected_set(
    query_mla: Tensor,  # [b, sq, np, hn]  DETACHED
    key_comp_mla: Tensor,  # [b, sk, hn] shared compressed KV, or legacy [b, sk, np, hn]
    topk_indices: Tensor,  # [b, sq, topk_eff] int32, -1 for invalid slots
    softmax_scale: float,
    tp_group=None,
) -> Tensor:
    """Construct attention target ``p[t, S_t]`` on the selected compressed set.

    Mathematically equivalent to ``_compute_dsa_indexer_loss`` with
    ``sparse_loss=True``, but evaluated only on the selected slots ``S_t``
    given by ``topk_indices`` instead of materializing the full ``[B,Sq,Sk]``
    distribution. Invalid (``-1``) slots are masked out before softmax and
    receive zero target probability after L1 normalization.

    The result has shape ``[b, sq, topk_eff]`` in fp32 and is the multi-head
    aggregated, L1 normalized target distribution used as the second argument
    of ``KL(target || index_prob)``.
    """
    b, sq, np, hn = query_mla.shape
    topk_eff = topk_indices.shape[-1]

    # Per-head full attention scores [b, np, sq, sk]. DSv4 compressed KV is
    # shared across query heads as [b, sk, hn]; the legacy per-head-expanded
    # [b, sk, np, hn] shape is still accepted for non-TileLang references.
    q = query_mla.transpose([0, 2, 1, 3]).cast("float32")  # [b, np, sq, hn]
    if len(key_comp_mla.shape) == 3:
        k = key_comp_mla.transpose([0, 2, 1]).cast("float32").unsqueeze(1)
    else:
        k = key_comp_mla.transpose([0, 2, 3, 1]).cast("float32")
    attn_scores = paddle.matmul(q, k) * float(softmax_scale)  # [b, np, sq, sk]

    # Replace -1 with 0 for safe gather; then mask back to -inf afterwards.
    valid = topk_indices >= 0  # [b, sq, topk_eff]
    safe_indices = paddle.where(
        valid, topk_indices, paddle.zeros_like(topk_indices)
    ).cast("int64")
    safe_indices_exp = safe_indices.unsqueeze(1).expand([b, np, sq, topk_eff])
    selected_logits = paddle.take_along_axis(
        attn_scores, safe_indices_exp, axis=-1
    )  # [b, np, sq, topk_eff]

    # Mask invalid slots so softmax assigns them zero probability.
    valid_bn = valid.unsqueeze(1)  # [b, 1, sq, topk_eff]
    neg_inf = paddle.full([1], float("-inf"), dtype="float32")
    selected_logits = paddle.where(valid_bn, selected_logits, neg_inf)

    # Avoid all-(-inf) rows producing NaN in softmax: zero such rows out.
    row_valid = valid.any(axis=-1, keepdim=True)  # [b, sq, 1]
    row_valid_bn = row_valid.unsqueeze(1)  # [b, 1, sq, 1]
    selected_logits = paddle.where(
        row_valid_bn, selected_logits, paddle.zeros_like(selected_logits)
    )

    probs = F.softmax(selected_logits, axis=-1, dtype="float32")
    # Re-zero fully invalid rows post-softmax (softmax of zeros is uniform).
    probs = probs * row_valid_bn.cast("float32")

    # Aggregate over heads, optional TP all-reduce, then L1 normalize.
    target = probs.sum(axis=1)  # [b, sq, topk_eff]
    if tp_group is not None and getattr(tp_group, "nranks", 1) > 1:
        paddle.distributed.all_reduce(target.contiguous(), group=tp_group)
    target = target / target.sum(axis=-1, keepdim=True).clip(min=1e-10)

    # Zero out invalid slots (so they contribute nothing to KL).
    target = paddle.where(valid, target, paddle.zeros_like(target))
    return target


def _compute_fused_csa_indexer_loss_forward(
    index_q: Tensor,
    weights: Tensor,
    index_k_comp: Tensor,
    query_mla: Tensor,
    key_comp_mla: Tensor,
    valid_range: Tensor,
    ratio: int,
    topk_effective: int,
    softmax_scale: float,
    loss_coeff: float,
    tp_group=None,
    seq_offset: int = 0,
    indexer_backend: str = "tilelang",
    loss_mask: Tensor | None = None,
    global_valid_count: float | None = None,
    startend_row_indices: Tensor | None = None,
    docmask_meta: CSADocMaskMetadata | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    from paddleformers.fleet.tilelang_ops import (
        csa_attn_target_reducesum,
        csa_indexer_topk_fwd,
    )

    if indexer_backend == "cudnn":
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        topk_indices, _, topk_scores = cudnn_indexer_topk_fwd(
            index_q,
            index_k_comp,
            weights,
            ratio=int(ratio),
            topk_effective=int(topk_effective),
            valid_range=valid_range,
            startend_row_indices=(
                docmask_meta.startend_row_indices
                if docmask_meta is not None
                else startend_row_indices
            ),
            doc_lens=docmask_meta.doc_lens_list
            if docmask_meta is not None
            else None,
            seq_offset=int(seq_offset),
            return_topk_scores=True,
        )
        invalid_mask = topk_indices < 0
        # Avoid NaN from softmax on all-(-inf) rows: zero them before softmax.
        row_valid = (~invalid_mask).any(axis=-1, keepdim=True)  # [B, Sq, 1]
        topk_scores = paddle.where(
            row_valid, topk_scores, paddle.zeros_like(topk_scores)
        )
        topk_probs = paddle.nn.functional.softmax(topk_scores, axis=-1)
        topk_probs = topk_probs * row_valid.cast(topk_probs.dtype)
    else:
        topk_indices, topk_probs = csa_indexer_topk_fwd(
            index_q,
            index_k_comp,
            weights,
            ratio=int(ratio),
            topk_effective=int(topk_effective),
            seq_offset=int(seq_offset),
            valid_range=valid_range,
        )

    if tp_group is not None and getattr(tp_group, "nranks", 1) > 1:
        target = _compute_attn_target_on_selected_set(
            query_mla, key_comp_mla, topk_indices, softmax_scale, tp_group
        )
    else:
        target = csa_attn_target_reducesum(
            query_mla,
            key_comp_mla,
            topk_indices,
            softmax_scale,
        )

    eps = 1e-10
    kl_per_elem = target * (
        paddle.log(target + eps) - paddle.log(topk_probs + eps)
    )
    # kl_per_elem: [B, Sq, topk] -> sum over topk -> [B, Sq]
    kl_per_pos = kl_per_elem.sum(axis=-1)
    if loss_mask is not None:
        lm = loss_mask.reshape(kl_per_pos.shape).astype(kl_per_pos.dtype)
        loss = (kl_per_pos * lm).sum() / global_valid_count * float(loss_coeff)
    else:
        loss = kl_per_pos.mean() * float(loss_coeff)
    return loss, topk_indices, topk_probs, target


class TileLangCSAIndexerLossAutoScaler(paddle.autograd.PyLayer):
    """Attach TileLang CSA indexer loss gradients to the main output.

    This is the TileLang analogue of ``DSAIndexerLossAutoScaler``. It avoids
    chaining a scalar-loss PyLayer behind another PyLayer in the full training
    graph while preserving the same gradient scale semantics.
    """

    @staticmethod
    def forward(
        ctx,
        output: Tensor,
        index_q: Tensor,
        weights: Tensor,
        index_k_comp: Tensor,
        topk_indices: Tensor,
        topk_probs: Tensor,
        target: Tensor,
        loss_coeff: float,
        indexer_backend: str = "tilelang",
        num_rows_override: float | None = None,
        loss_mask: Tensor | None = None,
    ) -> Tensor:
        ctx.save_for_backward(
            index_q.detach(),
            weights.detach(),
            index_k_comp.detach(),
            topk_indices.detach(),
            topk_probs.detach(),
            target.detach(),
        )
        ctx.loss_coeff = float(loss_coeff)
        ctx.indexer_backend = str(indexer_backend)
        ctx.loss_mask = loss_mask
        if num_rows_override is not None:
            ctx.num_rows = num_rows_override
        else:
            ctx.num_rows = float(target.shape[0] * target.shape[1])
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            index_q,
            weights,
            index_k_comp,
            topk_indices,
            topk_probs,
            target,
        ) = ctx.saved_tensor()

        scale = DSAIndexerLossAutoScaler._main_loss_backward_scale

        if ctx.indexer_backend == "cudnn":
            from paddleformers.fleet.cudnn_ops import csa_indexer_bwd

            # cuDNN multiplies its internal score-grad by ``grad_loss`` in
            # the GEMM kernel; pass the externally-set scaler as ``grad_loss``.
            if scale is None:
                grad_loss_arg = None
            elif isinstance(scale, paddle.Tensor):
                grad_loss_arg = scale
            else:
                grad_loss_arg = paddle.to_tensor(
                    float(scale), dtype=paddle.float32
                )

            # Apply loss_mask to mask out padding positions in backward
            bwd_target = target
            bwd_topk_probs = topk_probs
            if getattr(ctx, "loss_mask", None) is not None:
                lm = ctx.loss_mask.reshape(
                    [target.shape[0], target.shape[1], 1]
                ).astype(target.dtype)
                bwd_target = target * lm
                bwd_topk_probs = topk_probs * lm

            # cuDNN kernel internally divides by (B * S_q). When loss_mask is
            # provided, we want 1/global_valid_count instead. Compensate by
            # scaling loss_coeff so the kernel's internal division yields the
            # correct normalization.
            cudnn_loss_coeff = ctx.loss_coeff
            if getattr(ctx, "loss_mask", None) is not None:
                B_Sq = float(target.shape[0] * target.shape[1])
                cudnn_loss_coeff = (
                    ctx.loss_coeff
                    * B_Sq
                    / max(getattr(ctx, "num_rows", 1.0), 1.0)
                )

            grad_q, grad_weights, grad_k = csa_indexer_bwd(
                index_q,
                weights,
                index_k_comp,
                bwd_target,
                bwd_topk_probs,
                topk_indices,
                loss_coeff=cudnn_loss_coeff,
                grad_loss=grad_loss_arg,
            )
        elif ctx.indexer_backend == "tilelang":
            from paddleformers.fleet.tilelang_ops import csa_indexer_bwd

            grad_index_scores = (topk_probs - target) * (
                ctx.loss_coeff / max(getattr(ctx, "num_rows", 1.0), 1.0)
            )
            # Apply loss_mask to zero out gradients for padding positions
            if getattr(ctx, "loss_mask", None) is not None:
                lm = ctx.loss_mask.reshape(
                    [grad_index_scores.shape[0], grad_index_scores.shape[1], 1]
                ).astype(grad_index_scores.dtype)
                grad_index_scores = grad_index_scores * lm
            if scale is not None:
                grad_index_scores = grad_index_scores * scale

            grad_q, grad_weights, grad_k = csa_indexer_bwd(
                index_q,
                weights,
                index_k_comp,
                topk_indices,
                grad_index_scores,
            )
        else:
            raise NotImplementedError(
                f"CSA indexer backend {ctx.indexer_backend!r} not implemented."
            )

        if grad_q.dtype != index_q.dtype:
            grad_q = grad_q.cast(index_q.dtype)
        if grad_weights.dtype != weights.dtype:
            grad_weights = grad_weights.cast(weights.dtype)
        if grad_k.dtype != index_k_comp.dtype:
            grad_k = grad_k.cast(index_k_comp.dtype)

        grads = (grad_output, grad_q, grad_weights, grad_k, None, None, None)
        if getattr(ctx, "loss_mask", None) is not None:
            grads += (None,)
        return grads


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


@dataclass
class CompressorSublayersSpec:
    """Sublayer specifications for CSA Compressor."""

    linear_wkv: type | LayerSpec = None
    linear_wgate: type | LayerSpec = None
    norm: type | LayerSpec = None


class Compressor(nn.Layer):
    """Gated pooling compressor for CSA.

    Compresses a sequence by pooling groups of compress_ratio tokens using
    learned gated weights.

    For ratio=4: overlapping compression (coff=2)
    For ratio=128: non-overlapping compression (coff=1)
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: CompressorSublayersSpec,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
        rotary_pos_emb=None,
    ):
        super().__init__()
        self.config = config
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        # CSA layers (1 < ratio < 128) use overlapping compression (coff=2);
        # HCA (ratio 128) and window-only (ratio 0) do not overlap.
        self.overlap = 1 < compress_ratio < 128
        self.coff = 1 + int(self.overlap)
        self.rotate = rotate
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim or 0
        self.rotary_pos_emb = rotary_pos_emb

        proj_out_dim = self.coff * head_dim

        self.linear_wkv = build_spec_layer(
            sublayers_spec.linear_wkv,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            skip_weight_param_allocation=False,
            tp_group=None,
        )
        self.linear_wgate = build_spec_layer(
            sublayers_spec.linear_wgate,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            skip_weight_param_allocation=False,
            tp_group=None,
        )

        self.ape = self.create_parameter(
            shape=[compress_ratio, proj_out_dim],
            dtype="float32",
            default_initializer=nn.initializer.Normal(
                std=config.init_method_std
                if hasattr(config, "init_method_std")
                else 0.02
            ),
        )
        self._cast_to_low_precision = False

        self.norm = build_spec_layer(
            sublayers_spec.norm,
            config=config,
            hidden_size=head_dim,
            eps=getattr(config, "rms_norm_eps", 1e-5),
        )

        self.use_fp8_qat = getattr(config, "use_fp8_qat", False)
        self.use_fast_hadamard = getattr(config, "use_fast_hadamard", False)
        self.swa_high_precision_norm = getattr(
            config, "swa_high_precision_norm", False
        )
        self.high_precision_rope = getattr(config, "high_precision_rope", False)

    def _overlap_transform(
        self,
        tensor: Tensor,
        fill_value: float = 0,
        is_first: Tensor | None = None,
    ) -> Tensor:
        """Apply overlapping window transform for 4x compression.

        Input shape:  [b, n_groups, ratio, coff * head_dim]
        Output shape: [b, n_groups, 2 * ratio, head_dim]

        Args:
            tensor: input tensor.
            fill_value: fill value for positions without valid previous data.
            is_first: optional [n_groups] bool mask that is True for each
                compressed group that starts a new document (no valid
                predecessor). When provided, prevents pulling data across
                document boundaries.
        """
        b, n_groups, ratio, _ = tensor.shape
        d = self.head_dim
        new_tensor = paddle.full(
            [b, n_groups, 2 * ratio, d], fill_value, dtype=tensor.dtype
        )
        # Second half of each group's projection goes to positions [ratio:]
        new_tensor[:, :, ratio:, :] = tensor[:, :, :, d:]
        # First half of previous group goes to positions [:ratio] (skip group 0)
        new_tensor[:, 1:, :ratio, :] = tensor[:, :-1, :, :d]
        # Zero out at document boundaries: the first compressed group of each
        # document has no valid previous group to pull from.
        if is_first is not None:
            # is_first: [n_groups] bool mask; positions where is_first=True
            # should not use previous group data
            # is_first[0] is always True (handled by skipping group 0 above),
            # so we only need to handle is_first[1:] for groups 1..n_groups-1
            if n_groups > 1:
                boundary_mask = is_first[1:]  # [n_groups - 1]
                bm = boundary_mask.reshape([1, -1, 1, 1])
                new_tensor[:, 1:, :ratio, :] = paddle.where(
                    bm,
                    paddle.full([1], fill_value, dtype=tensor.dtype),
                    new_tensor[:, 1:, :ratio, :],
                )
        return new_tensor

    def forward(
        self,
        x: Tensor,
        cp_group=None,
        docmask_meta: CSADocMaskMetadata | None = None,
    ) -> Tensor | None:
        """Compress hidden states into shorter KV sequence.

        Args:
            x: [b, sq, hidden_size]
            cp_group: CP process group.
            docmask_meta: document-mask metadata, or None for simple causal mode.

        Returns:
            compressed_kv: [b, sq // ratio, head_dim] or None if too short.
            In CP mode, returns the local rank's slice of the global compressed KV.
        """
        b, sq, _ = x.shape
        ratio = self.compress_ratio

        if sq < ratio:
            return None
        if self.swa_high_precision_norm:
            kv = linear_bf16_fp32(
                x, self.linear_wkv.weight
            )  # [b, sq, coff * head_dim]
            score = linear_bf16_fp32(
                x, self.linear_wgate.weight
            )  # [b, sq, coff * head_dim]
        else:
            kv, _ = self.linear_wkv(x)  # [b, sq, coff * head_dim]
            score, _ = self.linear_wgate(x)  # [b, sq, coff * head_dim]

        # CP: gather projected KV globally before pooling (Miles pattern).
        # This lets the compressor pool across the full sequence while keeping
        # communication cheap (projected dim << hidden_size).
        # After all-gather, kv/score are global and sq is updated to sq_global.
        # The rest of the compression logic is shared with the non-CP path.
        if cp_group is not None and getattr(cp_group, "nranks", 1) > 1:
            kv = all_gather_cp(kv, dim=1, group=cp_group)
            score = all_gather_cp(score, dim=1, group=cp_group)
            b, sq, _ = kv.shape

        # Shared compression logic for both CP and non-CP paths.
        if docmask_meta is not None:
            # per-document cutoff, pack contiguously without padding
            n_compressed = sq // ratio
            actual_n_compressed = docmask_meta.actual_n_compressed
            cutoff_gather_indices = docmask_meta.cutoff_gather_indices

            # Pack only valid cutoff data contiguously (no padding)
            kv = paddle.gather(kv, cutoff_gather_indices, axis=1)
            score = paddle.gather(score, cutoff_gather_indices, axis=1)

            # Reshape: [b, actual_n_compressed, ratio, coff * head_dim]
            kv = kv.reshape([b, actual_n_compressed, ratio, -1])
            score = score.reshape([b, actual_n_compressed, ratio, -1])

            # APE: [ratio, coff * head_dim] -> [1, 1, ratio, coff * head_dim]
            ape = self.ape.reshape([1, 1, ratio, -1])
            ape = ape.cast(score.dtype) if _ACCURACY_COMPATIBLE_KERNEL else ape
            score = score + ape

            if self.overlap:
                is_first = docmask_meta.compressed_is_first
                kv = self._overlap_transform(
                    kv, fill_value=0, is_first=is_first
                )
                score = self._overlap_transform(
                    score, fill_value=float("-inf"), is_first=is_first
                )

            # TODO: should we cast?
            # Gated pooling: softmax over the pool_dim, weighted sum.
            kv = (kv * F.softmax(score, axis=2)).sum(axis=2)
            # kv: [b, actual_n_compressed, head_dim]

            if self.swa_high_precision_norm:
                kv = self.norm(
                    kv,
                    high_precision_norm=True,
                    return_high_precision_norm=True,
                )
            else:
                kv = self.norm(kv.cast(x.dtype))

            # Pad to n_compressed before RoPE
            if actual_n_compressed < n_compressed:
                pad_len = n_compressed - actual_n_compressed
                kv = paddle.concat(
                    [
                        kv,
                        paddle.zeros(
                            [b, pad_len, kv.shape[-1]], dtype=kv.dtype
                        ),
                    ],
                    axis=1,
                )

            # Apply RoPE with subsampled positions
            if self.rotary_pos_emb is not None and self.qk_pos_emb_head_dim > 0:
                kv = _apply_rope(
                    kv,
                    self.head_dim - self.qk_pos_emb_head_dim,
                    self.qk_pos_emb_head_dim,
                    self.rotary_pos_emb,
                    self.config,
                    n_compressed,
                    ratio=ratio,
                    compressed_pos_in_doc=docmask_meta.compressed_pos_in_doc,
                    high_precision_rope=self.high_precision_rope,
                )

            if self.rotate:
                kv = rotate_activation(
                    kv,
                    use_fast_hadamard=self.use_fast_hadamard,
                    high_precision_hadamard=self.swa_high_precision_norm,
                )
                if self.use_fp8_qat:
                    kv = fp8_simulate_qat(kv, 128)
            else:
                if self.use_fp8_qat:
                    nope_dim = self.head_dim - self.qk_pos_emb_head_dim
                    kv[..., :nope_dim] = fp8_simulate_qat(
                        kv[..., :nope_dim], 64
                    )

            if self.swa_high_precision_norm:
                kv = kv.cast(x.dtype)
            return kv  # [b, n_compressed, head_dim]
        else:
            # Original simple cutoff logic
            n_compressed = sq // ratio
            cutoff = n_compressed * ratio
            if cutoff < sq:
                kv = kv[:, :cutoff, :]
                score = score[:, :cutoff, :]
            doc_lens_cutoff = None

        # Reshape: [b, n_compressed, ratio, coff * head_dim]
        kv = kv.reshape([b, n_compressed, ratio, -1])
        score = score.reshape([b, n_compressed, ratio, -1])

        # APE: [ratio, coff * head_dim] -> [1, 1, ratio, coff * head_dim]
        ape = self.ape.reshape([1, 1, ratio, -1])
        ape = ape.cast(score.dtype) if _ACCURACY_COMPATIBLE_KERNEL else ape
        score = score + ape

        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0)
            score = self._overlap_transform(score, fill_value=float("-inf"))

        # TODO: old megatron-aligned logic. This will cause possible acc declining
        # weights = F.softmax(score, axis=2).cast(kv.dtype)
        # kv = (kv * weights).sum(axis=2)  # [b, n_compressed, head_dim]
        # Gated pooling: softmax over the pool_dim, weighted sum.
        kv = (kv * F.softmax(score, axis=2)).sum(
            axis=2
        )  # [b, n_compressed, head_dim]

        kv = self.norm(kv.cast(x.dtype))

        # Apply RoPE with subsampled positions
        if self.rotary_pos_emb is not None and self.qk_pos_emb_head_dim > 0:
            kv = _apply_rope(
                kv,
                self.head_dim - self.qk_pos_emb_head_dim,
                self.qk_pos_emb_head_dim,
                self.rotary_pos_emb,
                self.config,
                n_compressed,
                ratio=ratio,
                doc_lens_cutoff=doc_lens_cutoff,
            )

        if self.rotate:
            kv = rotate_activation(kv, use_fast_hadamard=self.use_fast_hadamard)
            if self.use_fp8_qat:
                kv = fp8_simulate_qat(kv, 128)
        else:
            if self.use_fp8_qat:
                nope_dim = self.head_dim - self.qk_pos_emb_head_dim
                kv[..., :nope_dim] = fp8_simulate_qat(kv[..., :nope_dim], 64)
        return kv  # [b, n_compressed, head_dim]


# ---------------------------------------------------------------------------
# CSAIndexer
# ---------------------------------------------------------------------------


@dataclass
class CSAIndexerSublayersSpec:
    """Sublayer specifications for CSAIndexer."""

    linear_wq_b: type | LayerSpec = None
    linear_weights_proj: type | LayerSpec = None
    compressor: type | LayerSpec = None


class CSAIndexer(nn.Layer):
    """Learned top-k retrieval over compressed positions for CSA.

    Computes index scores to select the most relevant compressed KV positions
    for each query token. Uses its own nested Compressor with Hadamard rotation
    for key generation.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: CSAIndexerSublayersSpec,
        compress_ratio: int,
        rotary_pos_emb=None,
    ):
        super().__init__()
        self.config = config
        self.compress_ratio = compress_ratio
        self.hidden_size = config.hidden_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim or 0
        self.q_lora_rank = config.q_lora_rank

        self.index_n_heads = config.dsa_index_n_heads
        self.index_head_dim = config.dsa_index_head_dim
        self.index_topk = config.dsa_index_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        self.rotary_pos_emb = rotary_pos_emb

        # Q projection: q_lora_rank -> n_heads * head_dim
        self.linear_wq_b = build_spec_layer(
            sublayers_spec.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            skip_weight_param_allocation=False,
            tp_group=None,
        )

        # Weights projection: hidden_size -> n_heads
        self.linear_weights_proj = build_spec_layer(
            sublayers_spec.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            skip_weight_param_allocation=False,
            tp_group=None,
        )

        # Own compressor (smaller head_dim, with Hadamard rotation)
        self.compressor = build_spec_layer(
            sublayers_spec.compressor,
            config=config,
            compress_ratio=compress_ratio,
            head_dim=self.index_head_dim,
            rotate=True,
            rotary_pos_emb=rotary_pos_emb,
        )

        self.use_fp8_qat = getattr(config, "use_fp8_qat", False)
        self.use_fast_hadamard = getattr(config, "use_fast_hadamard", False)

    def forward_before_topk(
        self,
        x: Tensor,  # [b, sq, hidden_size]
        qr: Tensor,  # [b, sq, q_lora_rank]
        position_offset: int = 0,
        cp_group=None,
        docmask_meta: CSADocMaskMetadata | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute Q, compressed K, and weights before top-k selection.

        In CP mode, position_offset and cp_group are forwarded to the internal
        compressor so that indexer K is computed over the full global sequence.
        """
        b, sq, _ = x.shape
        doc_lens = docmask_meta.doc_lens if docmask_meta is not None else None
        # Q path
        q, _ = self.linear_wq_b(qr)  # [b, sq, n_heads * head_dim]
        q = q.reshape([b, sq, self.index_n_heads, self.index_head_dim])
        if self.rotary_pos_emb is not None and self.qk_pos_emb_head_dim > 0:
            q = _apply_rope(
                q,
                self.index_head_dim - self.qk_pos_emb_head_dim,
                self.qk_pos_emb_head_dim,
                self.rotary_pos_emb,
                self.config,
                sq,
                ratio=1,
                doc_lens=doc_lens,
                position_offset=position_offset,
            )
        q = rotate_activation(q, use_fast_hadamard=self.use_fast_hadamard)

        # k QAT:
        if self.use_fp8_qat:
            q = fp8_simulate_qat(q, 128)

        # K path: own compressor (already applies RoPE and rotation internally)
        k = self.compressor(
            x,
            cp_group=cp_group,
            docmask_meta=docmask_meta,
        )  # [b, n_compressed, index_head_dim]

        # Weights
        weights, _ = self.linear_weights_proj(x)  # [b, sq, n_heads]
        weights = weights * (self.index_n_heads**-0.5)

        return q, k, weights

    def forward(
        self,
        x: Tensor,
        qr: Tensor,
        mask: Tensor | None = None,
        docmask_meta: CSADocMaskMetadata | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return (index_scores, topk_indices).

        Args:
            x: [b, sq, hidden_size]
            qr: [b, sq, q_lora_rank]
            mask: [b, sq, n_compressed] optional causal mask

        Returns:
            index_scores: [b, sq, n_compressed]
            topk_indices: [b, sq, topk]
        """
        q, k, weights = self.forward_before_topk(
            x, qr, docmask_meta=docmask_meta
        )
        effective_topk = min(self.index_topk, k.shape[1])
        weights = (
            weights * self.softmax_scale
        )  # 对齐 fwd 和 recompute fwd的一致性
        index_scores, topk_indices = fused_qk_topk_naive(
            q, k, weights, effective_topk, mask
        )
        return index_scores, topk_indices


# ---------------------------------------------------------------------------
# CompressedSparseAttention (core attention)
# ---------------------------------------------------------------------------


@dataclass
class CompressedSparseAttentionSublayersSpec:
    """Sublayer specifications for CompressedSparseAttention."""

    compressor: type | LayerSpec = None
    indexer: type | LayerSpec = None


class CompressedSparseAttention(FleetLayer):
    """Core attention combining sliding window + compressed KV attention.

    Conditionally builds Compressor and CSAIndexer based on compress_ratio:
      - ratio=0: window-only attention
      - ratio=4: window + 4x compressed + learned CSAIndexer
      - ratio=128: window + 128x compressed, attend to all compressed positions
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: CompressedSparseAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        is_mtp_layer: bool = False,
        is_swa: bool = False,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
        rotary_pos_emb: nn.Layer = None,
        compress_ratio: int = 0,
    ):
        super().__init__(config)
        DSAIndexerLossLoggingHelper.register_total_num_layers(config)
        self.config = config
        self.layer_number = layer_number
        if is_mtp_layer:
            self.layer_number += self.config.num_hidden_layers + 1
        self.pg_collection = pg_collection
        tp_size = int(getattr(config, "tensor_model_parallel_size", 1))
        if pg_collection is not None and pg_collection.tp is not None:
            tp_size = max(tp_size, int(getattr(pg_collection.tp, "nranks", 1)))
        if tp_size > 1:
            raise NotImplementedError(
                "CompressedSparseAttention does not support tensor parallelism "
                f"> 1, got tp={tp_size}."
            )
        self.tp_group = None
        self.compress_ratio = compress_ratio
        self.window_size = config.csa_window_size
        self.v_head_dim = config.v_head_dim
        self.n_local_heads = config.num_attention_heads
        self.softmax_scale = config.v_head_dim**-0.5

        # CP state: derived from pg_collection.cp; cp_size=1 means CP disabled
        cp_pg = pg_collection.cp if pg_collection is not None else None
        if cp_pg is not None and getattr(cp_pg, "nranks", 1) > 1:
            self.cp_group = cp_pg
            self.cp_size = cp_pg.nranks
            self.cp_rank = cp_pg.rank
            self.cp_enabled = True
        else:
            self.cp_group = None
            self.cp_size = 1
            self.cp_rank = 0
            self.cp_enabled = False

        # Learnable attention sink per head
        self.attn_sink = self.create_parameter(
            shape=[self.n_local_heads],
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.0),
        )
        self._cast_to_low_precision = False

        # Conditionally build Compressor (ratio > 1)
        if self.compress_ratio > 1:
            self.compressor = build_spec_layer(
                sublayers_spec.compressor,
                config=config,
                compress_ratio=self.compress_ratio,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=rotary_pos_emb,
            )
        else:
            self.compressor = None

        # Conditionally build Indexer for CSA layers (1 < ratio < 128) and not dense_mode.
        # ratio 128 (HCA) intentionally falls through to the attend-to-all path.
        # Keep this in sync with dsa_attention.py indexer-layer count.
        if 1 < self.compress_ratio < 128 and not config.csa_dense_mode:
            self.indexer = build_spec_layer(
                sublayers_spec.indexer,
                config=config,
                compress_ratio=self.compress_ratio,
                rotary_pos_emb=rotary_pos_emb,
            )
        else:
            self.indexer = None

    def _compute_indexer_compressed_topk_idxs(
        self,
        query: Tensor,
        x: Tensor,
        qr: Tensor,
        compressed_kv: Tensor,
        n_compressed: int,
        offset: int,
        loss_mask: Tensor | None = None,
        global_valid_count: float | None = None,
        docmask_meta: CSADocMaskMetadata | None = None,
    ) -> tuple[Tensor, Tensor | None, tuple | None]:
        """Build indexer-selected compressed KV indices and loss state."""
        b, sq, np_heads, _ = query.shape
        indexer_loss = None
        tilelang_indexer_loss_state = None

        x_det = x.detach()
        qr_det = qr.detach()
        if self.training:
            x_det.stop_gradient = False
            qr_det.stop_gradient = False

        # Loss and main attention intentionally use different top-k widths
        # during phase 2. ``dsa_indexer_use_sparse_loss=False`` expands only
        # the indexer loss to the full compressed range; the main CSA attention
        # remains sparse and consumes ``min(index_topk, n_compressed)``.
        indexer_backend = getattr(
            self.config, "csa_indexer_backend", "tilelang"
        )
        # The indexer loss path is only active during the grad-enabled forward.
        # Full recompute runs the first forward under no_grad; that pass should
        # only materialize main-attention indices. The backend branch remains
        # fixed across both forwards.
        need_indexer_loss = self.training and paddle.is_grad_enabled()
        loss_topk_effective = _resolve_csa_indexer_loss_topk_effective(
            self.config,
            self.indexer.index_topk,
            n_compressed,
        )
        attn_topk_effective = _resolve_csa_indexer_attn_topk_effective(
            self.indexer.index_topk,
            n_compressed,
        )

        causal_mask = _build_compressed_causal_mask(
            self.compress_ratio,
            b,
            sq,
            n_compressed,
            docmask_meta=docmask_meta,
        )
        valid_range = get_valid_range(
            int(self.compress_ratio),
            b,
            sq,
            docmask_meta=docmask_meta,
        )

        def compute_fused_indexer_loss(backend: str):
            # Training grad-enabled forward with TileLang/cuDNN backend.
            # The same backend's top-k-only kernel is used during recompute's
            # first no-grad forward below.
            indexer_loss_coeff = getattr(
                self.config, "dsa_indexer_loss_coeff", 0.0
            )
            q_indexer_bf, k_indexer_bf, weights_indexer_bf = (
                self.indexer.forward_before_topk(
                    x_det,
                    qr_det,
                    docmask_meta=docmask_meta,
                )
            )
            key_comp_mla = compressed_kv.detach()
            (
                loss,
                topk_indices,
                topk_probs,
                target,
            ) = _compute_fused_csa_indexer_loss_forward(
                q_indexer_bf,
                weights_indexer_bf,
                k_indexer_bf,
                query.detach(),
                key_comp_mla,
                valid_range,
                int(self.compress_ratio),
                int(loss_topk_effective),
                float(self.softmax_scale),
                float(indexer_loss_coeff),
                self.tp_group,
                indexer_backend=backend,
                loss_mask=loss_mask,
                global_valid_count=global_valid_count,
                startend_row_indices=docmask_meta.startend_row_indices
                if docmask_meta is not None
                else None,
                docmask_meta=docmask_meta,
            )
            loss_state = (
                q_indexer_bf,
                weights_indexer_bf,
                k_indexer_bf,
                topk_indices,
                topk_probs,
                target,
                float(indexer_loss_coeff),
                backend,
                global_valid_count if loss_mask is not None else None,
                loss_mask,
            )
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=loss,
                    layer_number=self.layer_number,
                    num_layers=DSAIndexerLossLoggingHelper.get_total_num_layers(
                        self.config
                    ),
                )
            return loss, topk_indices, loss_state

        if (
            indexer_backend == "cudnn"
        ):  # cuDNN branch for both recompute forwards; inner condition decides whether to compute loss.
            if need_indexer_loss:  # Grad-enabled recompute forward; compute cuDNN fused selected-set loss and top-k.
                (
                    indexer_loss,
                    topk_indices_compressed,
                    tilelang_indexer_loss_state,
                ) = compute_fused_indexer_loss("cudnn")
            else:  # First recompute no-grad forward; only materialize cuDNN top-k for attention.
                from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
                    cudnn_indexer_topk_fwd,
                )

                with paddle.no_grad():
                    q_indexer_cu, k_indexer_cu, weights_indexer_cu = (
                        self.indexer.forward_before_topk(
                            x_det,
                            qr_det,
                            docmask_meta=docmask_meta,
                        )
                    )
                    cu_topk_indices, _cu_topk_length = cudnn_indexer_topk_fwd(
                        q_indexer_cu,
                        k_indexer_cu,
                        weights_indexer_cu,
                        ratio=self.compress_ratio,
                        topk_effective=attn_topk_effective,
                        valid_range=valid_range,
                        startend_row_indices=docmask_meta.startend_row_indices
                        if docmask_meta is not None
                        else None,
                        doc_lens=docmask_meta.doc_lens_list
                        if docmask_meta is not None
                        else None,
                    )
                topk_indices_compressed = cu_topk_indices

        elif (
            indexer_backend == "tilelang"
        ):  # TileLang branch for both recompute forwards; inner condition decides whether to compute loss.
            if need_indexer_loss:  # Grad-enabled recompute forward; compute TileLang fused selected-set loss and top-k.
                (
                    indexer_loss,
                    topk_indices_compressed,
                    tilelang_indexer_loss_state,
                ) = compute_fused_indexer_loss("tilelang")
            else:  # First recompute no-grad forward; only materialize TileLang top-k for attention.
                from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd

                with paddle.no_grad():
                    q_indexer_tl, k_indexer_tl, weights_indexer_tl = (
                        self.indexer.forward_before_topk(
                            x_det,
                            qr_det,
                            docmask_meta=docmask_meta,
                        )
                    )
                    tl_topk_indices, _tl_topk_scores = csa_indexer_topk_fwd(
                        q_indexer_tl,
                        k_indexer_tl,
                        weights_indexer_tl,
                        ratio=self.compress_ratio,
                        topk_effective=attn_topk_effective,
                        valid_range=valid_range,
                    )
                topk_indices_compressed = tl_topk_indices

        elif (
            indexer_backend == "unfused"
        ):  # Unfused branch for both recompute forwards; inner condition decides whether to compute loss.
            if need_indexer_loss:  # Grad-enabled recompute forward; compute Paddle indexer loss and top-k.
                q_indexer, k_indexer, weights_indexer = (
                    self.indexer.forward_before_topk(
                        x_det,
                        qr_det,
                        docmask_meta=docmask_meta,
                    )
                )
                indexer_loss_coeff = getattr(
                    self.config, "dsa_indexer_loss_coeff", 0.0
                )
                key_for_loss = compressed_kv.unsqueeze(2).expand(
                    [-1, -1, np_heads, -1]
                )
                weights_for_loss = weights_indexer * self.indexer.softmax_scale
                mask_for_loss = causal_mask.unsqueeze(1)

                indexer_loss = FusedDSAIndexerLoss.apply(
                    q_indexer,
                    weights_for_loss,
                    k_indexer,
                    query.detach(),
                    key_for_loss.detach(),
                    self.softmax_scale,
                    min(self.indexer.index_topk, n_compressed),
                    indexer_loss_coeff,
                    mask_for_loss,
                    getattr(self.config, "dsa_indexer_use_sparse_loss", True),
                    self.tp_group,
                    loss_mask,
                    global_valid_count,
                )

                topk_indices_compressed = FusedDSAIndexerLoss._last_topk_indices

                if indexer_loss_coeff > 0:
                    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                        loss=indexer_loss,
                        layer_number=self.layer_number,
                        num_layers=DSAIndexerLossLoggingHelper.get_total_num_layers(
                            self.config
                        ),
                    )
            else:  # First recompute no-grad forward; only materialize unfused top-k for attention.
                _, topk_indices_compressed = self.indexer(
                    x_det,
                    qr_det,
                    mask=causal_mask,
                    docmask_meta=docmask_meta,
                )

        if (
            topk_indices_compressed.shape[-1] > attn_topk_effective
        ):  # Loss path may return wider top-k than attention consumes.
            topk_indices_compressed = topk_indices_compressed[
                ..., :attn_topk_effective
            ].contiguous()

        compress_topk_idxs = _map_compressed_topk_to_kv_full(
            topk_indices_compressed,
            sq,
            self.compress_ratio,
            offset,
        )

        return compress_topk_idxs, indexer_loss, tilelang_indexer_loss_state

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
        x: Tensor = None,
        qr: Tensor = None,
        input_ids: Tensor | None = None,
        docmask_meta: CSADocMaskMetadata | None = None,
    ) -> Tensor:
        """Forward pass for CompressedSparseAttention.

        Args:
            query: [b, sq, np, v_head_dim]
            key:   [b, sq, 1, v_head_dim] (single-head MQA)
            value: unused (key == value in DSv4 Hybrid MQA)
            attention_mask: unused (causal is implicit)
            x:     [b, sq, hidden_size] original hidden states
            qr:    [b, sq, q_lora_rank] compressed query representation

        Returns:
            output: [b, sq, np * v_head_dim]
        """
        b, sq, np_heads, hn = query.shape

        if docmask_meta is not None:
            assert b == 1, (
                "when docmask_meta is not None, ",
                f"only support batch_size == 1, current batch_size: {b}",
            )

        # Compute loss_mask from input_ids (mask out padding tokens)
        if input_ids is not None and self.indexer is not None:
            if (
                get_context_parallel_world_size() > 1
                and not self.config.experimental_dataflow
            ):
                # In EB data flow, we need to gather input_ids here to get right denom.
                input_ids_global = ContextParallelGatherOp.apply(
                    input_ids, axis=1, mode=self.config.cp_balance_mode
                )
            else:
                input_ids_global = input_ids

            pad_token_id = getattr(self.config, "pad_token_id", 0)
            assert pad_token_id is not None, (
                "pad_token_id must be set in config when input_ids is provided"
            )
            loss_mask_global = (input_ids_global != pad_token_id).astype(
                paddle.float32
            )
            if self.cp_enabled:
                # input_ids is global [b, sq_global]; scatter to local chunk
                loss_mask_global = loss_mask_global.reshape(
                    [b, self.cp_size * sq]
                )
                global_valid_count = max(float(loss_mask_global.sum()), 1.0)
                position_offset = self.cp_rank * sq
                loss_mask = loss_mask_global[
                    :, position_offset : position_offset + sq
                ]
            else:
                loss_mask = loss_mask_global.reshape([b, sq])
                global_valid_count = max(float(loss_mask.sum()), 1.0)
        else:
            loss_mask = None
            global_valid_count = None

        if self.cp_enabled:
            return self._forward_cp(
                query,
                key,
                x,
                qr,
                loss_mask=loss_mask,
                global_valid_count=global_valid_count,
                docmask_meta=docmask_meta,
            )

        if docmask_meta is not None and self.compress_ratio > 1:
            actual_n_compressed = docmask_meta.actual_n_compressed
        elif self.compress_ratio > 1:
            actual_n_compressed = sq // self.compress_ratio
        else:
            actual_n_compressed = 0

        # Step 1: Prepare single-head KV
        kv = key.squeeze(2)  # [b, sq, v_head_dim]

        # Step 2: Compression
        if (
            self.compressor is not None
            and self.compress_ratio > 1
            and actual_n_compressed > 0
        ):
            compressed_kv = self.compressor(
                x,
                docmask_meta=docmask_meta,
            )  # [b, n_compressed, v_head_dim]
            if compressed_kv is not None:
                kv_full = paddle.concat([kv, compressed_kv], axis=1)
                n_compressed = compressed_kv.shape[1]
            else:
                kv_full = kv
                n_compressed = 0
        else:
            kv_full = kv
            n_compressed = 0

        offset = sq  # compressed indices start after original positions

        # Step 3: Window indices
        window_idxs = get_window_topk_idxs(
            self.window_size,
            b,
            sq,
            docmask_meta=docmask_meta,
        )

        # Step 4: Compressed indices
        indexer_loss = None
        tilelang_indexer_loss_state = None

        if (
            self.compress_ratio > 1
            and n_compressed > 0
            and actual_n_compressed > 0
        ):
            if self.indexer is not None:
                (
                    compress_topk_idxs,
                    indexer_loss,
                    tilelang_indexer_loss_state,
                ) = self._compute_indexer_compressed_topk_idxs(
                    query,
                    x,
                    qr,
                    compressed_kv,
                    n_compressed,
                    offset,
                    loss_mask=loss_mask,
                    global_valid_count=global_valid_count,
                    docmask_meta=docmask_meta,
                )
            else:
                # ratio=128: attend to all compressed positions
                compress_topk_idxs = get_compress_topk_idxs(
                    self.compress_ratio,
                    b,
                    sq,
                    offset,
                    docmask_meta=docmask_meta,
                )

            if compress_topk_idxs.dtype != window_idxs.dtype:
                compress_topk_idxs = compress_topk_idxs.cast(window_idxs.dtype)
            topk_idxs = paddle.concat(
                [window_idxs, compress_topk_idxs], axis=-1
            )
        else:
            topk_idxs = window_idxs

        topk_idxs = topk_idxs.cast("int32")

        # Step 5: Sparse attention
        output = self.compressed_sparse_attn(
            query,
            kv_full,
            self.attn_sink,
            topk_idxs,
            self.softmax_scale,
        )

        # Step 6: Attach indexer loss
        if tilelang_indexer_loss_state is not None and self.training:
            output = TileLangCSAIndexerLossAutoScaler.apply(
                output,
                *tilelang_indexer_loss_state,
            )
        elif indexer_loss is not None and self.training:
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        return output

    def _forward_cp(
        self,
        query: Tensor,
        key: Tensor,
        x: Tensor,
        qr: Tensor,
        loss_mask: Tensor | None = None,
        global_valid_count: float | None = None,
        docmask_meta: CSADocMaskMetadata | None = None,
    ) -> Tensor:
        """CP-aware forward: local compress + all-gather, sparse attention.

        Mirrors the non-CP forward() structure exactly, with CP adaptations:
          1. All-gather KV + compress (Miles pattern: gather projected, pool globally)
          2. Indexer topk + fused loss (same three-branch logic as non-CP)
          3. Sparse attention (same compressed_sparse_attn dispatch)
          4. Attach loss (TileLangCSAIndexerLossAutoScaler or DSAIndexerLossAutoScaler)

        Gradient correctness:
          - all_gather_cp backward (reduce-scatter) routes attention/loss grads
          - indexer_loss / cp_size corrects local-mean to global-mean scaling
          - param grads are partial (each rank sees local Q); production ZeRO
            reduce_scatter(SUM) + optimizer x cp_size aggregates them correctly
        """
        b, sq, np_heads, hn = query.shape
        sq_global = sq * self.cp_size
        position_offset = self.cp_rank * sq
        q_positions = paddle.arange(
            position_offset, position_offset + sq, dtype="int64"
        )
        # Step 1: Window topk (CP-aware: uses global q_positions)
        if docmask_meta is None:
            window_idxs = get_window_topk_idxs_cp(
                q_positions, self.window_size, b, sq_global
            )
        else:
            full_window_idxs = get_window_topk_idxs(
                self.window_size,
                b,
                sq_global,
                docmask_meta=docmask_meta,
            )
            window_idxs = full_window_idxs[
                :, position_offset : position_offset + sq, ...
            ]

        # Step 2: All-gather KV + compress
        kv_local = key.squeeze(2)  # [b, sq, hn]
        kv_global = all_gather_cp(kv_local, dim=1, group=self.cp_group)

        compressed_kv_global = None
        n_compressed_local = 0
        if (
            self.compressor is not None
            and self.compress_ratio > 1
            and sq >= self.compress_ratio
        ):
            assert sq % self.compress_ratio == 0, (
                f"CP requires sq_local ({sq}) divisible by compress_ratio ({self.compress_ratio})"
            )
            n_compressed_local = sq // self.compress_ratio
        n_compressed_global = n_compressed_local * self.cp_size

        # Compute actual_n_compressed accounting for document boundaries
        if docmask_meta is not None and self.compress_ratio > 1:
            actual_n_compressed = docmask_meta.actual_n_compressed
        elif self.compress_ratio > 1:
            actual_n_compressed = n_compressed_global
        else:
            actual_n_compressed = 0

        offset = sq_global  # compressed indices follow vanilla KV in kv_full

        if (
            self.compressor is not None
            and self.compress_ratio > 1
            and n_compressed_local > 0
            and actual_n_compressed > 0
        ):
            # inside the compressor, we will all-gather all the compressed KV
            compressed_kv_global = self.compressor(
                x,
                cp_group=self.cp_group,
                docmask_meta=docmask_meta,
            )
            kv_full = paddle.concat([kv_global, compressed_kv_global], axis=1)
        else:
            kv_full = kv_global

        # Step 3: Compressed topk + optional fused indexer loss
        indexer_loss = None
        tilelang_indexer_loss_state = None

        if (
            self.compress_ratio > 1
            and n_compressed_global > 0
            and actual_n_compressed > 0
        ):
            if self.indexer is not None:
                x_det = x.detach()
                qr_det = qr.detach()
                if self.training:
                    x_det.stop_gradient = False
                    qr_det.stop_gradient = False

                indexer_backend = getattr(
                    self.config, "csa_indexer_backend", "tilelang"
                )
                use_tilelang_indexer = indexer_backend == "tilelang"
                use_cudnn_indexer = indexer_backend == "cudnn"
                use_fused_indexer_loss_path = (
                    (use_tilelang_indexer or use_cudnn_indexer)
                    and self.training
                    and paddle.is_grad_enabled()
                )
                loss_topk_effective = _resolve_csa_indexer_loss_topk_effective(
                    self.config, self.indexer.index_topk, n_compressed_global
                )
                attn_topk_effective = _resolve_csa_indexer_attn_topk_effective(
                    self.indexer.index_topk, n_compressed_global
                )

                # valid_range for varlen: [b, sq_local, 2] or None
                if docmask_meta is not None:
                    valid_range = docmask_meta.valid_range[
                        :, position_offset : position_offset + sq, :
                    ]
                else:
                    valid_range = None

                q_indexer_bf, k_indexer_global, weights_indexer_bf = (
                    self.indexer.forward_before_topk(
                        x_det,
                        qr_det,
                        position_offset=position_offset,
                        cp_group=self.cp_group,
                        docmask_meta=docmask_meta,
                    )
                )

                indexer_loss_coeff = getattr(
                    self.config, "dsa_indexer_loss_coeff", 0.0
                )

                if use_fused_indexer_loss_path:
                    # CP training grad-enabled forward with TileLang/cuDNN
                    # indexer backend.
                    # Fused TileLang/cuDNN: single path produces topk + loss.
                    # key_comp_mla is 3D [b, n_comp_global, hn] (shared across heads).
                    key_comp_mla = compressed_kv_global.detach()
                    (
                        indexer_loss,
                        topk_indices_compressed,
                        topk_probs,
                        target,
                    ) = _compute_fused_csa_indexer_loss_forward(
                        q_indexer_bf,
                        weights_indexer_bf,
                        k_indexer_global,
                        query.detach(),
                        key_comp_mla,
                        valid_range,
                        int(self.compress_ratio),
                        int(loss_topk_effective),
                        float(self.softmax_scale),
                        float(indexer_loss_coeff),
                        self.tp_group,
                        seq_offset=position_offset,
                        loss_mask=loss_mask,
                        global_valid_count=global_valid_count,
                        startend_row_indices=docmask_meta.startend_row_indices
                        if docmask_meta is not None
                        else None,
                        docmask_meta=docmask_meta,
                        indexer_backend=indexer_backend,
                    )
                    tilelang_indexer_loss_state = (
                        q_indexer_bf,
                        weights_indexer_bf,
                        k_indexer_global,
                        topk_indices_compressed,
                        topk_probs,
                        target,
                        float(indexer_loss_coeff)
                        if loss_mask is not None
                        else float(indexer_loss_coeff) / self.cp_size,
                        getattr(self.config, "csa_indexer_backend", "tilelang"),
                        global_valid_count if loss_mask is not None else None,
                        loss_mask,
                    )
                    if indexer_loss_coeff > 0:
                        # CP fused training path logs only when indexer loss is
                        # enabled.
                        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                            loss=indexer_loss,
                            layer_number=self.layer_number,
                            num_layers=self.config.num_hidden_layers,
                        )
                    # Scale: each rank's loss is mean over sq_local;
                    # global loss is mean over sq_global = sq_local * cp_size.
                    if loss_mask is None:
                        indexer_loss = indexer_loss / self.cp_size

                elif (
                    self.training
                    and not use_tilelang_indexer
                    and not use_cudnn_indexer
                ):
                    # CP training forward with unfused indexer backend.
                    # Paddle reference loss path
                    key_for_loss = (
                        compressed_kv_global.detach()
                        .unsqueeze(2)
                        .expand([-1, -1, np_heads, -1])
                    )

                    if docmask_meta is None:
                        causal_mask = build_causal_mask_cp(
                            q_positions,
                            n_compressed_global,
                            self.compress_ratio,
                            b,
                        )
                    else:
                        causal_mask_full = (
                            docmask_meta.get_compressed_causal_mask()
                        )
                        causal_mask = causal_mask_full[
                            :, position_offset : position_offset + sq, ...
                        ]

                    weights_for_loss = (
                        weights_indexer_bf * self.indexer.softmax_scale
                    )
                    mask_for_loss = causal_mask.unsqueeze(1)

                    indexer_loss = FusedDSAIndexerLoss.apply(
                        q_indexer_bf,
                        weights_for_loss,
                        k_indexer_global,
                        query.detach(),
                        key_for_loss.detach(),
                        self.softmax_scale,
                        min(self.indexer.index_topk, n_compressed_global),
                        indexer_loss_coeff,
                        mask_for_loss,
                        getattr(
                            self.config, "dsa_indexer_use_sparse_loss", True
                        ),
                        self.tp_group,
                        loss_mask,
                        global_valid_count,
                    )
                    topk_indices_compressed = (
                        FusedDSAIndexerLoss._last_topk_indices
                    )
                    if indexer_loss_coeff > 0:
                        # CP unfused training path logs only when indexer loss
                        # is enabled.
                        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                            loss=indexer_loss,
                            layer_number=self.layer_number,
                            num_layers=self.config.num_hidden_layers,
                        )
                    if loss_mask is None:
                        indexer_loss = indexer_loss / self.cp_size

                elif not use_tilelang_indexer and not use_cudnn_indexer:
                    # CP eval/no-grad forward with unfused backend; only
                    # materialize attention top-k.
                    # Inference-only Paddle topk (use already-gathered global K)
                    if docmask_meta is None:
                        causal_mask = build_causal_mask_cp(
                            q_positions,
                            n_compressed_global,
                            self.compress_ratio,
                            b,
                        )
                    else:
                        causal_mask_full = (
                            docmask_meta.get_compressed_causal_mask()
                        )
                        causal_mask = causal_mask_full[
                            :, position_offset : position_offset + sq, ...
                        ]

                    _, topk_indices_compressed = fused_qk_topk_naive(
                        q_indexer_bf,
                        k_indexer_global,
                        weights_indexer_bf,
                        attn_topk_effective,
                        causal_mask,
                    )

                # TileLang/cuDNN fwd-only topk (no loss, or loss already produced above)
                if (
                    (use_tilelang_indexer or use_cudnn_indexer)
                    and not use_fused_indexer_loss_path
                ):  # CP eval/no-grad or recompute first forward with TileLang/cuDNN backend.
                    with paddle.no_grad():
                        if use_cudnn_indexer:
                            from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
                                cudnn_indexer_topk_fwd,
                            )

                            topk_indices_compressed, _ = cudnn_indexer_topk_fwd(
                                q_indexer_bf,
                                k_indexer_global,
                                weights_indexer_bf,
                                ratio=self.compress_ratio,
                                topk_effective=attn_topk_effective,
                                valid_range=valid_range,
                                startend_row_indices=docmask_meta.startend_row_indices
                                if docmask_meta is not None
                                else None,
                                doc_lens=docmask_meta.doc_lens_list
                                if docmask_meta is not None
                                else None,
                                seq_offset=position_offset,
                            )
                        else:
                            from paddleformers.fleet.tilelang_ops import (
                                csa_indexer_topk_fwd,
                            )

                            topk_indices_compressed, _ = csa_indexer_topk_fwd(
                                q_indexer_bf,
                                k_indexer_global,
                                weights_indexer_bf,
                                ratio=self.compress_ratio,
                                topk_effective=attn_topk_effective,
                                seq_offset=position_offset,
                                valid_range=valid_range,
                            )

                if (
                    topk_indices_compressed.shape[-1] > attn_topk_effective
                ):  # CP loss path may return wider top-k than attention consumes.
                    topk_indices_compressed = topk_indices_compressed[
                        ..., :attn_topk_effective
                    ].contiguous()

                compress_topk_idxs = map_compressed_topk_to_kv_full_cp(
                    topk_indices_compressed,
                    q_positions,
                    self.compress_ratio,
                    offset,
                )
            else:
                # HCA path: attend to all compressed positions
                if docmask_meta is None:
                    compress_topk_idxs = get_compress_topk_idxs_cp(
                        q_positions,
                        self.compress_ratio,
                        b,
                        offset,
                        n_compressed_global,
                    )
                else:
                    compress_topk_idxs = docmask_meta.get_compress_topk_idxs(
                        offset
                    )
                    compress_topk_idxs = compress_topk_idxs[
                        :, position_offset : position_offset + sq, ...
                    ]

            if compress_topk_idxs.dtype != window_idxs.dtype:
                compress_topk_idxs = compress_topk_idxs.cast(window_idxs.dtype)
            topk_idxs = paddle.concat(
                [window_idxs, compress_topk_idxs], axis=-1
            )
        else:
            topk_idxs = window_idxs

        topk_idxs = topk_idxs.cast("int32")

        # Step 4: Sparse attention (same dispatch as non-CP)
        output = self.compressed_sparse_attn(
            query, kv_full, self.attn_sink, topk_idxs, self.softmax_scale
        )

        # Step 5: Attach indexer loss
        if tilelang_indexer_loss_state is not None and self.training:
            output = TileLangCSAIndexerLossAutoScaler.apply(
                output, *tilelang_indexer_loss_state
            )
        elif indexer_loss is not None and self.training:
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        return output

    def compressed_sparse_attn(
        self,
        query: Tensor,
        kv_full: Tensor,
        attn_sink: Tensor,
        topk_idxs: Tensor,
        softmax_scale: float,
    ):
        from paddleformers.fleet.fusions.csa_sparse_attn import csa_sparse_attn

        attn_sink_fp32 = (
            attn_sink.cast("bfloat16").cast("float32")
            if _ACCURACY_COMPATIBLE_KERNEL
            else attn_sink.cast("float32")
        )
        sparse_attn_backend = getattr(
            self.config, "csa_sparse_attn_backend", "tilelang"
        )
        return csa_sparse_attn(
            query,
            kv_full,
            attn_sink_fp32,
            topk_idxs,
            softmax_scale,
            backend=sparse_attn_backend,
        )
