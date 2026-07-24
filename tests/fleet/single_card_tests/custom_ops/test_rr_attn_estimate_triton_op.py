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

import math
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core

# from paddlefleet_ops._extensions.flashmask import (
from paddlefleet_ops import rr_attn_estimate_triton_func


class TestRRAttnEstimateTritonOP(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["bfloat16"]
        self.stride = [8]
        self.dim = [128]
        self.threshold = [0.3, 0.8]
        self.shape_cases = [
            (8, 128, 128, 4, 1),
            (1, 1024, 1023, 2, 1),
            (2, 2048, 2000, 1, 1),
            (1, 256, 127, 1, 1),
            (1, 127, 1000, 8, 2),
        ]
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

        np.random.seed(42)
        paddle.seed(42)

    def _run_reference_impl(
        self,
        query_states: paddle.Tensor,
        key_states: paddle.Tensor,
        mask_dense: paddle.Tensor,
        *,
        block_size: int = 128,
        stride: int = 8,
        causal: bool = False,
        chunk_size: int = 512,
    ):
        assert mask_dense is not None
        assert chunk_size % stride == 0, (
            "chunk_size must be divisible by stride"
        )
        assert block_size % stride == 0, (
            "block_size must be divisible by stride"
        )

        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])

        batch_size, num_q_head, q_len, head_dim = query_states.shape
        _, num_kv_head, k_len, _ = key_states.shape

        if num_q_head != num_kv_head:
            assert num_q_head % num_kv_head == 0
            num_groups = num_q_head // num_kv_head
            key_states = paddle.repeat_interleave(
                key_states, num_groups, axis=1
            )

        nheads_dense_mask = mask_dense.shape[1]
        mask_dense = mask_dense.astype("int32")
        if num_q_head != nheads_dense_mask:
            assert num_q_head % nheads_dense_mask == 0
            num_groups_indices = num_q_head // nheads_dense_mask
            mask_dense = paddle.repeat_interleave(
                mask_dense, num_groups_indices, axis=1
            )

        def get_pad_len(length, align):
            return (align - length % align) % align

        q_pad_len = get_pad_len(q_len, chunk_size)
        k_pad_len = get_pad_len(k_len, chunk_size)

        padded_q_len = q_len + q_pad_len
        padded_k_len = k_len + k_pad_len

        mask_dense = mask_dense.astype(paddle.float32)
        if q_pad_len > 0:
            query_states = F.pad(query_states, (0, 0, 0, q_pad_len), value=0)
            # Pad Mask Dense Height (Q dim)
            if mask_dense.shape[2] == q_len:
                mask_dense = F.pad(mask_dense, (0, 0, 0, q_pad_len), value=0)

        if k_pad_len > 0:
            key_states = F.pad(key_states, (0, 0, 0, k_pad_len), value=0)
            # Pad Mask Dense Width (K dim)
            if mask_dense.shape[3] == k_len:
                mask_dense = F.pad(mask_dense, (0, k_pad_len, 0, 0), value=0)

        num_k_strides = padded_k_len // stride
        k_global_indices = paddle.arange(padded_k_len)
        k_is_non_padding = k_global_indices < k_len
        k_is_non_padding = k_is_non_padding.reshape([num_k_strides, stride])
        stride_valid_length = k_is_non_padding.astype("float32").sum(axis=-1)

        head_offsets = (paddle.arange(num_q_head) % stride)[:, None]
        num_q_strides = padded_q_len // stride
        stride_starts = (paddle.arange(num_q_strides) * stride)[None, :]

        gather_indices = (
            (head_offsets + stride_starts).unsqueeze(-1).unsqueeze(0)
        )  # [1, H, QC, 1]
        gather_indices_expanded = paddle.expand(
            gather_indices, [batch_size, -1, -1, head_dim]
        )
        sampled_query = paddle.take_along_axis(
            query_states, gather_indices_expanded, axis=2
        )

        mask_gather_idx = gather_indices  # [1, H, QC, 1]
        if mask_dense.shape[1] == 1:
            mask_gather_idx = mask_gather_idx[:, 0:1, :, :]  # [1, 1, QC, 1]

        # Expand to K len
        mask_gather_idx = paddle.expand(
            mask_gather_idx, [batch_size, -1, -1, padded_k_len]
        )
        sampled_mask_dense = paddle.take_along_axis(
            mask_dense, mask_gather_idx, axis=2
        )

        attn_sums_list = []
        boundary_masks_list = []

        scale = 1.0 / math.sqrt(head_dim) / stride
        q_chunk_size = chunk_size // stride
        num_chunks = num_q_strides // q_chunk_size

        for i in range(num_chunks):
            st = i * q_chunk_size
            ed = (i + 1) * q_chunk_size

            q_chunk = sampled_query[:, :, st:ed, :]  # [B, H, qc, D]
            mask_chunk = sampled_mask_dense[:, :, st:ed, :]  # [B, H/1, qc, K]

            logits = paddle.matmul(q_chunk, key_states, transpose_y=True)
            logits = logits * scale

            logits = paddle.reshape(
                logits,
                [batch_size, num_q_head, q_chunk_size, num_k_strides, stride],
            )
            mask_chunk = paddle.reshape(
                mask_chunk,
                [batch_size, num_q_head, q_chunk_size, num_k_strides, stride],
            )

            logical_mask = mask_chunk

            if causal:
                q_idx_val = paddle.arange(q_chunk_size)[None, :]
                global_q_stride_idx = st + q_idx_val
                real_row = global_q_stride_idx * stride + head_offsets

                k_idx_val = paddle.arange(num_k_strides)[:, None]
                s_idx_val = paddle.arange(stride)[None, :]
                real_col = k_idx_val * stride + s_idx_val

                shift = k_len - q_len
                real_row = real_row.reshape([1, num_q_head, q_chunk_size, 1, 1])
                real_col = real_col.reshape([1, 1, 1, num_k_strides, stride])

                is_causal = (real_row + shift >= real_col).astype(logits.dtype)
                logical_mask = logical_mask * is_causal

            final_effective_mask = logical_mask * k_is_non_padding.astype(
                logits.dtype
            )
            logits = logits * final_effective_mask

            passed_counts = final_effective_mask.sum(axis=-1)
            total_valid_counts = stride_valid_length

            is_fully_masked = passed_counts == 0
            is_partially_masked = (passed_counts > 0) & (
                passed_counts < total_valid_counts
            )

            logits_stride = logits.sum(axis=-1)

            if is_fully_masked.any():
                neg_inf = paddle.to_tensor(
                    float("-inf"),
                    dtype=logits_stride.dtype,
                    place=logits_stride.place,
                )
                logits_stride = paddle.where(
                    is_fully_masked, neg_inf, logits_stride
                )

            scores_stride = F.softmax(logits_stride, axis=-1)
            scores_stride = paddle.nan_to_num(scores_stride, 0.0).astype(
                query_states.dtype
            )

            ratio = block_size // stride
            B, H, qc, ks = scores_stride.shape
            qb = qc // ratio
            kb = ks // ratio

            reshape_dims = [B, H, qb, ratio, kb, ratio]

            scores_reshaped = scores_stride.reshape(reshape_dims)
            attn_sum_chunk = scores_reshaped.sum(axis=[3, 5])
            attn_sums_list.append(attn_sum_chunk)

            boundary_stride = is_partially_masked.astype("float32")

            boundary_reshaped = boundary_stride.reshape(reshape_dims)
            boundary_mask_chunk = boundary_reshaped.max(axis=[3, 5])
            boundary_masks_list.append(boundary_mask_chunk.astype("bool"))

        final_attn_sums = paddle.concat(attn_sums_list, axis=2)
        final_boundary_mask = paddle.concat(boundary_masks_list, axis=2)

        valid_q_blocks = (q_len + block_size - 1) // block_size
        valid_k_blocks = (k_len + block_size - 1) // block_size

        return (
            final_attn_sums[:, :, :valid_q_blocks, :valid_k_blocks],
            final_boundary_mask[:, :, :valid_q_blocks, :valid_k_blocks],
        )

    def _run_rr_attn_estimate_test(
        self,
        batch_size,
        seqlen_q,
        seqlen_k,
        nheads,
        nheads_kv,
        nheads_startend_row_indices,
        dtype,
        gen_startend_row_indices,
        stride,
        dim,
        threshold,
    ):
        assert nheads % nheads_kv == 0

        q_ref_t = paddle.randn(
            shape=[batch_size, seqlen_q, nheads, dim], dtype="float32"
        )
        k_ref_t = paddle.randn(
            shape=[batch_size, seqlen_k, nheads_kv, dim], dtype="float32"
        )

        q_naive_t = q_ref_t.astype(dtype)
        k_naive_t = k_ref_t.astype(dtype)

        q_kernel_t = q_naive_t.detach().clone()
        k_kernel_t = k_naive_t.detach().clone()

        startend_row_indices, causal = gen_startend_row_indices(
            batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices
        )

        mask_dense = self._flashmask_to_densemask(
            startend_row_indices, seqlen_q, nheads_startend_row_indices, causal
        )

        print(
            f"Testing Config: B={batch_size}, Q={seqlen_q}, K={seqlen_k}, HQ={nheads}, H={nheads_kv}, Stride={stride}, Causal={causal}"
        )

        out_ref, bound_ref = self._run_reference_impl(
            q_ref_t,
            k_ref_t,
            mask_dense,
            block_size=128,
            stride=stride,
            causal=causal,
            chunk_size=2048,
        )

        out_naive, _ = self._run_reference_impl(
            q_naive_t,
            k_naive_t,
            mask_dense,
            block_size=128,
            stride=stride,
            causal=causal,
            chunk_size=2048,
        )

        out_kernel, bound_kernel, topp_kernel = rr_attn_estimate_triton_func(
            q=q_kernel_t,
            k=k_kernel_t,
            startend_row_indices=startend_row_indices,
            stride=stride,
            threshold=threshold,
            causal=causal,
        )

        out_ref = out_ref.astype("float32")
        bound_ref = bound_ref.astype("int32")

        out_naive = out_naive.astype("float32")

        out_kernel = out_kernel.astype("float32")
        bound_kernel = bound_kernel.astype("int32")

        # -----------------------------------------------------------
        # Test 1: Boundary Check Mask (Exact Match)
        # -----------------------------------------------------------
        print("\n--- Testing Boundary Mask ---")
        assert bound_ref.shape == bound_kernel.shape, (
            f"Shape Mismatch! Ref: {bound_ref.shape}, Kernel: {bound_kernel.shape}"
        )

        mask_diff_tensor = paddle.sum(paddle.abs(bound_ref - bound_kernel))
        mask_diff = mask_diff_tensor.item()
        total_elements = bound_ref.size

        print(
            f"Boundary Mask Mismatches: {mask_diff} / {total_elements} ({(mask_diff / total_elements) * 100:.4f}%)"
        )

        if mask_diff > 0:
            mismatch_indices = paddle.nonzero(bound_ref != bound_kernel)

            top_indices = mismatch_indices[:5]
            print("First 5 mismatches (Indices):", top_indices.tolist())

            ref_vals = paddle.gather_nd(bound_ref, top_indices)
            kernel_vals = paddle.gather_nd(bound_kernel, top_indices)

            print("Ref Values:", ref_vals.tolist())
            print("Kernel Values:", kernel_vals.tolist())

        assert mask_diff == 0, "[FAIL] Boundary masks do not match exactly!"

        # -----------------------------------------------------------
        # Test 2: Attention Score Estimation (Dynamic Tolerance)
        # -----------------------------------------------------------

        fwd_atol = (
            2 * paddle.max(paddle.abs(out_ref + 0.3 - 0.3 - out_ref)).item()
        )
        rtol = 2

        # Baseline Error
        naive_diff = paddle.abs(out_naive - out_ref)
        naive_err = paddle.max(naive_diff).item()

        # Kernel Error
        kernel_diff = paddle.abs(out_kernel - out_ref)
        kernel_err = paddle.max(kernel_diff).item()
        kernel_mean_err = paddle.mean(kernel_diff).item()

        allowed_error = rtol * naive_err + fwd_atol + 1e-4

        if kernel_err > allowed_error:
            print("[FAIL] Score error exceeds tolerance!")
            print(f"Max Diff: {kernel_err}")
            print(f"Allowed: {allowed_error}")

            flat_idx = paddle.argmax(kernel_diff).item()
            err_indices = []
            for dim in reversed(out_ref.shape):
                err_indices.append(flat_idx % dim)
                flat_idx //= dim
            err_indices = tuple(reversed(err_indices))

            ref_val = out_ref[err_indices].item()
            kernel_val = out_kernel[err_indices].item()
            print(
                f"Max Error at {err_indices}: Ref={ref_val}, Kernel={kernel_val}"
            )

        assert kernel_err <= allowed_error, (
            f"Output max diff {kernel_err} > Allowed {allowed_error}"
        )

        # -----------------------------------------------------------
        # Test 3: Top-p block selection
        # -----------------------------------------------------------

        self._verify_topp(out_kernel, topp_kernel, threshold)

    def _find_blocks_chunked(
        self,
        input_tensor,
        threshold,
    ):
        assert threshold is not None

        x = input_tensor.astype("float32")
        B, H, C, N = x.shape
        total_sum = x.sum(axis=-1, keepdim=True)
        cutoff = total_sum * float(threshold)

        sorted_values, sorted_idx = paddle.compat.sort(
            x, dim=-1, descending=True
        )  # both [B,H,C,N]

        prefix = paddle.cumsum(sorted_values, axis=-1)  # [B,H,C,N]
        keep = (prefix - sorted_values) < cutoff  # [B,H,C,N], bool

        mask0 = paddle.zeros_like(x, dtype="int32")
        mask_int = paddle.put_along_axis(
            mask0, sorted_idx, keep.astype("int32"), axis=-1
        )
        mask = mask_int.astype("bool")

        mask = paddle.logical_and(mask, total_sum > 0)
        return mask

    def _verify_topp(
        self,
        out_kernel,
        topp_mask_kernel,
        top_p_value,
    ):
        out_tensor = out_kernel.astype("float32")

        mask_py = self._find_blocks_chunked(out_tensor, threshold=top_p_value)

        mask_ker_bool = topp_mask_kernel.astype("bool")
        values_kernel = paddle.masked_select(out_tensor, mask_ker_bool)

        mask_py_bool = mask_py.astype("bool")
        values_py = paddle.masked_select(out_tensor, mask_py_bool)

        count_ker = values_kernel.shape[0]
        count_py = values_py.shape[0]
        # print(f"Selected Block Count - Kernel: {count_ker}, Python: {count_py}")

        assert count_ker == count_py, (
            f"Selection count mismatch! Diff: {abs(count_ker - count_py)}"
        )
        # if count_ker != count_py:
        #     print(f"Warning: Selection count mismatch! Diff: {abs(count_ker - count_py)}")
        #     assert False

        sum_ker = paddle.sum(values_kernel).item()
        sum_py = paddle.sum(values_py).item()
        sum_diff = abs(sum_ker - sum_py)

        # print(f"Selected Mass Sum    - Kernel: {sum_ker:.6f}, Python: {sum_py:.6f}")
        # print(f"Mass Diff: {sum_diff:.8f}")

        # if count_ker == count_py:
        if sum_ker == 0:
            return
        val_k_sorted = paddle.sort(values_kernel, descending=True)
        val_p_sorted = paddle.sort(values_py, descending=True)

        max_val_diff = paddle.max(
            paddle.abs(val_k_sorted - val_p_sorted)
        ).item()
        # print(f"Max Diff in Sorted Values: {max_val_diff:.8f}")

        assert max_val_diff < 1e-3, f"Values mismatch! Max diff: {max_val_diff}"
        # print("Value sets match perfectly (Sorted check passed).")

        # else:
        #     print("Counts differ, skipping sorted element-wise check.")
        #     assert sum_diff < 1e-2, f"Mass diff too high: {sum_diff}"

    def _flashmask_to_densemask(
        self, startend_row_indices, seqlen_q, nheads, causal=True
    ):
        if startend_row_indices is None:
            return None
        bz, num_head, seqlen_k, bound_num = startend_row_indices.shape
        assert nheads % num_head == 0
        m = paddle.ones((bz, num_head, seqlen_q, seqlen_k), dtype=paddle.int32)
        has_end = (causal and bound_num == 2) or (
            (not causal) and bound_num == 4
        )
        for bi in range(bz):
            for hi in range(num_head):
                for j in range(seqlen_k):
                    downstart = startend_row_indices[bi, hi, j, 0]
                    if has_end:
                        downend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, downstart:downend, j] = 0
                    else:
                        m[bi, hi, downstart:, j] = 0
                    if causal:
                        # from flash-attention 2.1 and in flash-attention 3, If seqlen_q != seqlen_k and causal=True,
                        # the causal mask is aligned to the bottom right corner of the attention matrix,
                        # instead of the top-left corner.
                        # See: https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#21-change-behavior-of-causal-flag
                        m[bi, hi, : max(0, j - (seqlen_k - seqlen_q)), j] = 0
                    else:
                        if has_end:
                            upstart = startend_row_indices[bi, hi, j, 2]
                            upend = startend_row_indices[bi, hi, j, 3]
                            m[bi, hi, upstart:upend, j] = 0
                        else:
                            upend = startend_row_indices[bi, hi, j, 1]
                            m[bi, hi, :upend, j] = 0
        m = paddle.repeat_interleave(x=m, repeats=nheads // num_head, axis=1)
        m = m.astype(paddle.bool)
        return m

    def _generate_sliding_window_mask(
        self, batch_size, seqlen_q, seqlen_k, h, window_size=None
    ):
        if window_size is None:
            window_size = 1024
            if seqlen_k != 8192:
                window_size = int(window_size * (seqlen_k / 8192))
                print(f"{seqlen_k=}, auto setting window_size to {window_size}")

        startend_row_indices = paddle.arange(
            window_size, seqlen_k + window_size, dtype="int32"
        ).reshape((1, 1, seqlen_k, 1))
        startend_row_indices = paddle.clip(
            startend_row_indices, max=seqlen_q
        ).repeat_interleave(batch_size, 0)

        causal = True
        return startend_row_indices, causal

    def _generate_causal_document_mask(
        self, batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None
    ):
        # TODO: this seems buggy, to be fixed
        if doc_seqlens is None:
            doc_seqlens = [2538, 1742, 3213]
            if seqlen_k != 8192:
                doc_seqlens = [
                    int(doc_seqlen * (seqlen_k / 8192))
                    for doc_seqlen in doc_seqlens
                ]
                print(f"{seqlen_k=}, auto setting doc_seqlens to {doc_seqlens}")
        total_seqlen = np.sum(doc_seqlens)
        assert total_seqlen <= seqlen_k
        assert len(doc_seqlens) >= 3
        padding = seqlen_k - np.sum(doc_seqlens)
        doc_seqlens[-1] += padding
        seq_cusums = np.cumsum(doc_seqlens)

        startend_row_indices = np.repeat(seq_cusums, doc_seqlens)
        startend_row_indices = (
            paddle.to_tensor(startend_row_indices, dtype=paddle.int32)
            .reshape((1, 1, seqlen_k, 1))
            .repeat_interleave(batch_size, 0)
        )
        startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)

        causal = True
        return startend_row_indices, causal

    def _generate_document_mask(
        self, batch_size, seqlen_q, seqlen_k, h, doc_seqlens=None
    ):
        # TODO: this seems buggy, to be fixed
        if doc_seqlens is None:
            doc_seqlens = [2538, 1742, 3213]
            if seqlen_k != 8192:
                doc_seqlens = [
                    int(doc_seqlen * (seqlen_k / 8192))
                    for doc_seqlen in doc_seqlens
                ]
                print(f"{seqlen_k=}, auto setting doc_seqlens to {doc_seqlens}")
        total_seqlen = np.sum(doc_seqlens)
        assert total_seqlen <= seqlen_k
        assert len(doc_seqlens) >= 3
        padding = seqlen_k - np.sum(doc_seqlens)

        down_left_row_indices = []
        up_right_row_indices = []

        cur_len_so_far = doc_seqlens[0]
        for i in range(len(doc_seqlens)):
            down_left_row_indices.extend([cur_len_so_far] * doc_seqlens[i])
            if i < len(doc_seqlens) - 1:
                cur_len_so_far += doc_seqlens[i + 1]
        if padding > 0:
            down_left_row_indices.extend([cur_len_so_far] * padding)

        cur_len_so_far = 0
        for i in range(len(doc_seqlens)):
            up_right_row_indices.extend([cur_len_so_far] * doc_seqlens[i])
            if i < len(doc_seqlens) - 1:
                cur_len_so_far += doc_seqlens[i + 1]
        if padding > 0:
            up_right_row_indices.extend([cur_len_so_far] * padding)

        down_left_row_indices = (
            paddle.to_tensor(down_left_row_indices, dtype=paddle.int32)
            .reshape((1, 1, seqlen_k, 1))
            .repeat_interleave(batch_size, 0)
        )
        up_right_row_indices = (
            paddle.to_tensor(up_right_row_indices, dtype=paddle.int32)
            .reshape((1, 1, seqlen_k, 1))
            .repeat_interleave(batch_size, 0)
        )
        startend_row_indices = paddle.concat(
            [down_left_row_indices, up_right_row_indices], axis=-1
        )
        startend_row_indices = paddle.clip(startend_row_indices, max=seqlen_q)

        causal = False
        return startend_row_indices, causal

    def _generate_shapes(self):
        for (
            batch_size,
            seqlen_q,
            seqlen_k,
            nheads,
            nheads_kv,
        ) in self.shape_cases:
            nheads_startend_row_indices_values = [1, nheads_kv]
            for (
                nheads_startend_row_indices
            ) in nheads_startend_row_indices_values:
                yield (
                    batch_size,
                    seqlen_q,
                    seqlen_k,
                    nheads,
                    nheads_kv,
                    nheads_startend_row_indices,
                )

    def _run_test_suite(self, dtype):
        mask_generators = [
            self._generate_sliding_window_mask,
            self._generate_causal_document_mask,
            self._generate_document_mask,
        ]

        for shape_params in self._generate_shapes():
            (
                batch_size,
                seqlen_q,
                seqlen_k,
                nheads,
                nheads_kv,
                nheads_startend_row_indices,
            ) = shape_params

            for gen_func in mask_generators:
                for stride in self.stride:
                    for dim in self.dim:
                        for threshold in self.threshold:
                            print(
                                f"\n[Run Test] dtype={dtype}, gen={gen_func.__name__}, "
                                f"stride={stride}, dim={dim}, threshold={threshold}"
                            )

                            self._run_rr_attn_estimate_test(
                                batch_size=batch_size,
                                seqlen_q=seqlen_q,
                                seqlen_k=seqlen_k,
                                nheads=nheads,
                                nheads_kv=nheads_kv,
                                nheads_startend_row_indices=nheads_startend_row_indices,
                                dtype=dtype,
                                gen_startend_row_indices=gen_func,
                                stride=stride,
                                dim=dim,
                                threshold=threshold,
                            )

    def test_rrattn_estimate_bf16(self):
        if core.is_bfloat16_supported(base.CUDAPlace(0)):
            self._run_test_suite("bfloat16")
