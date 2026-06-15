# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
Triton kernel implementation of the fused MoE TopK operation.

This module implements the TopK expert selection used in MoE (Mixture of
Experts) models, accelerated on GPU via Triton.

Main features:
- TopK expert selection: pick the k experts with the highest scores.
- Node Limit: group experts and select `topk_group` groups from all groups.
- Probability normalization: normalize the selected experts' probabilities.
- Routing map generation: build routing map and dispatch mask from the
  selected expert indices.
- Support for padding mask and pure-text mask.

"""

import paddle
import triton
import triton.language as tl

from .utils import enable_compat_on_triton_kernel


@enable_compat_on_triton_kernel
@triton.jit
def _fwd_kernel(
    ptr_gate,
    ptr_choice,
    ptr_out_probs,
    ptr_out_idx,
    ptr_out_sum,
    stride_gate_s,
    stride_gate_e,
    stride_choice_s,
    stride_choice_e,
    stride_out_s,
    stride_out_k,
    n_experts,
    moe_k: tl.constexpr,
    use_node_limit: tl.constexpr,
    n_group: tl.constexpr,
    topk_group: tl.constexpr,
    norm_gate_logits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward kernel for fused MoE TopK.

    For each sequence position, select the topk experts. Supports node limit
    and normalization.
    """
    pid = tl.program_id(0)

    # Calculate offset for this sequence row
    row_choice_ptr = ptr_choice + pid * stride_choice_s
    row_gate_ptr = ptr_gate + pid * stride_gate_s

    # Offsets for loading the expert row
    off_e = tl.arange(0, BLOCK_SIZE)
    mask_e = off_e < n_experts

    # Load choice probs into registers; init out of bounds with -inf
    choice_vals = tl.load(
        row_choice_ptr + off_e * stride_choice_e,
        mask=mask_e,
        other=float("-inf"),
    )

    # --- Node Limit Logic ---
    if use_node_limit:
        epg = n_experts // n_group
        selected_groups_mask = 0

        # Iteratively select topk groups using a simplistic approach suitable for small n_group
        for _ in range(topk_group):
            best_g_score = float("-inf")
            best_g_idx = -1

            # Evaluate all groups
            for g in range(n_group):
                # Check if this group index 'g' is already in the mask
                is_set = (selected_groups_mask >> g) & 1
                if is_set == 0:
                    g_start = g * epg

                    # Find top 2 sum in this group
                    m1 = float("-inf")
                    m2 = float("-inf")

                    # Iterate experts in group.
                    # Re-loading from memory here is necessary as Triton doesn't support
                    # dynamic indexing into register tensors (choice_vals).
                    # L1 cache should handle the bandwidth well given the small problem size per row.
                    for i in range(epg):
                        idx = g_start + i
                        if idx < n_experts:
                            val = tl.load(row_choice_ptr + idx * stride_choice_e)
                            if val > m1:
                                m2 = m1
                                m1 = val
                            elif val > m2:
                                m2 = val

                    score = m1 + m2
                    if score > best_g_score:
                        best_g_score = score
                        best_g_idx = g

            # Mark selected
            if best_g_idx != -1:
                selected_groups_mask = selected_groups_mask | (1 << best_g_idx)

        # Apply mask to choice_vals
        choice_group_idx = off_e // epg
        is_selected = (selected_groups_mask >> choice_group_idx) & 1
        choice_vals = tl.where((is_selected & mask_e), choice_vals, float("-inf"))

    # --- Choice TopK Logic ---
    row_out_probs = ptr_out_probs + pid * stride_out_s
    row_out_idx = ptr_out_idx + pid * stride_out_s

    for k_i in range(moe_k):
        # Find max across the block
        k_val = tl.max(choice_vals, axis=0)

        # Identify elements equal to max
        is_max = choice_vals == k_val

        # Determine index (prefer smallest index if tie)
        k_idx_candidates = tl.where(is_max, off_e, n_experts + 1)
        k_idx = tl.min(k_idx_candidates, axis=0)

        # Store index and value
        tl.store(row_out_idx + k_i * stride_out_k, k_idx)

        # Load gate probability for this index (fetching from global as needed)
        gate_val = tl.load(row_gate_ptr + k_idx * stride_gate_e)
        tl.store(row_out_probs + k_i * stride_out_k, gate_val)

        # Mask out this index so we don't pick it again
        choice_vals = tl.where(off_e != k_idx, choice_vals, float("-inf"))

    # --- Normalization ---
    if norm_gate_logits:
        # Sum the collected probs
        total_sum = 0.0
        for k_i in range(moe_k):
            total_sum += tl.load(row_out_probs + k_i * stride_out_k)

        tl.store(ptr_out_sum + pid, total_sum)

        denom = total_sum
        if denom < 1e-12:
            denom = 1e-12

        for k_i in range(moe_k):
            val_ptr = row_out_probs + k_i * stride_out_k
            val = tl.load(val_ptr)
            tl.store(val_ptr, val / denom)


@enable_compat_on_triton_kernel
@triton.jit
def _bwd_kernel(
    grad_out_probs_ptr,
    ind_ptr,
    normed_probs_ptr,
    sum_ptr,
    grad_gate_ptr,
    stride_grad_out_s,
    stride_grad_out_k,
    stride_ind_s,
    stride_ind_k,
    stride_normed_s,
    stride_normed_k,
    stride_grad_gate_s,
    stride_grad_gate_e,
    moe_k: tl.constexpr,
    norm_gate_logits: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,
):
    """
    Backward kernel for fused MoE TopK.

    Computes the gradient with respect to `gate_probs`, supporting the
    normalized-gradient path.
    """
    pid = tl.program_id(0)

    row_grad_out = grad_out_probs_ptr + pid * stride_grad_out_s
    row_ind = ind_ptr + pid * stride_ind_s
    row_normed = normed_probs_ptr + pid * stride_normed_s
    row_grad_gate_base = grad_gate_ptr + pid * stride_grad_gate_s

    # Optimization 1: load all k values at once via a mask.
    offs_k = tl.arange(0, K_BLOCK_SIZE)
    mask_k = offs_k < moe_k

    grad_out_vals = tl.load(row_grad_out + offs_k * stride_grad_out_k, mask=mask_k)
    indices = tl.load(row_ind + offs_k * stride_ind_k, mask=mask_k)

    # Optimization 2: branch on `norm_gate_logits` to avoid redundant work.
    if norm_gate_logits:
        # norm_gate_logits=True path: normalization is required.
        normed_vals = tl.load(row_normed + offs_k * stride_normed_k, mask=mask_k)
        sigma = tl.load(sum_ptr + pid)

        # Precompute inv_denom_masked = inv_denom * grad_sigma_mask.
        denom = tl.maximum(sigma, 1e-12)
        inv_denom = 1.0 / denom
        inv_denom_masked = tl.where(sigma > 1e-12, inv_denom, 0.0)

        # Vectorized computation of dot_prod and gradient.
        dot_prod = tl.sum(grad_out_vals * normed_vals)
        grad_vals = grad_out_vals * inv_denom - dot_prod * inv_denom_masked
    else:
        # norm_gate_logits=False path: use grad_out_vals directly, no extra work.
        grad_vals = grad_out_vals

    tl.store(
        row_grad_gate_base + indices * stride_grad_gate_e,
        grad_vals,
        mask=mask_k,
    )


class MoETopkFusion(paddle.autograd.PyLayer):
    """
    Fused MoE TopK operation accelerated with Triton.

    Supports node limit, grouped selection and probability normalization.
    """

    @staticmethod
    def forward(
        ctx,
        gate_probs,
        probs_for_choice,
        moe_k,
        use_node_limit,
        n_group,
        topk_group,
        norm_gate_logits,
    ):
        """
        Forward pass: select topk experts.

        Args:
            gate_probs: raw gate probabilities, shape [seq_len, n_experts].
            probs_for_choice: probabilities used for expert selection (may
                include correction bias), shape [seq_len, n_experts].
            moe_k: number of experts selected per token.
            use_node_limit: whether to apply the node limit.
            n_group: number of expert groups.
            topk_group: number of selected topk groups.
            norm_gate_logits: whether to normalize gate logits.

        Returns:
            topk_probs: normalized topk probabilities, shape [seq_len, moe_k].
            topk_indices: topk expert indices, shape [seq_len, moe_k].
        """
        seq_len, n_experts = gate_probs.shape

        topk_indices = paddle.empty((seq_len, moe_k), dtype="int32")
        topk_probs = paddle.empty((seq_len, moe_k), dtype=gate_probs.dtype)
        topk_sum = paddle.empty((seq_len,), dtype="float32") if norm_gate_logits else None

        # Block size must cover n_experts for the single-block reduction logic
        BLOCK_SIZE = triton.next_power_of_2(n_experts)
        if BLOCK_SIZE < 32:
            BLOCK_SIZE = 32

        # Use topk_probs as dummy pointer for sum if not needed, as it is writable
        ptr_sum_arg = topk_sum if norm_gate_logits else topk_probs

        _fwd_kernel[(seq_len,)](
            gate_probs,
            probs_for_choice,
            topk_probs,
            topk_indices,
            ptr_sum_arg,
            int(gate_probs.stride(0)),
            int(gate_probs.stride(1)),
            int(probs_for_choice.stride(0)),
            int(probs_for_choice.stride(1)),
            int(topk_probs.stride(0)),
            int(topk_probs.stride(1)),
            n_experts,
            moe_k,
            use_node_limit,
            n_group if use_node_limit else 1,
            topk_group if use_node_limit else 1,
            norm_gate_logits,
            BLOCK_SIZE,
        )

        ctx.save_for_backward(topk_indices, topk_probs, topk_sum)
        ctx.input_shape = gate_probs.shape
        ctx.norm_gate_logits = norm_gate_logits
        ctx.moe_k = moe_k

        return topk_probs, topk_indices.to(paddle.int64)

    @staticmethod
    def backward(ctx, grad_output_probs, grad_output_indices):
        """
        Backward: compute the gradient with respect to gate_probs.
        """
        topk_indices, topk_normed_probs, topk_sum = ctx.saved_tensor()

        grad_gate_probs = paddle.zeros(ctx.input_shape, dtype=grad_output_probs.dtype)

        # Dummy ptr for sum if not used
        ptr_sum_arg = topk_sum if ctx.norm_gate_logits else grad_output_probs

        K_BLOCK_SIZE = triton.next_power_of_2(ctx.moe_k)

        _bwd_kernel[(ctx.input_shape[0],)](
            grad_output_probs,
            topk_indices,
            topk_normed_probs,
            ptr_sum_arg,
            grad_gate_probs,
            int(grad_output_probs.stride(0)),
            int(grad_output_probs.stride(1)),
            int(topk_indices.stride(0)),
            int(topk_indices.stride(1)),
            int(topk_normed_probs.stride(0)),
            int(topk_normed_probs.stride(1)),
            int(grad_gate_probs.stride(0)),
            int(grad_gate_probs.stride(1)),
            ctx.moe_k,
            ctx.norm_gate_logits,
            K_BLOCK_SIZE,
        )

        return grad_gate_probs, None


@enable_compat_on_triton_kernel
@triton.jit
def _routing_map_fwd_kernel(
    topk_indices_ptr,
    input_ids_ptr,
    is_pure_text_line_ptr,
    routing_map_ptr,
    topk_indices_out_ptr,
    dispatch_mask_ptr,
    stride_topk_s,
    stride_topk_k,
    stride_routing_s,
    stride_routing_e,
    n_experts,
    seq_len,  # explicit seq_len for boundary checks during block processing
    moe_k,  # runtime parameter: the actual moe_k value
    pad_token_id,  # runtime parameter: token id used for padding
    has_input_ids: tl.constexpr,
    has_pure_text_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,  # block size along the sequence dim (e.g., 32, 64)
    BLOCK_N: tl.constexpr,  # block size along the expert dim (e.g., 64, 128)
    BLOCK_K: tl.constexpr,  # block size along the moe_k dim (must be a power of 2)
):
    """
    Forward kernel for routing map generation.

    Builds the routing map and dispatch mask from topk indices, supporting
    padding mask and pure-text mask.
    """
    # -----------------------------------------------------------
    # 1. Coordinate and mask setup
    # -----------------------------------------------------------
    # pid_m handles the sequence dim, pid_n handles the expert dim.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets along the sequence dim.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Boundary mask: avoid processing rows beyond seq_len.
    mask_m = offs_m < seq_len

    # -----------------------------------------------------------
    # 2. Load data (take advantage of coalesced access)
    # -----------------------------------------------------------
    # Load TopK indices: [BLOCK_M, BLOCK_K].
    # Use BLOCK_K as a compile-time constant; handle the real moe_k via mask.
    offs_k = tl.arange(0, BLOCK_K)
    # moe_k-dim mask: only process valid k values.
    mask_k = offs_k < moe_k
    # Compute load addresses: base + row offset + column offset.
    indices_ptrs = topk_indices_ptr + (offs_m[:, None] * stride_topk_s) + (offs_k[None, :] * stride_topk_k)
    # Load indices; fill out-of-bounds with -1, honoring the moe_k boundary.
    indices = tl.load(indices_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=-1)

    # Compute the validity mask (is_valid): [BLOCK_M].
    is_valid = tl.full((BLOCK_M,), 1, dtype=tl.int1)

    if has_input_ids:
        in_ids = tl.load(input_ids_ptr + offs_m, mask=mask_m, other=pad_token_id)
        is_valid = is_valid & (in_ids != pad_token_id)

    if has_pure_text_mask:
        p_mask = tl.load(is_pure_text_line_ptr + offs_m, mask=mask_m, other=0)
        is_valid = is_valid & (p_mask > 0)

    # -----------------------------------------------------------
    # 3. Store topk indices (with mask handling)
    # -----------------------------------------------------------
    # Only the first block along the expert dim (pid_n == 0) performs the
    # write, to avoid duplicated stores.
    if pid_n == 0:
        # Set indices of invalid rows to -1.
        masked_indices = tl.where(is_valid[:, None], indices, -1)
        out_indices_ptrs = topk_indices_out_ptr + (offs_m[:, None] * stride_topk_s) + (offs_k[None, :] * stride_topk_k)
        # The store must honor both seq_len and moe_k boundaries.
        tl.store(
            out_indices_ptrs,
            masked_indices,
            mask=mask_m[:, None] & mask_k[None, :],
        )

    # -----------------------------------------------------------
    # 4. Build the routing map (core optimization)
    # -----------------------------------------------------------
    # Offsets along the expert dim [BLOCK_N].
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_experts

    # Use broadcasting for a parallel comparison, eliminating the inner loop.
    # indices: [BLOCK_M, moe_k] -> [BLOCK_M, moe_k, 1]
    # offs_n : [BLOCK_N]        -> [1,       1,     BLOCK_N]
    # result : [BLOCK_M, moe_k, BLOCK_N] (boolean)

    # Check whether experts in the current block are selected.
    matches = indices[:, :, None] == offs_n[None, None, :]

    # Reduce along the moe_k dim: set to 1 if any k picked this expert.
    # Use max to implement an "any" reduction
    # (max of [0,0,0,1] = 1, max of [0,0,0,0] = 0).
    routing_block = tl.max(matches.to(tl.float32), axis=1)

    # Apply the validity mask: rows marked invalid have routing map all zeros.
    routing_block = tl.where(is_valid[:, None], routing_block, 0.0)

    # -----------------------------------------------------------
    # 5. Write the routing map
    # -----------------------------------------------------------
    # Compute write addresses: [BLOCK_M, BLOCK_N].
    routing_out_ptrs = routing_map_ptr + (offs_m[:, None] * stride_routing_s) + (offs_n[None, :] * stride_routing_e)

    # Combined mask: honor both the sequence and expert boundaries.
    full_store_mask = mask_m[:, None] & mask_n[None, :]

    tl.store(routing_out_ptrs, routing_block, mask=full_store_mask)

    # -----------------------------------------------------------
    # 6. Compute the dispatch mask (sum along the sequence dim)
    # -----------------------------------------------------------
    # Accumulate routing_block values along the expert dim of the current
    # block. Use atomic add to handle multiple blocks writing the same expert.
    dispatch_block = tl.sum(routing_block, axis=0)  # [BLOCK_N]
    # Zero out dispatch_block entries that fall outside bounds.
    dispatch_block = tl.where(mask_n, dispatch_block, 0.0)
    # Cast to int64.
    dispatch_block = dispatch_block.to(tl.int64)
    # Compute target addresses.
    dispatch_ptrs = dispatch_mask_ptr + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Perform a vectorized atomic add.
    # Note: tl.atomic_add supports vectorized pointers and values.
    tl.atomic_add(dispatch_ptrs, dispatch_block, mask=mask_n)


# -----------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------


def routing_map_fusion_forward(
    gate_probs,
    topk_indices,
    input_ids=None,
    is_pure_text_line=None,
    pad_token_id=0,
):
    """
    Get routing_map using Triton kernel.

    Args:
        gate_probs: Gate probabilities [seq_len, n_experts]
        topk_indices: Topk expert indices [seq_len, moe_k]
        input_ids: Input token IDs [seq_len] (optional, for padding mask)
        is_pure_text_line: Pure text line mask [seq_len] (optional)
        pad_token_id: Token ID treated as padding (default 0)

    Returns:
        routing_map: Routing map [seq_len, n_experts]
        topk_indices_out: Topk indices with masking [seq_len, moe_k]
        dispatch_mask: Dispatch mask [n_experts]
    """
    seq_len, moe_k = topk_indices.shape
    n_experts = gate_probs.shape[1]

    # Prepare output tensors.
    routing_map = paddle.zeros((seq_len, n_experts), dtype=paddle.float32)
    topk_indices_out = paddle.empty_like(topk_indices)
    # Initialize dispatch_mask to 0; the kernel accumulates via atomic_add.
    dispatch_mask = paddle.zeros((n_experts,), dtype=paddle.int64)

    # Tuned block sizes.
    # BLOCK_M: number of rows processed per block. 32 or 64 is recommended;
    # larger values improve memory bandwidth but increase register pressure.
    # BLOCK_N: number of experts processed per block. 64 or 128 is recommended.
    # BLOCK_K: number of moe_k entries processed per block. Must be a power
    # of 2; use next_power_of_2 to round up.
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = triton.next_power_of_2(moe_k)

    grid = (triton.cdiv(seq_len, BLOCK_M), triton.cdiv(n_experts, BLOCK_N))

    # Prepare pointer args: Paddle tensors can be passed directly to Triton.
    _routing_map_fwd_kernel[grid](
        topk_indices_ptr=topk_indices,
        input_ids_ptr=input_ids if input_ids is not None else topk_indices,  # placeholder
        is_pure_text_line_ptr=is_pure_text_line if is_pure_text_line is not None else topk_indices,  # placeholder
        routing_map_ptr=routing_map,
        topk_indices_out_ptr=topk_indices_out,
        dispatch_mask_ptr=dispatch_mask,
        stride_topk_s=int(topk_indices.stride(0)),
        stride_topk_k=int(topk_indices.stride(1)),
        stride_routing_s=int(routing_map.stride(0)),
        stride_routing_e=int(routing_map.stride(1)),
        n_experts=n_experts,
        seq_len=seq_len,
        moe_k=moe_k,
        pad_token_id=pad_token_id,
        has_input_ids=input_ids is not None,
        has_pure_text_mask=is_pure_text_line is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return routing_map, topk_indices_out, dispatch_mask
