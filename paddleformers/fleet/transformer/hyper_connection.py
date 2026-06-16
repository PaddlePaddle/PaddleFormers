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
Manifold-Constrained Hyper-Connections (mHC) module.

Implements the mHC propagation:
    x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

Reference: mHC paper - Manifold-Constrained Hyper-Connections for transformers.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn

from paddleformers.fleet.tensor_parallel.random import get_cuda_rng_tracker
from paddleformers.fleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddleformers.fleet.transformer.transformer_config import (
        TransformerConfig,
    )


_ACCURACY_COMPATIBLE_KERNEL: bool = (
    os.environ.get("FLAGS_use_accuracy_compatible_kernel", "0") == "1"
)


def _use_accuracy_compatible_kernel() -> bool:
    """Switch for Megatron-aligned (accuracy-compatible) numeric paths.

    Controlled by the ``FLAGS_use_accuracy_compatible_kernel`` env variable.
    """
    return _ACCURACY_COMPATIBLE_KERNEL


class SinkhornKnopp(paddle.autograd.PyLayer):
    """
    Differentiable Sinkhorn-Knopp algorithm for doubly stochastic projection.

    Projects a positive matrix onto the Birkhoff polytope (doubly stochastic matrices)
    via iterative row and column normalization.

    Reference: Eq. (9) in mHC paper - M^{(t)} = T_c(T_r(M^{(t-1)}))
    """

    @staticmethod
    def _sinkhorn_normalize(
        M: Tensor, num_iterations: int, eps: float = 1e-6
    ) -> Tensor:
        """
        Apply Sinkhorn-Knopp normalization iterations.

        Args:
            M: [..., n, n] - positive matrix to normalize
            num_iterations: Number of Sinkhorn iterations
            eps: Small constant for numerical stability

        Returns:
            M: [..., n, n] - doubly stochastic matrix
        """
        for _ in range(num_iterations):
            # T_r: Row normalization
            M = M / M.sum(axis=-1, keepdim=True).clip(min=eps)
            # T_c: Column normalization
            M = M / M.sum(axis=-2, keepdim=True).clip(min=eps)
        return M

    @staticmethod
    def forward(
        ctx, H_res_logits: Tensor, num_iterations: int, eps: float = 1e-6
    ) -> Tensor:
        """
        Project to doubly stochastic matrix via iterative row/col normalization.

        Args:
            H_res_logits: [..., n, n] - raw logits for residual mixing matrix
            num_iterations: Number of Sinkhorn iterations (paper uses 20)
            eps: Small constant for numerical stability

        Returns:
            H_res: [..., n, n] - doubly stochastic matrix
        """
        # Stabilized exp: subtract row-wise max to prevent overflow.
        # Under FLAGS_use_accuracy_compatible_kernel, force this outside AMP
        # so the Paddle native path preserves the BF16 contract used by
        # Megatron's torch implementation.
        if _use_accuracy_compatible_kernel():
            with paddle.amp.auto_cast(enable=False):
                M_init = paddle.exp(
                    H_res_logits - H_res_logits.max(axis=-1, keepdim=True)
                )
                M = SinkhornKnopp._sinkhorn_normalize(
                    M_init, num_iterations, eps
                )
        else:
            M_init = paddle.exp(
                H_res_logits - H_res_logits.max(axis=-1, keepdim=True)
            )
            M = SinkhornKnopp._sinkhorn_normalize(M_init, num_iterations, eps)

        # Save initial M for backward recomputation
        ctx.save_for_backward(M_init)
        ctx.num_iterations = num_iterations
        ctx.eps = eps
        return M

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
        """
        Backward through Sinkhorn-Knopp iterations using recomputation.
        """
        (M_init,) = ctx.saved_tensor()
        num_iterations = ctx.num_iterations
        eps = ctx.eps

        with paddle.enable_grad():
            # Recompute forward with autograd enabled
            M_input = M_init.detach()
            M_input.stop_gradient = False

            M_current = SinkhornKnopp._sinkhorn_normalize(
                M_input, num_iterations, eps
            )

            # Compute dL/dM_input via autograd
            grad_M_init = paddle.grad(
                outputs=[M_current],
                inputs=[M_input],
                grad_outputs=[grad_output],
                create_graph=False,
            )[0]

        # Apply chain rule: dL/dH = dL/dM_init * dM_init/dH = dL/dM_init * M_init
        # Since M_init = exp(H_res_logits), d(exp(x))/dx = exp(x) = M_init
        grad_input = grad_M_init * M_init

        return grad_input


def native_sinkhorn(
    input_logits: Tensor, num_iterations: int, eps: float = 1e-6
) -> Tensor:
    """Native Sinkhorn-Knopp (PyLayer wrapper)."""
    return SinkhornKnopp.apply(input_logits, num_iterations, eps)


def native_proj_rms(
    x: Tensor, weight: Tensor, eps: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """Native fused projection + RMS normalization."""
    nC = x.shape[-1]
    r = x.norm(axis=-1, keepdim=True) / math.sqrt(nC)
    r = 1.0 / (r + eps)
    proj = paddle.matmul(x, weight)
    return proj, r


def native_h_aggregate(x_streams: Tensor, h_pre: Tensor) -> Tensor:
    """Native n-stream weighted aggregation: out = sum_j(h_pre_j * x_j)."""
    return (x_streams * h_pre.unsqueeze(-1)).sum(axis=-2)


def native_h_post_bda(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Tensor | None,
) -> Tensor:
    """Native H_res.T @ residual + H_post * (x [+ bias]).

    Args:
        h_res: [..., n, n] - residual mixing matrix
        original_residual: [..., n, C] - n-stream hidden states
        h_post: [..., n] - expansion weights
        x: [..., C] - layer output
        bias: [C] or None

    Returns:
        output: [..., n, C]
    """
    leading_shape = original_residual.shape[:-2]
    n, C = original_residual.shape[-2], original_residual.shape[-1]
    num_tokens = math.prod(leading_shape)

    h_res_batched = h_res.reshape([num_tokens, n, n]).transpose([0, 2, 1])
    residual_batched = original_residual.reshape([num_tokens, n, C])
    mixed = paddle.bmm(h_res_batched, residual_batched).reshape(
        [*leading_shape, n, C]
    )

    x_expanded = h_post.unsqueeze(-1) * x.unsqueeze(-2)  # [..., n, C]
    if bias is not None:
        bias = bias.reshape([1] * len(leading_shape) + [1, C])
        bias_expanded = h_post.unsqueeze(-1) * bias
        return x_expanded + bias_expanded + mixed
    return x_expanded + mixed


class HyperConnectionModule(nn.Layer):
    """
    Unified mHC (Manifold-Constrained Hyper-Connections) module.

    Implements the complete mHC propagation:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

    This module handles:
    1. Computing learnable mappings: H_pre, H_post, H_res (with Sinkhorn-Knopp projection)
    2. Aggregation: n-stream → 1-stream (H_pre @ x)
    3. Expansion: 1-stream → n-stream (H_post^T @ output)
    4. Residual merge: H_res @ x + expanded_output
    5. Block-level expand/contract for TransformerBlock boundaries

    Args:
        config: TransformerConfig with hyper-connection fields
        layer_number: Current layer index for initialization
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations

        # Projection weights for dynamic mappings
        # Input: [..., n*C] -> Output: n^2 + 2n values per token
        # - H_pre: n values
        # - H_post: n values
        # - H_res: n^2 values (before Sinkhorn projection)
        self.mapping_proj = nn.Linear(
            self.n * self.hidden_size,
            self.n * self.n + 2 * self.n,
            bias_attr=False,
        )

        init_alpha = config.mhc_init_gating_factor
        # Learnable scaling factors (Eq. 5 in paper)
        self.alpha_pre = self.create_parameter(
            shape=[1],
            dtype=self.config.params_dtype,
            default_initializer=nn.initializer.Constant(init_alpha),
        )
        self.alpha_post = self.create_parameter(
            shape=[1],
            dtype=self.config.params_dtype,
            default_initializer=nn.initializer.Constant(init_alpha),
        )
        self.alpha_res = self.create_parameter(
            shape=[1],
            dtype=self.config.params_dtype,
            default_initializer=nn.initializer.Constant(init_alpha),
        )

        # Static bias terms
        self.bias = self.create_parameter(
            shape=[self.n * self.n + 2 * self.n],
            default_initializer=nn.initializer.Constant(0.0),
        )

        self.norm_eps = 1e-6

        # Choose implementation: fused kernels vs native reference.
        if config.use_fused_mhc:
            from paddleformers.fleet.fusions.fused_mhc_kernels import (
                fused_h_aggregate,
                fused_h_post_bda,
                fused_proj_rms,
                fused_sinkhorn,
            )

            self._sinkhorn_op = fused_sinkhorn
            self._h_aggregate_op = fused_h_aggregate
            self._h_post_bda_op = fused_h_post_bda
            self._proj_rms_op = fused_proj_rms
        else:
            self._sinkhorn_op = native_sinkhorn
            self._h_aggregate_op = native_h_aggregate
            self._h_post_bda_op = native_h_post_bda
            self._proj_rms_op = native_proj_rms

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        # Xavier uniform for mapping projection.
        # Use the model-parallel RNG tracker so that the initialization is
        # controlled by PaddleFleet's RNG state (seeded once at model init)
        # rather than the per-layer pipeline seed (base_seed + layer_index).
        # This prevents layer_index shifts (e.g. from MTPEmbeddingLayer insertion
        # in magic_send mode) from changing the weights.
        if paddle.distributed.get_world_size() <= 1:
            nn.initializer.XavierUniform()(self.mapping_proj.weight)
        else:
            with get_cuda_rng_tracker().fork():
                nn.initializer.XavierUniform()(self.mapping_proj.weight)

        # Set sequence_parallel attribute on parameters for gradient synchronization
        if self.config.sequence_parallel:
            self.mapping_proj.weight.is_distributed = False
            self.alpha_pre.is_distributed = False
            self.alpha_post.is_distributed = False
            self.alpha_res.is_distributed = False
            self.bias.is_distributed = False

    def _projection_and_get_norm(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Project input hidden states to mapping space and apply RMS normalization.

        Args:
            x: [..., n*C] - n-stream hidden states
        """
        if _use_accuracy_compatible_kernel():
            nC = x.shape[-1]
            weight = self.mapping_proj.weight
            r = x.norm(axis=-1, keepdim=True) / math.sqrt(nC)  # [..., 1]
            r = (1.0 / (r + self.norm_eps)).astype(x.dtype)  # [..., 1]
            # Match Megatron clean path: torch.matmul(x, weight.t()). Paddle
            # nn.Linear uses a different BF16 cuBLAS path for this shape and
            # drifts before the first HC BDA.
            x_2d = x.reshape([-1, nC])
            weight_out_in = weight.t().contiguous()
            proj_2d = paddle.matmul(x_2d, weight_out_in, transpose_y=True)
            proj = proj_2d.reshape([*x.shape[:-1], weight.shape[-1]])
        else:
            ori_dtype = x.dtype
            proj, r = self._proj_rms_op(
                x, self.mapping_proj.weight.astype(ori_dtype), self.norm_eps
            )
            if not self.config.high_precision_mhc:
                r = r.astype(ori_dtype)

        return proj, r

    def _compute_h(
        self, proj: Tensor, r: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute h from projected hidden states and scaling factors.

        Args:
            proj: [..., n^2 + 2n] - projected hidden states
            r: [..., 1] - scaling factors

        Returns:
            h_pre: [..., n] - aggregation weights
            h_post: [..., n] - expansion weights
            h_res: [..., n^2] - residual mixing logits
        """
        alpha_ = paddle.concat(
            [
                self.alpha_pre.expand([self.n]),
                self.alpha_post.expand([self.n]),
                self.alpha_res.expand([self.n * self.n]),
            ],
            axis=-1,
        )
        h = r * proj * alpha_ + self.bias
        # H_pre = σ(α_pre * (θ_pre @ x̃) + b_pre)
        h_pre = h[..., : self.n].sigmoid()  # [..., n]
        # H_post = 2σ(α_post * (θ_post @ x̃) + b_post)
        h_post = h[..., self.n : 2 * self.n].sigmoid() * 2  # [..., n]
        h_res = h[..., 2 * self.n :]
        if _use_accuracy_compatible_kernel():
            h_pre = h_pre.astype(proj.dtype)
            h_post = h_post.astype(proj.dtype)
        return h_pre, h_post, h_res

    def compute_mappings(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute mHC mappings from input hidden states.

        Reference: Eq. (5) and (8) in mHC paper

        Args:
            x: [..., n*C] - n-stream hidden states

        Returns:
            h_pre: [..., n] - aggregation weights (sigmoid activated)
            h_post: [..., n] - expansion weights (2*sigmoid activated)
            h_res: [..., n, n] - residual mixing matrix (doubly stochastic)
        """
        leading_shape = x.shape[:-1]
        proj, r = self._projection_and_get_norm(x)
        h_pre, h_post, h_res = self._compute_h(proj, r)
        h_res = self._sinkhorn_op(
            h_res.reshape([*leading_shape, self.n, self.n]),
            self.sinkhorn_iterations,
            self.norm_eps,
        )  # [..., n, n]

        return h_pre, h_post, h_res

    def aggregate(self, x: Tensor, h_pre: Tensor) -> Tensor:
        """
        Aggregate n-stream to 1-stream using H_pre weights.

        Computes: sum_i(h_pre_i * x_stream_i)

        Args:
            x: [..., n*C] - n-stream hidden states
            h_pre: [..., n] - aggregation weights

        Returns:
            aggregated: [..., C] - single stream hidden states
        """
        leading_shape = x.shape[:-1]
        C = self.hidden_size

        # Reshape to [..., n, C]
        x_streams = x.reshape([*leading_shape, self.n, C])

        if _use_accuracy_compatible_kernel():
            # Weighted sum: [..., n, C] * [..., n, 1] -> sum over n -> [..., C]
            aggregated = (x_streams * h_pre.unsqueeze(-1)).sum(axis=-2)
            if aggregated.dtype != x.dtype:
                aggregated = aggregated.astype(x.dtype)
            return aggregated
        else:
            aggregated = self._h_aggregate_op(x_streams, h_pre)
            if aggregated.dtype != x.dtype:
                aggregated = aggregated.astype(x.dtype)
            return aggregated

    def apply_h_res(self, h_res: Tensor, residual: Tensor) -> Tensor:
        """
        Apply H_res to residual using H_res weights.

        Computes: H_res.T @ residual

        Args:
            h_res: [..., n, n] - residual mixing matrix
            residual: [..., n*C] - n-stream hidden states
        """
        leading_shape = residual.shape[:-1]
        n = self.n
        C = self.hidden_size
        num_tokens = math.prod(leading_shape)

        if _use_accuracy_compatible_kernel():
            # Megatron clean path applies H_res.T to residual.
            ndim = h_res.ndim
            perm = [*list(range(ndim - 2)), ndim - 1, ndim - 2]
            h_res_batched = (
                h_res.astype(residual.dtype)
                .transpose(perm)
                .reshape([num_tokens, n, n])
            )
        else:
            # Reshape for bmm: [..., n, n] -> [batch, n, n]
            ndim = h_res.ndim
            perm = [*list(range(ndim - 2)), ndim - 1, ndim - 2]
            h_res_batched = h_res.transpose(perm).reshape([num_tokens, n, n])
        # [..., n*C] -> [..., n, C] -> [batch, n, C]
        residual_batched = residual.reshape([num_tokens, n, C])

        # Batch matrix multiply: [batch, n, n] @ [batch, n, C] -> [batch, n, C]
        mixed = paddle.bmm(h_res_batched, residual_batched)

        return mixed.reshape([*leading_shape, n * C])

    def _apply_h_post(self, x: Tensor, h_post: Tensor) -> Tensor:
        """
        Core implementation of H_post application to a single tensor.

        Computes: H_post^T @ x

        Args:
            x: Input tensor, can be either:
               - [..., C] - standard hidden states
               - [C] - bias tensor (will be broadcast)
            h_post: [..., n] - expansion weights

        Returns:
            output: [..., n*C] - expanded tensor
        """
        n = self.n
        leading_shape = h_post.shape[:-1]

        if x.dim() == 1:
            # x is bias with shape [C], broadcast to [..., 1, C]
            C = x.shape[0]
            x_expanded = x.reshape([1] * len(leading_shape) + [1, C])
            x_expanded = x_expanded.expand([*leading_shape, 1, C])
        else:
            # x is [..., C]
            C = x.shape[-1]
            x_expanded = x.unsqueeze(-2)  # [..., 1, C]

        # h_post^T @ x : [..., n, 1] * [..., 1, C] -> [..., n, C]
        result = h_post.unsqueeze(-1) * x_expanded
        return result.reshape([*leading_shape, n * C])

    def apply_h_post(
        self,
        x_with_bias: tuple[Tensor, Tensor | None],
        h_post: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Apply H_post to x and optionally bias.

        Args:
            x_with_bias: Tuple of (x, bias) where:
                - x: [..., C] - hidden states
                - bias: [C] or None - optional bias tensor
            h_post: [..., n] - expansion weights

        Returns:
            Tuple of (x_out, bias_out) where:
                - x_out: [..., n*C] - expanded hidden states
                - bias_out: [..., n*C] or None
        """
        x, bias = x_with_bias
        x_out = self._apply_h_post(x, h_post)
        bias_out = (
            self._apply_h_post(bias, h_post) if bias is not None else None
        )
        return x_out, bias_out

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Full mHC forward pass.

        Args:
            hidden_states: [..., n*C] - n-stream hidden states

        Returns:
            aggregated: [..., C] - aggregated input for layer computation
            h_res: [..., n, n] - residual mixing matrix (for fused kernel)
            h_post: [..., n] - expansion weights
        """
        with paddle.amp.auto_cast(enable=False):
            # Compute mappings
            if (
                not _use_accuracy_compatible_kernel()
                and self.config.high_precision_mhc
            ):
                hidden_states = hidden_states.astype("float32")
            h_pre, h_post, h_res = self.compute_mappings(hidden_states)

            # Aggregate for layer input
            aggregated = self.aggregate(hidden_states, h_pre)

        return aggregated, h_res, h_post

    # ==================== Block-level utilities ====================

    @staticmethod
    def input_expand(x: Tensor, n: int) -> Tensor:
        """
        Expand 1-stream to n-stream at TransformerBlock entry.

        Simple replication strategy: each stream initialized as a copy of input.

        Args:
            x: [..., C] - single stream hidden states
            n: Number of residual streams

        Returns:
            expanded: [..., n*C] - n-stream hidden states
        """
        leading_shape = x.shape[:-1]
        C = x.shape[-1]
        # Replicate input to n streams: [..., C] -> [..., n, C] -> [..., n*C]
        expanded = x.unsqueeze(-2).expand([*leading_shape, n, C])
        return expanded.reshape([*leading_shape, n * C])

    @staticmethod
    def output_contract(x: Tensor, n: int) -> Tensor:
        """
        Contract n-stream to 1-stream at TransformerBlock exit.

        Simple averaging strategy: average all streams.

        Args:
            x: [..., n*C] - n-stream hidden states
            n: Number of residual streams

        Returns:
            contracted: [..., C] - single stream hidden states
        """
        leading_shape = x.shape[:-1]
        nC = x.shape[-1]
        C = nC // n
        # Average all streams: [..., n*C] -> [..., n, C] -> mean -> [..., C]
        x_streams = x.reshape([*leading_shape, n, C])
        contracted = x_streams.mean(axis=-2)
        return contracted

    # ==================== Learned output contraction ====================

    @staticmethod
    def learned_output_contract(
        hidden_states: Tensor,
        head_fn: Tensor,
        base: Tensor,
        scale: Tensor,
        n: int,
        eps: float,
    ) -> Tensor:
        """Learned output contraction: n-stream → 1-stream via sigmoid-gated weighted sum.

        DSv4-style contraction using learnable parameters for gating.

        Args:
            hidden_states: [..., n*h] multi-stream hidden states
            head_fn: [n, n*h] learnable weight for gating
            base: [n] sigmoid bias
            scale: [1] scaling factor
            n: number of residual streams
            eps: epsilon for numerical stability

        Returns:
            contracted: [..., h] single-stream output
        """
        dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        head_fn = head_fn.astype("float32")
        base = base.astype("float32")
        scale = scale.astype("float32")

        rsqrt = paddle.rsqrt(
            hidden_states.square().mean(-1, keepdim=True) + eps
        )
        if _use_accuracy_compatible_kernel():
            # Match Torch F.linear(x, weight[out,in]) kernel selection. Paddle
            # F.linear(x, weight[in,out]) uses a different cuBLAS path and
            # causes BF16 ulp drift in DSv4 final output contraction.
            head_fn_out_in = head_fn.transpose([1, 0]).contiguous()
            with paddle.amp.auto_cast(False):
                proj = paddle.matmul(
                    hidden_states, head_fn_out_in, transpose_y=True
                )
            mixes = proj * rsqrt
        else:
            mixes = F.linear(hidden_states, head_fn) * rsqrt
        pre = F.sigmoid(mixes * scale + base) + eps
        y = paddle.sum(
            pre.unsqueeze(-1)
            * hidden_states.reshape([*hidden_states.shape[:-1], n, -1]),
            axis=-2,
        )
        return y.astype(dtype)

    # ==================== Fused kernel placeholder ====================

    def fused_h_res_h_post_bda(
        self,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias: tuple[Tensor, Tensor | None],
        dropout_prob: float,
        training: bool,
        fused: bool,
    ) -> Tensor:
        """
        Fused kernel combining apply_h_res, apply_h_post and bias-dropout-add.

        Currently implements the operations sequentially using native PaddlePaddle.

        The computation flow is:
            1. mixed = H_res^T @ original_residual (apply_h_res)
            2. expanded = H_post^T @ layer_output (apply_h_post)
            3. output = dropout(expanded + bias) + mixed (bias-dropout-add)

        Args:
            h_res: [..., n, n] - residual mixing matrix
            original_residual: [..., n*C] - n-stream hidden states
            h_post: [..., n] - expansion weights
            layer_output_with_bias: Tuple of (x, bias) where:
                - x: [..., C] - layer output (attention or MLP output)
                - bias: [C] or None - optional bias tensor
            dropout_prob: Dropout probability
            training: Whether in training mode
            fused: Whether to use fused BDA implementation (unused, kept for API compat)

        Returns:
            output: [..., n*C] - final output after all operations
        """
        with paddle.amp.auto_cast(enable=False):
            x, bias = layer_output_with_bias

            # Fast path: no dropout — use fused/native h_post_bda kernel
            if not _use_accuracy_compatible_kernel() and (
                dropout_prob == 0.0 or not training
            ):
                leading_shape = original_residual.shape[:-1]
                n = self.n
                C = self.hidden_size
                orig_reshaped = original_residual.reshape(
                    [*leading_shape, n, C]
                )
                if self.config.high_precision_mhc:
                    orig_reshaped = orig_reshaped.astype("float32")
                    x = x.astype("float32")
                    if bias is not None:
                        bias = bias.astype("float32")
                output = self._h_post_bda_op(
                    h_res, orig_reshaped, h_post, x, bias
                )
                return output.reshape([*leading_shape, n * C])

            # Sequential path: used when dropout required OR accuracy-compatible kernel is NOT enabled
            mixed = self.apply_h_res(h_res, original_residual)

            x_expanded = self._apply_h_post(x, h_post)
            bias_expanded = (
                self._apply_h_post(bias, h_post) if bias is not None else None
            )

            if bias_expanded is not None:
                x_expanded = x_expanded + bias_expanded
            out = paddle.nn.functional.dropout(
                x_expanded, p=dropout_prob, training=training
            )
            output = out + mixed

        return output


# ==================== Pipeline-compatible expand/contract layers ====================


class HyperConnectionExpandLayer(FleetLayer):
    """Pipeline-compatible layer that expands 1-stream to n-streams.

    Inserted before the first HyperConnectionTransformerLayer in the flat
    LayerDesc list of GPTModel. Receives and returns dict_args.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.n = config.num_residual_streams

    def forward(self, dict_args: dict) -> dict:
        dict_args["hidden_states"] = HyperConnectionModule.input_expand(
            dict_args["hidden_states"], self.n
        )
        return dict_args


class HyperConnectionContractLayer(FleetLayer):
    """Pipeline-compatible layer that contracts n-streams to 1-stream.

    Inserted after the last HyperConnectionTransformerLayer in the flat
    LayerDesc list of GPTModel. Receives and returns dict_args.

    Uses learned output contraction (DSv4 style) unconditionally.
    When MTP is enabled, additionally preserves the pre-contraction multi-stream
    tensor in dict_args["mhc_multistream"] for use by downstream MTP layers.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.n = config.num_residual_streams
        self.mtp_enabled = (
            getattr(config, "num_nextn_predict_layers", 0) > 0
            or getattr(config, "mtp_num_layers", 0) > 0
        )

        self.num_mtp = getattr(config, "num_nextn_predict_layers", 0) or 0
        self.magic_send = getattr(config, "enable_mtp_magic_send", False)

        # Learned contraction parameters (DSv4 style, always used)
        n = self.n
        hc_dim = config.hidden_size * n
        self.hc_head_fn = self.create_parameter(
            shape=[hc_dim, n],
            dtype=self.config.params_dtype,
            default_initializer=nn.initializer.XavierUniform(),
        )
        self.hc_head_base = self.create_parameter(
            shape=[n],
            dtype=self.config.params_dtype,
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.hc_head_scale = self.create_parameter(
            shape=[1],
            dtype=self.config.params_dtype,
            default_initializer=nn.initializer.Constant(1.0),
        )

        if config.sequence_parallel:
            self.hc_head_fn.is_distributed = False
            self.hc_head_base.is_distributed = False
            self.hc_head_scale.is_distributed = False

    def forward(self, dict_args: dict) -> dict:
        hidden_states = dict_args["hidden_states"]

        # When MTP is enabled, preserve multi-stream for MTP input
        if (
            self.mtp_enabled
            and self.num_mtp > 0
            and not getattr(self.config, "mtp_load_weight_only", False)
        ):
            dict_args["mhc_multistream"] = hidden_states

            if self.magic_send:
                # magic_send: hidden_states is pure backbone [B, S, n*H]
                # Magic send: backbone processes only main sequence, no MTP chunks concatenated.
                # Simply contract the entire tensor.
                dict_args["hidden_states"] = (
                    HyperConnectionModule.learned_output_contract(
                        hidden_states,
                        self.hc_head_fn,
                        self.hc_head_base,
                        self.hc_head_scale,
                        self.n,
                        self.config.rms_norm_eps,
                    )
                )
            else:
                # Non-magic_send: backbone output is [backbone_chunk | mtp_chunks...] concatenated.
                # Split, contract main backbone, slice MTP chunks.
                chunks = paddle.split(hidden_states, self.num_mtp + 1)

                # Main backbone: learned contraction [s, b, n*h] -> [s, b, h]
                main_contracted = HyperConnectionModule.learned_output_contract(
                    chunks[0],
                    self.hc_head_fn,
                    self.hc_head_base,
                    self.hc_head_scale,
                    self.n,
                    self.config.rms_norm_eps,
                )

                # 为了后面MTP slice、取shape的时候兼容,原本也是expand过来的[[s,b,h]...]
                mtp_contracted = [
                    c[..., : c.shape[-1] // self.n] for c in chunks[1:]
                ]

                dict_args["hidden_states"] = paddle.concat(
                    [main_contracted, *mtp_contracted]
                )

        else:
            # Learned output contraction: [s, b, n*h] -> [s, b, h]
            dict_args["hidden_states"] = (
                HyperConnectionModule.learned_output_contract(
                    hidden_states,
                    self.hc_head_fn,
                    self.hc_head_base,
                    self.hc_head_scale,
                    self.n,
                    self.config.rms_norm_eps,
                )
            )
        return dict_args
