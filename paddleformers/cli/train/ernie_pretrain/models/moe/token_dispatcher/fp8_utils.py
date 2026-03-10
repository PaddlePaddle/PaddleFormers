# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
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
FP8 Utilities for Mixture of Experts (MoE) Token Dispatcher

This module provides optimized operations for FP8 (8-bit floating point) computations
in Mixture of Experts architectures. Key features include:
- FP8 GEMM (General Matrix Multiply) operations for expert computations
- Specialized forward and backward passes for MoE layers
- Memory-efficient quantization and dequantization routines
- Support for both contiguous and non-contiguous memory layouts

The implementation leverages PaddlePaddle's FP8 incubator operations and provides
additional optimizations specific to MoE workloads.
"""

import numpy
import paddle
from paddle.incubate.fp8 import deep_gemm
from paddle.nn.functional import swiglu

from paddleformers.cli.train.ernie_pretrain.models.fp8_linear import fp8_gemm

__all__ = [
    "ExpertsGroupGemmNode",
    "ExpertsGroupGemmContiguousNode",
]


def _get_fp8_weight_and_scale(weight, stacked=False, transpose=False):
    if stacked:
        if transpose:
            fp8_weight, fp8_scale = (
                weight.fp8_weight_stacked_transpose,
                weight.fp8_scale_stacked_transpose,
            )
        else:
            fp8_weight, fp8_scale = weight.fp8_weight_stacked, weight.fp8_scale_stacked
    else:
        if transpose:
            fp8_weight, fp8_scale = (
                weight.fp8_weight_transpose,
                weight.fp8_scale_transpose,
            )
        else:
            fp8_weight, fp8_scale = weight.fp8_weight, weight.fp8_scale
    return fp8_weight, fp8_scale


def fused_stack_transpose_quant(weight_list, transpose=False):
    """
    Quant BF16 weight to FP8

    Args:
        weight_list (List[Tensor]): Input tensor list in BF16 format
        transpose (Boolean): Transpose operation flag

    Returns:
        Tuple[Tensor, Tensor]: The weight and scale after quant in FP8 format
    """
    if hasattr(weight_list[0], "fp8_weight_stacked"):
        w, scale = _get_fp8_weight_and_scale(weight_list[0], stacked=True, transpose=transpose)
    else:
        w, scale = paddle.incubate.nn.functional.fused_stack_transpose_quant(weight_list, transpose)
    return w, scale


def split_group_gemm(x_fp8, x_scale, w_fp8, w_scale, tokens_per_expert, gemm_out):
    """
    Perform grouped GEMM operation with FP8 tensors, splitting by expert tokens.

    Args:
        x_fp8 (Tensor): Input tensor in FP8 format
        x_scale (Tensor): Scaling factors for input tensor
        w_fp8 (Tensor): Weight tensor in FP8 format
        w_scale (Tensor): Scaling factors for weight tensor
        tokens_per_expert (List[int]): Number of tokens assigned to each expert
        gemm_out (Tensor): Output tensor for GEMM results

    Returns:
        Tensor: The GEMM output tensor with expert-specific computations

    Note:
        This implementation uses deep_gemm operations optimized for FP8 precision
        and handles the case where tokens may be unevenly distributed across experts.
    """
    start_idx = 0
    for i, token_num in enumerate(tokens_per_expert):
        if token_num == 0:
            continue
        end_idx = start_idx + token_num

        x_scale_tma_align = x_scale[start_idx:end_idx].T.contiguous().T

        deep_gemm.gemm_fp8_fp8_bf16_nt(
            (x_fp8[start_idx:end_idx], x_scale_tma_align),
            (w_fp8[i], w_scale[i]),
            gemm_out[start_idx:end_idx],
        )

        start_idx = end_idx

    return gemm_out


def has_config(config_map, key):
    return bool(config_map is not None and key in config_map and config_map[key])


class ExpertsGroupGemmNode:
    """
    Node for performing grouped GEMM operations in FP8 precision for MoE layers.

    This class handles both forward and backward passes for expert computations,
    including specialized operations for:
    - Gate projection (up_gate_proj)
    - SwiGLU activation
    - Down projection (down_proj)

    The implementation supports both standard and probability-weighted computations.
    """

    def __init__(self, experts, custom_map, name="moe_experts_node"):
        """
        Initialize the ExpertsGroupGemmNode.

        Args:
            experts (List[Module]): List of expert modules
            custom_map (CustomMap): Configuration mapping for expert operations
            name (str): Optional name for the node

        Attributes:
            o1 (Tensor): Cache for intermediate gate projection results
            unzipped_tokens (Tensor): Cache for input tokens
            custom_map (CustomMap): Expert configuration mapping
            unzipped_probs (Tensor): Cache for expert probabilities
            tokens_per_expert (List[int]): Token distribution across experts
            fp8_fused_ops_configs (Dict): Configuration for FP8 fused operations
        """
        self.o1 = None
        self.unzipped_tokens = None
        self.custom_map = custom_map
        self.unzipped_probs = None
        self.tokens_per_expert = None
        self.fp8_fused_ops_configs = custom_map.config.fp8_fused_ops_configs

    def reset_status(self):
        self.o1 = None
        self.unzipped_tokens = None
        self.unzipped_probs = None
        self.tokens_per_expert = None

    def fwd_gate_up(self, x_bf16, expert_w1, expert_w_count, tokens_per_expert):
        """
        Forward pass for gate projection in FP8 precision.

        Args:
            x_bf16 (Tensor): Input tensor in bfloat16 format
            expert_w1 (List[Tensor]): List of expert weights for gate projection
            expert_w_count (int): Number of experts
            tokens_per_expert (List[int]): Token distribution across experts

        Returns:
            Tensor: Output of gate projection in bfloat16 format

        Note:
            - Handles both stacked and individual expert weight quantization
            - Supports FP8 fused operations when configured
            - Maintains intermediate results for backward pass
        """
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            w1_t_quant, w1_t_scale = fused_stack_transpose_quant(expert_w1, transpose=True)
        else:
            stacked_w1 = paddle.stack(expert_w1, axis=0)
            stacked_w1_t = paddle.transpose(stacked_w1, [0, 2, 1]).contiguous()
            concated_w1_t = stacked_w1_t.reshape([-1, stacked_w1_t.shape[-1]])

            w1_t_quant, w1_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                concated_w1_t,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=False,
            )

        w1_t_quant = w1_t_quant.reshape([expert_w_count, -1, w1_t_quant.shape[-1]])
        w1_t_scale = w1_t_scale.reshape([expert_w_count, -1, w1_t_scale.shape[-1]])

        x_fp8, x_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            x_bf16,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=False,
        )

        x_fp8 = x_fp8.reshape([expert_w_count, -1, x_fp8.shape[-1]])
        x_scale = x_scale.reshape([expert_w_count, -1, x_scale.shape[-1]])
        x_scale = paddle.transpose(paddle.transpose(x_scale, [0, 2, 1]).contiguous(), [0, 2, 1])

        o1 = paddle.zeros([expert_w_count, x_fp8.shape[1], w1_t_quant.shape[1]], dtype=x_bf16.dtype)
        if numpy.prod(x_fp8.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (x_fp8, x_scale),
                (w1_t_quant, w1_t_scale),
                o1,
                tokens_per_expert,
                x_fp8.shape[1],
            )
        return o1

    def fwd_swiglu(self, o1):
        """
        Compute SwiGLU activation function.

        Args:
            o1 (Tensor): Input tensor from gate projection

        Returns:
            Tensor: Output after SwiGLU activation

        Note:
            Uses PaddlePaddle's optimized swiglu implementation
        """
        o2 = swiglu(o1)
        return o2

    def fwd_down(self, o1, unzipped_probs, expert_w_count, tokens_per_expert):
        """
        Forward pass for down projection with probability weighting.

        Args:
            o1 (Tensor): Input tensor from SwiGLU activation
            unzipped_probs (Tensor): Expert probabilities for each token
            expert_w_count (int): Number of experts
            tokens_per_expert (List[int]): Token distribution across experts

        Returns:
            Tuple[Tensor, Tensor]:
                - Output tensor after down projection
                - Reshaped probabilities tensor

        Note:
            - Handles both standard and fused FP8 quantization paths
            - Applies probability weighting to expert outputs
            - Uses grouped GEMM operations optimized for FP8
        """
        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            w2_quant, w2_scale = fused_stack_transpose_quant(expert_w2, transpose=True)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            stacked_w2_t = paddle.transpose(stacked_w2, [0, 2, 1]).contiguous()
            concated_w2_t = stacked_w2_t.reshape([-1, stacked_w2_t.shape[-1]])

            w2_quant, w2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                concated_w2_t,
                quant_method="128x128",
                input_transpose=False,
                output_scale_transpose=False,
            )
        w2_quant = w2_quant.reshape([expert_w_count, -1, w2_quant.shape[-1]])
        w2_scale = w2_scale.reshape([expert_w_count, -1, w2_scale.shape[-1]])
        o2 = self.fwd_swiglu(o1)
        unzipped_probs = unzipped_probs.unsqueeze(-1).reshape([expert_w_count, -1, 1])
        o2 = (o2 * unzipped_probs).cast(paddle.bfloat16)
        o2_reshape = o2.reshape([-1, o2.shape[-1]]).contiguous()
        o2_quant, o2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            o2_reshape,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=False,
        )

        o2_quant = o2_quant.reshape([expert_w_count, -1, o2_quant.shape[-1]])
        o2_scale = o2_scale.reshape([expert_w_count, -1, o2_scale.shape[-1]])
        o2_scale = paddle.transpose(paddle.transpose(o2_scale, [0, 2, 1]).contiguous(), [0, 2, 1])
        o3 = paddle.zeros([expert_w_count, o2_quant.shape[1], w2_quant.shape[1]], dtype=o1.dtype)
        if numpy.prod(o2_quant.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (o2_quant, o2_scale),
                (w2_quant, w2_scale),
                o3,
                tokens_per_expert,
                o2_quant.shape[1],
            )
        return o3, unzipped_probs

    def fwd_down_no_probs(self, o1, expert_w2, expert_w_count, tokens_per_expert):
        """
        Forward pass for down projection without probability weighting.

        Args:
            o1 (Tensor): Input tensor from SwiGLU activation
            expert_w2 (List[Tensor]): List of expert weights for down projection
            expert_w_count (int): Number of experts
            tokens_per_expert (List[int]): Token distribution across experts

        Returns:
            Tensor: Output tensor after down projection

        Note:
            - Simplified version of fwd_down without probability handling
            - Still maintains FP8 optimized computation path
        """
        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            w2_quant, w2_scale = fused_stack_transpose_quant(expert_w2, transpose=True)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            stacked_w2_t = paddle.transpose(stacked_w2, [0, 2, 1]).contiguous()
            concated_w2_t = stacked_w2_t.reshape([-1, stacked_w2_t.shape[-1]])

            w2_quant, w2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                concated_w2_t,
                quant_method="128x128",
                input_transpose=False,
                output_scale_transpose=False,
            )
        w2_quant = w2_quant.reshape([expert_w_count, -1, w2_quant.shape[-1]])
        w2_scale = w2_scale.reshape([expert_w_count, -1, w2_scale.shape[-1]])
        o2 = self.fwd_swiglu(o1)

        o2_reshape = o2.reshape([-1, o2.shape[-1]]).contiguous()
        o2_quant, o2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            o2_reshape,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=False,
        )

        o2_quant = o2_quant.reshape([expert_w_count, -1, o2_quant.shape[-1]])
        o2_scale = o2_scale.reshape([expert_w_count, -1, o2_scale.shape[-1]])
        o2_scale = paddle.transpose(paddle.transpose(o2_scale, [0, 2, 1]).contiguous(), [0, 2, 1])

        o3 = paddle.zeros([expert_w_count, o2_quant.shape[1], w2_quant.shape[1]], dtype=o1.dtype)
        if numpy.prod(o2_quant.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (o2_quant, o2_scale),
                (w2_quant, w2_scale),
                o3,
                tokens_per_expert,
                o2_quant.shape[1],
            )
        return o3

    def bwd_down_input(self, expert_w2, unzipped_grad, tokens_per_expert, expected_m):
        """
        Backward pass for down projection input gradient computation.

        Args:
            expert_w2 (List[Tensor]): List of expert weights for down projection
            unzipped_grad (Tensor): Gradient from downstream layer
            tokens_per_expert (List[int]): Token distribution across experts
            expected_m (int): Expected batch dimension size

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - Input gradient (do1)
                - SwiGLU output (o2_s)
                - Probability gradients

        Note:
            - Handles both standard and fused FP8 backprop paths
            - Computes gradients for SwiGLU activation and probability weighting
        """
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            bw_w2_quant, bw_w2_scale = fused_stack_transpose_quant(expert_w2, transpose=False)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            concated_w2 = stacked_w2.reshape([-1, stacked_w2.shape[-1]])

            bw_w2_quant, bw_w2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                concated_w2,
                quant_method="128x128",
                input_transpose=False,
                output_scale_transpose=False,
            )
        bw_w2_quant = bw_w2_quant.reshape([len(expert_w2), -1, bw_w2_quant.shape[-1]])
        bw_w2_scale = bw_w2_scale.reshape([len(expert_w2), -1, bw_w2_scale.shape[-1]])

        unzipped_grad_fp8, unzipped_grad_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            unzipped_grad,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=False,
        )
        unzipped_grad_scale = paddle.transpose(
            paddle.transpose(unzipped_grad_scale, [0, 2, 1]).contiguous(), [0, 2, 1]
        )
        do2_s = paddle.zeros(
            [len(expert_w2), unzipped_grad_fp8.shape[1], bw_w2_quant.shape[1]],
            dtype=unzipped_grad.dtype,
        )
        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (unzipped_grad_fp8, unzipped_grad_scale),
                (bw_w2_quant, bw_w2_scale),
                do2_s,
                tokens_per_expert,
                expected_m,
            )
        if has_config(self.fp8_fused_ops_configs, "swiglu_probs_bwd"):
            do1, probs_grad, o2_s = paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
                self.o1, do2_s, self.unzipped_probs
            )
        else:
            o2 = self.fwd_swiglu(self.o1)
            o2_s = (o2 * self.unzipped_probs).cast(paddle.bfloat16)
            do2 = (do2_s.cast(paddle.float32) * self.unzipped_probs).cast(paddle.bfloat16)

            probs_grad = (do2_s.cast(paddle.float32) * (o2.cast(paddle.float32))).sum(axis=-1)
            do1 = self.bwd_swiglu(self.o1, do2)

        return do1, o2_s, probs_grad

    def bwd_down_input_no_prob(self, expert_w2, unzipped_grad, tokens_per_expert, expected_m):
        o2 = self.fwd_swiglu(self.o1)
        o2_s = o2

        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            bw_w2_quant, bw_w2_scale = fused_stack_transpose_quant(expert_w2, transpose=False)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            concated_w2 = stacked_w2.reshape([-1, stacked_w2.shape[-1]])

            bw_w2_quant, bw_w2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                concated_w2,
                quant_method="128x128",
                input_transpose=False,
                output_scale_transpose=False,
            )
        bw_w2_quant = bw_w2_quant.reshape([len(expert_w2), -1, bw_w2_quant.shape[-1]])
        bw_w2_scale = bw_w2_scale.reshape([len(expert_w2), -1, bw_w2_scale.shape[-1]])

        unzipped_grad_fp8, unzipped_grad_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            unzipped_grad,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=False,
        )
        do2_s = paddle.zeros(
            [len(expert_w2), unzipped_grad_fp8.shape[1], bw_w2_quant.shape[1]],
            dtype=unzipped_grad.dtype,
        )
        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (unzipped_grad_fp8, unzipped_grad_scale),
                (bw_w2_quant, bw_w2_scale),
                do2_s,
                tokens_per_expert,
                expected_m,
            )

        return do2_s, o2_s

    def bwd_swiglu(self, o1, do2):
        """
        Backward pass for SwiGLU activation function.

        Args:
            o1 (Tensor): Original input to SwiGLU
            do2 (Tensor): Gradient from downstream layer

        Returns:
            Tensor: Gradient with respect to SwiGLU input

        Note:
            Uses PaddlePaddle's optimized swiglu_grad operation
        """
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        return do1

    def bwd_gate_up_input(self, do1, expert_w1, tokens_per_expert, expected_m):
        """
        Backward pass for gate projection input gradient computation.

        Args:
            do1 (Tensor): Gradient from downstream layer
            expert_w1 (List[Tensor]): List of expert weights for gate projection
            tokens_per_expert (List[int]): Token distribution across experts
            expected_m (int): Expected batch dimension size

        Returns:
            Tensor: Input gradient (dx)

        Note:
            - Performs FP8 optimized GEMM for gradient computation
            - Handles both standard and fused quantization paths
        """
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            bw_w1_quant, bw_w1_scale = fused_stack_transpose_quant(expert_w1, transpose=False)
        else:
            stacked_w1 = paddle.stack(expert_w1, axis=0)
            concated_w1_t_2d = stacked_w1.reshape([-1, stacked_w1.shape[-1]])

            bw_w1_quant, bw_w1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                concated_w1_t_2d,
                quant_method="128x128",
                input_transpose=False,
                output_scale_transpose=False,
            )
        bw_w1_quant = bw_w1_quant.reshape([len(expert_w1), -1, bw_w1_quant.shape[-1]])
        bw_w1_scale = bw_w1_scale.reshape([len(expert_w1), -1, bw_w1_scale.shape[-1]])

        do1_fp8_reshape = do1.reshape([-1, do1.shape[-1]]).contiguous()
        do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            do1_fp8_reshape,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=False,
        )

        do1_fp8 = (do1_fp8.reshape([len(expert_w1), -1, do1_fp8.shape[-1]])).contiguous()
        do1_scale = do1_scale.reshape([len(expert_w1), -1, do1_scale.shape[-1]]).contiguous()
        do1_scale = paddle.transpose(paddle.transpose(do1_scale, [0, 2, 1]).contiguous(), [0, 2, 1])

        dx = paddle.zeros(
            shape=[len(expert_w1), do1_fp8.shape[1], bw_w1_quant.shape[1]],
            dtype=paddle.bfloat16,
        )
        if numpy.prod(do1_fp8.shape) != 0:
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (do1_fp8, do1_scale),
                (bw_w1_quant, bw_w1_scale),
                dx,
                tokens_per_expert,
                expected_m,
            )
        return dx

    def bwd_down_weight(self, out_grad, o2, expert_w2):
        """
        Backward pass for down projection weight gradient computation.

        Args:
            out_grad (Tensor): Gradient from downstream layer
            o2 (Tensor): Output from SwiGLU activation
            expert_w2 (List[Tensor]): List of expert weights for down projection

        Note:
            - Computes weight gradients using FP8 optimized GEMM
            - Handles both main_grad and standard grad accumulation
            - Maintains proper gradient scaling for FP8 precision
        """
        group_num = len(expert_w2)
        H2 = o2.shape[-1]

        o2_t = (
            o2.reshape([group_num, -1, H2])
            .transpose([0, 2, 1])
            .contiguous()
            .reshape([group_num * H2, -1])
            .contiguous()
        )

        o2_t_fp8, o2_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            o2_t,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=True,
        )

        o2_t_fp8 = o2_t_fp8.reshape([group_num, int(o2_t_fp8.shape[0] / group_num), o2_t_fp8.shape[-1]])
        o2_t_scale = paddle.split(o2_t_scale, num_or_sections=group_num, axis=-1)

        H1 = out_grad.shape[-1]
        out_grad = (
            out_grad.reshape([group_num, -1, H1])
            .transpose([0, 2, 1])
            .contiguous()
            .reshape([group_num * H1, -1])
            .contiguous()
        )

        out_grad_fp8, out_grad_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            out_grad,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=True,
        )

        out_grad_fp8 = out_grad_fp8.reshape([group_num, H1, -1])
        out_grad_scale = paddle.split(out_grad_scale, num_or_sections=group_num, axis=-1)

        for i in range(len(expert_w2)):
            if hasattr(expert_w2[i], "main_grad"):
                if expert_w2[i].main_grad is None:
                    expert_w2[i].main_grad = paddle.zeros(shape=expert_w2[i].shape, dtype=paddle.float32)
                fp8_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    out_grad_fp8[i],
                    out_grad_scale[i],
                    True,
                    True,
                    expert_w2[i].main_grad,
                    paddle.float32,
                )
            else:
                if expert_w2[i].grad is None:
                    expert_w2[i].grad = paddle.zeros(shape=expert_w2[i].shape, dtype=paddle.float32)
                fp8_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    out_grad_fp8[i],
                    out_grad_scale[i],
                    True,
                    True,
                    expert_w2[i].grad,
                    paddle.float32,
                )
            if hasattr(expert_w2[i], "_apply_backward_hook"):
                expert_w2[i]._apply_backward_hook()

    def bwd_gate_up_weight(self, do1, input_x, expert_w1):
        group_num = len(expert_w1)
        """
        Backward pass for gate projection weight gradient computation.

        Args:
            do1 (Tensor): Gradient from downstream layer
            input_x (Tensor): Original input to gate projection
            expert_w1 (List[Tensor]): List of expert weights for gate projection

        Note:
            - Computes weight gradients using FP8 optimized GEMM
            - Handles both main_grad and standard grad accumulation
            - Maintains proper gradient scaling for FP8 precision
        """
        H1 = input_x.shape[-1]
        input_x = (
            input_x.reshape([group_num, -1, H1])
            .transpose([0, 2, 1])
            .contiguous()
            .reshape([group_num * H1, -1])
            .contiguous()
        )

        input_x_fp8, input_x_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            input_x,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=True,
        )
        input_x_scale = paddle.split(input_x_scale, num_or_sections=group_num, axis=-1)

        H2 = do1.shape[-1]
        do1 = (
            do1.reshape([group_num, -1, H2])
            .transpose([0, 2, 1])
            .contiguous()
            .reshape([group_num * H2, -1])
            .contiguous()
        )
        do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            do1,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=True,
        )
        do1_scale = paddle.split(do1_scale, num_or_sections=group_num, axis=-1)

        for i in range(len(expert_w1)):
            if hasattr(expert_w1[i], "main_grad"):
                if expert_w1[i].main_grad is None:
                    expert_w1[i].main_grad = paddle.zeros(shape=expert_w1[i].shape, dtype=paddle.float32)
                fp8_gemm(
                    input_x_fp8[i],
                    input_x_scale[i],
                    do1_fp8[i],
                    do1_scale[i],
                    True,
                    True,
                    expert_w1[i].main_grad,
                    paddle.float32,
                )
            else:
                if expert_w1[i].grad is None:
                    expert_w1[i].grad = paddle.zeros(shape=expert_w1[i].shape, dtype=paddle.float32)
                fp8_gemm(
                    input_x_fp8[i],
                    input_x_scale[i],
                    do1_fp8[i],
                    do1_scale[i],
                    True,
                    True,
                    expert_w1[i].grad,
                    paddle.float32,
                )
            if hasattr(expert_w1[i], "_apply_backward_hook"):
                expert_w1[i]._apply_backward_hook()

    @paddle.no_grad()
    def forward(self, hs_out, unzipped_probs, tokens_per_expert):
        expert_w1 = [x.up_gate_proj.weight for x in self.custom_map.experts if x is not None]
        expert_w_count = len(expert_w1)

        o1 = self.fwd_gate_up(hs_out, expert_w1, expert_w_count, tokens_per_expert)
        self.o1 = o1

        o3, unzipped_probs = self.fwd_down(
            o1=o1,
            unzipped_probs=unzipped_probs,
            expert_w_count=expert_w_count,
            tokens_per_expert=tokens_per_expert,
        )

        self.unzipped_probs = unzipped_probs
        self.unzipped_tokens = hs_out
        return o3

    @paddle.no_grad()
    def backward(self, out_grad, tokens_per_expert, dispatched_indices, expected_m):
        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]
        expert_w1 = [x.up_gate_proj.weight for x in self.custom_map.experts if x is not None]

        do1, o2_s, probs_grad = self.bwd_down_input(expert_w2, out_grad, tokens_per_expert, expected_m)

        dx = self.bwd_gate_up_input(do1, expert_w1, tokens_per_expert, expected_m)
        dx = dx.reshape([-1, dx.shape[-1]])
        self.bwd_down_weight(out_grad, o2_s, expert_w2)
        self.bwd_gate_up_weight(do1, self.unzipped_tokens, expert_w1)

        self.reset_status()
        return dx, probs_grad

    @paddle.no_grad()
    def forward_no_prob(self, hs_out, tokens_per_expert):
        expert_w1 = [x.up_gate_proj.weight for x in self.custom_map.experts if x is not None]
        expert_w_count = len(expert_w1)

        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]
        o1 = self.fwd_gate_up(hs_out, expert_w1, expert_w_count, tokens_per_expert)
        self.o1 = o1
        o3 = self.fwd_down_no_probs(o1, expert_w2, expert_w_count, tokens_per_expert)
        self.unzipped_tokens = hs_out
        return o3

    @paddle.no_grad()
    def backward_no_prob(self, out_grad, tokens_per_expert):
        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]
        expert_w1 = [x.up_gate_proj.weight for x in self.custom_map.experts if x is not None]

        expected_m = int(numpy.prod(out_grad.shape[:-1]) // len(expert_w1))

        out_grad = out_grad.reshape([-1, out_grad.shape[-1]])

        do2, o2_s = self.bwd_down_input_no_prob(expert_w2, out_grad, tokens_per_expert, expected_m)

        do1 = self.bwd_swiglu(self.o1, do2)

        dx = self.bwd_gate_up_input(do1, expert_w1, tokens_per_expert, expected_m)
        dx = dx.reshape([-1, dx.shape[-1]])

        self.bwd_down_weight(out_grad, o2_s, expert_w2)
        self.bwd_gate_up_weight(do1, self.unzipped_tokens, expert_w1)

        self.reset_status()
        return dx


class ExpertsGroupGemmContiguousNode:
    """
    Node for performing grouped GEMM operations with contiguous memory layout.

    This optimized version provides better performance for certain hardware configurations
    by ensuring memory access patterns are more cache-friendly. Key differences from
    ExpertsGroupGemmNode include:
    - Contiguous memory layout for all intermediate tensors
    - Specialized handling for recomputation scenarios
    - Optional input dequantization support
    - Split group GEMM optimization when configured
    """

    def __init__(
        self,
        custom_map,
        recompute_fwd_gate_up=False,
        dequant_input=False,
        group=None,
        name="experts_group_gemm_contiguous_node",
    ):
        """
        Initialize the ExpertsGroupGemmContiguousNode.

        Args:
            custom_map (CustomMap): Configuration mapping for expert operations
            recompute_fwd_gate_up (bool): Whether to recompute gate projection in backward pass
            dequant_input (bool): Whether to dequantize input tensors
            name (str): Optional name for the node

        Attributes:
            custom_map (CustomMap): Expert configuration mapping
            recompute_fwd_gate_up (bool): Recompute flag
            dequant_input (bool): Input dequantization flag
            tokens_per_expert (List[int]): Token distribution across experts
            m_indices (Tensor): Expert indices for contiguous operations
            unzipped_probs (Tensor): Cache for expert probabilities
            input (Tensor): Cache for input tensor (bf16)
            input_fp8 (Tensor): Cache for input tensor (FP8)
            input_scale (Tensor): Cache for input scaling factors
            o1 (Tensor): Cache for intermediate gate projection results
            fp8_fused_ops_configs (Dict): Configuration for FP8 fused operations
            is_split_group_gemm (bool): Whether split group GEMM optimization is enabled
        """
        self.custom_map = custom_map
        self.recompute_fwd_gate_up = recompute_fwd_gate_up
        self.dequant_input = dequant_input
        self.tokens_per_expert = None
        self.m_indices = None
        self.unzipped_probs = None
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None
        self.fp8_fused_ops_configs = custom_map.config.fp8_fused_ops_configs
        self.is_split_group_gemm = has_config(self.fp8_fused_ops_configs, "split_group_gemm")
        self.group = group

    def reset_status(self):
        self.tokens_per_expert = None
        self.m_indices = None
        self.unzipped_probs = None
        self.input = None
        self.input_fp8 = None
        self.input_scale = None
        self.o1 = None

    def gen_m_indices(self, tokens_per_expert):
        """
        Generate expert indices tensor for contiguous operations.

        Args:
            tokens_per_expert (List[int]): Token distribution across experts

        Returns:
            Tensor: Contiguous tensor of expert indices

        Note:
            This creates a flat tensor where each element indicates which expert
            should process the corresponding token, enabling efficient batched
            operations with contiguous memory access.
        """
        tokens = []
        for i in range(len(tokens_per_expert)):
            tokens.append(paddle.full([tokens_per_expert[i]], i, dtype="int32"))
        out = paddle.concat(tokens, axis=0)
        return out

    def fwd_gate_up(self, x, expert_w1, num_expert, tokens_per_expert, scale=None):
        """
        Forward pass for gate projection with contiguous memory layout.

        Args:
            x (Tensor): Input tensor in bfloat16 or float8 format
            expert_w1 (List[Tensor]): List of expert weights for gate projection
            num_expert (int): Number of experts
            tokens_per_expert (List[int]): Token distribution across experts
            scale (Tensor|None): Scale tensor for dequantization, optional.

        Returns:
            Tensor: Output of gate projection in bfloat16 format

        Note:
            - Optimized for contiguous memory access patterns
            - Supports both split and non-split group GEMM variants
            - Handles input caching for recomputation scenarios
            - Maintains FP8 precision for compute-intensive operations
        """
        self.tokens_per_expert = tokens_per_expert
        if not self.is_split_group_gemm:
            self.m_indices = self.gen_m_indices(tokens_per_expert)
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            w1_t_quant, w1_t_scale = fused_stack_transpose_quant(expert_w1, transpose=True)
        else:
            stacked_w1 = paddle.stack(expert_w1, axis=0)
            stacked_w1_t = paddle.transpose(stacked_w1, [0, 2, 1]).contiguous()
            concated_w1_t = stacked_w1_t.reshape([-1, stacked_w1_t.shape[-1]])
            w1_t_quant, w1_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                concated_w1_t,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=False,
            )
        w1_t_quant = w1_t_quant.reshape([num_expert, -1, w1_t_quant.shape[-1]])
        w1_t_scale = w1_t_scale.reshape([num_expert, -1, w1_t_scale.shape[-1]])

        if x is None:
            x_fp8, x_scale = self.input_fp8, self.input_scale
            assert x_fp8 is not None and x_scale is not None
        elif scale is not None:
            x_fp8, x_scale = x, scale
            assert (
                self.dequant_input
            ), "If a scale is provided, it indicates that a2a is using fp8. Dequant_input must be enabled."
        else:
            x_fp8, x_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                x,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=True,
            )
            x_scale = x_scale.T

        o1 = paddle.empty([x_fp8.shape[0], w1_t_quant.shape[1]], dtype=expert_w1[0].dtype)
        if numpy.prod(x_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(x_fp8, x_scale, w1_t_quant, w1_t_scale, tokens_per_expert, o1)
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (x_fp8, x_scale),
                    (w1_t_quant, w1_t_scale),
                    o1,
                    m_indices=self.m_indices,
                )

        if self.dequant_input:
            self.input_fp8 = x_fp8
            self.input_scale = x_scale
        else:
            self.input = x
        return o1

    def fwd_swiglu(self, o1):
        o2 = swiglu(o1)
        return o2

    def fwd_down(self, o1, unzipped_probs, expert_w2, num_expert):
        """
        Forward pass for down projection with contiguous memory layout.

        Args:
            o1 (Tensor): Input tensor from SwiGLU activation
            unzipped_probs (Tensor): Expert probabilities for each token
            expert_w2 (List[Tensor]): List of expert weights for down projection
            num_expert (int): Number of experts

        Returns:
            Tuple[Tensor, Tensor]:
                - Output tensor after down projection
                - Reshaped probabilities tensor

        Note:
            - Uses contiguous memory layout for all intermediate tensors
            - Supports fused SwiGLU activation and quantization when configured
            - Handles both split and non-split group GEMM variants
        """
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            w2_quant, w2_scale = fused_stack_transpose_quant(expert_w2, transpose=True)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            stacked_w2_t = paddle.transpose(stacked_w2, [0, 2, 1]).contiguous()
            concated_w2_t = stacked_w2_t.reshape([-1, stacked_w2_t.shape[-1]])
            w2_quant, w2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                concated_w2_t,
                quant_method="128x128",
                input_transpose=False,
                output_scale_transpose=False,
            )
        w2_quant = w2_quant.reshape([num_expert, -1, w2_quant.shape[-1]])
        w2_scale = w2_scale.reshape([num_expert, -1, w2_scale.shape[-1]])

        if has_config(self.fp8_fused_ops_configs, "spaq"):
            with paddle.amp.auto_cast(False):
                o2_fp8, o2_scale = paddle.incubate.nn.functional.fused_weighted_swiglu_act_quant(
                    o1, unzipped_probs, using_pow2_scaling=True
                )
            o2_scale = paddle.transpose(paddle.transpose(o2_scale, [1, 0]).contiguous(), [1, 0])
            unzipped_probs = unzipped_probs.unsqueeze(-1)
        else:
            o2 = self.fwd_swiglu(o1)
            unzipped_probs = unzipped_probs.unsqueeze(-1)
            o2 = (o2 * unzipped_probs).cast(paddle.bfloat16)
            o2_fp8, o2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                o2,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=False,
            )

        o3 = paddle.empty([o2_fp8.shape[0], w2_quant.shape[1]], dtype=o1.dtype)
        if numpy.prod(o2_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(o2_fp8, o2_scale, w2_quant, w2_scale, self.tokens_per_expert, o3)
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (o2_fp8, o2_scale),
                    (w2_quant, w2_scale),
                    o3,
                    m_indices=self.m_indices,
                )
        return o3, unzipped_probs

    def bwd_down_input(self, expert_w2, unzipped_grad, o1):
        """
        Backward pass for down projection input gradient (contiguous version).

        Args:
            expert_w2 (List[Tensor]): List of expert weights for down projection
            unzipped_grad (Tensor): Gradient from downstream layer
            o1 (Tensor): Original input to SwiGLU activation

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - Input gradient (do1)
                - SwiGLU output (o2_s)
                - Probability gradients

        Note:
            - Optimized for contiguous memory access patterns
            - Supports both standard and fused backprop paths
            - Handles split group GEMM when configured
        """
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            bw_w2_quant, bw_w2_scale = fused_stack_transpose_quant(expert_w2, transpose=False)
        else:
            stacked_w2 = paddle.stack(expert_w2, axis=0)
            concated_w2 = stacked_w2.reshape([-1, stacked_w2.shape[-1]])
            bw_w2_quant, bw_w2_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                concated_w2,
                quant_method="128x128",
                input_transpose=False,
                output_scale_transpose=False,
            )
        bw_w2_quant = bw_w2_quant.reshape([len(expert_w2), -1, bw_w2_quant.shape[-1]])
        bw_w2_scale = bw_w2_scale.reshape([len(expert_w2), -1, bw_w2_scale.shape[-1]])

        unzipped_grad_fp8, unzipped_grad_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            unzipped_grad,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=True,
        )
        unzipped_grad_scale = unzipped_grad_scale.T
        do2_s = paddle.empty(
            [unzipped_grad_fp8.shape[0], bw_w2_quant.shape[1]],
            dtype=unzipped_grad.dtype,
        )
        if numpy.prod(unzipped_grad_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(
                    unzipped_grad_fp8,
                    unzipped_grad_scale,
                    bw_w2_quant,
                    bw_w2_scale,
                    self.tokens_per_expert,
                    do2_s,
                )
            else:

                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (unzipped_grad_fp8, unzipped_grad_scale),
                    (bw_w2_quant, bw_w2_scale),
                    do2_s,
                    m_indices=self.m_indices,
                )

        if has_config(self.fp8_fused_ops_configs, "swiglu_probs_bwd"):
            do1, probs_grad, o2_s = paddle.incubate.nn.functional.fused_swiglu_weighted_bwd(
                o1, do2_s, self.unzipped_probs.squeeze(-1)
            )
        else:
            o2 = self.fwd_swiglu(o1)
            o2_s = (o2 * self.unzipped_probs).cast(paddle.bfloat16)
            do2 = (do2_s.cast(paddle.float32) * self.unzipped_probs).cast(paddle.bfloat16)
            probs_grad = (do2_s.cast(paddle.float32) * (o2.cast(paddle.float32))).sum(axis=-1)
            do1 = self.bwd_swiglu(o1, do2)

        return do1, o2_s, probs_grad

    def bwd_swiglu(self, o1, do2):
        do1, _ = paddle._C_ops.swiglu_grad(o1, None, do2)
        return do1

    def bwd_gate_up_input(self, do1, expert_w1):
        """
        Args:
            do1 (Tensor): Gradient from downstream layer
            expert_w1 (List[Tensor]): List of expert weights for gate projection

        Returns:
            Tensor: Input gradient (dx)

        Note:
            - Uses contiguous memory layout for all operations
            - Supports both standard and fused quantization paths
            - Handles split group GEMM when configured
        """
        if has_config(self.fp8_fused_ops_configs, "stack_quant"):
            bw_w1_quant, bw_w1_scale = fused_stack_transpose_quant(expert_w1, transpose=False)
        else:
            stacked_w1 = paddle.stack(expert_w1, axis=0)
            concated_w1_t_2d = stacked_w1.reshape([-1, stacked_w1.shape[-1]])
            bw_w1_quant, bw_w1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                concated_w1_t_2d,
                quant_method="128x128",
                input_transpose=False,
                output_scale_transpose=False,
            )
        bw_w1_quant = bw_w1_quant.reshape([len(expert_w1), -1, bw_w1_quant.shape[-1]])
        bw_w1_scale = bw_w1_scale.reshape([len(expert_w1), -1, bw_w1_scale.shape[-1]])

        do1_fp8, do1_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
            do1,
            quant_method="1x128",
            input_transpose=False,
            output_scale_transpose=True,
        )
        do1_scale = do1_scale.T

        dx = paddle.empty(shape=[do1_fp8.shape[0], bw_w1_quant.shape[1]], dtype=paddle.bfloat16)
        if numpy.prod(do1_fp8.shape) != 0:
            if self.is_split_group_gemm:
                split_group_gemm(
                    do1_fp8,
                    do1_scale,
                    bw_w1_quant,
                    bw_w1_scale,
                    self.tokens_per_expert,
                    dx,
                )
            else:
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (do1_fp8, do1_scale),
                    (bw_w1_quant, bw_w1_scale),
                    dx,
                    m_indices=self.m_indices,
                )

        return dx

    def fused_transpose_split_quant(self, x, tokens_per_expert, pow_2_scales):
        """
        Fused operation combining W-L-C-H transpose, split and quantization.

        Args:
            x (Tensor): Input tensor to process
            tokens_per_expert (List[int]): Token distribution across experts
            pow_2_scales (bool): Whether to use power-of-2 scaling

        Returns:
            Tuple[Tensor, Tensor]:
                - Quantized and split tensor with W-L-C-H layout
                - Corresponding scaling factors

        Note:
            This optimized operation:
            - Reshapes input into [World_size, Local_experts, Channels, Hidden]
            - Performs fused transpose/split/quant in single kernel
            - Maintains W-L-C-H memory layout throughout
            - Reduces memory bandwidth requirements
        """
        with paddle.amp.auto_cast(False):
            out, scale = paddle.incubate.nn.functional.fused_transpose_split_quant(x, tokens_per_expert, pow_2_scales)
        return out, scale

    def bwd_down_weight(self, do3, o2, expert_w2):
        """
        Backward pass for down projection weight gradient (contiguous version).

        Args:
            do3 (Tensor): Gradient from downstream layer
            o2 (Tensor): Output from SwiGLU activation
            expert_w2 (List[Tensor]): List of expert weights for down projection

        Note:
            - Uses contiguous memory layout for all operations
            - Supports both standard and fused transpose/split/quant paths
            - Handles both main_grad and standard grad accumulation
        """
        if has_config(self.fp8_fused_ops_configs, "transpose_split_quant"):
            o2_t_fp8, o2_t_scale = self.fused_transpose_split_quant(o2, self.tokens_per_expert, True)
        else:
            o2_t = o2.transpose([1, 0]).contiguous()
            o2_t_fp8, o2_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                o2_t,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=True,
            )
            o2_t_scale = paddle.split(
                o2_t_scale,
                num_or_sections=[int(x / 128) for x in self.tokens_per_expert],
                axis=0,
            )

        if has_config(self.fp8_fused_ops_configs, "transpose_split_quant"):
            do3_t_fp8, do3_t_scale = self.fused_transpose_split_quant(do3, self.tokens_per_expert, True)
        else:
            do3_t = do3.transpose([1, 0]).contiguous()
            do3_t_fp8, do3_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do3_t,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=True,
            )
            do3_t_scale = paddle.split(
                do3_t_scale,
                num_or_sections=[int(x / 128) for x in self.tokens_per_expert],
                axis=0,
            )

        for i in range(len(expert_w2)):
            if hasattr(expert_w2[i], "main_grad"):
                if expert_w2[i].main_grad is None:
                    expert_w2[i].main_grad = paddle.zeros(shape=expert_w2[i].shape, dtype=paddle.float32)
                fp8_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    do3_t_fp8[i],
                    do3_t_scale[i],
                    True,
                    True,
                    expert_w2[i].main_grad,
                    paddle.float32,
                )
            else:
                if expert_w2[i].grad is None:
                    expert_w2[i].grad = paddle.zeros(shape=expert_w2[i].shape, dtype=paddle.float32)
                fp8_gemm(
                    o2_t_fp8[i],
                    o2_t_scale[i],
                    do3_t_fp8[i],
                    do3_t_scale[i],
                    True,
                    True,
                    expert_w2[i].grad,
                    paddle.float32,
                )
            if hasattr(expert_w2[i], "_apply_backward_hook"):
                expert_w2[i]._apply_backward_hook()

    def bwd_gate_up_weight(self, do1, input_x, expert_w1):
        if has_config(self.fp8_fused_ops_configs, "transpose_split_quant"):
            input_x_t_fp8, input_x_t_scale = self.fused_transpose_split_quant(input_x, self.tokens_per_expert, True)
        else:
            input_x_t = input_x.transpose([1, 0]).contiguous()
            input_x_t_fp8, input_x_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                input_x_t,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=True,
            )
            input_x_t_scale = paddle.split(
                input_x_t_scale,
                num_or_sections=[int(x / 128) for x in self.tokens_per_expert],
                axis=0,
            )

        if has_config(self.fp8_fused_ops_configs, "transpose_split_quant"):
            do1_t_fp8, do1_t_scale = self.fused_transpose_split_quant(do1, self.tokens_per_expert, True)
        else:
            do1_t = do1.transpose([1, 0]).contiguous()
            do1_t_fp8, do1_t_scale = paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                do1_t,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=True,
            )
            do1_t_scale = paddle.split(
                do1_t_scale,
                num_or_sections=[int(x / 128) for x in self.tokens_per_expert],
                axis=0,
            )

        for i in range(len(expert_w1)):
            if hasattr(expert_w1[i], "main_grad"):
                if expert_w1[i].main_grad is None:
                    expert_w1[i].main_grad = paddle.zeros(shape=expert_w1[i].shape, dtype=paddle.float32)
                fp8_gemm(
                    input_x_t_fp8[i],
                    input_x_t_scale[i],
                    do1_t_fp8[i],
                    do1_t_scale[i],
                    True,
                    True,
                    expert_w1[i].main_grad,
                    paddle.float32,
                )
            else:
                if expert_w1[i].grad is None:
                    expert_w1[i].grad = paddle.zeros(shape=expert_w1[i].shape, dtype=paddle.float32)
                fp8_gemm(
                    input_x_t_fp8[i],
                    input_x_t_scale[i],
                    do1_t_fp8[i],
                    do1_t_scale[i],
                    True,
                    True,
                    expert_w1[i].grad,
                    paddle.float32,
                )
            if hasattr(expert_w1[i], "_apply_backward_hook"):
                expert_w1[i]._apply_backward_hook()

    @paddle.no_grad()
    def forward(
        self,
        hs_out,
        unzipped_probs,
        tokens_per_expert,
        origin_token_per_experts,
        scale=None,
    ):
        self.origin_token_per_experts = origin_token_per_experts
        if hs_out.shape[0] == 0:
            o3 = paddle.zeros_like(hs_out)
            self.unzipped_probs = unzipped_probs.unsqueeze(-1)
            return o3
        expert_w1 = [x.up_gate_proj.weight for x in self.custom_map.experts if x is not None]
        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]
        num_expert = len(expert_w1)
        o1 = self.fwd_gate_up(hs_out, expert_w1, num_expert, tokens_per_expert, scale=scale)
        if not self.recompute_fwd_gate_up:
            self.o1 = o1
        o3, unzipped_probs = self.fwd_down(o1, unzipped_probs, expert_w2, num_expert)
        self.unzipped_probs = unzipped_probs
        return o3

    @paddle.no_grad()
    def backward(self, out_grad, a2a_async_fn=None):
        if out_grad.shape[0] == 0:
            dx = paddle.zeros_like(out_grad)
            probs_grad = paddle.zeros_like(self.unzipped_probs)

            for expert in self.custom_map.experts:
                if expert is None:
                    continue

                if hasattr(expert.down_proj.weight, "main_grad"):
                    if expert.down_proj.weight.main_grad is None:
                        expert.down_proj.weight.main_grad = paddle.zeros(
                            shape=expert.down_proj.weight.shape, dtype=paddle.float32
                        )
                else:
                    if expert.down_proj.weight.grad is None:
                        expert.down_proj.weight.grad = paddle.zeros(
                            shape=expert.down_proj.weight.shape, dtype=paddle.float32
                        )

                if hasattr(expert.up_gate_proj.weight, "main_grad"):
                    if expert.up_gate_proj.weight.main_grad is None:
                        expert.up_gate_proj.weight.main_grad = paddle.zeros(
                            shape=expert.up_gate_proj.weight.shape, dtype=paddle.float32
                        )
                else:
                    if expert.up_gate_proj.weight.grad is None:
                        expert.up_gate_proj.weight.grad = paddle.zeros(
                            shape=expert.up_gate_proj.weight.shape, dtype=paddle.float32
                        )

            if a2a_async_fn:
                dx, task = a2a_async_fn(dx)
                task.wait()
            return dx, probs_grad

        expert_w2 = [x.down_proj.weight for x in self.custom_map.experts if x is not None]
        expert_w1 = [x.up_gate_proj.weight for x in self.custom_map.experts if x is not None]

        if self.recompute_fwd_gate_up:
            o1 = self.fwd_gate_up(self.input, expert_w1, len(expert_w1), self.tokens_per_expert)
        else:
            o1 = self.o1

        do1, o2_s, probs_grad = self.bwd_down_input(expert_w2, out_grad, o1)
        del o1
        if not self.recompute_fwd_gate_up:
            self.o1 = None

        if self.dequant_input:
            input = paddle.incubate.nn.functional.fused_act_dequant(self.input_fp8, self.input_scale)
            self.input_scale = None
        else:
            input = self.input

        if a2a_async_fn is None:
            # dw1
            self.bwd_gate_up_weight(do1, input, expert_w1)
            del input

            if not self.dequant_input:
                self.input = None
            # dx
            dx = self.bwd_gate_up_input(do1, expert_w1)

            # release do1 and input
            del do1

            self.bwd_down_weight(out_grad, o2_s, expert_w2)
        else:
            # dx
            dx = self.bwd_gate_up_input(do1, expert_w1)

            dx, task = a2a_async_fn(dx)

            # dw1
            self.bwd_gate_up_weight(do1, input, expert_w1)
            del input

            if not self.dequant_input:
                self.input = None

            # release do1 and input
            del do1

            self.bwd_down_weight(out_grad, o2_s, expert_w2)

            task.wait()

        self.reset_status()
        return dx, probs_grad


class ExpertsGroupGemmWLCHNode(ExpertsGroupGemmContiguousNode):
    """
    Node for performing grouped GEMM operations with W-L-C-H memory layout.

    This specialized version optimizes for distributed MoE scenarios with:
    - World-size (W) dimension for distributed expert parallelism
    - Local-expert (L) dimension for per-node expert processing
    - Channel (C) dimension for feature processing
    - Hidden (H) dimension for output features

    Inherits from ExpertsGroupGemmContiguousNode and adds:
    - W-L-C-H memory layout optimizations
    - Specialized fused transpose/split/quant operations
    - Distributed expert parallelism support
    """

    def __init__(
        self,
        custom_map,
        recompute_fwd_gate_up=False,
        dequant_input=False,
        group=None,
        name="experts_group_gemm_WLCH_node",
    ):
        """
        Initialize the ExpertsGroupGemmWLCHNode.

        Args:
            custom_map (CustomMap): Configuration mapping for expert operations
            recompute_fwd_gate_up (bool): Whether to recompute gate projection in backward pass
            dequant_input (bool): Whether to dequantize input tensors
            name (str): Optional name for the node

        Attributes:
            w (int): World size for distributed expert parallelism
            l (int): Number of local experts per node
            fp8_fused_ops_configs (Dict): Configuration for FP8 fused operations
        """
        super().__init__(
            custom_map,
            recompute_fwd_gate_up=recompute_fwd_gate_up,
            dequant_input=dequant_input,
            group=group,
            name=name,
        )

        self.fp8_fused_ops_configs["transpose_split_quant"] = True
        self.fp8_fused_ops_configs["split_group_gemm"] = False

        self.w = custom_map.world_size
        self.l = custom_map.num_local_experts

    def gen_m_indices(self, tokens_per_expert):
        """
        Generate expert indices tensor with W-L-C-H memory layout.

        Args:
            tokens_per_expert (List[int]): Token distribution across experts

        Returns:
            Tensor: Contiguous tensor of expert indices with W-L-C-H layout

        Note:
            - Creates indices tensor optimized for distributed expert parallelism
            - Layout: [World_size, Local_experts, Channels, Hidden]
            - Ensures contiguous memory access across distributed experts
        """
        m_indices = paddle.arange(self.l, dtype=paddle.int32).repeat_interleave(tokens_per_expert[0])
        m_indices = m_indices.reshape([self.w, self.l, -1]).transpose([1, 0, 2]).contiguous().reshape([-1])

        return m_indices

    def fused_transpose_split_quant(self, x, tokens_per_expert, pow_2_scales):
        s, h = x.shape
        x = x.reshape([self.w, self.l, -1, h])
        out, scale = paddle.incubate.nn.functional.fused_transpose_wlch_split_quant(
            x, tokens_per_expert, pow_2_scales=pow_2_scales
        )
        return out, scale
