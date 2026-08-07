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

# Refer to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


import paddle
import paddle.distributed as dist

from ..parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from .utils import VocabUtility


class VocabParallelCrossEntropy:
    """
    Computes the Cross Entropy Loss splitting the Vocab size across tensor parallel
    ranks. This implementation is used in both fused and unfused cross entropy implementations
    """

    @staticmethod
    def calculate_logits_max(
        vocab_parallel_logits: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Calculates logits_max."""

        vocab_parallel_logits = vocab_parallel_logits.float()
        # Maximum value along vocab dimension across all GPUs.
        logits_max = paddle.max(vocab_parallel_logits, axis=-1)

        return vocab_parallel_logits, logits_max

    @staticmethod
    def calculate_predicted_logits(
        vocab_parallel_logits: paddle.Tensor,
        target: paddle.Tensor,
        logits_max: paddle.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
    ) -> tuple[
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
    ]:
        """Calculates predicted logits."""

        # In-place subtraction reduces memory pressure.
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        partition_vocab_size = vocab_parallel_logits.shape[-1]
        logits_2d = vocab_parallel_logits.view([-1, partition_vocab_size])
        masked_target_1d = masked_target.view([-1])
        arange_1d = paddle.arange(
            start=0, end=logits_2d.shape[0], device=logits_2d.place
        )
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        # exp_logits = vocab_parallel_logits
        exp_logits = paddle.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        return (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        )

    @staticmethod
    def calculate_cross_entropy_loss(
        exp_logits: paddle.Tensor,
        predicted_logits: paddle.Tensor,
        sum_exp_logits: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Calculates cross entropy loss."""

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = paddle.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        return exp_logits, loss

    @staticmethod
    def prepare_gradient_calculation_operands(
        softmax: paddle.Tensor, target_mask: paddle.Tensor
    ) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Prepare gradient calculation operands."""

        # All the inputs have softmax as their gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.shape[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = paddle.arange(
            start=0, end=grad_2d.shape[0], device=grad_2d.place
        )

        softmax_update = 1.0 - target_mask.view(-1).float()

        return grad_2d, arange_1d, softmax_update, grad_input

    @staticmethod
    def calculate_gradients(
        grad_2d: paddle.Tensor,
        arange_1d: paddle.Tensor,
        masked_target_1d: paddle.Tensor,
        softmax_update: paddle.Tensor,
        grad_input: paddle.Tensor,
        grad_output: paddle.Tensor,
    ) -> paddle.Tensor:
        """Calculates gradients."""

        grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input


class _VocabParallelCrossEntropy(paddle.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0):
        """Vocab parallel cross entropy forward function."""

        vocab_parallel_logits, logits_max = (
            VocabParallelCrossEntropy.calculate_logits_max(
                vocab_parallel_logits
            )
        )

        tp_group = get_tensor_model_parallel_group(check_initialized=False)
        if tp_group is not None:
            paddle.distributed.all_reduce(
                logits_max,
                op=paddle.distributed.ReduceOp.MAX,
                group=tp_group,
            )

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]

        if tp_group is not None:
            rank = get_tensor_model_parallel_rank()
            world_size = get_tensor_model_parallel_world_size()
        else:
            rank = 0
            world_size = 1
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size
        )

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            vocab_parallel_logits,
            target,
            logits_max,
            vocab_start_index,
            vocab_end_index,
        )

        if tp_group is not None:
            # All reduce is needed to get the chunks from other GPUs.
            paddle.distributed.all_reduce(
                predicted_logits,
                op=dist.ReduceOp.SUM,
                group=tp_group,
            )

            paddle.distributed.all_reduce(
                sum_exp_logits,
                op=dist.ReduceOp.SUM,
                group=tp_group,
            )

        exp_logits, loss = (
            VocabParallelCrossEntropy.calculate_cross_entropy_loss(
                exp_logits, predicted_logits, sum_exp_logits
            )
        )

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            r"""
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """  # pylint: disable=line-too-long
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities.
            # So we can just take the log to get log-probs.
            log_probs = paddle.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """Vocab parallel cross entropy backward function."""

        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensor()
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        (grad_2d, arange_1d, softmax_update, grad_input) = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(
                softmax, target_mask
            )
        )

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (
                1.0 - smoothing
            ) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad

            # Finally elementwise multiplication with the output gradients.
            grad_input.mul_(grad_output.unsqueeze(dim=-1))
        else:
            grad_input = VocabParallelCrossEntropy.calculate_gradients(
                grad_2d,
                arange_1d,
                masked_target_1d,
                softmax_update,
                grad_input,
                grad_output,
            )

        return grad_input, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits, target, label_smoothing=0.0
):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
            dimension is [sequence_length, batch_size, vocab_size/num_parallel_ranks]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]

        label_smoothing: smoothing factor, must be in range [0.0, 1.0)
                         default is no smoothing (=0.0)
    """
    return _VocabParallelCrossEntropy.apply(
        vocab_parallel_logits, target, label_smoothing
    )
