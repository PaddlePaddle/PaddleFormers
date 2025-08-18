# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import paddle
import paddle.distributed.fleet as fleet

try:
    from paddle.nn.layer.layers import in_declarative_mode
except:
    from paddle.fluid.dygraph.base import in_declarative_mode
import paddle.distributed as dist
from paddle.autograd import PyLayer

from ..utils.tools import get_env_device


def parallel_matmul(lm_output, logit_weights, tensor_parallel_output=True, training=True):
    """
    Parallel matmul
    Args:
        lm_output: x for matmul
        logit_weights: y for matmul
        tensor_parallel_output: the output is paralleled or not
        training: args for xpu

    Returns:
        rst for matmul
    """
    if get_env_device() == "xpu":
        try:
            from paddle_xpu.layers.nn import parallel_matmul as xpu_parallel_matmul

            xpu_parallel_matmul = xpu_parallel_matmul()
            logits = xpu_parallel_matmul(
                lm_output,
                logit_weights,
                tensor_parallel_output=tensor_parallel_output,
                training=training,
            )
            return logits
        except ImportError:
            pass

    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False

    is_logit_weight_distributed = logit_weights.is_distributed
    #  `is_distributed` in static mode is always False
    if in_declarative_mode() and tensor_parallel_degree > 1:
        is_logit_weight_distributed = True

    if is_fleet_init and tensor_parallel_degree > 1 and is_logit_weight_distributed:
        input_parallel = paddle.distributed.collective._c_identity(lm_output, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


def parallel_linear(lm_output, logit_weights, bias, tensor_parallel_output=True):
    is_fleet_init = True
    tensor_parallel_degree = 1
    try:
        hcg = fleet.get_hybrid_communicate_group()
        model_parallel_group = hcg.get_model_parallel_group()
        tensor_parallel_degree = hcg.get_model_parallel_world_size()
    except:
        is_fleet_init = False

    is_logit_weight_distributed = logit_weights.is_distributed
    #  `is_distributed` in static mode is always False
    if in_declarative_mode() and tensor_parallel_degree > 1:
        is_logit_weight_distributed = True

    if is_fleet_init and tensor_parallel_degree > 1 and is_logit_weight_distributed:
        input_parallel = paddle.distributed.collective._c_identity(lm_output, group=model_parallel_group)
        bias_parallel = paddle.distributed.collective._c_identity(bias, group=model_parallel_group)
        logits = paddle.matmul(input_parallel, logit_weights)
        logits += bias_parallel

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights)
        logits += bias
        return logits


def fused_head_and_loss_fn(
    hidden_states,
    lm_head_weight,
    lm_head_bias,
    labels,
    loss_mask,
    transpose_y,
    num_embeddings,
    tensor_parallel_degree,
    tensor_parallel_output,
    fused_linear,
    loop_chunk_size,
    return_token_loss,
    ignore_index,
):
    """Run FusedHeadAndCrossEntropy."""
    return FusedHeadAndCrossEntropy.apply(
        hidden_states,
        lm_head_weight,
        lm_head_bias,
        labels,
        loss_mask,
        transpose_y,
        num_embeddings,
        tensor_parallel_degree,
        tensor_parallel_output,
        fused_linear,
        loop_chunk_size,
        return_token_loss,
        ignore_index,
    )


class FusedHeadAndCrossEntropy(PyLayer):
    """Fuse LM Head and CrossEntropyLoss into one module."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: paddle.Tensor,
        lm_head_weight: paddle.Tensor,
        lm_head_bias: paddle.Tensor,
        labels: paddle.Tensor,
        loss_mask: paddle.Tensor,
        transpose_y: bool,
        num_embeddings: int,
        tensor_parallel_degree: int,
        tensor_parallel_output: bool,
        fused_linear: bool,
        loop_chunk_size: int,
        return_token_loss: bool,
        ignore_index: int,
    ):
        """Run blockwise parallel cross entropy calculation.

        Args:
            ctx: PyLayerContext
            hidden_states (`paddle.Tensor` of shape `(batch_size, max_seq_len, hidden_size)`): the input features.
            lm_head_weight (`paddle.Tensor` of shape `(hidden_size, vocab_size)`)
            lm_head_bias (`paddle.Tensor` of shape `(vocab_size)`)
            labels (`paddle.Tensor` of shape `(batch_size, max_seq_len)`)
            loss_mask (`paddle.Tensor` of shape `(batch_size, max_seq_len)`)
            transpose_y: bool
            num_embeddings: int
            tensor_parallel_degree: int
            tensor_parallel_output: bool
            fused_linear: bool
            loop_chunk_size: int, default is LOOP_CHUNK_SIZE
            return_token_loss: bool
            ignore_index: int

        Returns:
            loss (`paddle.Tensor` of shape `()`: the output loss.
        """
        if fused_linear:
            # print("Cannot support fused_linear while using use_fused_head_and_loss_fn now!")
            fused_linear = False  # NOTE(hehuang): Cannot support fused_linear now
        # initialize distributed settings
        dtype = hidden_states.dtype
        if tensor_parallel_degree > 1:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            tensor_parallel_degree = hcg.get_model_parallel_world_size()

        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, original_shape[-1]])
        labels = labels.reshape([-1])
        if loss_mask is None:
            ctx.aux_num = 1
            loss_mask = (labels != ignore_index).astype("float32")
        else:
            ctx.aux_num = 2
            loss_mask = loss_mask.reshape([-1]).astype("float32")
        ctx.return_token_loss = return_token_loss
        if return_token_loss:
            divisor = 1
        else:
            divisor = loss_mask.sum()

        n_tokens = hidden_states.shape[0]
        n_classes = lm_head_weight.shape[0] if transpose_y else lm_head_weight.shape[1]

        # cast lm_head weight & bias
        lm_head_weight_cast = lm_head_weight.astype(dtype)
        if lm_head_bias is not None:
            lm_head_bias_cast = lm_head_bias.astype(dtype)

        # initialize indices for labels_one_hot
        if tensor_parallel_degree > 1 and tensor_parallel_output:
            rank = hcg.get_model_parallel_rank()
            per_part_size = num_embeddings // tensor_parallel_degree
            indices = paddle.arange(
                rank * per_part_size,
                rank * per_part_size + n_classes,
                dtype=labels.dtype,
            ).unsqueeze(0)
        else:
            indices = paddle.arange(num_embeddings, dtype=labels.dtype).unsqueeze(0)

        # initialize gradients
        if not return_token_loss:
            if not lm_head_weight.stop_gradient:
                grad_lm_head_weight = paddle.zeros_like(lm_head_weight)
            else:
                grad_lm_head_weight = None
            if lm_head_weight is not None and not lm_head_weight.stop_gradient:
                grad_lm_head_bias = paddle.zeros_like(lm_head_bias)
            else:
                grad_lm_head_bias = None
            if hidden_states.stop_gradient:
                grad_hidden_states = paddle.zeros_like(hidden_states)
            else:
                grad_hidden_states = None

        # initialize outputs
        token_loss = paddle.empty((n_tokens,), dtype=paddle.float32)

        # blockwise calculations
        for i in range(0, n_tokens, loop_chunk_size):
            token_start_idx = i
            token_end_idx = min(i + loop_chunk_size, n_tokens)
            cur_chunk_range = paddle.arange(token_start_idx, token_end_idx)
            hidden_states_chunk = paddle.gather(hidden_states, cur_chunk_range, axis=0)
            labels_chunk = paddle.gather(labels, cur_chunk_range, axis=0)
            loss_mask_chunk = paddle.gather(loss_mask, cur_chunk_range, axis=0)

            # logits calculations
            logits_chunk_cast = paddle.matmul(
                hidden_states_chunk,
                lm_head_weight_cast,
                transpose_y=transpose_y,
            )
            if lm_head_bias is not None:
                logits_chunk_cast += lm_head_bias_cast
            if tensor_parallel_degree > 1 and not tensor_parallel_output:
                logits_chunk_cast_lst = []
                dist.all_gather(
                    logits_chunk_cast_lst,
                    logits_chunk_cast,
                    group=model_parallel_group,
                )
                logits_chunk_cast = paddle.concat(logits_chunk_cast_lst, axis=-1)
            logits_chunk = logits_chunk_cast.astype("float32")

            # log softmax
            max_logits = paddle.max(logits_chunk, axis=-1, keepdim=True)
            if tensor_parallel_degree > 1 and tensor_parallel_output:
                dist.all_reduce(max_logits, op=dist.ReduceOp.MAX, group=model_parallel_group)
            normalized_logits = logits_chunk - max_logits
            exp_logits = paddle.exp(normalized_logits)
            sum_exp_logits = paddle.sum(exp_logits, axis=-1, keepdim=True)
            if tensor_parallel_degree > 1 and tensor_parallel_output:
                dist.all_reduce(
                    sum_exp_logits,
                    op=dist.ReduceOp.SUM,
                    group=model_parallel_group,
                )
            log_sum_exp_logits = paddle.log(sum_exp_logits)

            # cross entropy
            labels_one_hot = labels_chunk.unsqueeze(1) == indices
            label_logits = paddle.sum(
                paddle.where(
                    labels_one_hot,
                    normalized_logits,
                    paddle.zeros_like(normalized_logits),
                ),
                axis=-1,
                keepdim=True,
            )
            if tensor_parallel_degree > 1 and tensor_parallel_output:
                dist.all_reduce(
                    label_logits,
                    op=dist.ReduceOp.SUM,
                    group=model_parallel_group,
                )
            token_loss_chunk = (log_sum_exp_logits - label_logits).squeeze(1) / divisor
            cond = loss_mask_chunk.astype("bool")
            token_loss_chunk = paddle.where(cond, token_loss_chunk, paddle.zeros_like(token_loss_chunk))
            paddle.scatter_(token_loss, cur_chunk_range, token_loss_chunk, overwrite=True)

            # gradients calculations
            if not return_token_loss:
                if tensor_parallel_degree > 1 and not tensor_parallel_output:
                    exp_logits = exp_logits.split(model_parallel_group.nranks, axis=-1)[model_parallel_group.rank]
                    labels_one_hot = labels_one_hot.split(model_parallel_group.nranks, axis=-1)[
                        model_parallel_group.rank
                    ]
                grad_logits_chunk = (exp_logits / sum_exp_logits - labels_one_hot.astype("float32")) / divisor
                grad_logits_chunk = grad_logits_chunk.astype(dtype)
                grad_logits_chunk = paddle.where(
                    cond.unsqueeze(1),
                    grad_logits_chunk,
                    paddle.zeros_like(grad_logits_chunk),
                )

                if grad_hidden_states is not None:
                    paddle.scatter_(
                        grad_hidden_states,
                        cur_chunk_range,
                        paddle.matmul(grad_logits_chunk, lm_head_weight_cast, transpose_y=not transpose_y),
                        overwrite=True,
                    )
                if grad_lm_head_weight is not None:
                    if transpose_y:
                        grad_lm_head_weight += paddle.matmul(
                            grad_logits_chunk,
                            hidden_states_chunk,
                            transpose_x=True,
                        )
                    else:
                        grad_lm_head_weight += paddle.matmul(
                            hidden_states_chunk,
                            grad_logits_chunk,
                            transpose_x=True,
                        )
                if grad_lm_head_bias is not None:
                    grad_lm_head_bias += grad_logits_chunk.astype("float32").sum(axis=0).astype(dtype)

        if return_token_loss:
            loss = token_loss.reshape(original_shape[:-1])

            ctx.save_for_backward(
                hidden_states,
                lm_head_weight,
                lm_head_bias,
                labels,
                loss_mask,
            )
            ctx.transpose_y = transpose_y
            ctx.num_embeddings = num_embeddings
            ctx.loop_chunk_size = loop_chunk_size
            ctx.tensor_parallel_degree = tensor_parallel_degree
            ctx.tensor_parallel_output = tensor_parallel_output

            ctx.original_shape = original_shape
        else:
            loss = token_loss.sum()

            ctx.hidden_states_has_grad = grad_hidden_states is not None
            ctx.lm_head_weight_has_grad = grad_lm_head_weight is not None
            ctx.lm_head_bias_has_grad = grad_lm_head_bias is not None

            grad_args = []
            if ctx.hidden_states_has_grad:
                if tensor_parallel_degree > 1:
                    dist.all_reduce(
                        grad_hidden_states,
                        op=dist.ReduceOp.SUM,
                        group=model_parallel_group,
                    )
                grad_args.append(grad_hidden_states.reshape(original_shape))
            if ctx.lm_head_weight_has_grad:
                grad_args.append(grad_lm_head_weight)
            if ctx.lm_head_bias_has_grad:
                grad_args.append(grad_lm_head_bias)

            ctx.save_for_backward(*grad_args)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """Run the backward of blockwise parallel cross entropy calculation."""
        if not ctx.return_token_loss:
            grad_args = ctx.saved_tensor()
            idx = 0
            if ctx.hidden_states_has_grad:
                grad_hidden_states = grad_args[idx] * grad_output.astype(grad_args[idx].dtype)
                idx += 1
            else:
                grad_hidden_states = None

            if ctx.lm_head_weight_has_grad:
                grad_lm_head_weight = grad_args[idx] * grad_output.astype(grad_args[idx].dtype)
                idx += 1
            else:
                grad_lm_head_weight = None

            if ctx.lm_head_bias_has_grad:
                grad_lm_head_bias = grad_args[idx] * grad_output.astype(grad_args[idx].dtype)
                idx += 1
            else:
                grad_lm_head_bias = None

            if ctx.aux_num == 1:
                return (
                    grad_hidden_states,
                    grad_lm_head_weight,
                    grad_lm_head_bias,
                    None,
                )
            else:
                return (
                    grad_hidden_states,
                    grad_lm_head_weight,
                    grad_lm_head_bias,
                    None,
                    None,
                )

        # return_token_loss = True
        grad_token_loss = grad_output.reshape([-1])
        (
            hidden_states,
            lm_head_weight,
            lm_head_bias,
            labels,
            loss_mask,
        ) = ctx.saved_tensor()
        transpose_y = ctx.transpose_y
        num_embeddings = ctx.num_embeddings
        loop_chunk_size = ctx.loop_chunk_size
        tensor_parallel_degree = ctx.tensor_parallel_degree
        tensor_parallel_output = ctx.tensor_parallel_output

        # initialize distributed settings
        dtype = hidden_states.dtype
        if tensor_parallel_degree > 1:
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            tensor_parallel_degree = hcg.get_model_parallel_world_size()

        n_tokens = hidden_states.shape[0]
        n_classes = lm_head_weight.shape[0] if transpose_y else lm_head_weight.shape[1]

        # cast lm_head weight & bias
        lm_head_weight_cast = lm_head_weight.astype(dtype)
        if lm_head_bias is not None:
            lm_head_bias_cast = lm_head_bias.astype(dtype)

        # initialize indices for labels_one_hot
        if tensor_parallel_degree > 1 and tensor_parallel_output:
            rank = hcg.get_model_parallel_rank()
            per_part_size = num_embeddings // tensor_parallel_degree
            indices = paddle.arange(
                rank * per_part_size,
                rank * per_part_size + n_classes,
                dtype=labels.dtype,
            ).unsqueeze(0)
        else:
            indices = paddle.arange(num_embeddings, dtype=labels.dtype).unsqueeze(0)

        # initialize gradients
        if not lm_head_weight.stop_gradient:
            grad_lm_head_weight = paddle.zeros_like(lm_head_weight)
        else:
            grad_lm_head_weight = None
        if lm_head_weight is not None and not lm_head_weight.stop_gradient and lm_head_bias is not None:
            grad_lm_head_bias = paddle.zeros_like(lm_head_bias)
        else:
            grad_lm_head_bias = None
        if hidden_states.stop_gradient:
            grad_hidden_states = paddle.zeros_like(hidden_states)
        else:
            grad_hidden_states = None

        # blockwise calculations
        for i in range(0, n_tokens, loop_chunk_size):
            token_start_idx = i
            token_end_idx = min(i + loop_chunk_size, n_tokens)
            cur_chunk_range = paddle.arange(token_start_idx, token_end_idx)
            hidden_states_chunk = paddle.gather(hidden_states, cur_chunk_range, axis=0)
            labels_chunk = paddle.gather(labels, cur_chunk_range, axis=0)
            loss_mask_chunk = paddle.gather(loss_mask, cur_chunk_range, axis=0)

            # logits calculations
            logits_chunk_cast = paddle.matmul(
                hidden_states_chunk,
                lm_head_weight_cast,
                transpose_y=transpose_y,
            )
            if lm_head_bias is not None:
                logits_chunk_cast += lm_head_bias_cast
            if tensor_parallel_degree > 1 and not tensor_parallel_output:
                logits_chunk_cast_lst = []
                dist.all_gather(
                    logits_chunk_cast_lst,
                    logits_chunk_cast,
                    group=model_parallel_group,
                )
                logits_chunk_cast = paddle.concat(logits_chunk_cast_lst, axis=-1)
            logits_chunk = logits_chunk_cast.astype("float32")

            # log softmax
            max_logits = paddle.max(logits_chunk, axis=-1, keepdim=True)
            if tensor_parallel_degree > 1 and tensor_parallel_output:
                dist.all_reduce(max_logits, op=dist.ReduceOp.MAX, group=model_parallel_group)
            normalized_logits = logits_chunk - max_logits
            exp_logits = paddle.exp(normalized_logits)
            sum_exp_logits = paddle.sum(exp_logits, axis=-1, keepdim=True)
            if tensor_parallel_degree > 1 and tensor_parallel_output:
                dist.all_reduce(
                    sum_exp_logits,
                    op=dist.ReduceOp.SUM,
                    group=model_parallel_group,
                )

            labels_one_hot = labels_chunk.unsqueeze(1) == indices
            if tensor_parallel_degree > 1 and not tensor_parallel_output:
                exp_logits = exp_logits.split(model_parallel_group.nranks, axis=-1)[model_parallel_group.rank]
                labels_one_hot = labels_one_hot.split(model_parallel_group.nranks, axis=-1)[model_parallel_group.rank]
            grad_logits_chunk = exp_logits / sum_exp_logits - labels_one_hot.astype("float32")
            # NOTE(hehuang): scaling grad_logits_chunk by grad_token_loss
            grad_logits_chunk *= paddle.gather(grad_token_loss, cur_chunk_range, axis=0).unsqueeze(1)
            grad_logits_chunk = grad_logits_chunk.astype(dtype)
            cond = loss_mask_chunk.astype("bool")
            grad_logits_chunk = paddle.where(
                cond.unsqueeze(1),
                grad_logits_chunk,
                paddle.zeros_like(grad_logits_chunk),
            )

            if grad_hidden_states is not None:
                paddle.scatter_(
                    grad_hidden_states,
                    cur_chunk_range,
                    paddle.matmul(grad_logits_chunk, lm_head_weight_cast, transpose_y=not transpose_y),
                    overwrite=True,
                )
            if grad_lm_head_weight is not None:
                if transpose_y:
                    grad_lm_head_weight += paddle.matmul(grad_logits_chunk, hidden_states_chunk, transpose_x=True)
                else:
                    grad_lm_head_weight += paddle.matmul(hidden_states_chunk, grad_logits_chunk, transpose_x=True)
            if grad_lm_head_bias is not None:
                grad_lm_head_bias += grad_logits_chunk.astype("float32").sum(axis=0).astype(dtype)

        if grad_hidden_states is not None:
            if tensor_parallel_degree > 1:
                dist.all_reduce(
                    grad_hidden_states,
                    op=dist.ReduceOp.SUM,
                    group=model_parallel_group,
                )
            grad_hidden_states = grad_hidden_states.reshape(ctx.original_shape)

        if ctx.aux_num == 1:
            return (
                grad_hidden_states,
                grad_lm_head_weight,
                grad_lm_head_bias,
            )
        else:
            return (
                grad_hidden_states,
                grad_lm_head_weight,
                grad_lm_head_bias,
                None,
            )
