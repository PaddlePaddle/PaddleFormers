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

import logging
from functools import partial

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddle.incubate.nn.functional import cal_aux_loss, int_bincount
from paddle.nn.clip import _squared_l2_norm
from paddle.utils import unique_name

try:
    from paddleformers.cli.train.ernie_pretrain.src.utils.misc import (
        global_training_logs,
    )
except ModuleNotFoundError:
    global_training_logs = {}

logger = logging.getLogger(__name__)


def cal_aux_loss_func(
    gate_prob,
    dispatch_mask,
    tokens_mask,
    dispatch_tokens_mask,
    num_experts,
    use_group,
    moe_k,
    global_aux_loss=False,
    rank=None,
    group=None,
):
    if tokens_mask is not None and tokens_mask.dtype != gate_prob.dtype:
        tokens_mask = tokens_mask.astype(gate_prob.dtype)

    scale = None
    if dispatch_tokens_mask is not None:
        seqlen_float = dispatch_tokens_mask.astype(gate_prob.dtype).sum()
        if tokens_mask is not None and gate_prob.shape[0] != dispatch_tokens_mask.shape[0]:
            scale = seqlen_float / paddle.clip(tokens_mask.sum(), min=1e-6)
    elif tokens_mask is not None:
        seqlen_float = tokens_mask.sum()
    else:
        seqlen_float = gate_prob.numel().astype(gate_prob.dtype) / num_experts
    seqlen_float = paddle.clip(seqlen_float, min=1e-6)

    if len(dispatch_mask.shape) == 2:
        dispatch_mask = dispatch_mask.sum(0)
    ce = dispatch_mask.astype(gate_prob.dtype).detach() / seqlen_float
    me = paddle.sum(gate_prob, axis=0) / seqlen_float
    if global_aux_loss:
        me_list, ce_list = [], []
        dist.all_gather(me_list, me, group=group)
        dist.all_gather(ce_list, ce, group=group)

        me_list[rank] = me
        ce_list[rank] = ce
        me = paddle.stack(me_list).mean(0)
        ce = paddle.stack(ce_list).mean(0)

    l_aux = paddle.sum(me * ce) * num_experts
    if use_group:
        l_aux = l_aux / moe_k

    if scale is not None:
        l_aux = l_aux + (scale - 1) * l_aux.detach()

    return l_aux


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


class CalAuxLossFunctor(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        gate_prob,
        dispatch_mask,
        tokens_mask,
        dispatch_tokens_mask,
        num_experts,
        use_group,
        moe_k,
        clip_min=1e-6,
    ):
        if tokens_mask is not None and tokens_mask.dtype != gate_prob.dtype:
            tokens_mask = tokens_mask.astype(gate_prob.dtype)
        loss, seqlen_float, ce = cal_aux_loss(
            gate_prob,
            dispatch_mask,
            tokens_mask,
            dispatch_tokens_mask,
            num_experts,
            use_group,
            moe_k,
            clip_min,
        )
        ctx.save_for_backward(gate_prob, seqlen_float, ce)
        ctx.num_experts = num_experts
        ctx.use_group = use_group
        ctx.moe_k = moe_k
        return loss

    @staticmethod
    def backward(ctx, out_grad):
        gate_prob, seqlen_float, ce = ctx.saved_tensor()
        num_experts = ctx.num_experts
        use_group = ctx.use_group
        moe_k = ctx.moe_k
        return paddle._C_ops.cal_aux_loss_grad(gate_prob, seqlen_float, ce, out_grad, num_experts, use_group, moe_k)


def cast_if_needed(x, dtype):
    return x.cast(dtype) if x.dtype != dtype else x


class FusedGateDetachMatmul(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, w):
        ctx.dtype = paddle.float32
        ctx.save_for_backward(x, w)
        return F.linear(cast_if_needed(x, ctx.dtype), cast_if_needed(w, ctx.dtype))

    @staticmethod
    def backward(ctx, y_grad):
        x, w = ctx.saved_tensor()
        assert ctx.dtype == y_grad.dtype, "dtype not match"
        x_g, w_g = paddle._C_ops.matmul_grad(
            cast_if_needed(x, ctx.dtype),
            cast_if_needed(w, ctx.dtype),
            y_grad,
            False,
            False,
        )
        return cast_if_needed(x_g, x.dtype), cast_if_needed(w_g, w.dtype)


def gate_detach_matmul(x, weight, use_fuse):
    if use_fuse:
        return FusedGateDetachMatmul.apply(x, weight)
    else:
        x = cast_if_needed(x, paddle.float32)
        return F.linear(x, weight)


@paddle.no_grad()
def compute_optimal_transport(M, r, c, lam=1.0, epsilon=1e-8, max_iters: int = 10):
    n, _ = M.shape
    P = F.softmax(-M / lam)
    u = paddle.zeros(n, "float32")
    for _ in range(max_iters):
        if (u - P.sum(1)).abs().max() < epsilon:
            break
        u = P.sum(1)
        P *= (r / (u + 1e-8)).reshape((-1, 1))
        P *= (c / (P.sum(0) + 1e-8)).reshape((1, -1))
    P = paddle.where(~P.isnan(), P, paddle.zeros_like(P))
    return P, _


class Top2Gate(nn.Layer):
    """Gating network for Top-2 Mixture of Experts (MoE) routing.

    This gate computes routing weights for each token and selects the top-2 experts
    for each input token. Supports both standard and balanced routing strategies.

    Attributes:
        config: Configuration object containing hyperparameters.
        layer_idx (int): Identifier for the layer in the overall model.
        group (dist.ProcessGroup): Process group for distributed computation.
        gate_weight (nn.Parameter, optional): Learnable gating weights.
    """

    def __init__(self, config, layer_idx: int, group, gate_weight=None) -> None:
        """Initialize the Top-2 gating network.

        Args:
            config: Configuration object containing:
            layer_idx (int): Identifier for this gating layer (used for logging).
            group (dist.ProcessGroup): Process group for distributed operations.
            gate_weight (nn.Parameter, optional): Pre-initialized gating weight matrix.
                If None, will be initialized internally. Shape: (d_model, num_experts).
        """

        super().__init__()

        self.config = config
        self.fuse_gate_detach_matmul = config.fuse_gate_detach_matmul

        self.model_dim = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.num_experts_tensor = config.moe_num_experts

        self.cap = config.moe_capacity
        self.group = group

        self.layer_idx = layer_idx
        self.global_aux_loss = config.global_aux_loss
        if self.global_aux_loss:
            self.rank = dist.get_rank(self.group)

        self.use_correction_bias = config.moe_use_aux_free

        if config.moe_gate_act == "softmax":
            self.act = partial(F.softmax, axis=-1)
        elif config.moe_gate_act == "sigmoid":
            self.act = F.sigmoid
        else:
            raise ValueError(f"{config.moe_gate_act} is not supported.")

        self.expert_drop = False
        self.norm_gate_logits = config.moe_norm_gate_logits
        self.one = paddle.ones([], dtype="float32")

        self.moe_aux_loss_lambda = paddle.to_tensor(config.moe_aux_loss_lambda, dtype="float32")
        self.moe_orthogonal_loss_lambda = paddle.to_tensor(config.moe_orthogonal_loss_lambda, dtype="float32")
        if self.moe_aux_loss_lambda.ndim == 0:
            self.moe_aux_loss_lambda = self.moe_aux_loss_lambda.unsqueeze(0)
        if self.moe_orthogonal_loss_lambda.ndim == 0:
            self.moe_orthogonal_loss_lambda = self.moe_orthogonal_loss_lambda.unsqueeze(0)

        self.experts_type_ids = None
        if config.moe_orthogonal_loss_lambda:
            if hasattr(fleet.fleet, "_user_defined_strategy"):
                strategy = fleet.fleet._user_defined_strategy
                sharding_configs = strategy.hybrid_configs["sharding_configs"]
                pp_config = strategy.hybrid_configs["pp_configs"]
                assert (
                    not sharding_configs.comm_overlap and not pp_config.sharding_comm_overlap
                ), "orthogonal loss will cause twice gradient accumulate, will break pp/sharding overlap"

        self.eps = paddle.to_tensor([1e-12], dtype="float32")
        self.num_experts_list = [self.num_experts]
        if gate_weight is not None:
            self.weight = gate_weight
            logger.info("moe use gate_weight from outside")
            self._cast_to_low_precision = False
            self._cast_to_low_precision = False
        else:
            self._create_gate_parameter()

    def _create_gate_parameter(self):
        self.weight = self.create_parameter(
            shape=[self.model_dim, self.num_experts],
            dtype="float32",
            attr=paddle.ParamAttr(name=unique_name.generate("moe_gate")),
        )

        self._cast_to_low_precision = False
        self._cast_to_low_precision = False

    def forward(
        self,
        input,
        token_type_ids,
        transform_weight,
        correction_bias,
    ):
        orig_dtype = input.dtype
        weight = self.weight
        with paddle.amp.auto_cast(False):
            logits = gate_detach_matmul(input, weight, self.fuse_gate_detach_matmul)
            (
                capacity,
                dispatch_mask,
                combine_weights,
                scatter_index,
                l_aux,
            ) = self.top2_gating(logits, correction_bias=correction_bias)
            orthogonal_loss = self._cal_orthogonal_loss()
            router_loss = l_aux * self.moe_aux_loss_lambda + orthogonal_loss * self.moe_orthogonal_loss_lambda
            router_loss.stop_gradient = False

        combine_weights = combine_weights.cast(orig_dtype)
        return (
            capacity,
            dispatch_mask,
            combine_weights,
            scatter_index,
            router_loss,
            logits,
        )

    def get_capacity(self, num_tokens, cap_factor=None):
        num_experts = self.num_experts
        if cap_factor is not None:
            cap = cap_factor
        else:
            if self.training:
                cap = self.cap[0]
            elif num_tokens < num_experts:
                cap = self.cap[2]
            else:
                cap = self.cap[1]
        capacity = int(cap * num_tokens // num_experts)
        assert capacity > 0, f"requires capacity to >= 0. cap={cap}, num_tokens={num_tokens}"
        return capacity

    def top2_gating(self, logits, cap=None, correction_bias=None):
        gates = self.act(logits)

        assert logits.ndim == 2, logits.shape
        num_experts = gates.shape[1]
        capacity = self.get_capacity(logits.shape[0], cap)

        score_for_argmax = gates + correction_bias.unsqueeze(0) if correction_bias is not None else gates
        indices1_s = paddle.argmax(score_for_argmax, axis=1)
        mask1 = F.one_hot(indices1_s, num_classes=num_experts).cast(paddle.int64)

        l_aux = self._cal_aux_loss(gates, mask1.sum(axis=0), self.num_experts_tensor)
        logits_w_noise = logits

        logits_except1 = masked_fill(logits_w_noise, mask1.cast(paddle.bool), float("-inf"))
        score_for_argmax = (
            self.act(logits_except1) + correction_bias.unsqueeze(0) if correction_bias is not None else logits_except1
        )
        indices2_s_original = paddle.argmax(score_for_argmax, axis=1)

        mask2 = F.one_hot(indices2_s_original, num_classes=self.num_experts).cast(paddle.int64)

        locations1 = paddle.cumsum(mask1, axis=0) - 1
        locations2 = paddle.cumsum(mask2, axis=0) - 1
        locations2 += paddle.sum(mask1, axis=0, keepdim=True)

        mask1 *= (locations1 < capacity).cast(paddle.int64)
        mask2 *= (locations2 < capacity).cast(paddle.int64)

        locations1_s = paddle.sum(locations1 * mask1, axis=1)
        locations2_s = paddle.sum(locations2 * mask2, axis=1)

        mask1_float = mask1.cast(paddle.float32)
        mask2_float = mask2.cast(paddle.float32)
        gates1_s = (gates * mask1_float).sum(axis=-1)
        gates2_s = (gates * mask2_float).sum(axis=-1)

        if self.norm_gate_logits:
            denom_s = gates1_s + gates2_s
            denom_s = paddle.clip(denom_s, min=1e-6)
            gates1_s /= denom_s
            gates2_s /= denom_s
        if self.training and self.expert_drop:
            gates2_s = paddle.where(
                2 * gates2_s < paddle.rand_like(gates2_s),
                paddle.zeros_like(gates2_s),
                gates2_s,
            )

        gates1 = gates1_s.unsqueeze(1) * mask1_float
        gates2 = gates2_s.unsqueeze(1) * mask2_float

        expert1_index = paddle.argmax(gates1, -1)
        combine1_weight = paddle.max(gates1, -1, keepdim=True)
        scatter1_index = expert1_index * capacity + locations1_s
        scatter1_index = scatter1_index.cast("int64")
        dispatch1_mask = combine1_weight.cast(paddle.bool).detach()

        expert2_index = paddle.argmax(gates2, -1)
        combine2_weight = paddle.max(gates2, -1, keepdim=True)
        scatter2_index = expert2_index * capacity + locations2_s
        scatter2_index = scatter2_index.cast("int64")
        dispatch2_mask = combine2_weight.cast(paddle.bool).detach()

        return (
            capacity,
            paddle.concat((dispatch1_mask, dispatch2_mask), 1),
            paddle.concat((combine1_weight, combine2_weight), 1),
            paddle.stack((scatter1_index, scatter2_index), 1),
            l_aux,
        )

    def _cal_aux_loss(
        self,
        gate_prob,
        dispatch_mask,
        num_experts=None,
        use_group=None,
        tokens_mask=None,
        dispatch_tokens_mask=None,
    ):
        if self.act is F.sigmoid:
            gate_prob = gate_prob / gate_prob.sum(-1, keepdim=True)

        if self.use_correction_bias:
            if tokens_mask is not None:
                gate_prob_this_modality = gate_prob[tokens_mask.astype("bool")]
                if gate_prob_this_modality.shape[0]:
                    _, top_idx = gate_prob_this_modality.topk(k=self.config.moe_k, axis=-1)
                    dispatch_mask = int_bincount(top_idx, 0, gate_prob.shape[-1], paddle.int64)
                else:
                    dispatch_mask = paddle.zeros(gate_prob.shape[-1], dtype="int64")
                dist.stream.all_reduce(
                    dispatch_mask,
                    group=self.group,
                    use_calc_stream=True,
                )
            else:
                _, top_idx = gate_prob.topk(k=self.config.moe_k, axis=-1)
                dispatch_mask = int_bincount(top_idx, 0, gate_prob.shape[-1], paddle.int64)

        if num_experts is None:
            num_experts = self.num_experts_tensor
        if use_group is None:
            use_group = self.config.moe_group_experts

        return CalAuxLossFunctor.apply(
            gate_prob,
            dispatch_mask,
            tokens_mask,
            dispatch_tokens_mask,
            num_experts,
            use_group,
            self.config.moe_k,
            clip_min=1e-6,
        )

    def _cal_orthogonal_loss(self, weight_id=None, use_group=None):
        if use_group is None:
            use_group = self.config.moe_group_experts and self.config.moe_group_orthogonal_loss

        if weight_id is not None:
            if weight_id == 0:
                w_ = self.weight
            else:
                assert self.config.multimodel_experts
                w_ = getattr(self, f"weight_{weight_id}")
            return self._cal_orthogonal_loss_opt_each_weight(w_, use_group)

        orthogonal_loss = self._cal_orthogonal_loss_opt_each_weight(self.weight, use_group)
        return orthogonal_loss

    def _cal_orthogonal_loss_opt_each_weight(self, weight, use_group):
        if weight.dtype != paddle.float32:
            weight = weight.astype(paddle.float32)

        return cal_orthogonal_loss_opt_each_weight_func(weight, self.config.moe_k, use_group, self.eps, self.training)


def cal_orthogonal_loss_opt_each_weight_func(weight, moe_k, use_group, eps, training=True):
    weight = weight.transpose([1, 0]).contiguous()  # transpose weight here
    wnorm = weight.norm(axis=1)
    weight = weight / paddle.maximum(wnorm, eps).unsqueeze(1)

    if use_group:
        weight = weight.reshape([moe_k, -1, weight.shape[1]])  # [K, E/K, H]
        eye_matrix = paddle.eye(weight.shape[1], dtype=weight.dtype).unsqueeze(0)
    else:
        eye_matrix = paddle.eye(weight.shape[0], dtype=weight.dtype)

    weight_matmul = paddle.matmul(weight, weight, transpose_y=True)

    orthogonal_loss = weight_matmul - eye_matrix
    orthogonal_loss = _squared_l2_norm(orthogonal_loss) / orthogonal_loss.size
    return orthogonal_loss


class TopKGateFused(Top2Gate):
    def forward(
        self,
        input,
        token_type_ids=None,
        transform_weight=True,
    ):
        capacity = self.get_capacity(input.shape[0])
        weight = self.weight
        with paddle.amp.auto_cast(False):
            logits = gate_detach_matmul(input, weight, self.fuse_gate_detach_matmul)
            router_loss = paddle.zeros([1], dtype="float32")
            router_loss.stop_gradient = False
        return logits, capacity, router_loss
