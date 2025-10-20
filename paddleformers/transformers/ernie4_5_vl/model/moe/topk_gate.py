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
top2gate
"""

from functools import partial
from typing import Tuple

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import Tensor, _C_ops, nn
from paddle.distributed import fleet
from paddle.incubate.nn.functional import int_bincount
from paddle.nn.clip import _squared_l2_norm
from paddle.utils import unique_name
from paddleformers.utils.log import logger

from ernie.fusion_ops import cal_aux_loss


def masked_fill(x, mask, value):
    """
    Fills elements of the input tensor with a given value where mask is True.

    Args:
        x (Tensor): Input tensor to be modified
        mask (Tensor): Boolean mask tensor (same shape as x)
        value (float|int): Value to fill masked elements with

    Returns:
        Tensor: New tensor with masked elements replaced by value
    """
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


@paddle.no_grad()
def compute_optimal_transport(M, r, c, lam=1.0, epsilon=1e-8, max_iters: int = 10):
    """
    Computes optimal transport matrix and Sinkhorn distance using Sinkhorn-Knopp algorithm.

    Args:
        M (Tensor): Cost matrix (n x m)
        r (Tensor): Source marginals (n,)
        c (Tensor): Target marginals (m,)
        lam (float): Entropic regularization strength
        epsilon (float): Convergence threshold
        max_iters (int): Maximum iterations

    Returns:
        tuple: (optimal transport matrix, Sinkhorn distance)
    """
    n, _ = M.shape
    # P = (- lam * M).exp()
    # P /= P.sum()
    P = F.softmax(-M / lam)
    u = paddle.zeros(n, "float32")
    # normalize this matrix
    for _ in range(max_iters):
        if (u - P.sum(1)).abs().max() < epsilon:
            break
        u = P.sum(1)
        P *= (r / (u + 1e-8)).reshape((-1, 1))
        P *= (c / (P.sum(0) + 1e-8)).reshape((1, -1))
    P = paddle.where(~P.isnan(), P, paddle.zeros_like(P))
    return P, _


def cast_if_needed(x, dtype):
    """
    Casts tensor to specified dtype if not already in that dtype.

    Args:
        x (Tensor): Input tensor
        dtype: Target dtype

    Returns:
        Tensor: Casted tensor
    """
    return x.cast(dtype) if x.dtype != dtype else x


class FusedGateDetachMatmul(paddle.autograd.PyLayer):
    """
    Custom autograd function for fused gate-detached matrix multiplication.
    Optimizes forward/backward passes for MoE routing computations.
    """

    @staticmethod
    def forward(ctx, x, w):
        """
        Forward pass for fused matmul operation.

        Args:
            ctx: Context object
            x (Tensor): Input tensor
            w (Tensor): Weight matrix

        Returns:
            Tensor: Result of matrix multiplication
        """
        ctx.dtype = paddle.float32
        ctx.save_for_backward(x, w)
        return F.linear(cast_if_needed(x, ctx.dtype), cast_if_needed(w, ctx.dtype))

    @staticmethod
    def backward(ctx, y_grad):
        """
        Backward pass for gradient computation.

        Args:
            ctx: Context object
            y_grad (Tensor): Gradient from upstream

        Returns:
            tuple: Gradients with respect to inputs
        """
        x, w = ctx.saved_tensor()
        assert ctx.dtype == y_grad.dtype, "dtype not match"
        x_g, w_g = _C_ops.matmul_grad(cast_if_needed(x, ctx.dtype), cast_if_needed(w, ctx.dtype), y_grad, False, False)

        # Especially fix for lora training.
        if w.stop_gradient:
            return cast_if_needed(x_g, x.dtype), None
        return cast_if_needed(x_g, x.dtype), cast_if_needed(w_g, w.dtype)


def gate_detach_matmul(x, weight, use_fuse):
    """
    Performs gate-detached matrix multiplication with optimization options.

    Args:
        x (Tensor): Input tensor
        weight (Tensor): Weight matrix
        use_fuse (bool): Whether to use fused implementation

    Returns:
        Tensor: Result of matrix multiplication
    """
    if use_fuse:
        return FusedGateDetachMatmul.apply(x, weight)
    else:
        x = cast_if_needed(x, paddle.float32)
        return F.linear(x, weight)


class TopKGate(nn.Layer):
    """
    Fused version of TopK gate for improved performance.
    """

    def __init__(self, config, layer_idx: int, group, gate_weight=None) -> None:
        """
        Initialize the MoE (Mixture of Experts) layer.

        Args:
            config: Model configuration containing MoE parameters
            layer_idx: Index of this layer in the model
            group: Distributed communication group
            gate_weight: Optional pre-existing gate weight tensor
        """
        super().__init__()
        self.config = config

        self.fuse_gate_detach_matmul = config.fuse_gate_detach_matmul

        self.model_dim = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.num_experts_tensor = sum(config.moe_num_experts) if config.multimodel_experts else config.moe_num_experts

        self.cap = config.moe_capacity
        self.group = group

        self.layer_idx = layer_idx
        self.global_aux_loss = config.global_aux_loss
        if self.global_aux_loss:
            self.rank = dist.get_rank(self.group)

        self.sinkhorn_2gate = config.sinkhorn_2gate
        self.sinkhorn_temp = config.sinkhorn_temp
        self.use_correction_bias = config.moe_use_aux_free  # true
        self.use_token_type_bias = config.get("moe_use_token_type_bias", False)

        if config.moe_gate_act == "softmax":
            self.act = partial(F.softmax, axis=-1)  # [S,E]
        elif config.moe_gate_act == "sigmoid":
            self.act = F.sigmoid
        else:
            raise ValueError(f"{config.moe_gate_act} is not supported.")
        self.no_jitter = True
        self.expert_drop = False
        self.eye_matrix = None
        self.eye_matrix_size = None
        self.norm_gate_logits = config.moe_norm_gate_logits  # true
        self.one = paddle.ones([], dtype="float32")

        self.moe_aux_loss_lambda = paddle.to_tensor(config.moe_aux_loss_lambda, dtype="float32")
        self.moe_z_loss_lambda = paddle.to_tensor(config.moe_z_loss_lambda, dtype="float32")
        self.moe_orthogonal_loss_lambda = paddle.to_tensor(config.moe_orthogonal_loss_lambda, dtype="float32")
        if self.moe_aux_loss_lambda.ndim == 0:
            self.moe_aux_loss_lambda = self.moe_aux_loss_lambda.unsqueeze(0)
        if self.moe_z_loss_lambda.ndim == 0:
            self.moe_z_loss_lambda = self.moe_z_loss_lambda.unsqueeze(0)
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
        if config.multimodel_experts:
            if config.get("moe_use_hard_gate", False):
                self.num_experts_list = []
                self.experts_type_mask = []
                # hard-gate + group_experts 需要对gate_logits不同部分分开计算
                experts_ids = paddle.zeros([sum(self.num_experts)], dtype="int64").reshape([config.moe_world_size, -1])
                offset = 0
                for i, expert_num in enumerate(self.num_experts):
                    experts_ids[:, offset : offset + expert_num // config.moe_world_size] = i
                    offset += expert_num // config.moe_world_size
                self.experts_type_ids = experts_ids.reshape([-1])
                logger.info(f"use moe_use_hard_gate, experts_ids: {self.experts_type_ids}")
                for i, expert_num in enumerate(self.num_experts):
                    self.experts_type_mask.append(
                        self.experts_type_ids == i,
                    )
                    self.num_experts_list.append(expert_num)
            else:
                # 非group_experts, 依赖token_type_bias实现hard-gate能力。
                assert not config.moe_group_experts, "group_experts must use hard_gate when multimodel_experts is True"
        else:
            self.num_experts_list = [self.num_experts]
        if gate_weight is not None:
            self.weight = gate_weight
            assert (
                not self.config.moe_use_token_type_bias
            ), "gate_weights is from outside, token_type_bias can't be used"
            logger.info("moe use gate_weight from outside")
            # use fp32 pecison in amp
            self._cast_to_low_precision = False
            self._cast_to_low_precison = False
        else:
            self._create_gate_parameter()
        logger.info(
            f"{config.moe_gate}: w/ capacity: {self.cap} experts:{self.num_experts} "
            f"use_token_type_bias:{self.use_token_type_bias} "
            f"gate_act:{config.moe_gate_act} "
            f"norm_gate_logits={self.norm_gate_logits} use_correction_bias={self.use_correction_bias}"
        )

    def _create_gate_parameter(self):
        """
        Create gate weight parameter.
        """
        if self.config.multimodel_experts:
            # support setting lambda for each expert group
            self.moe_z_loss_lambda = self.moe_z_loss_lambda.expand(len(self.num_experts))
            self.moe_aux_loss_lambda = self.moe_aux_loss_lambda.expand(len(self.num_experts))
            self.moe_orthogonal_loss_lambda = self.moe_orthogonal_loss_lambda.expand(len(self.num_experts))

            for i, num_experts in enumerate(self.num_experts):
                if i == 1:
                    with paddle.utils.unique_name.guard(f"mm_gate_{self.layer_idx}_"):
                        p = self.create_parameter(
                            shape=[self.model_dim, num_experts],
                            dtype="float32",
                            attr=paddle.ParamAttr(name=unique_name.generate("moe_gate")),
                        )
                else:
                    p = self.create_parameter(
                        shape=[self.model_dim, num_experts],
                        dtype="float32",
                        attr=paddle.ParamAttr(name=unique_name.generate("moe_gate")),
                    )
                p.expert_type = f"expert_type_{i}"
                self.add_parameter(
                    "weight" if i == 0 else f"weight_{i}",  # 为了对齐原 state-dict，第一个 gate-weight 不改名.
                    p,
                )
        else:
            self.weight = self.create_parameter(
                shape=[self.model_dim, self.num_experts],
                dtype="float32",
                attr=paddle.ParamAttr(name=unique_name.generate("moe_gate")),  # for resume dense-ckpt
            )
        # use fp32 pecison in amp
        self._cast_to_low_precision = False
        self._cast_to_low_precison = False

    def get_gate_weight(self, transform_weight):
        """
        在`multimodel_experts` 的情况下，将多个 weights merge 成一个整体
        transform_weight: bool, 按照 local-expert id 将 多模态 weight 交叠
        """
        if not self.config.multimodel_experts:
            return self.weight
        if not transform_weight:
            return paddle.concat(
                [getattr(self, "weight" if i == 0 else f"weight_{i}") for i in range(len(self.num_experts))], -1
            )
        weight = paddle.zeros(
            [
                self.model_dim,
                self.config.moe_world_size,
                sum(self.num_experts) // self.config.moe_world_size,
            ],
            dtype="float32",
        )
        offset = 0
        for i, num_experts in enumerate(self.num_experts):
            weight[:, :, offset : offset + num_experts // self.config.moe_world_size] = getattr(
                self, "weight" if i == 0 else f"weight_{i}"
            ).reshape([self.model_dim, self.config.moe_world_size, -1])
            offset += num_experts // self.config.moe_world_size
        weight = weight.reshape([self.model_dim, -1])
        return weight

    def forward(
        self,
        input: Tensor,
        token_type_ids: Tensor = None,
        transform_weight: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for fused gate.

        Args:
            input: Input tensor
            token_type_ids: Token type IDs
            transform_weight: Whether to transform weights

        Returns:
            tuple: (logits, capacity, router_loss)
        """
        capacity = self.get_capacity(input.shape[0])
        weight = self.get_gate_weight(transform_weight)
        with paddle.amp.auto_cast(False):
            logits = gate_detach_matmul(input, weight, self.fuse_gate_detach_matmul)
            if self.use_token_type_bias:
                assert token_type_ids is not None
                assert (
                    token_type_ids.max() < self.bias.shape[0]
                ), f"token_type_ids {token_type_ids.max()} >= bias shape {self.bias.shape[0]}"
                bias = self.bias[token_type_ids]  # [seq]
                logits = logits + bias

            router_loss = paddle.zeros([1], dtype="float32")
            router_loss.stop_gradient = False

        return logits, capacity, router_loss

    def get_capacity(self, num_tokens, cap_factor=None):
        """
        Calculate capacity based on number of tokens.

        Args:
            num_tokens: Number of input tokens
            cap_factor: Optional capacity factor override

        Returns:
            int: Calculated capacity
        """
        num_experts = sum(self.num_experts) if self.config.multimodel_experts else self.num_experts
        if cap_factor is not None:
            cap = cap_factor
        else:
            if self.training:
                cap = self.cap[0]
            elif num_tokens < num_experts:  # seqlen < num_expert
                cap = self.cap[2]
            else:
                cap = self.cap[1]
        # capacity = 2S/E
        capacity = int(cap * num_tokens // num_experts)
        assert capacity > 0, f"requires capacity to >= 0. cap={cap}, num_tokens={num_tokens}"
        return capacity

    def _cal_aux_loss(
        self, gate_prob, dispatch_mask, num_experts=None, use_group=None, tokens_mask=None, dispatch_tokens_mask=None
    ):
        """
        Calculate auxiliary loss for router.

        Args:
            gate_prob: Gate probabilities tensor
            dispatch_mask: Dispatch mask tensor
            num_experts: Number of experts
            use_group: Whether to use expert groups
            tokens_mask: Tokens mask
            dispatch_tokens_mask: Dispatch tokens mask

        Returns:
            Tensor: Calculated auxiliary loss
        """
        if self.act is F.sigmoid:
            gate_prob = gate_prob / gate_prob.sum(-1, keepdim=True)

        if self.use_correction_bias:
            if tokens_mask is not None:
                gate_prob_this_modality = gate_prob[tokens_mask.astype("bool")]
                if gate_prob_this_modality.shape[0]:
                    _, top_idx = gate_prob_this_modality.topk(k=self.config.moe_k, axis=-1)
                    dispatch_mask = int_bincount(top_idx.reshape([-1]), 0, gate_prob.shape[-1], paddle.int64)
                else:
                    dispatch_mask = paddle.zeros(gate_prob.shape[-1], dtype="int64")
                dist.stream.all_reduce(
                    dispatch_mask,
                    group=self.group,
                    use_calc_stream=True,
                )
            else:
                _, top_idx = gate_prob.topk(k=self.config.moe_k, axis=-1)
                dispatch_mask = int_bincount(top_idx.reshape([-1]), 0, gate_prob.shape[-1], paddle.int64)
        if num_experts is None:
            num_experts = self.num_experts_tensor
        if use_group is None:
            use_group = self.config.moe_group_experts

        if (
            (tokens_mask is None or len(tokens_mask.shape) == 1)
            and (tokens_mask is None or tokens_mask.shape[0] == gate_prob.shape[0])
            and gate_prob.shape[0] >= gate_prob.shape[1]
        ):
            if tokens_mask is not None and tokens_mask.dtype != gate_prob.dtype:
                tokens_mask = tokens_mask.astype(gate_prob.dtype)
            l_aux, seqlen_float, ce = cal_aux_loss(
                gate_prob,
                dispatch_mask,
                tokens_mask,
                dispatch_tokens_mask,
                num_experts,
                use_group,
                self.config.moe_k,
                clip_min=1e-6,
            )
            return l_aux

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
        # me = paddle.mean(gate_prob, axis=0)
        # ce = paddle.mean(dispatch_mask.cast("float32"), axis=0)
        if self.global_aux_loss:
            me_list, ce_list = [], []
            dist.all_gather(me_list, me, group=self.group)
            dist.all_gather(ce_list, ce, group=self.group)

            me_list[self.rank] = me
            ce_list[self.rank] = ce
            me = paddle.stack(me_list).mean(0)
            ce = paddle.stack(ce_list).mean(0)
        l_aux = paddle.sum(me * ce) * num_experts
        if use_group:
            l_aux = l_aux / self.config.moe_k

        if scale is not None:
            # forward local me, backward global me
            l_aux = l_aux + (scale - self.one) * l_aux.detach()

        return l_aux

    def _cal_z_loss(self, logits, loss_mask=None):
        """
        Calculate Z-loss for router.

        Args:
            logits: Input logits tensor
            loss_mask: Optional loss mask

        Returns:
            Tensor: Calculated Z-loss
        """

        # l_zloss = logits.exp().sum(1).log().square().mean()
        if loss_mask is not None:
            loss_mask = loss_mask.astype(logits.dtype)
            l_zloss = (logits.logsumexp(1).square() * loss_mask).sum() / paddle.clip(loss_mask.sum(), min=1e-6)
        else:
            l_zloss = logits.logsumexp(1).square().mean()
        # TODO group_experts 分group计算zloss
        return l_zloss

    def _cal_orthogonal_loss_opt_each_weight(self, weight, use_group):
        """
        Calculate optimized orthogonal loss for each weight.

        Args:
            weight: Weight tensor
            use_group: Whether to use expert groups

        Returns:
            Tensor: Calculated orthogonal loss
        """
        if weight.dtype != paddle.float32:
            weight = weight.astype(paddle.float32)

        weight = weight.transpose([1, 0]).contiguous()  # transpose weight here
        wnorm = weight.norm(axis=1)
        weight = weight / paddle.maximum(wnorm, self.eps).unsqueeze(1)

        if use_group:
            weight = weight.reshape([self.config.moe_k, -1, weight.shape[1]])  # [K, E/K, H]
            eye_matrix = paddle.eye(weight.shape[1], dtype=weight.dtype).unsqueeze(0)
        else:
            eye_matrix = paddle.eye(weight.shape[0], dtype=weight.dtype)

        weight_matmul = paddle.matmul(weight, weight, transpose_y=True)

        orthogonal_loss = weight_matmul - eye_matrix
        orthogonal_loss = _squared_l2_norm(orthogonal_loss) / orthogonal_loss.size
        return orthogonal_loss

    def _cal_orthogonal_loss(self, weight_id=None, use_group=None):
        """
        Calculate orthogonal loss for router weights.

        Args:
            weight_id: Optional weight ID
            use_group: Whether to use expert groups

        Returns:
            Tensor: Calculated orthogonal loss
        """
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
        if self.config.multimodel_experts:
            for i in range(1, len(self.config.moe_num_experts)):
                w_ = getattr(self, f"weight_{i}")
                orthogonal_loss += self._cal_orthogonal_loss_opt_each_weight(w_, use_group=False)
        return orthogonal_loss
