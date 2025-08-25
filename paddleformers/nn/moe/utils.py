import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import Tensor, _C_ops, framework, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.communication import stream
from paddle.distributed.communication.group import Group
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.nn.functional import moe_combine, moe_gate_dispatch
from paddleformers.utils.log import logger
from paddleformers.transformers.ernie4_5.sequence_parallel_utils import ScatterOp

from paddleformers.transformers.ernie4_5.distributed.common_dist_utils import (
    AllGatherGroupOp,
    ReduceScatterGroupOp,
    all_gather_group,
    get_async_loader,
    hack_offload_wait,
    reduce_scatter_group,
)


def manual_backward(f: Callable, is_first_fwd: bool, *args: List[Any]):
    """
    Perform manual backward pass with gradient tracing control.

    Args:
        f: Function to execute
        is_first_fwd: Whether this is the first forward pass
        args: Arguments for the function

    Returns:
        tuple: (backward function, function outputs)
    """
    tracer = framework._dygraph_tracer()
    orig = tracer._has_grad
    if not is_first_fwd:
        tracer._has_grad = True  # turn on grad trace so we can manual backward

    detached_args = detach_and_requires_grad_(*args)
    detached_args_clone = [
        FakeClone.apply(a) if a is not None else None for a in detached_args
    ]
    out = f(*detached_args_clone)
    if isinstance(out, list):
        out = tuple(out)
    elif not isinstance(out, tuple):
        out = (out,)

    if is_first_fwd:
        tracer._has_grad = orig
        return None, out

    out_cached = [
        FakeClone.apply(o) for o in out if o is not None
    ]  # do not cache stop_gradient output

    for o in out_cached:
        o._clear_dataptr()  # free mem
    tracer._has_grad = orig

    def bwd_f(*grad):
        nonlocal out_cached, detached_args, f
        grad = list(grad)
        grad = [g for g in grad if g is not None]
        assert grad and out_cached, (len(grad), len(out_cached))
        # out 中的 stop_graident 参数，也会收到 gradient，在这里过滤掉
        grad, out_cached = zip(
            *[(g, o) for g, o in zip(grad, out_cached) if not o.stop_gradient]
        )

        assert len(grad) == len(out_cached), (len(grad), len(out_cached), f)
        # out, grad = zip(*[(o, g) for o, g in zip(out, grad) if g is not None])
        paddle.autograd.backward(out_cached, grad)
        return tuple([t.grad for t in detached_args if t is not None])

    return bwd_f, out

def combine_expert_output(expert_output, combine_weights, scatter_index):
    """
    Combine expert outputs using combination weights.

    Args:
        expert_output: Expert outputs [num_experts, capacity, dim]
        combine_weights: Combination weights
        scatter_index: Scatter indices

    Returns:
        Tensor: Combined output [seqlen, dim]
    """
    expert_output = expert_output.reshape(
        [-1, expert_output.shape[-1]]
    )  # [e*1,c,m]
    combined_output = combining(expert_output, combine_weights, scatter_index)

    # if self.output_postprocess is not None:
    #     combined_output = self.output_postprocess(combined_output)

    return combined_output

class GateCombine(PyLayer):
    """
    Custom PyLayer for gate combination operations with backward pass.
    """

    @staticmethod
    def forward(ctx, x, combine_weights, scatter_index):
        """
        Forward pass for gate combination.

        Args:
            x: Input tensor
            combine_weights: Combination weights
            scatter_index: Scatter indices

        Returns:
            Tensor: Combined output
        """
        ctx.x = x
        ctx.combine_weights = combine_weights
        ctx.scatter_index = scatter_index
        ret = moe_combine(x, combine_weights, scatter_index)
        return ret

    @staticmethod
    def backward(ctx, grad_y, *_):
        """
        Backward pass for gate combination.

        Args:
            grad_y: Gradient of output [seqlen, hidden_size]

        Returns:
            tuple: (grad_x, grad_combine_weight, None)
        """
        grad_x, grad_combine_weight_helper = _C_ops.moe_combine_grad(
            ctx.x, ctx.combine_weights, ctx.scatter_index, grad_y
        )
        # grad_combine_weight_helper is the same shape with grad x [seqlen * K, dim]
        # reduce the hidden shape
        # TODO: implement reduce in cuda ops
        grad_combine_weight = grad_combine_weight_helper.sum(-1)
        return grad_x, grad_combine_weight.reshape(ctx.combine_weights.shape), None

def combining(x, combine_weights, scatter_index, hard_gate=False):
    """
    Fused version of combining operation.

    Args:
        x: Input tensor [seq, dim]
        combine_weights: Combination weights [s, k]
        scatter_index: Scatter indices [k, s]
        hard_gate: Whether to use hard gating

    Returns:
        Tensor: Combined output [s, dim]
    """
    if hard_gate:
        x_gatherd = F.embedding(scatter_index, x)  # [s,k,dim]
        return x_gatherd.squeeze(-2)
    if paddle.device.is_compiled_with_custom_device("npu"):
        from ernie.fusion_ops.npu_fusion_ops import npu_combining

        ret = npu_combining(x, combine_weights, scatter_index)
    else:
        ret = GateCombine.apply(x, combine_weights, scatter_index)
    ret.stop_gradient = False
    return ret

def _calc_router_loss(
    self,
    dispatch_mask,
    gate_logits,
    gate_prob,
    num_experts,
    use_group,
    layer_idx,
    token_type=None,
    tokens_type_mask=None,
    dispatch_tokens_mask=None,
    prefix="",
    gate: nn.Layer,
):
    """
    Calculate router loss including auxiliary loss, z-loss and orthogonal loss.

    Args:
        dispatch_mask: Dispatch mask
        gate_logits: Gate logits
        gate_prob: Gate probabilities
        num_experts: Number of experts
        use_group: Whether to use expert groups
        layer_idx: Layer index
        token_type: Token type
        tokens_type_mask: Token type mask
        dispatch_tokens_mask: Dispatch tokens mask
        prefix: Prefix for logging

    Returns:
        Tensor: Total router loss
    """
    router_loss, l_aux, orthogonal_loss, zloss = 0.0, None, None, None
    if gate.config.moe_aux_loss_lambda:
        l_aux = gate._cal_aux_loss(
            gate_prob,
            dispatch_mask,
            num_experts,
            use_group,
            tokens_type_mask,
            dispatch_tokens_mask,
        )
        router_loss += gate.moe_aux_loss_lambda[token_type or 0] * l_aux
    else:
        zero = paddle.to_tensor(0, dtype=paddle.float32)
        router_loss += (
            zero * gate_prob[0, 0]
        )  # must use gate prob to avoid zero pointer
    if gate.config.moe_orthogonal_loss_lambda:
        orthogonal_loss = gate._cal_orthogonal_loss(token_type, use_group)
        router_loss += (
            gate.moe_orthogonal_loss_lambda[token_type or 0] * orthogonal_loss
        )
    if gate.config.moe_z_loss_lambda:
        zloss = gate._cal_z_loss(gate_logits, tokens_type_mask)
        router_loss += gate.moe_z_loss_lambda[token_type or 0] * zloss
    return router_loss

class ReshardCombineWeight(PyLayer):
    """
    Perform weights transform.
    """

    @staticmethod
    def forward(ctx, input, group=None):
        """Converts expert-partitioned weights to sequence-partitioned format.

        Args:
            ctx: PyLayer context object
            input (Tensor): Expert-wise partitioned weights [Seq, k] where:
                            - Non-local experts are zeroed out
                            - Seq: sequence dimension (may be sharded)
                            - k: expert capacity
            group (ProcessGroup): Model parallel group (default:)

        Returns:
            Tensor: Sequence-wise partitioned weights [Seq/n, k] via reduce-scatter
        """

        ctx.mask = input == 0.0
        ctx.group = group
        return reduce_scatter_group(input, group=group)

    @staticmethod
    def backward(ctx, grad):
        """Reconstructs expert-partitioned gradients from sequence-wise gradients.

        Args:
            grad (Tensor): Sequence-wise partitioned gradients [Seq/n, k]

        Returns:
            Tensor: Expert-wise partitioned gradients [Seq, k] with zeros for
                   non-local experts
        """
        gathered = all_gather_group(grad, group=ctx.group)
        return gathered.masked_fill(
            ctx.mask,
            0.0,
        )