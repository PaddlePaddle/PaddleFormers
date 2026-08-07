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


from contextlib import nullcontext
from copy import deepcopy

import paddle
import paddle.nn.functional as F
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
    shard_weight,
)

from paddleformers.fleet import utils
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_expert_parallel_rng_tracker_name,
)
from paddleformers.fleet.transformer.layer import FleetLayer
from paddleformers.fleet.transformer.mlp import MLP, MLPSublayersSpec
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

from .fusion_layer_utils import run_sonic_moe
from .moe_utils import (
    k_grouped_bf16_gemm_tn_contiguous_aligned,
)

try:
    from paddlefleet_ops import deep_gemm as paddlefleet_deep_gemm
    from paddlefleet_ops.sonicmoe.functional import (
        _refresh_fp8_config,
        clear_all_fp8_weight_caches,
    )
    from paddlefleet_ops.sonicmoe.quack_utils import quantize_native_fp8_weights
except (ImportError, RuntimeError):
    pass


try:
    from paddlefleet_ops.sonicmoe.ernie_compat.weight_layout_fusion import (
        fused_grouped_w1_to_sonic,
        fused_sonic_w1_to_grouped,
        fused_transpose_w2_layout,
    )
except (ImportError, RuntimeError):
    fused_grouped_w1_to_sonic = None
    fused_sonic_w1_to_grouped = None
    fused_transpose_w2_layout = None


class BMMFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, y, batch_sizes, trans_y=False):
        ctx.save_for_backward(x, y)
        ctx.batch_sizes = batch_sizes
        ctx.trans_y = trans_y
        return paddle.incubate.nn.functional.batched_gemm(
            x, y, batch_sizes, trans_rhs=trans_y
        )

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensor()
        batch_sizes = ctx.batch_sizes
        trans_y = ctx.trans_y

        if x.stop_gradient:
            dx = None
        else:
            dx = paddle.incubate.nn.functional.batched_gemm(
                grad, y, batch_sizes, trans_rhs=not trans_y
            )
        if y.stop_gradient:
            dy = None
        else:
            lhs, rhs = (grad, x) if trans_y else (x, grad)
            dy = paddle.incubate.nn.functional.batched_gemm(
                lhs, rhs, batch_sizes, trans_lhs=True, trans_rhs=False
            )
        return dx, dy


class DeepGEMMBMMFunction(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, y, batch_sizes):
        ctx.save_for_backward(x, y)
        ctx.batch_sizes = batch_sizes
        out = paddle.zeros([x.shape[0], y.shape[2]], dtype="bfloat16")

        tokens_per_expert_indices = paddle.repeat_interleave(
            paddle.arange(batch_sizes.shape[0]), batch_sizes
        ).cast("int32")

        paddlefleet_deep_gemm.m_grouped_bf16_gemm_nn_contiguous(
            x, y, out, tokens_per_expert_indices
        )

        del tokens_per_expert_indices
        return out

    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensor()
        batch_sizes = ctx.batch_sizes

        tokens_per_expert_indices = paddle.repeat_interleave(
            paddle.arange(batch_sizes.shape[0]), batch_sizes
        ).cast("int32")

        dx = paddle.zeros_like(x)
        paddlefleet_deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            grad,
            y,
            dx,
            tokens_per_expert_indices,
        )
        dx = paddle.cast(dx, paddle.float)

        dy = paddle.zeros_like(y, dtype=paddle.float)
        k_grouped_bf16_gemm_tn_contiguous_aligned(
            a=x,
            b=grad,
            d=dy,
            ks=paddle.tolist(batch_sizes),
            ks_tensor=batch_sizes.cast("int32"),
            c=paddle.zeros_like(y, dtype=paddle.float),
        )

        del tokens_per_expert_indices
        return dx, dy


class GroupedMLPExpert(FleetLayer):
    """An efficient implementation of the Experts layer using GroupedGEMM without TP/DP.

    Executes multiple experts in parallel using only expert parallelism.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        moe_deep_gemm,
        pg_collection: ProcessGroupCollection | None = None,
        intermediate_size_per_partition: int | None = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.config.hidden_act = F.silu
        self.num_local_experts = num_local_experts
        self.moe_deep_gemm = moe_deep_gemm
        # Intermediate size for the local shard of every expert. When using the
        # 'allgather' MoE dispatcher every expert is sharded along its
        # intermediate dim across the EP group, so this can be smaller than
        # ``config.moe_intermediate_size``.
        self.intermediate_size_per_partition = (
            intermediate_size_per_partition
            if intermediate_size_per_partition is not None
            else self.config.moe_intermediate_size
        )
        assert not config.use_bias, (
            "Bias not supported in Grouped GEMM yet, please set 'use_bias' to False."
        )

        self.ep_group = pg_collection.ep if pg_collection else None
        self.expert_parallel = (
            utils.get_pg_size(self.ep_group) > 1 if self.ep_group else False
        )

        if self.config.gated_linear_unit:
            if self.config.hidden_act not in [F.silu, F.gelu]:
                raise ValueError(
                    "Activation function must be silu or gelu when using GroupedMLP."
                )

            def glu(x):
                x = paddle.chunk(x, 2, dim=-1)
                return self.config.hidden_act(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.hidden_act
        self.activation_recompute = (
            self.config.recompute_granularity == "selective"
            and "moe_act" in self.config.recompute_modules
        )
        if self.activation_recompute and self.config.fp8:
            raise ValueError(
                "moe_act recompute for fp8 cannot work with the legacy GroupedMLP."
            )

        # No tensor parallel - full sizes
        fc1_output_size = self.intermediate_size_per_partition
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2

        fc2_input_size = self.intermediate_size_per_partition

        dtype = "bfloat16"
        w1_shape = [
            self.num_local_experts,
            self.config.hidden_size,
            fc1_output_size,
        ]
        w2_shape = [
            self.num_local_experts,
            fc2_input_size,
            self.config.hidden_size,
        ]

        rng_ctx = (
            get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name())
            if paddle.distributed.get_world_size() > 1 and self.expert_parallel
            else nullcontext()
        )

        with rng_ctx:
            self.weight1 = paddle.create_parameter(
                shape=w1_shape,
                dtype=dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            self.weight2 = paddle.create_parameter(
                shape=w2_shape,
                dtype=dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            # Use config.init_method / config.output_layer_init_method
            # which are functions that take a tensor and initialize it using sharedbuffer.
            if self.config.perform_initialization:
                self.config.init_method(self.weight1)
                self.config.output_layer_init_method(self.weight2)
        self.weight1.is_distributed = self.expert_parallel
        self.weight2.is_distributed = self.expert_parallel

    def forward(
        self,
        permuted_local_hidden_states: paddle.Tensor,
        tokens_per_expert: paddle.Tensor,
    ):
        """Forward step of the GroupedMLP without TP/DP."""

        if permuted_local_hidden_states.numel() != 0:
            tokens_per_expert = tokens_per_expert.cpu().tolist()
            tokens_per_expert = [int(x) for x in tokens_per_expert]

            if self.moe_deep_gemm:
                fc1_output = DeepGEMMBMMFunction.apply(
                    permuted_local_hidden_states,
                    self.weight1,
                    paddle.to_tensor(tokens_per_expert, dtype="int32"),
                )
            else:
                fc1_output = BMMFunction.apply(
                    permuted_local_hidden_states,
                    self.weight1,
                    tokens_per_expert,
                )
            if self.activation_recompute:
                raise NotImplementedError(
                    "Recompute in GroupedMLPExpert is not implemented"
                )
            else:
                intermediate_parallel = self.activation_func(fc1_output)
                if self.moe_deep_gemm:
                    fc2_output = DeepGEMMBMMFunction.apply(
                        intermediate_parallel,
                        self.weight2,
                        paddle.to_tensor(tokens_per_expert, dtype="int32"),
                    )
                else:
                    fc2_output = BMMFunction.apply(
                        intermediate_parallel, self.weight2, tokens_per_expert
                    )
        else:
            # No token is allocated for local experts.
            assert paddle.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.reshape(self.config.hidden_size, -1)
            w2 = self.weight2.reshape(-1, self.config.hidden_size)
            h = paddle.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                raise NotImplementedError(
                    "Recompute in GroupedMLPExpert is not implemented"
                )
            else:
                h = self.activation_func(h)
                fc2_output = paddle.matmul(h, w2)

        return fc2_output, None

    def backward_dw(self):
        """Performs backward pass for weight gradients in Experts.
        Empty implementation for compatibility with SequentialMLP and TEGroupedMLP.
        """
        pass

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        # standard:  shard on expert dim (axis=0), each rank owns disjoint experts.
        # allgather: shard on intermediate dim, each rank owns all experts.
        # Cross-topology reshard is handled by the DCP resharder.
        # See _get_intermediate_sharded_state_dict for allgather details.
        is_intermediate_sharded = (
            getattr(self.config, "moe_token_dispatcher_type", None)
            == "allgather"
            and self.expert_parallel
        )

        state_dict = self.state_dict(structured_name_prefix="")

        if is_intermediate_sharded:
            return self._get_intermediate_sharded_state_dict(
                state_dict, structured_name_prefix
            )

        model_type = getattr(self.config, "model_type", "none")
        if "qwen3_vl" not in model_type and "qwen3_5" not in model_type:
            w1 = state_dict["weight1"].reshape(-1, self.weight1.shape[-1])
            w2 = state_dict["weight2"].reshape(-1, self.weight2.shape[-1])
            w1.name = self.weight1.name
            w2.name = self.weight2.name
            state_dict["weight1"] = w1
            state_dict["weight2"] = w2

        sharded_dict = {}
        full_key1 = f"{structured_name_prefix}weight1"
        full_key2 = f"{structured_name_prefix}weight2"
        if self.ep_group is None:
            sharded_dict = build_sharded_state_dict(
                state_dict, None, structured_name_prefix
            )
        else:
            sharded_dict[full_key1] = shard_weight(
                key=full_key1,
                weight=state_dict["weight1"],
                axis=0,
                group=self.ep_group,
            )
            sharded_dict[full_key1].grouped_gemm_param = True
            sharded_dict[full_key2] = shard_weight(
                key=full_key2,
                weight=state_dict["weight2"],
                axis=0,
                group=self.ep_group,
            )
            sharded_dict[full_key2].grouped_gemm_param = True
        return sharded_dict

    def _get_intermediate_sharded_state_dict(
        self, state_dict, structured_name_prefix: str
    ):
        """Build sharded state dict for the allgather EP layout.

        weight1 [E, H, 2*I_local] is reshaped to [E, H, 2, I_local] and
        sharded on axis=3 → global [E, H, 2, I_full].
        Separating gate (axis=2, idx=0) and up (axis=2, idx=1) before
        sharding ensures they stay contiguous across ranks.

        weight2 [E, I_local, H] is sharded on axis=1 → global [E, I_full, H].

        In allgather mode E = num_experts (all experts per rank); in deepep
        mode E = num_experts // EP (disjoint experts per rank).  The DCP
        resharder handles this difference during cross-topology loading.
        """
        if self.ep_group is None:
            raise ValueError(
                "intermediate-sharded layout requires an EP group; "
                f"got ep_group=None. Check moe_token_dispatcher_type "
                f"({self.config.moe_token_dispatcher_type!r}) and EP "
                f"initialisation."
            )
        if not self.config.gated_linear_unit:
            raise ValueError(
                "intermediate-sharded layout currently assumes a gated "
                "linear unit (weight1 last dim = 2 * intermediate); "
                "non-gated MLP support is not implemented."
            )
        ep_size = self.ep_group.nranks
        if (
            self.intermediate_size_per_partition * ep_size
            != self.config.moe_intermediate_size
        ):
            raise ValueError(
                "intermediate-sharded layout inconsistency: "
                f"intermediate_size_per_partition="
                f"{self.intermediate_size_per_partition} * ep_size="
                f"{ep_size} != moe_intermediate_size="
                f"{self.config.moe_intermediate_size}"
            )
        w1_raw = state_dict["weight1"]
        w2_raw = state_dict["weight2"]
        # weight1 must be 3-D [E, H, 2*I_local]
        if len(w1_raw.shape) != 3:
            raise ValueError(
                f"weight1 must be 3-D [E, H, 2*I_local], got shape "
                f"{w1_raw.shape}"
            )
        E = w1_raw.shape[0]
        H = w1_raw.shape[1]
        I_local = self.intermediate_size_per_partition
        expected_last = 2 * I_local
        if w1_raw.shape[-1] != expected_last:
            raise ValueError(
                f"weight1 last dim {w1_raw.shape[-1]} "
                f"!= 2 * intermediate_size_per_partition "
                f"({expected_last})"
            )
        # weight2 shape must be [E, I_local, H]
        expected_w2_shape = [E, I_local, H]
        if list(w2_raw.shape) != expected_w2_shape:
            raise ValueError(
                f"weight2 shape {list(w2_raw.shape)} != expected "
                f"{expected_w2_shape} [E, I_local, H]"
            )
        # Reshape [E, H, 2*I_local] -> [E, H, 2, I_local], shard axis=3.
        # Gate (idx 0) and up (idx 1) are separated on axis=2, so each
        # stays contiguous when axis=3 is sharded across ranks.
        w1 = w1_raw.reshape([E, H, 2, I_local])
        w1.name = self.weight1.name
        # weight2 [E, I_local, H], shard axis=1
        w2 = w2_raw
        w2.name = self.weight2.name
        w1_axis = 3
        w2_axis = 1

        sharded_dict = {}
        full_key1 = f"{structured_name_prefix}weight1"
        full_key2 = f"{structured_name_prefix}weight2"
        sharded_dict[full_key1] = shard_weight(
            key=full_key1,
            weight=w1,
            axis=w1_axis,
            group=self.ep_group,
        )
        sharded_dict[full_key1].grouped_gemm_param = True
        sharded_dict[full_key2] = shard_weight(
            key=full_key2,
            weight=w2,
            axis=w2_axis,
            group=self.ep_group,
        )
        sharded_dict[full_key2].grouped_gemm_param = True
        return sharded_dict


class SonicMoEExpert(GroupedMLPExpert):
    _GROUPED_LAYOUT = "grouped"
    _SONIC_LAYOUT = "sonic"

    @staticmethod
    def _is_tensor_initialized(tensor):
        return (
            not hasattr(tensor, "_is_initialized") or tensor._is_initialized()
        )

    @staticmethod
    def _grouped_w1_to_sonic(weight):
        if fused_grouped_w1_to_sonic is not None:
            return fused_grouped_w1_to_sonic(weight)
        else:
            target_shape = [weight.shape[0], weight.shape[2], weight.shape[1]]
            gate, up = paddle.chunk(weight, 2, axis=-1)
            gate = gate.transpose([0, 2, 1])
            up = up.transpose([0, 2, 1])
            return paddle.stack([gate, up], axis=2).reshape(target_shape)

    @staticmethod
    def _sonic_w1_to_grouped(weight):
        if fused_sonic_w1_to_grouped is not None:
            return fused_sonic_w1_to_grouped(weight)
        else:
            target_shape = [weight.shape[0], weight.shape[2], weight.shape[1]]
            weight = weight.reshape([weight.shape[0], -1, 2, weight.shape[2]])
            gate = weight[:, :, 0, :].transpose([0, 2, 1])
            up = weight[:, :, 1, :].transpose([0, 2, 1])
            return paddle.concat([gate, up], axis=-1)

    @staticmethod
    def _transpose_w2_layout(weight):
        # if not SonicMoEExpert._is_tensor_initialized(weight):
        #     return weight
        if fused_transpose_w2_layout is not None:
            return fused_transpose_w2_layout(weight)
        else:
            return weight.transpose([0, 2, 1])

    @staticmethod
    def _assign_tensor(tensor, value):
        if tensor is value:
            return
        if not value.is_contiguous():
            value = value.contiguous()
        if list(tensor.shape) != list(value.shape):
            tensor.reshape_(list(value.shape))
        # tensor[...] = value
        paddle.assign(value, output=tensor)

    def __init__(
        self,
        num_local_experts: int,
        topk: int,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
        intermediate_size_per_partition: int | None = None,
    ):
        assert config.gated_linear_unit is True, (
            "Sonic MoE must use SwiGLU, i.e. set gated_linear_unit=True."
        )
        super().__init__(
            num_local_experts=num_local_experts,
            config=config,
            moe_deep_gemm=False,
            pg_collection=pg_collection,
            intermediate_size_per_partition=intermediate_size_per_partition,
        )
        self.hidden_size = self.config.hidden_size
        self.K = topk
        self._weights_layout = self._GROUPED_LAYOUT
        self.sonic_moe_config = _refresh_fp8_config()
        self.sonic_moe_config.enabled = self.config.fp8 is not None
        self.sonic_moe_config.fp8_wgrad = self.config.fp8_wgrad
        self.sonic_moe_config.fuse_y1_quant = True
        self.sonic_moe_config.fuse_y1_bf16_trunc = True

    def _convert_grad_layout(self, param, converter, convert_main_grad=True):
        main_grad = getattr(param, "main_grad", None)
        if convert_main_grad and main_grad is not None:
            self._assign_tensor(main_grad, converter(main_grad))
        if param.grad is not None and (
            main_grad is None or param.grad.data_ptr() != main_grad.data_ptr()
        ):
            self._assign_tensor(param.grad, converter(param.grad))

    def _convert_layout(
        self, target_layout, weight1_converter, weight2_converter
    ):
        if self._weights_layout == target_layout:
            return
        with paddle.no_grad():
            for param, converter in (
                (self.weight1, weight1_converter),
                (self.weight2, weight2_converter),
            ):
                if not SonicMoEExpert._is_tensor_initialized(param):
                    shape = param.shape
                    param.get_tensor()._set_dims([shape[0], shape[2], shape[1]])
                else:
                    self._assign_tensor(param, converter(param))
                # weight2's main_grad stays in the original grouped layout
                # ([E, I, H]); its grouped->sonic->grouped transpose is elided and
                # the down-proj bf16/fp8 wgrad accumulates into it via a permute
                # view. weight1's main_grad must still be converted because the
                # grouped->sonic conversion also interleaves gate/up (a perfect
                # shuffle the wgrad kernel does not undo).
                self._convert_grad_layout(
                    param, converter, convert_main_grad=(param is self.weight1)
                )
        self._weights_layout = target_layout

    def convert_weights_to_sonic_layout(self):
        self._convert_layout(
            self._SONIC_LAYOUT,
            self._grouped_w1_to_sonic,
            self._transpose_w2_layout,
        )

    def convert_weights_to_grouped_layout(self):
        self._convert_layout(
            self._GROUPED_LAYOUT,
            self._sonic_w1_to_grouped,
            self._transpose_w2_layout,
        )

    def flush_to_grouped_layout(self):
        self.convert_weights_to_grouped_layout()

    def step(self):
        self.flush_to_grouped_layout()

    @paddle.no_grad()
    def quant_weight(self):
        self.convert_weights_to_sonic_layout()

        payload = quantize_native_fp8_weights(
            self.weight1,
            self.weight2,
        )
        assert payload["format"] == "1x32", (
            f"quant strategy {payload.get('format')} is not supported."
        )
        w1_fp8, w1_scale, w1t_fp8, w1t_scale = payload["w1"]
        w2_fp8, w2_scale, w2t_fp8, w2t_scale = payload["w2"]
        self.weight1.fp8 = (w1_fp8, w1_scale)
        self.weight1.transposed_fp8 = (w1t_fp8, w1t_scale)
        self.weight2.fp8 = (w2_fp8, w2_scale)
        self.weight2.transposed_fp8 = (w2t_fp8, w2t_scale)

    def clear_fp8_weights(self):
        clear_all_fp8_weight_caches()
        for weight in (self.weight1, self.weight2):
            weight.fp8 = None
            weight.transposed_fp8 = None

    def need_quant_weight(self):
        for w in [self.weight1, self.weight2]:
            if not hasattr(w, "fp8") or w.fp8 is None:
                return True
            if not hasattr(w, "transposed_fp8") or w.transposed_fp8 is None:
                return True
        return False

    def forward(
        self,
        hidden_states,
        topk_indices,
        topk_scores,
        use_fp8=False,
        tokens_per_expert=None,
        fp8_scale=None,
        fp8_combine_grad_handle=None,
    ):
        self.convert_weights_to_sonic_layout()
        if self.sonic_moe_config.enabled is True and self.need_quant_weight():
            self.quant_weight()
        hidden_states = run_sonic_moe(
            hidden_states,
            topk_indices,
            topk_scores,
            self.K,
            self.num_local_experts,
            self.weight1,
            self.weight2,
            use_fp8,
            tokens_per_expert=tokens_per_expert,
            fp8_scale=fp8_scale,
            fp8_combine_grad_handle=fp8_combine_grad_handle,
            fp8_config=self.sonic_moe_config,
        )
        return hidden_states

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        self.convert_weights_to_grouped_layout()
        return super().sharded_state_dict(structured_name_prefix)


class StandardMLPExpert(MLP):
    def __init__(
        self,
        config: TransformerConfig,
        moe_intermediate_size: int,
        is_expert: bool,
        mlp_spec: MLPSublayersSpec,
    ):
        if moe_intermediate_size == config.intermediate_size:
            super().__init__(
                config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                # tp_group=pg_collection.expt_tp,
            )
        else:
            # Local SequentialMLP can still be used here by overriding the intermediate_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.intermediate_size = moe_intermediate_size
            super().__init__(
                sequential_mlp_config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                # tp_group=pg_collection.expt_tp,
            )
