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

from __future__ import annotations

import functools
import hashlib
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
import paddlefleet_ops
from paddle import framework, nn
from paddle.autograd import PyLayer
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    GatherOp,
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)

if TYPE_CHECKING:
    from paddle.distributed.fleet.meta_parallel import LayerSpec

    from paddleformers.fleet.process_groups_config import ProcessGroupCollection
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig

from paddleformers.fleet import utils
from paddleformers.fleet.recompute_utils import need_recompute_in_first_n
from paddleformers.fleet.transformer.utils import profile

from .fp8_utils import fused_stack_quant_without_cache
from .fused_a2a import configure_buffer
from .fusion_layer_utils import (
    FusionMoePyLayer,
    HybridEPMoePyLayer,
)
from .moe_expert import GroupedMLPExpert, SonicMoEExpert, StandardMLPExpert
from .moe_router import TopKRouter
from .moe_shared_expert import StandardMLPSharedExpert
from .moe_utils import AddAuxiliaryLoss, use_accuracy_compatible_kernel
from .token_dispatcher import (
    AllGatherTokenDispatcher,
    AllToAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    is_hybrid_ep_backend_selected,
)

logger = logging.getLogger(__name__)


# MD5 logging for MoE precision debugging
_LOG_LAYER_MD5 = os.environ.get("LOG_LAYER_MD5", "0") == "1"


def _log_moe_md5(tensor, name, layer_idx=None):
    """Log MD5 of a tensor for MoE precision alignment debugging."""
    from paddleformers.fleet.transformer.transformer_layer import TransformerLayer

    if _LOG_LAYER_MD5 and TransformerLayer._gpt_model_use_experimental_version:
        if TransformerLayer._skip_mtp_probes:
            return  # Skip MTP passes — EC has no MTP
        data = tensor.detach().cast("float32").numpy().tobytes()
        md5 = hashlib.md5(data).hexdigest()
        rank = (
            paddle.distributed.get_rank()
            if paddle.distributed.is_initialized()
            else 0
        )
        layer_str = f" Layer={layer_idx}" if layer_idx is not None else ""
        print(
            f"[MD5 MoE] Rank={rank}{layer_str} {name} MD5={md5} shape={list(tensor.shape)}",
            flush=True,
        )


from .moe_utils import (
    global_moe_balance_training_logs_enabled,
    log_moe_balance,
    log_moe_losses,
    permute,
    unpermute,
)


class GradDtypeGuard(PyLayer):
    """Guard the grad's dtype if different from input's dtype."""

    @staticmethod
    def forward(ctx, x, dtype):
        """forward"""
        return paddle.empty([0], dtype=dtype), {"x": x}

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        return grad


class GradDtypeUnguard(PyLayer):
    """Remove grad dtype guard."""

    @staticmethod
    def forward(ctx, x, status):
        """forward"""
        if hasattr(ctx, "set_grad_in_dtype_consistent"):
            ctx.set_grad_in_dtype_consistent(False)
        return status["x"]

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        return grad


@dataclass
class MoESublayers:
    """MoE Layer Sublayers spec"""

    mlp_spec: LayerSpec | type = None  # Used by experts


class MoELayer(nn.Layer):
    def __init__(
        self,
        config: TransformerConfig,
        sublayers: MoESublayers | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__()
        self.config = config
        self.moe_sublayers = sublayers
        routed_expert_config = deepcopy(config)
        shared_expert_config = deepcopy(config)
        global_use_bias = routed_expert_config.use_bias
        moe_routed_expert_use_bias = config.moe_routed_expert_use_bias
        if moe_routed_expert_use_bias is not None:
            routed_expert_config.use_bias = moe_routed_expert_use_bias
            logger.info(
                "PaddleFleet MoELayer moe_routed_expert_use_bias overrides "
                "routed_expert_config.use_bias: global_use_bias=%s moe_routed_expert_use_bias=%s",
                global_use_bias,
                moe_routed_expert_use_bias,
            )
        self.pg_collection = pg_collection
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.moe_shared_expert_intermediate_size = None
        if self.n_shared_experts:
            self.moe_shared_expert_intermediate_size = (
                self.moe_intermediate_size * self.n_shared_experts
            )
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_act = config.hidden_act
        self.sequence_parallel = config.sequence_parallel
        self.tensor_model_parallel_size = config.tensor_model_parallel_size
        self.moe_token_dispatcher_type = config.moe_token_dispatcher_type
        self.moe_allgather_gate_overlap = config.moe_allgather_gate_overlap
        self.use_hybrid_ep_backend = False
        self.moe_shared_expert_overlap = config.moe_shared_expert_overlap
        self.fp8 = config.fp8
        self.use_ue8m0 = config.use_ue8m0
        self.dw_p2p_overlap = getattr(config, "dw_p2p_overlap", False)
        self.using_sonic_moe = self.config.using_sonic_moe
        self.fp8_dispatch = bool(config.fp8)
        self.fp8_wgrad = config.fp8_wgrad
        self.fp8_dispatch_bwd = (
            self.fp8_dispatch and self.using_sonic_moe and self.fp8_wgrad
        )
        self.moe_expert_fusion = config.moe_expert_fusion
        self.moe_subbatch_token_num_after_dispatch = (
            config.moe_subbatch_token_num_after_dispatch
        )
        if self.using_sonic_moe:
            assert paddlefleet_ops.is_sonic_moe_available(), (
                paddlefleet_ops.blocked_import_messages[
                    "paddlefleet_ops.sonicmoe"
                ]
            )
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.moe_deep_gemm = config.moe_deep_gemm

        if self.moe_deep_gemm:
            incompatible_reasons = []
            if not self.moe_expert_fusion:
                incompatible_reasons.append("moe_expert_fusion must be True")
            if incompatible_reasons:
                logging.warning(
                    "moe_deep_gemm=True is ignored because %s; "
                    "setting moe_deep_gemm to False.",
                    " and ".join(incompatible_reasons),
                )
                self.moe_deep_gemm = False
        self.moe_ep_barrier = config.moe_ep_barrier

        # Latent MoE initialization
        self.use_latent_moe = (
            self.config.moe_latent_size is not None
            and self.config.moe_latent_size > 0
        )
        if self.use_latent_moe:
            logging.info(
                f"Latent MoE enabled: hidden_size={self.config.hidden_size} -> moe_latent_size={self.config.moe_latent_size}"
            )
            self.fc1_latent_proj = nn.Linear(
                self.config.hidden_size,
                self.config.moe_latent_size,
                bias_attr=self.config.use_bias,
            )
            self.fc2_latent_proj = nn.Linear(
                self.config.moe_latent_size,
                self.config.hidden_size,
                bias_attr=self.config.use_bias,
            )
            # Override default XavierUniform with config init methods
            self.config.init_method(self.fc1_latent_proj.weight)
            self.config.output_layer_init_method(self.fc2_latent_proj.weight)
            # Update expert config to use latent size
            routed_expert_config.hidden_size = self.config.moe_latent_size
        # Cached latent-space projection from _maybe_pre_allgather_overlap;
        # consumed (and cleared) by _project_to_latent. Initialised here so the
        # attribute always exists regardless of which forward entry path is
        # taken (custom_forward vs fusion_moe_forward) and whether overlap fired.
        self._latent_hidden = None
        self.moe_group = pg_collection.ep
        self.expert_model_parallel_size = (
            utils.get_pg_size(self.moe_group)
            if self.moe_group is not None
            else 1
        )
        self.num_local_experts = (
            self.num_experts // self.expert_model_parallel_size
        )
        # MoE-Related Configs
        self._init_expert_parallel()

        self.gate = TopKRouter(config=config, pg_collection=pg_collection)

        self.expert_class = StandardMLPExpert
        self.shared_expert_class = StandardMLPSharedExpert

        if (
            self.expert_model_parallel_size <= 1
            and self.sequence_parallel
            and self.tensor_model_parallel_size > 1
        ):
            routed_expert_config.sequence_parallel = False
            if not self.config.gpt_model_use_experimental_version:
                shared_expert_config.sequence_parallel = False
        elif (
            self.expert_model_parallel_size > 1
            and self.tensor_model_parallel_size >= 1
            or paddle.version.cuda() == "12.6"
        ):
            routed_expert_config.tensor_model_parallel_size = 1

        if (
            paddle.is_compiled_with_cuda()
            and paddle.device.get_device_capability()[0] < 9
        ):
            # TODO: Support Ampere architecture after upgrade deepep in paddlepaddle
            if self.moe_token_dispatcher_type in ("deepep", "hybridep"):
                logger.info(
                    "deepep/hybridep in paddlepaddle does not support compute capability < 9.0, "
                    "fallback to alltoall token dispatcher."
                )
                self.moe_token_dispatcher_type = "alltoall"
            if self.moe_deep_gemm:
                logger.warning(
                    "moe_deep_gemm is not supported when device capability < 9.0."
                )
                self.moe_deep_gemm = False

        self.moe_use_fusion_node = config.moe_use_fusion_node
        if self.expert_model_parallel_size > 1:
            if self.moe_token_dispatcher_type in ("deepep", "hybridep"):
                self.use_hybrid_ep_backend = is_hybrid_ep_backend_selected(
                    self.moe_token_dispatcher_type
                )
                if (
                    self.moe_use_fusion_node
                    and self.use_hybrid_ep_backend
                    and self.moe_shared_expert_overlap
                ):
                    logger.info(
                        "HybridEP backend does not support moe_shared_expert_overlap; disabling it."
                    )
                    self.moe_shared_expert_overlap = False
            elif self.moe_token_dispatcher_type == "allgather":
                self._validate_allgather_config()
            else:
                logger.info(
                    "moe_use_fusion_node is only supported when moe_token_dispatcher_type is 'deepep' or 'hybridep'; disabling it."
                )
                self.moe_use_fusion_node = False
                if self.moe_expert_fusion:
                    raise ValueError(
                        "moe_expert_fusion is only supported when moe_token_dispatcher_type is 'deepep' or 'hybridep' and on GPU architecture SM90 or higher. If these conditions are not met, please set it to false in the configuration yaml."
                    )
                self.fp8_dispatch = False

        if self.fp8:
            if paddle.version.cuda() == "12.6":
                raise NotImplementedError(
                    "fp8 is not supported when cuda version == 12.6."
                )
            assert self.moe_use_fusion_node, (
                "fp8 can only be used when moe_use_fusion_node = True."
            )

        if self.use_ue8m0:
            assert paddle.device.cuda.get_device_capability()[0] == 10, (
                "use_ue8m0 requires Blackwell GPU (SM100)"
            )

        expert_args = {}
        expert_args["config"] = routed_expert_config
        expert_args["moe_intermediate_size"] = self.moe_intermediate_size
        expert_args["is_expert"] = True
        expert_args["mlp_spec"] = self.moe_sublayers.mlp_spec

        use_fused_weight = self.moe_expert_fusion
        if (
            self.fp8
            and (self.moe_expert_fusion is False)
            and self.moe_deep_gemm
        ):
            raise ValueError(
                "For fp8 deep_gemm (i.e. use k-grouped gemm in backward), moe_expert_fusion must be True."
            )
        if (
            self.fp8
            and self.moe_expert_fusion
            and self.moe_deep_gemm is False
            and self.using_sonic_moe is False
        ):
            use_fused_weight = False
        if self.using_sonic_moe:
            assert use_fused_weight is True, (
                "for sonic moe, expert weight must be fused."
            )

        if use_fused_weight:
            if (
                self.moe_token_dispatcher_type == "allgather"
                and self.expert_model_parallel_size > 1
            ):
                # AllGather EP>1: every rank holds all experts, sharded
                # along intermediate dim (I // EP per rank).
                self.grouped_gemm_experts = SonicMoEExpert(
                    self.num_experts,
                    self.num_experts_per_tok,
                    routed_expert_config,
                    pg_collection,
                    intermediate_size_per_partition=(
                        self.moe_intermediate_size
                        // self.expert_model_parallel_size
                    ),
                )
            elif self.using_sonic_moe:
                # TODO: replace grouped_gemm_experts with fusion_experts
                self.grouped_gemm_experts = SonicMoEExpert(
                    self.num_local_experts,
                    self.num_experts_per_tok,
                    routed_expert_config,
                    pg_collection,
                )
            else:
                # TODO: replace grouped_gemm_experts with fusion_experts
                self.grouped_gemm_experts = GroupedMLPExpert(
                    self.num_local_experts,
                    routed_expert_config,
                    self.moe_deep_gemm,
                    pg_collection,
                )
        else:
            self.experts = nn.LayerList([])
            for i in range(self.num_experts):
                if i // self.num_experts_per_device == self.moe_rank:
                    self.experts.append(self.expert_class(**expert_args))
                else:
                    self.experts.append(None)

        shared_expert_args = deepcopy(expert_args)
        if self.config.gpt_model_use_experimental_version:
            shared_expert_args["is_expert"] = False
            shared_expert_args["config"] = shared_expert_config
        shared_expert_args["config"].use_bias = shared_expert_config.use_bias
        shared_expert_args["config"].hidden_size = self.config.hidden_size
        shared_expert_args["moe_intermediate_size"] = (
            self.moe_shared_expert_intermediate_size
        )
        shared_expert_args["is_expert"] = False
        if self.n_shared_experts > 0:
            self.shared_experts = self.shared_expert_class(**shared_expert_args)
        else:
            self.shared_experts = None

        # when sp is enabled, mark shared_experts as sequence parallel, because:
        # 1. shared_experts only process local tokens which shape is [s/tp,b,h]
        # 2. shared_experts'weight and bias will not be splited across tp ranks
        if (
            not self.config.gpt_model_use_experimental_version
            and self.sequence_parallel
            and self.expert_model_parallel_size > 1
            and self.shared_experts is not None
        ):
            mark_as_sequence_parallel_parameter(
                self.shared_experts.up_gate_proj.weight
            )
            if shared_expert_config.use_bias:
                mark_as_sequence_parallel_parameter(
                    self.shared_experts.up_gate_proj.bias
                )
            mark_as_sequence_parallel_parameter(
                self.shared_experts.down_proj.weight
            )
            if shared_expert_config.use_bias:
                mark_as_sequence_parallel_parameter(
                    self.shared_experts.down_proj.bias
                )

        if self.expert_model_parallel_size > 1:
            if self.moe_token_dispatcher_type in ("deepep", "hybridep"):
                # Set NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN automatically if not set by user.
                if (
                    self.moe_token_dispatcher_type == "hybridep"
                    and os.getenv("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN")
                    is None
                ):
                    # We limit the default domain size to 64 due to NVL72 topology. If user wants to use
                    # a larger domain size, they can set the environment variable manually.
                    num_of_hybrid_ep_ranks_per_nvlink_domain = min(
                        self.expert_model_parallel_size, 64
                    )
                    os.environ["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = (
                        str(num_of_hybrid_ep_ranks_per_nvlink_domain)
                    )
                    logger.info(
                        "Automatically set NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=%d for hybrid EP backend.",
                        num_of_hybrid_ep_ranks_per_nvlink_domain,
                    )
                self.token_dispatcher = MoEFlexTokenDispatcher(
                    self.num_experts_per_device,
                    self.num_experts_per_tok,
                    self.num_experts,
                    self.moe_group,
                    self.moe_ep_barrier,
                    dispatcher_type=self.moe_token_dispatcher_type,
                    hybridep_buffer_configs=getattr(
                        config, "hybridep_buffer_configs", None
                    ),
                    moe_deep_gemm=self.moe_deep_gemm,
                )
                if (
                    self.moe_token_dispatcher_type == "deepep"
                    and getattr(config, "deepep_buffer_configs", None)
                    is not None
                ):
                    configure_buffer(**config.deepep_buffer_configs)
            elif self.moe_token_dispatcher_type == "alltoall":
                local_expert_indices = list(
                    range(
                        self.moe_rank * self.num_experts_per_device,
                        (self.moe_rank + 1) * self.num_experts_per_device,
                    )
                )
                self.token_dispatcher = AllToAllTokenDispatcher(
                    self.moe_group,
                    self.expert_model_parallel_size,
                    self.num_experts_per_device,
                    local_expert_indices,
                )
            elif self.moe_token_dispatcher_type == "allgather":
                self.token_dispatcher = AllGatherTokenDispatcher(
                    self.moe_group,
                    self.expert_model_parallel_size,
                    self.num_experts,
                    fp8_dispatch=self.fp8_dispatch,
                    use_ue8m0=self.use_ue8m0,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported moe_token_dispatcher_type {self.moe_token_dispatcher_type}"
                )

        self.recompute_moe_gate_up = getattr(
            self.config, "recompute_moe_gate_up", False
        ) or (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "moe_gate_up" in self.config.recompute_modules
        )
        self.recompute_moe_premute = getattr(
            self.config, "recompute_moe_premute", False
        ) or (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "moe_premute" in self.config.recompute_modules
        )
        self.use_auto_subbatch = getattr(
            self.config, "use_auto_subbatch", False
        )
        self.moe_subbatch_diag = getattr(
            self.config, "moe_subbatch_diag", False
        )
        self.auto_subbatch_mode = getattr(
            self.config, "auto_subbatch_mode", None
        )

        if self.expert_model_parallel_size > 1:
            self.is_mp_moe = False
            self.is_ep_moe = True
            # Color routed-expert params with the default "moe_expert" now,
            # matching the historical construction-time behavior, UNLESS
            # mtp_shared_last_layer is enabled. In the shared-MTP case the color
            # (moe_expert vs the no-hook variant) depends on the layer number,
            # which is unknown here, so coloring is deferred to
            # set_layer_number()/_color_expert_params(). Paddle forbids
            # reassigning a non-None color, so coloring must happen exactly once.
            color_experts_now = not getattr(
                self.config, "mtp_shared_last_layer", False
            )
            fusion_experts = None
            if hasattr(self, "grouped_gemm_experts"):
                fusion_experts = self.grouped_gemm_experts
            if fusion_experts is not None:
                for p in fusion_experts.parameters():
                    p.is_moe_param = True
                    # Default color set here; deferred when mtp_shared_last_layer
                    # is on (see set_layer_number/_color_expert_params).
                    if color_experts_now:
                        p.color = {
                            "color": "moe_expert",
                            "group": self.moe_grad_group,
                        }
                    p.no_sync = not self.is_mp_moe
                    p.expert = not self.is_mp_moe
                    if self.is_mp_moe or self.is_ep_moe:
                        p.is_distributed = True
            else:
                assert self.experts is not None, (
                    "experts should be initialized."
                )
                for p in self.experts.parameters():
                    p.is_moe_param = True
                    # Default color set here; deferred when mtp_shared_last_layer
                    # is on (see set_layer_number/_color_expert_params).
                    if color_experts_now:
                        p.color = {
                            "color": "moe_expert",
                            "group": self.moe_grad_group,
                        }
                    p.no_sync = not self.is_mp_moe
                    p.expert = not self.is_mp_moe
                    if self.is_mp_moe or self.is_ep_moe:
                        p.is_distributed = True

        self.use_rr_deepep_combine = False

    def rr_recompute_update(self, in_full_recompute, in_mlp_recompute):
        if (
            self.config.recompute_modules is not None
            and "moe_combine" in self.config.recompute_modules
        ):
            if (
                self.moe_token_dispatcher_type != "deepep"
                or not self.moe_shared_expert_overlap
            ):
                raise ValueError(
                    "moe_combine RR is only supported in DeepEP mode with "
                    "moe_shared_expert_overlap enabled (combine_overlap scenario)."
                )
            if self.config.recompute_granularity is None:
                raise ValueError(
                    "recompute_granularity must be set when moe_combine RR is enabled."
                )
            if isinstance(self.config.recompute_modules, list):
                self.use_rr_deepep_combine = True
            elif isinstance(self.config.recompute_modules, dict):
                # dict mode only supports first_n: uniform applies recompute to all layers
                # (use list mode instead), block is not yet implemented but can be extended.
                if self.config.recompute_method != "first_n":
                    raise ValueError(
                        "recompute_modules dict mode for moe_combine RR requires "
                        f"recompute_method='first_n', got '{self.config.recompute_method}'."
                    )
                if not hasattr(self, "layer_number"):
                    raise ValueError(
                        "layer_number must be set before rr_recompute_update is called in dict mode. "
                        "Ensure set_layer_number() is called first."
                    )
                self.use_rr_deepep_combine = not need_recompute_in_first_n(
                    self.layer_number,
                    self.config,
                    self.config.recompute_modules["moe_combine"],
                )
        if (
            (not in_full_recompute)
            and (not in_mlp_recompute)
            and self.use_rr_deepep_combine
        ):
            raise ValueError(
                "Enabling rr for moe_combine is meaningless when neither full_recompute "
                "nor mlp_recompute is active."
            )

    def _init_expert_parallel(self):
        def _parse_moe_expert_parallel(
            num_experts: int, expert_model_parallel_size: int
        ) -> int:
            """
            Args:
                num_experts: Total number of experts
                expert_model_parallel_size: Expert parallel groups

            Returns:
                n_routed_experts_per_device: Number of experts per device
            """
            assert num_experts >= expert_model_parallel_size, (
                f"expert num_experts={num_experts} >= moe_world_size={expert_model_parallel_size}"
            )
            assert num_experts % expert_model_parallel_size == 0, (
                f"expert num_experts={num_experts} % moe_world_size={expert_model_parallel_size} == 0"
            )

            n_routed_experts_per_device = (
                num_experts // expert_model_parallel_size
            )
            return n_routed_experts_per_device

        if self.expert_model_parallel_size > 1:
            self.moe_grad_group = self.pg_collection.expt_dp
            self.moe_rank = utils.get_pg_rank(self.moe_group)
            self.moe_rank = max(self.moe_rank, 0)
            if self.moe_token_dispatcher_type == "allgather":
                # AllGather: every rank holds a shard of every expert.
                self.num_experts_per_device = self.num_experts
            else:
                self.num_experts_per_device = _parse_moe_expert_parallel(
                    self.num_experts, self.expert_model_parallel_size
                )
        else:
            self.moe_group = None
            self.moe_rank = 0
            self.expert_model_parallel_size = 1
            self.num_experts_per_device = self.num_experts

    def expert_forward(
        self,
        dispatched_input,
        tokens_per_expert,
    ):
        outputs = []
        tokens_per_expert = (
            tokens_per_expert.tolist()
            if not isinstance(tokens_per_expert, list)
            else tokens_per_expert
        )
        chunks = paddle.split(
            dispatched_input, num_or_sections=tokens_per_expert, axis=0
        )
        scale_chunks = None
        if use_accuracy_compatible_kernel():
            per_token_scale = getattr(
                self.token_dispatcher, "global_input_probs", None
            )
            if per_token_scale is None:
                raise RuntimeError(
                    "FLAGS_use_accuracy_compatible_kernel requires dispatched "
                    "router probabilities from the token dispatcher."
                )
            scale_chunks = paddle.split(
                per_token_scale, num_or_sections=tokens_per_expert, axis=0
            )
        for i, chunk in enumerate(chunks):
            if tokens_per_expert[i] == 0:
                continue
            chunk = chunk.contiguous()
            current_expert_idx = i + self.moe_rank * self.num_experts_per_device
            expert = self.experts[current_expert_idx]
            if scale_chunks is None:
                expert_output = expert(chunk)[0]
            else:
                expert_output = expert(chunk, per_token_scale=scale_chunks[i])[
                    0
                ]
            outputs += [expert_output]

        if not outputs:
            return dispatched_input

        return paddle.concat(outputs, axis=0)

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
        async_finish: bool = False,
    ):
        hidden_states = self.token_dispatcher.dispatch_preprocess(
            hidden_states, probs, routing_map, topk_weights, topk_indices
        )
        hidden_states, fp8_dispatched_handle = (
            self.token_dispatcher.token_dispatch(
                hidden_states,
                self.fp8_dispatch,
                async_finish=async_finish,
                use_ue8m0=self.use_ue8m0,
                using_sonic_moe=self.using_sonic_moe,
            )
        )
        return hidden_states, fp8_dispatched_handle

    def permute(self, hidden_states: paddle.Tensor):
        global_input_tokens, tokens_per_expert = (
            self.token_dispatcher.dispatch_postprocess(hidden_states)
        )
        return global_input_tokens, tokens_per_expert

    def unpermute(self, hidden_states: paddle.Tensor):
        return self.token_dispatcher.combine_preprocess(hidden_states)

    def combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        fp8_combine_grad_handle: dict | None = None,
    ):
        """Combine expert outputs back to the local token shard.

        For the 'allgather' and 'alltoall' dispatchers: ``token_combine``
        issues the reverse communication, then ``combine_postprocess``
        finalizes it.

        For other dispatchers (deepep / hybridep): delegates to
        ``_comm_manager.combine`` directly, which already returns the restored
        tensor (no separate combine_postprocess step).
        """
        if self.moe_token_dispatcher_type in ("allgather", "alltoall"):
            hidden_states = self.token_dispatcher.token_combine(
                hidden_states,
                combine_overlap_handle=combine_overlap_handle,
                async_finish=async_finish,
                fp8_combine_grad_handle=fp8_combine_grad_handle,
            )
            return self.token_dispatcher.combine_postprocess(hidden_states)
        return self.token_dispatcher._comm_manager.combine(
            hidden_states,
            combine_overlap_handle,
            use_rr_deepep_combine=self.use_rr_deepep_combine,
            fp8_dispatch=self.fp8_dispatch_bwd,
            combine_grad_handle=fp8_combine_grad_handle,
        )

    def routed_experts_compute(
        self,
        hidden_states: paddle.Tensor,
    ):
        global_input_tokens, tokens_per_expert = self.permute(hidden_states)
        expert_outs = self.expert_forward(
            global_input_tokens,
            tokens_per_expert,
        )
        return self.unpermute(expert_outs)

    def _maybe_pre_allgather_overlap(self, hidden_states: paddle.Tensor):
        """Pre-issue async AllGather on comm stream to overlap with gate MLP.

        allgather + EP>1 + moe_allgather_gate_overlap only. Result consumed
        in dispatch_preprocess. For latent MoE, fc1_latent_proj is hoisted
        here so AllGather targets latent-space tensor.
        """
        if (
            self.moe_token_dispatcher_type == "allgather"
            and self.expert_model_parallel_size > 1
            and self.moe_allgather_gate_overlap
        ):
            if self.use_latent_moe:
                self._latent_hidden = self.fc1_latent_proj(hidden_states)
                self.token_dispatcher.pre_allgather(self._latent_hidden)
            else:
                self._latent_hidden = None
                self.token_dispatcher.pre_allgather(hidden_states)
        else:
            self._latent_hidden = None

    def _validate_allgather_config(self):
        """Validate and force-correct config flags for the allgather dispatcher.

        AllGather + ReduceScatter EP pattern: every expert is sharded along its
        intermediate dim across the EP group.  Requires SonicMoE fused kernels;
        fp8 dispatch quantization is handled by ``AllGatherTokenDispatcher``
        (see ``_quantize_and_pack_fp8``) and fp8 expert compute by
        ``run_sonic_moe``.
        """
        if not self.using_sonic_moe:
            raise ValueError(
                "moe_token_dispatcher_type='allgather' requires "
                "using_sonic_moe=True; the allgather path is only "
                "implemented for SonicMoE fused kernels."
            )
        if not self.moe_use_fusion_node:
            logger.warning(
                "moe_token_dispatcher_type='allgather' only "
                "support moe_use_fusion_node; forcing moe_use_fusion_node=True."
            )
            self.moe_use_fusion_node = True
        if not self.moe_expert_fusion:
            logger.warning(
                "moe_token_dispatcher_type='allgather' requires "
                "fused expert weights; forcing moe_expert_fusion=True."
            )
            self.moe_expert_fusion = True
        if self.moe_deep_gemm:
            logger.warning(
                "moe_token_dispatcher_type='allgather' does not "
                "support moe_deep_gemm; forcing moe_deep_gemm=False."
            )
            self.moe_deep_gemm = False
        if self.moe_intermediate_size % self.expert_model_parallel_size != 0:
            raise ValueError(
                f"moe_intermediate_size={self.moe_intermediate_size} "
                f"must be divisible by EP="
                f"{self.expert_model_parallel_size} in 'allgather' mode."
            )
        if self.fp8:
            intermediate_per_rank = (
                self.moe_intermediate_size // self.expert_model_parallel_size
            )
            if intermediate_per_rank % 128 != 0:
                raise ValueError(
                    f"allgather + fp8 requires "
                    f"moe_intermediate_size / EP to be divisible by 128 "
                    f"(fp8 block-scale tile), got "
                    f"moe_intermediate_size={self.moe_intermediate_size}, "
                    f"EP={self.expert_model_parallel_size}, "
                    f"intermediate_per_rank={intermediate_per_rank}. "
                    f"Consider reducing EP to a divisor of "
                    f"moe_intermediate_size // 128 = "
                    f"{self.moe_intermediate_size // 128}."
                )

    def _project_to_latent(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """Project hidden_states to latent space, consuming any cached
        projection from the AllGather overlap path if available.
        """
        if not self.use_latent_moe:
            return hidden_states
        if self._latent_hidden is not None:
            hidden_states = self._latent_hidden
            self._latent_hidden = None
        else:
            hidden_states = self.fc1_latent_proj(hidden_states)
        return hidden_states

    # MoE forward: dispatch -> permute -> compute ->unpermute -> combine
    def custom_forward(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        # Latent MoE: project hidden_states to latent space before dispatch
        if self.use_latent_moe:
            hidden_states = self.fc1_latent_proj(hidden_states)

        should_log_balance = framework._dygraph_tracer()._has_grad
        with profile("dispatch"):
            hidden_states, _ = self.dispatch(
                hidden_states, probs, routing_map, topk_weights, topk_indices
            )
        if should_log_balance and global_moe_balance_training_logs_enabled():
            log_moe_balance(
                self.layer_number,
                self.moe_group,
                self.num_experts_per_tok,
                self.token_dispatcher.get_dispatched_routing()[2],
            )
        with profile("fusion_mlp"):
            hidden_states = self.routed_experts_compute(hidden_states)
        with profile("combine"):
            hidden_states = self.combine(hidden_states)

        # Latent MoE: project back from latent space to hidden_size
        if self.use_latent_moe:
            hidden_states = self.fc2_latent_proj(hidden_states)

        return hidden_states

    def fusion_moe_forward(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        combine_overlap_handle: dict,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        hidden_states = self._project_to_latent(hidden_states)

        should_log_balance = framework._dygraph_tracer()._has_grad
        with profile("dispatch"):
            dispatched_hidden_states, fp8_dispatched_handle = self.dispatch(
                hidden_states, probs, routing_map, topk_weights, topk_indices
            )

        dispatched_indices, dispatched_probs, tokens_per_expert = (
            self.token_dispatcher.get_dispatched_routing()
        )
        if should_log_balance and global_moe_balance_training_logs_enabled():
            log_moe_balance(
                self.layer_number,
                self.moe_group,
                self.num_experts_per_tok,
                tokens_per_expert,
            )
        fp8_combine_grad_handle = {} if self.fp8_dispatch_bwd else None
        # fp8_combine_grad_handle = None

        with profile("fusion_mlp"):
            if self._use_hybrid_ep_fusion():
                hidden_states = self._run_hybrid_ep_fusion(
                    dispatched_hidden_states,
                    dispatched_probs,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                )
            elif self.using_sonic_moe:
                use_fp8 = self.fp8 is not None
                fp8_scale = None
                if fp8_dispatched_handle is not None:
                    fp8_scale = fp8_dispatched_handle["scale"]
                hidden_states = self.grouped_gemm_experts(
                    dispatched_hidden_states,
                    dispatched_indices,
                    dispatched_probs,
                    use_fp8,
                    tokens_per_expert=tokens_per_expert,
                    fp8_scale=fp8_scale,
                    fp8_combine_grad_handle=fp8_combine_grad_handle,
                )
            else:
                hidden_states = FusionMoePyLayer.apply(
                    dispatched_hidden_states,
                    dispatched_probs,
                    dispatched_indices,
                    self,
                    self.num_experts_per_tok,
                    use_fp8_mlp=self.fp8,
                    moe_deep_gemm=self.moe_deep_gemm,
                    recompute_moe_gate_up=self.recompute_moe_gate_up,
                    recompute_moe_premute=self.recompute_moe_premute,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                    use_bf16_gemm_weight_grad=not self.fp8_wgrad,
                    use_auto_subbatch=self.use_auto_subbatch,
                    auto_subbatch_mode=self.auto_subbatch_mode,
                    moe_expert_fusion=self.moe_expert_fusion,
                    moe_subbatch_token_num_after_dispatch=self.moe_subbatch_token_num_after_dispatch,
                    moe_subbatch_diag=self.moe_subbatch_diag,
                    use_ue8m0=self.use_ue8m0,
                    dw_p2p_overlap=self.dw_p2p_overlap,
                    clamp_value=self.config.activation_func_clamp_value,
                    is_first_fwd=not framework._dygraph_tracer()._has_grad,
                    use_accuracy_compatible=getattr(
                        self.config, "use_accuracy_compatible", False
                    ),
                )

        with profile("combine"):
            hidden_states = self.combine(
                hidden_states,
                combine_overlap_handle=combine_overlap_handle,
                fp8_combine_grad_handle=fp8_combine_grad_handle,
            )

        # Latent MoE: project back from latent space to hidden_size
        if self.use_latent_moe:
            hidden_states = self.fc2_latent_proj(hidden_states)

        return hidden_states

    def compute_gate(
        self, hidden_states, input_ids=None, origin_input_ids=None
    ):
        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
        return self.gate(
            hidden_states,
            input_ids=input_ids,
            origin_input_ids=origin_input_ids,
        )

    def _use_hybrid_ep_fusion(self):
        return self.moe_use_fusion_node and self.use_hybrid_ep_backend

    def _run_hybrid_ep_fusion(
        self,
        dispatched_hidden_states,
        dispatched_probs,
        fp8_dispatched_handle=None,
        is_first_fwd=False,
    ):
        dispatched_hidden_states.stop_gradient = False
        dispatched_probs.stop_gradient = False
        return HybridEPMoePyLayer.apply(
            dispatched_hidden_states,
            dispatched_probs,
            self,
            use_fp8_mlp=self.fp8,
            moe_deep_gemm=self.moe_deep_gemm,
            moe_expert_fusion=self.moe_expert_fusion,
            use_ue8m0=self.use_ue8m0,
            recompute_moe_gate_up=self.recompute_moe_gate_up,
            use_bf16_gemm_weight_grad=not self.fp8_wgrad,
            fp8_dispatched_handle=fp8_dispatched_handle,
            is_first_fwd=is_first_fwd,
            dw_p2p_overlap=self.dw_p2p_overlap,
            clamp_value=self.config.activation_func_clamp_value,
            use_accuracy_compatible=getattr(
                self.config, "use_accuracy_compatible", False
            ),
        )

    def dispatch_preprocess(self, args):
        hidden_states, token_probs, token_indices = args
        if self.use_latent_moe:
            hidden_states = self.fc1_latent_proj(hidden_states)
        assert isinstance(self.token_dispatcher, MoEFlexTokenDispatcher)
        hidden_states = self.token_dispatcher.dispatch_preprocess_overlap(
            hidden_states, token_probs, token_indices
        )
        token_probs = self.token_dispatcher._comm_manager.token_probs
        token_indices = self.token_dispatcher._comm_manager.token_indices
        return hidden_states, token_indices, token_probs

    def compute_dispatch(self, args, async_finish=False):
        hidden_states, token_indices, token_weights = args
        if self.moe_use_fusion_node:
            dispatched_hidden_states, fp8_dispatched_handle = (
                self.token_dispatcher.token_dispatch_overlap(
                    hidden_states,
                    token_indices,
                    token_weights,
                    self.fp8_dispatch,
                    async_finish=async_finish,
                    use_ue8m0=self.use_ue8m0,
                )
            )
            dispatched_probs = (
                self.token_dispatcher._comm_manager.dispatched_probs
            )
            # NOTE: tokens_per_expert_list is stateful and should be saved for recompute.
            tokens_per_expert = (
                self.token_dispatcher._comm_manager.tokens_per_expert
            )
            # dispatched_hidden_states's dtype is fp8, but its gradient's dtype is bf16, so type separation is required; the actual values are passed via a dictionary.
            dispatched_hidden_states, guard_status = GradDtypeGuard.apply(
                dispatched_hidden_states, hidden_states.dtype
            )
            guard_status["x"].stop_gradient = True
            dispatched_indices = None
            if not self._use_hybrid_ep_fusion():
                dispatched_indices = (
                    self.token_dispatcher._comm_manager.dispatched_indices
                )
            return (
                dispatched_hidden_states,
                dispatched_indices,
                dispatched_probs,
                fp8_dispatched_handle,
                tokens_per_expert,
                guard_status,
            )

    def compute_experts(self, args, is_first_fwd=False):
        if self.moe_use_fusion_node:
            (
                dispatched_hidden_states,
                dispatched_indices,
                dispatched_probs,
                fp8_dispatched_handle,
                tokens_per_expert,
                guard_status,
            ) = args
            self.token_dispatcher._comm_manager.tokens_per_expert = (
                tokens_per_expert
            )
            dispatched_hidden_states = GradDtypeUnguard.apply(
                dispatched_hidden_states, guard_status
            )

            if self._use_hybrid_ep_fusion():
                hidden_states = self._run_hybrid_ep_fusion(
                    dispatched_hidden_states,
                    dispatched_probs,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                    is_first_fwd=is_first_fwd,
                )
            else:
                hidden_states = FusionMoePyLayer.apply(
                    dispatched_hidden_states,
                    dispatched_probs,
                    dispatched_indices.clone()
                    if is_first_fwd
                    else dispatched_indices,
                    self,
                    self.num_experts_per_tok,
                    use_fp8_mlp=self.fp8,
                    moe_deep_gemm=self.moe_deep_gemm,
                    recompute_moe_gate_up=self.recompute_moe_gate_up,
                    recompute_moe_premute=self.recompute_moe_premute,
                    fp8_dispatched_handle=fp8_dispatched_handle,
                    use_bf16_gemm_weight_grad=not self.fp8_wgrad,
                    use_auto_subbatch=self.use_auto_subbatch,
                    auto_subbatch_mode=self.auto_subbatch_mode,
                    moe_expert_fusion=self.moe_expert_fusion,
                    moe_subbatch_token_num_after_dispatch=self.moe_subbatch_token_num_after_dispatch,
                    moe_subbatch_diag=self.moe_subbatch_diag,
                    use_ue8m0=self.use_ue8m0,
                    dw_p2p_overlap=self.dw_p2p_overlap,
                    clamp_value=self.config.activation_func_clamp_value,
                    use_accuracy_compatible=getattr(
                        self.config, "use_accuracy_compatible", False
                    ),
                )

            if is_first_fwd:
                hidden_states.stop_gradient = False
        else:
            hidden_states, topk_weights = args
            hidden_states = self.routed_experts_compute(hidden_states)
        return hidden_states

    def compute_combine(self, hidden_states, async_finish=False):
        # Note: RR (use_rr_deepep_combine) is NOT passed here because this method
        # is used by TransformerLayerWithOverlap where shared expert computation is
        # managed by the scheduler separately, not via combine_overlap_handle.
        if self.moe_use_fusion_node:
            hidden_states = self.token_dispatcher._comm_manager.combine(
                hidden_states,
                None,
                async_finish=async_finish,
            )
        else:
            hidden_states = self.combine(hidden_states)
        return hidden_states

    def aux_loss_compute(self, args):
        hidden_states, aux_loss, z_loss, residuals = args
        if self.use_latent_moe:
            hidden_states = self.fc2_latent_proj(hidden_states)
        if self.training and self.router_aux_loss_coef and aux_loss is not None:
            aux_loss = aux_loss * float(self.router_aux_loss_coef)
            output = AddAuxiliaryLoss.apply(hidden_states, aux_loss)
        else:
            output = hidden_states
        if self.training and z_loss is not None:
            output = AddAuxiliaryLoss.apply(output, z_loss)
        output = output.reshape(residuals.shape)
        if self.shared_experts is not None:
            shared_output = self.shared_experts(residuals)[0]
            output = output + shared_output

        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            output = ScatterOp.apply(output)
        return output

    # ------------------------------------------------------------------
    # Overridable hooks (Template Method pattern)
    # Subclasses (e.g. Gemma4MoELayer) can override these to customize
    # gate/expert input transformation and output post-processing without
    # rewriting the full forward logic.
    # ------------------------------------------------------------------

    def _prepare_gate_input(self, hidden_states, residual):
        """Return the tensor fed into the router. Default: hidden_states."""
        return hidden_states

    def _prepare_expert_input(self, hidden_states, residual):
        """Return the tensor fed into routed experts. Default: hidden_states."""
        return hidden_states

    def _post_routed_output(self, output):
        """Post-process routed expert output before combining with shared. Default: identity."""
        return output

    def _post_shared_output(self, shared_output):
        """Post-process shared expert output before combining. Default: identity."""
        return shared_output

    def forward(
        self,
        hidden_states: paddle.Tensor,
        input_ids: paddle.Tensor | None = None,
        residual: paddle.Tensor | None = None,
        origin_input_ids: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        """
        Args:
            hidden_states: Shape: [batch_size, seq_len, hidden_size]
            input_ids: Shape: [batch_size, seq_len], optional token ids from embedding input.
            residual: Shape: [batch_size, seq_len, hidden_size], optional separate residual
                      for routing/expert input (used by Gemma4 dual-branch topology).
            origin_input_ids: Shape: [batch_size, seq_len + num_mtp_layers], optional original input_ids.
                Only passed when gpt_model_use_experimental_version is True.

        Returns:
            output: Shape: [batch_size, seq_len, hidden_size]
        """
        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
            if residual is not None:
                residual = GatherOp.apply(residual)

        orig_shape = hidden_states.shape
        residuals = hidden_states

        layer_idx = getattr(self, "layer_number", None)
        _log_moe_md5(hidden_states, "moe_input", layer_idx)

        self._maybe_pre_allgather_overlap(hidden_states)
        gate_input = self._prepare_gate_input(hidden_states, residual)

        (
            capacity,
            topk_weights,
            topk_indices,
            probs,
            mask,
            priorities,
            aux_loss,
            z_loss,
        ) = self.gate(
            gate_input,
            input_ids=input_ids,
            origin_input_ids=origin_input_ids,
        )
        # topk_weights, topk_indices: Shape is [seq_len, moe_router_topk]
        # probs: combine weights in [S, E] sparse layout (non-selected positions are 0) [seq_len, num_experts]
        # mask (routing_map): binary selection matrix [seq_len, num_experts]
        # capacity, priorities are used for dropping tokens, currently they are not used

        _log_moe_md5(probs, "probs", layer_idx)
        _log_moe_md5(mask, "routing_mask", layer_idx)
        if framework._dygraph_tracer()._has_grad:
            log_moe_losses(layer_idx, aux_loss=aux_loss, z_loss=z_loss)

        if (
            self.shared_experts is not None
            and self.moe_shared_expert_overlap
            and self.moe_use_fusion_node
            and self.expert_model_parallel_size > 1
        ):
            combine_overlap_handle = {
                "fn": self.shared_experts,
                "fn_args": (residuals,),
            }
        else:
            combine_overlap_handle = None

        expert_input = self._prepare_expert_input(hidden_states, residual)
        if self.expert_model_parallel_size > 1:
            if self.moe_use_fusion_node:
                output = self.fusion_moe_forward(
                    expert_input,
                    probs,
                    mask,
                    combine_overlap_handle,
                    topk_weights=topk_weights,
                    topk_indices=topk_indices,
                )
            else:
                output = self.custom_forward(
                    expert_input,
                    probs,
                    mask,
                    topk_weights=topk_weights,
                    topk_indices=topk_indices,
                )
        else:
            if len(expert_input.shape) == 3:
                batch_size, seq_len, d_model = expert_input.shape
                reshaped_input = expert_input.reshape([-1, d_model])
            else:
                reshaped_input = expert_input
            # Latent MoE: project to latent space before single-card MoE
            if self.use_latent_moe:
                reshaped_input = self.fc1_latent_proj(reshaped_input)
            if self.moe_expert_fusion:
                output = self._forward_single_card_grouped_gemm_moe(
                    reshaped_input, mask, probs, topk_indices, topk_weights
                )
            else:
                output = self._forward_single_card_moe(
                    reshaped_input, topk_indices, topk_weights
                )
            # Latent MoE: project back from latent space
            if self.use_latent_moe:
                output = self.fc2_latent_proj(output)

        _log_moe_md5(output, "moe_routed_output", layer_idx)

        if self.training and self.router_aux_loss_coef and aux_loss is not None:
            aux_loss = aux_loss * float(self.router_aux_loss_coef)
            output = AddAuxiliaryLoss.apply(output, aux_loss)

        if self.training and z_loss is not None:
            output = AddAuxiliaryLoss.apply(output, z_loss)

        output = output.reshape(orig_shape)
        output = self._post_routed_output(output)

        if self.shared_experts is not None:
            if combine_overlap_handle is not None:
                shared_output = combine_overlap_handle["fn_out"][0]
            else:
                shared_output = self.shared_experts(residuals)[0]
            shared_output = self._post_shared_output(shared_output)
            output = output + shared_output

        _log_moe_md5(output, "moe_final_output", layer_idx)

        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            output = ScatterOp.apply(output)
        return output, None  # None is bias

    def _forward_single_card_moe(
        self,
        hidden_states: paddle.Tensor,
        selected_experts: paddle.Tensor,
        topk_weights: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Forward without expert parallelism

        Args:
            hidden_states: Input hidden states, shape: [batch_size*seq_len, hidden_size]
            selected_experts: TopK experts indices, shape: [seq_len, num_experts_per_tok]
            topk_weights: TopK weights, shape: [seq_len, num_experts_per_tok]

        Returns:
            output: Output hidden states, shape: [seq_len, hidden_size]
        """

        _, d_model = hidden_states.shape
        final_hidden_states = paddle.zeros_like(
            hidden_states, dtype=hidden_states.dtype
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = paddle.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).transpose([2, 1, 0])
        tokens_per_expert = expert_mask.reshape([expert_mask.shape[0], -1]).sum(
            axis=-1
        )
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, idx = paddle.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            if tokens_per_expert[expert_idx] <= 0.1:
                continue
            current_state = hidden_states[idx, None].reshape([-1, d_model])
            expert_out = expert_layer(current_state)[0]
            current_weight = topk_weights[idx, top_x].unsqueeze(-1)
            current_hidden_states = expert_out * current_weight

            # use scatter to replace index_add
            final_hidden_states_tmp = paddle.zeros_like(final_hidden_states)
            final_hidden_states_tmp = paddle.scatter(
                final_hidden_states_tmp,
                idx.reshape([-1]),
                current_hidden_states.to(hidden_states.dtype),
                overwrite=False,
            )
            final_hidden_states = final_hidden_states + final_hidden_states_tmp
        return final_hidden_states.cast(hidden_states.dtype)

    def _forward_single_card_grouped_gemm_moe(
        self,
        hidden_states: paddle.Tensor,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_indices: paddle.Tensor | None = None,
        topk_weights: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        """
        Forward without expert parallelism

        Args:
            hidden_states: Input hidden states, shape: [batch_size*seq_len, hidden_size]
            routing_map: Routing map, shape: [seq_len, num_experts]
            probs: Probabilities of selecting each expert, shape: [seq_len, num_experts]

        Returns:
            output: Output hidden states, shape: [seq_len, hidden_size]
        """

        def _convert_routing_map_and_probs(
            routing_map: paddle.Tensor, probs: paddle.Tensor, topk: int
        ):
            routing_map = routing_map.astype("bool")
            masked_probs = probs * routing_map.astype("float32")
            weights, indices = paddle.topk(masked_probs, k=topk, axis=-1)
            return indices, weights

        if self.using_sonic_moe:
            use_fp8 = self.fp8 is not None
            final_hidden_states = self.grouped_gemm_experts(
                hidden_states,
                topk_indices,
                topk_weights,
                use_fp8,
            )
            return final_hidden_states.cast(hidden_states.dtype)
        else:
            tokens_per_expert = routing_map.sum(axis=0)
            permuted_local_hidden_states, sorted_indices = permute(
                hidden_states, routing_map, tokens_per_expert
            )
            grouped_expert_out = self.grouped_gemm_experts(
                permuted_local_hidden_states, tokens_per_expert
            )[0]
            final_hidden_states = unpermute(
                grouped_expert_out,
                sorted_indices,
                restore_shape=hidden_states.shape,
                probs=probs,
                routing_map=routing_map,
            )
            return final_hidden_states.cast(hidden_states.dtype)

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        if not (self.moe_use_fusion_node and self.fp8):
            return
        if hasattr(self, "grouped_gemm_experts") and isinstance(
            self.grouped_gemm_experts, SonicMoEExpert
        ):
            self.grouped_gemm_experts.quant_weight()
            return

        def quantize_weights(
            weight_list, weight_obj=None, quant_transpose=None
        ):
            """Helper function to quantize a list of weights."""
            if weight_obj is None:
                weight_obj = weight_list[0]

            # 始终量化非转置版（行为对齐，fp8_weight_stacked 始终存在）
            fp8_weight, fp8_scale = fused_stack_quant_without_cache(
                weight_list, transpose=False, use_ue8m0=self.use_ue8m0
            )
            weight_obj.fp8_weight_stacked = fp8_weight
            weight_obj.fp8_scale_stacked = fp8_scale

            if quant_transpose is None or quant_transpose is True:
                fp8_weight_t, fp8_scale_t = fused_stack_quant_without_cache(
                    weight_list, transpose=True, use_ue8m0=self.use_ue8m0
                )
                weight_obj.fp8_weight_stacked_transpose = fp8_weight_t
                weight_obj.fp8_scale_stacked_transpose = fp8_scale_t
            else:
                weight_obj.fp8_weight_stacked_transpose = None
                weight_obj.fp8_scale_stacked_transpose = None
                if self.use_ue8m0:
                    from paddleformers.fleet.triton_ops import (
                        fuse_stack_ue8m0_scale_transpose,
                    )

                    converted_scale = fuse_stack_ue8m0_scale_transpose(
                        fp8_scale,
                        len(weight_list),
                        weight_list[0].shape[0],
                        weight_list[0].shape[1],
                    )
                    weight_obj.fp8_scale_stacked_transpose = converted_scale

        if hasattr(self, "grouped_gemm_experts"):
            if batch_mode:
                expert_w1 = self.grouped_gemm_experts.weight1
                expert_w2 = self.grouped_gemm_experts.weight2
                local_expert_num = expert_w1.shape[0]
                expert_w1_list = [
                    expert_w1[i, :, :] for i in range(local_expert_num)
                ]
                expert_w2_list = [
                    expert_w2[i, :, :] for i in range(local_expert_num)
                ]

                # Batch mode: process all experts' weights together
                if expert_w1_list:
                    quantize_weights(
                        expert_w1_list,
                        self.grouped_gemm_experts.weight1,
                        quant_transpose,
                    )
                if expert_w2_list:
                    quantize_weights(
                        expert_w2_list,
                        self.grouped_gemm_experts.weight2,
                        quant_transpose,
                    )

            else:
                raise NotImplementedError(
                    "Not support individual mode for fuse_expert_fp8_weight_quant yet."
                )

            return

        if batch_mode:
            # Batch mode: process all experts' weights together
            expert_w1_list = [
                expert.up_gate_proj.weight
                for expert in self.experts
                if expert is not None
            ]
            expert_w2_list = [
                expert.down_proj.weight
                for expert in self.experts
                if expert is not None
            ]
            if expert_w1_list:
                quantize_weights(
                    expert_w1_list, expert_w1_list[0], quant_transpose
                )
            if expert_w2_list:
                quantize_weights(
                    expert_w2_list, expert_w2_list[0], quant_transpose
                )

        else:
            # Individual mode: process each expert's weights separately
            for expert in self.experts:
                if expert is not None:
                    quantize_weights(
                        [expert.up_gate_proj.weight],
                        quant_transpose=quant_transpose,
                    )
                    quantize_weights(
                        [expert.down_proj.weight],
                        quant_transpose=quant_transpose,
                    )

    def clear_fp8_quant_weight(self):
        """Clear cached FP8 quantized weights to release memory."""

        logger.info(
            "Clearing FP8 quantized weights in MoE layer: "
            "[fp8_weight_stacked, fp8_scale_stacked, "
            "fp8_weight_stacked_transpose, fp8_scale_stacked_transpose]"
        )

        if not (self.moe_use_fusion_node and self.fp8):
            return

        fp8_attrs = (
            "fp8_weight_stacked",
            "fp8_scale_stacked",
            "fp8_weight_stacked_transpose",
            "fp8_scale_stacked_transpose",
        )

        def _clear_attrs(weight_obj):
            for attr in fp8_attrs:
                if hasattr(weight_obj, attr):
                    delattr(weight_obj, attr)

        if hasattr(self, "grouped_gemm_experts"):
            _clear_attrs(self.grouped_gemm_experts.weight1)
            _clear_attrs(self.grouped_gemm_experts.weight2)
        else:
            for expert in self.experts:
                if expert is not None:
                    _clear_attrs(expert.up_gate_proj.weight)
                    _clear_attrs(expert.down_proj.weight)

    def use_fp8(self):
        if self.moe_use_fusion_node and self.fp8:
            return True
        return False

    def set_layer_number(self, layer_number, is_mtp_layer: bool = False):
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        # Assign routed-expert 'color' now that the layer number is known. This
        # is the single place color is set for experts (Paddle forbids
        # reassigning it): the MTP-shared last layer uses the no-hook color.
        self._color_expert_params()
        assert hasattr(self.gate, "set_layer_number"), (
            "expect gate has method 'set_layer_number'"
        )
        # Hash routing activation (moe_n_hash_layers) is decided by the router
        # itself based on layer_number. See TopKRouter._setup_hash_layer.
        self.gate.set_layer_number(layer_number, is_mtp_layer=is_mtp_layer)

    def _color_expert_params(self):
        """Set the sharding 'color' on routed-expert params (called once).

        Only needed when ``mtp_shared_last_layer`` is enabled: in that case the
        expert params were intentionally left uncolored at construction (the
        moe_expert vs no-hook choice depends on the layer number). Picks
        ``moe_weight_no_hook`` for the MTP-shared backbone last layer so the
        sharding-stage1 optimizer reduces those shared params synchronously (no
        overlap hook); otherwise the normal ``moe_expert`` color is used.

        Params already colored at construction (the common, non-shared-MTP case)
        are skipped: Paddle forbids reassigning a non-None color, and their
        color would be ``moe_expert`` either way.
        """
        if self.expert_model_parallel_size <= 1:
            return
        # Lazy import to avoid a circular import: transformer_layer imports
        # MoELayer from this module.
        from paddleformers.fleet.transformer.transformer_layer import (
            is_mtp_shared_last_layer,
        )

        fusion_experts = getattr(self, "grouped_gemm_experts", None)
        if fusion_experts is not None:
            expert_params = fusion_experts.parameters()
        else:
            assert self.experts is not None, "experts should be initialized."
            expert_params = self.experts.parameters()
        color_key = (
            "moe_weight_no_hook"
            if is_mtp_shared_last_layer(
                self.config, self.layer_number, self.is_mtp_layer
            )
            else "moe_expert"
        )
        for p in expert_params:
            # Skip params already colored at construction (the non-shared-MTP
            # case); Paddle forbids reassigning a non-None color. Uncolored
            # params carry no color attribute (None) or the -1 sentinel.
            color = getattr(p, "color", None)
            if color not in (None, -1):
                continue
            p.color = {"color": color_key, "group": self.moe_grad_group}


class Gemma4TopKRouter(TopKRouter):
    """Gemma4 MoE router aligned with ms-swift/HF Gemma4TextRouter.

    Reuses TopKRouter for padding mask, SP/CP, aux/z loss handling.
    Only adds:
    1. Input normalization: scaleless RMSNorm + learned scale * (1/sqrt(d))
    2. Config overrides: softmax scoring, norm_topk_prob, learnable per-expert scale
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        # Use a shallow copy to avoid polluting the shared config object.
        import copy

        config = copy.copy(config)
        # Configure TopKRouter to match Gemma4 behavior
        config.scoring_func = "softmax"
        config.norm_topk_prob = True
        config.topk_method = "greedy"
        config.routed_scaling_factor_learnable = True
        config.routed_scaling_factor = 1.0
        config.router_aux_loss_coef = 0.0
        config.router_z_loss_coef = 0.0
        # Greedy topk is incompatible with moe_topk_fusion (requires e_score_correction_bias
        # which is only created for topk_method == "noaux_tc").
        config.moe_topk_fusion = False
        super().__init__(config, pg_collection)

        # Gemma4-specific: input normalization scale (learnable, aligned with HF nn.Parameter)
        hidden_size = config.hidden_size
        self.router_input_scale = paddle.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self._inv_sqrt_d = hidden_size**-0.5

    def _normalize_input(self, hidden_states):
        """Scaleless RMSNorm + learned scale."""
        h = hidden_states.cast("float32")
        rms = (h.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        h = (h / rms).cast(hidden_states.dtype)
        return h * self.router_input_scale * self._inv_sqrt_d

    def forward(self, input, input_ids=None, origin_input_ids=None):
        """Normalize input, then delegate to TopKRouter for full routing logic."""
        normalized_input = self._normalize_input(input)
        return super().forward(
            normalized_input,
            input_ids=input_ids,
            origin_input_ids=origin_input_ids,
        )


class Gemma4MoELayer(MoELayer):
    """Gemma4 MoE via base-class hooks (no forward override).

    Customizations over base MoELayer:
      - Gate: Gemma4TopKRouter (internal RMS norm + per_expert_scale)
      - Activation: GeGLU (gelu_tanh(gate) * up)
      - Dual-branch topology via hooks:
        * _prepare_gate_input  → route on residual
        * _prepare_expert_input → pre_feedforward_layernorm_2(residual)
        * _post_routed_output  → post_moe_layernorm
        * _post_shared_output  → post_shared_expert_layernorm

    Norms (aligned with HF naming):
      - post_shared_expert_layernorm (= HF post_feedforward_layernorm_1)
      - pre_feedforward_layernorm_2 (= HF pre_feedforward_layernorm_2)
      - post_moe_layernorm (= HF post_feedforward_layernorm_2)
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers: MoESublayers | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        if (
            not hasattr(config, "n_shared_experts")
            or config.n_shared_experts is None
        ):
            config.n_shared_experts = 1
        super().__init__(config, sublayers, pg_collection)

        self.gate = Gemma4TopKRouter(config=config, pg_collection=pg_collection)

        shared_size = getattr(
            config, "moe_shared_expert_intermediate_size", None
        )
        if shared_size and self.shared_experts is not None:
            shared_expert_config = deepcopy(config)
            shared_expert_config.intermediate_size = shared_size
            self.shared_experts = StandardMLPSharedExpert(
                config=shared_expert_config,
                moe_intermediate_size=shared_size,
                is_expert=False,
                mlp_spec=self.moe_sublayers.mlp_spec,
            )

        self._activation_type = "geglu"

        if (
            hasattr(self, "grouped_gemm_experts")
            and self.grouped_gemm_experts is not None
        ):
            gelu_tanh = functools.partial(F.gelu, approximate=True)

            def _gemma4_glu(x):
                x = paddle.chunk(x, 2, dim=-1)
                return gelu_tanh(x[0]) * x[1]

            self.grouped_gemm_experts.activation_func = _gemma4_glu
            self.grouped_gemm_experts.config.hidden_act = gelu_tanh

        from paddleformers.fleet.transformer.paddle_norm import RMSNorm

        self.post_shared_expert_layernorm = RMSNorm(config)
        self.pre_feedforward_layernorm_2 = RMSNorm(config)
        self.post_moe_layernorm = RMSNorm(config)

        if (
            hasattr(self, "grouped_gemm_experts")
            and self.grouped_gemm_experts is not None
        ):
            import types

            from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
                build_sharded_state_dict,
                shard_weight,
            )

            grouped = self.grouped_gemm_experts

            def _gemma4_grouped_sharded_state_dict(
                self_inner, structured_name_prefix=""
            ):
                state_dict = self_inner.state_dict(structured_name_prefix="")
                sharded_dict = {}
                full_key1 = f"{structured_name_prefix}weight1"
                full_key2 = f"{structured_name_prefix}weight2"
                if self_inner.ep_group is None:
                    sharded_dict = build_sharded_state_dict(
                        state_dict, None, structured_name_prefix
                    )
                else:
                    sharded_dict[full_key1] = shard_weight(
                        key=full_key1,
                        weight=state_dict["weight1"],
                        axis=0,
                        group=self_inner.ep_group,
                    )
                    sharded_dict[full_key1].grouped_gemm_param = True
                    sharded_dict[full_key2] = shard_weight(
                        key=full_key2,
                        weight=state_dict["weight2"],
                        axis=0,
                        group=self_inner.ep_group,
                    )
                    sharded_dict[full_key2].grouped_gemm_param = True
                return sharded_dict

            grouped.sharded_state_dict = types.MethodType(
                _gemma4_grouped_sharded_state_dict, grouped
            )

    # ------------------------------------------------------------------
    # Hook overrides: dual-branch topology (shared from hidden_states,
    # routed from residual with extra norms)
    # ------------------------------------------------------------------

    def _prepare_gate_input(self, hidden_states, residual):
        """Route on residual (Gemma4TopKRouter applies internal normalization)."""
        return residual if residual is not None else hidden_states

    def _prepare_expert_input(self, hidden_states, residual):
        """Apply pre_feedforward_layernorm_2 to residual before expert compute."""
        src = residual if residual is not None else hidden_states
        return self.pre_feedforward_layernorm_2(src)

    def _post_routed_output(self, output):
        """Apply post_moe_layernorm after routed expert combine."""
        return self.post_moe_layernorm(output)

    def _post_shared_output(self, shared_output):
        """Apply post_shared_expert_layernorm to shared expert output."""
        return self.post_shared_expert_layernorm(shared_output)
