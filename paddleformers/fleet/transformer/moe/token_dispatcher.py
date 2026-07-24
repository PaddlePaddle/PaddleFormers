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
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import paddle
from paddle import nn

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

logger = logging.getLogger(__name__)

from .fp8_utils import FP8_ALIGN
from .fused_a2a import (
    HYBRIDEP_TOKEN_ALIGNMENT,
    DeepEPCombineAsyncRefinedRecompute,
    fused_combine,
    fused_dispatch,
    get_hybrid_ep_buffer,
    hybrid_ep_combine,
    hybrid_ep_dispatch,
    quantize_activation_blockscaled_fast,
)
from .moe_utils import (
    AllGatherGroupOp,
    ReduceScatterGroupOp,
    _AllToAll,
    all_gather_group,
    manual_backward,
    permute,
    reduce_scatter_group,
    sort_chunks_by_idxs,
    unpermute,
    use_accuracy_compatible_kernel,
)

HAVE_HYBRID_EP = False
HYBRID_EP_LOAD_CACHED_KERNELS = True


def _sort_chunks_like_tokens(
    input: paddle.Tensor,
    split_sizes: list[int],
    sorted_idxs: list[int],
) -> paddle.Tensor:
    chunks = paddle.split(input, split_sizes, axis=0)
    return paddle.concat([chunks[i] for i in sorted_idxs], axis=0)


try:
    from paddlefleet_ops import is_hybrid_ep_available

    HAVE_HYBRID_EP = is_hybrid_ep_available()
except ImportError:
    HAVE_HYBRID_EP = False


def is_hybrid_ep_backend_selected(
    dispatcher_type: str | None = None,
) -> bool:
    selected_dispatcher = dispatcher_type or "deepep"
    if selected_dispatcher not in (
        "allgather",
        "alltoall",
        "deepep",
        "hybridep",
    ):
        raise ValueError(
            "moe_token_dispatcher_type must be one of: allgather, alltoall, deepep, hybridep"
        )
    if selected_dispatcher != "hybridep":
        return False
    if not HAVE_HYBRID_EP:
        raise ImportError(
            "moe_token_dispatcher_type=hybridep but HybridEP runtime is unavailable."
        )
    return True


def _try_setup_router_topk_metadata(
    manager,
    num_tokens: int,
    topk_weights: paddle.Tensor | None,
    topk_indices: paddle.Tensor | None,
) -> bool:
    if topk_weights is None or topk_indices is None:
        return False
    manager.token_probs = topk_weights.reshape(
        [num_tokens, manager.router_topk]
    )
    manager.token_indices = topk_indices.reshape(
        [num_tokens, manager.router_topk]
    )
    manager.token_indices.stop_gradient = True
    return True


class _DispatchManager(ABC):
    """
    A manager class to handle dispatch and combine processes for MoE models.

    DispatcherManager handles token dispatching according to the routing_map of format
    [num_local_tokens, world_size, num_instances]. The routing_map is a 3D tensor where each
    element indicates whether a token should be sent to a specific rank.

    num_instances is the maximum number of tokens instances dispatched into a target rank, it
    can be the number of local experts, or the size of sub_group.
    """

    @abstractmethod
    def setup_metadata(
        self,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        """Set up metadata of routing_map and probs.

        If ``topk_weights`` and ``topk_indices`` are provided (e.g. produced by
        the router), they will be used directly and the internal ``paddle.topk``
        call will be skipped.
        """
        pass

    @abstractmethod
    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool,
        async_finish: bool,
    ) -> paddle.Tensor:
        """Dispatch the hidden_states according to the routing_map."""
        pass

    @abstractmethod
    def combine(
        self, hidden_states: paddle.Tensor, combine_overlap_handle: dict | None
    ) -> paddle.Tensor:
        """Combine the hidden_states after expert processing."""
        pass

    @abstractmethod
    def get_dispatched_metadata(self) -> paddle.Tensor:
        """Get the metadata of the dispatched hidden_states."""
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        """Get the permuted hidden states by instances."""
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        """Get the restored hidden states by instances."""
        pass


class _HybridEPManager(_DispatchManager):
    """
    HybridEP path using dispatch_with_permute/combine_with_unpermute only.

    The manager owns per-layer handles and count metadata. The communication
    buffer is shared at fused_a2a module scope.
    """

    def __init__(
        self,
        group: Group,
        router_topk: int,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        moe_ep_barrier: bool = True,
        hybridep_buffer_configs: dict | None = None,
        moe_deep_gemm: bool = False,
    ):
        if not HAVE_HYBRID_EP:
            raise ImportError("HybridEP runtime is not available.")

        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.routing_map = None
        self.routing_probs = None
        self.token_indices = None
        self.token_probs = None
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.tokens_per_expert = None
        self.padded_tokens_per_expert = None
        self.num_permuted_tokens = None
        self.handle = None
        self._active_buffer = None
        self.hybridep_buffer_configs = hybridep_buffer_configs or {}
        self._moe_deep_gemm = moe_deep_gemm
        self._reset_dispatch_state()
        self._num_unpadded_tokens = None

    def _reset_dispatch_state(self):
        self._dispatch_uses_fp8 = None
        self._dispatch_pad_multiple = None

    def _set_dispatch_state(self, use_fp8: bool):
        self._dispatch_uses_fp8 = use_fp8
        self._dispatch_pad_multiple = (
            FP8_ALIGN if use_fp8 or self._moe_deep_gemm else None
        )

    def _get_max_num_tokens_per_rank(self, num_local_tokens: int, place) -> int:
        max_num_tokens = num_local_tokens
        if self.group.nranks > 1:
            max_num_tokens_tensor = paddle.to_tensor(
                [num_local_tokens], dtype="int64", place=place
            )
            paddle.distributed.all_reduce(
                max_num_tokens_tensor,
                op=paddle.distributed.ReduceOp.MAX,
                group=self.group,
            )
            max_num_tokens = int(max_num_tokens_tensor.item())
        return (
            (max_num_tokens + HYBRIDEP_TOKEN_ALIGNMENT - 1)
            // HYBRIDEP_TOKEN_ALIGNMENT
            * HYBRIDEP_TOKEN_ALIGNMENT
        )

    def _pad_tokens_to_rank_max(
        self, tensor: paddle.Tensor | None, max_num_tokens: int
    ) -> paddle.Tensor | None:
        if tensor is None or tensor.shape[0] == max_num_tokens:
            return tensor
        assert tensor.shape[0] < max_num_tokens, (
            f"HybridEP token padding expects local tokens <= EP max, got "
            f"{tensor.shape[0]} > {max_num_tokens}."
        )
        pad_shape = [max_num_tokens - tensor.shape[0], *tensor.shape[1:]]
        padding = paddle.zeros(pad_shape, dtype=tensor.dtype)
        return paddle.concat([tensor, padding], axis=0)

    def _get_buffer(
        self,
        hidden_states: paddle.Tensor,
        max_num_of_tokens_per_rank: int | None = None,
    ):
        hidden_dim = hidden_states.shape[-1]
        if max_num_of_tokens_per_rank is None:
            max_num_of_tokens_per_rank = hidden_states.shape[0]
        self._active_buffer = get_hybrid_ep_buffer(
            group=self.group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_num_of_tokens_per_rank,
            num_local_experts=self.num_local_experts,
            load_cached_kernels=HYBRID_EP_LOAD_CACHED_KERNELS,
            **self.hybridep_buffer_configs,
        )
        return self._active_buffer

    def _get_num_permuted_tokens_upper_bound(
        self, num_local_tokens: int
    ) -> int:
        total_routed_tokens = (
            num_local_tokens * self.group.nranks * self.router_topk
        )
        if FP8_ALIGN > 1:
            total_routed_tokens += self.num_local_experts * (FP8_ALIGN - 1)
        return total_routed_tokens

    def _indices_to_dense_metadata(
        self,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor | None,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        safe_indices = paddle.where(
            token_indices >= 0,
            token_indices,
            paddle.zeros_like(token_indices),
        ).astype("int64")
        one_hot = paddle.nn.functional.one_hot(
            safe_indices, num_classes=self.num_experts
        )
        valid_mask = (token_indices >= 0).astype(one_hot.dtype).unsqueeze(-1)
        one_hot = one_hot * valid_mask
        routing_map = paddle.sum(one_hot, axis=1).astype("bool")

        probs = None
        if token_weights is not None:
            probs = paddle.sum(
                one_hot.astype(token_weights.dtype)
                * token_weights.unsqueeze(-1),
                axis=1,
            )
            if probs.dtype != paddle.float32:
                probs = probs.astype("float32")
        return routing_map, probs

    def _get_dispatch_metadata(
        self,
        token_indices: paddle.Tensor | None,
        token_weights: paddle.Tensor | None,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        if self.routing_map is not None:
            return self.routing_map, self.routing_probs
        assert token_indices is not None, (
            "HybridEP dispatch requires routing metadata."
        )
        return self._indices_to_dense_metadata(token_indices, token_weights)

    def setup_metadata(
        self,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        num_tokens = routing_map.shape[0]
        self.routing_map = routing_map.reshape(
            [num_tokens, self.num_experts]
        ).astype("bool")
        self.routing_probs = probs.reshape([num_tokens, self.num_experts])
        if self.routing_probs.dtype != paddle.float32:
            self.routing_probs = self.routing_probs.astype("float32")
        if _try_setup_router_topk_metadata(
            self, num_tokens, topk_weights, topk_indices
        ):
            return
        self.token_probs, self.token_indices = paddle.topk(
            self.routing_probs, self.router_topk, axis=-1
        )

    def _extract_tokens_per_expert(
        self,
        num_dispatched_tokens: int,
        local_expert_routing_map: paddle.Tensor,
    ):
        return (
            local_expert_routing_map[:num_dispatched_tokens]
            .astype("int64")
            .sum(axis=0)
        )

    def _set_num_permuted_tokens(self, tokens_per_expert: paddle.Tensor) -> int:
        self.num_permuted_tokens = int(
            paddle.sum(tokens_per_expert.astype("int64")).item()
        )
        return self.num_permuted_tokens

    def dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
    ) -> paddle.Tensor:
        del async_finish
        self.token_indices = token_indices
        self.token_probs = token_weights
        hidden_states, self.dispatched_probs, scale = hybrid_ep_dispatch(
            hidden_states,
            token_indices,
            token_weights,
            self,
            fp8_dispatch,
        )
        self.dispatched_indices = None
        return hidden_states, None if scale is None else {"scale": scale}

    def _dispatch_with_permute_impl(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        use_fp8: bool = False,
    ):
        num_unpadded_tokens = hidden_states.shape[0]
        max_num_tokens = self._get_max_num_tokens_per_rank(
            num_unpadded_tokens, hidden_states.place
        )
        self._num_unpadded_tokens = num_unpadded_tokens
        routing_map, probs = self._get_dispatch_metadata(
            token_indices, token_weights
        )
        hidden_states = self._pad_tokens_to_rank_max(
            hidden_states, max_num_tokens
        )
        routing_map = self._pad_tokens_to_rank_max(routing_map, max_num_tokens)
        probs = self._pad_tokens_to_rank_max(probs, max_num_tokens)
        buffer = self._get_buffer(hidden_states, max_num_tokens)
        num_permuted_tokens = self._get_num_permuted_tokens_upper_bound(
            max_num_tokens
        )
        scaling_factor = None
        if use_fp8:
            hidden_states, scaling_factor = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    hidden_states,
                    quant_method="1x128",
                    input_transpose=False,
                    output_scale_transpose=True,
                    return_transpose_only=False,
                )
            )
            scaling_factor = scaling_factor.T.contiguous()
        self._set_dispatch_state(use_fp8)
        (
            hidden_states,
            dispatched_probs,
            scale,
            tokens_per_expert,
            self.handle,
        ) = buffer.dispatch_with_permute(
            hidden=hidden_states,
            routing_map=routing_map,
            probs=probs,
            num_of_experts_per_rank=self.num_local_experts,
            use_fp8=use_fp8,
            scaling_factor=scaling_factor,
            pad_multiple=self._dispatch_pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=True,
        )
        self.padded_tokens_per_expert = tokens_per_expert
        num_permuted_tokens = self._set_num_permuted_tokens(tokens_per_expert)
        hidden_states = hidden_states[:num_permuted_tokens]
        if dispatched_probs is not None:
            dispatched_probs = dispatched_probs[:num_permuted_tokens]
        if scale is not None:
            scale = scale[:num_permuted_tokens]
        (
            _sparse_to_dense_map,
            _rdma_to_attn_map,
            _attn_to_rdma_map,
            num_dispatched_tokens_tensor,
            local_expert_routing_map,
            *_,
        ) = self.handle
        num_dispatched_tokens = int(num_dispatched_tokens_tensor.item())
        self.tokens_per_expert = self._extract_tokens_per_expert(
            num_dispatched_tokens,
            local_expert_routing_map,
        )
        return hidden_states, dispatched_probs, scale

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ) -> paddle.Tensor:
        return self.dispatch_overlap(
            hidden_states,
            self.token_indices,
            self.token_probs,
            fp8_dispatch=fp8_dispatch,
            async_finish=async_finish,
        )

    def combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        use_rr_deepep_combine: bool = False,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ) -> paddle.Tensor:
        del async_finish, use_rr_deepep_combine
        if combine_overlap_handle is not None:
            raise NotImplementedError(
                "HybridEP backend does not support combine overlap in PaddleFleet."
            )
        hidden_states = hybrid_ep_combine(
            hidden_states, self, self.num_permuted_tokens
        )
        self.dispatched_probs = None
        self.handle = None
        self.num_permuted_tokens = None
        self._reset_dispatch_state()
        if (
            self._num_unpadded_tokens is not None
            and hidden_states.shape[0] != self._num_unpadded_tokens
        ):
            hidden_states = hidden_states[: self._num_unpadded_tokens]
        self._num_unpadded_tokens = None
        return hidden_states

    def get_dispatched_metadata(self) -> paddle.Tensor:
        if self.dispatched_indices is None or self.dispatched_probs is None:
            raise NotImplementedError(
                "HybridEP backend does not expose fused-node dispatch metadata for the current mode."
            )
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> paddle.Tensor:
        return self.tokens_per_expert

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        return hidden_states

    def get_restored_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        if self.dispatched_probs is None:
            return hidden_states
        return hidden_states * self.dispatched_probs.astype(
            hidden_states.dtype
        ).unsqueeze(-1)


class _DeepEPManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: Group,
        router_topk: int,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        moe_ep_barrier: bool = True,
    ):
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.moe_ep_barrier = moe_ep_barrier

        # Metadata
        self.token_indices = None
        self.token_probs = None
        # Handle used for combine operation
        self.handle = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not supported in your paddlepaddle whl package."
            )
        self._rr_fusedcombined = None

    def setup_metadata(
        self,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        num_tokens = routing_map.shape[0]

        if _try_setup_router_topk_metadata(
            self, num_tokens, topk_weights, topk_indices
        ):
            return

        routing_map = routing_map.reshape([num_tokens, self.num_experts])
        probs = probs.reshape([num_tokens, self.num_experts])
        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = paddle.topk(
            probs, self.router_topk, axis=-1
        )

    def dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
    ) -> paddle.Tensor:
        hidden_states, dispatched_probs, states, scale = fused_dispatch(
            hidden_states,
            token_indices,
            token_weights,
            self.num_experts,
            self.group,
            fp8_dispatch=fp8_dispatch,
            async_finish=async_finish,
            use_ue8m0=use_ue8m0,
        )
        self.handle = states["handle"]
        self.tokens_per_expert = states["tokens_per_expert"]
        self.dispatched_indices = states["dispatched_indices"]
        self.dispatched_probs = dispatched_probs

        return hidden_states, scale

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ) -> paddle.Tensor:
        hidden_states, dispatched_probs, states, scale = fused_dispatch(
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            self.group,
            fp8_dispatch=fp8_dispatch,
            async_finish=async_finish,
            moe_ep_barrier=self.moe_ep_barrier,
            use_ue8m0=use_ue8m0,
            using_sonic_moe=using_sonic_moe,
        )
        self.handle = states["handle"]
        self.tokens_per_expert = states["tokens_per_expert"]
        self.dispatched_indices = states["dispatched_indices"]
        self.dispatched_probs = dispatched_probs

        return hidden_states, scale

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (paddle.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (paddle.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            tuple[paddle.Tensor, paddle.Tensor]:
                - routing_map: Multihot vector.
                - probs: Multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = paddle.zeros(
            (batch_size, self.num_local_experts), dtype=paddle.int64
        )

        multihot_probs = paddle.zeros(
            (batch_size, self.num_local_experts), dtype=paddle.float32
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = paddle.arange(batch_size).repeat_interleave(
            mask.sum(axis=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.cast(paddle.bool), multihot_probs

    def get_dispatched_metadata(self) -> paddle.Tensor:
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> paddle.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        use_rr_deepep_combine: bool = False,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ) -> paddle.Tensor:
        if combine_overlap_handle is not None and use_rr_deepep_combine:
            if self._rr_fusedcombined is None:
                self._rr_fusedcombined = DeepEPCombineAsyncRefinedRecompute()
            elif not isinstance(
                self._rr_fusedcombined, DeepEPCombineAsyncRefinedRecompute
            ):
                raise RuntimeError(
                    f"_rr_fusedcombined type mismatch: expected DeepEPCombineAsyncRefinedRecompute, "
                    f"got {type(self._rr_fusedcombined).__name__}."
                )
        if fp8_dispatch is True:
            assert combine_grad_handle is not None, (
                "fp8_dispatch=True, but combine_grad_handle is None."
            )
        hidden_states = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            _rr_fusedcombined=self._rr_fusedcombined,
            combine_overlap_handle=combine_overlap_handle,
            async_finish=async_finish,
            moe_ep_barrier=self.moe_ep_barrier,
            use_rr_deepep_combine=use_rr_deepep_combine,
            fp8_dispatch=fp8_dispatch,
            combine_grad_handle=combine_grad_handle,
        )
        # Release the handle and token_indices after combine operation
        self.handle = None
        self.token_indices = None
        self.token_probs = None
        self.dispatched_probs = None
        self.dispatched_indices = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        self.dispatched_routing_map, self.dispatched_probs = (
            self._indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs
            )
        )
        self.hidden_shape_before_permute = hidden_states.shape
        hidden_states, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            num_out_tokens=sum(self.tokens_per_expert),
        )
        return hidden_states

    def get_restored_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        input_dtype = hidden_states.dtype
        assert self.dispatched_probs.dtype == paddle.float32, (
            "DeepEP only supports float32 probs"
        )
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            probs=self.dispatched_probs,
        )
        return hidden_states.to(input_dtype)


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self, ep_group) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self._ep_group = ep_group

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return self._ep_group

    @property
    def ep_size(self):
        """Get expert model parallel world_size."""
        return self.ep_group.world_size

    @abstractmethod
    def token_permutation(
        self,
        tokens: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (paddle.Tensor): Input tokens.
            probs (paddle.Tensor): The routing probability tensor [num_tokens, num_experts].
            routing_map (paddle.Tensor): Token to expert mapping tensor.

        Returns:
            paddle.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(
        self, expert_output: paddle.Tensor, bias: paddle.Tensor = None
    ):
        """Restores the expert output to its original ordering.

        Args:
            expert_output (paddle.Tensor): The output tensor from the expert models.
            bias (paddle.Tensor): The bias tensor.

        Returns:
            (paddle.Tensor, paddle.Tensor): Unpermuted activation and optional bias.
        """
        raise NotImplementedError("Restore function not implemented.")


class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """
    Flexible token dispatcher for MoE models with Efficient-A2A communication kernels.
    """

    def __init__(
        self,
        num_local_experts: int,
        num_experts_per_tok: int,
        n_routed_experts: int,
        ep_group: Group,
        moe_ep_barrier: bool = True,
        dispatcher_type: str | None = None,
        hybridep_buffer_configs: dict | None = None,
        moe_deep_gemm: bool = False,
    ):
        super().__init__(ep_group)

        self.num_local_experts = num_local_experts
        assert self.ep_size > 1, "Flex token dispatcher requires EP > 1"
        manager_cls = (
            _HybridEPManager
            if is_hybrid_ep_backend_selected(dispatcher_type)
            else _DeepEPManager
        )
        manager_kwargs = {
            "group": self.ep_group,
            "router_topk": num_experts_per_tok,
            "num_experts": n_routed_experts,
            "num_local_experts": self.num_local_experts,
            "moe_ep_barrier": moe_ep_barrier,
        }
        if manager_cls is _HybridEPManager:
            manager_kwargs["hybridep_buffer_configs"] = hybridep_buffer_configs
            manager_kwargs["moe_deep_gemm"] = moe_deep_gemm
        self._comm_manager = manager_cls(**manager_kwargs)

    def dispatch_preprocess(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])
        self._comm_manager.setup_metadata(
            routing_map, probs, topk_weights, topk_indices
        )
        return hidden_states

    def dispatch_preprocess_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_probs: paddle.Tensor,
        token_indices: paddle.Tensor,
    ):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])
        self._comm_manager.routing_map = None
        self._comm_manager.routing_probs = None
        self._comm_manager.token_probs = token_probs
        self._comm_manager.token_indices = token_indices
        return hidden_states

    def token_dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool,
        async_finish: bool = False,
        use_ue8m0: bool = False,
    ):
        return self._comm_manager.dispatch_overlap(
            hidden_states,
            token_indices,
            token_weights,
            fp8_dispatch,
            async_finish,
            use_ue8m0=use_ue8m0,
        )

    def token_dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ):
        return self._comm_manager.dispatch(
            hidden_states,
            fp8_dispatch,
            async_finish,
            use_ue8m0=use_ue8m0,
            using_sonic_moe=using_sonic_moe,
        )

    def dispatch_postprocess(
        self,
        hidden_states: paddle.Tensor,
    ):
        global_input_tokens = (
            self._comm_manager.get_permuted_hidden_states_by_experts(
                hidden_states
            )
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()

        return global_input_tokens, tokens_per_expert

    def combine_preprocess(self, hidden_states: paddle.Tensor):
        return self._comm_manager.get_restored_hidden_states_by_experts(
            hidden_states
        )

    def token_combine(self, hidden_states: paddle.Tensor, async_finish=False):
        return self._comm_manager.combine(
            hidden_states, async_finish=async_finish
        )

    def combine_postprocess(self, hidden_states: paddle.Tensor):
        return hidden_states.reshape(self.hidden_shape)

    def get_dispatched_routing(self):
        """Return (dispatched_indices, dispatched_probs, tokens_per_expert)."""
        return (
            self._comm_manager.dispatched_indices,
            self._comm_manager.dispatched_probs,
            self._comm_manager.tokens_per_expert,
        )

    def token_permutation(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])

        self._comm_manager.setup_metadata(
            routing_map, probs, topk_weights, topk_indices
        )
        hidden_states, scale = self._comm_manager.dispatch(hidden_states)
        global_input_tokens = (
            self._comm_manager.get_permuted_hidden_states_by_experts(
                hidden_states
            )
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: paddle.Tensor, bias: paddle.Tensor | None = None
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        assert bias is None, "Bias is not supported in MoEFlexTokenDispatcher"
        hidden_states = (
            self._comm_manager.get_restored_hidden_states_by_experts(
                hidden_states
            )
        )
        hidden_states = self._comm_manager.combine(hidden_states)

        hidden_states = hidden_states.reshape(self.hidden_shape)
        return hidden_states, None


class AllToAllTokenDispatcher(nn.Layer):
    """
    All-to-All EP
    """

    def __init__(
        self,
        moe_group: Group,
        expert_model_parallel_size: int,
        num_experts_per_device: int,
        local_expert_indices: list,
    ):
        nn.Layer.__init__(self)
        self.moe_group = moe_group
        self.expert_model_parallel_size = expert_model_parallel_size
        self.num_experts_per_device = num_experts_per_device
        self.local_expert_indices = local_expert_indices
        self.num_local_experts = len(local_expert_indices)

    def dispatch_preprocess(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        mask: paddle.Tensor,  # routing_map
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        self.routing_map = mask
        self.probs = probs
        self.num_experts = (
            self.num_experts_per_device * self.expert_model_parallel_size
        )
        mask = mask.to(paddle.int32)

        if len(hidden_states.shape) == 3:
            batch_size, seq_len, d_model = hidden_states.shape
        else:
            seq_len, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model])
        self.d_model = d_model
        self.reshaped_input_shape = reshaped_input.shape
        tokens_per_expert = mask.sum(axis=0)  # Shape: [num_experts]
        tokens_per_expert = tokens_per_expert.detach()
        tokens_per_ep_rank = tokens_per_expert.reshape(
            [self.expert_model_parallel_size, -1]
        ).sum(axis=1)
        # First All-to-All: Exchange expert token counts across ranks
        # Returns `tokens_per_expert_group` is for current rank
        num_global_tokens_per_expert = AllGatherGroupOp.apply(
            tokens_per_expert, group=self.moe_group
        ).reshape(self.expert_model_parallel_size, self.num_experts)
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].clone()

        # Can also use the two AllToAll functions below instead of the above AllGather
        # It will save memory , but also has more accuracy diff with DeepEP version
        # global_tokens_per_expert = _AllToAll.apply(
        #     [tokens_per_expert.shape[0]],
        #     tokens_per_expert,
        #     group=self.moe_group,
        # )
        # num_global_tokens_per_local_expert = global_tokens_per_expert.reshape(self.expert_model_parallel_size, self.num_local_experts)

        if num_global_tokens_per_local_expert.sum().item() == 0:
            self.is_empty_tokens = True
        else:
            self.is_empty_tokens = False

        self.tokens_per_expert = num_global_tokens_per_local_expert.sum(axis=0)

        num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(
            axis=1
        )

        self.num_global_tokens_per_local_expert = (
            num_global_tokens_per_local_expert.reshape(
                -1, self.num_local_experts
            )
        )

        self.output_splits = num_global_tokens_per_rank.cpu().tolist()
        num_local_tokens_per_expert = self.routing_map.sum(dim=0)
        self.input_split_sizes = num_local_tokens_per_expert.reshape(
            self.expert_model_parallel_size, self.num_local_experts
        ).sum(axis=1)
        self.output_shape_tokens = [
            num_global_tokens_per_rank.sum().cpu().item(),
            d_model,
        ]

        (
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
        ) = permute(reshaped_input, self.routing_map)
        if use_accuracy_compatible_kernel():
            num_routed_tokens = int(tokens_per_expert.sum().item())
            routing_map = self.routing_map.cast(paddle.bool).T.contiguous()
            flat_sorted = paddle.argsort(
                routing_map.reshape([-1]).cast("int32"),
                descending=True,
                stable=True,
            )[:num_routed_tokens]
            self.permuted_local_probs = paddle.index_select(
                self.probs.T.contiguous().reshape([-1]),
                flat_sorted,
                axis=0,
            )
        self.permutated_local_input_tokens_shape = (
            permutated_local_input_tokens.shape
        )

        return permutated_local_input_tokens

    def token_dispatch(
        self,
        permutated_local_input_tokens: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ):
        # Second All-to-All: Exchange expert tokens across ranks. `gathered_tokens` are the tokens that will be processed by current rank
        global_input_tokens = _AllToAll.apply(
            self.output_shape_tokens,
            permutated_local_input_tokens,  # sorted_tokens,
            out_split_sizes=self.output_splits,
            in_split_sizes=self.input_split_sizes,
            group=self.moe_group,
        )
        if use_accuracy_compatible_kernel():
            # Match Megatron's all-to-all backward numerics by routing probs through a
            # 2D [tokens, 1] tensor, like hidden-state dispatch.
            global_input_probs_2d = _AllToAll.apply(
                [self.output_shape_tokens[0], 1],
                self.permuted_local_probs.unsqueeze(-1),
                out_split_sizes=self.output_splits,
                in_split_sizes=self.input_split_sizes,
                group=self.moe_group,
            )
            self.global_input_probs = global_input_probs_2d.squeeze(-1)

        return global_input_tokens, None

    def dispatch_postprocess(
        self,
        global_input_tokens: paddle.Tensor,
    ):
        input_chunk_idxs = paddle.arange(self.num_experts)
        # [num_local_experts, ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        if self.num_local_experts > 1 and not self.is_empty_tokens:
            split_sizes_list = (
                self.num_global_tokens_per_local_expert.ravel().tolist()
            )
            sorted_idxs_list = self.sort_input_by_local_experts.tolist()
            global_input_tokens, _ = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
            )
            if use_accuracy_compatible_kernel():
                self.global_input_probs = _sort_chunks_like_tokens(
                    self.global_input_probs,
                    split_sizes_list,
                    sorted_idxs_list,
                )
        sorted_tokens = global_input_tokens
        self.tokens_per_expert_post_gather = self.tokens_per_expert
        return sorted_tokens, self.tokens_per_expert_post_gather

    def get_dispatched_routing(self):
        """Return (dispatched_indices, dispatched_probs, tokens_per_expert).

        AllToAll uses tokens_per_expert-based expert processing
        (expert_forward), so dispatched_indices and dispatched_probs are None.
        The corresponding branch in ``fusion_moe_forward`` selects
        ``expert_forward`` instead of index-based fusion kernels.
        """
        return (None, None, self.tokens_per_expert)

    def combine_preprocess(self, hidden_states: paddle.Tensor):
        if self.num_local_experts > 1 and not self.is_empty_tokens:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
            )
        return hidden_states

    def token_combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        fp8_combine_grad_handle: dict | None = None,
    ):
        permutated_local_input_tokens = _AllToAll.apply(
            self.permutated_local_input_tokens_shape,
            hidden_states,
            out_split_sizes=self.input_split_sizes,
            in_split_sizes=self.output_splits,
            group=self.moe_group,
        )
        return permutated_local_input_tokens

    def combine_postprocess(self, permutated_local_input_tokens: paddle.Tensor):
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.reshaped_input_shape,
            probs=(None if use_accuracy_compatible_kernel() else self.probs),
            routing_map=self.routing_map,
        )
        return output


class _RouterAllGather(paddle.autograd.PyLayer):
    """AllGather for router topk weights, used only by the allgather dispatcher.

    Forward:  [T_local, K] --AllGather(EP)--> [T_global, K]  (identical on all ranks)
    Backward: [T_global, K] --ReduceScatter(EP, SUM)--> [T_local, K]

    In the allgather dispatcher every EP rank holds an intermediate-dim shard
    of every expert and computes a partial expert output for ALL global tokens
    using the same all-gathered router weights.  Each rank therefore produces
    its own partial gradient for the shared weight tensor; the backward must
    sum these partials (reduce) and return each rank its own token segment
    (scatter).  A plain scatter would keep only the origin rank's partial and
    discard the rest, under-training the router by ~1/nranks.
    """

    @staticmethod
    def forward(ctx, input, group):
        ctx.group = group
        ctx.input_shape = list(input.shape)
        if group is None or group.nranks == 1:
            return input.clone()
        output_shape = list(input.shape)
        output_shape[0] = output_shape[0] * group.nranks
        output = paddle.empty(shape=output_shape, dtype=input.dtype)
        # NOTE on stream placement (measured, see allgather perf optimization):
        # The topk_weights AllGather intentionally runs on the CALC stream
        # (synchronous) rather than the comm stream.  The indices AllGather runs
        # async on the comm stream; issuing BOTH on the comm stream would
        # serialize the small weights gather behind the much larger fused
        # fp8-data AllGather already queued there, increasing dispatch latency
        # (measured +~4ms/step).  Keeping weights on calc lets it overlap with
        # the in-flight comm-stream gathers.  Backward (reduce-scatter) is on
        # the comm stream and unchanged.
        paddle.distributed.stream.all_gather(
            output, input, group=group, use_calc_stream=True
        )
        return output

    @staticmethod
    def backward(ctx, grad):
        group = ctx.group
        local_shape = ctx.input_shape
        if group is None or group.nranks == 1:
            if list(grad.shape) != local_shape:
                grad = grad.reshape(local_shape)
            return grad
        global_shape = [local_shape[0] * group.nranks, *local_shape[1:]]
        if list(grad.shape) != global_shape:
            expected_numel = 1
            for _d in global_shape:
                expected_numel *= _d
            if int(grad.numel()) != expected_numel:
                raise ValueError(
                    "_RouterAllGather.backward: incoming grad has "
                    f"{int(grad.numel())} elements but the AllGather'd router "
                    f"tensor requires {expected_numel} (global_shape="
                    f"{global_shape})."
                )
            grad = grad.reshape(global_shape)
        out = reduce_scatter_group(grad.contiguous(), group=group)
        if list(out.shape) != local_shape:
            out = out.reshape(local_shape)
        return out


class _PreAllGatherResult(paddle.autograd.PyLayer):
    """Consume a pre-issued async AllGather of hidden_states.

    Forward waits for the async NCCL task and returns the filled buffer.
    Backward is ReduceScatter (dual of AllGather).
    The ``handle`` is a plain dict, not a tensor, so Paddle's PyLayer does not
    expect a gradient for it — backward returns only one value.
    """

    @staticmethod
    def forward(ctx, hidden_states, handle):
        handle["task"].wait()
        ctx.group = handle["group"]
        return handle["output"]

    @staticmethod
    def backward(ctx, grad):
        grad_input = ReduceScatterGroupOp.apply(grad, ctx.group)
        return grad_input


class _PreAllGatherFP8Result(paddle.autograd.PyLayer):
    """FP8 variant of _PreAllGatherResult.

    Forward waits for the single fused async AllGather task (fp8 data ++ block
    scale packed per token) and returns ``(x_fp8_global, scale_global)`` after
    splitting.  The fp8 tensor is consumed directly by SonicMoE's _UpProjection
    via prequant_activation_payload.

    _UpProjection.backward produces a bf16 dx for the activation input, but
    the forward output here is fp8.  set_grad_in_dtype_consistent(False) and
    set_materialize_grads(False) tell Paddle to pass the bf16 grad through
    without dtype coercion.  Backward then ReduceScatters that bf16 grad back
    to the local token shard.
    """

    @staticmethod
    def forward(ctx, hidden_states, handle):
        handle["task"].wait()
        ctx.group = handle["group"]
        x_fp8_global, scale_global = _split_fused_fp8_gather(
            handle["fused_global"],
            handle["H"],
            handle["H128"],
            handle["scale_dtype"],
        )
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        return x_fp8_global, scale_global

    @staticmethod
    def backward(ctx, grad_output, grad_scale=None):
        del grad_scale
        group = ctx.group
        if grad_output is None:
            return None
        if group is None or group.nranks == 1:
            return grad_output
        grad_input = ReduceScatterGroupOp.apply(grad_output, group)
        return grad_input


def _reduce_scatter_async(input, group):
    """Async ReduceScatter (SUM, axis 0) on the comm stream.
    Returns (output, task)."""
    input = input.contiguous()
    out_shape = list(input.shape)
    if out_shape[0] % group.nranks != 0:
        raise ValueError(
            f"ReduceScatter input rows {out_shape[0]} not divisible by "
            f"nranks {group.nranks}"
        )
    out_shape[0] //= group.nranks
    output = paddle.empty(shape=out_shape, dtype=input.dtype)
    task = paddle.distributed.stream.reduce_scatter(
        output,
        input,
        op=paddle.distributed.ReduceOp.SUM,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    return output, task


def _all_gather_async(input, group):
    """Async AllGather (axis 0) on the comm stream.
    Returns (output, task)."""
    input = input.contiguous()
    out_shape = list(input.shape)
    out_shape[0] *= group.nranks
    output = paddle.empty(shape=out_shape, dtype=input.dtype)
    task = paddle.distributed.stream.all_gather(
        output,
        input,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    return output, task


def _quantize_and_pack_fp8(x):
    """Quantize ``x`` to fp8 e4m3 + int32 1x128 block scale and pack into a
    single uint8 row ``[T_local, H + 4*H128]`` (fp8 data ++ scale bytes).

    Returns ``(fused_local, H, H128, scale_dtype)``. The caller is responsible
    for AllGather (sync or async) and for unpacking via
    :func:`_split_fused_fp8_gather`. Bit-identical to two separate gather of
    data and scale: AllGather is a lossless row concat and the 1x128 scale
    lives within a single token's hidden vector, so packing along axis 1
    before the gather yields the same per-rank bytes as gathering separately.
    """
    if quantize_activation_blockscaled_fast is None:
        raise RuntimeError(
            "Cannot find quantize_activation_blockscaled_fast, "
            "please update sonicmoe."
        )
    x = x.contiguous()
    x_fp8, scale = quantize_activation_blockscaled_fast(
        x, scale_dtype=paddle.int32
    )
    _, H = x_fp8.shape
    H128 = scale.shape[1]
    scale_dtype = scale.dtype
    fused_local = paddle.concat(
        [x_fp8.view("uint8"), scale.view("uint8")], axis=1
    ).contiguous()
    return fused_local, H, H128, scale_dtype


def _fused_fp8_all_gather_async(x, group):
    """Quantize local tensor to fp8 e4m3 + int32 1x128 block scale, then a SINGLE
    async AllGather of the fused (fp8-data-as-uint8 ++ scale-as-uint8) per-token
    byte row on the comm stream.  Returns
    ``(fused_global, H, H128, scale_dtype, task)``.

    Fusing data and scale into one AllGather (vs two back-to-back collectives)
    is bit-identical: AllGather concatenates rows along axis 0 and every rank
    carries an identical ``T_local``, so packing the two payloads along axis 1
    before the gather yields the same per-rank bytes as gathering them
    separately.  It removes one NCCL kernel launch (and its peer-wait fixed
    cost) per collective.

    Because AllGather is a lossless row concat and the 1x128 scale lives within
    a single token's hidden vector, this is bit-identical to
    bf16-AllGather-then-quantize (halving on-wire bytes vs bf16).  The caller
    waits ``task`` and then calls :func:`_split_fused_fp8_gather` to recover
    ``(data_e4m3_global, scale_global)``.

    Used both by the combine backward (``_AllGatherCombineAsync.backward``) and
    by ``AllGatherTokenDispatcher.pre_allgather`` for the forward activation.
    """
    fused_local, H, H128, scale_dtype = _quantize_and_pack_fp8(x)
    T_global = fused_local.shape[0] * group.nranks
    fused_global = paddle.empty([T_global, fused_local.shape[1]], dtype="uint8")
    task = paddle.distributed.stream.all_gather(
        fused_global,
        fused_local,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    return fused_global, H, H128, scale_dtype, task


def _split_fused_fp8_gather(fused_global, H, H128, scale_dtype):
    """Recover ``(data_e4m3_global, scale_global)`` from a fused gather buffer.

    Inverse of the packing in :func:`_fused_fp8_all_gather_async` /
    :meth:`AllGatherTokenDispatcher.pre_allgather`. Must be called only after the
    gather task has completed.
    """
    T_global = fused_global.shape[0]
    data_global_u8 = fused_global[:, :H].contiguous()
    scale_global = fused_global[:, H:].contiguous().view(scale_dtype)
    return (
        data_global_u8.view("float8_e4m3fn"),
        scale_global.reshape([T_global, H128]),
    )


class _AllGatherCombineAsync(paddle.autograd.PyLayer):
    """Fuse the combine ReduceScatter with a shared-expert subgraph for overlap.

    Forward:
      1. Issue async ReduceScatter of expert output on comm stream.
      2. Run shared-expert subgraph on calc stream (concurrent with step 1).
      3. Wait for ReduceScatter, return (combined_x,) + fn_out.

    Backward (dual):
      - bf16 path: async AllGather of grad on comm stream while shared-expert
        backward runs on calc stream.
      - fp8 path (fp8_combine_grad_handle != None): quantize local grad to fp8,
        async AllGather both data and scale on comm stream, write results into
        the handle for _DownProjection.backward to consume directly.

    Overlap is safe because fn's inputs are independent of x (expert output).
    """

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        *fn_args,
        fn,
        is_first_fwd=False,
        fp8_combine_grad_handle=None,
    ):
        if fn is None:
            raise ValueError(
                "_AllGatherCombineAsync requires a non-None fn for overlap."
            )
        ctx.group = group
        ctx.fp8_combine_grad_handle = fp8_combine_grad_handle
        if fp8_combine_grad_handle is not None:
            ctx.set_grad_in_dtype_consistent(False)
            ctx.set_materialize_grads(False)

        if group is None or group.nranks == 1:
            combined_x = x.clone()
            ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
            return (combined_x,) + fn_out

        combined_x, task = _reduce_scatter_async(x, group)
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
        task.wait()

        return (combined_x,) + fn_out

    @staticmethod
    def backward(ctx, grad_output, *fn_out_grads):
        group = ctx.group
        handle = ctx.fp8_combine_grad_handle
        if group is None or group.nranks == 1:
            grad_x = grad_output.clone()
            fn_args_grads = ctx.bwf(*fn_out_grads)
            return (grad_x,) + fn_args_grads

        if handle is not None:
            fused_global, _H, _H128, _sdt, task = _fused_fp8_all_gather_async(
                grad_output, group
            )
            fn_args_grads = ctx.bwf(*fn_out_grads)
            task.wait()
            data_e4m3, scale_global = _split_fused_fp8_gather(
                fused_global, _H, _H128, _sdt
            )
            handle["data"] = data_e4m3
            handle["scale"] = scale_global
            return (data_e4m3,) + fn_args_grads

        grad_x, task = _all_gather_async(grad_output, group)
        fn_args_grads = ctx.bwf(*fn_out_grads)
        task.wait()
        return (grad_x,) + fn_args_grads


class _AllGatherCombineNoOverlap(paddle.autograd.PyLayer):
    """ReduceScatter combine without overlap subgraph (sync on calc stream).

    Mirrors ``_AllGatherCombineAsync``'s fp8/bf16 grad collection but skips the
    shared-expert overlap. All collectives use the same sync calc-stream
    wrappers (``reduce_scatter_group`` / ``all_gather_group``) as
    :class:`ReduceScatterGroupOp` and the ``_PreAllGather*Result`` backward
    paths, so they FIFO-order naturally after the expert MLP forward and
    ``_DownProjection.backward`` without cross-stream event synchronization.
    Needed because the no-overlap path must still populate
    ``fp8_combine_grad_handle`` in backward for ``_DownProjection.backward``
    to consume; previously the no-overlap branch returned ``hidden_states``
    directly and dropped the handle.
    """

    @staticmethod
    def forward(ctx, x, group, fp8_combine_grad_handle=None):
        ctx.group = group
        ctx.fp8_combine_grad_handle = fp8_combine_grad_handle
        if fp8_combine_grad_handle is not None:
            ctx.set_grad_in_dtype_consistent(False)
            ctx.set_materialize_grads(False)
        if group is None:
            return x.clone()
        return reduce_scatter_group(x, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        handle = ctx.fp8_combine_grad_handle
        if group is None:
            return grad_output.clone()
        if handle is not None:
            fused_local, H, H128, scale_dtype = _quantize_and_pack_fp8(
                grad_output
            )
            fused_global = all_gather_group(fused_local, group=group)
            data_e4m3, scale_global = _split_fused_fp8_gather(
                fused_global, H, H128, scale_dtype
            )
            handle["data"] = data_e4m3
            handle["scale"] = scale_global
            return data_e4m3
        return all_gather_group(grad_output, group=group)


@paddle.no_grad()
def _tokens_per_expert_histogram(indices, num_experts):
    """Count tokens per expert WITHOUT any GPU->CPU synchronization.

    ``indices`` is the [..., K] routing tensor that may contain ``-1`` for
    padding tokens.  The result is an int32 ``[num_experts]`` histogram.

    This replaces ``masked_select`` + ``bincount``: ``masked_select`` produces a
    *variable-length* output (its size is data-dependent), which forces Paddle
    to copy the element count back to the CPU, stalling the host.  ``bincount``
    likewise must resolve ``max(input)+1`` on the CPU to size its output.  Both
    serialize the pipeline.

    Instead we build a fixed-size one-hot histogram: padding (-1) is sent to a
    sink column ``== num_experts`` so it is dropped, and every other value is in
    ``[0, num_experts)``.  All shapes are known a priori, so no host sync occurs.
    Numerically identical to ``bincount(masked_select(indices, indices>=0),
    minlength=num_experts)``.
    """
    flat = indices.reshape([-1]).cast("int64")
    sink = paddle.full_like(flat, num_experts)
    clamped = paddle.where(flat >= 0, flat, sink)
    # scatter(overwrite=False) accumulates updates[i] into counts[clamped[i]]
    # via atomic add — fixed-shape [num_experts + 1] output, no host sync,
    # and O(T*K + E) temp memory instead of O(T*K*E) from a full one-hot.
    counts = paddle.zeros([num_experts + 1], dtype="int32")
    ones = paddle.ones([flat.shape[0]], dtype="int32")
    counts = paddle.scatter(counts, clamped, ones, overwrite=False)
    return counts[:num_experts]


class AllGatherTokenDispatcher(nn.Layer):
    """AllGather + ReduceScatter EP dispatcher (SonicMoE fused-kernel only).

    Every expert is sharded along its intermediate dim into EP partitions;
    every rank holds all experts but only its I/EP shard of each.  The forward
    data flow is:

        [T_local, H] --AllGather--> [T_global, H] --SonicMoE fused
        _UpProjection / _DownProjection (all tokens, partial I)-->
        [T_global, H] --ReduceScatter(SUM)--> [T_local, H]

    Routing metadata (topk_indices, topk_weights) is also AllGathered so every
    rank sees the same global token assignments.  No explicit permute/unpermute
    is needed — the SonicMoE fused kernels handle token gather/scatter
    internally based on the indices.
    """

    def __init__(
        self,
        moe_group: Group,
        expert_model_parallel_size: int,
        num_experts: int,
        fp8_dispatch: bool = False,
        use_ue8m0: bool = False,
    ):
        nn.Layer.__init__(self)
        self.moe_group = moe_group
        self.ep_size = expert_model_parallel_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts  # every rank holds all experts
        self.fp8_dispatch = fp8_dispatch
        self.use_ue8m0 = use_ue8m0
        self._pre_ag_handle: dict | None = None
        self._global_topk_indices = None
        self._global_topk_weights = None
        self._fp8_dispatch_scale = None
        self._overlap_combined = None

    def pre_allgather(self, hidden_states: paddle.Tensor):
        """Issue an async AllGather of hidden_states on the comm stream.

        Called before gate computation so the AllGather overlaps with the gate
        MLP on the calc stream.  Result is stored in self._pre_ag_handle and
        consumed by dispatch_preprocess.

        bf16 path: single async AllGather.
        fp8 path: quantize + a single fused async AllGather of (data ++ scale)
            via ``_fused_fp8_all_gather_async``.
        """
        if self.moe_group is None or self.moe_group.nranks == 1:
            self._pre_ag_handle = None
            return

        # Drain leftover handle from a possibly-aborted previous forward.
        if self._pre_ag_handle is not None:
            try:
                self._pre_ag_handle["task"].wait()
            except (RuntimeError, OSError) as _e:
                logger.warning(
                    "pre_allgather: leftover async task wait failed (%s), "
                    "discarding handle.",
                    _e,
                )
            self._pre_ag_handle = None

        if len(hidden_states.shape) == 3:
            _, _, d_model = hidden_states.shape
        else:
            _, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model]).contiguous()

        if self.fp8_dispatch:
            (
                fused_global,
                H,
                H128,
                scale_dtype,
                task,
            ) = _fused_fp8_all_gather_async(reshaped_input, self.moe_group)
            self._pre_ag_handle = {
                "fused_global": fused_global,
                "H": H,
                "H128": H128,
                "scale_dtype": scale_dtype,
                "task": task,
                "group": self.moe_group,
                "fp8": True,
            }
            return

        output_shape = list(reshaped_input.shape)
        output_shape[0] = output_shape[0] * self.moe_group.nranks
        global_hidden_states = paddle.empty(
            shape=output_shape, dtype=reshaped_input.dtype
        )
        task = paddle.distributed.stream.all_gather(
            global_hidden_states,
            reshaped_input,
            group=self.moe_group,
            sync_op=False,
            use_calc_stream=False,
        )

        self._pre_ag_handle = {
            "output": global_hidden_states,
            "task": task,
            "group": self.moe_group,
        }

    def dispatch_preprocess(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        mask: paddle.Tensor,  # routing_map
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        """AllGather hidden_states and routing metadata across the EP group.

        Steps:
        1. Reshape to [T_local, H].
        2. AllGather hidden_states (reuse pre-issued async handle if available).
        3. AllGather topk_indices async on comm stream (int32, no gradient).
        4. AllGather topk_weights via _RouterAllGather on calc stream (has
           gradient, backward = reduce-scatter).
        5. Wait for indices, build padding mask (indices < 0), zero padding weights.
        6. Return global hidden_states (unpermuted — SonicMoE handles gather).

        Caches _global_topk_indices, _global_topk_weights for downstream use.

        Note:
            If ``_pre_ag_handle`` is None on entry (gate-overlap did not fire,
            e.g. ``moe_allgather_gate_overlap=False`` or a direct call that
            bypasses ``_maybe_pre_allgather_overlap``) and ``fp8_dispatch`` is
            True, this method issues the fp8 AllGather inline via
            ``pre_allgather`` and immediately waits on it.  This is correct but
            forfeits the gate-compute overlap that the pre-issued path provides.
        """
        if len(hidden_states.shape) == 3:
            _, _, d_model = hidden_states.shape
        else:
            _, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model]).contiguous()

        self._fp8_dispatch_scale = None

        if self._pre_ag_handle is not None:
            if self._pre_ag_handle.get("fp8", False):
                global_hidden_states, self._fp8_dispatch_scale = (
                    _PreAllGatherFP8Result.apply(
                        reshaped_input, self._pre_ag_handle
                    )
                )
            else:
                global_hidden_states = _PreAllGatherResult.apply(
                    reshaped_input, self._pre_ag_handle
                )
            self._pre_ag_handle = None
        elif self.fp8_dispatch:
            self.pre_allgather(reshaped_input)
            global_hidden_states, self._fp8_dispatch_scale = (
                _PreAllGatherFP8Result.apply(
                    reshaped_input, self._pre_ag_handle
                )
            )
            self._pre_ag_handle = None
        else:
            global_hidden_states = AllGatherGroupOp.apply(
                reshaped_input, self.moe_group
            )

        if topk_indices is None or topk_weights is None:
            raise ValueError(
                "AllGatherTokenDispatcher requires topk_indices and "
                "topk_weights to be provided."
            )
        # AllGather indices as int32 on comm stream (async, no gradient).
        # Issued before weights AllGather so both collectives are in flight.
        topk_indices_i32 = topk_indices.detach().cast("int32").contiguous()
        if self.moe_group is None or self.moe_group.nranks == 1:
            self._global_topk_indices = topk_indices_i32.clone()
            _idx_task = None
        else:
            _idx_out_shape = list(topk_indices_i32.shape)
            _idx_out_shape[0] *= self.moe_group.nranks
            self._global_topk_indices = paddle.empty(
                shape=_idx_out_shape, dtype=topk_indices_i32.dtype
            )
            _idx_task = paddle.distributed.stream.all_gather(
                self._global_topk_indices,
                topk_indices_i32,
                group=self.moe_group,
                sync_op=False,
                use_calc_stream=False,
            )
        # AllGather router weights on calc stream (has gradient).
        self._global_topk_weights = _RouterAllGather.apply(
            topk_weights.cast(probs.dtype), self.moe_group
        )
        # Wait for indices AllGather right before its first consumer.
        if _idx_task is not None:
            _idx_task.wait()
        # Build padding mask and zero corresponding weights.
        padding_mask = self._global_topk_indices < 0
        self._global_topk_weights = paddle.where(
            padding_mask,
            paddle.zeros_like(self._global_topk_weights),
            self._global_topk_weights,
        )
        self.tokens_per_expert = None
        return global_hidden_states

    def token_dispatch(
        self,
        permuted_global_input_tokens: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = True,
    ):
        """No-op pass-through.  AllGather already happened in dispatch_preprocess,
        so every rank already holds the full global token list.  Returns
        (tokens, fp8_handle) where fp8_handle carries the dispatch scale if
        fp8_dispatch is active."""
        if not using_sonic_moe:
            raise ValueError(
                "AllGatherTokenDispatcher requires using_sonic_moe=True; "
                "the AllGather path is only wired for the fused SonicMoE "
                "expert kernels. Switch dispatcher type or enable SonicMoE."
            )
        fp8_handle = (
            {"scale": self._fp8_dispatch_scale}
            if self._fp8_dispatch_scale is not None
            else None
        )
        return permuted_global_input_tokens, fp8_handle

    def get_dispatched_routing(self):
        """Return (global_indices, global_weights, tokens_per_expert).

        tokens_per_expert uses a sync-free scatter histogram
        (:func:`_tokens_per_expert_histogram`) — no GPU->CPU sync, no
        full one-hot materialization.
        """
        tokens_per_expert = _tokens_per_expert_histogram(
            self._global_topk_indices, self.num_experts
        )
        return (
            self._global_topk_indices,
            self._global_topk_weights,
            tokens_per_expert,
        )

    def dispatch_postprocess(
        self,
        global_input_tokens: paddle.Tensor,
    ):
        """Return (global_tokens, tokens_per_expert). tokens_per_expert is None
        on this path — SonicMoE kernels recompute it from indices."""
        return global_input_tokens, self.tokens_per_expert

    def combine_preprocess(self, hidden_states: paddle.Tensor):
        """No-op pass-through."""
        return hidden_states

    def token_combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        fp8_combine_grad_handle: dict | None = None,
    ):
        """Combine expert outputs via ReduceScatter.

        If combine_overlap_handle is provided, fuse the ReduceScatter with the
        shared-expert subgraph via _AllGatherCombineAsync for overlap.  The
        combined output is cached for combine_postprocess to return.

        fp8_combine_grad_handle, when non-None, enables fp8 quantization of the
        combine backward gradient (halves bandwidth vs bf16).  The gathered fp8
        data+scale are written into the handle for _DownProjection.backward.
        """
        if combine_overlap_handle is None:
            # Must wrap in a PyLayer so backward populates
            # fp8_combine_grad_handle for _DownProjection.backward.
            combined_x = _AllGatherCombineNoOverlap.apply(
                hidden_states, self.moe_group, fp8_combine_grad_handle
            )
            self._overlap_combined = combined_x
            return combined_x
        if not isinstance(combine_overlap_handle, dict):
            raise TypeError(
                "combine_overlap_handle must be a dict, got "
                f"{type(combine_overlap_handle).__name__}"
            )
        if (
            "fn" not in combine_overlap_handle
            or "fn_args" not in combine_overlap_handle
        ):
            raise ValueError(
                "combine_overlap_handle must contain 'fn' and 'fn_args' keys"
            )
        if not isinstance(combine_overlap_handle["fn_args"], tuple):
            raise TypeError(
                "combine_overlap_handle['fn_args'] must be a tuple, got "
                f"{type(combine_overlap_handle['fn_args']).__name__}"
            )
        from paddle import framework as _framework

        combined_x, *fn_out = _AllGatherCombineAsync.apply(
            hidden_states,
            self.moe_group,
            *(combine_overlap_handle["fn_args"]),
            fn=combine_overlap_handle["fn"],
            is_first_fwd=not _framework._dygraph_tracer()._has_grad,
            fp8_combine_grad_handle=fp8_combine_grad_handle,
        )
        combine_overlap_handle["fn_out"] = tuple(fn_out)
        self._overlap_combined = combined_x
        return combined_x

    def combine_postprocess(self, hidden_states: paddle.Tensor):
        """Return cached ReduceScatter result from token_combine.

        token_combine sets _overlap_combined on both paths (overlap and
        no-overlap), so the fallback ReduceScatterGroupOp is defensive only.
        """
        if getattr(self, "_overlap_combined", None) is not None:
            out = self._overlap_combined
            self._overlap_combined = None
            return out
        return ReduceScatterGroupOp.apply(hidden_states, self.moe_group)
