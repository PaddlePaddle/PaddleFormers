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

import contextlib
import copy
import logging

import paddle

import paddlefleet_ops
from paddleformers.fleet.transformer.moe.fp8_utils import (
    ExpertsGroupGemmContiguousNode,
)

from .fp8_utils import (
    FP8_ALIGN,
    USE_INPLACE_SWIGLU_BWD,
    moe_token_padding_alignment,
    tilewise_quant,
)
from .moe_utils import get_auto_sb_history
from .vmm_utils import (
    allocator_free_block_info,
    auto_subbatch_allocator_backend,
    find_max_concurrent_subbatch_size,
    find_max_sequence_subbatch_size,
    merge_subbatch_cast,
    tokens_zip_unique_add_with_subbatch,
)

_scatter_router_scores_i32 = None
if paddlefleet_ops.is_sonic_moe_available():
    from paddlefleet_ops.sonicmoe.enums import ActivationType
    from paddlefleet_ops.sonicmoe.ernie_compat.deepep_metadata import (
        deepep_topk_to_sonic_metadata,
    )

    try:
        from paddlefleet_ops.sonicmoe.ernie_compat.deepep_metadata import (
            deepep_topk_to_sonic_metadata_with_scales,
        )
    except ImportError:
        # Older installed paddlefleet_ops binaries (pre-#1348) do not export
        # the fp8-scales variant; the sibling optional imports below use the
        # same guard. Only the fp8 + fp8_scale MoE path calls it, which this
        # config does not exercise.
        deepep_topk_to_sonic_metadata_with_scales = None
    from paddlefleet_ops.sonicmoe.ernie_compat.mlp_node_v2 import (
        _differentiable_router_scores,
    )
    from paddlefleet_ops.sonicmoe.functional import (
        _DownProjection,
        _UpProjection,
    )
    from paddlefleet_ops.sonicmoe.functional.utils import enable_fp8

    try:
        from paddlefleet_ops.sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            _scatter_router_scores_i32,
        )
    except (ImportError, RuntimeError):
        pass

logger = logging.getLogger(__name__)


def _copy_sonic_fp8_weight_attrs(src, dst):
    for attr_name in ("fp8", "transposed_fp8"):
        attr = getattr(src, attr_name, None)
        if attr is None:
            raise RuntimeError(
                f"Sonic FP8 forward requires weight.{attr_name} attribute; "
                "call quant_weight() before FP8 forward."
            )
        setattr(dst, attr_name, attr)


class _SonicFP8WeightCarrier(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, weight):
        carrier = paddle.empty([1], dtype=weight.dtype).as_strided(
            [weight.shape[1], weight.shape[2], weight.shape[0]], [0, 0, 0]
        )
        carrier.stop_gradient = weight.stop_gradient
        _copy_sonic_fp8_weight_attrs(weight, carrier)
        return carrier

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None
        return grad.transpose([2, 0, 1])


def _make_sonic_fp8_weight_carrier(weight):
    return _SonicFP8WeightCarrier.apply(weight)


class _SonicRouterScoresFromMetadata(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, topk_scores, metadata_scores, score_src_idx):
        if len(topk_scores.shape) != 2:
            raise ValueError(
                f"topk_scores: expected rank 2, got shape {topk_scores.shape}"
            )
        if len(metadata_scores.shape) != 1:
            raise ValueError(
                "metadata_scores: expected rank 1, got shape "
                f"{metadata_scores.shape}"
            )
        if len(score_src_idx.shape) != 1:
            raise ValueError(
                f"score_src_idx: expected rank 1, got shape {score_src_idx.shape}"
            )
        if metadata_scores.shape[0] < score_src_idx.shape[0]:
            raise ValueError(
                "metadata_scores must include every real score referenced by "
                f"score_src_idx; got {metadata_scores.shape[0]} scores and "
                f"{score_src_idx.shape[0]} indices"
            )
        if "int32" not in str(score_src_idx.dtype):
            raise ValueError(
                f"score_src_idx: expected int32, got {score_src_idx.dtype}"
            )
        metadata_scores.stop_gradient = True
        score_src_idx.stop_gradient = True
        ctx.save_for_backward(score_src_idx)
        ctx.input_shape = list(topk_scores.shape)
        ctx.n_total = int(topk_scores.shape[0]) * int(topk_scores.shape[1])
        scores = metadata_scores.clone()
        scores.stop_gradient = topk_scores.stop_gradient
        return scores

    @staticmethod
    def backward(ctx, grad_out):
        (score_src_idx,) = ctx.saved_tensor()
        if _scatter_router_scores_i32 is None:
            raise RuntimeError(
                "SonicMoE metadata router score backward requires "
                "paddlefleet_ops.sonicmoe.quack_utils.blockscaled_fp8_gemm."
                "_scatter_router_scores_i32; update paddlefleet_ops or use "
                "the differentiable router-score fallback."
            )
        grad_flat = _scatter_router_scores_i32(
            grad_out.contiguous(), score_src_idx, ctx.n_total
        )
        return grad_flat.reshape(ctx.input_shape), None, None


class UnZipNode:
    """
    UnZipNode 类用于对输入的token 矩阵根据分发索引进行解压操作,得到专家需要处理的 token。
    """

    def __init__(self, token_dispatcher, name="unzip"):
        self.token_dispatcher = token_dispatcher
        self.name = name
        self.unzipped_probs = None
        self.zipped_expertwise_rowmap = None

    def reset_state(self):
        """
        重置模型的状态。

        Args:
            无

        Returns:
            无

        """
        self.unzipped_probs = None
        self.zipped_expertwise_rowmap = None

    def cached_tensors(self):
        """
        cached_tensors
        """
        return [self.unzipped_probs, self.zipped_expertwise_rowmap]

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        self.unzipped_probs, self.zipped_expertwise_rowmap = tensors

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        self.set_cached_tensors([None] * len(self.cached_tensors()))

    @paddle.no_grad()
    def forward(
        self,
        hs_2d_dispatched,
        dispatched_indices,
        dispatched_probs,
        topk,
        num_experts,
        tokens_per_expert,
        fill_output=True,
        padding_alignment=FP8_ALIGN,
    ):
        """
        前向传播函数，用于解压输入的张量。

        Args:
            hs_2d_dispatched: 原始输入的token。
            dispatched_indices: 分发索引。
            dispatched_probs: 分发概率。

        Returns:
            tuple: 返回解压后的令牌、压缩后的专家行映射、解压后的概率。
        """
        if isinstance(hs_2d_dispatched, tuple):
            assert len(hs_2d_dispatched) == 2, (
                f"hs_2d_dispatched should has at most 2 tensors, but bot {len(hs_2d_dispatched)}"
            )
            hidden_states, scale = hs_2d_dispatched
        else:
            hidden_states, scale = hs_2d_dispatched, None

        with paddle.amp.auto_cast(False):
            using_ue8m0_scale = (
                scale is not None and scale.dtype == paddle.int32
            )
            (
                unzipped_tokens,
                zipped_expertwise_rowmap,
                unzipped_probs,
                unzipped_scale,
            ) = paddle.nn.functional.moe_permute(
                hidden_states,
                scale,
                dispatched_indices,
                dispatched_probs,
                num_experts=num_experts,
                tokens_per_expert=tokens_per_expert,
                padding_alignment=padding_alignment,
                do_gather=fill_output,
                using_ue8m0_scale=using_ue8m0_scale,
            )

        if scale is None:
            # NOTE: 由于自定义算子不能返回None, 所以scale为None时
            # unzipped_scale会返回一个0shape的fake ouutput
            assert unzipped_scale.shape[0] == 0
            unzipped_scale = None

        self.unzipped_probs = unzipped_probs
        self.zipped_expertwise_rowmap = zipped_expertwise_rowmap
        return (
            unzipped_tokens,
            zipped_expertwise_rowmap,
            unzipped_probs,
            unzipped_scale,
        )

    @paddle.no_grad()
    def backward(
        self,
        dx,
        hidden_states_out_grad_shape,
        probs_grad,
        dispatched_indices,
        num_experts,
    ):
        with paddle.amp.auto_cast(False):
            weighted_zipped_tokens, probs_grad_zipped = (
                paddle.nn.functional.moe_unpermute(
                    dx,
                    self.zipped_expertwise_rowmap,
                    dispatched_indices,
                    probs_grad,
                    total_zipped_tokens=hidden_states_out_grad_shape[0],
                    num_experts=num_experts,
                )
            )
        self.reset_state()
        return weighted_zipped_tokens, probs_grad_zipped


class ZipNode:
    """
    与 UnzipNode 相反，类用将解压后的 token 张量压缩回原始状态。
    """

    def __init__(self, token_dispatcher, name="zip"):
        self.token_dispatcher = token_dispatcher
        self.name = name

    def cached_tensors(self):
        """
        cached_tensors
        """
        return []

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        assert len(tensors) == 0

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        pass

    @paddle.no_grad()
    def forward(
        self,
        expert_out,
        zipped_expertwise_rowmap,
        routemap_topk,
        unzipped_probs,
        total_zipped_tokens,
        num_experts,
    ):
        with paddle.amp.auto_cast(False):
            expert_out_zipped, zipped_probs_topk = (
                paddle.nn.functional.moe_unpermute(
                    expert_out,
                    zipped_expertwise_rowmap,
                    routemap_topk,
                    unzipped_probs,
                    total_zipped_tokens,
                    num_experts,
                )
            )
        return expert_out_zipped

    @paddle.no_grad()
    def backward(
        self,
        grad_output,
        dispatched_indices,
        dispatched_probs,
        top_k,
        num_experts,
        tokens_per_expert,
        fill_output=True,
        padding_alignment=FP8_ALIGN,
    ):
        with paddle.amp.auto_cast(False):
            (
                unzipped_grad,
                zipped_expertwise_rowmap_grad,
                unzipped_probs_grad,
                unzipped_scale_grad,
            ) = paddle.nn.functional.moe_permute(
                grad_output,
                None,
                dispatched_indices,
                dispatched_probs,
                num_experts,
                tokens_per_expert,
                padding_alignment=padding_alignment,
                do_gather=fill_output,
            )
        return unzipped_grad


class MlpNode:
    """
    The FusedMoeLayer class includes operations for unzipping, expert computation, and zipping.
    """

    def __init__(
        self,
        custom_map,
        num_experts_per_tok,
        recompute_moe_gate_up=False,
        dequant_input=False,
        moe_expert_fusion=False,
        recompute_moe_premute=False,
        moe_subbatch_token_num_after_dispatch=None,
        use_bf16_gemm_weight_grad=False,
        use_fp8_mlp=True,
        moe_deep_gemm=False,
        use_auto_subbatch=False,
        auto_subbatch_mode=None,
        moe_subbatch_diag=False,
        use_ue8m0=False,
        dw_p2p_overlap=False,
        clamp_value=None,
        activation_type=None,
        use_accuracy_compatible=False,
    ):
        """
        Constructor
        """
        self.token_dispatcher = custom_map.token_dispatcher
        self.moe_expert_fusion = moe_expert_fusion
        self.experts = getattr(custom_map, "experts", None)
        if activation_type is None:
            activation_type = getattr(custom_map, "_activation_type", "swiglu")
        self.activation_type = activation_type

        self.moe_rank = getattr(custom_map, "moe_rank", 0)
        self.tokens_per_expert = (
            self.token_dispatcher._comm_manager.tokens_per_expert
        )
        self.num_experts_per_device = getattr(
            custom_map,
            "num_experts_per_device",
            len(self.tokens_per_expert),
        )
        if recompute_moe_premute:
            assert moe_expert_fusion == moe_deep_gemm, (
                f"recompute_moe_premute requires moe_expert_fusion == moe_deep_gemm"
                f" (got moe_expert_fusion={moe_expert_fusion}, moe_deep_gemm={moe_deep_gemm})"
            )
            assert recompute_moe_gate_up, (
                "recompute_moe_gate_up must be enabled when recompute_moe_premute = True"
            )
            assert dequant_input, (
                "dequant_input must be enabled with recompute_moe_premute = True"
            )
        self.recompute_moe_premute = recompute_moe_premute

        self.moe_subbatch_token_num_after_dispatch = (
            moe_subbatch_token_num_after_dispatch
        )
        _has_static_subbatch = (
            self.moe_subbatch_token_num_after_dispatch is not None
            and self.moe_subbatch_token_num_after_dispatch > 0
        )

        if _has_static_subbatch:
            assert (
                self.moe_subbatch_token_num_after_dispatch % FP8_ALIGN == 0
            ), self.moe_subbatch_token_num_after_dispatch
            assert moe_expert_fusion == moe_deep_gemm, (
                "static subbatch requires moe_expert_fusion == moe_deep_gemm"
                f" (got moe_expert_fusion={moe_expert_fusion}, moe_deep_gemm={moe_deep_gemm})"
            )
            assert dequant_input, (
                "dequant_input must be enabled when moe_subbatch_token_num_after_dispatch > 0"
            )
            if not recompute_moe_gate_up:
                logger.warning(
                    "Auto-enabling recompute_moe_gate_up: static subbatch splits"
                    " forward into multiple chunks that overwrite intermediate"
                    " activations (o1), recompute is required to reconstruct them"
                    " during backward."
                )
                recompute_moe_gate_up = True

        valid_auto_subbatch_modes = {None, "post_permute", "pre_permute"}
        if auto_subbatch_mode not in valid_auto_subbatch_modes:
            raise ValueError(
                "auto_subbatch_mode must be one of None, 'post_permute', "
                f"or 'pre_permute', got {auto_subbatch_mode!r}"
            )
        if use_auto_subbatch and auto_subbatch_mode == "pre_permute":
            assert moe_expert_fusion, (
                "auto_subbatch_mode='pre_permute' requires "
                f"moe_expert_fusion=True, got {moe_expert_fusion}"
            )

        # Per-expert gemm node list is needed when:
        #   no subbatch:     never
        #   static subbatch: always (regardless of deep_gemm)
        #   auto_subbatch:   only when moe_expert_fusion=False (fusion mode uses runtime fallback)
        #   pre_permute:     uses fusion node (group_gemm per chunk), no per-expert list needed
        _need_per_expert_nodes = _has_static_subbatch or (
            use_auto_subbatch and not moe_expert_fusion
        )
        if _need_per_expert_nodes:
            # For deep_gemm: set moe_expert_fusion=False at MlpNode level to route into
            # per-expert loop; each node internally keeps moe_expert_fusion=True for
            # grouped gemm API with sliced [1,K,N] weight.
            if moe_deep_gemm:
                self.moe_expert_fusion = False
            self.experts_group_gemm_node = [
                ExpertsGroupGemmContiguousNode(
                    custom_map,
                    recompute_moe_gate_up=recompute_moe_gate_up,
                    dequant_input=dequant_input,
                    expert_id=self._global_expert_id(local_expert_id),
                    moe_subbatch_token_num_after_dispatch=moe_subbatch_token_num_after_dispatch,
                    use_bf16_gemm_weight_grad=use_bf16_gemm_weight_grad,
                    use_fp8_mlp=use_fp8_mlp,
                    moe_deep_gemm=moe_deep_gemm,
                    use_ue8m0=use_ue8m0,
                    dw_p2p_overlap=dw_p2p_overlap,
                    moe_expert_fusion=moe_expert_fusion,
                    clamp_value=clamp_value,
                    activation_type=activation_type,
                    use_accuracy_compatible=use_accuracy_compatible,
                )
                for local_expert_id in range(self.num_experts_per_device)
            ]
        else:
            self.experts_group_gemm_node = ExpertsGroupGemmContiguousNode(
                custom_map,
                recompute_moe_gate_up=recompute_moe_gate_up,
                dequant_input=dequant_input,
                moe_subbatch_token_num_after_dispatch=moe_subbatch_token_num_after_dispatch,
                use_bf16_gemm_weight_grad=use_bf16_gemm_weight_grad,
                use_fp8_mlp=use_fp8_mlp,
                moe_deep_gemm=moe_deep_gemm,
                use_ue8m0=use_ue8m0,
                dw_p2p_overlap=dw_p2p_overlap,
                moe_expert_fusion=moe_expert_fusion,
                clamp_value=clamp_value,
                activation_type=activation_type,
                use_accuracy_compatible=use_accuracy_compatible,
            )
        self.unzip_node = UnZipNode(self.token_dispatcher)
        self.zip_node = ZipNode(self.token_dispatcher)
        self.hs_2d_dispatched_fp8 = None
        self.hs_2d_dispatched_scale = None
        self.hs_2d_dispatched_bf16 = None
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.unzipped_probs = None
        # == [MG accuracy-alignment diff · ref PF PR#968] per-expert padding alignment ==
        # When use_accuracy_compatible=True and non-fp8/non-grouped_gemm, use
        #   alignment=1 (real token count) so the permute and per-expert GEMM M dim
        #   equal the real tokens_per_expert and cuBLAS picks the same algorithm as
        #   MG SequentialMLP; otherwise align to FP8_ALIGN (kernel requirement) to
        #   preserve the original behavior.
        self.moe_permute_padding_alignment = moe_token_padding_alignment(
            use_fp8_mlp=use_fp8_mlp,
            moe_grouped_gemm=moe_expert_fusion,
            use_accuracy_compatible=use_accuracy_compatible,
        )
        self.padding_token_per_experts = [
            (x + self.moe_permute_padding_alignment - 1)
            // self.moe_permute_padding_alignment
            * self.moe_permute_padding_alignment
            for x in self.tokens_per_expert
        ]
        self.token_offsets = [0]
        for padding_token in self.padding_token_per_experts:
            self.token_offsets.append(self.token_offsets[-1] + padding_token)
        self.router_topk = num_experts_per_tok
        self.use_fp8_mlp = use_fp8_mlp
        self.use_auto_subbatch = use_auto_subbatch
        self.moe_subbatch_diag = moe_subbatch_diag
        # Resolve effective auto_subbatch_mode. use_auto_subbatch is the
        # master switch; auto_subbatch_mode only selects the enabled strategy.
        if use_auto_subbatch:
            self.auto_subbatch_mode = auto_subbatch_mode or "post_permute"
        else:
            self.auto_subbatch_mode = None
        if self.moe_subbatch_token_num_after_dispatch is not None:
            self.min_auto_subbatch_rows = (
                self.moe_subbatch_token_num_after_dispatch
            )
        else:
            self.min_auto_subbatch_rows = FP8_ALIGN**2 // 2

    def _global_expert_id(self, local_expert_id):
        return self.moe_rank * self.num_experts_per_device + local_expert_id

    def _gemm_node(self, local_expert_id):
        return self.experts_group_gemm_node[local_expert_id]

    def cached_tensors(self):
        """
        cached tensors
        """
        if self.experts_group_gemm_node is not None:
            if isinstance(self.experts_group_gemm_node, list):
                gemm_node_tensors = []
                for gemm_node in self.experts_group_gemm_node:
                    gemm_node_tensors.extend(gemm_node.cached_tensors())
            else:
                gemm_node_tensors = (
                    self.experts_group_gemm_node.cached_tensors()
                )
        else:
            gemm_node_tensors = []

        return (
            gemm_node_tensors
            + self.unzip_node.cached_tensors()
            + self.zip_node.cached_tensors()
            + [
                self.hs_2d_dispatched_fp8,
                self.hs_2d_dispatched_scale,
                self.dispatched_indices,
                self.dispatched_probs,
                self.unzipped_probs,
                self.tokens_per_expert,
                self.router_topk,
            ]
        )

    def set_cached_tensors(self, tensors):
        """
        set_cached_tensors
        """
        idx = 0
        if self.experts_group_gemm_node is not None:
            if isinstance(self.experts_group_gemm_node, list):
                for expert_id, gemm_node in enumerate(
                    self.experts_group_gemm_node
                ):
                    num = len(gemm_node.cached_tensors())
                    gemm_node.set_cached_tensors(tensors[idx : idx + num])
                    idx += num
            else:
                num = len(self.experts_group_gemm_node.cached_tensors())
                self.experts_group_gemm_node.set_cached_tensors(
                    tensors[idx : idx + num]
                )
                idx += num

        num = len(self.unzip_node.cached_tensors())
        self.unzip_node.set_cached_tensors(tensors[idx : idx + num])
        idx += num

        num = len(self.zip_node.cached_tensors())
        self.zip_node.set_cached_tensors(tensors[idx : idx + num])
        idx += num

        (
            self.hs_2d_dispatched_fp8,
            self.hs_2d_dispatched_scale,
            self.dispatched_indices,
            self.dispatched_probs,
            self.unzipped_probs,
            self.tokens_per_expert,
            self.router_topk,
        ) = tensors[idx:]

    def clear_cached_tensors(self):
        """
        clear_cached_tensors
        """
        self.set_cached_tensors([None] * len(self.cached_tensors()))

    def reset_state(self):
        """
        重置所有状态变量。

        Args:
            无。

        Returns:
            无。

        """
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.unzipped_probs = None
        self.tokens_per_expert = None
        self.padding_token_per_experts = None
        self.router_topk = None
        self.unzip_node.reset_state()
        self._pre_permute_cached_chunks = None
        self._pre_permute_chunk_bounds = None
        self.release_mem()

    def release_mem(self):
        """
            释放内存，将变量置为None。
        这个函数应该在程序结束时调用，以便释放不再需要的资源。

        Args:
            无参数。

        Returns:
            无返回值，直接修改了类实例中的变量。
        """
        if isinstance(self.experts_group_gemm_node, list):
            for node in self.experts_group_gemm_node:
                node.reset_state()
        else:
            self.experts_group_gemm_node.reset_state()
        self.experts_group_gemm_node = None

    # ==================== auto_subbatch helper methods ====================

    def subbatch_unzip_and_prepare_gemm_node(
        self, hs_2d_dispatched, zipped_expertwise_rowmap, expert_id
    ):
        """
        zip_unzip_fusion=False 时的单专家输入准备：
        从 zipped 空间 gather 出该专家的 token，设置到 gemm_node 上。
        返回 expert_unzipped_idx，供后续 scatter-add回zipped空间使用。

        Example (expert_id=1, tokens_per_expert=[2,3,1], FP8_ALIGN=4):

            zipped 空间 (hs_2d_dispatched):     unzip gather 后 (expert_id=1):
            ┌─────────────┐                     ┌─────────────┐
            │ tok0 (E0,E1)│ ──────────────────► │ tok0        │ row 0
            │ tok1 (E0)   │                     │ tok2        │ row 1
            │ tok2 (E1,E2)│ ──────────────────► │ tok4        │ row 2
            │ tok3 (E0)   │                     │ <pad>       │ row 3 (pad to FP8_ALIGN=4)
            │ tok4 (E1)   │ ──────────────────► └─────────────┘
            │ tok5 (E2)   │                     gemm_node.input_fp8 = 上面 4 行
            └─────────────┘                     expert_unzipped_idx = [0, 2, 4]
        """
        hs_2d_dispatched, hs_2d_dispatched_scale = hs_2d_dispatched
        # 从 zipped 空间按 expert_id gather，输出已 pad 到 FP8_ALIGN 对齐
        (
            expert_out,
            expert_out_scale,
            expert_unzipped_idx,
        ) = paddlefleet_ops.tokens_unzip_gather(
            hs_2d_dispatched,
            hs_2d_dispatched_scale,
            zipped_expertwise_rowmap,
            expert_id,
            self.tokens_per_expert,
            FP8_ALIGN,
        )
        # 将 gather 出的输入设置到对应专家的 gemm_node 上
        gemm_node = self._gemm_node(expert_id)
        if self.use_fp8_mlp is not None:
            gemm_node.input_fp8 = expert_out
            gemm_node.input_scale = expert_out_scale
        else:
            expert_out = paddle.incubate.nn.functional.fused_act_dequant(
                expert_out, expert_out_scale
            )
            gemm_node.input = expert_out
        return expert_unzipped_idx

    def subbatch_prepare_gemm_node(self, unzipped_hs_2d, expert_id):
        """
        Prepare input for this node. Dequant if needed.
        """
        input_fp8, input_scale = unzipped_hs_2d
        gemm_node = self._gemm_node(expert_id)
        if self.use_fp8_mlp is not None:
            gemm_node.input_fp8 = input_fp8
            gemm_node.input_scale = input_scale
        else:
            gemm_node.input = paddle.incubate.nn.functional.fused_act_dequant(
                input_fp8, input_scale
            )

    def gemm_forward_subbatch(
        self,
        expert_id,
        unzipped_probs,
        unzipped_idx,
        output,
        total_zipped_tokens,
        unzipped_out=None,
        start_idx=None,
        end_idx=None,
    ):
        """
        对单个专家执行一次（或一个 subbatch 的）前向 GEMM，并将结果写回输出。

        Example (expert_id=1, 该专家有 300 个 token, subbatch_rows=128):

            gemm_node.input_fp8 (300 tokens, padded to 384):
            ┌──────────────────────────────────────────────┐
            │ tok0 ... tok127 │ tok128 ... tok255 │ tok256 ... tok299 + pad │
            └──────────────────────────────────────────────┘
                 subbatch 0        subbatch 1         subbatch 2
              start=0,end=128    start=128,end=256   start=256,end=300

            每个 subbatch 独立执行:
            1. _slice 截取输入/probs/输出 → 临时替换 gemm_node 上的引用
            2. gemm_node.forward: gate_up → SwiGLU → down_proj
            3. 写回结果:
               - unzipped_out 非 None → in-place 写入预分配 buffer (zip_unzip_fusion)
               - unzipped_out 为 None → scatter-add 到 float32 累加器
            4. 恢复 gemm_node 的原始引用，下一个 subbatch 再切

        Args:
            expert_id: 专家编号。
            unzipped_probs: 该专家的 token 权重。
            unzipped_idx: scatter-add 回 zipped 空间的索引（zip_unzip_fusion=False 时使用）。
            output: 累加输出 buffer（float32 累加器或 list[Tensor]）。
            total_zipped_tokens: zipped 空间的总 token 数。
            unzipped_out: 预分配的输出 buffer（zip_unzip_fusion=True 时传入，GEMM 结果 in-place 写入）。
            start_idx/end_idx: subbatch 切片范围。None 表示不切片，整个专家一次算完。
        """
        gemm_node = self._gemm_node(expert_id)
        if start_idx is not None:
            # --- subbatch 切片：从完整专家的输入/输出中截取 [start_idx, end_idx) ---
            tokens_per_expert = end_idx - start_idx
            padding_token_per_experts = (
                (tokens_per_expert + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN
            )
            padding_end_idx = start_idx + padding_token_per_experts

            unzipped_probs = unzipped_probs._slice(start_idx, padding_end_idx)
            unzipped_idx = unzipped_idx._slice(start_idx, end_idx)
            if self.use_fp8_mlp is not None:
                origin_input_fp8 = gemm_node.input_fp8
                origin_input_scale = gemm_node.input_scale
                gemm_node.input_fp8 = origin_input_fp8._slice(
                    start_idx, padding_end_idx
                )
                gemm_node.input_scale = origin_input_scale.contiguous()._slice(
                    start_idx, padding_end_idx
                )
            else:
                origin_input = gemm_node.input
                gemm_node.input = origin_input._slice(
                    start_idx, padding_end_idx
                )
            if unzipped_out is not None:
                unzipped_out = unzipped_out._slice(start_idx, padding_end_idx)
            gemm_node.tokens_per_expert = [padding_token_per_experts]
        else:
            # --- 不切片：整个专家一次算完 ---
            tokens_per_expert = self.tokens_per_expert[expert_id]
            padding_token_per_experts = self.padding_token_per_experts[
                expert_id
            ]

        # 执行 gate_up → SwiGLU → down_proj GEMM
        # hs_out=None 表示从 gemm_node.input_fp8 取输入（已在 prepare 阶段设置）
        expert_out = gemm_node.forward(
            None,
            unzipped_probs,
            [padding_token_per_experts],
            output=unzipped_out,
        )

        # recompute_moe_premute 场景下，forward 完成后释放 input_fp8
        if start_idx is None and self.recompute_moe_premute:
            gemm_node.input_fp8 = None
            gemm_node.input_scale = None

        # zip_unzip_fusion=False 时，需要 scatter-add 到 output 累加器，output是一个list
        # zip_unzip_fusion=True 时，结果已 in-place 写入 unzipped_out，无需额外操作
        if unzipped_out is None:
            output = tokens_zip_unique_add_with_subbatch(
                output,
                expert_out,
                unzipped_idx,
                zipped_rows=total_zipped_tokens,
                subbatch_rows=(
                    self.moe_subbatch_token_num_after_dispatch
                    if isinstance(output, paddle.Tensor)
                    else output[0].shape[0]
                ),
            )

        # subbatch 切片后恢复 gemm_node 的原始输入引用
        if start_idx is not None:
            if self.use_fp8_mlp is not None:
                gemm_node.input_fp8 = origin_input_fp8
                gemm_node.input_scale = origin_input_scale
            else:
                gemm_node.input = origin_input
            gemm_node.tokens_per_expert = [
                self.padding_token_per_experts[expert_id]
            ]

        return output

    # ==================== forward methods ====================

    def _ensure_weight_grad(self):
        """Pre-allocate weight grads so VMM free-memory query reflects true availability."""
        if self.experts is not None:
            for expert in self.experts:
                if expert is None:
                    continue
                for weight in (
                    expert.up_gate_proj.weight,
                    expert.down_proj.weight,
                ):
                    grad_attr = (
                        "main_grad" if hasattr(weight, "main_grad") else "grad"
                    )
                    if getattr(weight, grad_attr) is None:
                        setattr(
                            weight,
                            grad_attr,
                            paddle.zeros(weight.shape, dtype=paddle.float32),
                        )
            return

        # deep_gemm: stacked weight
        nodes = self.experts_group_gemm_node
        if isinstance(nodes, list):
            first_sliced = getattr(nodes[0], "grouped_gemm_experts", None)
            if first_sliced is None or not hasattr(first_sliced, "_parent"):
                return
            parent = first_sliced._parent
        else:
            parent = getattr(nodes, "grouped_gemm_experts", None)
            if parent is None:
                return

        for attr in ("weight1", "weight2"):
            pw = getattr(parent, attr)
            grad_attr = "main_grad" if hasattr(pw, "main_grad") else "grad"
            if getattr(pw, grad_attr) is None:
                setattr(
                    pw, grad_attr, paddle.zeros(pw.shape, dtype=paddle.float32)
                )

    def _slice_weight_grad(self):
        """Set up grad views on sliced weights pointing back to parent grad."""
        for gemm_node in self.experts_group_gemm_node:
            sliced = getattr(gemm_node, "grouped_gemm_experts", None)
            if sliced is None or not hasattr(sliced, "_parent"):
                continue
            parent = sliced._parent
            lid = sliced._local_id
            for attr in ("weight1", "weight2"):
                pw = getattr(parent, attr)
                sw = getattr(sliced, attr)
                grad_attr = "main_grad" if hasattr(pw, "main_grad") else "grad"
                if getattr(sw, grad_attr, None) is None:
                    setattr(
                        sw,
                        grad_attr,
                        getattr(pw, grad_attr)._slice(lid, lid + 1),
                    )

    def _gate_up_out_dim(self, hidden_size):
        """Return the gate_up projection output width
        Two expert layouts, both with the output width as the last weight dim:
          - per-expert (non-fused): up_gate_proj.weight [H, 2*inter]
          - grouped deep_gemm:      grouped_gemm_experts.weight1 [E, H, 2*inter]
        Falls back to 2 * hidden_size if the weight cannot be resolved.
        """
        # per-expert (non-fused) path
        if self.experts is not None:
            for expert in self.experts:
                if expert is None:
                    continue
                w = getattr(
                    getattr(expert, "up_gate_proj", None), "weight", None
                )
                if w is not None and len(w.shape) >= 1:
                    return int(w.shape[-1])

        # grouped deep_gemm path (stacked weight1)
        nodes = self.experts_group_gemm_node
        node = nodes[0] if isinstance(nodes, list) else nodes
        parent = getattr(node, "grouped_gemm_experts", None)
        w1 = getattr(parent, "weight1", None) if parent is not None else None
        if w1 is not None and len(w1.shape) >= 1:
            return int(w1.shape[-1])

        return hidden_size * 2

    def _bwd_pre_permute_feature_sizes(
        self, hidden_size, gate_up_out_dim, inter_dim
    ):
        """Byte-size of each concurrent backward buffer per FP8_ALIGN unzipped
        tokens, for find_max_concurrent_subbatch_size.
        """
        gemm_node = self.experts_group_gemm_node
        if isinstance(gemm_node, list):
            gemm_node = gemm_node[0] if gemm_node else None
        use_bf16_wgrad = getattr(gemm_node, "use_bf16_gemm_weight_grad", False)
        # Mirror the actual swiglu-bwd op dispatch (see bwd_down_input_fp8):
        #   clamp_value>0          -> fused_swiglu_weighted_clamp_bwd (out-of-place,
        #                             do1 = empty_like(o1), a separate buffer)
        #   USE_INPLACE_SWIGLU_BWD -> _fused_swiglu_probs_bwd inplace (do1 reuses o1)
        #   otherwise              -> fused_swiglu_weighted_bwd (out-of-place)
        # so clamp forces the out-of-place peak even when USE_INPLACE_SWIGLU_BWD.
        clamp_value = getattr(gemm_node, "clamp_value", None)
        clamp_active = clamp_value is not None and clamp_value > 0
        used_inplace = USE_INPLACE_SWIGLU_BWD and not clamp_active

        if use_bf16_wgrad:
            if used_inplace:
                # do1 reuses o1 buffer (inplace), peak at dw1 dequant:
                #   out_grad(H*2) + o1/do1(inter*2*2) + o2_s(inter*2)
                #   + input_fp8(H) + dequant_x(H*2)
                return [
                    FP8_ALIGN
                    * hidden_size
                    * 2,  # out_grad (permuted_grad) [N,H] bf16
                    FP8_ALIGN
                    * gate_up_out_dim
                    * 2,  # o1/do1 [N,inter*2] bf16 (inplace)
                    FP8_ALIGN
                    * inter_dim
                    * 2,  # o2_s [N,inter] bf16 (held for dw2)
                    FP8_ALIGN
                    * hidden_size,  # permuted_input [N,H] fp8 (alive at dw1)
                    FP8_ALIGN
                    * hidden_size
                    * 2,  # dw1 dequant x [N,H] bf16 (the peak)
                ]
            else:
                # do1 is a separate buffer (out-of-place), o1 stays alive:
                #   out_grad(H*2) + o1(inter*2*2) + do1(inter*2*2) + o2_s(inter*2)
                #   + input_fp8(H) + dequant_x(H*2)
                return [
                    FP8_ALIGN
                    * hidden_size
                    * 2,  # out_grad (permuted_grad) [N,H] bf16
                    FP8_ALIGN * gate_up_out_dim * 2,  # o1 [N,inter*2] bf16
                    FP8_ALIGN
                    * gate_up_out_dim
                    * 2,  # do1 [N,inter*2] bf16 (out-of-place)
                    FP8_ALIGN
                    * inter_dim
                    * 2,  # o2_s [N,inter] bf16 (held for dw2)
                    FP8_ALIGN
                    * hidden_size,  # permuted_input [N,H] fp8 (alive at dw1)
                    FP8_ALIGN
                    * hidden_size
                    * 2,  # dw1 dequant x [N,H] bf16 (the peak)
                ]

        # fp8 wgrad path
        if used_inplace:
            return [
                FP8_ALIGN
                * hidden_size
                * 2,  # out_grad (permuted_grad) [N,H] bf16
                FP8_ALIGN
                * gate_up_out_dim
                * 2,  # o1/do1 [N,inter*2] bf16 (inplace)
                FP8_ALIGN * inter_dim * 2,  # o2_s [N,inter] bf16
                FP8_ALIGN * hidden_size,  # input_x_t_fp8 [N,H] fp8
                FP8_ALIGN * gate_up_out_dim,  # do1_t_fp8 [N,inter*2] fp8
            ]
        else:
            return [
                FP8_ALIGN
                * hidden_size
                * 2,  # out_grad (permuted_grad) [N,H] bf16
                FP8_ALIGN * gate_up_out_dim * 2,  # o1 [N,inter*2] bf16
                FP8_ALIGN
                * gate_up_out_dim
                * 2,  # do1 [N,inter*2] bf16 (out-of-place)
                FP8_ALIGN * inter_dim * 2,  # o2_s [N,inter] bf16
                FP8_ALIGN * hidden_size,  # input_x_t_fp8 [N,H] fp8
                FP8_ALIGN * gate_up_out_dim,  # do1_t_fp8 [N,inter*2] fp8
            ]

    def _fwd_pre_permute_feature_sizes(
        self, hidden_size, gate_up_out_dim, inter_dim, unpermute_tmp_per_N
    ):
        """Byte-size of each concurrent forward buffer per FP8_ALIGN unzipped
        tokens, for find_max_concurrent_subbatch_size.
        """
        return [
            # max(o1 [N,2*inter], o3 [N,H]) bf16 (not concurrent: clear_o1)
            FP8_ALIGN * max(gate_up_out_dim * 2, hidden_size * 2),
            FP8_ALIGN * hidden_size,  # permuted_input [N,H] fp8
            FP8_ALIGN * inter_dim,  # o2_fp8 [N,inter] fp8
            FP8_ALIGN * unpermute_tmp_per_N,  # unpermute tmp (per-N 折算)
        ]

    def fallback_to_no_expert_fusion(self):
        """
        Fallback from expert_fusion=True to per-expert mode, splitting the fused
        experts_group_gemm_node into individual per-expert nodes.

        Called by auto_subbatch when free memory is insufficient for a single group_gemm.
        Shallow-copies the fused node for each expert and slices forward-saved tensors.
        """
        fused_gemm_node = self.experts_group_gemm_node
        self.experts_group_gemm_node = []
        self.moe_expert_fusion = False

        for local_id, tokens_per_expert in enumerate(
            self.padding_token_per_experts
        ):
            global_expert_id = self._global_expert_id(local_id)
            gemm_node = copy.copy(fused_gemm_node)
            gemm_node.is_split_group_gemm = True
            gemm_node.recompute_moe_gate_up = fused_gemm_node.o1 is None
            gemm_node.expert_id = global_expert_id
            gemm_node.tokens_per_expert = [tokens_per_expert]

            if self.experts is not None:
                # Non deep_gemm: per-expert weight list
                gemm_node.moe_expert_fusion = False
                gemm_node.experts = [self.experts[global_expert_id]]
            else:
                # deep_gemm: slice stacked weight to [1, K, N]
                gemm_node.moe_expert_fusion = True
                parent = fused_gemm_node.grouped_gemm_experts
                if hasattr(parent.weight1, "fp8_weight_stacked"):
                    from paddleformers.fleet.transformer.moe.fp8_utils import (
                        _PerExpertWeightProxy,
                    )

                    gemm_node.grouped_gemm_experts = _PerExpertWeightProxy(
                        parent, local_id
                    )
                else:
                    sliced = type("_SlicedGroupedExpert", (), {})()
                    sliced.weight1 = parent.weight1._slice(
                        local_id, local_id + 1
                    )
                    sliced.weight2 = parent.weight2._slice(
                        local_id, local_id + 1
                    )
                    sliced._parent = parent
                    sliced._local_id = local_id
                    gemm_node.grouped_gemm_experts = sliced
                # Regenerate m_indices for single expert; global m_indices has wrong range
                gemm_node.m_indices = gemm_node.gen_m_indices(
                    [tokens_per_expert]
                )

            self.experts_group_gemm_node.append(gemm_node)

            start_idx = self.token_offsets[local_id]
            end_idx = self.token_offsets[local_id + 1]

            # 如果是在反向才 fallback，需要将前向保存的 input_fp8/o1 切分给每个专家
            if fused_gemm_node.input_fp8 is not None:
                gemm_node.input_fp8 = fused_gemm_node.input_fp8._slice(
                    start_idx, end_idx
                )
                gemm_node.input_scale = (
                    fused_gemm_node.input_scale.contiguous()._slice(
                        start_idx, end_idx
                    )
                )
            if fused_gemm_node.o1 is not None:
                gemm_node.o1 = fused_gemm_node.o1._slice(start_idx, end_idx)

    @contextlib.contextmanager
    def slice_fp8_weight(self, expert_id):
        """
        Temporarily slice FP8 stacked weights for a single expert during per-expert fallback.

        When expert_fusion=True, FP8 weights are stacked on experts[0]. After fallback,
        each expert needs its own weight slice. This context manager sets up and restores them.
        """
        # deep_gemm: weights already sliced in fallback_to_no_expert_fusion
        if self.experts is None:
            yield
            return

        # Stacked FP8 weights live on local expert 0; slice them for the current expert.
        stacked_weight_owner_global_id = self._global_expert_id(0)
        current_expert_global_id = self._global_expert_id(expert_id)

        def has_fp8_weight(expert):
            weight = expert.up_gate_proj.weight
            return getattr(weight, "fp8_weight_stacked", None) is not None

        if not (
            self.num_experts_per_device > 1
            and has_fp8_weight(self.experts[stacked_weight_owner_global_id])
            and not has_fp8_weight(
                self.experts[stacked_weight_owner_global_id + 1]
            )
        ):
            yield
            return

        w1 = self.experts[stacked_weight_owner_global_id].up_gate_proj.weight
        w2 = self.experts[stacked_weight_owner_global_id].down_proj.weight
        w1_weight, w1_scale = w1.fp8_weight_stacked, w1.fp8_scale_stacked
        w2_weight, w2_scale = w2.fp8_weight_stacked, w2.fp8_scale_stacked

        def slice_expert(t):
            chunk_size = t.shape[0] // self.num_experts_per_device
            return t._slice(
                chunk_size * expert_id, chunk_size * (expert_id + 1)
            )

        cur_w1 = self.experts[current_expert_global_id].up_gate_proj.weight
        cur_w2 = self.experts[current_expert_global_id].down_proj.weight
        cur_w1.fp8_weight_stacked = slice_expert(w1_weight)
        cur_w1.fp8_scale_stacked = slice_expert(w1_scale)
        cur_w1.fp8_weight_stacked_transpose = None
        cur_w1.fp8_scale_stacked_transpose = None
        cur_w2.fp8_weight_stacked = slice_expert(w2_weight)
        cur_w2.fp8_scale_stacked = slice_expert(w2_scale)
        cur_w2.fp8_weight_stacked_transpose = None
        cur_w2.fp8_scale_stacked_transpose = None

        try:
            yield
        finally:
            if expert_id == 0:
                w1.fp8_weight_stacked, w1.fp8_scale_stacked = (
                    w1_weight,
                    w1_scale,
                )
                w2.fp8_weight_stacked, w2.fp8_scale_stacked = (
                    w2_weight,
                    w2_scale,
                )
            else:
                del cur_w1.fp8_weight_stacked, cur_w1.fp8_scale_stacked
                del cur_w2.fp8_weight_stacked, cur_w2.fp8_scale_stacked
            del (
                cur_w1.fp8_weight_stacked_transpose,
                cur_w1.fp8_scale_stacked_transpose,
            )
            del (
                cur_w2.fp8_weight_stacked_transpose,
                cur_w2.fp8_scale_stacked_transpose,
            )

    def _prepare_forward(
        self,
        hs_2d_dispatched,
        dispatched_indices,
        dispatched_probs,
        fill_output,
        padding_alignment=None,
    ):
        """
        前向计算的公共预处理，被 forward() 和 forward_auto_subbatch() 共用。
        完成 4 步操作：cast indices → unzip → record_stream → 条件 quant。

        Example (6 tokens, 3 experts, topk=2, FP8_ALIGN=128):

            输入:
              hs_2d_dispatched: [6, 4096] (bf16 或 FP8 tuple)
              dispatched_indices: [6, 2] = [[0,1], [0,-1], [1,2], [0,-1], [1,-1], [2,-1]]
              dispatched_probs:   [6, 2]

            Step 1 - unzip (moe_permute):
              按专家分组 + pad 到 FP8_ALIGN 对齐
              tokens_per_expert = [3, 3, 2]  →  padding 后 = [128, 128, 128]

              fill_output=True 时:
                unzipped_tokens: [384, 4096]  ← 实际拷贝数据（fusion 路径）
              fill_output=False 时:
                unzipped_tokens: [384, 4096]  ← 数据未填充，只计算 rowmap（逐专家 gather 路径）

              zipped_expertwise_rowmap: 索引映射（供后续 gather/scatter 使用）
              unzipped_probs: [384, 1]

            Step 2 - record_stream:
              标记输入 tensor 可被 CUDA stream 异步释放

            Step 3 - 条件 FP8 量化:
              fill_output=False（逐专家路径）:
                tilewise_quant(hs_2d_dispatched) → hs_2d_dispatched_fp8 [6, 4096] (FP8)
                                                   hs_2d_dispatched_scale [6, 32]
                释放原始 bf16 数据
              fill_output=True（fusion 路径）:
                不做量化（直接用 unzipped_tokens），fp8/scale = None

            Step 4 - 保存 recompute 输入:
              recompute_moe_premute=True 时保存 fp8/scale 供反向重算

        Args:
            hs_2d_dispatched: 输入 token。bf16 Tensor 或 (FP8 Tensor, scale) tuple。
            dispatched_indices: 专家分配索引 [S, topk]。
            dispatched_probs: 专家分配权重 [S, topk]。
            fill_output: 控制 moe_permute 是否实际 gather 数据。
                True  → unzipped_tokens 有数据（fusion 路径 / zip_unzip_fusion=True）
                False → 只计算 rowmap（逐专家 gather 路径）

        Returns:
            tuple:
                - use_fp8_dispatch_a2a (bool): 输入是否已经是 FP8（a2a 阶段已量化）。
                - num_experts (int): 专家总数。
                - num_zipped_tokens (int): zipped 空间的 token 数（即原始序列长度 S）。
                - hidden_size (int): 隐藏层维度 H。
                - unzipped_tokens (Tensor): 按专家分组后的 token [N_total_padded, H]。
                    fill_output=True 时有数据，False 时数据未填充（需后续逐专家 gather）。
                - zipped_expertwise_rowmap (Tensor): unzip/zip 的索引映射，供 gather/scatter 使用。
                - unzipped_probs (Tensor): 按专家分组后的权重 [N_total_padded, 1]。
                - unzipped_scale (Tensor|None): FP8 量化 scale，非 FP8 输入时为 None。
                - hs_2d_dispatched_fp8 (Tensor|None): zipped 空间的 FP8 数据。
                    逐专家路径用于后续 gather；fusion 路径为 None。
                - hs_2d_dispatched_scale (Tensor|None): 对应的 FP8 scale。
        """
        use_fp8_dispatch_a2a = isinstance(hs_2d_dispatched, tuple)
        num_experts = len(self.tokens_per_expert)

        # 1. unzip: 按专家分组 + pad 到 FP8_ALIGN 对齐
        self.dispatched_indices = dispatched_indices.to(paddle.int32)
        (
            unzipped_tokens,
            zipped_expertwise_rowmap,
            unzipped_probs,
            unzipped_scale,
        ) = self.unzip_node.forward(
            hs_2d_dispatched,
            self.dispatched_indices,
            dispatched_probs,
            topk=self.router_topk,
            num_experts=num_experts,
            tokens_per_expert=self.tokens_per_expert,
            fill_output=fill_output,
            **(
                {}
                if padding_alignment is None
                else {"padding_alignment": padding_alignment}
            ),
        )
        self.unzipped_probs = unzipped_probs

        # 2. 获取 shape 信息 + record_stream（标记 tensor 可被异步释放）
        if use_fp8_dispatch_a2a:
            num_zipped_tokens = hs_2d_dispatched[0].shape[0]
            hidden_size = hs_2d_dispatched[0].shape[-1]
            hs_2d_dispatched[0]._record_stream()
            hs_2d_dispatched[1]._record_stream()
        else:
            num_zipped_tokens = hs_2d_dispatched.shape[0]
            hidden_size = hs_2d_dispatched.shape[-1]
            hs_2d_dispatched._record_stream()
        dispatched_indices._record_stream()
        dispatched_probs._record_stream()
        if self.dispatched_indices.dtype is not dispatched_indices.dtype:
            dispatched_indices._clear_to_zero_allocation()

        # 3. FP8 量化（逐专家路径需要从 zipped 空间 gather，所以需要量化后的 zipped 数据）
        #    fusion 路径直接使用 unzipped_tokens，不需要量化 zipped 数据
        hs_2d_dispatched_fp8, hs_2d_dispatched_scale = None, None

        if use_fp8_dispatch_a2a:
            hs_2d_dispatched_fp8, hs_2d_dispatched_scale = hs_2d_dispatched
        else:
            if self.use_auto_subbatch:
                hs_2d_dispatched_fp8, hs_2d_dispatched_scale = tilewise_quant(
                    hs_2d_dispatched
                )
            hs_2d_dispatched._clear_to_zero_allocation()

            # 4. 保存输入供 recompute 使用（仅逐专家路径需要）
        if self.recompute_moe_premute and hs_2d_dispatched_fp8 is not None:
            self.hs_2d_dispatched_fp8 = hs_2d_dispatched_fp8
            self.hs_2d_dispatched_scale = hs_2d_dispatched_scale

        return (
            use_fp8_dispatch_a2a,
            num_experts,
            num_zipped_tokens,
            hidden_size,
            unzipped_tokens,
            zipped_expertwise_rowmap,
            unzipped_probs,
            unzipped_scale,
            hs_2d_dispatched_fp8,
            hs_2d_dispatched_scale,
        )

    def forward_auto_subbatch(
        self, hs_2d_dispatched, dispatched_indices, dispatched_probs
    ):
        """
        AutoSubbatch 前向: 根据 VMM 空闲显存动态决定每次处理多少 token.

        背景
        ----
        MoE 前向 = unzip(按专家展开) -> expert_gemm -> zip(合并回原序).
        expert_gemm 产生大量中间变量 (o1, o2 等), 显存可能不够一次做完,
        因此需要把 token 切成多个 subbatch 分批计算.

        两个关键决策
        -----------
        1) zip_unzip_fusion: 能否一次性分配 unzip 后的完整 buffer?
           - True:  分配 n2[N,H] + o3[N,H], unzip/zip 各做一次
           - False: 显存不够, 不分配整块, 每个专家单独 gather/scatter
        2) subbatch_rows: 每个 subbatch 处理多少 token?
           由 VMM 空闲显存 / 单个 subbatch 峰值 (o1[2H] + o2[H/2]) 决定.
           如果 subbatch_rows >= 总token数 且 fusion, 则走 group_gemm 一次算完.

        流程 (S=seq_len, H=hidden_size, N=unzipped_tokens)
        --------------------------------------------------
        0. 预分配 n3[S,H] (最终输出, 必须整块), 判断 zip_unzip_fusion
        1. unzip: 按专家重排 token + FP8 量化
        2. subbatch planning:
           - 如果 not zip_unzip_fusion: 预分配 n2 placeholder 占位,
             分配 n3 累加 buffer
           - del n3 释放空间给专家计算
           - 查询 VMM 空闲 -> 算出 subbatch_rows
        3. expert_gemm:
           a) group_gemm: subbatch_rows >= N -> 一次完成
           b) per_expert: 逐专家循环, 每个专家按 subbatch_rows 切片:
              o1=gate_up(n2), o2=swiglu(o1), o3=down(o2)
              not zip_unzip_fusion 时: 先 gather n2 再算, 算完 scatter 回 n3
        4. zip: 合并专家输出回原序 -> 最终 output[S,H]
        """
        use_fp8_dispatch_a2a = isinstance(hs_2d_dispatched, tuple)

        # 先分配 n3，因为 n3 必须整个分配，避免先分配了后面的 n2/o3 导致 n3 分配不出来
        zipped_out = paddle.empty(
            shape=(
                hs_2d_dispatched[0].shape
                if use_fp8_dispatch_a2a
                else hs_2d_dispatched.shape
            ),
            dtype=(
                paddle.bfloat16
                if use_fp8_dispatch_a2a
                else hs_2d_dispatched.dtype
            ),
        )

        num_unzipped_tokens = self.token_offsets[-1]
        hidden_size = zipped_out.shape[1]

        # 基于 warmup 历史峰值预测剩余 forward 是否需要降级
        _free_blocks_now = allocator_free_block_info()
        _allocator_total_free = sum(s for s, _ in _free_blocks_now)
        _history = get_auto_sb_history()
        _history.record_forward(_allocator_total_free)
        _should_degrade = _history.should_degrade(_allocator_total_free)
        if _should_degrade:
            zip_unzip_fusion = False
        else:
            zip_unzip_fusion = (
                find_max_concurrent_subbatch_size(
                    [
                        num_unzipped_tokens * zipped_out.shape[1] * 2,
                        num_unzipped_tokens * zipped_out.shape[1],
                    ],
                    upper=1,
                )
                > 0
            )

        if zip_unzip_fusion:
            expert_unzipped_out = paddle.empty(
                [num_unzipped_tokens, zipped_out.shape[1]], zipped_out.dtype
            )

        # 1. 公共预处理：unzip → record_stream → quant
        (
            use_fp8_dispatch_a2a,
            num_experts,
            num_zipped_tokens,
            hidden_size,
            unzipped_tokens,
            zipped_expertwise_rowmap,
            unzipped_probs,
            unzipped_scale,
            hs_2d_dispatched_fp8,
            hs_2d_dispatched_scale,
        ) = self._prepare_forward(
            hs_2d_dispatched,
            dispatched_indices,
            dispatched_probs,
            fill_output=zip_unzip_fusion,
            padding_alignment=self.moe_permute_padding_alignment,
        )

        # 2. subbatch planning
        # 分配 n3 的累加 buffer（如需）
        if zip_unzip_fusion or num_zipped_tokens == 0:
            output = paddle.empty([0, hidden_size], dtype=paddle.float32)
        else:
            # 由于 unzip 必须整专家进行，需要预留 n2 空间，
            # 若不重计算，则需要预留所有专家的 n2
            if self.recompute_moe_premute:
                unzipped_tokens_placeholder = paddle.empty(
                    [max(self.padding_token_per_experts), hidden_size],
                    dtype=unzipped_tokens.dtype,
                )
            else:
                unzipped_tokens_placeholder = [
                    paddle.empty(
                        [num_tokens, hidden_size],
                        dtype=unzipped_tokens.dtype,
                    )
                    for num_tokens in self.padding_token_per_experts
                ]

            # 当 zip 不是一次性完成时，需要为 n3 分配更高精度的累加 buffer
            n3_subbatch_rows = find_max_sequence_subbatch_size(
                feature_size=hidden_size * 4, length=num_zipped_tokens
            )
            n3_subbatch_rows = max(
                n3_subbatch_rows, self.min_auto_subbatch_rows
            )
            output = [
                paddle.zeros(
                    [
                        min(n3_subbatch_rows, num_zipped_tokens - idx),
                        hidden_size,
                    ],
                    dtype=paddle.float32,
                )
                for idx in range(0, num_zipped_tokens, n3_subbatch_rows)
            ]

        # 在专家计算过程中 n3 可以暂时释放，因为 n3 和专家计算的中间变量的生命周期不重叠
        del zipped_out

        # 找到最大的 subbatch_rows
        # 只需考虑 o1 和 o2，因为 o3 可复用 o1 或预分配 buffer
        subbatch_rows = (
            find_max_concurrent_subbatch_size(
                [FP8_ALIGN * hidden_size * 2, FP8_ALIGN * hidden_size // 2],
                upper=self.token_offsets[-1] // FP8_ALIGN,
            )
            * FP8_ALIGN
        )
        subbatch_rows = max(subbatch_rows, self.min_auto_subbatch_rows)
        # 3. experts
        fwd_path = "unknown"
        if (
            self.moe_expert_fusion
            and zip_unzip_fusion
            and subbatch_rows >= self.token_offsets[-1]
        ):
            fwd_path = "group_gemm"
            # 3a) 显存充足, subbatch_rows 大于总 token 数时，直接用 group_gemm 一次计算所有专家
            self.experts_group_gemm_node.forward(
                unzipped_tokens,
                unzipped_probs,
                self.padding_token_per_experts,
                output=expert_unzipped_out,
                scale=unzipped_scale,
            )
        else:
            fwd_path = "per_expert"
            # 3b) 显存不足或非 fusion 模式，回退到 expert_fusion=False, 逐专家处理
            if self.moe_expert_fusion:
                self.fallback_to_no_expert_fusion()

            for expert_id, tokens_per_expert in enumerate(
                self.tokens_per_expert
            ):
                gemm_node = self._gemm_node(expert_id)
                start_idx, end_idx = (
                    self.token_offsets[expert_id],
                    self.token_offsets[expert_id + 1],
                )
                expert_unzipped_idx = paddle.empty(
                    [tokens_per_expert, 0], dtype=paddle.int64
                )
                tmp_unzipped_probs = unzipped_probs[start_idx:end_idx]
                tmp_expert_unzipped_out = None

                # 如果不切 unzip，则专家输入直接通过切片引用；否则每个专家都要执行一次 unzip (gather)
                if zip_unzip_fusion:
                    self.subbatch_prepare_gemm_node(
                        (
                            unzipped_tokens[start_idx:end_idx],
                            unzipped_scale[start_idx:end_idx],
                        ),
                        expert_id,
                    )
                    tmp_expert_unzipped_out = expert_unzipped_out[
                        start_idx:end_idx
                    ]
                else:
                    # 释放 placeholder，为实际 n2 buffer 腾出空间
                    if self.recompute_moe_premute:
                        unzipped_tokens_placeholder = None
                    else:
                        unzipped_tokens_placeholder[expert_id] = None
                    expert_unzipped_idx = (
                        self.subbatch_unzip_and_prepare_gemm_node(
                            (hs_2d_dispatched_fp8, hs_2d_dispatched_scale),
                            zipped_expertwise_rowmap,
                            expert_id,
                        )
                    )

                # 遍历当前专家的每个 subbatch（可能只有一个subbatch）

                with self.slice_fp8_weight(expert_id):
                    for sb_start in range(0, tokens_per_expert, subbatch_rows):
                        sb_end = min(
                            sb_start + subbatch_rows, tokens_per_expert
                        )
                        if sb_start == 0 and sb_end == tokens_per_expert:
                            sb_start = sb_end = None
                        output = self.gemm_forward_subbatch(
                            expert_id,
                            tmp_unzipped_probs,
                            expert_unzipped_idx,
                            output,
                            num_zipped_tokens,
                            unzipped_out=tmp_expert_unzipped_out,
                            start_idx=sb_start,
                            end_idx=sb_end,
                        )
                if self.recompute_moe_premute:
                    gemm_node.input_fp8 = None
                    gemm_node.input_scale = None
                    gemm_node.input = None

                del expert_unzipped_idx
                del tmp_expert_unzipped_out, tmp_unzipped_probs

        # 4. zip
        # 如果不切 zip，则在这里做一次完整的 zip；否则 zip 已经在前面分批完成，这里只需合并结果
        if zip_unzip_fusion:
            output = self.zip_node.forward(
                expert_unzipped_out,
                zipped_expertwise_rowmap,
                self.dispatched_indices,
                unzipped_probs,
                total_zipped_tokens=num_zipped_tokens,
                num_experts=num_experts,
            )
        else:
            output_dtype = (
                paddle.bfloat16
                if use_fp8_dispatch_a2a
                else hs_2d_dispatched.dtype
            )
            output = merge_subbatch_cast(output, output_dtype)

        self.dispatched_probs = dispatched_probs
        output.stop_gradient = False

        if self.moe_subbatch_diag:
            _predicted_need = _history.predicted_need_for_remaining()
            _in_warmup = _history.in_warmup()
            logger.info(
                "[AutoSubbatch FWD] backend=%s, path=%s, total_tokens=%d, "
                "subbatch_rows=%d, zip_unzip_fusion=%s, "
                "history_step=%d, history_forward_count=%d, history_backward_count=%d, "
                "history_prev_steps=%d, history_prev_max_delta_mb=%.2f, history_max_delta_mb=%.2f, "
                "history_in_warmup=%s, allocator_free_mb=%.2f, "
                "predicted_need_mb=%.2f, should_degrade=%s",
                auto_subbatch_allocator_backend(),
                fwd_path,
                num_unzipped_tokens,
                subbatch_rows,
                zip_unzip_fusion,
                _history.step_idx,
                _history.forward_count,
                _history.backward_count,
                _history.prev_total_steps,
                _history.prev_max_delta / 1024 / 1024,
                _history.max_delta / 1024 / 1024,
                _in_warmup,
                _allocator_total_free / 1024 / 1024,
                _predicted_need / 1024 / 1024,
                _should_degrade,
            )
        return output

    @paddle.no_grad()
    def forward_auto_subbatch_pre_permute(
        self, hs_2d_dispatched, dispatched_indices, dispatched_probs
    ):
        """
        Pre-permute auto subbatch 前向: 在 dispatched 空间 (S 维度) 切 chunk，
        每个 chunk 独立完成 permute→compute→unpermute，峰值显存仅取决于单个 chunk 的展开大小。

        与 post_permute 模式对比:
          post_permute: 全量 moe_permute → [N, H] → subbatch on N
          pre_permute:  切 S → per-chunk permute → per-chunk compute → per-chunk unpermute

        正确性保证: 每个 token 的路由独立（由 dispatched_indices[i] 决定），
        按 S 维度切块后每个 chunk 内部 permute→compute→unpermute 是自洽闭环。
        """
        use_fp8_dispatch_a2a = isinstance(hs_2d_dispatched, tuple)
        if use_fp8_dispatch_a2a:
            hs_input, hs_scale = hs_2d_dispatched
            S = hs_input.shape[0]
            hidden_size = hs_input.shape[1]
            output_dtype = paddle.bfloat16
        else:
            S = hs_2d_dispatched.shape[0]
            hidden_size = hs_2d_dispatched.shape[1]
            output_dtype = hs_2d_dispatched.dtype

        num_experts = len(self.tokens_per_expert)

        # Cast indices to int32 and save for backward
        self.dispatched_indices = dispatched_indices.to(paddle.int32)
        self.dispatched_probs = dispatched_probs

        # Quantize input to FP8 only when use_fp8_mlp=True or fp8_dispatch_a2a
        # When use_fp8_mlp=False (BF16 gemm path), keep input as BF16
        gemm_node = self.experts_group_gemm_node
        use_fp8_path = getattr(gemm_node, "use_fp8_mlp", True)

        if use_fp8_dispatch_a2a:
            hs_fp8 = hs_input
            hs_fp8_scale = hs_scale
            hs_input._record_stream()
            hs_scale._record_stream()
            self.hs_2d_dispatched_bf16 = None
        elif use_fp8_path:
            hs_fp8, hs_fp8_scale = tilewise_quant(hs_2d_dispatched)
            hs_2d_dispatched._clear_to_zero_allocation()
            self.hs_2d_dispatched_bf16 = None
        else:
            # BF16 path: no FP8 quantization
            hs_fp8 = None
            hs_fp8_scale = None
            self.hs_2d_dispatched_bf16 = hs_2d_dispatched

        # Save input for backward (always keep hs_2d_dispatched_fp8 so backward
        # can fall back to recompute path if cached path has insufficient memory)
        self.hs_2d_dispatched_fp8 = hs_fp8
        self.hs_2d_dispatched_scale = hs_fp8_scale

        # History tracking
        _free_blocks_now = allocator_free_block_info()
        _allocator_total_free = sum(s for s, _ in _free_blocks_now)
        _history = get_auto_sb_history()
        _history.record_forward(_allocator_total_free)

        if self.recompute_moe_premute:
            self._pre_permute_cached_chunks = None
        else:
            self._pre_permute_cached_chunks = []
            self._pre_permute_chunk_bounds = []

        # Global expansion ratio for initial chunk size estimate
        N_total = self.token_offsets[-1]
        global_ratio = N_total / max(S, 1)

        # Pre-allocate final output [S, H] before estimating chunk_size,
        # so that find_max_concurrent_subbatch_size sees the true free memory
        # available for the loop.
        final_output = paddle.empty([S, hidden_size], dtype=output_dtype)

        # Determine chunk_size based on VMM free memory
        # unpermute tmp: chunk_size = N_chunk / global_ratio
        # → unpermute_tmp per N token = H * 2 / global_ratio (bytes)
        gate_up_out_dim = self._gate_up_out_dim(hidden_size)  # == inter*2
        inter_dim = max(gate_up_out_dim // 2, 1)
        _unpermute_tmp_per_N = (
            int(hidden_size * 2 / global_ratio) if global_ratio > 0 else 0
        )
        fwd_feature_sizes = self._fwd_pre_permute_feature_sizes(
            hidden_size, gate_up_out_dim, inter_dim, _unpermute_tmp_per_N
        )
        max_N_chunk = (
            find_max_concurrent_subbatch_size(
                fwd_feature_sizes,
                upper=N_total // FP8_ALIGN if N_total > 0 else 1,
            )
            * FP8_ALIGN
        )
        if max_N_chunk > 0 and global_ratio > 0:
            chunk_size = int(max_N_chunk / global_ratio)
        else:
            chunk_size = S
        # Align to FP8_ALIGN
        chunk_size = (chunk_size // FP8_ALIGN) * FP8_ALIGN
        # Clamp
        chunk_size = max(chunk_size, num_experts * FP8_ALIGN)
        chunk_size = min(chunk_size, S)
        # Snap trailing gap smaller than one FP8_ALIGN unit to S,
        # so we don't emit a tiny tail chunk purely due to alignment.
        if 0 < S - chunk_size <= FP8_ALIGN:
            chunk_size = S
        # Allow test override for forced multi-chunk
        if hasattr(self, "max_pre_permute_chunk_size_fwd"):
            chunk_size = min(chunk_size, self.max_pre_permute_chunk_size_fwd)
        elif hasattr(self, "max_pre_permute_chunk_size"):
            chunk_size = min(chunk_size, self.max_pre_permute_chunk_size)

        # Main loop
        sb_start = 0
        initial_chunk_size = chunk_size
        single_chunk = chunk_size >= S
        num_chunks = 0

        # Capture decision-time memory state for diagnostics
        if not single_chunk:
            _decision_free = _allocator_total_free
            _decision_top5_mb = sorted(
                (s / 1024 / 1024 for s, _ in _free_blocks_now),
                reverse=True,
            )[:5]

        while sb_start < S:
            sb_end = min(sb_start + chunk_size, S)

            # Slice chunk
            chunk_indices = self.dispatched_indices[sb_start:sb_end]
            chunk_probs = dispatched_probs[sb_start:sb_end]
            if use_fp8_path:
                chunk_fp8 = hs_fp8[sb_start:sb_end]
                chunk_scale = (
                    hs_fp8_scale[sb_start:sb_end]
                    if hs_fp8_scale is not None
                    else None
                )
                chunk_bf16 = None
            else:
                chunk_fp8 = None
                chunk_scale = None
                chunk_bf16 = self.hs_2d_dispatched_bf16[sb_start:sb_end]

            # Compute tokens_per_expert for this chunk via bincount.
            # Map any -1 (invalid) indices to the num_experts bin, then discard
            # that extra bin. This avoids masked_select + D2H sync.
            flat_indices = chunk_indices.flatten().to(paddle.int32)
            flat_indices = paddle.where(
                flat_indices >= 0,
                flat_indices,
                paddle.full_like(flat_indices, num_experts),
            )
            chunk_tpe_tensor = paddle.bincount(
                flat_indices, minlength=num_experts + 1
            )[:num_experts]
            chunk_tpe = chunk_tpe_tensor.tolist()

            # Pad to FP8_ALIGN
            padded_chunk_tpe = [
                (t + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN for t in chunk_tpe
            ]
            N_chunk = sum(padded_chunk_tpe)

            # Memory check: use fragmentation-aware estimation
            # (same logic as initial chunk_size, but re-query current free blocks)
            # Skip when single_chunk: the initial estimate already covers it.
            if N_chunk > 0 and not single_chunk:
                max_N_now = (
                    find_max_concurrent_subbatch_size(
                        fwd_feature_sizes,
                        upper=N_chunk // FP8_ALIGN,
                    )
                    * FP8_ALIGN
                )
                if max_N_now < N_chunk and chunk_size > num_experts * FP8_ALIGN:
                    # Shrink: scale chunk_size by how much we can actually fit
                    shrink_factor = max_N_now / N_chunk * 0.85
                    chunk_size = int(chunk_size * shrink_factor)
                    chunk_size = (chunk_size // FP8_ALIGN) * FP8_ALIGN
                    chunk_size = max(chunk_size, num_experts * FP8_ALIGN)
                    single_chunk = False
                    continue  # retry with smaller chunk

            chunk_num_tokens = sb_end - sb_start

            if N_chunk == 0:
                # All tokens in this chunk go to no expert, output zeros
                final_output[sb_start:sb_end] = 0
                sb_start = sb_end
                chunk_size = initial_chunk_size
                num_chunks += 1
                continue

            # Permute this chunk
            using_ue8m0_scale = (
                chunk_scale is not None and chunk_scale.dtype == paddle.int32
            )
            if use_fp8_path:
                with paddle.amp.auto_cast(False):
                    (
                        permuted_tokens,
                        chunk_rowmap,
                        permuted_probs,
                        permuted_scale,
                    ) = paddle.nn.functional.moe_permute(
                        chunk_fp8,
                        chunk_scale,
                        chunk_indices,
                        chunk_probs,
                        num_experts=num_experts,
                        tokens_per_expert=chunk_tpe,
                        padding_alignment=FP8_ALIGN,
                        do_gather=True,
                        using_ue8m0_scale=using_ue8m0_scale,
                    )

                # Group gemm (FP8 path): set input_fp8/scale, call forward(None, ...)
                gemm_node = self.experts_group_gemm_node
                gemm_node.input_fp8 = permuted_tokens
                gemm_node.input_scale = permuted_scale

                expert_out = gemm_node.forward(
                    None, permuted_probs, padded_chunk_tpe
                )

                # Cleanup gemm node state
                gemm_node.input_fp8 = None
                gemm_node.input_scale = None
                gemm_node.input = None
            else:
                # BF16 path: permute without scale
                with paddle.amp.auto_cast(False):
                    (
                        permuted_tokens,
                        chunk_rowmap,
                        permuted_probs,
                        _permuted_scale,
                    ) = paddle.nn.functional.moe_permute(
                        chunk_bf16,
                        None,
                        chunk_indices,
                        chunk_probs,
                        num_experts=num_experts,
                        tokens_per_expert=chunk_tpe,
                        padding_alignment=FP8_ALIGN,
                        do_gather=True,
                        using_ue8m0_scale=False,
                    )
                permuted_scale = None

                # Group gemm (BF16 path): pass input directly
                gemm_node = self.experts_group_gemm_node

                expert_out = gemm_node.forward(
                    permuted_tokens, permuted_probs, padded_chunk_tpe
                )

                # Cleanup gemm node state
                gemm_node.input_fp8 = None
                gemm_node.input_scale = None
                gemm_node.input = None

            # Unpermute this chunk's output back to dispatched space
            with paddle.amp.auto_cast(False):
                chunk_output, _ = paddle.nn.functional.moe_unpermute(
                    expert_out,
                    chunk_rowmap,
                    chunk_indices,
                    permuted_probs,
                    chunk_num_tokens,
                    num_experts,
                )

            final_output[sb_start:sb_end] = chunk_output

            # Cache permuted input for backward (non-recompute path)
            if self._pre_permute_cached_chunks is not None:
                self._pre_permute_cached_chunks.append(
                    (permuted_tokens.detach(), permuted_scale, chunk_rowmap)
                )
                self._pre_permute_chunk_bounds.append((sb_start, sb_end))

            # Per-chunk diagnostic
            if not single_chunk:
                logger.info(
                    "[AutoSubbatch PRE_PERMUTE FWD] chunk %d/%d done: "
                    "sb=[%d,%d), N_chunk=%d, decision_free_mb=%.1f, "
                    "decision_top5_mb=[%s]",
                    num_chunks + 1,
                    (S + initial_chunk_size - 1) // initial_chunk_size,
                    sb_start,
                    sb_end,
                    N_chunk,
                    _decision_free / 1024 / 1024,
                    ", ".join(f"{b:.1f}" for b in _decision_top5_mb),
                )

            # Advance
            sb_start = sb_end
            chunk_size = initial_chunk_size
            num_chunks += 1

            # Cleanup chunk tensors (only delete if not cached)
            if self._pre_permute_cached_chunks is None:
                del permuted_tokens, chunk_rowmap, permuted_scale
            del permuted_probs, expert_out, chunk_output

        # Save unzipped_probs as None (not used in pre_permute backward the same way)
        self.unzipped_probs = None

        final_output.stop_gradient = False

        if self.moe_subbatch_diag:
            logger.info(
                "[AutoSubbatch PRE_PERMUTE FWD] mode=pre_permute, total_tokens=%d, "
                "S=%d, chunk_size=%d, num_chunks=%d, global_ratio=%.2f, "
                "allocator_free_mb=%.2f",
                N_total,
                S,
                initial_chunk_size,
                num_chunks,
                global_ratio,
                _allocator_total_free / 1024 / 1024,
            )

        return final_output

    # ==================== backward methods ====================

    def backward_auto_subbatch(self, hidden_states_out_grad):
        """
        AutoSubbatch 反向: 与前向对称, 根据 VMM 空闲显存动态决定 subbatch 大小.

        背景
        ----
        MoE 反向 = zip_grad(梯度按专家展开) -> expert_bwd -> unzip_grad(合并回原序).
        与前向对称, 但反向的显存峰值更大 (需要重算前向中间变量 + 存储梯度),
        因此更需要 subbatch 切分.

        两个关键决策 (与前向相同)
        -----------------------
        1) zip_unzip_fusion: 能否一次性分配展开后的完整梯度 buffer?
           - True:  分配 do3[N,H], zip_grad/unzip_grad 各做一次
           - False: 显存不够, 每个专家单独 gather/scatter_add
        2) subbatch_rows: 由 VMM 空闲 / 反向峰值决定.
           峰值与 swiglu_bwd 是否 inplace 相关:
             inplace:     o1/do1 共享(2H) + o2_s(H) + n2_s(2H) = 5H
             out-of-place: o1(2H) + do1(2H) + o2_s(H) + n2_s(2H) = 7H

        流程 (S=seq_len, H=hidden_size, N=unzipped_tokens)
        --------------------------------------------------
        0. 判断 zip_unzip_fusion
        1. zip_grad: dn3[S,H] -> do3[N,H] (梯度按专家展开)
           如果 recompute_premute + fusion, 同时重算前向的 n2
        2. subbatch planning:
           - 如果 not zip_unzip_fusion: 预分配 do3/n2 placeholder 占位,
             分配 dn1 累加 buffer
           - 查询 VMM 空闲 -> 算出 subbatch_rows -> 释放 placeholder
        3. expert_bwd:
           a) group_gemm: subbatch_rows >= N -> 一次完成
           b) per_expert: 逐专家循环, 每个专家按 subbatch_rows 切片:
              do2=do3@W2, do1=swiglu_bwd(o1), dW2=o2^T@do3, dW1=n2^T@do1, dn2=do1@W1
              not zip_unzip_fusion 时: gather do3 -> 算完 -> scatter_add 回 dn1
        4. unzip_grad: 合并梯度回原序 -> dn1[S,H]
           释放 dn3 物理页给 dn1 复用
        """
        num_unzipped_tokens = self.token_offsets[-1]
        num_zipped_tokens = hidden_states_out_grad.shape[0]
        hidden_size = hidden_states_out_grad.shape[-1]
        output = paddle.empty([0, hidden_size], dtype=paddle.float32)
        probs_grad_list = []

        # 如果 do3 和 n2 (如果recompute) 能同时分配，则可以不用切 zip_grad/unzip，避免重复读写
        zip_unzip_features = [num_unzipped_tokens * hidden_size * 2]
        if self.recompute_moe_premute:
            zip_unzip_features.append(num_unzipped_tokens * hidden_size)
        zip_unzip_fusion = (
            find_max_concurrent_subbatch_size(zip_unzip_features, upper=1) > 0
        )
        # 1. zip_grad and unzip (recompute)
        unzipped_grad = self.zip_node.backward(
            hidden_states_out_grad,
            self.dispatched_indices,
            self.dispatched_probs,
            top_k=self.router_topk,
            num_experts=len(self.tokens_per_expert),
            tokens_per_expert=self.tokens_per_expert,
            fill_output=zip_unzip_fusion,
            padding_alignment=self.moe_permute_padding_alignment,
        )
        if self.recompute_moe_premute and zip_unzip_fusion:
            (unzipped_tokens, _, _, unzipped_scale) = self.unzip_node.forward(
                (self.hs_2d_dispatched_fp8, self.hs_2d_dispatched_scale),
                self.dispatched_indices,
                self.dispatched_probs,
                topk=self.router_topk,
                num_experts=len(self.tokens_per_expert),
                tokens_per_expert=self.tokens_per_expert,
                fill_output=True,
                padding_alignment=self.moe_permute_padding_alignment,
            )

        # 2. subbatch planning
        # 分配 dn1 的累加 buffer（如需）
        if not zip_unzip_fusion:
            # 由于 zip_grad/unzip 不切专家，我们需要保证最大的专家的 unzipped_grad/unzipped_tokens 能够完整分配，
            # 所以我们先分配两者，再计算 subbatch_rows，避免 subbatch_rows 过大导致两者分配不出来
            max_unzipped_tokens_per_expert = (
                (max(self.tokens_per_expert) + FP8_ALIGN - 1)
                // FP8_ALIGN
                * FP8_ALIGN
            )
            unzipped_grad_placeholder = paddle.empty(
                [max_unzipped_tokens_per_expert, hidden_size],
                dtype=hidden_states_out_grad.dtype,
            )
            if self.recompute_moe_premute:
                unzipped_tokens_placeholder = paddle.empty(
                    [max_unzipped_tokens_per_expert, hidden_size],
                    dtype=self.hs_2d_dispatched_fp8.dtype,
                )

            # 当 unzip_grad 不是一次性完成时，需要为 dn1 分配更高精度的的累加 buffer
            dn1_subbatch_rows = find_max_sequence_subbatch_size(
                feature_size=hidden_size * 4, length=num_zipped_tokens
            )
            dn1_subbatch_rows = max(
                dn1_subbatch_rows, self.min_auto_subbatch_rows
            )
            output = [
                paddle.zeros(
                    [
                        min(dn1_subbatch_rows, num_zipped_tokens - idx),
                        hidden_size,
                    ],
                    dtype=paddle.float32,
                )
                for idx in range(0, num_zipped_tokens, dn1_subbatch_rows)
            ]
            if num_zipped_tokens == 0:
                output = paddle.empty([0, hidden_size], dtype=paddle.float32)

        # 找到最大的 subbatch_rows
        # 反向需要考虑3个临时变量：o1、do2、n2_s；其他：n2、do3 已分配，do1、o2_s、dn2 是原地复用
        # o1与swiglu_bwd 是否 inplace相关：
        #
        # inplace（USE_INPLACE_SWIGLU_BWD=True：
        #   do1 复用 o1 buffer，峰值在 dw1（D 点）：
        #   do1/o1(2H) + o2_s(H) + n2_s(2H) = 5H
        #   → feature_sizes = [2H, H, 2H]
        #
        # out-of-place（USE_INPLACE_SWIGLU_BWD=False）：
        #   do1 是独立新 buffer，o1 延迟释放，峰值在 dw1（D 点）：
        #   o1(2H) + do1(2H) + o2_s(H) + n2_s(2H) = 7H
        #   → feature_sizes = [2H, 2H, H, 2H]

        # Pre-allocate grads before subbatch decision so VMM query accounts for grad memory
        self._ensure_weight_grad()

        if USE_INPLACE_SWIGLU_BWD:
            bwd_feature_sizes = [
                FP8_ALIGN * hidden_size * 2,  # o1/do1（inplace 共享）
                FP8_ALIGN * hidden_size,  # o2_s
                FP8_ALIGN * hidden_size * 2,  # n2_s
            ]
        else:
            bwd_feature_sizes = [
                FP8_ALIGN * hidden_size * 2,  # o1
                FP8_ALIGN * hidden_size * 2,  # do1（out-of-place 独立 buffer）
                FP8_ALIGN * hidden_size,  # o2_s
                FP8_ALIGN * hidden_size * 2,  # n2_s
            ]
        subbatch_rows = (
            find_max_concurrent_subbatch_size(
                bwd_feature_sizes,
                upper=self.token_offsets[-1] // FP8_ALIGN,
            )
            * FP8_ALIGN
        )
        subbatch_rows = max(subbatch_rows, self.min_auto_subbatch_rows)

        # 确定 subbatch_rows 后，就可以把刚才占位的显存释放了
        if not zip_unzip_fusion:
            del unzipped_grad_placeholder
            if self.recompute_moe_premute:
                del unzipped_tokens_placeholder

        # 3. experts
        bwd_path = "unknown"
        # 3a) 如果前向走了 group_gemm，且当前显存也足够，则反向也使用 group_gemm
        if self.moe_expert_fusion:
            if zip_unzip_fusion and subbatch_rows >= self.token_offsets[-1]:
                bwd_path = "group_gemm"
                unzipped_grad, unzipped_probs_grad = (
                    self.experts_group_gemm_node.backward(
                        unzipped_grad, self.unzipped_probs
                    )
                )
                probs_grad_list.append(unzipped_probs_grad)
            else:
                # 显存不够做 group_gemm，回退到回退到 expert_fusion=False，逐专家处理
                bwd_path = "per_expert (fallback)"
                self.fallback_to_no_expert_fusion()

        # 3b) 逐专家处理
        if not self.moe_expert_fusion:
            if bwd_path == "unknown":
                bwd_path = "per_expert"
            self._slice_weight_grad()
            for expert_id, tokens_per_expert in enumerate(
                self.tokens_per_expert
            ):
                gemm_node = self._gemm_node(expert_id)
                start_idx, end_idx = (
                    self.token_offsets[expert_id],
                    self.token_offsets[expert_id + 1],
                )

                gemm_node.moe_subbatch_token_num_after_dispatch = (
                    subbatch_rows if subbatch_rows < tokens_per_expert else None
                )

                # 如果前面 zip_grad 一次做完了，这里只需进行切片；否则需要做一次专家级的 zip_grad
                if zip_unzip_fusion:
                    expert_unzipped_grad = unzipped_grad[start_idx:end_idx]
                    if self.recompute_moe_premute:
                        self.subbatch_prepare_gemm_node(
                            (
                                unzipped_tokens[start_idx:end_idx],
                                unzipped_scale[start_idx:end_idx],
                            ),
                            expert_id,
                        )
                else:
                    (
                        expert_unzipped_grad,
                        _,
                        unzipped_grad_idx,
                    ) = paddlefleet_ops.tokens_unzip_gather(
                        hidden_states_out_grad,
                        None,
                        self.unzip_node.zipped_expertwise_rowmap,
                        expert_id=expert_id,
                        tokens_per_expert=self.tokens_per_expert,
                        padding_multiplex=FP8_ALIGN,
                    )
                    if self.recompute_moe_premute:
                        self.subbatch_unzip_and_prepare_gemm_node(
                            (
                                self.hs_2d_dispatched_fp8,
                                self.hs_2d_dispatched_scale,
                            ),
                            self.unzip_node.zipped_expertwise_rowmap,
                            expert_id,
                        )

                # 进行单个专家的 backward，注意 expert_unzipped_grad 是原地修改
                with self.slice_fp8_weight(expert_id):
                    expert_unzipped_grad, unzipped_probs_grad = (
                        gemm_node.backward(
                            expert_unzipped_grad,
                            self.unzipped_probs[start_idx:end_idx],
                        )
                    )

                # 如果 unzip_grad 不是一次做完，则需要每个专家分别做一次 unzip_grad (scatter_add)
                if not zip_unzip_fusion:
                    output = tokens_zip_unique_add_with_subbatch(
                        output,
                        expert_unzipped_grad,
                        unzipped_grad_idx,
                        zipped_rows=num_zipped_tokens,
                        subbatch_rows=(
                            None
                            if isinstance(output, paddle.Tensor)
                            else output[0].shape[0]
                        ),
                    )
                    del unzipped_grad_idx

                if len(unzipped_probs_grad.shape) > 1:
                    unzipped_probs_grad = unzipped_probs_grad.squeeze(-1)
                assert len(unzipped_probs_grad.shape) == 1, (
                    unzipped_probs_grad.shape
                )
                probs_grad_list.append(unzipped_probs_grad)

                # gemm_node.moe_subbatch_token_num_after_dispatch = (
                #     original_subbatch_rows
                # )

                del expert_unzipped_grad

        # 4. unzip_grad
        hidden_states_out_grad._clear_to_zero_allocation()  # dn1 复用 dn3
        if self.recompute_moe_premute and zip_unzip_fusion:
            del unzipped_tokens, unzipped_scale

        # 如果不切 unzip_grad，则在这里做一次完整的 unzip_grad；否则 unzip_grad 已经在前面分批完成，这里只需合并结果
        if zip_unzip_fusion:
            probs_grad = paddle.concat(probs_grad_list)
            del probs_grad_list
            hs_fp8_dispatched_grad, dispatched_probs_grad = (
                self.unzip_node.backward(
                    unzipped_grad,
                    hidden_states_out_grad.shape,
                    probs_grad,
                    self.dispatched_indices,
                    num_experts=len(self.tokens_per_expert),
                )
            )
        else:
            hs_fp8_dispatched_grad = merge_subbatch_cast(
                output, hidden_states_out_grad.dtype
            )
            del output
            dispatched_probs_grad = paddlefleet_ops.tokens_zip_prob(
                probs_grad_list,
                self.unzip_node.zipped_expertwise_rowmap,
                self.dispatched_indices,
            )

        self.reset_state()

        _history = get_auto_sb_history()
        _iteration_finished = _history.record_backward()

        if self.moe_subbatch_diag:
            logger.info(
                "[AutoSubbatch BWD] backend=%s, path=%s, total_tokens=%d, "
                "subbatch_rows=%d, zip_unzip_fusion=%s, "
                "history_step=%d, history_forward_count=%d, history_backward_count=%d, "
                "history_prev_steps=%d, history_iteration_finished=%s",
                auto_subbatch_allocator_backend(),
                bwd_path,
                num_unzipped_tokens,
                subbatch_rows,
                zip_unzip_fusion,
                _history.step_idx,
                _history.forward_count,
                _history.backward_count,
                _history.prev_total_steps,
                _iteration_finished,
            )

        return hs_fp8_dispatched_grad, dispatched_probs_grad

    @paddle.no_grad()
    def backward_auto_subbatch_pre_permute(self, hidden_states_out_grad):
        """
        Pre-permute auto subbatch 反向: 在 dispatched 空间 (S 维度) 切 chunk，
        每个 chunk 独立完成 permute_grad→expert_bwd→unpermute_grad。

        两种路径:
        - recompute_moe_premute=True: 反向独立决策 chunk 边界，重新 permute 前向输入
        - recompute_moe_premute=False: 复用前向 chunk 边界，使用缓存的 permuted 结果
        """
        S = hidden_states_out_grad.shape[0]
        hidden_size = hidden_states_out_grad.shape[1]

        # Early return for empty input (EP dispatch sent 0 tokens to this rank)
        if S == 0:
            topk = (
                self.router_topk
                if self.router_topk is not None
                else (
                    self.dispatched_probs.shape[1]
                    if self.dispatched_probs is not None
                    else 1
                )
            )
            self.reset_state()
            return (
                paddle.empty(
                    [0, hidden_size], dtype=hidden_states_out_grad.dtype
                ),
                paddle.empty([0, topk], dtype=paddle.float32),
            )

        num_experts = len(self.tokens_per_expert)
        N_total = self.token_offsets[-1]
        global_ratio = N_total / max(S, 1)

        # Pre-allocate weight grads so VMM query reflects true availability
        self._ensure_weight_grad()

        # Determine if FP8 or BF16 path
        gemm_node = self.experts_group_gemm_node
        use_fp8_path = getattr(gemm_node, "use_fp8_mlp", True)

        # Check if we have cached permuted results from forward (non-recompute path)
        use_cached = (
            self._pre_permute_cached_chunks is not None
            and len(self._pre_permute_cached_chunks) > 0
        )
        # Pre-allocate final grads as zeros so cached backward leaves
        # all-empty chunks with correct zero gradients.
        final_input_grad = paddle.zeros(
            [S, hidden_size], dtype=hidden_states_out_grad.dtype
        )
        dispatched_probs_grad = paddle.zeros(
            [S, self.router_topk], dtype=paddle.float32
        )

        # Backward per-unzipped-token peak buffers (bytes per FP8_ALIGN tokens),
        gate_up_out_dim = self._gate_up_out_dim(hidden_size)  # == inter*2
        inter_dim = max(gate_up_out_dim // 2, 1)
        bwd_feature_sizes = self._bwd_pre_permute_feature_sizes(
            hidden_size, gate_up_out_dim, inter_dim
        )

        if use_cached:
            # Check if backward peak fits in current free memory (fragmentation
            # aware); if not, discard cache and fall back to recompute path
            # which can independently decide smaller chunk boundaries.
            max_chunk_S = max(
                sb_end - sb_start
                for sb_start, sb_end in self._pre_permute_chunk_bounds
            )
            max_N_estimate = int(max_chunk_S * global_ratio)
            max_N_fit = (
                find_max_concurrent_subbatch_size(
                    bwd_feature_sizes,
                    upper=max(max_N_estimate // FP8_ALIGN, 1),
                )
                * FP8_ALIGN
            )
            if max_N_fit < max_N_estimate:
                logger.info(
                    "[AutoSubbatch PRE_PERMUTE BWD] cached path memory "
                    "insufficient (need_N=%d, fit_N=%d), "
                    "falling back to recompute path",
                    max_N_estimate,
                    max_N_fit,
                )
                self._pre_permute_cached_chunks = None
                self._pre_permute_chunk_bounds = None
                use_cached = False

        if use_cached:
            # Non-recompute path: use forward's chunk boundaries and cached permuted inputs
            chunk_bounds = self._pre_permute_chunk_bounds
            cached_chunks = self._pre_permute_cached_chunks
            num_chunks = len(chunk_bounds)

            for chunk_idx, (sb_start, sb_end) in enumerate(chunk_bounds):
                chunk_indices = self.dispatched_indices[sb_start:sb_end]
                chunk_probs = self.dispatched_probs[sb_start:sb_end]
                chunk_grad = hidden_states_out_grad[sb_start:sb_end]
                chunk_num_tokens = sb_end - sb_start

                # Get cached permuted input and rowmap from forward
                permuted_input, permuted_input_scale, chunk_rowmap = (
                    cached_chunks[chunk_idx]
                )

                # Compute tokens_per_expert for this chunk (needed for permute_grad)
                # Map any -1 (invalid) indices to the num_experts bin, then
                # discard that extra bin. Avoids masked_select + D2H sync.
                flat_indices = chunk_indices.flatten().to(paddle.int32)
                flat_indices = paddle.where(
                    flat_indices >= 0,
                    flat_indices,
                    paddle.full_like(flat_indices, num_experts),
                )
                chunk_tpe_tensor = paddle.bincount(
                    flat_indices, minlength=num_experts + 1
                )[:num_experts]
                chunk_tpe = chunk_tpe_tensor.tolist()
                padded_chunk_tpe = [
                    (t + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN
                    for t in chunk_tpe
                ]
                # Step 1: Permute grad (zip_grad direction)
                using_ue8m0_scale = False
                with paddle.amp.auto_cast(False):
                    (
                        permuted_grad,
                        _,
                        permuted_probs,
                        _,
                    ) = paddle.nn.functional.moe_permute(
                        chunk_grad,
                        None,
                        chunk_indices,
                        chunk_probs,
                        num_experts=num_experts,
                        tokens_per_expert=chunk_tpe,
                        padding_alignment=FP8_ALIGN,
                        do_gather=True,
                        using_ue8m0_scale=using_ue8m0_scale,
                    )

                # Step 2: Use cached permuted_input directly (no re-permute needed)
                if use_fp8_path:
                    gemm_node.input_fp8 = permuted_input
                    gemm_node.input_scale = permuted_input_scale
                else:
                    gemm_node.input = permuted_input
                    gemm_node.input_fp8 = None
                    gemm_node.input_scale = None
                gemm_node.tokens_per_expert = padded_chunk_tpe
                gemm_node.o1 = None
                orig_recompute = gemm_node.recompute_moe_gate_up
                gemm_node.recompute_moe_gate_up = True
                permuted_input_grad, permuted_probs_grad = gemm_node.backward(
                    permuted_grad, permuted_probs
                )
                gemm_node.recompute_moe_gate_up = orig_recompute
                gemm_node.input_fp8 = None
                gemm_node.input_scale = None
                gemm_node.input = None
                gemm_node.tokens_per_expert = None

                # Step 3: Unpermute grad back to dispatched space
                with paddle.amp.auto_cast(False):
                    chunk_input_grad, chunk_probs_grad = (
                        paddle.nn.functional.moe_unpermute(
                            permuted_input_grad,
                            chunk_rowmap,
                            chunk_indices,
                            permuted_probs_grad,
                            chunk_num_tokens,
                            num_experts,
                        )
                    )

                final_input_grad[sb_start:sb_end] = chunk_input_grad
                dispatched_probs_grad[sb_start:sb_end] = chunk_probs_grad

                # Cleanup
                del permuted_grad, permuted_probs, permuted_probs_grad
                del permuted_input_grad, chunk_input_grad, chunk_probs_grad

            # Release cached data
            self._pre_permute_cached_chunks = None
            self._pre_permute_chunk_bounds = None

        else:
            # Recompute path: independently decide chunk boundaries for backward
            _free_before_chunk_blocks = allocator_free_block_info()
            _free_before_chunk_calc = sum(
                s for s, _ in _free_before_chunk_blocks
            )
            _effective_free = _free_before_chunk_calc

            # BWD recompute path chunk sizing: fragmentation-aware, using the
            # shared bwd_feature_sizes (real gate_up/inter widths + dx output),
            # mirroring the forward path's find_max_concurrent_subbatch_size.
            max_N_chunk = (
                find_max_concurrent_subbatch_size(
                    bwd_feature_sizes,
                    upper=N_total // FP8_ALIGN if N_total > 0 else 1,
                )
                * FP8_ALIGN
            )
            if max_N_chunk > 0 and global_ratio > 0:
                chunk_size = int(max_N_chunk / global_ratio)
            else:
                chunk_size = S
            chunk_size = (chunk_size // FP8_ALIGN) * FP8_ALIGN
            chunk_size = max(chunk_size, num_experts * FP8_ALIGN)
            chunk_size = min(chunk_size, S)
            # Snap trailing gap smaller than one FP8_ALIGN unit to S,
            # so we don't emit a tiny tail chunk purely due to alignment.
            if 0 < S - chunk_size <= FP8_ALIGN:
                chunk_size = S
            # Allow test override for forced multi-chunk
            if hasattr(self, "max_pre_permute_chunk_size_bwd"):
                chunk_size = min(
                    chunk_size, self.max_pre_permute_chunk_size_bwd
                )
            elif hasattr(self, "max_pre_permute_chunk_size"):
                chunk_size = min(chunk_size, self.max_pre_permute_chunk_size)

            # Main loop
            sb_start = 0
            initial_chunk_size = chunk_size
            single_chunk = chunk_size >= S
            num_chunks = 0

            # Capture decision-time memory state for diagnostics
            if not single_chunk:
                _decision_free = _free_before_chunk_calc
                _decision_top5_mb = sorted(
                    (s / 1024 / 1024 for s, _ in _free_before_chunk_blocks),
                    reverse=True,
                )[:5]

            while sb_start < S:
                sb_end = min(sb_start + chunk_size, S)

                # Slice chunk
                chunk_indices = self.dispatched_indices[sb_start:sb_end]
                chunk_probs = self.dispatched_probs[sb_start:sb_end]
                chunk_grad = hidden_states_out_grad[sb_start:sb_end]
                chunk_num_tokens = sb_end - sb_start

                # Compute tokens_per_expert for this chunk via bincount.
                # Map any -1 (invalid) indices to the num_experts bin, then
                # discard that extra bin. Avoids masked_select + D2H sync.
                flat_indices = chunk_indices.flatten().to(paddle.int32)
                flat_indices = paddle.where(
                    flat_indices >= 0,
                    flat_indices,
                    paddle.full_like(flat_indices, num_experts),
                )
                chunk_tpe_tensor = paddle.bincount(
                    flat_indices, minlength=num_experts + 1
                )[:num_experts]
                chunk_tpe = chunk_tpe_tensor.tolist()

                # Pad to FP8_ALIGN
                padded_chunk_tpe = [
                    (t + FP8_ALIGN - 1) // FP8_ALIGN * FP8_ALIGN
                    for t in chunk_tpe
                ]
                N_chunk = sum(padded_chunk_tpe)

                # Memory recheck: fragmentation-aware, mirroring the forward
                # path. Skip when single_chunk (the initial fragmentation-aware
                # estimate already covers the whole-S case).
                if N_chunk > 0 and not single_chunk:
                    max_N_now = (
                        find_max_concurrent_subbatch_size(
                            bwd_feature_sizes,
                            upper=N_chunk // FP8_ALIGN,
                        )
                        * FP8_ALIGN
                    )
                    if (
                        max_N_now < N_chunk
                        and chunk_size > num_experts * FP8_ALIGN
                    ):
                        shrink_factor = max_N_now / N_chunk * 0.85
                        chunk_size = int(chunk_size * shrink_factor)
                        chunk_size = (chunk_size // FP8_ALIGN) * FP8_ALIGN
                        chunk_size = max(chunk_size, num_experts * FP8_ALIGN)
                        single_chunk = False
                        continue

                if N_chunk == 0:
                    final_input_grad[sb_start:sb_end] = 0
                    dispatched_probs_grad[sb_start:sb_end] = 0
                    sb_start = sb_end
                    chunk_size = initial_chunk_size
                    num_chunks += 1
                    continue

                # Step 1: Permute grad (zip_grad direction)
                using_ue8m0_scale = False
                with paddle.amp.auto_cast(False):
                    (
                        permuted_grad,
                        chunk_rowmap,
                        permuted_probs,
                        _,
                    ) = paddle.nn.functional.moe_permute(
                        chunk_grad,
                        None,
                        chunk_indices,
                        chunk_probs,
                        num_experts=num_experts,
                        tokens_per_expert=chunk_tpe,
                        padding_alignment=FP8_ALIGN,
                        do_gather=True,
                        using_ue8m0_scale=using_ue8m0_scale,
                    )

                # Step 2: Recompute forward input for gemm backward
                if use_fp8_path:
                    chunk_input_fp8 = self.hs_2d_dispatched_fp8[sb_start:sb_end]
                    chunk_input_scale = (
                        self.hs_2d_dispatched_scale[sb_start:sb_end]
                        if self.hs_2d_dispatched_scale is not None
                        else None
                    )
                    recompute_ue8m0 = (
                        chunk_input_scale is not None
                        and chunk_input_scale.dtype == paddle.int32
                    )
                    with paddle.amp.auto_cast(False):
                        (
                            permuted_input,
                            _,
                            _,
                            permuted_input_scale,
                        ) = paddle.nn.functional.moe_permute(
                            chunk_input_fp8,
                            chunk_input_scale,
                            chunk_indices,
                            chunk_probs,
                            num_experts=num_experts,
                            tokens_per_expert=chunk_tpe,
                            padding_alignment=FP8_ALIGN,
                            do_gather=True,
                            using_ue8m0_scale=recompute_ue8m0,
                        )
                    # Last/only chunk: the un-permuted dispatched source is now
                    # fully consumed. Release it before gemm_node.backward so the
                    # dw1 fused_act_dequant peak no longer holds both the source
                    # [S,H] fp8 and the freshly-permuted permuted_input [N,H] fp8
                    # (single-chunk otherwise pays an extra 1H).
                    if sb_end >= S:
                        del chunk_input_fp8, chunk_input_scale
                        self.hs_2d_dispatched_fp8 = None
                        self.hs_2d_dispatched_scale = None
                else:
                    # BF16 path: re-permute BF16 input
                    chunk_input_bf16 = self.hs_2d_dispatched_bf16[
                        sb_start:sb_end
                    ]
                    with paddle.amp.auto_cast(False):
                        (
                            permuted_input,
                            _,
                            _,
                            _,
                        ) = paddle.nn.functional.moe_permute(
                            chunk_input_bf16,
                            None,
                            chunk_indices,
                            chunk_probs,
                            num_experts=num_experts,
                            tokens_per_expert=chunk_tpe,
                            padding_alignment=FP8_ALIGN,
                            do_gather=True,
                            using_ue8m0_scale=False,
                        )
                    permuted_input_scale = None
                    # Last/only chunk: release the un-permuted BF16 source before
                    # gemm backward (mirrors the fp8 branch).
                    if sb_end >= S:
                        del chunk_input_bf16
                        self.hs_2d_dispatched_bf16 = None

                # Step 3: Group gemm backward (all experts in one call)
                gemm_node = self.experts_group_gemm_node
                if use_fp8_path:
                    gemm_node.input_fp8 = permuted_input
                    gemm_node.input_scale = permuted_input_scale
                else:
                    gemm_node.input = permuted_input
                    gemm_node.input_fp8 = None
                    gemm_node.input_scale = None
                gemm_node.tokens_per_expert = padded_chunk_tpe
                gemm_node.o1 = None
                orig_recompute = gemm_node.recompute_moe_gate_up
                gemm_node.recompute_moe_gate_up = True
                permuted_input_grad, permuted_probs_grad = gemm_node.backward(
                    permuted_grad, permuted_probs
                )
                gemm_node.recompute_moe_gate_up = orig_recompute
                gemm_node.input_fp8 = None
                gemm_node.input_scale = None
                gemm_node.input = None
                gemm_node.tokens_per_expert = None

                # Step 4: Unpermute grad back to dispatched space
                with paddle.amp.auto_cast(False):
                    chunk_input_grad, chunk_probs_grad = (
                        paddle.nn.functional.moe_unpermute(
                            permuted_input_grad,
                            chunk_rowmap,
                            chunk_indices,
                            permuted_probs_grad,
                            chunk_num_tokens,
                            num_experts,
                        )
                    )

                final_input_grad[sb_start:sb_end] = chunk_input_grad
                dispatched_probs_grad[sb_start:sb_end] = chunk_probs_grad

                # Per-chunk diagnostic
                if not single_chunk:
                    logger.info(
                        "[AutoSubbatch PRE_PERMUTE BWD] chunk %d/%d done: "
                        "sb=[%d,%d), N_chunk=%d, decision_free_mb=%.1f, "
                        "decision_top5_mb=[%s]",
                        num_chunks + 1,
                        (S + chunk_size - 1) // chunk_size,
                        sb_start,
                        sb_end,
                        N_chunk,
                        _decision_free / 1024 / 1024,
                        ", ".join(f"{b:.1f}" for b in _decision_top5_mb),
                    )

                # Advance
                sb_start = sb_end
                chunk_size = initial_chunk_size
                num_chunks += 1

                # Cleanup
                del (
                    permuted_grad,
                    chunk_rowmap,
                    permuted_probs,
                    permuted_probs_grad,
                )
                del permuted_input_grad
                del chunk_input_grad, chunk_probs_grad
                del permuted_input, permuted_input_scale

        # Reset state
        self.reset_state()

        # History tracking
        _history = get_auto_sb_history()
        _iteration_finished = _history.record_backward()

        if self.moe_subbatch_diag:
            if use_cached:
                _diag_chunk_size = (
                    chunk_bounds[0][1] - chunk_bounds[0][0]
                    if chunk_bounds
                    else S
                )
            else:
                _diag_chunk_size = initial_chunk_size
            logger.info(
                "[AutoSubbatch PRE_PERMUTE BWD] mode=pre_permute, total_tokens=%d, "
                "S=%d, chunk_size=%d, num_chunks=%d, global_ratio=%.2f, "
                "allocator_free_mb=%.2f, use_cached=%s",
                N_total,
                S,
                _diag_chunk_size,
                num_chunks,
                global_ratio,
                sum(s for s, _ in allocator_free_block_info()) / 1024 / 1024,
                use_cached,
            )

        return final_input_grad, dispatched_probs_grad

    @paddle.no_grad()
    def forward(self, hs_2d_dispatched, dispatched_indices, dispatched_probs):
        """
        对输入数据进行前向传播计算。

        Args:
            hs_2d_dispatched (Tensor): 表示被分派到各个专家的输入数据。
            dispatched_indices (Tensor):表示输入数据被分派到的专家索引。
            dispatched_probs (Tensor): 表示输入数据被分派到各个专家的概率。

        Returns:
            Tensor: 经过前向传播计算后的输出数据。

        """
        if self.use_auto_subbatch:
            if self.auto_subbatch_mode == "pre_permute":
                return self.forward_auto_subbatch_pre_permute(
                    hs_2d_dispatched, dispatched_indices, dispatched_probs
                )
            return self.forward_auto_subbatch(
                hs_2d_dispatched, dispatched_indices, dispatched_probs
            )
        if (
            not self.moe_expert_fusion
            and self.moe_subbatch_token_num_after_dispatch is not None
            and self.moe_subbatch_token_num_after_dispatch > 0
        ):
            fill_output = False
        else:
            fill_output = True
        # 1. 公共预处理：unzip → record_stream → quant
        (
            use_fp8_dispatch_a2a,
            num_experts,
            total_zipped_tokens,
            hidden_size,
            unzipped_tokens,
            zipped_expertwise_rowmap,
            unzipped_probs,
            unzipped_scale,
            hs_2d_dispatched_fp8,
            hs_2d_dispatched_scale,
        ) = self._prepare_forward(
            hs_2d_dispatched,
            dispatched_indices,
            dispatched_probs,
            fill_output=fill_output,
            padding_alignment=self.moe_permute_padding_alignment,
        )
        fwd_path = "unknown"
        if (
            not self.moe_expert_fusion
            and self.moe_subbatch_token_num_after_dispatch is not None
            and self.moe_subbatch_token_num_after_dispatch > 0
        ):
            fwd_path = "per_expert"
            # 路径 2：逐专家 gather → 逐专家 GEMM → scatter-add
            expected_output_dtype = (
                paddle.bfloat16
                if use_fp8_dispatch_a2a
                else hs_2d_dispatched.dtype
            )
            output = paddle.empty([0, hidden_size], dtype=paddle.float32)

            for expert_id, tokens_per_expert in enumerate(
                self.tokens_per_expert
            ):
                expert_unzipped_idx = self.subbatch_unzip_and_prepare_gemm_node(
                    (hs_2d_dispatched_fp8, hs_2d_dispatched_scale),
                    zipped_expertwise_rowmap,
                    expert_id,
                )

                if (
                    self.moe_subbatch_token_num_after_dispatch is not None
                    and self.moe_subbatch_token_num_after_dispatch > 0
                    and tokens_per_expert
                    > self.moe_subbatch_token_num_after_dispatch
                ):
                    num_subbatches = (
                        tokens_per_expert
                        + self.moe_subbatch_token_num_after_dispatch
                        - 1
                    ) // self.moe_subbatch_token_num_after_dispatch
                    for i in range(num_subbatches):
                        sb_start = (
                            i * self.moe_subbatch_token_num_after_dispatch
                        )
                        sb_end = min(
                            sb_start
                            + self.moe_subbatch_token_num_after_dispatch,
                            tokens_per_expert,
                        )
                        output = self.gemm_forward_subbatch(
                            expert_id,
                            unzipped_probs[
                                self.token_offsets[
                                    expert_id
                                ] : self.token_offsets[expert_id + 1]
                            ],
                            expert_unzipped_idx,
                            output,
                            total_zipped_tokens,
                            start_idx=sb_start,
                            end_idx=sb_end,
                        )
                    # nparts>1 的 expert 全部 subbatch 跑完后，释放 input_fp8
                    if self.recompute_moe_premute:
                        gemm_node = self._gemm_node(expert_id)
                        gemm_node.input_fp8 = None
                        gemm_node.input_scale = None
                else:
                    output = self.gemm_forward_subbatch(
                        expert_id,
                        unzipped_probs[
                            self.token_offsets[expert_id] : self.token_offsets[
                                expert_id + 1
                            ]
                        ],
                        expert_unzipped_idx,
                        output,
                        total_zipped_tokens,
                    )

            expert_out = merge_subbatch_cast(output, expected_output_dtype)
        else:
            # 路径 1：一次性 group GEMM → zip
            fwd_path = "group_gemm"
            if not use_fp8_dispatch_a2a:
                hs_2d_dispatched._clear_to_zero_allocation()
            expert_out = self.experts_group_gemm_node.forward(
                unzipped_tokens,
                unzipped_probs,
                self.padding_token_per_experts,
                output=unzipped_tokens,
                scale=unzipped_scale,  # maybe None
            )

            expert_out = expert_out.reshape([-1, expert_out.shape[-1]])

            expert_out = self.zip_node.forward(
                expert_out,
                zipped_expertwise_rowmap,
                self.dispatched_indices,
                unzipped_probs,
                total_zipped_tokens=total_zipped_tokens,
                num_experts=num_experts,
            )

        self.dispatched_probs = dispatched_probs
        expert_out.stop_gradient = False

        if self.moe_subbatch_diag:
            logger.info(
                "[FWD] path=%s, total_tokens=%d",
                fwd_path,
                total_zipped_tokens,
            )

        return expert_out

    @paddle.no_grad()
    def backward(self, hidden_states_out_grad):
        """
        反向传播函数。

        Args:
            hidden_states_out_grad (Tensor): 隐藏状态梯度。

        Returns:
            Tuple[Tensor, Tensor]: 包含两个元素，分别为hs_fp8_dispatched_grad和dispatched_probs_grad。
                - hs_fp8_dispatched_grad (Tensor): 解压后的隐藏状态梯度。
                - dispatched_probs_grad (Tensor): 分发概率梯度。

        """
        if self.use_auto_subbatch:
            if self.auto_subbatch_mode == "pre_permute":
                return self.backward_auto_subbatch_pre_permute(
                    hidden_states_out_grad
                )
            return self.backward_auto_subbatch(hidden_states_out_grad)

        if (
            not self.moe_expert_fusion
            and self.moe_subbatch_token_num_after_dispatch is not None
            and self.moe_subbatch_token_num_after_dispatch > 0
        ):
            fill_output = False
        else:
            fill_output = True

        # zip_grad
        hidden_states_out_grad_shape = hidden_states_out_grad.shape
        unzipped_grad = self.zip_node.backward(
            hidden_states_out_grad,
            self.dispatched_indices,
            self.dispatched_probs,
            top_k=self.router_topk,
            num_experts=len(self.tokens_per_expert),
            tokens_per_expert=self.tokens_per_expert,
            fill_output=fill_output,
            padding_alignment=self.moe_permute_padding_alignment,
        )
        hidden_states_out_grad._record_stream()
        bwd_path = "unknown"
        if (
            not self.moe_expert_fusion
            and self.moe_subbatch_token_num_after_dispatch is not None
            and self.moe_subbatch_token_num_after_dispatch > 0
        ):
            # Per-expert backward path (non-fusion)
            bwd_path = "per_expert"

            self._ensure_weight_grad()
            self._slice_weight_grad()

            output = paddle.empty(
                [0, hidden_states_out_grad_shape[-1]], dtype=paddle.float32
            )
            probs_grad_list = []
            for expert_id, tokens_per_expert in enumerate(
                self.tokens_per_expert
            ):
                (
                    expert_unzipped_grad,
                    _,
                    unzipped_grad_idx,
                ) = paddlefleet_ops.tokens_unzip_gather(
                    hidden_states_out_grad,
                    None,
                    self.unzip_node.zipped_expertwise_rowmap,
                    expert_id=expert_id,
                    tokens_per_expert=self.tokens_per_expert,
                    padding_multiplex=FP8_ALIGN,
                )

                if self.recompute_moe_premute:
                    self.subbatch_unzip_and_prepare_gemm_node(
                        (
                            self.hs_2d_dispatched_fp8,
                            self.hs_2d_dispatched_scale,
                        ),
                        self.unzip_node.zipped_expertwise_rowmap,
                        expert_id,
                    )

                gemm_node = self._gemm_node(expert_id)
                expert_unzipped_grad, unzipped_probs_grad = gemm_node.backward(
                    expert_unzipped_grad,
                    self.unzipped_probs[
                        self.token_offsets[expert_id] : self.token_offsets[
                            expert_id + 1
                        ]
                    ],
                )

                output = tokens_zip_unique_add_with_subbatch(
                    output,
                    expert_unzipped_grad,
                    unzipped_grad_idx,
                    zipped_rows=hidden_states_out_grad_shape[0],
                    subbatch_rows=self.moe_subbatch_token_num_after_dispatch,
                )
                del unzipped_grad_idx

                if len(unzipped_probs_grad.shape) > 1:
                    unzipped_probs_grad = unzipped_probs_grad.squeeze(-1)
                probs_grad_list.append(unzipped_probs_grad)

                del expert_unzipped_grad

            hidden_states_out_grad._clear_to_zero_allocation()
            hs_fp8_dispatched_grad = merge_subbatch_cast(
                output, hidden_states_out_grad.dtype
            )
            del output

            dispatched_probs_grad = paddlefleet_ops.tokens_zip_prob(
                probs_grad_list,
                self.unzip_node.zipped_expertwise_rowmap,
                self.dispatched_indices,
            )
        else:
            bwd_path = "group_gemm"
            hidden_states_out_grad._clear_to_zero_allocation()

            # expert_grad
            expert_out, probs_grad = self.experts_group_gemm_node.backward(
                unzipped_grad, self.unzipped_probs
            )
            del unzipped_grad

            hs_fp8_dispatched_grad, dispatched_probs_grad = (
                self.unzip_node.backward(
                    expert_out,
                    hidden_states_out_grad_shape,
                    probs_grad,
                    self.dispatched_indices,
                    num_experts=len(self.tokens_per_expert),
                )
            )
        self.reset_state()

        if self.moe_subbatch_diag:
            logger.info(
                "[BWD] path=%s, total_tokens=%d",
                bwd_path,
                hs_fp8_dispatched_grad.shape[0],
            )

        return hs_fp8_dispatched_grad, dispatched_probs_grad


class FusionMoePyLayer(paddle.autograd.PyLayer):
    """
    The Fp8FusedMoeFunc class includes operations for unzipping, expert computation, and zipping.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        dispatched_probs,
        dispatched_indices,
        custom_map,
        num_experts_per_tok,
        use_fp8_mlp=True,
        moe_deep_gemm=False,
        recompute_moe_gate_up=False,
        dequant_input=True,
        moe_expert_fusion=True,
        recompute_moe_premute=False,
        moe_subbatch_token_num_after_dispatch=None,
        use_bf16_gemm_weight_grad=False,
        is_first_fwd=False,
        fp8_dispatched_handle=None,
        use_auto_subbatch=False,
        auto_subbatch_mode=None,
        moe_subbatch_diag=False,
        use_ue8m0=False,
        dw_p2p_overlap=False,
        clamp_value=None,
        activation_type=None,
        use_accuracy_compatible=False,
    ):
        """
        根据给定的参数执行前向传播操作。

        Args:
            hidden_states (tensor): 输入的隐藏状态张量。
            dispatched_probs (tensor): 分派概率张量。
            dispatched_indices (tensor): 分派索引张量。
            num_experts_per_tok (int): topk。
            activation_type (str, optional): Activation type, "swiglu" or "geglu".
                Defaults to custom_map._activation_type if present, otherwise "swiglu".

        Returns:
            tensor: 前向传播的结果张量。
        """
        if activation_type is None:
            activation_type = getattr(custom_map, "_activation_type", "swiglu")
        ctx.node = MlpNode(
            custom_map,
            num_experts_per_tok,
            recompute_moe_gate_up=recompute_moe_gate_up,
            dequant_input=dequant_input,
            recompute_moe_premute=recompute_moe_premute,
            moe_subbatch_token_num_after_dispatch=moe_subbatch_token_num_after_dispatch,
            use_bf16_gemm_weight_grad=use_bf16_gemm_weight_grad,
            use_fp8_mlp=use_fp8_mlp,
            moe_deep_gemm=moe_deep_gemm,
            moe_expert_fusion=moe_expert_fusion,
            use_auto_subbatch=use_auto_subbatch,
            auto_subbatch_mode=auto_subbatch_mode,
            moe_subbatch_diag=moe_subbatch_diag,
            use_ue8m0=use_ue8m0,
            dw_p2p_overlap=dw_p2p_overlap,
            clamp_value=clamp_value,
            activation_type=activation_type,
            use_accuracy_compatible=use_accuracy_compatible,
        )

        if fp8_dispatched_handle is not None:
            assert hidden_states.dtype == paddle.float8_e4m3fn
            scale = fp8_dispatched_handle["scale"]
            hidden_states = (hidden_states, scale)

        out = ctx.node.forward(
            hidden_states, dispatched_indices, dispatched_probs
        )

        if is_first_fwd:
            # Under full recompute's first forward (no_grad), the inner PyLayer's
            # backward will never be called. Release intermediate state immediately.
            ctx.node.release_mem()

        # Expose node on moe_layer for diagnostic access
        custom_map._fusion_node = ctx.node

        if is_first_fwd:
            ctx.node.clear_cached_tensors()
        else:
            cached_tensors = ctx.node.cached_tensors()
            ctx.save_for_backward(cached_tensors)
            ctx.node.clear_cached_tensors()
        return out

    @staticmethod
    def backward(ctx, output_grad):
        """
        计算反向传播梯度。

        Args:
            output_grad (Tensor): 输出梯度张量。

        Returns:
            Tuple[Tensor, Tensor, None]: 返回三个梯度张量，前两个分别是隐藏状态和派发概率的梯度，
                                            第三个为None，表示没有需要传递给更前向节点的梯度。

        """
        (cached_tensors,) = ctx.saved_tensor()
        ctx.node.set_cached_tensors(cached_tensors)

        del cached_tensors
        ctx.container = None
        hidden_states_grad, dispatched_probs_grad = ctx.node.backward(
            output_grad
        )
        return hidden_states_grad, dispatched_probs_grad, None


def _hybrid_ep_prepare_expert_counts(
    custom_map,
    use_fp8_mlp,
    moe_expert_fusion,
):
    manager = custom_map.token_dispatcher._comm_manager
    padded_tokens_per_expert = manager.padded_tokens_per_expert
    assert padded_tokens_per_expert is not None, (
        "HybridEP manager must populate padded_tokens_per_expert before "
        "HybridEPMoePyLayer runs."
    )
    num_permuted_tokens = manager.num_permuted_tokens
    assert num_permuted_tokens is not None, (
        "HybridEP manager must populate num_permuted_tokens before "
        "HybridEPMoePyLayer runs."
    )
    padded_tokens_per_expert_tensor = padded_tokens_per_expert.astype("int64")

    if not use_fp8_mlp or not moe_expert_fusion:
        padded_tokens_per_expert_list = padded_tokens_per_expert_tensor.tolist()
        return padded_tokens_per_expert_list, num_permuted_tokens
    return padded_tokens_per_expert_tensor, num_permuted_tokens


def _pad_front_rows(tensor, target_shape):
    if tuple(tensor.shape) == tuple(target_shape):
        return tensor
    padded_tensor = paddle.zeros(target_shape, dtype=tensor.dtype)
    padded_tensor[: tensor.shape[0]] = tensor
    return padded_tensor


def _restore_hybrid_ep_prob_grad_shape(
    dispatched_probs_grad,
    original_probs_shape,
):
    assert len(original_probs_shape) == 1, (
        "HybridEP dispatched_probs is expected to stay 1D on the public "
        f"contract, got original shape {original_probs_shape}"
    )

    if (
        len(dispatched_probs_grad.shape) == 2
        and dispatched_probs_grad.shape[-1] == 1
    ):
        dispatched_probs_grad = dispatched_probs_grad.squeeze(-1)
    assert len(dispatched_probs_grad.shape) == 1, (
        "HybridEP probs grad must normalize back to 1D, "
        f"got shape {tuple(dispatched_probs_grad.shape)}"
    )

    return _pad_front_rows(dispatched_probs_grad, original_probs_shape)


class HybridEPMoePyLayer(paddle.autograd.PyLayer):
    """
    Expert compute for HybridEP's permuted dispatch contract.

    HybridEP dispatch_with_permute already produces expert-contiguous tokens, so
    this layer intentionally skips FusionMoePyLayer's unzip/zip stages and only
    reuses the grouped expert GEMM node for both bf16 and fp8.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        dispatched_probs,
        custom_map,
        use_fp8_mlp=True,
        moe_deep_gemm=True,
        moe_expert_fusion=False,
        recompute_moe_gate_up=False,
        use_bf16_gemm_weight_grad=False,
        fp8_dispatched_handle=None,
        is_first_fwd=False,
        dw_p2p_overlap=False,
        clamp_value=None,
        use_ue8m0=False,
        use_accuracy_compatible=False,
    ):
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            recompute_moe_gate_up=recompute_moe_gate_up,
            dequant_input=True,
            use_bf16_gemm_weight_grad=use_bf16_gemm_weight_grad,
            use_fp8_mlp=use_fp8_mlp,
            moe_deep_gemm=moe_deep_gemm,
            moe_expert_fusion=moe_expert_fusion,
            use_ue8m0=use_ue8m0,
            dw_p2p_overlap=dw_p2p_overlap,
            clamp_value=clamp_value,
            activation_type=getattr(custom_map, "_activation_type", "swiglu"),
            use_accuracy_compatible=use_accuracy_compatible,
        )
        original_hidden_shape = tuple(hidden_states.shape)
        original_probs_shape = tuple(dispatched_probs.shape)
        (
            padded_tokens_per_expert,
            num_permuted_tokens,
        ) = _hybrid_ep_prepare_expert_counts(
            custom_map,
            use_fp8_mlp,
            moe_expert_fusion,
        )
        hidden_states = hidden_states[:num_permuted_tokens]
        dispatched_probs = dispatched_probs[:num_permuted_tokens]
        scale = None
        if fp8_dispatched_handle is not None:
            assert hidden_states.dtype == paddle.float8_e4m3fn
            scale = fp8_dispatched_handle["scale"][:num_permuted_tokens]

        out = node.forward(
            hidden_states,
            dispatched_probs,
            padded_tokens_per_expert,
            scale=scale,
        )
        out.stop_gradient = False

        ctx.node = node
        ctx.original_hidden_shape = original_hidden_shape
        ctx.original_probs_shape = original_probs_shape
        if is_first_fwd:
            node.clear_cached_tensors()
        ctx.save_for_backward([*node.cached_tensors(), dispatched_probs])
        node.clear_cached_tensors()
        return out

    @staticmethod
    def backward(ctx, output_grad):
        (cached_tensors,) = ctx.saved_tensor()
        dispatched_probs = cached_tensors[-1]
        ctx.node.set_cached_tensors(cached_tensors[:-1])
        hidden_states_grad, dispatched_probs_grad = ctx.node.backward(
            output_grad, dispatched_probs
        )
        ctx.node.reset_state()
        hidden_states_grad = _pad_front_rows(
            hidden_states_grad, ctx.original_hidden_shape
        )
        dispatched_probs_grad = _restore_hybrid_ep_prob_grad_shape(
            dispatched_probs_grad,
            ctx.original_probs_shape,
        )
        return hidden_states_grad, dispatched_probs_grad


def run_sonic_moe(
    hidden_states,
    topk_indices,
    topk_scores,
    K,
    E,
    w1,
    w2,
    fp8=False,
    tokens_per_expert=None,
    fp8_scale=None,
    fp8_combine_grad_handle=None,
    fp8_config=None,
):
    T = hidden_states.shape[0]
    stream_id = paddle.device.current_stream()

    if tokens_per_expert is None:
        valid = topk_indices >= 0
        valid_experts = topk_indices[valid].cast(paddle.int32)
        tokens_per_expert = paddle.bincount(valid_experts, minlength=E).cast(
            paddle.int32
        )

    fp8_scale_packed = None
    if fp8 and fp8_scale is not None:
        (
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            _router_scores,
            TK_padded,
            total_pad_rows,
            _N_recv,
            _score_src_idx,
            fp8_scale_packed,
        ) = deepep_topk_to_sonic_metadata_with_scales(
            topk_indices.cast(paddle.int32),
            topk_scores,
            tokens_per_expert,
            E,
            fp8_scale,
            int(hidden_states.shape[1]),
            block=128,
        )
    else:
        (
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            _router_scores,
            TK_padded,
            total_pad_rows,
            _N_recv,
            _score_src_idx,
        ) = deepep_topk_to_sonic_metadata(
            topk_indices.cast(paddle.int32),
            topk_scores,
            tokens_per_expert,
            E,
            block=128 if fp8 else 1,
        )

    s_scatter_idx.stop_gradient = True
    activation_type = ActivationType("swiglu")

    total_expert_freq = TK_padded
    if _score_src_idx is not None and _scatter_router_scores_i32 is not None:
        scores_for_down = _SonicRouterScoresFromMetadata.apply(
            topk_scores, _router_scores, _score_src_idx
        )
    else:
        scores_for_down = _differentiable_router_scores(
            topk_scores,
            topk_indices.cast(paddle.int32),
            num_activated_expert_per_token_offset,
            TK_padded - total_pad_rows,
            TK_padded,
            E,
            score_src_idx=_score_src_idx,
        )

    fp8_hidden_states = None
    if fp8_scale is not None:
        fp8_hidden_states = (
            (hidden_states, fp8_scale, fp8_scale_packed)
            if fp8_scale_packed is not None
            else (hidden_states, fp8_scale)
        )

    # if fp8:
    #     w1_sonic = _make_sonic_fp8_weight_carrier(w1)
    #     w2_sonic = _make_sonic_fp8_weight_carrier(w2)
    # else:
    #     w1_sonic = w1.permute([1, 2, 0])
    #     w2_sonic = w2.permute([1, 2, 0])

    with enable_fp8(fp8):
        # _refresh_fp8_config()
        y1, z = _UpProjection.apply(
            hidden_states,
            w1,
            None,
            expert_frequency_offset,
            total_expert_freq,
            K,
            stream_id,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            True,  # is_varlen_k
            activation_type,
            is_inference_mode_enabled=False,
            use_low_precision_postact_buffer=False,
            prequant_activation_payload=fp8_hidden_states,
            fp8_config=fp8_config,
        )
        hidden_states = _DownProjection.apply(
            y1,
            z,
            w2,
            None,
            scores_for_down,
            s_scatter_idx,
            expert_frequency_offset,
            T,
            K,
            stream_id,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            True,  # is_varlen_k
            activation_type,
            None,
            fp8_combine_grad_handle,
            fp8_config=fp8_config,
        )

    return hidden_states
