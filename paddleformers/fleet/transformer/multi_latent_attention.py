# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import math
import os
from dataclasses import dataclass
from functools import partial
from typing import NoReturn

import paddle
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    WeightGradStore,
)
from paddle.distributed.fleet.utils import recompute

from paddleformers.fleet.context_parallel_utils import ContextParallelScatterOp
from paddleformers.fleet.models.common.embeddings import (
    apply_rotary_pos_emb,
)
from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding as YarnRotaryEmbedding,
    _yarn_get_mscale,
)
from paddleformers.fleet.parallel_state import (
    get_context_parallel_world_size,
)
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.recompute_utils import (
    need_recompute_in_block,
    need_recompute_in_first_n,
)
from paddleformers.fleet.tensor_parallel import RecomputeWithoutOutput
from paddleformers.fleet.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from paddleformers.fleet.transformer.attention import Attention
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.utils import get_pg_rank, get_pg_size


def build_hysparse_valid_range(
    attn_mask_startend_row_indices,
    seq_len,
    batch_size,
    window_size=None,
):
    """Build ``valid_range`` [B, S, 2] int32 for the HySparse TileLang ops.

    Each query token ``t`` gets a half-open valid key-column range ``[bos, eos)``:

    * ``eos = t + 1`` (causal upper bound).
    * ``bos`` = start of the document containing ``t`` (document mask). When
      ``window_size`` is given, ``bos`` is additionally clamped up to
      ``t - window_size + 1`` (causal sliding window).

    Document boundaries are recovered from the flashmask
    ``attn_mask_startend_row_indices`` of shape ``[B, *, S, *]`` whose first
    head / first channel holds, per token, the **exclusive end** of the document
    that token belongs to (same convention as
    ``utils.get_doc_lens`` / ``csa_attention._derive_csa_doc_boundaries``).
    When it is ``None`` a single document (``bos`` document part = 0) is assumed.

    The block grid used by the block-score / block-sparse operators is anchored
    at ``bos`` (document-relative blocks), so the *document* range (no window
    clamp) must be used for block scoring and the block-sparse branch, while the
    windowed range is used for the sliding-window main path.
    """
    positions = paddle.arange(seq_len, dtype="int64").unsqueeze(0)  # [1, S]
    if attn_mask_startend_row_indices is not None:
        # (C) Convention guard: we read the exclusive document-end from
        # channel [:, 0, :, 0]. This only holds for the flashmask layout
        # [B, num_masks, S, num_bounds] whose first mask / first bound carries
        # the per-token exclusive doc-end (== utils.get_doc_lens /
        # csa_attention._derive_csa_doc_boundaries). A silent upstream layout
        # change (extra mask channels, bidirectional bounds, transposed axes)
        # would make the [:, 0, :, 0] slice mean something else and corrupt
        # every downstream bos -> block bucket. Assert the structural shape
        # (host-side, free) so such a change fails loudly here instead of
        # silently mis-bucketing.
        # Use explicit raises (not assert): this is a production forward path
        # and asserts are stripped under `python -O`, which would let a changed
        # upstream layout silently mis-bucket every bos -> block instead of
        # failing loudly here.
        if attn_mask_startend_row_indices.ndim != 4:
            raise ValueError(
                "attn_mask_startend_row_indices must be 4-D "
                "[B, num_masks, S, num_bounds] so [:, 0, :, 0] is the "
                "per-token exclusive doc-end; got ndim="
                f"{attn_mask_startend_row_indices.ndim} "
                f"shape={attn_mask_startend_row_indices.shape}"
            )
        if attn_mask_startend_row_indices.shape[2] != seq_len:
            raise ValueError(
                "attn_mask_startend_row_indices axis-2 must be the "
                f"query length S={seq_len}; got shape="
                f"{attn_mask_startend_row_indices.shape} "
                "(layout changed? [:, 0, :, 0] would no longer be the doc-end)"
            )
        # A legal document mask carries a single exclusive-doc-end bound on the
        # last axis: the per-token exclusive doc-end read via [:, 0, :, 0].
        # shape[3] > 1 (e.g. bidirectional start+end bounds) would make bound 0
        # mean something other than the doc-end and mis-bucket every bos ->
        # block; reject it here. (Axis 1 may be > 1: a multi-head flashmask
        # whose heads share one doc layout is valid and read via head 0.)
        if attn_mask_startend_row_indices.shape[3] != 1:
            raise ValueError(
                "attn_mask_startend_row_indices must be a document mask with a "
                "single exclusive-doc-end bound (shape[3] == 1); got shape="
                f"{attn_mask_startend_row_indices.shape} "
                "([:, 0, :, 0] would no longer be the doc-end)"
            )
        # [B, *, S, *] -> [B_mask, S] exclusive document end per token.
        de = attn_mask_startend_row_indices[:, 0, :, 0].cast("int64")  # [Bm, S]
        # The flashmask row indices may carry a batch of 1 that broadcasts over
        # the data batch (all sequences share one document layout). Expand so
        # the produced valid_range matches the query/key/value batch instead of
        # the mask's batch.
        if de.shape[0] == 1 and batch_size > 1:
            de = de.expand([batch_size, seq_len])
        bsz = de.shape[0]
        pos_b = positions.expand([bsz, seq_len])  # [B, S]
        is_boundary = paddle.zeros([bsz, seq_len], dtype="bool")
        is_boundary[:, 0] = True
        # a new document starts at t when the previous token's doc-end equals t
        # and the doc-end value actually changes.
        is_boundary[:, 1:] = (pos_b[:, 1:] == de[:, :-1]) & (
            de[:, 1:] != de[:, :-1]
        )
        doc_start = paddle.cummax(
            is_boundary.cast("int64") * pos_b, axis=1
        ).values  # [B, S] most-recent document start <= t
    else:
        bsz = batch_size
        doc_start = paddle.zeros([bsz, seq_len], dtype="int64")

    pos_b = positions.expand([bsz, seq_len])
    bos = doc_start
    if window_size is not None and window_size > 0:
        bos = paddle.maximum(doc_start, pos_b - window_size + 1)
    eos = pos_b + 1
    valid_range = paddle.stack([bos, eos], axis=-1).cast("int32")  # [B, S, 2]
    return valid_range.contiguous()


def _ec_compatible_rope_apply(
    q_pe,
    k_pe,
    seq_len,
    rope_base=1000000.0,
    position_offset=0,
    position_ids=None,
    cp_balance_mode="dualchunk_allgather",
):
    """Apply RoPE using EC's complex multiplication method (no YaRN, no mscale).

    This exactly matches ErnieCore's compute_freqs_cis_mrope_and_apply_rotary_3d
    when position_ids are sequential [0, 1, 2, ..., seq_len-1] (text-only case
    where all 3 mRoPE axes have the same value).

    Args:
        q_pe: [B, S, H, D] query positional embedding portion
        k_pe: [B, S, 1, D] key positional embedding portion
        seq_len: sequence length
        rope_base: base frequency (default 1e6)
        position_offset: starting position index for autoregressive decode
        position_ids: optional [S] position ID in fastdeploy decode mode.
                     If None, defaults to [0, 1, ..., seq_len-1] (offset by position_offset).
    """
    head_dim = q_pe.shape[-1]
    # inv_freq same as EC: 1 / (base^(arange(0, dim, 2) / dim))
    freqs = 1.0 / (
        rope_base
        ** (paddle.arange(0, head_dim, 2, dtype="float32") / float(head_dim))
    )
    if get_context_parallel_world_size() > 1:
        # In EB dataflow and CP size > 1, shape of q is [b, s/cp, h, d],
        # we need to get full seq_len here
        seq_len = seq_len * get_context_parallel_world_size()

    # Compute positions: prefer 1D position_ids (fastdeploy decode), else use sequential with offset
    if position_ids is not None and position_ids.ndim == 1:
        positions = position_ids.astype(freqs.dtype)
    else:
        # position ids: [position_offset, position_offset+1, ..., position_offset+seq_len-1]
        positions = paddle.arange(
            position_offset, position_offset + seq_len, dtype="float32"
        )
    # freqs_table: [S, D/2]
    freqs_table = paddle.outer(positions, freqs)
    # Expand for batch: [1, S, D/2]
    freqs_expanded = freqs_table.unsqueeze(0)
    # Expand to match q_pe batch size: [B, S, D/2]
    freqs_expanded = freqs_expanded.expand(
        [q_pe.shape[0], seq_len, head_dim // 2]
    )
    # freqs_cis: complex [B, S, D/2] -> [B, S, 1, D/2]
    freqs_cis = paddle.polar(paddle.ones_like(freqs_expanded), freqs_expanded)
    freqs_cis = freqs_cis.unsqueeze(2)  # [B, S, 1, D/2]

    if get_context_parallel_world_size() > 1:
        # In EB dataflow and CP size > 1, freqs_cis is [b, s/cp, 1, d] in local
        # so, we need to scatter freqs_cis here
        freqs_cis = ContextParallelScatterOp.apply(
            freqs_cis, axis=1, mode=cp_balance_mode
        )

    # MD5 debug
    import hashlib as _hl

    _log_md5 = os.environ.get("LOG_LAYER_MD5", "0") == "1"
    if _log_md5:
        import paddle.distributed as _dist

        _r = _dist.get_rank() if _dist.is_initialized() else 0
        _fc_real = paddle.as_real(freqs_cis)
        _md5 = _hl.md5(_fc_real.cast("float32").numpy().tobytes()).hexdigest()
        print(
            f"[MD5 Probe PF] Rank={_r} freqs_cis MD5={_md5} shape={list(freqs_cis.shape)} q_dtype={q_pe.dtype}",
            flush=True,
        )

    # Apply to q_pe via complex multiplication (EC style: interleaved pairs)
    xq = paddle.reshape(
        q_pe.cast("float32"), [*q_pe.shape[:-1], -1, 2]
    )  # [B,S,H,D/2,2]
    xk = paddle.reshape(
        k_pe.cast("float32"), [*k_pe.shape[:-1], -1, 2]
    )  # [B,S,1,D/2,2]
    xq_ = paddle.as_complex(xq)  # [B,S,H,D/2]
    xk_ = paddle.as_complex(xk)  # [B,S,1,D/2]

    if _log_md5:
        _xq_data = paddle.as_real(xq_).cast("float32").numpy().tobytes()
        _xq_md5 = _hl.md5(_xq_data).hexdigest()
        print(
            f"[MD5 Probe PF] Rank={_r} xq_complex MD5={_xq_md5} shape={list(xq_.shape)}",
            flush=True,
        )

    xq_out = paddle.as_real(xq_ * freqs_cis)  # [B,S,H,D/2,2]
    xk_out = paddle.as_real(xk_ * freqs_cis)  # [B,S,1,D/2,2]

    xq_out = paddle.flatten(xq_out, start_axis=3)  # [B,S,H,D]
    xk_out = paddle.flatten(xk_out, start_axis=3)  # [B,S,1,D]

    return xq_out.cast(q_pe.dtype), xk_out.cast(k_pe.dtype)


@dataclass
class MLASelfAttentionSublayersSpec:
    """Sublayers for MLA self-attention layer."""

    q_a_layernorm: LayerSpec | type = None
    kv_a_layernorm: LayerSpec | type = None

    q_proj: LayerSpec | type = None
    q_a_proj: LayerSpec | type = None
    q_b_proj: LayerSpec | type = None
    kv_a_proj_with_mqa: LayerSpec | type = None
    kv_b_proj: LayerSpec | type = None
    core_attention: LayerSpec | type = None
    o_proj: LayerSpec | type = None
    gate_proj: LayerSpec | type = None


class FP8OverlapProj(paddle.autograd.PyLayer):
    """
    Replaces RowParallelLinear (no bias, mp==1) with explicit split backward.
    Defers dw computation via WeightGradStore to overlap with P2P communication.
    Bit-exact with F.linear(x, weight) for arbitrary batch dimensions.
    """

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        # Bit-exact with RowParallelLinear mp==1, no bias:
        # F.linear(x, weight) = x @ weight, weight shape: [in, out]
        return paddle.nn.functional.linear(x, weight)

    @staticmethod
    def backward(ctx, out_grad):
        x, weight = ctx.saved_tensor()

        def _compute_weight_grad(x, out_grad, weight):
            with paddle.amp.auto_cast(False):
                # Flatten all leading batch dims to 2D before matmul,
                # so dw = x_2d.T @ out_grad_2d has shape [in, out] == weight.shape
                x_2d = x.reshape([-1, x.shape[-1]])  # [B*S, in]
                og_2d = out_grad.reshape([-1, out_grad.shape[-1]])  # [B*S, out]
                w_grad = paddle.matmul(
                    x_2d, og_2d, transpose_x=True
                )  # [in, out]
                # print("w_grad compute")

            if hasattr(weight, "main_grad"):
                if weight.main_grad is None:
                    weight.main_grad = paddle.zeros(
                        weight.shape, dtype=paddle.float32
                    )
                weight.main_grad.add_(w_grad)
            else:
                raise AssertionError("fp8 overlap need main_grad attribute")

            if hasattr(weight, "_apply_backward_hook"):
                weight._apply_backward_hook()

        # dx = out_grad @ weight.T, weight: [in, out] -> [out, in]
        dx = paddle.matmul(out_grad, weight, transpose_y=True)

        # dw computation (deferred via WeightGradStore)
        if not weight.stop_gradient:
            # print("enter overlap weight grad")
            WeightGradStore.enabled = True
            WeightGradStore.put(
                partial(
                    _compute_weight_grad, x.detach(), out_grad.detach(), weight
                )
            )
            WeightGradStore.enabled = False

        return dx, None


class MultiLatentAttention(Attention):
    """Multi-Latent Attention layer abstract class."""

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLASelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        is_mtp_layer: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        self.config: TransformerConfig

        self.out_projection_size = self.v_head_dim * self.num_attention_heads

        if (
            self.is_swa
            and getattr(self.config, "swa_qk_nope_head_dim", None) is not None
        ):
            self.qk_nope_head_dim = self.config.swa_qk_nope_head_dim
        else:
            self.qk_nope_head_dim = self.config.qk_nope_head_dim

        if (
            self.is_swa
            and getattr(self.config, "swa_qk_rope_head_dim", None) is not None
        ):
            self.qk_rope_head_dim = self.config.swa_qk_rope_head_dim
        else:
            self.qk_rope_head_dim = self.config.qk_rope_head_dim

        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        mscale = _yarn_get_mscale(
            self.config.rotary_scaling_factor, self.config.mscale_all_dim
        )
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)
        # mscale == 1.0 means softmax_scale equals default 1/sqrt(d), no need to pass explicitly
        self._softmax_scale_arg = None if mscale == 1.0 else self.softmax_scale

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_rope_head_dim,
                rotary_interleaved=self.config.rotary_interleaved,
                rotary_percent=1.0,
                rotary_base=self.rope_theta,
                cp_group=self.pg_collection.cp,
                use_accuracy_compatible=getattr(
                    self.config, "use_accuracy_compatible", False
                ),
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_rope_head_dim,
                rotary_interleaved=self.config.rotary_interleaved,
                rotary_base=self.rope_theta,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                # cp_group=self.pg_collection.cp,
                use_accuracy_compatible=getattr(
                    self.config, "use_accuracy_compatible", False
                ),
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
                "'rope' and 'yarn'"
            )

        self.core_attention = build_spec_layer(
            sublayers_spec.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            is_mtp_layer=self.is_mtp_layer,
            is_swa=self.is_swa,
            softmax_scale=self._softmax_scale_arg,
            k_channels=self.q_head_dim,
            v_channels=self.v_head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=1,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
        )

        # Output.
        self.o_proj = build_spec_layer(
            sublayers_spec.o_proj,
            self.out_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.use_bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=self.pg_collection.tp,
        )

        # Gated attention
        self.gated_attention = getattr(self.config, "gated_attention", False)
        self.gated_attn_use_q_lora = getattr(
            self.config, "gated_attn_use_q_lora", False
        )
        if self.gated_attention and sublayers_spec.gate_proj is not None:
            # Gate input source: q_compressed (post q_a_layernorm, dim=q_lora_rank) when
            # gated_attn_use_q_lora is set, otherwise the full hidden_states.
            if self.gated_attn_use_q_lora:
                assert self.config.q_lora_rank is not None, (
                    "gated_attn_use_q_lora=True requires q_lora_rank is not None"
                )
                gate_in_dim = self.config.q_lora_rank
            else:
                gate_in_dim = self.config.hidden_size
            self.gate_proj = build_spec_layer(
                sublayers_spec.gate_proj,
                gate_in_dim,
                self.out_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.use_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="mla_gate",
                tp_group=self.pg_collection.tp,
            )
            print(
                f"[GatedAttnCheck][init] layer={getattr(self, 'layer_number', -1)} "
                f"gated_attention={self.gated_attention} "
                f"gated_attn_use_q_lora={self.gated_attn_use_q_lora} "
                f"q_lora_rank={self.config.q_lora_rank} "
                f"hidden_size={self.config.hidden_size} "
                f"gate_in_dim={gate_in_dim} "
                f"gate_out_dim={self.out_projection_size}",
                flush=True,
            )
        else:
            self.gated_attention = False
            self.gate_proj = None

        self.recompute_gated_attn = (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "gated_attn" in self.config.recompute_modules
        )

        self.recompute_qkv_up_porj_and_rope = False
        if self.config.recompute_granularity == "selective":
            modules = self.config.recompute_modules
            if isinstance(modules, list) and "mla_qkv_recompute" in modules:
                self.recompute_qkv_up_porj_and_rope = (
                    True
                    if self.config.recompute_num_layers is None
                    else (
                        need_recompute_in_block(
                            self.layer_number,
                            self.config,
                            self.config.recompute_num_layers,
                        )
                        if self.config.recompute_method == "block"
                        else need_recompute_in_first_n(
                            self.layer_number,
                            self.config,
                            self.config.recompute_num_layers,
                        )
                    )
                )
            elif isinstance(modules, dict) and "mla_qkv_recompute" in modules:
                assert self.config.recompute_method in ["first_n", "block"]
                num_layers = modules["mla_qkv_recompute"]
                self.recompute_qkv_up_porj_and_rope = (
                    need_recompute_in_block(
                        self.layer_number, self.config, num_layers
                    )
                    if self.config.recompute_method == "block"
                    else need_recompute_in_first_n(
                        self.layer_number, self.config, num_layers
                    )
                )

    def _compute_absorbed_q(self, query):
        """
        Compute absorbed query for FD MLA decode kernel.

        The MLA decode kernel expects q in absorbed form:
            q_absorbed = [q_nope @ W_k_b, q_pe] per head
        where per-head dim = kv_lora_rank + qk_rope_head_dim (e.g. 576)

        Also returns wv_b for V de-absorption on the kernel output.

        Args:
            query: [b, s, heads, qk_nope_head_dim + qk_rope_head_dim]

        Returns:
            q_absorbed: [b, s, heads, kv_lora_rank + qk_rope_head_dim]
            wv_b: [heads, kv_lora_rank, v_head_dim]
        """
        qk_nope_head_dim = self.qk_nope_head_dim
        qk_rope_head_dim = self.qk_rope_head_dim
        kv_lora_rank = self.config.kv_lora_rank
        v_head_dim = self.v_head_dim
        num_heads = self.num_attention_heads_per_partition

        # Split query into nope and rope parts
        q_nope = query[
            ..., :qk_nope_head_dim
        ]  # [b, s, heads, qk_nope_head_dim]
        q_pe = query[..., qk_nope_head_dim:]  # [b, s, heads, qk_rope_head_dim]

        # Get kv_b_proj weight and reshape to per-head form
        # kv_b_proj.weight: [kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]
        kv_b_weight = self.kv_b_proj.weight
        w = kv_b_weight.reshape([kv_lora_rank, num_heads, -1]).transpose(
            perm=[1, 2, 0]
        )

        # w: [heads, qk_nope + v_head, kv_lora_rank]
        # wk_b: [heads, qk_nope_head_dim, kv_lora_rank]
        wk_b = w[:, :qk_nope_head_dim, :]
        # wv_b: [heads, kv_lora_rank, v_head_dim]
        wv_b = w[:, -v_head_dim:, :].transpose(perm=[0, 2, 1])

        # Absorption: q_nope @ wk_b => q_nope_absorbed
        # q_nope: [b, s, heads, qk_nope] -> [b*s, heads, qk_nope] -> [heads, b*s, qk_nope]
        orig_shape = q_nope.shape  # [b, s, heads, qk_nope]
        bs = orig_shape[0] * orig_shape[1]

        q_nope_3d = q_nope.reshape([bs, num_heads, qk_nope_head_dim]).transpose(
            [1, 0, 2]
        )
        q_pe_3d = q_pe.reshape([bs, num_heads, qk_rope_head_dim])
        # bmm: [heads, b*s, qk_nope] @ [heads, qk_nope, kv_lora_rank] -> [b*s, heads, kv_lora_rank]

        q_nope_absorbed = paddle.bmm(q_nope_3d, wk_b).transpose([1, 0, 2])
        # Concat: [b, s, heads, kv_lora_rank + qk_rope_head_dim]
        q_absorbed = paddle.concat([q_nope_absorbed, q_pe_3d], axis=-1)
        q_absorbed = q_absorbed.reshape(
            orig_shape[0], orig_shape[1], num_heads, -1
        )
        return q_absorbed, wv_b

    def forward(
        self,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices: Tensor | None = None,
        key_value_states=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        in_recompute: bool = False,
        position_ids=None,
        shared_kv: list[Tensor] | None = None,
        **kwargs,
    ):
        """Forward pass for multi-latent attention"""
        from paddleformers.fleet.transformer.transformer_layer import TransformerLayer

        _log = TransformerLayer._log_md5

        assert rotary_pos_emb is None, (
            "Rotary position embeddings should not be passed into MLA."
        )
        assert attention_bias is None, (
            "Attention bias should not be passed into MLA."
        )
        assert rotary_pos_cos is None and rotary_pos_sin is None, (
            "MLA does not support Flash Decoding"
        )

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention
        # Also get q_compressed for DSA indexer (if enabled)
        query, key, value, q_compressed, kv_compressed, k_pos_emb = (
            self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
            )
        )

        layer_num = getattr(self, "layer_number", -1)
        _log(query, "attn_query", layer_num)
        _log(key, "attn_key", layer_num)
        if value is not None:
            _log(value, "attn_value", layer_num)

        attn_mask_type = self.attn_mask_type
        query = query.contiguous()
        key = key.contiguous()

        if value is not None:
            value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================

        # NOTE: For sequence parallel, the input is [seq, b, h],
        # transpose back to [b, seq, h] for attention computation
        # TODO: supports [seq, b, h] input in attention computation
        if self.config.sequence_parallel:
            query = query.transpose([1, 0, 2, 3]).contiguous()
            key = key.transpose([1, 0, 2, 3]).contiguous()
            value = value.transpose([1, 0, 2, 3]).contiguous()

        # Extract inference kwargs to pass through to core_attention
        past_key_values = kwargs.get("past_key_values")
        layer_idx = kwargs.get("layer_idx")
        use_cache = kwargs.get("use_cache", False)

        if hasattr(self.core_attention.config, "forward_meta"):  # decode mode
            # Compute absorbed query and V de-absorption weight for FD MLA decode kernel
            # q_absorbed: [b, s, heads, kv_lora_rank + qk_rope_head_dim]
            # wv_b: [heads, kv_lora_rank, v_head_dim]
            q_absorbed, wv_b = self._compute_absorbed_q(query)
        else:
            q_absorbed, wv_b = None, None

        if self.config.enable_hy_sparse_attention and shared_kv is not None:
            # HySparse full-attention layer. The full (dense) attention here is
            # computed by the MHA block-score TileLang op, which additionally
            # emits per-(query, key-block) max logits. We select the top-k key
            # blocks and share both the compressed KV latent and the selected
            # block indices with the downstream SWA layers' block-sparse branch.
            #
            # This branch is checked BEFORE recompute_core_attention: the FA4
            # block-score full path is a distinct computation that must produce
            # block_indices for the downstream SWA layer, and running the plain
            # recompute(core_attention) branch here would leave block_indices
            # undefined (UnboundLocalError at the shared_kv.append below) while
            # also failing to emit the top-k blocks. Activation recompute for
            # HySparse full layers is handled at the layer level
            # (HySparseTransformerLayer.full_recompute).
            core_attn_out, block_indices = self._hy_sparse_full_attention(
                query,
                key,
                value,
                attn_mask_startend_row_indices,
            )
        elif self.recompute_core_attention and self.training:
            core_attn_out = recompute(
                self.core_attention,
                query,
                key,
                value,
                attention_mask.clone() if attention_mask is not None else None,
                attn_mask_startend_row_indices.clone()
                if attn_mask_startend_row_indices is not None
                else None,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention,
                # DSA-specific parameters
                x=hidden_states,
                qr=q_compressed,
                # fastdeploy support
                kv_compressed=kv_compressed,
                k_pos_emb=k_pos_emb,
                q_absorbed=q_absorbed,
                v_b_proj_weight=wv_b,
            )
        else:
            # Static batching attention kernel.
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_startend_row_indices,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention
                and in_recompute,
                past_key_values=past_key_values,
                layer_idx=layer_idx,
                use_cache=use_cache,
                # DSA-specific parameters
                x=hidden_states,
                qr=q_compressed,
                # fastdeploy support
                kv_compressed=kv_compressed,
                k_pos_emb=k_pos_emb,
                q_absorbed=q_absorbed,
                v_b_proj_weight=wv_b,
            )

        if self.recompute_qkv_up_porj_and_rope and self.training:
            assert getattr(self, "_qkv_recompute", None) is not None
            self._qkv_recompute.discard_output_and_register_recompute(
                core_attn_out
            )
            self._qkv_recompute = None

        _log(core_attn_out, "core_attn_out", layer_num)

        if self.config.enable_hy_sparse_attention and shared_kv is not None:
            # Compressed KV latent shared with block-sparse attention in SWA
            # layers (single MQA head): [B, S, 1, kv_lora_rank + qk_rope_head_dim].
            shared_key = paddle.concat(
                [kv_compressed.unsqueeze(2), k_pos_emb], axis=-1
            )
            shared_kv.append(shared_key)
            # block_indices produced by the MHA block-score path above.
            shared_kv.append(block_indices)

        # =================
        # Output. [b, sq, h]
        # =================
        if self.config.sequence_parallel:
            core_attn_out = core_attn_out.transpose([1, 0, 2]).contiguous()

        # Apply gated attention
        if self.gated_attention:
            # Gate input source: q_compressed (post q_a_layernorm, dim=q_lora_rank) when
            # gated_attn_use_q_lora is set, otherwise hidden_states.
            gate_source = (
                q_compressed if self.gated_attn_use_q_lora else hidden_states
            )
            # [GatedAttnCheck][forward] debug print (kept commented for on-demand use):
            # if not getattr(self, "_gated_attn_fwd_logged", False):
            #     self._gated_attn_fwd_logged = True
            #     _src_is_qc = gate_source is q_compressed
            #     print(
            #         f"[GatedAttnCheck][forward] layer={getattr(self, 'layer_number', -1)} "
            #         f"gated_attn_use_q_lora={self.gated_attn_use_q_lora} "
            #         f"gate_source_is_q_compressed={_src_is_qc} "
            #         f"gate_source.shape={list(gate_source.shape)} "
            #         f"q_compressed.shape={list(q_compressed.shape)} "
            #         f"hidden_states.shape={list(hidden_states.shape)} "
            #         f"recompute_gated_attn={self.recompute_gated_attn}",
            #         flush=True,
            #     )
            if self.recompute_gated_attn:
                gate_recompute = RecomputeWithoutOutput()
                core_attn_out = gate_recompute.recompute(
                    self._gate,
                    gate_source,
                    core_attn_out,
                    preserve_rng_state=False,
                    share_grad_holder=True,
                )
            else:
                core_attn_out = self._gate(gate_source, core_attn_out)

        if getattr(self.config, "dw_p2p_overlap", False) and not getattr(
            self.config, "use_bias", False
        ):
            output = FP8OverlapProj.apply(core_attn_out, self.o_proj.weight)
            bias = None
        else:
            output, bias = self.o_proj(core_attn_out)

        if self.gated_attention and self.recompute_gated_attn:
            gate_recompute.discard_output_and_register_recompute(output)

        _log(output, "attn_o_proj_out", layer_num)

        return output, bias

    def _gate(self, gate_source, core_attn_out):
        gate, _ = self.gate_proj(gate_source)
        if self.config.sigmoid_gate_fusion:
            from paddleformers.fleet.triton_ops import SigmoidGateFusionTriton

            core_attn_out = SigmoidGateFusionTriton.apply(core_attn_out, gate)
        else:
            core_attn_out = core_attn_out * paddle.nn.functional.sigmoid(gate)
        return core_attn_out

    def _hy_sparse_full_attention(
        self,
        query,
        key,
        value,
        attn_mask_startend_row_indices,
    ):
        """HySparse full-attention layer using the FA4-fused block-score op.

        Runs dense (decompressed) MHA attention through the FA4 sm100 kernel,
        whose softmax epilogue additionally emits per-(query, key-block) max
        logits at near-zero extra cost (``block_score_fa4_attn_fwd``). From those
        we recover block scores and select the top-k key blocks per query token.
        The selected block indices (shared across heads, document-relative) are
        returned so the downstream SWA layers' block-sparse branch can gather
        exactly the same blocks.

        Args:
            query: [B, S, H, Dk] decompressed query (H independent heads).
            key:   [B, S, H, Dk] decompressed key.
            value: [B, S, H, Dv] decompressed value.
            attn_mask_startend_row_indices: flashmask doc boundaries or ``None``.

        Returns:
            core_attn_out: [B, S, H*Dv] dense attention output.
            block_indices: [B, S, topk] int32 selected block ids (-1 padding).
        """
        from paddleformers.fleet.tilelang_ops.hysparse import (
            block_score_fa4_attn_fwd,
            select_topk_blocks,
        )

        use_tl = getattr(self.config, "hy_sparse_full_attn_use_tilelang", False)
        if use_tl:
            from paddleformers.fleet.tilelang_ops.hysparse.block_score_mha import (
                block_score_mha_attn_fwd,
            )

        b, s, h, _dv = value.shape
        block_B = self.config.hy_sparse_block_size
        topk = self.config.hy_sparse_topk
        sm_scale = self.softmax_scale

        # Document valid_range (no window clamp): full-layer block scoring and
        # the SWA block-sparse branch must share the same document-anchored
        # blocks so the selected indices are transferable across layers. It also
        # anchors select_topk_blocks' per-token block bounds. FA4 itself masks
        # via causal + the raw flashmask ``startend_row_indices`` (same doc
        # structure valid_range is derived from), so the fused block-max --
        # taken after FA4's mask_fn -- honours the identical document mask.
        valid_range = build_hysparse_valid_range(
            attn_mask_startend_row_indices, s, b
        )

        if use_tl:
            # Independent TileLang MHA scorer: masks purely via valid_range
            # (document + causal), no flashmask input needed.
            out, lse, block_logit = block_score_mha_attn_fwd(
                query,
                key,
                value,
                valid_range=valid_range,
                sm_scale=sm_scale,
                block_B=block_B,
                causal=True,
            )
        else:
            out, lse, block_logit = block_score_fa4_attn_fwd(
                query,
                key,
                value,
                valid_range=valid_range,
                sm_scale=sm_scale,
                block_B=block_B,
                causal=True,
                startend_row_indices=attn_mask_startend_row_indices,
            )
        block_indices = select_topk_blocks(
            block_logit,
            lse,
            valid_range,
            topk,
            block_B,
        )
        core_attn_out = out.reshape([b, s, h * _dv])
        return core_attn_out, block_indices


class MLASelfAttention(MultiLatentAttention):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLASelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        is_mtp_layer: bool = False,
    ):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        if self.config.q_lora_rank is None:
            # Not projecting query
            self.q_proj = build_spec_layer(
                sublayers_spec.q_proj,
                self.config.hidden_size,
                self.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_proj",
            )

        else:
            self.q_a_proj = build_spec_layer(
                sublayers_spec.q_a_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_a_proj",
                skip_weight_param_allocation=False,
                tp_group=pg_collection.tp,
            )

            self.q_b_proj = build_spec_layer(
                sublayers_spec.q_b_proj,
                self.config.q_lora_rank,
                self.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_b_proj",
                tp_group=pg_collection.tp,
            )

        self.kv_a_proj_with_mqa = build_spec_layer(
            sublayers_spec.kv_a_proj_with_mqa,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.qk_rope_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_a_proj_with_mqa",
            skip_weight_param_allocation=False,
            tp_group=pg_collection.tp,
        )

        self.kv_b_proj = build_spec_layer(
            sublayers_spec.kv_b_proj,
            self.config.kv_lora_rank,
            self.num_attention_heads
            * (self.qk_nope_head_dim + self.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_b_proj",
            tp_group=pg_collection.tp,
        )

        if self.config.q_lora_rank is not None:
            self.q_a_layernorm = build_spec_layer(
                sublayers_spec.q_a_layernorm,
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.rms_norm_eps,
            )

        self.kv_a_layernorm = build_spec_layer(
            sublayers_spec.kv_a_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.rms_norm_eps,
        )

    def _is_cudagraph_active(self) -> bool:
        """Check if CUDA Graph capture or replay is currently active.

        Uses forward_meta.step_use_cudagraph flag set on core_attention.config
        by FastDeploy's model runner before CUDA graph capture/replay.
        """
        forward_meta = getattr(self.core_attention.config, "forward_meta", None)
        if forward_meta is None:
            return False
        return getattr(forward_meta, "step_use_cudagraph", False)

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # b = batch size, s = sequence length, h = hidden size, n = num attention heads
        # Attention heads [b, s, n*h]
        assert hidden_states.ndim == 3, (
            f"hidden_states should be 3D, [b, s, n*h], got {hidden_states.ndim}D"
        )

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb:[1, s, 1, 64]
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        packed_seq = (
            packed_seq_params is not None
            and packed_seq_params.qkv_format == "thd"
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq,
                position_ids=None if self.training else position_ids,
            )
        else:
            if self.config.apply_rope_fusion:
                rotary_pos_cos, rotary_pos_sin = (
                    self.rotary_pos_emb.get_cached_cos_sin(
                        rotary_seq_len,
                        dtype=hidden_states.dtype,
                        packed_seq=packed_seq,
                    )
                )
                rotary_pos_emb = None
                from paddleformers.fleet.triton_ops.fused_mla_yarn_rope_apply import (
                    fused_apply_mla_rope_for_kv,
                    fused_apply_mla_rope_for_q,
                )

                assert (
                    fused_apply_mla_rope_for_q is not None
                    and fused_apply_mla_rope_for_kv is not None
                ), "Fused MLA RoPE apply is not imported successfully"
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq,
                    position_ids=None if self.training else position_ids,
                )
                # mscale is already accounted for in self.softmax_scale; set to 1.0 to avoid double-applying
                # mscale = 1.0

        cp_size = get_context_parallel_world_size()
        if cp_size > 1:
            # Keep RoPE inputs local to the current CP rank before the fused
            # and non-fused apply paths consume them.
            if packed_seq_params is not None:
                raise ValueError(
                    "Context parallel RoPE scatter in MLA does not support "
                    "packed_seq_params yet."
                )
            if self.config.sequence_parallel:
                local_seq_len = (
                    hidden_states.shape[0]
                    * self.config.tensor_model_parallel_size
                )
            else:
                local_seq_len = hidden_states.shape[1]
            expected_rotary_seq_len = cp_size * local_seq_len
            if rotary_seq_len != expected_rotary_seq_len:
                raise ValueError(
                    "Context parallel requires rotary_seq_len to be the global "
                    f"sequence length, got rotary_seq_len={rotary_seq_len}, "
                    f"expected={expected_rotary_seq_len}, cp_size={cp_size}, "
                    f"local_seq_len={local_seq_len}, "
                    f"sequence_parallel={self.config.sequence_parallel}, "
                    f"tensor_model_parallel_size="
                    f"{self.config.tensor_model_parallel_size}."
                )
            if rotary_pos_cos is not None and rotary_pos_sin is not None:
                if (
                    rotary_pos_cos.shape[1] != rotary_seq_len
                    or rotary_pos_sin.shape[1] != rotary_seq_len
                ):
                    raise ValueError(
                        "Context parallel requires rotary_pos_cos/sin sequence "
                        f"length to match rotary_seq_len, got "
                        f"cos={rotary_pos_cos.shape}, "
                        f"sin={rotary_pos_sin.shape}, "
                        f"rotary_seq_len={rotary_seq_len}."
                    )
                rotary_pos_cos = ContextParallelScatterOp.apply(
                    rotary_pos_cos, axis=1, mode=self.config.cp_balance_mode
                ).contiguous()
                rotary_pos_sin = ContextParallelScatterOp.apply(
                    rotary_pos_sin, axis=1, mode=self.config.cp_balance_mode
                ).contiguous()
            elif rotary_pos_emb is not None:
                if rotary_pos_emb.shape[1] != rotary_seq_len:
                    raise ValueError(
                        "Context parallel requires rotary_pos_emb sequence "
                        f"length to match rotary_seq_len, got "
                        f"rotary_pos_emb={rotary_pos_emb.shape}, "
                        f"rotary_seq_len={rotary_seq_len}."
                    )
                rotary_pos_emb = ContextParallelScatterOp.apply(
                    rotary_pos_emb, axis=1, mode=self.config.cp_balance_mode
                )
            else:
                raise ValueError(
                    "Context parallel requires rotary_pos_emb or rotary_pos_cos/sin "
                    "to be prepared before applying MLA RoPE."
                )

        if (
            packed_seq_params is not None
            and packed_seq_params.qkv_format == "thd"
        ):
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            # if q_a_proj is ColumnParallelLinear:
            #     q_compressed: [b, s, q_lora_rank / TP]
            q_compressed, _ = self.q_a_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear):
            # Gather output to restore output dim q_lora_rank;
            # Scatter sequence back to s / TP if sequence-parallel
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(
                    q_compressed
                )
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(
                        q_compressed
                    )
        else:
            q_compressed = hidden_states

        # if kv_a_proj_with_mqa is ColumnParallelLinear:
        #     kv_combined: [b, s, (kv_lora_rank + qk_rope_head_dim) / TP]
        kv_combined, _ = self.kv_a_proj_with_mqa(hidden_states)
        if (
            kv_combined.size(-1)
            != self.config.kv_lora_rank + self.qk_rope_head_dim
        ):
            # kv_combined: [b, s, (kv_lora_rank + qk_rope_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[b, s, kv_lora_rank], k_pos_emb: [b, s, qk_rope_head_dim]
            kv_compressed, k_pos_emb = paddle.split(
                kv_combined,
                [self.config.kv_lora_rank, self.qk_rope_head_dim],
                axis=-1,
            )
            if self.config.sequence_parallel:
                # kv_compressed:[b, s / TP, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(
                    kv_compressed
                )
        else:
            # kv_compressed:[b, s / TP, kv_lora_rank], k_pos_emb: [b, s / TP, qk_rope_head_dim]
            kv_compressed, k_pos_emb = paddle.split(
                kv_combined,
                [self.config.kv_lora_rank, self.qk_rope_head_dim],
                axis=-1,
            )
            if (
                get_pg_size(self.pg_collection.tp) > 1
                and self.config.sequence_parallel
            ):
                # k_pos_emb: [b, s, qk_rope_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(
                    k_pos_emb, group=self.pg_collection.tp
                )

        # if packed_seq_params is not None:
        #     # PaddleFleet batch-first: [b=1, t, h] -> squeeze dim0 (batch) -> [t, h]
        #     # (SP seq-first: [t, b=1, h] -> squeeze dim1 (batch) -> [t, h])
        #     batch_dim = 1 if self.config.sequence_parallel else 0
        #     q_compressed = q_compressed.squeeze(batch_dim)
        #     kv_compressed = kv_compressed.squeeze(batch_dim)
        #     k_pos_emb = k_pos_emb.squeeze(batch_dim)

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = self.q_a_layernorm(q_compressed)

        kv_compressed = self.kv_a_layernorm(kv_compressed)

        # === MD5 probes for MLA intermediate values ===
        from paddleformers.fleet.transformer.transformer_layer import TransformerLayer

        _log = TransformerLayer._log_md5
        _log(q_compressed, "mla_q_compressed_normed", self.layer_number)
        _log(kv_compressed, "mla_kv_compressed_normed", self.layer_number)
        _log(k_pos_emb, "mla_k_pos_emb_raw", self.layer_number)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(
            q_compressed,
            kv_compressed,
            k_pos_emb,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            position_ids=None,
        ):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [b, s, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [b, s, ...] or [t, ...] for two cases.
            """
            if self.config.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_nope_head_dim + qk_rope_head_dim)]
                q, _ = self.q_b_proj(q_compressed)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_nope_head_dim + qk_rope_head_dim)]
                q, _ = self.q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(
                *q.size()[:-1],
                self.num_attention_heads_per_partition,
                self.q_head_dim,
            )

            # kv: [num_tokens, n * (qk_nope_head_dim + v_head_dim)]
            kv, _ = self.kv_b_proj(kv_compressed)

            # Debug: print kv shape
            # if self.layer_number == 0:
            #     print(f"[DEBUG MLA layer {self.layer_number}] kv shape after kv_b_proj: {kv.shape}", flush=True)

            # kv: [num_tokens, n, (qk_nope_head_dim + v_head_dim)]
            kv = kv.view(
                *kv.size()[:-1],
                self.num_attention_heads_per_partition,
                self.qk_nope_head_dim + self.v_head_dim,
            )

            # if self.layer_number == 0:
            #     print(f"[DEBUG MLA layer {self.layer_number}] kv shape after view: {kv.shape}", flush=True)

            # [num_tokens, qk_rope_head_dim] -> [num_tokens, 1, qk_rope_head_dim]
            k_pos_emb = paddle.unsqueeze(k_pos_emb, -2)

            if self.config.apply_rope_fusion:
                from paddleformers.fleet.triton_ops.fused_mla_yarn_rope_apply import (
                    fused_apply_mla_rope_for_kv,
                    fused_apply_mla_rope_for_q,
                )

                assert not self.config.sequence_parallel, (
                    "sequence_parallel for apply_rope_fusion in mla is not supported yet."
                )
                assert cu_seqlens_q is None, (
                    "thd for apply_rope_fusion in mla is not supported yet."
                )
                cp_size = get_pg_size(self.pg_collection.cp)
                cp_rank = get_pg_rank(self.pg_collection.cp)
                q_len = q.size(1)
                if (
                    packed_seq_params is None
                    or self.config.context_parallel_size == 1
                ) and self.config.rope_type == "rope":
                    # During training, the sequence length is always
                    # the full rotary_pos_emb length, except for sequence packing + CP.
                    # We need the full rotary_pos_emb to cover the full sequence,
                    # so we do not shorten it here.
                    rotary_pos_emb = rotary_pos_emb[:, 0:q_len]
                if self.config.rope_type == "rope":
                    cos = paddle.cos(rotary_pos_emb).contiguous()
                    sin = paddle.sin(rotary_pos_emb).contiguous()
                else:
                    cos = rotary_pos_cos
                    sin = rotary_pos_sin
                if cos.shape[1] != q_len or sin.shape[1] != q_len:
                    raise ValueError(
                        "Fused MLA RoPE requires local cos/sin sequence "
                        f"length to match q_len, got cos={cos.shape}, "
                        f"sin={sin.shape}, q_len={q_len}."
                    )
                query = fused_apply_mla_rope_for_q(
                    q,
                    cos,
                    sin,
                    self.qk_nope_head_dim,
                    self.qk_rope_head_dim,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                )
                key, value = fused_apply_mla_rope_for_kv(
                    kv,
                    k_pos_emb,
                    cos,
                    sin,
                    self.qk_rope_head_dim,
                    self.qk_nope_head_dim,
                    self.v_head_dim,
                    cu_seqlens_kv,
                    cp_rank,
                    cp_size,
                )

                # dynamic_inference not supported for now
                if not self.training:
                    raise NotImplementedError(
                        "apply_rope_fusion does not support dynamic inference yet."
                    )

                k_pe = None
            else:
                # Determine seq length:
                #   packed 3D [t, n, d]      -> dim 0
                #   SP     4D [s, b, n, d]   -> dim 0
                #   normal 4D [b, s, n, d]   -> dim 1
                if q.ndim == 3 or self.config.sequence_parallel:
                    q_len = q.size(0)
                else:
                    q_len = q.size(1)

                # Determine RoPE start position from position_ids (for decode offset)
                # .item() triggers D2H sync which is forbidden inside CUDA graph capture:
                # it causes cudaErrorStreamCaptureUnsupported (900), invalidating the stream
                # BEFORE the try/except can save it.  Guard with _is_cudagraph_active() so
                # we never attempt the sync during capture at all.
                start_pos = 0
                if position_ids is not None and not self._is_cudagraph_active():
                    # Normal path: works when not inside CUDA Graph capture
                    if position_ids.numel() == q_len:
                        start_pos = int(position_ids.flatten()[0].item())

                if get_context_parallel_world_size() == 1 and (
                    packed_seq_params is None
                    or self.config.context_parallel_size == 1
                ):
                    if rotary_pos_emb.shape[1] >= start_pos + q_len:
                        rotary_pos_emb = rotary_pos_emb[
                            :, start_pos : start_pos + q_len
                        ]
                    else:
                        # During inference with KV cache, rotary_pos_emb was
                        # computed for the current input length only, but
                        # position_ids indicate we need embeddings at start_pos.
                        # Recompute with the correct offset.
                        if self.config.rope_type == "rope":
                            rotary_pos_emb = self.rotary_pos_emb(
                                q_len,
                                offset=start_pos,
                                packed_seq=packed_seq,
                                position_ids=None
                                if self.training
                                else position_ids,
                            )
                        else:
                            # mscale is constant for Yarn (depends only on
                            # model hyper-params), so we can safely drop the
                            # recomputed value and keep the outer-scope one.
                            rotary_pos_emb, _ = self.rotary_pos_emb(
                                q_len,
                                offset=start_pos,
                                packed_seq=packed_seq,
                                position_ids=None
                                if self.training
                                else position_ids,
                            )

                if packed_seq_params is not None:
                    raise ValueError(
                        "MLA qkv_up_proj_and_rope_apply does not support "
                        "packed_seq_params yet."
                    )
                expected_rotary_pos_emb_len = q_len
                if rotary_pos_emb.shape[1] != expected_rotary_pos_emb_len:
                    raise ValueError(
                        "MLA RoPE requires local rotary_pos_emb sequence "
                        f"length to match expected length, got "
                        f"rotary_pos_emb={rotary_pos_emb.shape}, "
                        f"expected={expected_rotary_pos_emb_len}, q_len={q_len}, "
                        f"sequence_parallel={self.config.sequence_parallel}, "
                        f"tensor_model_parallel_size="
                        f"{self.config.tensor_model_parallel_size}."
                    )

                # Replace paddle.split with zero-copy slice views.
                q_no_pe = q[..., : self.qk_nope_head_dim]
                q_pos_emb = q[..., self.qk_nope_head_dim :]

                # k_no_pe: [num_tokens, n, qk_nope_head_dim]
                # value: [num_tokens, n, v_head_dim]
                k_no_pe, value = paddle.split(
                    kv,
                    [self.qk_nope_head_dim, self.v_head_dim],
                    axis=-1,
                )

                # When sequence_parallel is enabled and not packed,
                # q/k are seq-first [s, b, n, d] but rotary_pos_emb is
                # batch-first [1, s, 1, d]. Transpose to [s, 1, 1, d]
                # so broadcasting aligns correctly in _apply_rotary_pos_emb_bshd.
                if self.config.sequence_parallel and rotary_pos_emb.ndim == 4:
                    rotary_pos_emb = rotary_pos_emb.transpose([1, 0, 2, 3])

                if self.config.gpt_model_use_experimental_version:
                    # EC-compatible RoPE: complex rotation, no YaRN, no mscale
                    from paddleformers.fleet.transformer.transformer_layer import (
                        TransformerLayer,
                    )

                    _log = TransformerLayer._log_md5
                    _log(q_pos_emb, "mla_q_pe_before_rope", self.layer_number)
                    _log(k_pos_emb, "mla_k_pe_before_rope", self.layer_number)
                    q_pos_emb, k_pos_emb = _ec_compatible_rope_apply(
                        q_pos_emb,
                        k_pos_emb,
                        q_len,
                        rope_base=self.rope_theta,  # Must match EC's config.rope_theta
                        position_offset=start_pos,
                        position_ids=position_ids,
                        cp_balance_mode=self.config.cp_balance_mode,
                    )
                    _log(q_pos_emb, "mla_q_pe_after_rope", self.layer_number)
                    _log(k_pos_emb, "mla_k_pe_after_rope", self.layer_number)
                else:
                    # q_pos_emb: [num_tokens, n, qk_rope_head_dim]
                    q_pos_emb = apply_rotary_pos_emb(
                        q_pos_emb,
                        rotary_pos_emb,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        mscale=mscale,
                        cp_group=self.pg_collection.cp,
                    )
                    # k_pos_emb:[num_tokens, 1, qk_rope_head_dim]
                    k_pos_emb = apply_rotary_pos_emb(
                        k_pos_emb,
                        rotary_pos_emb,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        mscale=mscale,
                        cp_group=self.pg_collection.cp,
                        sp_group=self.pg_collection.tp
                        if self.config.sequence_parallel
                        else None,
                    )

                # query: [num_tokens, n, (qk_nope_head_dim + qk_rope_head_dim)]
                k_pe = k_pos_emb
                query = paddle.cat([q_no_pe, q_pos_emb], axis=-1)

                # key: [num_tokens, n, (qk_nope_head_dim + qk_rope_head_dim)]
                if k_pos_emb.ndim == 4:
                    k_pos_emb = k_pos_emb.expand(
                        -1, -1, self.num_attention_heads_per_partition, -1
                    )
                else:
                    assert k_pos_emb.ndim == 3
                    k_pos_emb = k_pos_emb.expand(
                        -1, self.num_attention_heads_per_partition, -1
                    )
                key = paddle.cat([k_no_pe, k_pos_emb], axis=-1)

            # if self.layer_number == 0:
            #     print(f"[DEBUG MLA layer {self.layer_number}] key final shape: {key.shape}, head_dim={key.shape[-1]}", flush=True)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            return query, key, value, k_pe

        if self.recompute_qkv_up_porj_and_rope and self.training:
            self._qkv_recompute = RecomputeWithoutOutput()
            query, key, value, k_pos_emb = self._qkv_recompute.recompute(
                qkv_up_proj_and_rope_apply,
                q_compressed,
                kv_compressed,
                k_pos_emb,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                position_ids,
                preserve_rng_state=False,
                share_grad_holder=True,
            )
        else:
            query, key, value, k_pos_emb = qkv_up_proj_and_rope_apply(
                q_compressed,
                kv_compressed,
                k_pos_emb,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                position_ids,
            )

        return query, key, value, q_compressed, kv_compressed, k_pos_emb

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation"""
        self._backward_kv_proj()
        self._backward_q_proj()
        self._backward_output_proj()
        # GATE backward?

    def _backward_kv_proj(self):
        """Computes weight gradients of KV projection layers"""
        self.kv_b_proj.backward_dw()
        self.kv_a_proj_with_mqa.backward_dw()

    def _backward_q_proj(self):
        """Computes weight gradients of Q projection layers"""
        if self.config.q_lora_rank is None:
            self.q_proj.backward_dw()
        else:
            self.q_a_proj.backward_dw()
            self.q_b_proj.backward_dw()

    def _backward_output_proj(self):
        """Computes weight gradients of output projection layer"""
        self.o_proj.backward_dw()


class MQASelfAttention(MLASelfAttention):
    """Multi-Query Attention."""

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLASelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        assert not self.config.apply_rope_fusion, (
            "MQA does not support rope fusion."
        )

        # Use MQA only when HySparse is enabled and this is an SWA layer.
        # Otherwise, use its parent's forward method (MLA).
        self.is_mqa = config.enable_hy_sparse_attention and self.is_swa

        if self.is_mqa:
            # Adjust absorbed kv channels for core attention
            k_channels = self.config.kv_lora_rank + self.qk_rope_head_dim
            v_channels = self.config.kv_lora_rank

            self.core_attention.hidden_size_per_partition = (
                k_channels * self.num_attention_heads_per_partition
            )
            self.core_attention.k_channels = k_channels
            self.core_attention.v_channels = v_channels

            # The MQA path never calls ``self.core_attention(...)`` (it runs the
            # TileLang / cuDNN MQA kernels directly), so the ``softmax_offset``
            # that DotProductAttention registers for SWA layers under
            # ``add_swa_attention_sink_bias`` never participates in the forward
            # and keeps a zero gradient, tripping distributed unused-parameter
            # checks. The real sink logits live in ``swa_attn_sink`` /
            # ``sparse_attn_sink`` below, so drop this redundant parameter.
            # ``del`` goes through ``paddle.nn.Layer.__delattr__`` which removes
            # the entry from ``_parameters``; reset to ``None`` afterwards so any
            # generic ``core_attention.softmax_offset`` lookup still resolves.
            if getattr(self.core_attention, "softmax_offset", None) is not None:
                del self.core_attention.softmax_offset
                self.core_attention.softmax_offset = None

            # Gate for block sparse attention
            if self.gated_attention:
                self.sparse_gate_proj = build_spec_layer(
                    sublayers_spec.gate_proj,
                    self.gate_proj.input_size,
                    self.gate_proj.output_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=self.config.use_bias,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name="mla_gate",
                    tp_group=self.pg_collection.tp,
                )

            # Learnable attention-sink bias for the SWA MQA path. Gated by
            # ``add_swa_attention_sink_bias`` (mirrors the DotProductAttention
            # SWA sink promotion). The two HySparse MQA branches are independent
            # softmaxes (main sliding-window path + block-sparse DSA path), so
            # each gets its own per-head sink logit; both are zero-initialised
            # (a zero sink logit == an off-by-one-style sink at logit 0). When
            # the switch is off, both stay ``None`` and the kernels run their
            # plain sinkless softmax exactly as before.
            self.add_swa_attention_sink_bias = getattr(
                self.config, "add_swa_attention_sink_bias", False
            )
            if self.add_swa_attention_sink_bias:
                num_heads = self.num_attention_heads_per_partition
                self.swa_attn_sink = self.create_parameter(
                    shape=[num_heads],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
                self.sparse_attn_sink = self.create_parameter(
                    shape=[num_heads],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )
            else:
                self.swa_attn_sink = None
                self.sparse_attn_sink = None

    def forward(
        self,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices: Tensor | None = None,
        key_value_states=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        in_recompute: bool = False,
        position_ids=None,
        shared_kv: list[Tensor] | None = None,
        **kwargs,
    ):
        """Forward pass for multi-latent attention"""
        from paddleformers.fleet.transformer.transformer_layer import TransformerLayer

        _log = TransformerLayer._log_md5

        assert rotary_pos_emb is None, (
            "Rotary position embeddings should not be passed into MQA."
        )
        assert attention_bias is None, (
            "Attention bias should not be passed into MQA."
        )
        assert rotary_pos_cos is None and rotary_pos_sin is None, (
            "MQA does not support Flash Decoding"
        )
        if get_context_parallel_world_size() > 1:
            raise ValueError("MQA does not support context parallel.")
        if get_pg_size(self.pg_collection.tp) != 1:
            raise ValueError("MQA does not support tensor parallel.")

        if not self.is_mqa:
            return super().forward(
                hidden_states,
                attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                key_value_states=key_value_states,
                packed_seq_params=packed_seq_params,
                in_recompute=in_recompute,
                position_ids=position_ids,
                shared_kv=shared_kv,
                **kwargs,
            )

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention
        # Also get q_compressed for DSA indexer (if enabled)
        query, key, value, q_compressed, kv_compressed, k_pos_emb = (
            self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
            )
        )

        layer_num = getattr(self, "layer_number", -1)
        _log(query, "attn_query", layer_num)
        _log(key, "attn_key", layer_num)
        if value is not None:
            _log(value, "attn_value", layer_num)

        attn_mask_type = self.attn_mask_type
        query = query.contiguous()
        key = key.contiguous()

        if value is not None:
            value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        from paddleformers.fleet.tilelang_ops.hysparse import (
            sliding_window_mqa_attention,
        )

        b, s = query.shape[0], query.shape[1]
        block_B = self.config.hy_sparse_block_size
        sm_scale = self.softmax_scale
        window_size = self.config.sliding_window[0]

        # Absorbed-MLA MQA: one shared K/V head with
        # Dk=kv_lora_rank+qk_rope_head_dim and Dv=kv_lora_rank. Squeeze the head
        # axis to the [B, S_kv, D] layout the TileLang MQA kernels expect.
        shared_k = key.squeeze(2).contiguous()
        shared_v = value.squeeze(2).contiguous()

        # Windowed valid_range for the sliding-window main path; document-anchored
        # valid_range (no window clamp) for the block-sparse branch so its blocks
        # match the full layer's block scoring / selected indices.
        window_valid_range = build_hysparse_valid_range(
            attn_mask_startend_row_indices, s, b, window_size=window_size
        )
        doc_valid_range = build_hysparse_valid_range(
            attn_mask_startend_row_indices, s, b
        )

        # Sliding-window main path over the absorbed MQA dimensions.
        core_attn_out, _ = sliding_window_mqa_attention(
            query,
            shared_k,
            shared_v,
            window_valid_range,
            attn_sink=getattr(self, "swa_attn_sink", None),
            sm_scale=sm_scale,
            block_B=block_B,
        )
        core_attn_out = core_attn_out.reshape(
            [
                b,
                s,
                self.num_attention_heads_per_partition
                * self.config.kv_lora_rank,
            ]
        )

        _log(core_attn_out, "core_attn_out", layer_num)

        # =================
        # Absorb value. [b, sq, num_heads * v_head_dim]
        # =================

        kv_lora_rank = self.config.kv_lora_rank
        num_heads = self.num_attention_heads_per_partition

        v_absorb_weight = self.kv_b_proj.weight.reshape(
            [kv_lora_rank, num_heads, -1]
        )[:, :, self.qk_nope_head_dim :]

        def compute_absorbed_v(core_attn_out):
            core_attn_out = core_attn_out.view(
                *core_attn_out.shape[:-1], num_heads, kv_lora_rank
            )
            core_attn_out = paddle.einsum(
                "bshl,lhv->bshv", core_attn_out, v_absorb_weight
            )
            core_attn_out = core_attn_out.view(
                *core_attn_out.shape[:-2], num_heads * self.v_head_dim
            )
            return core_attn_out

        core_attn_out = compute_absorbed_v(core_attn_out)

        # =================
        # Sparse attention computation
        # =================

        shared_key, shared_block_indices = shared_kv
        # Shared compressed KV latent from the full layer, with
        # Dk=kv_lora_rank+qk_rope_head_dim. Squeeze to [B, S_kv, D]; its leading
        # kv_lora_rank channels are the values.
        shared_key_sq = shared_key.squeeze(2).contiguous()

        # Block-sparse gather branch over the absorbed-MQA shared-head layout
        # (value == the leading kv_lora_rank slice of the shared latent).
        use_tl = getattr(
            self.config, "hy_sparse_block_sparse_use_tilelang", False
        )
        if use_tl:
            from paddleformers.fleet.tilelang_ops.hysparse.block_sparse_mqa_tl import (
                block_sparse_mqa_attention_tl,
            )

            sparse_core_attn_out, _ = block_sparse_mqa_attention_tl(
                query,
                shared_key_sq,
                shared_block_indices,
                doc_valid_range,
                sm_scale=sm_scale,
                block_B=block_B,
                kv_lora_rank=self.config.kv_lora_rank,
                attn_sink=getattr(self, "sparse_attn_sink", None),
            )
        else:
            from paddleformers.fleet.cudnn_ops import (
                block_sparse_mqa_attention_dsa,
                is_dsa_available,
            )

            if not is_dsa_available():
                raise RuntimeError(
                    "HySparse block-sparse attention requires the DSA backend "
                    "(FlashMLA sparse fwd + cuDNN DSA bwd), unavailable here: it "
                    "needs SM100 + FlashMLA + the cuDNN frontend."
                )
            sparse_core_attn_out, _ = block_sparse_mqa_attention_dsa(
                query,
                shared_key_sq,
                shared_block_indices,
                doc_valid_range,
                sm_scale=sm_scale,
                block_B=block_B,
                kv_lora_rank=self.config.kv_lora_rank,
                attn_sink=getattr(self, "sparse_attn_sink", None),
            )
        sparse_core_attn_out = sparse_core_attn_out.reshape(
            [
                b,
                s,
                self.num_attention_heads_per_partition
                * self.config.kv_lora_rank,
            ]
        )

        sparse_core_attn_out = compute_absorbed_v(sparse_core_attn_out)

        # =================
        # Output. [b, sq, h]
        # =================
        # Apply gated attention
        if self.gated_attention:
            # Gate input source: q_compressed (post q_a_layernorm, dim=q_lora_rank) when
            # gated_attn_use_q_lora is set, otherwise hidden_states.
            gate_source = (
                q_compressed if self.gated_attn_use_q_lora else hidden_states
            )
            core_attn_out = self._gate(gate_source, core_attn_out)

        # Add sparse attention output
        if self.gated_attention:
            gate, _ = self.sparse_gate_proj(gate_source)
            sparse_core_attn_out = (
                sparse_core_attn_out * paddle.nn.functional.sigmoid(gate)
            )
        core_attn_out += sparse_core_attn_out

        output, bias = self.o_proj(core_attn_out)

        _log(output, "attn_o_proj_out", layer_num)

        return output, bias

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        if not self.is_mqa:
            return super().get_query_key_value_tensors(
                hidden_states,
                key_value_states=key_value_states,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
            )

        # b = batch size, s = sequence length, h = hidden size, n = num attention heads
        # Attention heads [b, s, n*h]
        assert hidden_states.ndim == 3, (
            f"hidden_states should be 3D, [b, s, n*h], got {hidden_states.ndim}D"
        )

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb: [1, s, 1, pe_dim]
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None

        # Explicit raises (not assert): production forward path, asserts are
        # stripped under `python -O` and an unsupported rope_type /
        # packed_seq_params would then silently feed the RoPE/TileLang/DSA
        # kernels instead of failing here.
        if self.config.rope_type != "rope":
            raise ValueError(
                "MQA only supports rope_type 'rope', got "
                f"{self.config.rope_type}"
            )
        if packed_seq_params is not None:
            raise ValueError("MQA doesn't support packed_seq_params")

        rotary_pos_emb = self.rotary_pos_emb(
            rotary_seq_len,
            position_ids=None if self.training else position_ids,
        )

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            q_compressed, _ = self.q_a_proj(hidden_states)
        else:
            q_compressed = hidden_states

        # kv_combined: [b, s, (kv_lora_rank + qk_rope_head_dim)]
        kv_combined, _ = self.kv_a_proj_with_mqa(hidden_states)

        # kv_compressed: [b, s, kv_lora_rank], k_pos_emb: [b, s, qk_rope_head_dim]
        kv_compressed, k_pos_emb = paddle.split(
            kv_combined,
            [self.config.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
        )

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = self.q_a_layernorm(q_compressed)

        kv_compressed = self.kv_a_layernorm(kv_compressed)

        # === MD5 probes for MLA intermediate values ===
        from paddleformers.fleet.transformer.transformer_layer import TransformerLayer

        _log = TransformerLayer._log_md5
        _log(q_compressed, "mla_q_compressed_normed", self.layer_number)
        _log(kv_compressed, "mla_kv_compressed_normed", self.layer_number)
        _log(k_pos_emb, "mla_k_pos_emb_raw", self.layer_number)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(
            q_compressed,
            kv_compressed,
            k_pos_emb,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            position_ids=None,
        ):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [b, s, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [b, s, ...] or [t, ...] for two cases.
            """
            if self.config.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_nope_head_dim + qk_rope_head_dim)]
                q, _ = self.q_b_proj(q_compressed)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_nope_head_dim + qk_rope_head_dim)]
                q, _ = self.q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(
                *q.size()[:-1],
                self.num_attention_heads_per_partition,
                self.q_head_dim,
            )

            kv_lora_rank = self.config.kv_lora_rank
            num_heads = self.num_attention_heads_per_partition

            q_no_pe = q[..., : self.qk_nope_head_dim]
            q_pos_emb = q[..., self.qk_nope_head_dim :]

            q_absorb_weight = self.kv_b_proj.weight.reshape(
                [kv_lora_rank, num_heads, -1]
            )[:, :, : self.qk_nope_head_dim]
            q_nope_absorbed = paddle.einsum(
                "bshd,lhd->bshl", q_no_pe, q_absorb_weight
            )

            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                config=self.config,
                mscale=mscale,
            )
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb.unsqueeze(-2),
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                config=self.config,
                mscale=mscale,
            )

            kv_compressed = kv_compressed.unsqueeze(-2)

            query = paddle.concat([q_nope_absorbed, q_pos_emb], axis=-1)
            key = paddle.concat([kv_compressed, k_pos_emb], axis=-1)
            value = kv_compressed

            return query, key, value, k_pos_emb

        query, key, value, k_pos_emb = qkv_up_proj_and_rope_apply(
            q_compressed,
            kv_compressed,
            k_pos_emb,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            position_ids,
        )

        return query, key, value, q_compressed, kv_compressed, k_pos_emb
