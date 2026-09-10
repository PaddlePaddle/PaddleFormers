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

"""HyperEncoder network assembly: Provider / `TransformerBlock` subclass / top-level Model.

Layout convention: the `TransformerBlock` subclass, the Provider, and the
top-level Model all live here in PaddleFormers; the PaddleFleet side holds only
the spec builders and the layers specific to this model.

---

# ``HyperEncoderBlock`` -- a thin `TransformerBlock` subclass whose only job is to forward the mask geometry

## Why it is needed

`TransformerBlock.forward` builds `dict_args` with only 9 keys and does NOT
include `attn_mask_startend_row_indices`; but `TransformerLayer._forward_impl`
does explicitly accept it, and the handoff point is
`self._forward_impl(**dict_args, ...)`. So putting one extra key into
`dict_args` is all it takes -- zero intrusion, no need to change the shared
class.

## Why not change `TransformerBlock` itself

Adding a parameter to the shared class is a small change but:
* it affects the `dict_args` shape of every model in the repo;
* those 9 keys are the transport contract between PP stages
  (`TransformerLayer.forward(dict_args) -> dict` exists precisely so it can be a
  `PipelineLayer` stage), so an extra key would ripple into the PP key
  allowlist.

A thin subclass avoids those risks and is the better trade-off.

## What each of the three attention paths needs

| Path | Carrier | How this class passes it |
|---|---|---|
| (1) eager | dense `attention_mask` `[1,1,L,L]` | uses the parent's existing `attention_mask` parameter; this class does nothing extra |
| (2) FlashMask | `attn_mask_startend_row_indices` `[1,1,T',4]` | this class injects it into `dict_args` |
| (3) Triton | `packed_seq_params.prefix_lm_layout` dict | attached to `packed_seq_params` (a plain dataclass, adding a field has no side effect), already forwarded by the parent |

Paths (1) and (2) cannot be passed together. Under bf16, if only the dense mask
is passed without row indices, `DotProductAttention` takes its SDPA branch and
forces `is_causal=True`, silently breaking the bidirectional semantics -- this
is guarded by the `_attn_implementation='eager'` assertion in the layer specs.
Conversely, if both are passed, the flashmask branch takes priority and the
dense mask is silently ignored. So this class raises directly when both are
non-None, rather than guessing a priority.
"""

from __future__ import annotations

from dataclasses import dataclass

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddlefleet.models.gpt.gpt_config import GPTConfig
from paddlefleet.models.hyperencoder.attn_backend import use_packed_decoder
from paddlefleet.packed_seq_params import PackedSeqParams
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.prefix_lm_triton_core import PREFIX_LM_LAYOUT_ATTR
from paddlefleet.transformer.transformer_block import TransformerBlock

from ..model_provider import ModelProviderMixin
from ..model_utils import PretrainedModel
from .configuration import HyperEncoderConfig

__all__ = ["HyperEncoderBlock", "HyperEncoderModel", "HyperEncoderProvider", "HyperEncoderModelFleet"]


class HyperEncoderBlock(TransformerBlock):
    """The 12-layer backbone. The only difference from the parent is that
    `forward` forwards one extra mask-geometry key."""

    def forward(  # type: ignore[override]
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        attn_mask_startend_row_indices: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        rotary_pos_emb: Tensor | None = None,
        **kwargs,
    ):
        """
        Args:
            attention_mask: path (1), dense mask `[1,1,L,L]` bool, `True` = masked out.
            attn_mask_startend_row_indices: path (2), FlashMask 4-column indices `[1,1,T',4]` int32.
            packed_seq_params: path (3) attaches `prefix_lm_layout` to it.
        """
        if attention_mask is not None and attn_mask_startend_row_indices is not None:
            raise ValueError(
                "attention_mask (dense) and attn_mask_startend_row_indices (FlashMask "
                "column-sparse) cannot be passed together: the flashmask branch of "
                "dot_product_attention takes priority and the dense mask would be "
                "silently ignored. Choose exactly one path."
            )
        # Path (3) (packed + triton) passes neither mask parameter; the geometry
        # is entirely in `packed_seq_params.prefix_lm_layout`. So "pass nothing"
        # is only valid when a layout is attached -- otherwise it truly degrades
        # to no mask at all.
        _has_layout = getattr(packed_seq_params, PREFIX_LM_LAYOUT_ATTR, None) is not None
        if attention_mask is None and attn_mask_startend_row_indices is None and not _has_layout:
            raise ValueError(
                "exactly one of attention_mask / attn_mask_startend_row_indices / "
                f"packed_seq_params.{PREFIX_LM_LAYOUT_ATTR} must be provided -- "
                "providing none silently degrades to no mask, losing the prefix-LM "
                "three-region semantics without raising."
            )
        if _has_layout and (attention_mask is not None or attn_mask_startend_row_indices is not None):
            raise ValueError(
                f"packed_seq_params.{PREFIX_LM_LAYOUT_ATTR} cannot be passed together "
                "with a dense mask / FlashMask row indices: the three paths are "
                "mutually exclusive, and passing more than one means the caller has "
                "not decided which path to take."
            )

        # The parent's forward does not know about attn_mask_startend_row_indices,
        # so the key cannot be added after running its implementation (dict_args is
        # built internally there). Passing row indices via an instance attribute is
        # not an option either: recompute re-runs forward and instance attributes
        # are not recompute-safe. So we build dict_args ourselves. The block below
        # mirrors the structure of the parent's dict_args construction.
        from contextlib import nullcontext

        from paddlefleet import tensor_parallel
        from paddlefleet.utils import WrappedTensor

        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        if not self.pre_process:
            hidden_states = self.input_tensor

        rng_context = tensor_parallel.get_cuda_rng_tracker().fork() if self.config.sequence_parallel else nullcontext()

        with rng_context:
            dict_args = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "context": kwargs.get("context"),
                "context_mask": kwargs.get("context_mask"),
                "rotary_pos_emb": rotary_pos_emb,
                "rotary_pos_cos": kwargs.get("rotary_pos_cos"),
                "rotary_pos_sin": kwargs.get("rotary_pos_sin"),
                "attention_bias": kwargs.get("attention_bias"),
                "packed_seq_params": packed_seq_params,
                # This extra key is the entire reason this class exists.
                "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            }
            for layer in self.layers:
                dict_args = layer(dict_args)
                hidden_states = dict_args["hidden_states"]

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class HyperEncoderModel(FleetLayer):
    """The top-level HyperEncoder: front end + 12-layer backbone + output projection.

    It combines the front end, backbone, and `forward_decoder` with the output
    projection and context assembly.

    ## Parameter list

    ===============================  ==================  ==========
    Parameter                          Shape               Notes
    ===============================  ==================  ==========
    ``embed_tokens.weight``          ``[129280, 1280]``
    ``image_encoder.*``              see modality_encoders
    ``audio_encoder.*``              same as above
    ``projector.layers.{weight,bias}``  ``[1280,768]``   image/audio share the same instance
    ``query_short.weight``           ``[256, 1280]``
    ``query_long.weight``            ``[long_q, 1280]``
    ``block.*``                      12 layers
    ``out_projector.{weight,bias}``  ``[1280, H_lm]``
    ===============================  ==================  ==========

    ## The two projector levels must not be conflated

    * ``projector``: modality-level 768 -> 1280, shared by image/audio, on ``is_first_stage``;
    * ``out_projector``: encoder -> LLM 1280 -> ``language_hidden_size``, on ``is_last_stage``,
      a ``RowParallelLinear(bias=True, input_is_parallel=False)`` that returns an
      ``(out, bias)`` tuple; the caller adds bias by hand.
    """

    def __init__(self, config, language_hidden_size: int = 1280):
        # FleetLayer.__init__ requires config (used to record recompute / dtype context).
        super().__init__(config=config)
        from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )
        from paddlefleet.models.hyperencoder import (
            AudioEncoderConv,
            ImageEncoderConv,
            MlpProjector,
            get_hyperencoder_block_spec,
        )
        from paddlefleet.tensor_parallel.layers import RowParallelLinear

        self.config = config
        self.language_hidden_size = language_hidden_size
        short_q, long_q = config.hyperencoder_query_lengths

        # ---- Front end ----
        # embed_tokens uses a plain Embedding rather than VocabParallelEmbedding:
        # the vocabulary is not sharded across TP (it is marked TP-replicated).
        self.embed_tokens = paddle.nn.Embedding(config.vocab_size, config.hidden_size)
        self.image_encoder = ImageEncoderConv(img_size=728, patch_size=14, in_chans=3, embed_dim=768, out_chans=256)
        self.audio_encoder = AudioEncoderConv(num_mel_bins=128, embed_dim=768)
        self.projector = MlpProjector({"projector_type": "linear", "input_dim": 768, "n_embed": config.hidden_size})
        # Two query tables: built as nn.Embedding but never indexed -- only
        # `.weight` is used as a whole matrix. The Embedding form is kept so the
        # weight key matches (`query_short.weight`).
        self.query_short = paddle.nn.Embedding(short_q, config.hidden_size)
        self.query_long = paddle.nn.Embedding(long_q, config.hidden_size)

        # ---- Backbone ----
        self.block = HyperEncoderBlock(
            config=config,
            spec=get_hyperencoder_block_spec(config),
            post_layer_norm=True,
            pre_process=True,
            post_process=True,
        )

        # ---- RoPE ----
        self.rotary_pos_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rope_theta,
            # Must be enabled: the `pow` for `inv_freq` is computed on CPU for
            # numerical stability. The GPU `pow` differs from the reference by
            # one fp32 ULP, which after `outer(seq, inv_freq)` gets amplified in
            # the freqs table and, through `sin` + bf16 cast, flips a few
            # elements, propagating a per-element difference in q/k all the way
            # to the backward pass. Computing on CPU and moving to GPU makes the
            # freqs table bit-identical.
            #
            # This switch is an existing parameter of `RotaryEmbedding`, so
            # enabling it at the call site is enough; the shared
            # rotary embedding module does not need to change.
            use_accuracy_compatible=True,
        )

        # ---- Output projection ----
        # Use a shallow copy of config with `sequence_parallel` forced to False,
        # rather than sharing the same config. This is required: combining
        # `input_is_parallel=False` with `sequence_parallel=True` raises
        # "To enable `sequence_parallel`, `input_is_parallel` must be `True`".
        # At TP=1 sequence_parallel is already False, so this only bites at TP>1.
        import copy as _copy

        self.projector_config = _copy.copy(config)
        self.projector_config.sequence_parallel = False
        self.out_projector = RowParallelLinear(
            config.hidden_size,
            language_hidden_size,
            config=self.projector_config,
            init_method=config.init_method,
            bias=True,
            input_is_parallel=False,
            skip_bias_add=False,
        )

        # ---- Cast the front end uniformly to params_dtype ----
        # These modules are built with paddle's default dtype (fp32), while the
        # backbone and inputs are bf16.
        for m in (
            self.embed_tokens,
            self.image_encoder,
            self.audio_encoder,
            self.projector,
            self.query_short,
            self.query_long,
        ):
            m.to(dtype=config.params_dtype)

        # The front end and query tables are TP-replicated parameters.
        for m in (
            self.embed_tokens,
            self.image_encoder,
            self.audio_encoder,
            self.projector,
            self.query_short,
            self.query_long,
        ):
            for p in m.parameters():
                p.average_gradients_across_tp_domain = True
        # One separate case: the output projection's bias. `out_projector` is a
        # RowParallelLinear -- its weight is sharded by TP but the bias is
        # replicated -- so the bias wgrad is computed independently on each rank
        # and must be averaged/summed across the TP domain, otherwise at TP>1
        # this term would be computed as one of tp_size shares. It is not among
        # the six modules in the loop above; missing it does not raise, it just
        # silently computes the wrong value.
        if getattr(self.out_projector, "bias", None) is not None:
            self.out_projector.bias.average_gradients_across_tp_domain = True

    # ------------------------------------------------------------------ #
    def build_context_embeds(self, context_ids, image=None, audio=None):
        """context_ids -> context_embeds, splicing image/audio features in by "truncate and append".

        Note the known coordinate mismatch that is reproduced intentionally:
        ``audio_start_token_pos`` is computed from the original ``context_ids``,
        while ``embeds`` may already have been rebuilt by the image branch. This
        is only harmless when the number of placeholders exactly equals the
        number of feature rows.

        Also reproduces three gradient-keepalive tricks: when there is no
        image/audio, a dummy is still pushed through the tower and
        ``feat.sum() * 0.0`` is added. This is bit-identically an identity on the
        forward (`x + 0.0`); what it actually affects is the set of parameters
        that receive gradients.
        """
        from paddleformers.transformers.hyperencoder.configuration import (
            AUDIO_PATCH_TOKEN,
            IM_PATCH_TOKEN,
        )

        embeds = self.embed_tokens(context_ids)  # [B, C, H]
        bs = embeds.shape[0]
        imgs = image if image else [None] * bs
        auds = audio if audio else [None] * bs

        out = []
        for i in range(bs):
            ids_i, emb_i = context_ids[i], embeds[i]

            pos = paddle.nonzero(ids_i == IM_PATCH_TOKEN)
            if pos.numel() > 0 and isinstance(imgs[i], paddle.Tensor):
                feat = self.projector(self.image_encoder(imgs[i].unsqueeze(0)))
                # [1,h',w',H] -> [h'*w', H]
                feat = feat.transpose([0, 3, 1, 2]).flatten(2).transpose([0, 2, 1]).squeeze(0)
                emb_i = paddle.concat([emb_i[: int(pos[0])], feat], axis=0)  # truncate and append
            else:
                dummy = paddle.zeros([1, 3, 128, 128], dtype=emb_i.dtype)
                feat = self.projector(self.image_encoder(dummy))
                emb_i = emb_i + feat.sum() * 0.0  # gradient keepalive

            pos = paddle.nonzero(ids_i == AUDIO_PATCH_TOKEN)
            if pos.numel() > 0 and isinstance(auds[i], paddle.Tensor):
                feat = self.projector(self.audio_encoder(auds[i].unsqueeze(0)))
                feat = feat.transpose([0, 3, 1, 2]).flatten(2).transpose([0, 2, 1]).squeeze(0)
                emb_i = paddle.concat([emb_i[: int(pos[0])], feat], axis=0)
            else:
                dummy = paddle.zeros([1, 128, 10], dtype=emb_i.dtype)
                feat = self.projector(self.audio_encoder(dummy))
                emb_i = emb_i + feat.sum() * 0.0

            out.append(emb_i)
        return paddle.stack(out, axis=0)

    def _add_unused_query_dependency(self, latents, use_long_query: bool):
        """Attach a zero gradient to the query table that was NOT used.

        Without this, that table's ``.grad`` is ``None``, and the set of
        parameters receiving gradients would be inconsistent between runs.
        """
        unused = (self.query_short if use_long_query else self.query_long).weight.sum() * 0.0
        return latents + unused.astype(latents.dtype)

    def forward_decoder(self, context_embeds, use_long_query: bool):
        """Non-packed single-segment forward."""
        from paddlefleet.transformer.prefix_lm_mask import (
            build_dense_mask,
            prefix_lm_pad_len,
        )

        bs, n_context, _ = context_embeds.shape
        qw = (self.query_long if use_long_query else self.query_short).weight
        queries = qw.unsqueeze(0).expand([bs, -1, -1])
        n_queries = queries.shape[1]

        x = paddle.concat([context_embeds, queries], axis=1)
        seq_len = x.shape[1]
        pad = prefix_lm_pad_len(
            seq_len,
            self.config.tensor_model_parallel_size,
            self.config.sequence_parallel,
            align=self.config.hyperencoder_seq_align,
        )
        if pad > 0:
            x = paddle.concat([x, paddle.zeros([bs, pad, x.shape[-1]], dtype=x.dtype)], axis=1)
        mask = build_dense_mask([n_context], [n_queries], pad)
        # The mask is NOT sharded: with SP enabled, `linear_qkv` (ColumnParallel +
        # sequence_parallel) first all-gathers `[S/tp,B,H]` back to `[S,B,H]`, so
        # attention still sees the full sequence.

        rope = self.rotary_pos_emb(max_seq_len=x.shape[1])
        if self.config.sequence_parallel:
            # PaddleFleet's `RotaryEmbedding.forward` always returns `[1, S, 1, D]`,
            # which is meant for the `[B,S,H]` layout. With SP enabled the tensor
            # becomes time-major `[S,B,...]` and this freqs shape would broadcast
            # incorrectly and silently: query `[640,1,5,128]` x freqs
            # `[1,640,1,128]` -> `[640,640,5,128]`, then fail at
            # `paddle.cat((t, t_pass))` (t_pass is still `[640,1,5,0]`).
            # Here we transpose it at the call site to time-major `[S,1,1,D]`, so
            # the shared rotary core needs no change:
            #   * ndim is still 4, so the `sp_group` sharding inside
            #     `_apply_rotary_pos_emb_bshd` (which only triggers for ndim 2/3)
            #     does not fire -- which is what we want: with SP enabled,
            #     `qkv_proj` has already all-gathered the sequence back to the
            #     full S, so freqs must never be sharded again;
            #   * `len(freqs.shape) == len(t.shape)`, so no unsqueeze is done and
            #     `[S,1,1,D]` broadcasts directly over `[S,B,ng,D]`.
            rope = rope.transpose([1, 0, 2, 3])
            h = self._sp_scatter(x)
            h = self.block(hidden_states=h, attention_mask=mask, rotary_pos_emb=rope)
            h = self._sp_gather(h)
            h = h[:seq_len].transpose([1, 0, 2])
        else:
            # Without SP, PaddleFleet's TransformerBlock expects `[B,S,H]`, so no
            # transpose here; a purely layout convention difference, no numerical
            # effect.
            h = self.block(hidden_states=x, attention_mask=mask, rotary_pos_emb=rope)
            h = h[:, :seq_len]  # non-packed path drops the padding
        return h[:, n_context:, :]  # take only the trailing Q rows

    def forward_decoder_packed(self, context_embeds, use_long_query: bool):
        """Packed single-call forward.

        The only difference from :meth:`forward_decoder` is the geometry: the B
        `(context, query)` segments in the batch are concatenated end-to-end into
        one long sequence, and the mask is no longer a dense `[1,1,T,T]` but the
        integer plan `packed_seq_params.prefix_lm_layout`, consumed directly by
        the `PrefixLMTritonCore` Triton kernel.

        The return shape matches :meth:`forward_decoder` (`[B, Q, H]`) so that
        the output-projection loop in :meth:`forward` is identical on both paths;
        the only difference between the two paths is confined to the attention
        geometry.

        ``use_long_query`` accepts either a **bool** (homogeneous pack, returns
        [B,Q,H]) or a **list[bool]** (mixed short/long query pack, returns a
        per-segment list list[[Q_i,H]]). Real production may pack Q=256 segments
        and Q=8192 segments into the same sequence.
        """
        from paddlefleet.transformer.prefix_lm_mask import prefix_lm_pad_len

        # ``use_long_query`` may be a single **bool** (homogeneous pack, returns
        # ``[B,Q,H]``) or a **list[bool]** per segment (mixed short/long query
        # pack). Mixed packs have different ``Q_i`` per segment, so a per-segment
        # list ``list[[Q_i,H]]`` is returned (the list form lets the
        # output-projection loop in :meth:`forward` treat both paths uniformly).
        # The kernel side already consumes the layout as a per-segment
        # ``n_queries`` tuple.
        # ``context_embeds`` may be an **equal-length rectangle**
        # ``[bs, n_context, H]`` (synthetic use case), or a **variable-length
        # segment list** ``list[[C_i, H]]`` (real production ragged pack). The
        # rectangular branch is allowed additionally so existing equal-length
        # cases stay bit-unchanged (the rectangular branch emits the same op
        # sequence as before).
        if isinstance(context_embeds, (list, tuple)):
            seg_ctx = [c.squeeze(0) if c.ndim == 3 else c for c in context_embeds]  # each [C_i,H]
            n_contexts = [int(c.shape[0]) for c in seg_ctx]
            hidden = int(seg_ctx[0].shape[1])
        else:
            bs, n_context, hidden = context_embeds.shape
            seg_ctx = [context_embeds[i] for i in range(bs)]
            n_contexts = [n_context] * bs
        bs = len(seg_ctx)

        # Per-segment use_long_query (bool -> homogeneous pack; list -> mixed
        # pack, length must equal the number of segments)
        if isinstance(use_long_query, bool):
            uql = [use_long_query] * bs
        else:
            uql = [bool(u) for u in use_long_query]
            if len(uql) != bs:
                raise ValueError(f"use_long_query list length {len(uql)} does not match " f"segment count {bs}")
        # Per-segment query table
        qws = [(self.query_long if u else self.query_short).weight for u in uql]
        nq_list = [int(q.shape[0]) for q in qws]

        # ---- Concatenate segments and record the layout ----
        segments, starts, running = [], [], 0
        for ci, nc, q, nq in zip(seg_ctx, n_contexts, qws, nq_list):
            segments.append(paddle.concat([ci, q], axis=0))
            starts.append(running)
            running += nc + nq
        x = paddle.concat(segments, axis=0).unsqueeze(0)
        seq_len = x.shape[1]

        pad = prefix_lm_pad_len(
            seq_len,
            self.config.tensor_model_parallel_size,
            self.config.sequence_parallel,
            align=self.config.hyperencoder_seq_align,
        )
        if pad > 0:
            x = paddle.concat([x, paddle.zeros([1, pad, hidden], dtype=x.dtype)], axis=1)

        # ---- Per-segment sequence lengths: the last segment absorbs the pad ----
        # The pad is not a segment of its own: it is folded into the last
        # segment's [start, end), and its positions are handled by the kernel's
        # pad diagonal term (`tok_seg_start == -1`). If it were a separate
        # segment, the rope segmentation and the layout segmentation would no
        # longer line up.
        seg_lens = [nc + nq for nc, nq in zip(n_contexts, nq_list)]
        seg_lens[-1] += pad
        max_seqlen = max(seg_lens)

        # `PackedSeqParams` on this path is only the carrier for
        # `prefix_lm_layout`; its numeric fields are left empty -- in particular
        # `cu_seqlens` is NOT set.
        #
        # Why: as soon as `attention.py` sees `cu_seqlens`, it switches rope to
        # the thd branch (`rope_utils._apply_rotary_pos_emb_thd`), whose sharding
        # is hard-coded batch-major (`split(..., axis=1 if ndim==4 else 0)`,
        # `freqs[:, ...]`); with SP enabled the backbone is time-major
        # `[S,B,N,D]`, so it would shard the wrong axis and the `split` sizes
        # would not match. That is a shared file used by models unrelated to
        # HyperEncoder, so it should not be changed for us.
        #
        # So freqs are assembled on the host side instead (`_packed_rope` below):
        # the result is bit-equivalent to the thd branch (at offset=0, cp=1 the
        # thd helper reduces to `freqs[:, :L_i]` then `cat`), while rope takes the
        # bshd branch, which already handles `time_major` correctly. The op
        # sequence is unchanged and the shared file is untouched.
        #
        # Cost: `cu_seqlens` is unused by `PrefixLMTritonCore` anyway (it only
        # reads `prefix_lm_layout`). If the core is later swapped for an
        # implementation that needs `cu_seqlens`, this must change too.
        psp = PackedSeqParams()
        setattr(
            psp,
            PREFIX_LM_LAYOUT_ATTR,
            {
                "segment_starts": tuple(starts),
                "n_contexts": tuple(n_contexts),
                "n_queries": tuple(nq_list),
                "pad_len": pad,
            },
        )

        rope = self._packed_rope(max_seqlen, seg_lens)
        if self.config.sequence_parallel:
            # Same handling as the non-packed path (see the long comment in
            # `forward_decoder`): freqs is `[1,T,1,D]`, and with SP enabled the
            # backbone tensor is time-major `[T,B,H]`, so freqs must also be
            # transposed to `[T,1,1,D]`.
            rope = rope.transpose([1, 0, 2, 3])
            h = self._sp_scatter(x)
            h = self.block(hidden_states=h, rotary_pos_emb=rope, packed_seq_params=psp)
            h = self._sp_gather(h)
            # The packed path keeps the pad (slices to x's length, not to
            # seq_len), because per-segment row extraction uses offsets computed
            # against the pad-inclusive layout.
            h = h[: x.shape[1]].transpose([1, 0, 2])
        else:
            h = self.block(hidden_states=x, rotary_pos_emb=rope, packed_seq_params=psp)

        # ---- Extract the query suffix per segment ----
        # Equal-length (all nq identical, including the bool path): stack back to
        # `[B, Q, H]`, sharing the output loop with the non-packed path;
        # mixed (nq differs per segment): return a per-segment list
        # `list[[Q_i, H]]`, projected segment by segment by the caller.
        outs = [h[0, s + nc : s + nc + nq] for s, nc, nq in zip(starts, n_contexts, nq_list)]
        if len(set(nq_list)) == 1:
            return paddle.stack(outs, axis=0)
        return outs

    def _packed_rope(self, max_seqlen: int, seg_lens: list[int]):
        """Assemble the RoPE freqs for the packed sequence: `[1, ΣL_i, 1, D]`.

        Each segment's position starts from 0 again (the prefix-LM in-segment
        relative position), so this just concatenates `freqs[:, :L_i]` per
        segment.

        ## Why assemble on the host side instead of using the framework's thd branch

        The framework path (switching to thd when `cu_seqlens` is seen) does
        exactly this at `cp=1` (the thd helper at offset=0 equals `freqs[:, :L_i]`
        then `cat`) -- bit-equivalent. But its sharding is hard-coded batch-major,
        and with SP enabled the backbone is time-major, so it would shard the
        wrong axis; and that is a shared file used by unrelated models, so it
        should not be changed for us.

        Assembling here lets rope take the bshd branch, which already handles
        `time_major` correctly (the non-packed path always runs this way). The op
        sequence is unchanged and the shared file is untouched.

        With a single segment the result is `freqs[:, :L_0]`, identical to the
        non-packed case.
        """
        # `RotaryEmbedding.forward` always returns `[1, S, 1, D]`
        freqs = self.rotary_pos_emb(max_seq_len=max_seqlen)
        if len(seg_lens) == 1:
            return freqs[:, : seg_lens[0]]
        return paddle.concat([freqs[:, :L] for L in seg_lens], axis=1)

    def _sp_scatter(self, x):
        """`[B,S,H]` -> `[S/tp,B,H]`."""
        from paddlefleet.tensor_parallel.mappings import (
            scatter_to_sequence_parallel_region,
        )

        return scatter_to_sequence_parallel_region(x.transpose([1, 0, 2]))

    def _sp_gather(self, h):
        """`[S/tp,B,H]` -> `[S,B,H]`.

        This deliberately keeps the default `tensor_parallel_output_grad=True`,
        i.e. reduce-scatter on the backward. At this call site that is arguably
        wrong -- the downstream `out_projector` is a
        ``RowParallelLinear(input_is_parallel=False)``, so each rank's backward
        already receives the full, mutually identical gradient, and summing it
        again scales the backbone gradient by `tp_size` (exactly x2 in practice).
        The "correct" form would be `tensor_parallel_output_grad=False`.

        However, the default value is retained here to keep parity with the
        intended reference behavior; changing it to the "correct" form would
        break the TP configuration's expected gradients.
        """
        from paddlefleet.tensor_parallel.mappings import (
            gather_from_sequence_parallel_region,
        )

        return gather_from_sequence_parallel_region(h)

    def forward(self, context_ids, image=None, audio=None, use_long_query: bool = False):
        """``context_ids`` -> latent ``Z``, shape ``[B*Q, language_hidden_size]``.

        `use_long_query` is an explicit argument here. It selects the short vs
        long query table directly, rather than being inferred from a token stream.
        """
        embeds = self.build_context_embeds(context_ids, image, audio)
        if use_packed_decoder():
            latents = self.forward_decoder_packed(embeds, use_long_query)
        else:
            latents = self.forward_decoder(embeds, use_long_query)
        latents = self._add_unused_query_dependency(latents, use_long_query)
        # Per-batch output projection: the projection returns (out, bias); add
        # bias by hand, then concat along dim 0.
        outs = []
        for i in range(latents.shape[0]):
            projected, bias = self.out_projector(latents[i])
            if bias is not None:
                projected = projected + bias
            outs.append(projected)
        return paddle.concat(outs, axis=0)


@dataclass
class HyperEncoderProvider(GPTConfig, ModelProviderMixin["HyperEncoderModel"]):
    """``HyperEncoderConfig`` -> network builder.

    ## What it replaces

    An earlier ``build_hyperencoder_config()`` factory manually translated a
    long block of literal assignments into a ``GPTConfig``. This class rewrites
    that using the framework's generic mechanism:

    * non-``GPTConfig``-default values become this class's **dataclass field
      defaults** (the block below);
    * names that differ between the HF side and the Fleet side go into
      :attr:`transform_rules`;
    * the four values that must be **derived** go into :meth:`__post_init__`;
    * how ``PretrainedConfig`` values arrive is handled by the inherited
      ``TransformerConfig.from_config`` (``object.__new__`` +
      ``register_attributes`` + ``__post_init__``).

    ``from_config`` uses ``object.__new__`` and does NOT run the dataclass
    ``__init__`` -- field defaults take effect because a simple default becomes a
    class attribute. So this class's fields must NOT use
    ``field(default_factory=...)`` (that would be missing under
    ``object.__new__``).

    ## Field defaults = the model's intended non-default values
    """

    # ---- Structure ----
    gated_linear_unit: bool = True
    normalization: str = "RMSNorm"
    use_bias: bool = False  # add_bias_linear=False
    # ---- Positional encoding ----
    position_embedding_type: str = "rope"
    rotary_percent: float = 1.0
    # ---- MoE ----
    moe_token_dispatcher_type: str = "alltoall"
    moe_expert_fusion: bool = False  # moe_grouped_gemm=False
    moe_router_load_balancing_type: str = "seq_aux_loss"
    routed_scaling_factor_learnable: bool = False
    # ---- dtype ----
    params_dtype: paddle.dtype = paddle.bfloat16
    bf16: bool = True
    attention_softmax_in_fp32: bool = False
    # ---- Fusion switches ----
    masked_softmax_fusion: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = False
    # ---- Kernel-selection compatibility ----
    # Must stay False. It is a bundled switch that would force
    # `attention_softmax_in_fp32=True`, which conflicts with this model's
    # intended False. The MoE permute/unpermute path reads an env flag, not this
    # field, so keeping it False does not lose any needed behavior.
    use_accuracy_compatible: bool = False

    # ---- HyperEncoder-specific geometry (config fields here) ----
    hyperencoder_query_lengths: tuple = (256, 8192)
    hyperencoder_seq_align: int = 128
    #: Output-projection width for encoder -> LLM. In production this is
    #: determined by the LLM's `hidden_size` and passed in as a constructor arg.
    language_hidden_size: int = 1280

    # No `transform_rules`: `HyperEncoderConfig` field names are exactly the same
    # as on the Fleet side (`num_hidden_layers` / `hidden_size` /
    # `n_routed_experts` / `rms_norm_eps` / `rope_theta` ... are all formal
    # fields of `TransformerConfig`), so none needs renaming. Defining an
    # identity table would instead shadow the base class's DSA/CSA mapping table.

    def __post_init__(self) -> None:
        """Four derivations plus one divisibility check.

        ``from_config`` runs ``register_attributes`` then this method, so the
        values seen here are already those from ``config.json`` / yaml;
        derivations must go here (class defaults would be overwritten).

        Two things must be done first, two last:

        * ``params_dtype`` must be restored first -- ``PretrainedConfig.__init__``
          always sets ``self.dtype = None``, and ``_process_attribute`` maps
          ``dtype`` to ``params_dtype``, so our ``bfloat16`` class default would
          be overwritten by ``None``;
        * ``recompute_*`` must be set AFTER ``super().__post_init__()`` -- the
          intended sequence sets them only after ``GPTConfig(...)`` is
          constructed, so the parent's post_init sees "recompute off". A
          different order would take a different validation branch in the parent.
        """
        if getattr(self, "params_dtype", None) is None:
            self.params_dtype = paddle.bfloat16

        # `hidden_act` can only be set here, NOT as a class field default --
        # `from_config` uses `object.__new__`, and field defaults exist as class
        # attributes; a plain function on a class attribute becomes a method, so
        # `getattr` returns a *bound method* and downstream calls would get an
        # extra self argument.
        self.hidden_act = F.silu  # activation_func

        # For the fields below, the ``LlmMetaConfig`` defaults in PaddleFormers
        # differ from the ``GPTConfig`` dataclass defaults in PaddleFleet, and
        # ``register_attributes`` copies the former in. The intended config uses
        # the latter, so we explicitly restore the Fleet defaults here. Pinning
        # these to the model's intended (non-fused) kernels matters: without
        # pinning, base-config defaults would flip them and change the kernels
        # used (`moe_router_fusion` / `situ_glu_fusion` /
        # `fp32_residual_connection` all switch kernels). This table is guarded
        # field by field by test_hyperencoder_config_roundtrip.py.
        self.moe_router_fusion = False
        self.situ_glu_fusion = False
        self.fp32_residual_connection = False
        self.moe_expert_capacity_factor = None
        self.moe_subbatch_token_num_before_dispatch = None
        self.train_mtp_only = False
        self.pad_token_id = 0

        # Detach recompute first, so the parent's post_init sees "recompute off"
        # -- matching the intended sequence (which sets it only after
        # `GPTConfig(...)` is constructed). Without detaching, the parent would
        # hit `recompute_method is None` and raise
        # "when recompute_granularity=full, recompute_method must be one of ...".
        # The two fields below are pinned values that this class would otherwise
        # overwrite; they must be caught explicitly rather than silently ignored,
        # because "yaml changed but had no effect" looks exactly like a real
        # numerical bug.
        for name, pinned, where in (
            ("recompute_method", "uniform", "recompute_method"),
            ("recompute_num_layers", 1, "recompute_num_layers"),
        ):
            got = getattr(self, name, None)
            if got is not None and got != pinned:
                raise ValueError(
                    f"{name}={got!r} is not allowed ({where} is pinned to {pinned!r}). "
                    "HyperEncoder recompute has only one degree of freedom: "
                    "`recompute_granularity in (None, 'full')`."
                )
        rg = getattr(self, "recompute_granularity", None)
        if rg not in (None, "full"):
            raise ValueError(
                f"recompute_granularity={rg!r} is not supported; only 'full' is " "allowed, set None to disable."
            )
        if getattr(self, "sequence_parallel", False):
            raise ValueError(
                "Do not set sequence_parallel explicitly: it is derived from "
                "`(tp_size > 1)`. Set only tensor_model_parallel_size."
            )

        _recompute_on = rg is not None
        self.recompute_granularity = None

        ql = tuple(int(v) for v in self.hyperencoder_query_lengths)
        if len(ql) != 2 or ql[0] <= 0 or ql[1] <= 0:
            raise ValueError(f"hyperencoder_query_lengths must be two positive integers, got {ql}")
        self.hyperencoder_query_lengths = ql
        # seq_length = max_position_embeddings = long_q + 8192.
        # This is distinct from the LLM-side --seq-length; also, the SFT workflow
        # writes data_args.max_seq_len onto the config, so it must be overridden
        # back here rather than relying on the value in config.
        self.max_sequence_length = ql[1] + 8192

        self.head_dim = self.hidden_size // self.num_attention_heads

        # moe_layer_freq (= [0] + [1]*11) does NOT need to be computed by hand --
        # the parent's post_init derives the same list from `first_k_dense_replace`
        # (`[0]*k + [1]*(n-k)`), and providing both would raise. The old factory
        # computed moe_layer_freq manually because it did not run the parent's
        # post_init.

        # ETP == TP; SP is derived from the real tp_size.
        tp = int(self.tensor_model_parallel_size or 1)
        self.expert_tensor_parallel_size = tp
        self.sequence_parallel = tp > 1

        # The encoder only supports PP=1 (see `_check_divisibility`), so VPP is
        # meaningless. `HyperEncoderConfig` may carry a 1 from `LlmMetaConfig`;
        # normalize it to None here, and raise for any other value rather than
        # silently ignoring it.
        vpp = getattr(self, "virtual_pipeline_model_parallel_size", None)
        if vpp not in (None, 1):
            raise ValueError(
                f"virtual_pipeline_model_parallel_size={vpp} is not supported: encoder only supports PP=1"
            )
        self.virtual_pipeline_model_parallel_size = None

        self._check_divisibility()

        super().__post_init__()

        # `first_k_dense_replace`'s only purpose is to let the parent derive
        # `moe_layer_freq` (see the comment above). After derivation it is reset
        # to None to match the intended config and avoid any downstream reader.
        self.first_k_dense_replace = None

        # ---- Recompute. See the docstring: must be set after super() ----
        if _recompute_on:
            self.recompute_granularity = "full"
            self.recompute_method = "uniform"
            self.recompute_num_layers = 1

    def _check_divisibility(self) -> None:
        """The divisibility assertions for the model geometry.

        Note that the TP divisibility of ``moe_shared_expert_intermediate_size``
        is intentionally NOT checked here, matching the intended behavior; adding
        the check would raise in some configurations where the model otherwise
        would not, causing a behavior difference.
        """
        tp = int(self.tensor_model_parallel_size or 1)
        ep = int(self.expert_model_parallel_size or 1)
        pp = int(self.pipeline_model_parallel_size or 1)
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("num_attention_heads", self.num_attention_heads),
            ("ffn_hidden_size", self.intermediate_size),
            ("moe_ffn_hidden_size", self.moe_intermediate_size),
        ):
            if value % tp != 0:
                raise ValueError(f"encoder {name}={value} must be divisible by encoder TP={tp}")
        if self.n_routed_experts % ep != 0:
            raise ValueError(f"encoder num_moe_experts={self.n_routed_experts} must be divisible by encoder EP={ep}")
        if pp != 1:
            raise ValueError("HyperEncoder currently supports encoder PP=1 only")

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "HyperEncoderModel":
        """Build the model. Signature matches ``ModelProviderMixin.provide``.

        The encoder currently only supports PP=1 (``_check_divisibility``), so
        ``pre_process`` / ``post_process`` / ``vp_stage`` are accepted but unused
        -- they are kept to satisfy the mixin contract.
        """
        return HyperEncoderModel(self, language_hidden_size=self.language_hidden_size)


class HyperEncoderModelFleet(PretrainedModel):
    """The `AutoModel` / CLI entry wrapper.

    `__new__` returns a `HyperEncoderModel` (a `FleetLayer`), NOT an instance of
    `cls`, so `PretrainedModel.__init__` is not called (Python semantics). The
    `PretrainedModel` base class exists only so `AutoModel` can recognize it and
    obtain `config_class`.
    """

    # Registration goes through `MODEL_NAMES_MAPPING` by exact class name (value
    # = "HyperEncoderModelFleet"), not the `{prefix}Model` naming convention, so
    # the wrapper name is free and the `HyperEncoderModel` network need not be
    # renamed.
    config_class = HyperEncoderConfig
    base_model_prefix = "hyperencoder"

    def __new__(cls, config: HyperEncoderConfig):
        # Clamp parallelism degrees to >= 1 (None/0 if not set in yaml).
        for name in (
            "tensor_model_parallel_size",
            "expert_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
        ):
            setattr(config, name, max(int(getattr(config, name, 1) or 1), 1))
        provider = HyperEncoderProvider.from_config(config)
        model = provider.provide()
        model.config_to_save = config
        model._gen_aoa_config = cls._gen_aoa_config
        model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        return model

    # ---- Warm-start: AoA weight-name mapping (used by flex_checkpoint) ----
    #
    # This AoA mapping is derived from the offline HF-to-model conversion plus
    # the `fused_qkv` / `fused_ffn` / `^T` tags shared by all MoE models in the
    # repo.
    #
    # HF source key prefixes:
    #   decoder backbone  model.encoder.decoder.model.model.{embed_tokens,norm,layers.N.*}
    #   image/audio towers  model.encoder.{image_encoder,audio_encoder,projector}.*
    #   query tables      model.encoder.decoder.query_{short,long}.weight
    #   output projection model.projector.{weight,bias}
    _HF_DEC = "model.encoder.decoder.model.model"
    _HF_ENC = "model.encoder"

    @classmethod
    def _gen_aoa_config(cls, config):
        """Declarative mapping from HF checkpoint key to this model's state_dict key."""
        dec, enc = cls._HF_DEC, cls._HF_ENC
        n_layers = int(config.num_hidden_layers)
        # `first_k_dense_replace` is reset to None after use in
        # Provider.__post_init__ (it was only used to derive moe_layer_freq), so
        # it cannot be read here. The number of dense layers = the count of
        # leading zeros in moe_layer_freq (moe_layer_freq[i]==0 means layer i is
        # dense).
        freq = list(getattr(config, "moe_layer_freq", None) or [1] * n_layers)
        dense_layers = {i for i, v in enumerate(freq) if v == 0}
        nh = int(config.num_attention_heads)
        kvh = int(getattr(config, "num_key_value_heads", nh) or nh)

        st = []
        # --- Front end: towers / embedding / query / output projection ---
        # conv / pos_embed layouts match on both sides, so no transpose
        for name in (
            "image_encoder.pos_embed",
            "image_encoder.patch_embed.proj.weight",
            "image_encoder.patch_embed.proj.bias",
            "audio_encoder.pos_embed",
            "audio_encoder.conv1.weight",
            "audio_encoder.conv1.bias",
            "audio_encoder.conv2.weight",
            "audio_encoder.conv2.bias",
        ):
            st.append(f"{enc}.{name} -> {name}")
        # The modality projector is a Linear, so transpose (paddle stores [in,out])
        st.append(f"{enc}.projector.layers.weight^T -> projector.layers.weight")
        st.append(f"{enc}.projector.layers.bias -> projector.layers.bias")
        # embedding: torch.nn.Embedding, both sides [vocab,hidden], no transpose
        st.append(f"{dec}.embed_tokens.weight -> embed_tokens.weight")
        # query tables: nn.Embedding weight, no transpose
        st.append(f"{enc}.decoder.query_short.weight -> query_short.weight")
        st.append(f"{enc}.decoder.query_long.weight -> query_long.weight")
        # Final backbone norm
        st.append(f"{dec}.norm.weight -> block.norm.weight")
        # Output projection (encoder->LLM), RowParallelLinear, so transpose
        st.append("model.projector.weight^T -> out_projector.weight")
        st.append("model.projector.bias -> out_projector.bias")

        # --- Per layer ---
        for L in range(n_layers):
            hf = f"{dec}.layers.{L}"
            me = f"block.layers.{L}"
            st.append(f"{hf}.input_layernorm.weight -> {me}.input_layernorm.weight")
            # post_attention_layernorm is used as pre_mlp_layernorm; our key still
            # calls it post_attention
            st.append(f"{hf}.post_attention_layernorm.weight -> {me}.post_attention_layernorm.weight")
            # QKV: interleave + transpose, reusing the fused_qkv tag (common to all MoE)
            st.append(
                f"{hf}.self_attn.q_proj.weight^T, {hf}.self_attn.k_proj.weight^T, "
                f"{hf}.self_attn.v_proj.weight^T -> {me}.self_attn.qkv_proj.weight, "
                f"fused_qkv, num_heads={nh}, num_key_value_groups={kvh}"
            )
            st.append(f"{hf}.self_attn.o_proj.weight^T -> {me}.self_attn.o_proj.weight")
            if L in dense_layers:
                # dense layer
                st.append(
                    f"{hf}.mlp.gate_proj.weight^T, {hf}.mlp.up_proj.weight^T -> "
                    f"{me}.mlp.up_gate_proj.weight, fused_ffn"
                )
                st.append(f"{hf}.mlp.down_proj.weight^T -> {me}.mlp.down_proj.weight")
            else:
                # MoE layer: router not transposed; shared + routed experts use fused_ffn
                st.append(f"{hf}.mlp.gate.weight -> {me}.mlp.gate.weight")
                st.append(
                    f"{hf}.mlp.shared_experts.gate_proj.weight^T, "
                    f"{hf}.mlp.shared_experts.up_proj.weight^T -> "
                    f"{me}.mlp.shared_experts.up_gate_proj.weight, fused_ffn"
                )
                st.append(
                    f"{hf}.mlp.shared_experts.down_proj.weight^T -> " f"{me}.mlp.shared_experts.down_proj.weight"
                )
                st.append(
                    f"{hf}.mlp.experts.$EXPERT_ID.gate_proj.weight^T, "
                    f"{hf}.mlp.experts.$EXPERT_ID.up_proj.weight^T -> "
                    f"{me}.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn"
                )
                st.append(
                    f"{hf}.mlp.experts.$EXPERT_ID.down_proj.weight^T -> "
                    f"{me}.mlp.experts.$EXPERT_ID.down_proj.weight"
                )
        return {"aoa_statements": st}

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        """Save direction (this model -> HF). Currently only used for `save_pretrained`."""
        # The reverse is derived automatically by flex_checkpoint from the
        # forward mapping; this placeholder satisfies the interface.
        raise NotImplementedError(
            "HyperEncoder inverse AoA is not implemented; flex_checkpoint derives "
            "the reverse automatically from _gen_aoa_config."
        )
