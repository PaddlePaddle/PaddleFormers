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

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import paddle
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    ScheduleNode,
    build_spec_layer,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ScatterOp,
)

from paddleformers.fleet.context_parallel_utils import (
    ContextParallelScatterOp,
    mark_context_parallel_parameter_disable_scale_grad,
)
from paddleformers.fleet.models.gpt.utils import fill_feature
from paddleformers.fleet.parallel_state import (
    get_context_parallel_world_size,
)
from paddleformers.fleet.tensor_parallel.mappings import (
    scatter_to_sequence_parallel_region,
)
from paddleformers.fleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddle import Tensor

    from paddleformers.fleet.packed_seq_params import PackedSeqParams
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig


@dataclass
class GPTEmbeddingSpec:
    language_embedding: LayerSpec
    rope_embedding: LayerSpec | None


class GPTEmbedding(FleetLayer):
    def __init__(
        self,
        sublayers_spec: GPTEmbeddingSpec,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "none"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        swa_rotary_base: int = 10000,
        rope_scaling: bool = False,
        mrope_section: list[int] | None = None,
    ):
        super().__init__(config)
        self.embedding = build_spec_layer(
            sublayers_spec.language_embedding,
            config=config,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type=position_embedding_type,
        )
        self.sequence_parallel = self.config.sequence_parallel

        self.multimodal_embedding = config.multimodal_embedding
        if self.sequence_parallel and (
            self.multimodal_embedding
            or (
                config.num_nextn_predict_layers is not None
                and self.config.num_nextn_predict_layers > 0
                and not config.mtp_load_weight_only
            )
        ):
            self.embedding.embed_tokens.reduce_scatter_embeddings = False
            self.embedding.scatter_to_sequence_parallel = False
            self.embedding.reduce_scatter_embeddings = False
            self.embedding.sequence_parallel = False

        if self.config.experimental_dataflow:
            # In EB data flow, since CP scatter is apply after embedding,
            # we need to disable scale grad for the parameters that need to be scattered to each cp local.
            mark_context_parallel_parameter_disable_scale_grad(
                self.embedding.embed_tokens
            )

        self.rotary_pos_emb = None
        self.swa_rotary_pos_emb = None
        self.mrope_section = mrope_section
        self.position_embedding_type = position_embedding_type
        if sublayers_spec.rope_embedding is not None:
            self.rotary_pos_emb = build_spec_layer(
                sublayers_spec.rope_embedding,
                head_dim=config.head_dim,
                rotary_percent=rotary_percent,
                rotary_interleaved=config.rotary_interleaved,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                use_accuracy_compatible=getattr(
                    config, "use_accuracy_compatible", False
                ),
            )

            if config.sliding_window is not None:
                if config.window_attn_skip_freq is None:
                    warnings.warn(
                        "sliding_window is set but window_attn_skip_freq is None. "
                        "is_layer_window_attention() will return True for all layers, "
                        "meaning all layers will use sliding window attention (SWA)."
                    )
                self.swa_rotary_pos_emb = build_spec_layer(
                    sublayers_spec.rope_embedding,
                    head_dim=config.swa_head_dim,
                    rotary_percent=rotary_percent,
                    rotary_interleaved=config.rotary_interleaved,
                    rotary_base=swa_rotary_base,
                    rope_scaling=rope_scaling,
                )

    @property
    def embedding_weight(self):
        return self.embedding.embedding_weight

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="GPTEmbedding")

    def forward(
        self,
        dict_args: dict,
        decoder_input: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        if self.config.gpt_model_use_experimental_version:
            assert (
                getattr(self.config, "max_sequence_length", None) is not None
            ), (
                "config.max_sequence_length must be set when gpt_model_use_experimental_version=True"
            )
            if self.config.sequence_parallel:
                assert not self.config.multi_latent_attention, (
                    "multi_latent_attention is not supported when gpt_model_use_experimental_version=True and sequence_parallel=True"
                )
        input_ids = dict_args["input_ids"]
        labels = dict_args.get("labels", None)
        if labels is not None:
            labels = labels.cuda()
        position_ids = dict_args.get("position_ids", None)
        device = paddle.device.get_device().split(":")[0].lower()
        position_ids = (
            position_ids.to(device) if position_ids is not None else None
        )
        attention_mask = dict_args.get("attention_mask", None)
        attn_mask_startend_row_indices = dict_args.get(
            "attn_mask_startend_row_indices", None
        )
        # Fallback: ernie5 trainer uses "startend_row_indices" key name
        if attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = dict_args.get(
                "startend_row_indices", None
            )
        attn_mask_startend_row_indices = (
            attn_mask_startend_row_indices.to(device)
            if attn_mask_startend_row_indices is not None
            else None
        )
        deepstack_image_embeds = dict_args.get("deepstack_image_embeds", None)
        deepstack_video_embeds = dict_args.get("deepstack_video_embeds", None)
        visual_pos_masks = None
        # Deepstack
        deepstack_visual_embeds = None
        visual_pos_mask = None
        mtp_emb_res = None
        if input_ids is None and decoder_input is None:
            assert dict_args["decoder_input"] is not None, (
                "input_ids or decoder_input must be provided"
            )
            decoder_input = dict_args["decoder_input"]

        # The input_ids_for_moe_mask for moe router is same as input_ids.
        # The moe router will use it to generate the padding mask for the current sequence.
        input_ids_for_moe_mask = None
        # Per-depth MTP input_ids for MoE routing in MTP layers.
        # Shape: [B, num_mtp, max_seq] when MTP is enabled, None otherwise.
        mtp_input_ids_for_moe_mask = None
        if decoder_input is None:
            decoder_input = self.embedding(
                input_ids=input_ids,
                position_ids=None
                if self.multimodal_embedding
                else position_ids,
            )
            # Padding-Token is 0，avoiding Grad updating (ernie_core fill_feature func）
            if (
                self.config.expert_model_parallel_size > 1
                and self.config.tensor_model_parallel_size < 2
                or self.config.gpt_model_use_experimental_version
            ):
                pad_token_id = getattr(self.config, "pad_token_id", 0)
                if pad_token_id is None:
                    pad_token_id = 0
                text_padding_indices = input_ids == pad_token_id
                decoder_input = fill_feature(
                    decoder_input, text_padding_indices, 0
                )
                input_ids_for_moe_mask = input_ids
            if (
                self.config.num_nextn_predict_layers is not None
                and self.config.num_nextn_predict_layers > 0
                and not self.config.mtp_load_weight_only
            ):
                assert not self.multimodal_embedding, (
                    "MTP not support mm for now."
                )
                # Split input_ids for MoE mask: main part for backbone, per-depth for MTP
                if input_ids_for_moe_mask is not None:
                    # Main backbone input_ids: [B, max_seq]
                    # Use .contiguous() because slices are non-contiguous and PP P2P send requires contiguous tensors.
                    input_ids_for_moe_mask = input_ids[
                        :, : -self.config.num_nextn_predict_layers
                    ].contiguous()
                    # Construct per-depth MTP input_ids: for depth k, use
                    # input_ids[:, (k+1):(k+1+max_seq)] matching embedding shift
                    seq_length = (
                        input_ids.shape[1]
                        - self.config.num_nextn_predict_layers
                    )
                    mtp_ids_list = []
                    for depth in range(self.config.num_nextn_predict_layers):
                        mtp_ids_list.append(
                            input_ids[:, (depth + 1) : (depth + 1 + seq_length)]
                        )
                    # [B, num_mtp, max_seq] - paddle.stack creates a new contiguous tensor
                    mtp_input_ids_for_moe_mask = paddle.stack(
                        mtp_ids_list, axis=1
                    )

                if self.config.enable_mtp_magic_send:
                    # Magic send: only truncate, skip shifted embedding pre-computation.
                    # input_ids will be broadcast to the last stage for re-embedding.
                    decoder_input = decoder_input[
                        :, : -self.config.num_nextn_predict_layers, :
                    ]

                    # Apply the same SP scatter as the non-magic-send path to ensure
                    # bit-for-bit identical main embedding output.
                    if (
                        get_context_parallel_world_size() > 1
                        and self.config.experimental_dataflow
                    ):
                        decoder_input = ContextParallelScatterOp.apply(
                            decoder_input,
                            axis=1,
                            mode=self.config.cp_balance_mode,
                        )
                    if (
                        self.config.gpt_model_use_experimental_version
                        and self.config.sequence_parallel
                    ):
                        decoder_input = decoder_input.astype(
                            self.embedding.embed_tokens.weight.dtype
                        )
                    if self.sequence_parallel:
                        batch_size, seq_length, hidden_size = (
                            decoder_input.shape
                        )
                        decoder_input = decoder_input.reshape(
                            [-1, decoder_input.shape[-1]]
                        )
                        decoder_input = ScatterOp.apply(decoder_input)
                        if not (
                            self.config.gpt_model_use_experimental_version
                            and self.config.sequence_parallel
                        ):
                            decoder_input = (
                                decoder_input.reshape(
                                    [batch_size, -1, hidden_size]
                                )
                                .permute(1, 0, 2)
                                .contiguous()
                            )  # change to [S/tp, B, H]
                else:
                    inputs_embeds_extra = decoder_input[
                        :, -self.config.num_nextn_predict_layers :, :
                    ]  # [B, S, H]
                    inputs_embeds = decoder_input[
                        :, : -self.config.num_nextn_predict_layers, :
                    ]
                    inputs_embeds_ori = inputs_embeds
                    batch_size, seq_length, hidden_size = inputs_embeds.shape

                    if (
                        get_context_parallel_world_size() > 1
                        and self.config.experimental_dataflow
                    ):
                        # In EB data flow, main input embed apply CP scatter here
                        inputs_embeds = ContextParallelScatterOp.apply(
                            inputs_embeds,
                            axis=1,
                            mode=self.config.cp_balance_mode,
                        )

                    if self.sequence_parallel:
                        inputs_embeds = inputs_embeds.reshape(
                            [-1, inputs_embeds.shape[-1]]
                        )
                        inputs_embeds = ScatterOp.apply(inputs_embeds)
                        inputs_embeds = (
                            inputs_embeds.reshape([batch_size, -1, hidden_size])
                            .permute(1, 0, 2)
                            .contiguous()
                        )  # change to [S, B, H]
                    mtp_emb_res = [inputs_embeds]
                    for depth in range(self.config.num_nextn_predict_layers):
                        inputs_embeds_mtp = paddle.concat(
                            [
                                inputs_embeds_ori[:, (depth + 1) :, :],
                                inputs_embeds_extra[:, : (depth + 1), :],
                            ],
                            axis=1,
                        )

                        if (
                            get_context_parallel_world_size() > 1
                            and self.config.experimental_dataflow
                        ):
                            # In EB data flow, mtp input embed apply CP scatter here
                            inputs_embeds_mtp = ContextParallelScatterOp.apply(
                                inputs_embeds_mtp,
                                axis=1,
                                mode=self.config.cp_balance_mode,
                            )

                        if self.sequence_parallel:
                            inputs_embeds_mtp = inputs_embeds_mtp.reshape(
                                [-1, inputs_embeds_mtp.shape[-1]]
                            )
                            inputs_embeds_mtp = ScatterOp.apply(
                                inputs_embeds_mtp
                            )
                            inputs_embeds_mtp = (
                                inputs_embeds_mtp.reshape(
                                    [batch_size, -1, hidden_size]
                                )
                                .permute(1, 0, 2)
                                .contiguous()
                            )  # change to [S, B, H]
                        mtp_emb_res.append(inputs_embeds_mtp)

            if self.multimodal_embedding:
                image_embeds = dict_args.get("image_embeds", None)
                video_embeds = dict_args.get("video_embeds", None)
                if image_embeds is not None:
                    image_mask, _ = self.get_placeholder_mask(
                        input_ids,
                        inputs_embeds=decoder_input,
                        image_features=image_embeds,
                    )
                    # Replace masked_scatter with arithmetic blend to avoid
                    # IndexingBackwardKernel (sparse scatter) in the backward pass.
                    #   image_mask : [B, S, H] bool
                    #   image_embeds: [N_img, H]  (N_img = number of image tokens)
                    # Expand image_embeds into the full [B, S, H] space by:
                    #   1. flatten decoder_input and image_mask to 1-D
                    #   2. use paddle.scatter (dense backward = gather) to place
                    #      image_embeds values at the True positions
                    #   3. blend with original decoder_input via mask arithmetic
                    #
                    # Optimization: reuse decoder_input's flattened buffer as the
                    # scatter base (scaled by (1-mask)) to avoid a separate
                    # paddle.zeros([n_total]) allocation (~192 MB bf16 tensor).
                    image_mask_f = image_mask.astype(
                        decoder_input.dtype
                    )  # [B,S,H] float
                    flat_indices = paddle.nonzero(
                        image_mask.reshape([-1])
                    ).squeeze(
                        -1
                    )  # [N_img*H] int64 — dense nonzero, no scatter bwd
                    # Scale the base tensor by (1 - mask) in-place before scatter
                    # so that visual positions are zero — no extra zeros allocation.
                    base_flat = (decoder_input * (1.0 - image_mask_f)).reshape(
                        [-1]
                    )
                    image_src_flat = paddle.scatter(
                        base_flat,
                        flat_indices,
                        image_embeds.astype(decoder_input.dtype).reshape([-1]),
                    )  # scatter bwd is a simple gather — no sparse atomics
                    decoder_input = image_src_flat.reshape(decoder_input.shape)
                    visual_pos_masks = image_mask[..., 0]
                    deepstack_visual_embeds = deepstack_image_embeds

                if video_embeds is not None:
                    _, video_mask = self.get_placeholder_mask(
                        input_ids,
                        inputs_embeds=decoder_input,
                        video_features=video_embeds,
                    )
                    video_mask_f = video_mask.astype(decoder_input.dtype)
                    flat_indices = paddle.nonzero(
                        video_mask.reshape([-1])
                    ).squeeze(-1)
                    base_flat = (decoder_input * (1.0 - video_mask_f)).reshape(
                        [-1]
                    )
                    video_src_flat = paddle.scatter(
                        base_flat,
                        flat_indices,
                        video_embeds.astype(decoder_input.dtype).reshape([-1]),
                    )
                    decoder_input = video_src_flat.reshape(decoder_input.shape)
                    visual_pos_masks = video_mask[..., 0]
                    deepstack_visual_embeds = deepstack_video_embeds

                if image_embeds is not None and video_embeds is not None:
                    image_mask = image_mask[..., 0]  # [B, S] bool
                    video_mask = video_mask[..., 0]  # [B, S] bool
                    visual_pos_masks = image_mask | video_mask
                    deepstack_visual_embeds = []
                    for img_embed, vid_embed in zip(
                        deepstack_image_embeds, deepstack_video_embeds
                    ):
                        # Build embed_joint [N_visual, H] without boolean-index
                        # scatter. Use dense mask arithmetic instead.
                        #   img_embed : [N_img, H]
                        #   vid_embed : [N_vid, H]
                        #   visual_pos_masks: [B, S] bool, N_visual True entries
                        # img_mask_in_visual[i] = True  iff visual position i is image
                        # Computed as: image_mask flattened, keep only visual positions,
                        # expressed as a dense [N_visual] float mask — no indexing.
                        h = img_embed.shape[-1]
                        n_visual = int(visual_pos_masks.sum())
                        # visual_pos_flat: [B*S] bool
                        visual_pos_flat = visual_pos_masks.reshape([-1])
                        image_mask_flat = image_mask.reshape([-1])  # [B*S] bool
                        video_mask_flat = video_mask.reshape([-1])  # [B*S] bool
                        # Dense [B*S] float masks, then compress to [N_visual] via
                        # paddle.masked_select (forward: gather, backward: scatter_add
                        # — but scalar backward is efficient, no sparse atomics)
                        img_mask_in_vis_f = paddle.masked_select(
                            image_mask_flat.astype(img_embed.dtype),
                            visual_pos_flat,
                        ).unsqueeze(-1)  # [N_visual, 1]
                        vid_mask_in_vis_f = paddle.masked_select(
                            video_mask_flat.astype(vid_embed.dtype),
                            visual_pos_flat,
                        ).unsqueeze(-1)  # [N_visual, 1]
                        embed_joint = (
                            img_embed.reshape([n_visual, h]) * img_mask_in_vis_f
                            + vid_embed.reshape([n_visual, h])
                            * vid_mask_in_vis_f
                        )
                        deepstack_visual_embeds.append(embed_joint)
                # Scatter decoder_input to SP format [S/tp, B, H] after multimodal
                # token replacement, since LanguageModelEmbedding's internal scatter
                # was disabled to allow image/video embedding insertion first.
                if self.sequence_parallel:
                    decoder_input = decoder_input.transpose(
                        [1, 0, 2]
                    ).contiguous()
                    decoder_input = scatter_to_sequence_parallel_region(
                        decoder_input, group=self.embedding.tp_group
                    )
                    if self.config.clone_scatter_output_in_embedding:
                        decoder_input = decoder_input.clone()
            # CP scatter for the plain (no-MTP, no-multimodal) path must happen
            # before rope generation so that get_rotary_seq_len sees local seq len.
            if (
                not self.multimodal_embedding
                and not (
                    self.config.num_nextn_predict_layers
                    and self.config.num_nextn_predict_layers > 0
                    and not self.config.mtp_load_weight_only
                )
                and get_context_parallel_world_size() > 1
                and self.config.experimental_dataflow
            ):
                assert not self.sequence_parallel, (
                    "sequence_parallel is not supported when context_parallel scatter "
                    "is applied in the plain (no-MTP, no-multimodal) path before RoPE "
                    "generation."
                )
                decoder_input = ContextParallelScatterOp.apply(
                    decoder_input, axis=1, mode=self.config.cp_balance_mode
                )

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        swa_rotary_pos_emb = None
        swa_rotary_pos_cos = None
        swa_rotary_pos_sin = None

        # For MTP mode: truncate position_ids to match the actual sequence length
        # MTP reduces sequence length by num_nextn_predict_layers
        mtp_position_ids = position_ids
        if (
            mtp_emb_res is not None
            and position_ids is not None
            and self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
        ):
            # mtp_emb_res[0] has shape [B, seq_len - num_nextn_predict_layers, H]
            actual_seq_len = mtp_emb_res[0].shape[1]
            if position_ids.shape[1] > actual_seq_len:
                mtp_position_ids = position_ids[:, :actual_seq_len]

        if (
            self.position_embedding_type == "rope"
            and self.rotary_pos_emb is not None
        ):
            rope_base = decoder_input if mtp_emb_res is None else mtp_emb_res[0]
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                rope_base, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
                position_ids=None if self.training else mtp_position_ids,
            )
        elif (
            self.position_embedding_type == "mrope"
            and self.rotary_pos_emb is not None
        ):
            rotary_pos_emb = self.rotary_pos_emb(
                position_ids, self.mrope_section
            )

        if rotary_pos_emb is not None:
            if self.config.apply_rope_fusion:
                rotary_pos_cos = paddle.cos(rotary_pos_emb)
                rotary_pos_sin = paddle.sin(rotary_pos_emb)
            if self.config.sequence_parallel:
                if self.position_embedding_type == "mrope":
                    # MRoPE: [B, S, head_dim] -> [S, B, head_dim]
                    rotary_pos_emb = rotary_pos_emb.transpose(
                        [1, 0, 2]
                    ).contiguous()
                else:
                    # RoPE: [1, S, 1, head_dim] -> [S, 1, 1, head_dim]
                    rotary_pos_emb = rotary_pos_emb.transpose(
                        [1, 0, 2, 3]
                    ).contiguous()

        if (
            self.position_embedding_type == "rope"
            and self.swa_rotary_pos_emb is not None
        ):
            rope_base = decoder_input if mtp_emb_res is None else mtp_emb_res[0]
            rotary_seq_len = self.swa_rotary_pos_emb.get_rotary_seq_len(
                rope_base, self.config, packed_seq_params
            )
            swa_rotary_pos_emb = self.swa_rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
                position_ids=position_ids,
            )

        elif (
            self.position_embedding_type == "mrope"
            and self.swa_rotary_pos_emb is not None
        ):
            swa_rotary_pos_emb = self.swa_rotary_pos_emb(
                position_ids, self.mrope_section
            )

        if swa_rotary_pos_emb is not None:
            if self.config.apply_rope_fusion:
                swa_rotary_pos_cos = paddle.cos(swa_rotary_pos_emb)
                swa_rotary_pos_sin = paddle.sin(swa_rotary_pos_emb)
            if self.config.sequence_parallel:
                if self.position_embedding_type == "mrope":
                    # MRoPE: [B, S, head_dim] -> [S, B, head_dim]
                    swa_rotary_pos_emb = swa_rotary_pos_emb.transpose(
                        [1, 0, 2]
                    ).contiguous()
                else:
                    # RoPE: [1, S, 1, head_dim] -> [S, 1, 1, head_dim]
                    swa_rotary_pos_emb = swa_rotary_pos_emb.transpose(
                        [1, 0, 2, 3]
                    ).contiguous()

        if paddle.core._has_grad():
            decoder_input.stop_gradient = False  # Prevent errors in recompute_pylayer during LoRA training caused by base_weight lacking gradients.

        # NOTE(Waynezee):  gpt_model_use_experimental_version currently don't need values below
        if self.config.gpt_model_use_experimental_version:
            rotary_pos_emb = None
            rotary_pos_cos = None
            rotary_pos_sin = None
            swa_rotary_pos_emb = None
            swa_rotary_pos_cos = None
            swa_rotary_pos_sin = None

        if (
            get_context_parallel_world_size() > 1
            and self.config.experimental_dataflow
        ):
            if rotary_pos_emb is not None:
                rotary_pos_emb = ContextParallelScatterOp.apply(
                    rotary_pos_emb, axis=1, mode=self.config.cp_balance_mode
                )
            if swa_rotary_pos_emb is not None:
                swa_rotary_pos_emb = ContextParallelScatterOp.apply(
                    swa_rotary_pos_emb, axis=1, mode=self.config.cp_balance_mode
                )
            if rotary_pos_cos is not None:
                rotary_pos_cos = ContextParallelScatterOp.apply(
                    rotary_pos_cos, axis=1, mode=self.config.cp_balance_mode
                )
            if rotary_pos_sin is not None:
                rotary_pos_sin = ContextParallelScatterOp.apply(
                    rotary_pos_sin, axis=1, mode=self.config.cp_balance_mode
                )
            if swa_rotary_pos_cos is not None:
                swa_rotary_pos_cos = ContextParallelScatterOp.apply(
                    swa_rotary_pos_cos, axis=1, mode=self.config.cp_balance_mode
                )
            if swa_rotary_pos_sin is not None:
                swa_rotary_pos_sin = ContextParallelScatterOp.apply(
                    swa_rotary_pos_sin, axis=1, mode=self.config.cp_balance_mode
                )

        preproc_output = {
            "hidden_states": decoder_input.contiguous(),  # prepare for pp send
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "rotary_pos_emb": rotary_pos_emb,
            "rotary_pos_cos": rotary_pos_cos,
            "rotary_pos_sin": rotary_pos_sin,
            "swa_rotary_pos_emb": swa_rotary_pos_emb,
            "swa_rotary_pos_cos": swa_rotary_pos_cos,
            "swa_rotary_pos_sin": swa_rotary_pos_sin,
            "position_ids": position_ids,
            "deepstack_visual_emb": deepstack_visual_embeds,
            "visual_pos_masks": visual_pos_masks,
            "labels": labels,
            "input_ids": input_ids_for_moe_mask,
            "mtp_input_ids_for_moe_mask": mtp_input_ids_for_moe_mask,
            "origin_input_ids": (
                input_ids
                if self.config.gpt_model_use_experimental_version
                else None
            ),
        }
        # New dataflow: pass mtp_startend_row_indices_all and mtp_hidden_inputs_mask_all
        # through dict_args to MTP layer. They must both be present or both be absent.
        mtp_startend_row_indices_all = dict_args.get(
            "mtp_startend_row_indices_all", None
        )
        mtp_hidden_inputs_mask_all = dict_args.get(
            "mtp_hidden_inputs_mask_all", None
        )
        assert (mtp_startend_row_indices_all is None) == (
            mtp_hidden_inputs_mask_all is None
        ), (
            "mtp_startend_row_indices_all and mtp_hidden_inputs_mask_all must both be None or both be not None, "
            f"got mtp_startend_row_indices_all={'None' if mtp_startend_row_indices_all is None else 'not None'}, "
            f"mtp_hidden_inputs_mask_all={'None' if mtp_hidden_inputs_mask_all is None else 'not None'}"
        )
        if mtp_startend_row_indices_all is not None:
            # Ensure tensor is on GPU (dataloader may deliver it as pinned CPU memory).
            # PP P2P communication (NCCL) cannot send pinned tensors directly.
            if not mtp_startend_row_indices_all.place.is_gpu_place():
                mtp_startend_row_indices_all = (
                    mtp_startend_row_indices_all.cuda()
                )
            preproc_output["mtp_startend_row_indices_all"] = (
                mtp_startend_row_indices_all
            )
            if not mtp_hidden_inputs_mask_all.place.is_gpu_place():
                mtp_hidden_inputs_mask_all = mtp_hidden_inputs_mask_all.cuda()
            preproc_output["mtp_hidden_inputs_mask_all"] = (
                mtp_hidden_inputs_mask_all
            )
        if mtp_emb_res is not None:
            assert (
                self.config.num_nextn_predict_layers is not None
                and self.config.num_nextn_predict_layers > 0
                and not self.config.mtp_load_weight_only
            )
            assert len(mtp_emb_res) == self.config.num_nextn_predict_layers + 1
            hidden_states_concat = paddle.concat(mtp_emb_res)
            preproc_output["hidden_states"] = hidden_states_concat

        # Pass through KV cache kwargs for inference
        for key in ("past_key_values", "use_cache"):
            if key in dict_args and key not in preproc_output:
                preproc_output[key] = dict_args[key]

        for key in list(preproc_output.keys()):
            if preproc_output[key] is None:
                preproc_output.pop(key)
        return preproc_output

    def get_placeholder_mask(
        self,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        image_features: Tensor | None = None,
        video_features: Tensor | None = None,
    ):
        """
        Obtain the multimodal placeholder mask from the input and verify whether the number of placeholder tokens matches the length of the multimodal features.
        If the lengths do not match, an error is thrown.
        Args:
            input_ids: Tensor of input token IDs```
            inputs_embeds: input embedding tensor
            image_features: Tensor of image features, optional```
            video_features: Video feature tensor, optional
        Returns:
            tuple: (special_image_mask, special_video_mask) - Mask tensors for image and video tokens
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.embedding(
                paddle.to_tensor(self.config.image_token_id, dtype="int64")
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.embedding(
                paddle.to_tensor(self.config.video_token_id, dtype="int64")
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = int(special_image_mask.sum())
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(
            inputs_embeds
        )

        if (
            image_features is not None
            and n_image_tokens * inputs_embeds.shape[-1]
            != image_features.numel()
        ):
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = int(special_video_mask.sum())
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(
            inputs_embeds
        )
        if (
            video_features is not None
            and n_video_tokens * inputs_embeds.shape[-1]
            != video_features.numel()
        ):
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask
