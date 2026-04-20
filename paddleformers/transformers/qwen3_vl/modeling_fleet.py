# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import types
from collections import OrderedDict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import LayerSpec
from paddle.distributed.fleet.utils import recompute
from paddlefleet import parallel_state, tensor_parallel
from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddlefleet.models.common.vision_layer.vision_layer import VisionLayer
from paddlefleet.models.multimodal.llava_model import LLaVAModel as MCoreLLaVAModel
from paddlefleet.packed_seq_params import PackedSeqParams
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from paddlefleet.transformer.attention import SelfAttention, SelfAttentionSublayersSpec
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType, ModelType
from paddlefleet.transformer.identity_op import IdentityOp
from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.paddle_norm import FusedRMSNorm, LayerNorm
from paddlefleet.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
)
from paddlefleet.utils import WrappedTensor, deprecate_inference_params

from ...nn.criterion.interface import CriterionLayer
from ...nn.pp_model import GeneralModelForCausalLMPipe
from ..cache_utils import Cache
from ..gpt_provider import GPTModelProvider
from ..model_utils import PretrainedModel
from .configuration import Qwen3VLConfig
from .modeling import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLPretrainedModel,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
)


def safe_repeat_interleave_values(values, repeats):
    max_repeats = paddle.max(repeats)
    mask = paddle.arange(max_repeats).unsqueeze(0) < repeats.unsqueeze(1)
    expanded_values = values.unsqueeze(1).expand([values.shape[0], max_repeats])
    result = paddle.masked_select(expanded_values, mask)
    return result


def get_layer_spec(is_vit, normalization) -> LayerSpec:
    """Transformer Layer Spec."""
    attn_mask_type = AttnMaskType.no_mask if is_vit else AttnMaskType.causal
    if normalization == "LayerNorm":
        norm = LayerNorm
    elif normalization == "RMSNorm":
        norm = FusedRMSNorm
    else:
        raise RuntimeError(f"Unknown normalization: {normalization}")

    mlp = get_mlp_module_spec(use_te=False)

    return LayerSpec(
        layer=TransformerLayer,
        sublayers_spec=TransformerLayerSublayersSpec(
            input_layernorm=norm,
            self_attn=LayerSpec(
                layer=SelfAttention,
                extra_kwargs={"attn_mask_type": attn_mask_type},
                sublayers_spec=SelfAttentionSublayersSpec(
                    qkv_proj=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    o_proj=RowParallelLinear,
                    q_norm=IdentityOp,
                    k_norm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_attention_layernorm=norm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_mlp_module_spec(use_te: bool = True) -> LayerSpec:
    return LayerSpec(
        layer=MLP,
        sublayers_spec=MLPSublayersSpec(
            up_gate_proj=ColumnParallelLinear,
            down_proj=RowParallelLinear,
        ),
    )


def get_image_sequence_length(img_h, img_w, patch_dim, add_class_token, class_token_len):
    num_patches_per_dim_h = img_h // patch_dim
    num_patches_per_dim_w = img_w // patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    return num_patches + (class_token_len if add_class_token else 0)


class Qwen3VLTextTransformerLayer(TransformerLayer):
    """Qwen3VL text model for adapt deepstack process"""

    def forward(
        self,
        dict_args: dict,
    ):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        dict_args.pop("dynamic_inference_decode_only", None)
        dict_args.pop("position_ids", None)
        deepstack_visual_emb = dict_args.get("deepstack_visual_emb", None)
        visual_pos_masks = dict_args.get("visual_pos_masks", None)

        if self.full_recompute:
            hidden_states = dict_args["hidden_states"]
            attention_mask = dict_args.get("attention_mask", None)
            attn_mask_startend_row_indices = dict_args.get("attn_mask_startend_row_indices", None)
            context = dict_args.get("context", None)
            context_mask = dict_args.get("context_mask", None)
            rotary_pos_emb = dict_args.get("rotary_pos_emb", None)
            rotary_pos_cos = dict_args.get("rotary_pos_cos", None)
            rotary_pos_sin = dict_args.get("rotary_pos_sin", None)
            attention_bias = dict_args.get("attention_bias", None)
            packed_seq_params = dict_args.get("packed_seq_params", None)

            assert (rotary_pos_sin is None) == (rotary_pos_cos is None)

            outputs = recompute(
                self._forward_impl,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices.clone()  # Clone is necessary!
                if attn_mask_startend_row_indices is not None
                else None,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb.clone() if rotary_pos_emb is not None else None,  # Clone is necessary!
                rotary_pos_cos=rotary_pos_cos.clone() if rotary_pos_cos is not None else None,
                rotary_pos_sin=rotary_pos_sin.clone() if rotary_pos_sin is not None else None,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            outputs = self._forward_impl(**dict_args)

        if isinstance(outputs, tuple):
            output, context = outputs[0], outputs[1]
        else:
            output, context = outputs, None

        # Apply deepstack visual embedding outside of recompute to avoid issues
        # with recompute not properly handling list-of-tensors (deepstack_visual_emb)
        if deepstack_visual_emb and self.layer_number in range(len(deepstack_visual_emb)):
            output = self._deepstack_process(
                hidden_states=output,
                visual_embeds=deepstack_visual_emb[self.layer_number],
                visual_pos_masks=visual_pos_masks,
            )

        rst = OrderedDict()
        rst = {"hidden_states": output}
        if context is not None:
            rst["context"] = context
        rst = {**dict_args, **rst}
        return rst

    def _forward_impl(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor = None,
        attn_mask_startend_row_indices: paddle.Tensor = None,
        context: paddle.Tensor = None,
        context_mask: paddle.Tensor = None,
        rotary_pos_emb: paddle.Tensor = None,
        rotary_pos_cos: paddle.Tensor = None,
        rotary_pos_sin: paddle.Tensor = None,
        attention_bias: paddle.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ):
        # Fix for mRoPE + apply_rope_fusion: ensure sin/cos have correct 4D shape
        if self.config.apply_rope_fusion and rotary_pos_cos is not None and rotary_pos_sin is not None:
            if rotary_pos_cos.ndim == 2 or rotary_pos_cos.ndim == 3:
                # We need to reshape to [1, seq_len, 1, head_dim] (4D) for fused_rotary_position_embedding
                rotary_pos_cos = rotary_pos_cos.reshape([1, -1, 1, rotary_pos_cos.shape[-1]])
                rotary_pos_sin = rotary_pos_sin.reshape([1, -1, 1, rotary_pos_sin.shape[-1]])

        hidden_states, context = self._forward_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
        hidden_states = self._forward_mlp(hidden_states)
        if context is not None:
            return hidden_states, context
        return hidden_states

    def _deepstack_process(
        self, hidden_states: paddle.Tensor, visual_pos_masks: paddle.Tensor, visual_embeds: paddle.Tensor
    ):
        # SP layout is [S/tp, B, H] (seq-first); transpose to [B, S/tp, H] so that
        # flatten(0,1) produces batch-first [B*S/tp, H], consistent with visual_pos_masks [B, S].
        _sp_transposed = False
        if getattr(self.config, "sequence_parallel", False) and hidden_states.ndim == 3:
            hidden_states = hidden_states.transpose([1, 0, 2])  # [S/tp,B,H] -> [B,S/tp,H]
            _sp_transposed = True
        # Save original_shape AFTER the SP transpose so that reshape restores the
        # batch-first [B, S/tp, H] form (needed for the final back-transpose).
        original_shape = hidden_states.shape
        if hidden_states.ndim > 2:
            hidden_states = hidden_states.flatten(start_axis=0, stop_axis=1)

        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        # Sequence Parallelism (SP) row slicing.
        # visual_pos_masks is [B, S] (full sequence), hidden_states is [B*S/tp, H]
        # (batch-major after transpose+flatten). We must slice along the S dimension
        # (dim=1) to match the batch-major layout, NOT flatten-then-chunk which
        # breaks when B > 1.
        if visual_pos_masks.ndim > 1 and visual_pos_masks.shape[1] > hidden_states.shape[0] // max(
            visual_pos_masks.shape[0], 1
        ):
            # visual_pos_masks: [B, S], hidden_states: [B*S/tp, H]
            try:
                from paddle.distributed.fleet import get_hybrid_communicate_group

                hcg = get_hybrid_communicate_group()
                mp_rank = hcg.get_model_parallel_rank()
                mp_size = hcg.get_model_parallel_world_size()
            except (ImportError, AttributeError):
                batch_size = visual_pos_masks.shape[0]
                full_seq_len = visual_pos_masks.shape[1]
                mp_size = (batch_size * full_seq_len) // hidden_states.shape[0]
                mp_rank = paddle.distributed.get_rank() % mp_size

            full_seq_len = visual_pos_masks.shape[1]
            chunk_s = full_seq_len // mp_size
            start_s = mp_rank * chunk_s

            # Slice along S dimension: [B, S] -> [B, S/tp]
            local_mask = visual_pos_masks[:, start_s : start_s + chunk_s]
            batch_size = visual_pos_masks.shape[0]

            # Gather per-sample visual_embeds.
            # visual_embeds is ordered as [sample0_all_vis, sample1_all_vis, ...].
            # Each rank only needs the visual tokens that fall within its local
            # sequence chunk [start_s, start_s+chunk_s) for each sample.
            per_sample_total = paddle.cast(visual_pos_masks, "int32").sum(axis=1)  # [B]
            per_sample_pre = (
                paddle.cast(visual_pos_masks[:, :start_s], "int32").sum(axis=1)
                if start_s > 0
                else paddle.zeros([batch_size], dtype="int32")
            )  # [B]
            per_sample_local = paddle.cast(local_mask, "int32").sum(axis=1)  # [B]

            gather_indices = []
            cumulative_total = 0
            for i in range(batch_size):
                total_i = int(per_sample_total[i].item())
                pre_i = int(per_sample_pre[i].item())
                count_i = int(per_sample_local[i].item())
                if count_i > 0:
                    gather_indices.append(paddle.arange(cumulative_total + pre_i, cumulative_total + pre_i + count_i))
                cumulative_total += total_i

            if gather_indices:
                gather_indices = paddle.concat(gather_indices)
                visual_embeds = visual_embeds[gather_indices]
            else:
                visual_embeds = visual_embeds[:0]  # empty

            # Flatten local mask to [B*S/tp] matching hidden_states batch-major layout
            visual_pos_masks = local_mask.flatten()
        elif visual_pos_masks.ndim > 1:
            visual_pos_masks = visual_pos_masks.flatten()

        # If TP is enabled, hidden_states has shape [..., Hidden_Dim / TP_Size],
        # but visual_embeds usually has full [Hidden_Dim]. We need to slice visual_embeds column-wise.
        if hidden_states.shape[-1] != visual_embeds.shape[-1]:
            try:
                from paddle.distributed.fleet import get_hybrid_communicate_group

                hcg = get_hybrid_communicate_group()
                tp_rank = hcg.get_model_parallel_rank()
                tp_size = hcg.get_model_parallel_world_size()
            except (ImportError, AttributeError):
                # Fallback simple estimation
                tp_size = visual_embeds.shape[-1] // hidden_states.shape[-1]
                tp_rank = paddle.distributed.get_rank() % tp_size

            if tp_size > 1:
                embed_dim = visual_embeds.shape[-1]
                slice_width = embed_dim // tp_size
                start_col = tp_rank * slice_width
                end_col = start_col + slice_width
                visual_embeds = visual_embeds[:, start_col:end_col]

        hidden_states = hidden_states.clone()
        update_indices = paddle.nonzero(visual_pos_masks)
        # Under SP, visual tokens are unevenly distributed across ranks. After row-slicing
        # visual_pos_masks and visual_embeds to the local sequence chunk, some ranks may
        # have zero visual tokens (local_visual_count == 0), producing visual_embeds with
        # shape [0, H]. Guard against passing an empty updates tensor to scatter_nd_add,
        # whose behavior is undefined / backend-dependent in that case.
        if visual_embeds.shape[0] > 0:
            hidden_states = paddle.scatter_nd_add(hidden_states, update_indices, visual_embeds)

        # [Supplement 3] Restore original shape [B*S, D] -> [B, S, D] if necessary
        if len(original_shape) > 2:
            hidden_states = hidden_states.reshape(original_shape)
        if _sp_transposed:
            hidden_states = hidden_states.transpose([1, 0, 2])  # [B,S/tp,H] -> [S/tp,B,H]

        return hidden_states


@dataclass
class Qwen3VLTextProvider(GPTModelProvider):
    """
    Base config for Qwen3 Models.
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    use_bias: bool = False
    add_qkv_bias: bool = True
    seq_length: int = 4096
    init_method_std: int = 0.02
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    vocab_size: int = 151936
    share_embeddings_and_output_weights: bool | None = False
    rms_norm_eps: float = 1e-6
    rotary_base: float = 1000000.0
    position_embedding_type: str = "rope"
    bias_activation_fusion: bool = True
    use_qk_norm: bool = True
    specific_layer: type = Qwen3VLTextTransformerLayer
    max_sequence_length: int = 262144
    multimodal_embedding: bool = False
    _save_to_hf: bool = False
    use_fused_linear_cross_entropy: bool = True
    high_precision_rope: bool = True
    moe_grouped_gemm: bool = True

    n_shared_experts: int = 0
    transform_rules = {
        "dtype": "params_dtype",
        "num_heads": "num_attention_heads",
        "depth": "num_hidden_layers",
        "initializer_range": "init_method_std",
        "num_experts": "n_routed_experts",
    }

    def __post_init__(self):
        super().__post_init__()
        self.mrope_section = self.rope_scaling.get("mrope_section", [24, 20, 20])


@dataclass
class Qwen3VLVisionProvider(TransformerConfig):
    """Qwen3VL Vidion Model Configuration."""

    patch_size: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    num_position_embeddings: int = 2304
    embed_dim: int = (1152,)
    hidden_size: int = 1152
    out_hidden_size: int = 4096
    in_channels: int = 3
    spatial_merge_size: int = 2
    spatial_patch_size: int = 16
    temporal_patch_size: int = 2
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    intermediate_size: int = 4304
    initializer_range: float = 0.02
    gated_linear_unit: bool = False
    activation_func: Callable = F.gelu
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = "LayerNorm"
    apply_rope_fusion: bool = True
    rms_norm_eps: float = 1e-6
    transformer_layer_spec: LayerSpec = None
    model_version: str = "qwen3_vl"
    img_h: int = 336
    img_w: int = 336
    add_class_token: bool = False
    class_token_len: int = 1
    high_precision_rope: bool = True
    # _save_to_hf: bool = False
    # use_fused_linear_cross_entropy: bool = True
    # fuse_linear: bool = True
    # transform_rules: dict = field(default_factory=lambda: {
    #     "num_heads": "num_attention_heads",
    #     "depth": "num_hidden_layers"
    # })
    transform_rules = {
        "dtype": "params_dtype",
        "num_heads": "num_attention_heads",
        "depth": "num_hidden_layers",
        "initializer_range": "init_method_std",
    }

    def provide(self) -> "Qwen3VLVisionModel":
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, LayerSpec):
            transformer_layer_spec = get_layer_spec(is_vit=True, normalization=self.normalization)
        model = Qwen3VLVisionModel(
            config=self,
            transformer_layer_spec=transformer_layer_spec,
        )

        return model


class Qwen3VLVisionTransformerBlock(TransformerBlock):
    """
    Qwen3-VL Vision Transformer Block.
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: TransformerBlockSublayersSpec | LayerSpec,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: ProcessGroupCollection = None,
        vp_stage: int = None,
    ):
        super().__init__(
            config=config,
            spec=spec,
            post_layer_norm=False,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        # print("vision_model transformer_layer ",config.num_hidden_layers)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )
        self.merger = Qwen3VLVisionPatchMerger(
            config,
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor | WrappedTensor,
        attention_mask: paddle.Tensor | None,
        context: paddle.Tensor | None = None,
        context_mask: paddle.Tensor | None = None,
        rotary_pos_emb: paddle.Tensor | None = None,
        rotary_pos_cos: paddle.Tensor | None = None,
        rotary_pos_sin: paddle.Tensor | None = None,
        attention_bias: paddle.Tensor | None = None,
        inference_context=None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        *,
        inference_params=None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
            packed_seq_params_full (PackedSeqParams, optional): Parameters for packed sequence
                processing for full attention.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Delete the obsolete reference to the initial input tensor if necessary.
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            hidden_states = self.input_tensor

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        # print("fleet vision 0 hidden_states", hidden_states.shape)

        with rng_context:
            deepstack_feature_lists = []
            for l_no, layer in enumerate(self.layers):
                packed_seq_params_now = packed_seq_params
                input_dict = {
                    "hidden_states": hidden_states,
                    "attention_mask": attention_mask,
                    "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
                    "context": context,
                    "rotary_pos_emb": rotary_pos_emb,
                    "rotary_pos_cos": rotary_pos_cos,
                    "rotary_pos_sin": rotary_pos_sin,
                    "attention_bias": attention_bias,
                    "packed_seq_params": packed_seq_params_now,
                }
                output = layer(input_dict)
                hidden_states, context = output["hidden_states"], output["context"]
                if (
                    paddle.is_grad_enabled()
                    and self.config.cpu_offloading
                    and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

                if l_no in self.deepstack_visual_indexes:
                    deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(l_no)](
                        hidden_states.squeeze(0)
                    )
                    deepstack_feature_lists.append(deepstack_feature)
                    # print(f"fleet vision {l_no} hidden_states", hidden_states.shape)

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        hidden_states = self.merger(hidden_states.squeeze(0))
        # print("vision merger output ",hidden_states.shape)
        return hidden_states, deepstack_feature_lists


class Qwen3VLVisionModel(VisionLayer):
    is_fleet = True

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: LayerSpec,
    ):
        super().__init__(config=config)
        # print("Qwen3VLVisionModel transformer_layer nums ",config.num_hidden_layers)
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.merge_hidden_size = self.embed_dim * (config.spatial_merge_size**2)

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.model_type = ModelType.encoder_or_decoder

        self.decoder = Qwen3VLVisionTransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
        )

    def _build_token_image_mapping(self, grid_thw):
        """Build token-to-image mapping, shared by rot_pos_emb and fast_pos_embed_interpolate"""
        heights = grid_thw[:, 1]
        widths = grid_thw[:, 2]
        frames = grid_thw[:, 0]

        num_tokens = frames * heights * widths  # [N]

        total_tokens = num_tokens.sum().item()  # 1 D2H
        max_hw = paddle.max(paddle.maximum(heights, widths)).item()  # 1 D2H

        # token-to-image mapping: image_id[j] = i, where cu_tokens[i] <= j < cu_tokens[i+1]
        cu_tokens = paddle.concat([paddle.zeros([1], dtype="int64"), num_tokens.cumsum(0)])
        global_idx = paddle.arange(total_tokens, dtype="int64")
        image_id = (global_idx.unsqueeze(-1) >= cu_tokens[:-1].unsqueeze(0)).astype("int64").sum(-1) - 1

        local_idx = global_idx - cu_tokens[image_id]

        # frame-local index
        token_hw = (heights * widths)[image_id]
        frame_local_idx = local_idx % token_hw

        return image_id, frame_local_idx, total_tokens, max_hw

    def rot_pos_emb(self, grid_thw, image_id=None, frame_local_idx=None, total_tokens=None, max_hw=None):
        m = self.spatial_merge_size
        widths = grid_thw[:, 2]
        merged_w = widths // m

        if image_id is None:
            image_id, frame_local_idx, total_tokens, max_hw = self._build_token_image_mapping(grid_thw)

        freq_table = self.rotary_pos_emb(max_hw)

        token_mw = merged_w[image_id]  # [total_tokens]

        # Decompose linear index to coordinates: layout [merged_h, merged_w, m, m]
        mm = m * m
        mw_mm = token_mw * mm
        block_row = frame_local_idx // mw_mm
        r1 = frame_local_idx % mw_mm
        block_col = r1 // mm
        r2 = r1 % mm
        intra_row = r2 // m
        intra_col = r2 % m

        row_idx = block_row * m + intra_row
        col_idx = block_col * m + intra_col

        pos_ids = paddle.stack([row_idx, col_idx], axis=-1)  # [total_tokens, 2]

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(start_axis=1)
        return embeddings

    def fast_pos_embed_interpolate(
        self, grid_thw, image_id=None, frame_local_idx=None, total_tokens=None, max_hw=None
    ):
        N = self.num_grid_per_side
        m = self.spatial_merge_size
        heights = grid_thw[:, 1]
        widths = grid_thw[:, 2]
        merged_w = widths // m

        if image_id is None:
            image_id, frame_local_idx, total_tokens, max_hw = self._build_token_image_mapping(grid_thw)

        token_mw = merged_w[image_id]

        # Decompose linear index to coordinates (same layout as rot_pos_emb)
        mm = m * m
        mw_mm = token_mw * mm
        block_row = frame_local_idx // mw_mm
        r1 = frame_local_idx % mw_mm
        block_col = r1 // mm
        r2 = r1 % mm
        intra_row = r2 // m
        intra_col = r2 % m

        # Pixel coordinates
        j_h = (block_row * m + intra_row).astype("float32")
        j_w = (block_col * m + intra_col).astype("float32")

        # Bilinear interpolation: h_idx = j_h * (N-1) / (h-1)
        token_h = heights[image_id].astype("float32")
        token_w = widths[image_id].astype("float32")
        h_denom = (token_h - 1).clip(min=1.0)
        w_denom = (token_w - 1).clip(min=1.0)
        h_idx = j_h * (N - 1) / h_denom
        w_idx = j_w * (N - 1) / w_denom

        h_floor = h_idx.astype("int32")
        w_floor = w_idx.astype("int32")
        h_ceil = (h_floor + 1).clip(max=N - 1)
        w_ceil = (w_floor + 1).clip(max=N - 1)

        dh = h_idx - h_floor.astype("float32")
        dw = w_idx - w_floor.astype("float32")

        base_h = h_floor * N
        base_h_ceil = h_ceil * N

        idx_tensor = paddle.stack(
            [
                (base_h + w_floor).astype("int64"),
                (base_h + w_ceil).astype("int64"),
                (base_h_ceil + w_floor).astype("int64"),
                (base_h_ceil + w_ceil).astype("int64"),
            ]
        )  # [4, total_tokens]

        weight_tensor = paddle.stack([(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw]).astype(
            self.pos_embed.weight.dtype
        )  # [4, total_tokens]

        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        # Already in (block_h, block_w, intra_h, intra_w) order, no merge_reshape needed
        return patch_pos_embeds

    def get_packed_seq_params(
        self,
        grid_thw: paddle.Tensor,
    ):
        hw = grid_thw[:, 1] * grid_thw[:, 2]
        frames = grid_thw[:, 0]

        # Build seqlens: repeat hw[i] by frames[i]
        total_seqs = frames.sum().item()  # 1 D2H
        cu_frames = paddle.concat([paddle.zeros([1], dtype="int64"), frames.cumsum(0)])
        seq_idx = paddle.arange(total_seqs, dtype="int64")
        seq_image_id = (seq_idx.unsqueeze(-1) >= cu_frames[:-1].unsqueeze(0)).astype("int64").sum(-1) - 1
        seqlens = hw[seq_image_id]

        cu_seqlens = paddle.concat(
            [
                paddle.zeros([1], dtype="int32"),
                seqlens.cumsum(0).astype("int32"),
            ]
        )
        max_seqlen = seqlens.max().item()
        total_seqlen = cu_seqlens[-1].item()

        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            total_seqlen_q=total_seqlen,
            total_seqlen_kv=total_seqlen,
            qkv_format="thd",
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        grid_thw: paddle.Tensor,
        attention_mask: paddle.Tensor | None = None,
        **kwargs
    ) -> paddle.Tensor:
        # Pathed embedding
        hidden_states = self.patch_embed(hidden_states).view(-1, self.embed_dim)

        # Share token-to-image mapping to avoid redundant computation
        image_id, frame_local_idx, total_tokens, max_hw = self._build_token_image_mapping(grid_thw)

        pos_embeds = self.fast_pos_embed_interpolate(
            grid_thw, image_id=image_id, frame_local_idx=frame_local_idx, total_tokens=total_tokens, max_hw=max_hw
        )
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape([seq_len, -1])
        hidden_states = hidden_states.unsqueeze(0)

        rotary_pos_emb = self.rot_pos_emb(
            grid_thw, image_id=image_id, frame_local_idx=frame_local_idx, total_tokens=total_tokens, max_hw=max_hw
        )
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        rotary_pos_emb = paddle.cat((rotary_pos_emb, rotary_pos_emb), axis=-1)
        # Cast freqs to float32 and compute cos/sin inside auto_cast(False) to match the
        # precision of _apply_rotary_pos_emb_bshd_fp32, which computes cos/sin on the same
        # bf16 freqs but under auto_cast(False) using a float32 kernel.
        with paddle.amp.auto_cast(False):
            _freqs_f32 = rotary_pos_emb.astype("float32")
            rotary_pos_cos = paddle.cos(_freqs_f32)
            rotary_pos_sin = paddle.sin(_freqs_f32)
        rotary_pos_emb = rotary_pos_emb[:, None, None, :]
        rotary_pos_emb = rotary_pos_emb.transpose([1, 0, 2, 3])

        packed_seq_params = self.get_packed_seq_params(grid_thw)

        # Pre-compute attn_mask_startend_row_indices once for all ViT layers
        cu_seqlens = packed_seq_params.cu_seqlens_kv
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        indices_per_segment = paddle.stack(
            [
                cu_seqlens[1:],  # col 0: lower_start = end_i
                paddle.full_like(cu_seqlens[1:], seq_len),  # col 1: lower_end   = total_seq
                paddle.zeros_like(cu_seqlens[:-1]),  # col 2: upper_start = 0
                cu_seqlens[:-1],  # col 3: upper_end   = start_i
            ],
            axis=1,
        )  # [num_segments, 4]
        attn_mask_startend_row_indices = (
            paddle.repeat_interleave(indices_per_segment, lengths, axis=0).unsqueeze(0).unsqueeze(0)
        )  # [1, 1, seq_len, 4]

        hidden_states = self.decoder(
            hidden_states,
            attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )

        return hidden_states


class Qwen3VLVisionModelFleet(Qwen3VLPretrainedModel):
    def __new__(cls, config):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = Qwen3VLVisionProvider
        model_provider = model_provider_class.from_config(config)
        vision_model = model_provider.provide()
        vision_model._gen_aoa_config = cls._gen_aoa_config
        vision_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        vision_model._get_tensor_parallel_mappings = cls._get_tensor_parallel_mappings
        vision_model.config_to_save = config

        return vision_model


@dataclass
class Qwen3VLProvider(TransformerConfig):
    text_config: Qwen3VLTextProvider | None = None
    vision_config: Qwen3VLVisionProvider | None = None

    drop_vision_class_token: bool = False
    vision_feature_layer: int = -2

    encoder_pipeline_model_parallel_size: int = 0
    encoder_tensor_model_parallel_size: int = 1

    seq_length: int = 1024

    language_model_from_pretrained: str | None = None
    vision_model_from_pretrained: str | None = None

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, tokenizer=None, vp_stage: int | None = None, loss_fn=None) -> "Qwen3VLModelDist":
        self.text_config.scatter_embedding_sequence_parallel = False
        self.text_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.text_config.tensor_parallel_output = self.tensor_parallel_output
        self.text_config.sequence_parallel = self.sequence_parallel
        self.text_config.context_parallel_size = self.context_parallel_size
        self.vision_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        # ViT always runs without sequence parallel. The vision encoder processes
        # images as packed variable-length sequences (qkv_format="thd", cu_seqlens),
        # whose token count per sample varies with image resolution and differs from
        # the text sequence length that SP splits across ranks. There is no meaningful
        # "equal-length sharding" to apply, so SP must stay disabled for the ViT
        # regardless of the global sequence_parallel setting.
        self.vision_config.sequence_parallel = False
        # self.vision_projection_config.tensor_model_parallel_size = self.tensor_model_parallel_size
        self.text_config.pipeline_model_parallel_size = self.pipeline_model_parallel_size

        if self.encoder_pipeline_model_parallel_size > 0:
            assert self.encoder_pipeline_model_parallel_size == 1, "ViT can only live on 1 pipeline stage."
            self.vision_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            # self.vision_projection_config.pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            self.text_config.encoder_pipeline_model_parallel_size = self.encoder_pipeline_model_parallel_size
            if self.encoder_tensor_model_parallel_size > 0:
                self.vision_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size
                # self.vision_projection_config.tensor_model_parallel_size = self.encoder_tensor_model_parallel_size

        config_attrs = [
            "cross_entropy_loss_fusion",
            "gradient_accumulation_fusion",
            "bias_activation_fusion",
            "bias_dropout_fusion",
            "masked_softmax_fusion",
            "attention_softmax_in_fp32",
            "apply_rope_fusion",
            "overlap_p2p_comm",
            "batch_p2p_comm",
        ]

        for config in [
            self.text_config,
            self.vision_config,
            # self.vision_projection_config,
        ]:
            for attr in config_attrs:
                setattr(config, attr, getattr(self, attr))

        self.text_config.tp_comm_overlap = self.tp_comm_overlap
        self.vision_config.tp_comm_overlap = False
        # self.vision_projection_config.tp_comm_overlap = False

        vp_stage = vp_stage or 0

        model = Qwen3VLModelDist(
            config=self,
            tokenizer=tokenizer,
            pre_process=parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
            or parallel_state.get_pipeline_model_parallel_rank() == self.encoder_pipeline_model_parallel_size,
            post_process=parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage),
            add_encoder=parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage),
            add_decoder=parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
            or parallel_state.get_pipeline_model_parallel_rank() >= self.encoder_pipeline_model_parallel_size,
            drop_vision_class_token=self.drop_vision_class_token,
            vp_stage=vp_stage,
            criterion=loss_fn,
        )

        return model

    # def __post_init__(self):
    #     if self.text_config is not None:
    #         for attr in MODEL_CONFIG_ATTR:
    #             setattr(self, attr, getattr(self.text_config, attr))
    #         self.text_config.position_embedding_type = "mrope"
    #         self.text_config.mrope_section = [24, 20, 20]
    #         self.text_config.multimodal_embedding = True

    @classmethod
    def from_config(cls, config):
        res = super().from_config(config)
        res.vision_config = Qwen3VLVisionProvider.from_config(config.vision_config)
        res.text_config = Qwen3VLTextProvider.from_config(config.text_config)
        res.vision_config.normalization = "LayerNorm"
        res.vision_config.gated_linear_unit = False
        res.text_config.multimodal_embedding = True
        res.text_config.position_embedding_type = "mrope"
        res.text_config.image_token_id = config.image_token_id
        res.text_config.video_token_id = config.video_token_id
        return res


class Qwen3VLModelDist(MCoreLLaVAModel):
    """Qwen3VL Model Base Model Class."""

    def __init__(
        self,
        config: Qwen3VLProvider,
        tokenizer=None,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        drop_vision_class_token: bool = False,
        vp_stage: int | None = None,
        model_version: str | None = None,
        criterion=False,
    ) -> None:
        super(MCoreLLaVAModel, self).__init__(config=config)

        language_transformer_config = config.text_config
        vision_transformer_config = config.vision_config
        self.model_version = vision_transformer_config.model_version if model_version is None else model_version
        self._language_max_sequence_length = language_transformer_config.max_sequence_length
        assert self.model_version is not None

        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.vp_stage = vp_stage

        self.encoder_hidden_state = None
        self.vision_model = None
        self.language_model = None
        self.image_token_index = config.image_token_id
        self.video_token_index = config.video_token_id

        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        assert not (self.context_parallel_lm > 1), f"qwenvl donnot support context parallel {self.context_parallel_lm}"
        self.share_embeddings_and_output_weights = False
        self.rope_deltas = None

        if self.add_decoder:
            self.language_model = language_transformer_config.provide(
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )
            self._language_is_pipeline_parallel = language_transformer_config.pipeline_model_parallel_size > 1

        if add_encoder:
            self.vision_model = Qwen3VLVisionModelFleet(vision_transformer_config)
            self._drop_vision_class_token = drop_vision_class_token

        self.freeze(
            freeze_language_model=config.freeze_language_model,
            freeze_vision_model=config.freeze_vision_model,
            freeze_vision_projection=config.freeze_vision_projection,
        )

        self.model_type = ModelType.encoder_or_decoder

        self._img_seq_len = get_image_sequence_length(
            img_h=vision_transformer_config.img_h,
            img_w=vision_transformer_config.img_w,
            patch_dim=vision_transformer_config.patch_size,
            add_class_token=not drop_vision_class_token,
            class_token_len=vision_transformer_config.class_token_len,
        )
        self.criterion = criterion

    def get_rope_index(
        self,
        input_ids: paddle.LongTensor | None = None,
        image_grid_thw: paddle.LongTensor | None = None,
        video_grid_thw: paddle.LongTensor | None = None,
        attention_mask: paddle.Tensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        if video_grid_thw is not None:
            video_grid_thw = paddle.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        # TODO when implemented data file.
        image_token_id = self.image_token_index
        video_token_id = self.video_token_index
        vision_start_token_id = 151652
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = paddle.ones_like(total_input_ids)
            position_ids = paddle.ones([3, input_ids.shape[0], input_ids.shape[1]], dtype=input_ids.dtype)
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = paddle.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(paddle.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = paddle.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = paddle.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = paddle.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(paddle.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(paddle.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = paddle.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = paddle.to_tensor(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = paddle.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
                mrope_position_deltas = paddle.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def get_video_features(
        self,
        pixel_values_videos: paddle.FloatTensor,
        video_grid_thw: paddle.LongTensor | None = None,
    ):
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: paddle.FloatTensor, image_grid_thw: paddle.LongTensor | None = None):
        image_embeds, deepstack_image_embeds = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        # print("vision_model output ",image_embeds.shape)
        # print(f"image_grid_thw {image_grid_thw.prod(-1)} spatial_merge_size {self.vision_model.spatial_merge_size ** 2}")
        split_sizes = (image_grid_thw.prod(-1) // self.vision_model.spatial_merge_size**2).tolist()
        image_embeds = paddle.split(image_embeds, split_sizes)
        # print(f"after split {split_sizes} image_embeds {image_embeds}")
        return image_embeds, deepstack_image_embeds

    def forward(
        self,
        input_ids: paddle.LongTensor = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.LongTensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        inference_params=None,
        pixel_values: paddle.Tensor | None = None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        runtime_gather_output: bool | None = None,
        cache_position: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        **kwargs,
    ) -> paddle.Tensor:
        assert loss_mask is None, "loss_mask is not supported yet"
        image_embeds, video_embeds, deepstack_image_embeds, deepstack_video_embeds = (None for _ in range(4))
        if self.add_encoder and pixel_values is not None:
            # Handle list[paddle.Tensor] input (from RL training pipeline)
            if isinstance(pixel_values, list):
                # Filter out None and concatenate tensors
                tensor_list = [elem for elem in pixel_values if elem is not None]
                if tensor_list:
                    pixel_values = paddle.concat(tensor_list, axis=0)
                else:
                    pixel_values = None
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.vision_model.parameters()[0].dtype)
                # Handle list[paddle.Tensor] for image_grid_thw
                if image_grid_thw is not None:
                    if isinstance(image_grid_thw, list):
                        tensor_list = [elem for elem in image_grid_thw if elem is not None]
                        if tensor_list:
                            image_grid_thw = paddle.concat(tensor_list, axis=0)
                        else:
                            image_grid_thw = None
                if self.config.freeze_vision_model:
                    with paddle.no_grad():
                        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                else:
                    image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = paddle.cat(image_embeds, dim=0)

        if self.add_encoder and pixel_values_videos is not None:
            # Handle list[paddle.Tensor] input (from RL training pipeline)
            if isinstance(pixel_values_videos, list):
                # Filter out None and concatenate tensors
                tensor_list = [elem for elem in pixel_values_videos if elem is not None]
                if tensor_list:
                    pixel_values_videos = paddle.concat(tensor_list, axis=0)
                else:
                    pixel_values_videos = None
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.to(self.vision_model.parameters()[0].dtype)
                # Handle list[paddle.Tensor] for video_grid_thw
                if video_grid_thw is not None:
                    if isinstance(video_grid_thw, list):
                        tensor_list = [elem for elem in video_grid_thw if elem is not None]
                        if tensor_list:
                            video_grid_thw = paddle.concat(tensor_list, axis=0)
                        else:
                            video_grid_thw = None
                if self.config.freeze_vision_model:
                    with paddle.no_grad():
                        video_embeds, deepstack_video_embeds = self.get_video_features(
                            pixel_values_videos, video_grid_thw
                        )
                else:
                    video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = paddle.cat(video_embeds, axis=0)

        if position_ids is None:
            if self.rope_deltas is None or cache_position is None or cache_position[0] == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                position_ids = paddle.arange(seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = cache_position[0] + self.rope_deltas
                else:
                    delta = paddle.zeros((batch_size, seq_length))
                delta = delta.repeat_interleave(batch_size // delta.shape[0], axis=1)
                position_ids = position_ids + delta
        else:
            # Handle position_ids with mrope format [batch_size, seq_len, 3] -> [3, batch_size, seq_len]
            if position_ids.ndim == 3 and position_ids.shape[-1] == 3:
                position_ids = position_ids.transpose([2, 0, 1])
            elif position_ids.shape == input_ids.shape:
                position_ids = position_ids.expand(3, position_ids.shape[0], -1)

        input_dict = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "decoder_input": None,
            "image_embeds": image_embeds,
            "video_embeds": video_embeds,
            "labels": labels,
            "deepstack_image_embeds": deepstack_image_embeds,
            "deepstack_video_embeds": deepstack_video_embeds,
            "runtime_gather_output": runtime_gather_output,
        }
        output = self.language_model(input_dict)
        # print("qwenvl criterion ",self.criterion)
        if labels is None:
            return output
        elif self.criterion is not None:
            # print("qwenvl output loss  ",self.criterion(output, labels))
            return self.criterion(output, labels)
        else:
            return output

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for llava"

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    # def get_input_embeddings(self):
    #     return self.language_model.get_input_embeddings()


class Qwen3VLPretrainedModelFleet(PretrainedModel):
    config_class = Qwen3VLConfig
    base_model_prefix = "model"
    input_modalities = ["image", "video", "text"]
    _no_split_modules = ["Qwen3VLTextTransformerLayer", "Qwen3VLVisionTransformerBlock"]
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv",
        "gate_proj",
        "up_proj",
        "down_proj",
        "proj",
        "linear_fc\d+",
        "up_gate_proj",
        "qkv_proj",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Qwen3VLConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        # visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        visual_target = "model.vision_model"
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        # language model
        aoa_config = {
            "aoa_statements": [
                f"model.language_model.embed_tokens.weight -> {llm_prefix}embedding.embed_tokens.weight",
                f"model.language_model.norm.weight -> {llm_prefix}norm.weight",
                f"model.language_model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.language_model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                f"model.language_model.layers.$LAYER_ID.input_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.language_model.layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.language_model.layers.$LAYER_ID.self_attn.q_norm.weight -> {llm_prefix}layers.$LAYER_ID.self_attn.q_norm.weight",
                f"model.language_model.layers.$LAYER_ID.self_attn.k_norm.weight -> {llm_prefix}layers.$LAYER_ID.self_attn.k_norm.weight",
            ]
        }

        # visual model
        aoa_config["aoa_statements"] += [
            stmt
            for layer_id in range(config.vision_config.depth)
            for stmt in (
                f"model.visual.blocks.{layer_id}.attn.qkv.weight -> model.visual.blocks.{layer_id}.attn.q.weight, model.visual.blocks.{layer_id}.attn.k.weight,model.visual.blocks.{layer_id}.attn.v.weight,axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.weight^T, model.visual.blocks.{layer_id}.attn.k.weight^T, model.visual.blocks.{layer_id}.attn.v.weight^T -> {visual_prefix}decoder.layers.{layer_id}.self_attn.qkv_proj.weight,fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads}",
                f"model.visual.blocks.{layer_id}.attn.qkv.bias -> model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias,axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias -> {visual_prefix}decoder.layers.{layer_id}.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads},axis=0",
            )
        ]
        aoa_config["aoa_statements"] += (
            [
                f"model.visual.blocks.$LAYER_ID.attn.proj.weight^T -> {visual_prefix}decoder.layers.$LAYER_ID.self_attn.o_proj.weight",
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.attn.proj.bias -> {visual_prefix}decoder.layers.$LAYER_ID.self_attn.o_proj.bias"
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.mlp.{x}.weight^T -> {visual_prefix}decoder.layers.$LAYER_ID.mlp.{y}.weight"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.mlp.{x}.bias -> {visual_prefix}decoder.layers.$LAYER_ID.mlp.{y}.bias"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
        )
        aoa_config["aoa_statements"] += [
            f"model.visual.patch_embed.proj.weight -> {visual_prefix}patch_embed.proj.weight",
            f"model.visual.patch_embed.proj.bias -> {visual_prefix}patch_embed.proj.bias",
            f"model.visual.pos_embed.weight -> {visual_prefix}pos_embed.weight",
            f"model.visual.merger.norm.weight -> {visual_prefix}decoder.merger.norm.weight",
            f"model.visual.merger.norm.bias -> {visual_prefix}decoder.merger.norm.bias",
            f"model.visual.blocks.$LAYER_ID.norm1.weight -> {visual_prefix}decoder.layers.$LAYER_ID.input_layernorm.weight",
            f"model.visual.blocks.$LAYER_ID.norm1.bias -> {visual_prefix}decoder.layers.$LAYER_ID.input_layernorm.bias",
            f"model.visual.blocks.$LAYER_ID.norm2.weight -> {visual_prefix}decoder.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"model.visual.blocks.$LAYER_ID.norm2.bias -> {visual_prefix}decoder.layers.$LAYER_ID.post_attention_layernorm.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"model.visual.merger.linear_fc1.weight^T -> {visual_prefix}decoder.merger.linear_fc1.weight",
            f"model.visual.merger.linear_fc1.bias -> {visual_prefix}decoder.merger.linear_fc1.bias",
            f"model.visual.merger.linear_fc2.weight^T -> {visual_prefix}decoder.merger.linear_fc2.weight",
            f"model.visual.merger.linear_fc2.bias -> {visual_prefix}decoder.merger.linear_fc2.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.weight^T -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc1.weight",
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.bias -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc1.bias",
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.weight^T -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc2.weight",
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.bias -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc2.bias",
            f"model.visual.deepstack_merger_list.$LAYER_ID.norm.weight -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.norm.weight",
            f"model.visual.deepstack_merger_list.$LAYER_ID.norm.bias -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.norm.bias",
        ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{layer_id}.self_attn.q_proj.weight^T, model.language_model.layers.{layer_id}.self_attn.k_proj.weight^T, model.language_model.layers.{layer_id}.self_attn.v_proj.weight^T -> {llm_prefix}layers.{layer_id}.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}"
            for layer_id in range(config.text_config.num_hidden_layers)
        ]

        # FFN
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{layer_id}.mlp.gate_proj.weight^T, model.language_model.layers.{layer_id}.mlp.up_proj.weight^T -> {llm_prefix}layers.{layer_id}.mlp.up_gate_proj.weight, fused_ffn"
            for layer_id in range(config.text_config.num_hidden_layers)
        ]

        # Qwen3_VLModel without lm_head
        if cls._tied_weights_keys:
            aoa_config["aoa_statements"] += [
                f"{'model.language_model.embed_tokens.weight' if config.tie_word_embeddings else 'lm_head.weight'} -> {llm_prefix}lm_head.weight",
            ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Qwen3VLConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        # visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        visual_target = "model.vision_model"
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        # language model
        aoa_config = {
            "aoa_statements": [
                f"{llm_prefix}embedding.embed_tokens.weight -> model.language_model.embed_tokens.weight",
                f"{llm_prefix}norm.weight -> model.language_model.norm.weight",
                f"{llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.language_model.layers.$LAYER_ID.mlp.down_proj.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.language_model.layers.$LAYER_ID.self_attn.o_proj.weight",
                f"{llm_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.language_model.layers.$LAYER_ID.input_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.language_model.layers.$LAYER_ID.post_attention_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.q_norm.weight -> model.language_model.layers.$LAYER_ID.self_attn.q_norm.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.k_norm.weight -> model.language_model.layers.$LAYER_ID.self_attn.k_norm.weight",
            ]
        }

        # visual model
        aoa_config["aoa_statements"] += [
            stmt
            for layer_id in range(config.vision_config.depth)
            for stmt in (
                f"{visual_prefix}decoder.layers.{layer_id}.self_attn.qkv_proj.weight -> model.visual.blocks.{layer_id}.attn.q.weight, model.visual.blocks.{layer_id}.attn.k.weight, model.visual.blocks.{layer_id}.attn.v.weight, fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads}",
                f"model.visual.blocks.{layer_id}.attn.q.weight^T, model.visual.blocks.{layer_id}.attn.k.weight^T, model.visual.blocks.{layer_id}.attn.v.weight^T -> model.visual.blocks.{layer_id}.attn.qkv.weight, axis=0",
                f"{visual_prefix}decoder.layers.{layer_id}.self_attn.qkv_proj.bias -> model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias, fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads},axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias -> model.visual.blocks.{layer_id}.attn.qkv.bias, axis=0",
            )
        ]
        aoa_config["aoa_statements"] += (
            [
                f"{visual_prefix}decoder.layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.visual.blocks.$LAYER_ID.attn.proj.weight"
            ]
            + [
                f"{visual_prefix}decoder.layers.$LAYER_ID.self_attn.o_proj.bias -> model.visual.blocks.$LAYER_ID.attn.proj.bias"
            ]
            + [
                f"{visual_prefix}decoder.layers.$LAYER_ID.mlp.{y}.weight^T -> model.visual.blocks.$LAYER_ID.mlp.{x}.weight"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
            + [
                f"{visual_prefix}decoder.layers.$LAYER_ID.mlp.{y}.bias -> model.visual.blocks.$LAYER_ID.mlp.{x}.bias"
                for x, y in (("linear_fc1", "up_gate_proj"), ("linear_fc2", "down_proj"))
            ]
        )
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}patch_embed.proj.weight -> model.visual.patch_embed.proj.weight",
            f"{visual_prefix}patch_embed.proj.bias -> model.visual.patch_embed.proj.bias",
            f"{visual_prefix}pos_embed.weight -> model.visual.pos_embed.weight",
            f"{visual_prefix}decoder.merger.norm.weight -> model.visual.merger.norm.weight",
            f"{visual_prefix}decoder.merger.norm.bias -> model.visual.merger.norm.bias",
            f"{visual_prefix}decoder.layers.$LAYER_ID.input_layernorm.weight -> model.visual.blocks.$LAYER_ID.norm1.weight",
            f"{visual_prefix}decoder.layers.$LAYER_ID.input_layernorm.bias -> model.visual.blocks.$LAYER_ID.norm1.bias",
            f"{visual_prefix}decoder.layers.$LAYER_ID.post_attention_layernorm.weight -> model.visual.blocks.$LAYER_ID.norm2.weight",
            f"{visual_prefix}decoder.layers.$LAYER_ID.post_attention_layernorm.bias -> model.visual.blocks.$LAYER_ID.norm2.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}decoder.merger.linear_fc1.weight^T -> model.visual.merger.linear_fc1.weight",
            f"{visual_prefix}decoder.merger.linear_fc1.bias -> model.visual.merger.linear_fc1.bias",
            f"{visual_prefix}decoder.merger.linear_fc2.weight^T -> model.visual.merger.linear_fc2.weight",
            f"{visual_prefix}decoder.merger.linear_fc2.bias -> model.visual.merger.linear_fc2.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc1.weight^T -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.weight",
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc1.bias -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.bias",
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc2.weight^T -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.weight",
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc2.bias -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.bias",
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.norm.weight -> model.visual.deepstack_merger_list.$LAYER_ID.norm.weight",
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.norm.bias -> model.visual.deepstack_merger_list.$LAYER_ID.norm.bias",
        ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{layer_id}.self_attn.qkv_proj.weight  -> model.language_model.layers.{layer_id}.self_attn.q_proj.weight, model.language_model.layers.{layer_id}.self_attn.k_proj.weight, model.language_model.layers.{layer_id}.self_attn.v_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups = {config.text_config.num_key_value_heads}"
            for layer_id in range(config.text_config.num_hidden_layers)
        ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.language_model.layers.{layer_id}.self_attn.{x}_proj.weight"
            for layer_id in range(config.text_config.num_hidden_layers)
            for x in ("q", "k", "v")
        ]

        # FFN
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{layer_id}.mlp.up_gate_proj.weight -> model.language_model.layers.{layer_id}.mlp.gate_proj.weight, model.language_model.layers.{layer_id}.mlp.up_proj.weight, fused_ffn"
            for layer_id in range(config.text_config.num_hidden_layers)
        ]
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{layer_id}.mlp.{x}_proj.weight^T -> model.language_model.layers.{layer_id}.mlp.{x}_proj.weight"
            for layer_id in range(config.text_config.num_hidden_layers)
            for x in ("gate", "up")
        ]

        # Qwen3VLModel without lm_head
        if cls._tied_weights_keys:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}lm_head.weight -> {'_' if config.tie_word_embeddings else 'lm_head.weight'}",
            ]

        return aoa_config


class Qwen3VLModel(Qwen3VLPretrainedModelFleet):
    config_class = Qwen3VLConfig

    def __new__(cls, config, have_criterion=True):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        criterion = None
        if have_criterion:
            criterion = CriterionLayer(config.text_config)
        model_provider_class = Qwen3VLProvider
        model_provider = model_provider_class.from_config(config)
        qwen3vl_model = Qwen3VLModelDist(model_provider, model_version=config.model_type, criterion=criterion)
        qwen3vl_model._gen_aoa_config = cls._gen_aoa_config
        qwen3vl_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        qwen3vl_model._get_tensor_parallel_mappings = cls._get_tensor_parallel_mappings
        qwen3vl_model.config_to_save = config

        return qwen3vl_model


class Qwen3VLForConditionalGeneration(Qwen3VLPretrainedModelFleet):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    config_class = Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        # model_provider = Qwen3VLProvider.from_config(config)
        self.model = Qwen3VLModel(
            config, have_criterion=False
        )  # Qwen3VLModel(model_provider, model_version=config.model_type)
        self.criterion = CriterionLayer(config.text_config)
        # self.tie_weights()

    def state_dict(self, *args, **kwargs):
        # Override state_dict method to handle language_model's custom state_dict
        state_dict = super().state_dict(*args, **kwargs)
        # Remove existing language_model keys to avoid duplicates
        delete_key = []
        for key in state_dict.keys():
            if key.startswith("model.language_model."):
                delete_key.append(key)
        for key in delete_key:
            state_dict.pop(key)
        if self.model.language_model is not None:
            # Get language_model's state_dict
            language_state_dict = self.model.language_model.state_dict(*args, **kwargs)

            # Merge language_model parameters into main state_dict
            for key, value in language_state_dict.items():
                state_dict[key] = value
        return state_dict

    # def get_input_embeddings(self):
    #     return self.model.get_input_embeddings()
    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
        cache_position: Optional[paddle.Tensor] = None,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`paddle.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`paddle.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`paddle.Tensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.

        Example:

        ```python
        >>> from paddleformers.transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        >>> model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pd"
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )

        logits = outputs

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            rope_deltas=None,
        )


class Qwen3VLForCausalLMPipe(Qwen3VLPretrainedModelFleet, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config, have_criterion=True):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        criterion = None
        if have_criterion:
            criterion = CriterionLayer(config.text_config)
        model_provider_class = Qwen3VLProvider
        model_provider = model_provider_class.from_config(config)
        qwen3vl_model = Qwen3VLModelDist(model_provider, model_version=config.model_type, criterion=criterion)
        qwen3vl_model._gen_aoa_config = cls._gen_aoa_config
        qwen3vl_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        qwen3vl_model._get_tensor_parallel_mappings = cls._get_tensor_parallel_mappings
        qwen3vl_model.config_to_save = config

        return qwen3vl_model


class Qwen3VLModelPipe(Qwen3VLPretrainedModelFleet, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config, have_criterion=True):
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)
        criterion = None
        if have_criterion:
            criterion = CriterionLayer(config.text_config)
        model_provider_class = Qwen3VLProvider
        model_provider = model_provider_class.from_config(config)
        qwen3vl_model = Qwen3VLModelDist(model_provider, model_version=config.model_type, criterion=criterion)
        qwen3vl_model._gen_aoa_config = cls._gen_aoa_config
        qwen3vl_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        qwen3vl_model._get_tensor_parallel_mappings = cls._get_tensor_parallel_mappings
        qwen3vl_model.get_hardware_flops = types.MethodType(cls.get_hardware_flops, qwen3vl_model)
        qwen3vl_model.config_to_save = config

        return qwen3vl_model


__all__ = [
    "Qwen3VLModel",
    "Qwen3VLForCausalLMPipe",
    "Qwen3VLModelPipe",
    "Qwen3VLForConditionalGeneration",
]
