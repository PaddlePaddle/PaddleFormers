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

"""PaliGemma2 model implementation in PaddlePaddle.

This module implements the PaliGemma2 multimodal model, combining a SigLIP
vision encoder with a Gemma2 text decoder. The implementation follows the
HuggingFace transformers architecture and is adapted for PaddlePaddle.

Reference: https://huggingface.co/google/paligemma2-3b-pt-448
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleformers.transformers.model_outputs import ModelOutput
from paddleformers.transformers.model_utils import PretrainedModel

from .configuration import Gemma2TextConfig, PaliGemma2Config, SiglipVisionConfig


@dataclass
class PaliGemma2ModelOutput(ModelOutput):
    """Output of the PaliGemma2 model."""

    loss: Optional[paddle.Tensor] = None
    logits: Optional[paddle.Tensor] = None
    past_key_values: Optional[Tuple] = None
    image_hidden_states: Optional[paddle.Tensor] = None


class SiglipVisionEmbeddings(nn.Layer):
    """Embeddings for the SigLIP vision encoder."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias_attr=True,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        pixel_values = pixel_values.cast(target_dtype)

        patch_embeds = self.patch_embedding(pixel_values)  # [B, C, H, W]
        patch_embeds = patch_embeds.flatten(2).transpose((0, 2, 1))  # [B, N, C]

        position_ids = paddle.arange(self.num_positions, dtype="int64").unsqueeze(0)
        embeddings = patch_embeds + self.position_embedding(position_ids)
        return embeddings


class SiglipVisionAttention(nn.Layer):
    """Multi-head attention for SigLIP vision encoder."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias_attr=True)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        bsz, tgt_len, _ = hidden_states.shape

        query_states = (
            self.q_proj(hidden_states).reshape([bsz, tgt_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        )
        key_states = (
            self.k_proj(hidden_states).reshape([bsz, tgt_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        )
        value_states = (
            self.v_proj(hidden_states).reshape([bsz, tgt_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        )

        attn_weights = paddle.matmul(query_states, key_states, transpose_y=True) * self.scale
        attn_weights = F.softmax(attn_weights.cast("float32"), axis=-1).cast(query_states.dtype)

        attn_output = paddle.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([bsz, tgt_len, self.embed_dim])

        return self.out_proj(attn_output)


class SiglipMLP(nn.Layer):
    """MLP for SigLIP vision encoder."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=True)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate=True)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Layer):
    """A single layer of the SigLIP vision encoder."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.self_attn = SiglipVisionAttention(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipVisionTransformer(nn.Layer):
    """SigLIP vision transformer encoder."""

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = nn.LayerList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.embeddings(pixel_values)
        for layer in self.encoder:
            hidden_states = layer(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states


class PaliGemma2MultiModalProjector(nn.Layer):
    """Projects vision features to the text decoder's hidden size."""

    def __init__(self, config: PaliGemma2Config):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias_attr=True,
        )

    def forward(self, image_features: paddle.Tensor) -> paddle.Tensor:
        return self.linear(image_features)


class Gemma2RMSNorm(nn.Layer):
    """RMS normalization for Gemma2."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = paddle.create_parameter(
            shape=[dim],
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.0),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        input_dtype = x.dtype
        x = x.cast("float32")
        output = x * paddle.rsqrt(paddle.mean(x * x, axis=-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.cast("float32"))
        return output.cast(input_dtype)


class Gemma2RotaryEmbedding(nn.Layer):
    """Rotary position embedding for Gemma2."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (paddle.arange(0, dim, 2).cast("float32") / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, position_ids: paddle.Tensor, dtype=None) -> paddle.Tensor:
        # position_ids: [bsz, seq_len]
        # inv_freq: [head_dim // 2]
        # freqs: [bsz, seq_len, head_dim // 2]
        freqs = position_ids.cast("float32").unsqueeze(-1) * self.inv_freq  # [bsz, seq_len, head_dim//2]
        emb = paddle.concat([freqs, freqs], axis=-1)  # [bsz, seq_len, head_dim]
        cos = emb.cos()
        sin = emb.sin()
        if dtype is not None:
            cos = cos.cast(dtype)
            sin = sin.cast(dtype)
        return cos, sin


def rotate_half(x: paddle.Tensor) -> paddle.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [bsz, num_heads, seq_len, head_dim]
    # cos, sin: [bsz, seq_len, head_dim] -> [bsz, 1, seq_len, head_dim]
    cos = cos.cast(q.dtype).unsqueeze(1)
    sin = sin.cast(q.dtype).unsqueeze(1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class Gemma2Attention(nn.Layer):
    """Multi-head attention for Gemma2 with sliding window support."""

    def __init__(self, config: Gemma2TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.config.query_pre_attn_scalar**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias_attr=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=False)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        bsz, q_len, _ = hidden_states.shape

        hidden_states = hidden_states.cast(self.q_proj.weight.dtype)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        # Repeat KV for GQA
        key_states = paddle.repeat_interleave(key_states, self.num_key_value_groups, axis=1)
        value_states = paddle.repeat_interleave(value_states, self.num_key_value_groups, axis=1)

        # Apply rotary embedding
        cos, sin = self._get_rotary_emb(q_len, position_ids, query_states.dtype)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Match HuggingFace Gemma2: softcap raw attention scores before masking.
        attn_weights = paddle.matmul(query_states, key_states, transpose_y=True) * self.scaling
        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = paddle.tanh(attn_weights) * self.config.attn_logit_softcapping
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights.cast("float32"), axis=-1).cast(query_states.dtype)
        attn_output = paddle.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([bsz, q_len, self.num_heads * self.head_dim])
        attn_output = attn_output.cast(self.o_proj.weight.dtype)
        return self.o_proj(attn_output)

    def _get_rotary_emb(self, seq_len: int, position_ids: paddle.Tensor, dtype):
        if not hasattr(self, "_rotary_emb"):
            self._rotary_emb = Gemma2RotaryEmbedding(self.head_dim, self.config.rope_theta)
        cos, sin = self._rotary_emb(seq_len, position_ids, dtype)
        return cos, sin


class Gemma2MLP(nn.Layer):
    """MLP for Gemma2 with GELU activation."""

    def __init__(self, config: Gemma2TextConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=False)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        gate_states = self.gate_proj(hidden_states)
        gate_states = F.gelu(gate_states, approximate=True)
        return self.down_proj(gate_states * self.up_proj(hidden_states))


class Gemma2DecoderLayer(nn.Layer):
    """A single decoder layer for Gemma2."""

    def __init__(self, config: Gemma2TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = Gemma2Attention(config, layer_idx)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        position_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Gemma2Model(nn.Layer):
    """Gemma2 text decoder model."""

    def __init__(self, config: Gemma2TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.LayerList([Gemma2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: paddle.Tensor,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        bsz, seq_len = hidden_states.shape[:2]
        if position_ids is None:
            position_ids = paddle.arange(seq_len).unsqueeze(0).expand([bsz, seq_len])

        causal_mask = (
            paddle.triu(
                paddle.full([seq_len, seq_len], float("-inf"), dtype="float32"),
                diagonal=1,
            )
            .unsqueeze([0, 1])
            .expand([bsz, 1, seq_len, seq_len])
        )
        if attention_mask is not None:
            padding_mask = paddle.where(
                attention_mask.unsqueeze([1, 2]).astype("bool"),
                paddle.zeros([bsz, 1, 1, seq_len], dtype="float32"),
                paddle.full([bsz, 1, 1, seq_len], float("-inf"), dtype="float32"),
            )
            causal_mask = causal_mask + padding_mask

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, causal_mask)

        return self.norm(hidden_states)


class PaliGemma2PreTrainedModel(PretrainedModel):
    """Base class for PaliGemma2 models."""

    config_class = PaliGemma2Config
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"position_ids"]


class PaliGemma2LMHead(nn.Layer):
    def __init__(self, config: Gemma2TextConfig):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[config.vocab_size, config.hidden_size],
            dtype=paddle.get_default_dtype(),
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.matmul(hidden_states, self.weight, transpose_y=True)


class PaliGemma2ForConditionalGeneration(PaliGemma2PreTrainedModel):
    """PaliGemma2 model for conditional generation (image + text -> text).

    This model combines a SigLIP vision encoder with a Gemma2 text decoder
    for multimodal tasks such as image captioning and visual question answering.

    Example:
        ```python
        >>> from paddleformers.transformers.paligemma2 import (
        ...     PaliGemma2ForConditionalGeneration,
        ...     PaliGemma2Config,
        ... )
        >>> config = PaliGemma2Config()
        >>> model = PaliGemma2ForConditionalGeneration(config)
        ```
    """

    _tied_weights_keys = {"lm_head.weight": "language_model.embed_tokens.weight"}

    def __init__(self, config: PaliGemma2Config):
        super().__init__(config)
        self.config = config
        self.vision_tower = SiglipVisionTransformer(config.vision_config)
        self.multi_modal_projector = PaliGemma2MultiModalProjector(config)
        self.language_model = Gemma2Model(config.text_config)
        self.lm_head = PaliGemma2LMHead(config.text_config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: paddle.Tensor,
        pixel_values: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
    ) -> PaliGemma2ModelOutput:
        """Forward pass of the model.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len].
            pixel_values: Image tensor of shape [batch_size, 3, H, W].
            attention_mask: Attention mask of shape [batch_size, seq_len].
            labels: Label token IDs for computing loss.

        Returns:
            PaliGemma2ModelOutput with loss and logits.
        """
        bsz, seq_len = input_ids.shape

        # The image placeholder token is outside the text vocabulary.
        llm_input_ids = paddle.where(
            input_ids == self.config.image_token_index,
            paddle.zeros_like(input_ids),
            input_ids,
        )
        inputs_embeds = self.language_model.embed_tokens(llm_input_ids)

        # Process image if provided
        image_hidden_states = None
        if pixel_values is not None:
            image_features = self.vision_tower(pixel_values)
            image_features = self.multi_modal_projector(image_features)
            image_features = image_features / (self.config.text_config.hidden_size**0.5)

            # Replace image token placeholders with image features
            image_token_mask = input_ids == self.config.image_token_index
            num_image_tokens = image_features.shape[1]

            if int(image_token_mask.astype("int64").sum().item()) != bsz * num_image_tokens:
                raise ValueError("Image features and image tokens do not match")

            for batch_idx in range(bsz):
                mask = image_token_mask[batch_idx]
                positions = paddle.nonzero(mask).flatten()
                if len(positions) > 0:
                    inputs_embeds[batch_idx, positions[:num_image_tokens]] = image_features[batch_idx]

            image_hidden_states = image_features

        position_ids = paddle.arange(1, seq_len + 1).unsqueeze(0).expand([bsz, seq_len])

        # Create causal additive mask without multiplying zero by -inf.
        causal_mask = (
            paddle.triu(
                paddle.full([seq_len, seq_len], float("-inf"), dtype="float32"),
                diagonal=1,
            )
            .unsqueeze([0, 1])
            .expand([bsz, 1, seq_len, seq_len])
        )
        if token_type_ids is not None:
            image_tokens = token_type_ids == 0
            bidirectional_image_mask = image_tokens.unsqueeze(2) & image_tokens.unsqueeze(1)
            causal_mask = paddle.where(
                bidirectional_image_mask.unsqueeze(1),
                paddle.zeros_like(causal_mask),
                causal_mask,
            )
        if attention_mask is not None:
            padding_mask = paddle.where(
                attention_mask.unsqueeze([1, 2]).astype("bool"),
                paddle.zeros([bsz, 1, 1, seq_len], dtype="float32"),
                paddle.full([bsz, 1, 1, seq_len], float("-inf"), dtype="float32"),
            )
            extended_mask = causal_mask + padding_mask
        else:
            extended_mask = causal_mask

        hidden_states = inputs_embeds * paddle.to_tensor(
            self.config.text_config.hidden_size**0.5,
            dtype=inputs_embeds.dtype,
        )
        for layer in self.language_model.layers:
            hidden_states = layer(hidden_states, position_ids, extended_mask)

        hidden_states = self.language_model.norm(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)
        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = paddle.tanh(logits) * self.config.text_config.final_logit_softcapping

        loss = None
        if labels is not None:
            flat_labels = labels.reshape([-1])
            loss_mask = (flat_labels != -100).astype(logits.dtype)
            safe_labels = paddle.where(flat_labels == -100, paddle.zeros_like(flat_labels), flat_labels)
            token_loss = F.cross_entropy(logits.reshape([-1, logits.shape[-1]]), safe_labels, reduction="none")
            loss = (token_loss * loss_mask).sum() / loss_mask.sum()

        return PaliGemma2ModelOutput(
            loss=loss,
            logits=logits,
            image_hidden_states=image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        token_type_ids=None,
        **kwargs,
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
        }

    @classmethod
    def _gen_aoa_config(cls, config):
        """AOA checkpoint conversion config.

        Linear weights need transpose (PyTorch [out,in] -> Paddle [in,out]).
        Embedding, LM head, and norm weights do not need transpose.
        """
        aoa_statements = [
            "language_model.embed_tokens.weight -> language_model.embed_tokens.weight",
            "language_model.norm.weight -> language_model.norm.weight",
            "lm_head.weight -> lm_head.weight",
            # Vision tower
            "vision_tower.embeddings.patch_embedding.weight -> vision_tower.embeddings.patch_embedding.weight",
            "vision_tower.embeddings.position_embedding.weight -> vision_tower.embeddings.position_embedding.weight",
            "vision_tower.post_layernorm.weight -> vision_tower.post_layernorm.weight",
            # Multi-modal projector
            "multi_modal_projector.linear.weight^T -> multi_modal_projector.linear.weight",
            "multi_modal_projector.linear.bias -> multi_modal_projector.linear.bias",
            # Per-layer weights
            "language_model.layers.$LAYER_ID.input_layernorm.weight -> language_model.layers.$LAYER_ID.input_layernorm.weight",
            "language_model.layers.$LAYER_ID.post_attention_layernorm.weight -> language_model.layers.$LAYER_ID.post_attention_layernorm.weight",
            "language_model.layers.$LAYER_ID.pre_feedforward_layernorm.weight -> language_model.layers.$LAYER_ID.pre_feedforward_layernorm.weight",
            "language_model.layers.$LAYER_ID.post_feedforward_layernorm.weight -> language_model.layers.$LAYER_ID.post_feedforward_layernorm.weight",
            "language_model.layers.$LAYER_ID.self_attn.q_proj.weight^T -> language_model.layers.$LAYER_ID.self_attn.q_proj.weight",
            "language_model.layers.$LAYER_ID.self_attn.k_proj.weight^T -> language_model.layers.$LAYER_ID.self_attn.k_proj.weight",
            "language_model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> language_model.layers.$LAYER_ID.self_attn.v_proj.weight",
            "language_model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> language_model.layers.$LAYER_ID.self_attn.o_proj.weight",
            "language_model.layers.$LAYER_ID.mlp.gate_proj.weight^T -> language_model.layers.$LAYER_ID.mlp.gate_proj.weight",
            "language_model.layers.$LAYER_ID.mlp.up_proj.weight^T -> language_model.layers.$LAYER_ID.mlp.up_proj.weight",
            "language_model.layers.$LAYER_ID.mlp.down_proj.weight^T -> language_model.layers.$LAYER_ID.mlp.down_proj.weight",
            # Vision encoder layers
            "vision_tower.encoder.$LAYER_ID.layer_norm1.weight -> vision_tower.encoder.$LAYER_ID.layer_norm1.weight",
            "vision_tower.encoder.$LAYER_ID.layer_norm2.weight -> vision_tower.encoder.$LAYER_ID.layer_norm2.weight",
            "vision_tower.encoder.$LAYER_ID.self_attn.q_proj.weight^T -> vision_tower.encoder.$LAYER_ID.self_attn.q_proj.weight",
            "vision_tower.encoder.$LAYER_ID.self_attn.k_proj.weight^T -> vision_tower.encoder.$LAYER_ID.self_attn.k_proj.weight",
            "vision_tower.encoder.$LAYER_ID.self_attn.v_proj.weight^T -> vision_tower.encoder.$LAYER_ID.self_attn.v_proj.weight",
            "vision_tower.encoder.$LAYER_ID.self_attn.out_proj.weight^T -> vision_tower.encoder.$LAYER_ID.self_attn.out_proj.weight",
            "vision_tower.encoder.$LAYER_ID.mlp.fc1.weight^T -> vision_tower.encoder.$LAYER_ID.mlp.fc1.weight",
            "vision_tower.encoder.$LAYER_ID.mlp.fc2.weight^T -> vision_tower.encoder.$LAYER_ID.mlp.fc2.weight",
        ]
        return {"aoa_statements": aoa_statements}


class PaliGemma2ForCausalLM(PaliGemma2PreTrainedModel):
    """PaliGemma2 model for causal language modeling (text-only)."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: PaliGemma2Config):
        super().__init__(config)
        self.config = config
        self.model = Gemma2Model(config.text_config)
        self.lm_head = PaliGemma2LMHead(config.text_config)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
    ) -> PaliGemma2ModelOutput:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = paddle.tanh(logits) * self.config.text_config.final_logit_softcapping

        loss = None
        if labels is not None:
            flat_labels = labels.reshape([-1])
            loss_mask = (flat_labels != -100).astype(logits.dtype)
            safe_labels = paddle.where(flat_labels == -100, paddle.zeros_like(flat_labels), flat_labels)
            token_loss = F.cross_entropy(logits.reshape([-1, logits.shape[-1]]), safe_labels, reduction="none")
            loss = (token_loss * loss_mask).sum() / loss_mask.sum()

        return PaliGemma2ModelOutput(loss=loss, logits=logits)


__all__ = [
    "PaliGemma2PreTrainedModel",
    "PaliGemma2ForConditionalGeneration",
    "PaliGemma2ForCausalLM",
]
