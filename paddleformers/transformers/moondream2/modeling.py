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

"""Paddle implementation of the Moondream2 vision-language architecture."""

import math
from typing import Optional, Tuple, Union

import paddle
from paddle import nn
from paddle.distributed.fleet.utils import recompute
from paddle.nn import functional as F

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.embedding import Embedding as GeneralEmbedding
from ...nn.linear import Linear as GeneralLinear
from ...nn.lm_head import LMHead as GeneralLMHead
from ...nn.norm import Norm as GeneralNorm
from ..cache_utils import Cache, DynamicCache
from ..model_outputs import ModelOutput
from ..model_utils import PretrainedModel, register_base_model
from .configuration import Moondream2Config, Moondream2TextConfig, Moondream2VisionConfig

__all__ = [
    "Moondream2CausalLMOutputWithPast",
    "Moondream2ForConditionalGeneration",
    "Moondream2Model",
    "Moondream2ModelOutputWithPast",
    "Moondream2PreTrainedModel",
    "Moondream2VisionModel",
    "build_moondream2_prefix_mask",
]


class Moondream2ModelOutputWithPast(ModelOutput):
    """Base-model output, mirroring the other multimodal PaddleFormers models."""

    last_hidden_state: Optional[paddle.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


class Moondream2CausalLMOutputWithPast(ModelOutput):
    """Causal-LM output carrying the Moondream2 logits and cache."""

    loss: Optional[paddle.Tensor] = None
    logits: Optional[paddle.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


def gelu_tanh(hidden_states):
    return F.gelu(hidden_states, approximate=True)


def build_moondream2_prefix_mask(query_length, key_value_length, prefix_length, past_length=0, dtype="float32"):
    """Build Moondream2's explicit additive attention mask.

    The first ``prefix_length`` positions are mutually visible.  Every later
    query can attend causally to all prior positions.  The returned tensor has
    shape ``[1, 1, query_length, key_value_length]`` and is suitable for eager
    attention scores.
    """
    query_positions = paddle.arange(past_length, past_length + query_length).unsqueeze(-1)
    key_positions = paddle.arange(key_value_length).unsqueeze(0)
    visible = key_positions <= query_positions
    prefix_visible = (query_positions < prefix_length) & (key_positions < prefix_length)
    visible = visible | prefix_visible
    zero = paddle.zeros([query_length, key_value_length], dtype=dtype)
    negative = paddle.full([query_length, key_value_length], -1e9, dtype=dtype)
    return paddle.where(visible, zero, negative).unsqueeze(0).unsqueeze(0)


def build_moondream2_block_mask(query_length, key_value_length, image_span=None, past_length=0, dtype="float32"):
    """Build the additive mask used by the decoder.

    Positions are causal, except that an image block (``[start, end)``) is
    visible in full to every query at or after its start. With ``image_span``
    set to ``(0, prefix_attn)`` this reproduces the reference bidirectional
    image prefix; with it set to an in-place placeholder span it makes the
    expanded image tokens attend to each other.
    """
    query_positions = paddle.arange(past_length, past_length + query_length).unsqueeze(-1)
    key_positions = paddle.arange(key_value_length).unsqueeze(0)
    visible = key_positions <= query_positions
    if image_span is not None:
        start, end = image_span
        in_block = (key_positions >= start) & (key_positions < end)
        visible = visible | (in_block & (query_positions >= start))
    zero = paddle.zeros([query_length, key_value_length], dtype=dtype)
    negative = paddle.full([query_length, key_value_length], -1e9, dtype=dtype)
    return paddle.where(visible, zero, negative).unsqueeze(0).unsqueeze(0)


def apply_partial_rope(query, key, position_ids, rotary_dim, inv_freq):
    """Apply Moondream2's non-interleaved RoPE to only the leading dimensions.

    ``inv_freq`` is precomputed once per attention layer so every forward pass
    and every decoder layer does not rebuild the frequency table.
    """
    if rotary_dim == 0:
        return query, key
    positions = position_ids.astype("float32").unsqueeze(-1)
    freqs = positions * inv_freq.unsqueeze(0).unsqueeze(0)
    cos = paddle.cos(freqs).unsqueeze(1)
    sin = paddle.sin(freqs).unsqueeze(1)

    def rotate(states):
        rotated, passthrough = states[..., :rotary_dim], states[..., rotary_dim:]
        half = rotary_dim // 2
        # Match the source implementation exactly: calculate the complex
        # rotation in FP32, then cast the concatenated rotated portion back to
        # the projection dtype.  Paddle previously downcast the trig factors
        # before multiplication, which changes the BF16 inference path.
        real = rotated[..., :half].astype("float32")
        imag = rotated[..., half:].astype("float32")
        out = paddle.concat([real * cos - imag * sin, real * sin + imag * cos], axis=-1).astype(states.dtype)
        return paddle.concat([out, passthrough], axis=-1)

    return rotate(query), rotate(key)


class Moondream2MLP(nn.Layer):
    def __init__(self, config, in_features, intermediate_size, out_features):
        super().__init__()
        self.fc1 = GeneralLinear.create(
            config=config, in_features=in_features, out_features=intermediate_size, has_bias=True, tp_plan="colwise"
        )
        self.fc2 = GeneralLinear.create(
            config=config,
            in_features=intermediate_size,
            out_features=out_features,
            has_bias=True,
            tp_plan="rowwise",
            input_is_parallel=True,
        )

    def forward(self, hidden_states):
        return self.fc2(gelu_tanh(self.fc1(hidden_states)))


class Moondream2VisionAttention(nn.Layer):
    def __init__(self, config: Moondream2VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.qkv = GeneralLinear.create(
            config=config,
            in_features=config.hidden_size,
            out_features=config.hidden_size * 3,
            has_bias=True,
            tp_plan="colwise",
            gather_output=True,
        )
        self.proj = GeneralLinear.create(
            config=config,
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            has_bias=True,
            tp_plan="rowwise",
        )

    def forward(self, hidden_states):
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape([batch_size, sequence_length, 3, self.num_heads, self.head_dim])
        query, key, value = qkv.unbind(axis=2)
        query, key, value = query.transpose([0, 2, 1, 3]), key.transpose([0, 2, 1, 3]), value.transpose([0, 2, 1, 3])
        scores = paddle.matmul(query * self.scale, key, transpose_y=True)
        attention = F.softmax(scores, axis=-1)
        output = paddle.matmul(attention, value).transpose([0, 2, 1, 3]).reshape([batch_size, sequence_length, -1])
        return self.proj(output)


class Moondream2VisionBlock(nn.Layer):
    def __init__(self, config: Moondream2VisionConfig):
        super().__init__()
        self.norm1 = GeneralNorm.create(
            config=config, norm_type="layer_norm", hidden_size=config.hidden_size, norm_eps=config.layer_norm_eps
        )
        self.attn = Moondream2VisionAttention(config)
        self.norm2 = GeneralNorm.create(
            config=config, norm_type="layer_norm", hidden_size=config.hidden_size, norm_eps=config.layer_norm_eps
        )
        self.mlp = Moondream2MLP(config, config.hidden_size, config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Moondream2VisionModel(nn.Layer):
    """ViT used for every global/local Moondream image crop."""

    def __init__(self, config: Moondream2VisionConfig):
        super().__init__()
        self.config = config
        self.grid_size = config.crop_size // config.patch_size
        self.patch_embed = nn.Linear(config.num_channels * config.patch_size * config.patch_size, config.hidden_size)
        self.position_embedding = self.create_parameter(
            [1, self.grid_size * self.grid_size, config.hidden_size], default_initializer=nn.initializer.Normal(std=0.02)
        )
        self.blocks = nn.LayerList([Moondream2VisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = GeneralNorm.create(
            config=config, norm_type="layer_norm", hidden_size=config.hidden_size, norm_eps=config.layer_norm_eps
        )

    def forward(self, pixel_values):
        batch_size, _, height, width = pixel_values.shape
        if height != self.config.crop_size or width != self.config.crop_size:
            raise ValueError(f"Moondream2 expects {self.config.crop_size}x{self.config.crop_size} crops, got {height}x{width}")
        patch = self.config.patch_size
        grid = self.grid_size
        patches = pixel_values.reshape([batch_size, self.config.num_channels, grid, patch, grid, patch])
        patches = patches.transpose([0, 2, 4, 1, 3, 5]).reshape([batch_size, grid * grid, -1])
        hidden_states = self.patch_embed(patches) + self.position_embedding
        for block in self.blocks:
            # A native image contains one global and up to twelve local crops.
            # Storing every ViT block's activations for all crops exceeds a 46 GB
            # GPU during full FP32 SFT, although inference itself fits normally.
            # Recompute each vision block in backward to retain only its input.
            # This changes memory/runtime trade-offs only while training and
            # leaves the eager inference/parity path untouched.
            if self.training:
                hidden_states = recompute(block, hidden_states)
            else:
                hidden_states = block(hidden_states)
        return self.norm(hidden_states)


class Moondream2VisionProjector(nn.Layer):
    def __init__(self, config: Moondream2VisionConfig):
        super().__init__()
        self.mlp = Moondream2MLP(
            config, config.hidden_size * 2, config.projection_inner_dim, config.projection_dim
        )

    def forward(self, global_features, reconstructed_features):
        return self.mlp(paddle.concat([global_features, reconstructed_features], axis=-1))


class Moondream2TextAttention(nn.Layer):
    def __init__(self, config: Moondream2TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_dim = config.rotary_dim
        self.rope_theta = config.rope_theta
        self.scale = self.head_dim**-0.5
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = 0.0
        self.config = config
        projection_size = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        self.qkv = GeneralLinear.create(
            config=config,
            in_features=config.hidden_size,
            out_features=projection_size,
            has_bias=True,
            tp_plan="colwise",
            gather_output=True,
        )
        self.out_proj = GeneralLinear.create(
            config=config,
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            has_bias=True,
            tp_plan="rowwise",
        )
        if self.rotary_dim > 0:
            self.register_buffer(
                "inv_freq",
                1.0
                / (
                    self.rope_theta
                    ** (paddle.arange(0, self.rotary_dim, 2, dtype="float32") / self.rotary_dim)
                ),
                persistable=False,
            )
        else:
            self.inv_freq = None

    def forward(
        self,
        hidden_states,
        position_ids,
        attention_mask,
        past_key_values=None,
        use_cache=False,
        attn_mask_startend_row_indices=None,
    ):
        interface_name = getattr(self.config, "_attn_implementation", "eager")
        if interface_name not in ALL_ATTENTION_FUNCTIONS:
            interface_name = "eager"
        # Flashmask only consumes start/end row indices, so the dense additive
        # mask built for the bidirectional image prefix keeps using eager.
        if interface_name == "flashmask" and attention_mask is not None and attn_mask_startend_row_indices is None:
            interface_name = "eager"
        batch_size, query_length, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query_end = self.num_heads * self.head_dim
        key_end = query_end + self.num_key_value_heads * self.head_dim
        query = qkv[..., :query_end].reshape([batch_size, query_length, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key = qkv[..., query_end:key_end].reshape([batch_size, query_length, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value = qkv[..., key_end:].reshape([batch_size, query_length, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        query, key = apply_partial_rope(query, key, position_ids, self.rotary_dim, self.inv_freq)

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            key, value = past_key_values.update(key, value, self.layer_idx)

        # Grouped-query attention is expanded by the attention backend itself
        # (eager repeats with ``num_key_value_groups``, sdpa uses ``enable_gqa``).
        attention_interface = ALL_ATTENTION_FUNCTIONS[interface_name]
        output, _ = attention_interface(
            self,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scale,
        )
        return self.out_proj(output), past_key_values


class Moondream2DecoderBlock(nn.Layer):
    """Parallel pre-norm residual block used by the original Moondream decoder."""

    def __init__(self, config: Moondream2TextConfig, layer_idx: int):
        super().__init__()
        self.norm = GeneralNorm.create(
            config=config, norm_type="layer_norm", hidden_size=config.hidden_size, norm_eps=config.layer_norm_eps
        )
        self.attn = Moondream2TextAttention(config, layer_idx)
        self.mlp = Moondream2MLP(config, config.hidden_size, config.intermediate_size, config.hidden_size)

    def forward(
        self,
        hidden_states,
        position_ids,
        attention_mask,
        past_key_values=None,
        use_cache=False,
        attn_mask_startend_row_indices=None,
    ):
        normalized = self.norm(hidden_states)
        attention_output, past_key_values = self.attn(
            normalized,
            position_ids,
            attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        return hidden_states + attention_output + self.mlp(normalized), past_key_values


class Moondream2RegionModel(nn.Layer):
    """Optional coordinate/size embeddings and decoders for detect/point tasks."""

    def __init__(self, config):
        super().__init__()
        # The public checkpoint stores Fourier frequencies as [input_dim,
        # feature_dim / 2]: scalar coordinates use [1, 128] and sizes use
        # [2, 256].  This preserves its exact tensor layout.
        self.coordinate_features = self.create_parameter(
            [1, config.coordinate_feature_dim // 2], default_initializer=nn.initializer.Normal(std=1.0)
        )
        self.size_features = self.create_parameter(
            [2, config.size_feature_dim // 2], default_initializer=nn.initializer.Normal(std=1.0)
        )
        self.coordinate_encoder = GeneralLinear.create(
            config=config, in_features=config.coordinate_feature_dim, out_features=config.hidden_size, has_bias=True
        )
        self.size_encoder = GeneralLinear.create(
            config=config, in_features=config.size_feature_dim, out_features=config.hidden_size, has_bias=True
        )
        self.coordinate_decoder = Moondream2MLP(
            config, config.hidden_size, config.intermediate_size, config.coordinate_output_dim
        )
        self.size_decoder = Moondream2MLP(config, config.hidden_size, config.intermediate_size, config.size_output_dim)

    @staticmethod
    def _fourier(values, weights):
        frequencies = 2.0 * math.pi * paddle.matmul(values, weights)
        return paddle.concat([paddle.cos(frequencies), paddle.sin(frequencies)], axis=-1)

    def encode_coordinates(self, coordinates):
        return self.coordinate_encoder(self._fourier(coordinates, self.coordinate_features))

    def encode_sizes(self, sizes):
        return self.size_encoder(self._fourier(sizes, self.size_features))

    def decode_coordinates(self, hidden_states):
        return self.coordinate_decoder(hidden_states)

    def decode_sizes(self, hidden_states):
        return self.size_decoder(hidden_states).reshape([*hidden_states.shape[:-1], 2, -1])


class Moondream2PreTrainedModel(PretrainedModel):
    config_class = Moondream2Config

    @classmethod
    def _moondream2_aoa_statements(cls, config, inverse=False):
        """Map the reference checkpoint names onto this implementation.

        The released checkpoint stores ``model.vision.*`` / ``model.text.*`` /
        ``model.region.*`` tensors, while this implementation keeps the vision
        tower and the grounding heads under ``vision_model`` / ``region_model``
        and the decoder under ``layers``.
        """
        model_prefix = "model." if cls != cls.base_model_class else ""
        statements = []

        def add(source, target, transpose=False):
            src, dst = (target, source) if inverse else (source, target)
            if transpose:
                src = f"{src}^T"
            statements.append(f"{src} -> {dst}")

        add("model.vision.patch_emb.weight", f"{model_prefix}vision_model.patch_embed.weight", True)
        add("model.vision.patch_emb.bias", f"{model_prefix}vision_model.patch_embed.bias")
        add("model.vision.pos_emb", f"{model_prefix}vision_model.position_embedding")
        add("model.vision.post_ln.weight", f"{model_prefix}vision_model.norm.weight")
        add("model.vision.post_ln.bias", f"{model_prefix}vision_model.norm.bias")
        for module in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            add(
                f"model.vision.blocks.$LAYER_ID.{module}.weight",
                f"{model_prefix}vision_model.blocks.$LAYER_ID.{module}.weight",
                True,
            )
            add(f"model.vision.blocks.$LAYER_ID.{module}.bias", f"{model_prefix}vision_model.blocks.$LAYER_ID.{module}.bias")
        for norm in ("ln1", "ln2"):
            target_norm = f"norm{norm[-1]}"
            for param in ("weight", "bias"):
                add(
                    f"model.vision.blocks.$LAYER_ID.{norm}.{param}",
                    f"{model_prefix}vision_model.blocks.$LAYER_ID.{target_norm}.{param}",
                )
        for module in ("fc1", "fc2"):
            add(
                f"model.vision.proj_mlp.{module}.weight",
                f"{model_prefix}vision_projector.mlp.{module}.weight",
                True,
            )
            add(f"model.vision.proj_mlp.{module}.bias", f"{model_prefix}vision_projector.mlp.{module}.bias")

        add("model.text.wte", f"{model_prefix}embed_tokens.weight")
        add("model.text.post_ln.weight", f"{model_prefix}norm.weight")
        add("model.text.post_ln.bias", f"{model_prefix}norm.bias")
        add("model.text.lm_head.weight", "lm_head.weight")
        add("model.text.lm_head.bias", "lm_head.bias")
        for module in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            target_module = "attn.out_proj" if module == "attn.proj" else module
            add(
                f"model.text.blocks.$LAYER_ID.{module}.weight",
                f"{model_prefix}layers.$LAYER_ID.{target_module}.weight",
                True,
            )
            add(f"model.text.blocks.$LAYER_ID.{module}.bias", f"{model_prefix}layers.$LAYER_ID.{target_module}.bias")
        for param in ("weight", "bias"):
            add(f"model.text.blocks.$LAYER_ID.ln.{param}", f"{model_prefix}layers.$LAYER_ID.norm.{param}")

        for source, target in (
            ("coord_features", "coordinate_features"),
            ("size_features", "size_features"),
        ):
            add(f"model.region.{source}", f"{model_prefix}region_model.{target}")
        for source, target in (
            ("coord_encoder", "coordinate_encoder"),
            ("size_encoder", "size_encoder"),
            ("coord_decoder", "coordinate_decoder"),
            ("size_decoder", "size_decoder"),
        ):
            if source.endswith("encoder"):
                add(f"model.region.{source}.weight", f"{model_prefix}region_model.{target}.weight", True)
                add(f"model.region.{source}.bias", f"{model_prefix}region_model.{target}.bias")
                continue
            for module in ("fc1", "fc2"):
                add(
                    f"model.region.{source}.{module}.weight",
                    f"{model_prefix}region_model.{target}.{module}.weight",
                    True,
                )
                add(f"model.region.{source}.{module}.bias", f"{model_prefix}region_model.{target}.{module}.bias")

        return statements

    @classmethod
    def _gen_aoa_config(cls, config):
        """Convert the reference Moondream2 checkpoint into this implementation."""
        return {"aoa_statements": cls._moondream2_aoa_statements(config)}

    @classmethod
    def _gen_inv_aoa_config(cls, config):
        """Convert this implementation back to the reference checkpoint layout."""
        return {"aoa_statements": cls._moondream2_aoa_statements(config, inverse=True)}
    base_model_prefix = "model"
    _no_split_modules = ["Moondream2DecoderBlock", "Moondream2VisionBlock"]
    _keep_in_fp32_modules = ["model.norm"]

    def _init_weights(self, layer):
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            if getattr(layer, "weight", None) is not None:
                nn.initializer.Normal(std=0.02)(layer.weight)
            if getattr(layer, "bias", None) is not None:
                nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.initializer.Constant(1.0)(layer.weight)
            nn.initializer.Constant(0.0)(layer.bias)


@register_base_model
class Moondream2Model(Moondream2PreTrainedModel):
    """Vision encoder, image-prefix injection and the causal text stack."""

    base_model_prefix = "model"

    def __init__(self, config: Moondream2Config):
        super().__init__(config)
        # The attention backend is chosen on the top-level config by the loader;
        # propagate it to the text sub-config before the decoder is built.
        if getattr(config, "_attn_implementation", None) is not None:
            config.text_config._attn_implementation = config._attn_implementation
        self.vision_model = Moondream2VisionModel(config.vision_config)
        self.vision_projector = Moondream2VisionProjector(config.vision_config)
        self.region_model = Moondream2RegionModel(config.region_config)
        self.embed_tokens = GeneralEmbedding.create(
            config=config.text_config,
            num_embeddings=config.text_config.vocab_size,
            embedding_dim=config.text_config.hidden_size,
        )
        self.layers = nn.LayerList(
            [Moondream2DecoderBlock(config.text_config, idx) for idx in range(config.text_config.num_hidden_layers)]
        )
        self.norm = GeneralNorm.create(
            config=config.text_config,
            norm_type="layer_norm",
            hidden_size=config.text_config.hidden_size,
            norm_eps=config.text_config.layer_norm_eps,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _expand_image_block(self, inputs_embeds, input_ids, attention_mask, image_block, image_token_id):
        """Replace each placeholder token with its image features in place."""
        block_length = image_block.shape[1]
        counts = paddle.sum((input_ids == image_token_id).astype("int64"), axis=1)
        if int(paddle.min(counts)) != 1 or int(paddle.max(counts)) != 1:
            raise ValueError(
                "In-place image expansion expects exactly one image placeholder per sample; "
                "only enable `image_token_id` for single-image prompts."
            )
        if image_block.shape[0] != inputs_embeds.shape[0]:
            raise ValueError("The number of images must match the batch size")

        merged_embeds, merged_masks, spans = [], [], []
        for index in range(inputs_embeds.shape[0]):
            position = int(paddle.nonzero(input_ids[index] == image_token_id, as_tuple=False)[0])
            row = inputs_embeds[index]
            merged_embeds.append(paddle.concat([row[:position], image_block[index], row[position + 1 :]], axis=0))
            if attention_mask is not None:
                mask_row = attention_mask[index]
                block_mask = paddle.ones([block_length], dtype=attention_mask.dtype)
                merged_masks.append(
                    paddle.concat([mask_row[:position], block_mask, mask_row[position + 1 :]], axis=0)
                )
            spans.append((position, position + block_length))

        inputs_embeds = paddle.stack(merged_embeds, axis=0)
        if attention_mask is not None:
            attention_mask = paddle.stack(merged_masks, axis=0)
        return inputs_embeds, attention_mask, spans

    def _reconstruct_local_features(self, crop_features, tiles_h, tiles_w):
        """Stitch local ViT features using the source overlap-boundary semantics."""
        grid = self.vision_model.grid_size
        margin = self.config.vision_config.overlap_margin
        tiles_h, tiles_w = int(tiles_h), int(tiles_w)
        if tiles_h * tiles_w != crop_features.shape[0]:
            raise ValueError(
                f"Local crop count ({crop_features.shape[0]}) does not match tiling ({tiles_h}, {tiles_w})"
            )

        # Match ``reconstruct_from_crops`` in the native implementation: discard
        # overlap only on edges shared with another local crop.  In particular a
        # 1x1 tiling must preserve all 27x27 features, rather than cropping it to
        # the 19x19 interior and upsampling it again.
        rows = []
        index = 0
        for tile_y in range(tiles_h):
            cells = []
            y_start = 0 if tile_y == 0 else margin
            y_end = grid if tile_y == tiles_h - 1 else grid - margin
            for tile_x in range(tiles_w):
                x_start = 0 if tile_x == 0 else margin
                x_end = grid if tile_x == tiles_w - 1 else grid - margin
                cell = crop_features[index].reshape([grid, grid, -1])
                cells.append(cell[y_start:y_end, x_start:x_end])
                index += 1
            rows.append(paddle.concat(cells, axis=1))
        reconstructed = paddle.concat(rows, axis=0).transpose([2, 0, 1]).unsqueeze(0)
        reconstructed = F.adaptive_avg_pool2d(reconstructed, output_size=[grid, grid])
        return reconstructed.squeeze(0).transpose([1, 2, 0]).reshape([grid * grid, -1])

    def encode_image(self, pixel_values, image_sizes=None, crop_counts=None):
        """Encode normalized global/local crops into the fixed 729-token image prefix."""
        pixel_values = pixel_values.astype(self.vision_model.patch_embed.weight.dtype)
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.ndim != 5:
            raise ValueError("pixel_values must have shape [batch, crops, channels, height, width]")
        batch_size, max_crops = pixel_values.shape[:2]
        if crop_counts is None:
            crop_counts = paddle.full([batch_size], max_crops, dtype="int64")
        outputs = []
        for batch_idx in range(batch_size):
            count = int(crop_counts[batch_idx])
            encoded = self.vision_model(pixel_values[batch_idx, :count])
            global_features = encoded[0]
            if count == 1:
                reconstructed = global_features
            else:
                if image_sizes is None:
                    tiles_h, tiles_w = 1, count - 1
                else:
                    tiles_h, tiles_w = int(image_sizes[batch_idx, 0]), int(image_sizes[batch_idx, 1])
                reconstructed = self._reconstruct_local_features(encoded[1:], tiles_h, tiles_w)
            outputs.append(self.vision_projector(global_features, reconstructed))
        return paddle.stack(outputs, axis=0)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        image_sizes: Optional[paddle.Tensor] = None,
        crop_counts: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Moondream2ModelOutputWithPast]:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        # PaddleFormers' generic VL collator supplies a binary causal mask with
        # shape [batch, 1, query, key].  Moondream2 constructs its own causal
        # and image-prefix mask below, so it only needs the corresponding 2-D
        # key-padding mask.  Reducing over heads and queries preserves real
        # token columns (including a final one-token sequence) and excludes
        # padded columns before the image prefix is prepended.
        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = paddle.max(attention_mask, axis=[1, 2])

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be specified")
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = inputs_embeds.astype(self.embed_tokens.weight.dtype)

        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_image_prefix = pixel_values is not None
        image_span = getattr(self, "_image_span", None)
        if has_image_prefix:
            image_features = self.encode_image(pixel_values, image_sizes=image_sizes, crop_counts=crop_counts)
            # The native image path pre-fills one BOS embedding followed by the
            # projected image tokens as a bidirectional block.
            bos_ids = paddle.full([inputs_embeds.shape[0], 1], self.config.bos_token_id, dtype="int64")
            bos_features = self.embed_tokens(bos_ids).astype(inputs_embeds.dtype)
            image_block = paddle.concat([bos_features, image_features.astype(inputs_embeds.dtype)], axis=1)

            image_token_id = getattr(self.config, "image_token_id", None)
            if image_token_id is not None and input_ids is not None and bool(paddle.any(input_ids == image_token_id)):
                # Expansion in place: the template keeps one placeholder token
                # per sample and the image features replace it, so packed rows
                # keep each document's own image next to its own text.
                inputs_embeds, attention_mask, spans = self._expand_image_block(
                    inputs_embeds, input_ids, attention_mask, image_block, int(image_token_id)
                )
                image_span = spans[0] if len(set(spans)) == 1 else spans
                self._image_expanded = True
            else:
                inputs_embeds = paddle.concat([image_block, inputs_embeds], axis=1)
                if attention_mask is not None:
                    prefix_attention = paddle.ones(
                        [attention_mask.shape[0], image_block.shape[1]], dtype=attention_mask.dtype
                    )
                    attention_mask = paddle.concat([prefix_attention, attention_mask], axis=1)
                image_span = (0, image_block.shape[1])
                self._image_expanded = False
            self._image_span = image_span
        elif past_length == 0:
            # Fresh text-only prefill: drop any span left by a previous image.
            image_span = None
            self._image_span = None

        batch_size, query_length, _ = inputs_embeds.shape
        key_value_length = past_length + query_length
        position_ids = paddle.arange(past_length, key_value_length, dtype="int64").unsqueeze(0).expand([batch_size, -1])
        if attn_mask_startend_row_indices is not None and has_image_prefix:
            raise ValueError(
                "attn_mask_startend_row_indices cannot be combined with pixel_values: the bidirectional "
                "image block needs a dense mask until it is expressed with row indices"
            )
        backend = getattr(self.config.text_config, "_attn_implementation", "eager")
        if attn_mask_startend_row_indices is not None or (
            image_span is None and attention_mask is None and backend in ("sdpa", "flashmask")
        ):
            # Either the packing layout is expressed with row indices, or a
            # causal-capable backend derives causality on its own. Eager still
            # needs the explicit additive mask built below.
            hidden_mask = None
        else:
            if image_span is None:
                hidden_mask = build_moondream2_block_mask(
                    query_length, key_value_length, None, past_length, dtype=inputs_embeds.dtype
                )
            elif isinstance(image_span, list):
                hidden_mask = paddle.concat(
                    [
                        build_moondream2_block_mask(
                            query_length, key_value_length, span, past_length, dtype=inputs_embeds.dtype
                        )
                        for span in image_span
                    ],
                    axis=0,
                )
            else:
                hidden_mask = build_moondream2_block_mask(
                    query_length, key_value_length, image_span, past_length, dtype=inputs_embeds.dtype
                )
            if attention_mask is not None:
                if attention_mask.ndim != 2:
                    raise ValueError("Moondream2 attention_mask must have shape [batch, sequence]")
                key_padding = attention_mask[:, None, None, :key_value_length].astype(inputs_embeds.dtype)
                hidden_mask = hidden_mask + (1.0 - key_padding) * -1e9

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states, past_key_values = layer(
                hidden_states,
                position_ids,
                hidden_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            )
        last_hidden_state = self.norm(hidden_states)
        if not return_dict:
            return (last_hidden_state, past_key_values)
        return Moondream2ModelOutputWithPast(last_hidden_state=last_hidden_state, past_key_values=past_key_values)


class Moondream2ForConditionalGeneration(Moondream2PreTrainedModel):
    """Moondream2 with an image-prefix-conditioned causal language head."""

    base_model_prefix = "model"

    def __init__(self, config: Moondream2Config):
        super().__init__(config)
        self.model = Moondream2Model(config)
        self.lm_head = GeneralLMHead(config)
        self.loss_fn = CriterionLayer(config)
        # Strict FP64 inference is only required by the dedicated cross-framework
        # logit-parity harness.  It must stay opt-in: SFT configurations that
        # request FP32 need actual FP32 parameters, optimizer states and kernels.
        self._strict_logits_alignment = False

    def to(self, *args, **kwargs):
        """Optionally promote the dedicated logit-parity harness to FP64."""
        requested_dtype = kwargs.get("dtype")
        if self._strict_logits_alignment and requested_dtype in ("float32", paddle.float32):
            kwargs["dtype"] = "float64"
        return super().to(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def _align_labels_to_image_block(self, labels):
        """Align SFT labels with the image block inserted by the base model.

        Prefix injection prepends the block, so the labels only need ignored
        targets in front. In-place expansion removes the placeholder token and
        inserts the block at its position, so the placeholder's own label is
        dropped and replaced by ignored targets.
        """
        spans = getattr(self.model, "_image_span", None)
        if spans is None:
            return labels
        if not isinstance(spans, list):
            spans = [spans] * labels.shape[0]
        expanded = getattr(self.model, "_image_expanded", False)

        rows = []
        for index in range(labels.shape[0]):
            start, end = spans[index]
            block_length = end - start
            ignored = paddle.full([block_length], -100, dtype=labels.dtype)
            row = labels[index]
            if expanded:
                rows.append(paddle.concat([row[:start], ignored, row[start + 1 :]], axis=0))
            else:
                rows.append(paddle.concat([ignored, row], axis=0))
        return paddle.stack(rows, axis=0)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        crop_counts=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            past_length = past_key_values.get_seq_length()
            input_ids = input_ids[:, -1:]
            inputs_embeds = None
            pixel_values = None
            image_sizes = None
            crop_counts = None

            # GenerationMixin extends the caller-provided text mask by one token
            # per decode step, but the cache also contains the BOS + image prefix.
            # Restore those omitted prefix columns before applying the cached
            # attention mask.
            if attention_mask is not None:
                key_value_length = past_length + input_ids.shape[1]
                missing_prefix = key_value_length - attention_mask.shape[-1]
                if missing_prefix > 0:
                    prefix_attention = paddle.ones(
                        [attention_mask.shape[0], missing_prefix],
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = paddle.concat(
                        [prefix_attention, attention_mask], axis=-1
                    )
        return {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "crop_counts": crop_counts,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        image_sizes: Optional[paddle.Tensor] = None,
        crop_counts: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Moondream2CausalLMOutputWithPast]:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            crop_counts=crop_counts,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).astype("float32")
        loss = None
        if labels is not None:
            if pixel_values is not None:
                labels = self._align_labels_to_image_block(labels)
            # PaddleFormers' SFT collator already shifts labels, so logits and
            # labels are position-aligned here and must not be shifted again.
            # ``CriterionLayer`` applies the same supervised-token normalization
            # as the reference training scripts and supports tensor-parallel
            # cross entropy when the vocab axis is sharded.
            criterion_output = self.loss_fn(logits, labels)
            loss = criterion_output[0] if isinstance(criterion_output, (tuple, list)) else criterion_output
        if not return_dict:
            output = (logits, outputs.past_key_values)
            return (loss,) + output if loss is not None else output
        return Moondream2CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)
