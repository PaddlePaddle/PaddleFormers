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

from __future__ import annotations

import os
from typing import Any, Callable

import paddle
from paddle import nn

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ..cache_utils import Cache
from ..llama.modeling import (
    LLamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRotaryEmbedding,
    rotate_half,
)
from ..model_utils import PretrainedModel
from ..modeling_rope_utils import dynamic_rope_update
from .configuration import JanusConfig
from .vision import JanusMlpProjector, JanusVisionModel, _effective_vision_layers

_RUNTIME_DTYPE_MAP = {
    "float16": paddle.float16,
    "bfloat16": paddle.bfloat16,
    "float32": paddle.float32,
    "float64": paddle.float64,
}

_HF_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".safetensors.index.json",
    ".bin",
    ".bin.index.json",
    ".pt",
    ".pth",
)


def _dtype_name(value) -> str:
    """Normalize Paddle dtype values and config strings for policy checks."""

    if value is None:
        return ""
    return str(value).lower().replace("paddle.", "")


def _is_bfloat16_materialization(config: JanusConfig) -> bool:
    """Return whether the current construction will materialize BF16 params."""

    candidates = (
        paddle.get_default_dtype(),
        getattr(config, "dtype", None),
        getattr(getattr(config, "language_config", None), "dtype", None),
        getattr(getattr(config, "language_config", None), "torch_dtype", None),
    )
    return any(_dtype_name(value) in ("bfloat16", "bf16") for value in candidates)


def _apply_default_bfloat16_policy(config: JanusConfig) -> None:
    """Make BF16 multimodal construction use the validated parity runtime.

    Checkpoint storage and execution dtype are independent.  An explicit
    ``checkpoint`` policy remains an opt-out for native-kernel experiments;
    otherwise an unspecified BF16 policy must not silently select the known
    cross-framework-divergent visual kernels.
    """

    if not _is_bfloat16_materialization(config):
        return
    if not config.vision_config or not config.aligner_config:
        return

    params = config.vision_config.get("params") or {}
    # ``native`` is used by the diagnostic checkpoint to request the raw
    # BF16 kernels explicitly.  Preserve that choice; otherwise a load of a
    # checkpoint with no serialized runtime policy would silently rewrite the
    # experiment into the parity policy below.
    if params.get("vision_parity_precision") == "native" or (
        "paddle_high_precision" in params and params["paddle_high_precision"] is False
    ):
        return

    if config.language_compute_dtype is None:
        config.language_compute_dtype = "float32"
    if config.vision_compute_dtype is None:
        config.vision_compute_dtype = "float64"

    params = config.vision_config.setdefault("params", {})
    if "vision_parity_precision" not in params and "paddle_high_precision" not in params:
        params["vision_parity_precision"] = "fp64_accumulate"


def _requested_runtime_dtype(config: JanusConfig, name: str):
    requested = getattr(config, name, None)
    if requested in (None, "checkpoint"):
        return None
    return _RUNTIME_DTYPE_MAP[requested]


def _validated_image_masks(
    images_seq_mask: paddle.Tensor,
    images_emb_mask: paddle.Tensor,
    batch_size: int,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    if images_seq_mask.ndim != 2 or images_emb_mask.ndim != 3:
        raise ValueError("images_seq_mask must be [batch, sequence] and images_emb_mask [batch, image, token]")
    if images_seq_mask.shape[0] != batch_size or images_emb_mask.shape[0] != batch_size:
        raise ValueError("image masks must have the same batch size as input_ids")

    seq_mask = images_seq_mask.astype("bool")
    image_mask = images_emb_mask.astype("bool").reshape([batch_size, -1])
    placeholder_counts = seq_mask.astype("int64").sum(axis=1)
    embedding_counts = image_mask.astype("int64").sum(axis=1)
    if not bool(paddle.equal(placeholder_counts, embedding_counts).all().item()):
        raise ValueError("the number of image placeholders must equal the number of image embeddings for each sample")
    return seq_mask, image_mask


def _looks_like_hf_weight_filename(name: str) -> bool:
    """Return whether a basename follows the conventional HF weight names."""

    name = os.path.basename(os.fspath(name)).lower()
    return name.startswith(("model.", "model-", "pytorch_model.", "pytorch_model-")) and name.endswith(
        _HF_WEIGHT_SUFFIXES
    )


def _probe_remote_hf_checkpoint_reference(reference, download_hub=None) -> bool:
    """Probe an unresolved repository for a conventional HF weight file.

    The parent loader has a native flex-checkpoint default, so an unknown
    remote reference must not be classified as a raw HF snapshot merely from
    its repository-id shape.  Cache inspection is attempted first for offline
    use; the Hub file listing is only a best-effort probe and all errors fall
    back to ``False`` so native checkpoints retain their normal route.
    """

    source = download_hub
    if source is None:
        source = os.environ.get("DOWNLOAD_SOURCE")
    if source is not None:
        source = str(getattr(source, "value", source)).lower()
        if source not in ("", "default", "huggingface", "hf"):
            return False

    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    try:
        from huggingface_hub import HfApi, try_to_load_from_cache

        for filename in candidates:
            cached = try_to_load_from_cache(reference, filename)
            if isinstance(cached, (str, os.PathLike)) and os.path.isfile(cached):
                return True

        files = HfApi().list_repo_files(reference, repo_type="model")
        return any(_looks_like_hf_weight_filename(filename) for filename in files)
    except Exception:
        # A transient network/auth/cache failure must not change the default
        # behavior for repositories that may contain native flex checkpoints.
        return False


def _is_raw_hf_checkpoint_reference(
    pretrained_model_name_or_path,
    convert_from_hf=True,
    download_hub=None,
    subfolder=None,
):
    """Detect a raw Hugging Face checkpoint before the shared loader picks flex format.

    Paddle's flex loader consumes ``*.metadata``/``*.distcp`` directories.  A
    Hugging Face snapshot instead contains ``model*.safetensors`` (or the
    equivalent PyTorch files), so Janus routes that case through the existing
    streaming safetensors loader.  The check is deliberately local to Janus;
    other models retain the repository-wide default.
    """

    if convert_from_hf is False or pretrained_model_name_or_path is None:
        return False

    path = os.fspath(pretrained_model_name_or_path)
    if os.path.isfile(path):
        return _looks_like_hf_weight_filename(path) or path.lower().endswith((".safetensors", ".bin", ".pt", ".pth"))
    if os.path.isdir(path):
        checkpoint_dir = os.path.join(path, os.fspath(subfolder)) if subfolder else path
        if not os.path.isdir(checkpoint_dir):
            return False
        names = os.listdir(checkpoint_dir)
        # Flex checkpoints always have both a metadata file and at least one
        # data shard.  Ignore unrelated ``*.metadata`` sidecars (for example
        # Hugging Face cache markers) when a raw HF export is present.
        has_metadata = any(name.endswith(".metadata") for name in names)
        has_data = any(name.endswith(".distcp") for name in names)
        if has_metadata and has_data:
            return False
        return any(_looks_like_hf_weight_filename(name) for name in names)

    # Non-local references are resolved by the parent loader.  Only a
    # Hugging Face source is known to provide the raw ``model*.safetensors``
    # layout.  Other hubs may expose native flex checkpoints and must retain
    # the repository-wide default.  ``download_hub`` can be an enum or its
    # string value, while an omitted value follows the shared resolver's
    # ``DOWNLOAD_SOURCE`` environment default.
    source = download_hub
    if source is not None:
        source = str(getattr(source, "value", source)).lower()
        if source in ("huggingface", "hf"):
            return True
        if source not in ("", "default"):
            return False

    return _probe_remote_hf_checkpoint_reference(pretrained_model_name_or_path, download_hub)


def janus_apply_rotary_pos_emb(
    q: paddle.Tensor,
    k: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Apply the official Janus RoPE arithmetic without changing shared Llama."""

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class JanusLlamaAttention(LLamaAttention):
    """Llama attention with Janus-local input-dtype rotary arithmetic."""

    def forward(
        self,
        hidden_states: paddle.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        position_embeddings: tuple[paddle.Tensor, paddle.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[paddle.Tensor, list[paddle.Tensor] | None]:
        if self.config.sequence_parallel:
            seq_len = self.config.max_sequence_length
            batch_size = hidden_states.shape[0] * self.config.tensor_model_parallel_size // seq_len
        else:
            batch_size, seq_len = hidden_states.shape[:2]

        q_shape = (batch_size, seq_len, -1, self.head_dim)
        kv_shape = (batch_size, seq_len, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).reshape(q_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).reshape(kv_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).reshape(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = janus_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS["sdpa"]
        if self.config._attn_implementation != "sdpa":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        if self.config.sequence_parallel:
            attn_output = attn_output.reshape([-1, attn_output.shape[-1]])
        return self.o_proj(attn_output), attn_weights


class JanusLlamaDecoderLayer(LlamaDecoderLayer):
    """Decoder layer whose attention implementation is private to Janus."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        attention_state = self.self_attn.state_dict()
        self.self_attn = JanusLlamaAttention(config, layer_idx)
        self.self_attn.set_state_dict(attention_state)


class JanusLlamaRotaryEmbedding(LlamaRotaryEmbedding):
    """RoPE module matching Torch's dtype and promoted-runtime behavior."""

    @dynamic_rope_update
    def forward(self, x: paddle.Tensor, position_ids: paddle.Tensor):
        with paddle.amp.auto_cast(enable=False):
            use_original_frequency = self.rope_type == "default" and x.dtype not in (paddle.bfloat16, paddle.float16)
            inverse_frequency = self.original_inv_freq if use_original_frequency else self.inv_freq
            inverse_frequency = inverse_frequency.to(x.place).astype(x.dtype)
            inv_freq_expanded = inverse_frequency[None, :, None].float().expand([position_ids.shape[0], -1, 1])
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class JanusLlamaModel(LlamaModel):
    """Llama model assembled with Janus-local decoder and RoPE layers."""

    def __init__(self, config):
        super().__init__(config)
        original_layers = list(self.layers)
        self.layers = nn.LayerList(
            [JanusLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        for new_layer, old_layer in zip(self.layers, original_layers):
            new_layer.set_state_dict(old_layer.state_dict())
        self.rotary_emb = JanusLlamaRotaryEmbedding(config)


def _make_janus_language_model(config) -> LlamaForCausalLM:
    """Build the shared Llama wrapper, replacing only its local decoder tower."""

    language_model = LlamaForCausalLM(config)
    original_model = language_model.model
    janus_model = JanusLlamaModel(config)
    janus_model.set_state_dict(original_model.state_dict())
    language_model.model = janus_model
    language_model.tie_weights()
    return language_model


class JanusPretrainedModel(PretrainedModel):
    """Base class for the Janus multimodal model."""

    config_class = JanusConfig
    base_model_prefix = "language_model"
    transpose_weight_keys = list(LlamaForCausalLM.transpose_weight_keys) + [
        "qkv",
        "q",
        "kv",
        "proj",
        "fc1",
        "fc2",
        r"aligner\.layers\.\d+",
    ]
    input_modalities = ["text", "image"]
    _keys_to_ignore_on_load_unexpected = [r"^gen_"]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Raw HF snapshots do not contain the metadata required by Paddle's
        # flex checkpoint reader.  Keep explicit caller choices untouched and
        # only select the regular streaming loader for an otherwise-default
        # Janus load.
        if (
            "load_checkpoint_format" not in kwargs
            and "state_dict" not in kwargs
            and not kwargs.get("enable_auto_parallel", False)
            and kwargs.get("flex_ckpt_comm_method") is None
            and _is_raw_hf_checkpoint_reference(
                pretrained_model_name_or_path,
                kwargs.get("convert_from_hf", True),
                kwargs.get("download_hub"),
                kwargs.get("subfolder"),
            )
        ):
            kwargs["load_checkpoint_format"] = ""
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    @classmethod
    def _gen_aoa_config(cls, config: JanusConfig):
        """Describe the official Janus HF layout for flex checkpoint loads.

        The understanding path intentionally keeps the source names.  Only
        Linear tensors need a transpose because HF/PyTorch stores them as
        ``[out_features, in_features]`` while Paddle stores them as
        ``[in_features, out_features]``.  Generator-only ``gen_*`` tensors are
        absent from these statements and are therefore ignored by the
        understanding model.
        """

        statements = [
            "language_model.model.embed_tokens.weight -> language_model.model.embed_tokens.weight",
            "language_model.model.norm.weight -> language_model.model.norm.weight",
        ]
        if config.language_config.tie_word_embeddings:
            statements.append("language_model.model.embed_tokens.weight -> language_model.lm_head.weight")
        else:
            statements.append("language_model.lm_head.weight -> language_model.lm_head.weight")

        # Expand transposed layer mappings explicitly.  Paddle's AOA layer
        # macro infers the layer count from the first source name; a ``^T``
        # suffix is not present in the checkpoint key and would otherwise
        # make every transposed rule expand to layer zero only.
        language_layers = int(config.language_config.num_hidden_layers)
        for layer_id in range(language_layers):
            layer_prefix = f"language_model.model.layers.{layer_id}"
            statements.extend(
                [
                    f"{layer_prefix}.input_layernorm.weight -> {layer_prefix}.input_layernorm.weight",
                    f"{layer_prefix}.post_attention_layernorm.weight -> {layer_prefix}.post_attention_layernorm.weight",
                ]
            )
            statements.extend(
                [
                    f"{layer_prefix}.self_attn.{name}.weight^T -> {layer_prefix}.self_attn.{name}.weight"
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                ]
            )
            statements.extend(
                [
                    f"{layer_prefix}.mlp.{name}.weight^T -> {layer_prefix}.mlp.{name}.weight"
                    for name in ("gate_proj", "up_proj", "down_proj")
                ]
            )

        vision_config = config.vision_config or {}
        vision_params = vision_config.get("params", {})
        if vision_config:
            vision_prefix = "vision_model.vision_tower."
            vision_layers = _effective_vision_layers(vision_params)
            statements.extend(
                [
                    f"{vision_prefix}pos_embed -> {vision_prefix}pos_embed",
                    f"{vision_prefix}patch_embed.proj.weight -> {vision_prefix}patch_embed.proj.weight",
                    f"{vision_prefix}patch_embed.proj.bias -> {vision_prefix}patch_embed.proj.bias",
                    f"{vision_prefix}norm.weight -> {vision_prefix}norm.weight",
                    f"{vision_prefix}norm.bias -> {vision_prefix}norm.bias",
                ]
            )
            for layer_id in range(vision_layers):
                block_prefix = f"{vision_prefix}blocks.{layer_id}"
                statements.extend(
                    [
                        f"{block_prefix}.norm1.weight -> {block_prefix}.norm1.weight",
                        f"{block_prefix}.norm1.bias -> {block_prefix}.norm1.bias",
                        f"{block_prefix}.norm2.weight -> {block_prefix}.norm2.weight",
                        f"{block_prefix}.norm2.bias -> {block_prefix}.norm2.bias",
                        *[
                            f"{block_prefix}.attn.{name}.weight^T -> {block_prefix}.attn.{name}.weight"
                            for name in ("qkv", "proj")
                        ],
                        *[
                            f"{block_prefix}.attn.{name}.bias -> {block_prefix}.attn.{name}.bias"
                            for name in ("qkv", "proj")
                        ],
                        *[
                            f"{block_prefix}.mlp.{name}.weight^T -> {block_prefix}.mlp.{name}.weight"
                            for name in ("fc1", "fc2")
                        ],
                        *[
                            f"{block_prefix}.mlp.{name}.bias -> {block_prefix}.mlp.{name}.bias"
                            for name in ("fc1", "fc2")
                        ],
                    ]
                )

            if vision_params.get("class_token", False):
                statements.append(f"{vision_prefix}cls_token -> {vision_prefix}cls_token")

            if vision_params.get("global_pool", "map") == "map":
                statements.extend(
                    [
                        f"{vision_prefix}attn_pool.latent -> {vision_prefix}attn_pool.latent",
                        f"{vision_prefix}attn_pool.norm.weight -> {vision_prefix}attn_pool.norm.weight",
                        f"{vision_prefix}attn_pool.norm.bias -> {vision_prefix}attn_pool.norm.bias",
                        f"{vision_prefix}attn_pool.q.weight^T -> {vision_prefix}attn_pool.q.weight",
                        f"{vision_prefix}attn_pool.q.bias -> {vision_prefix}attn_pool.q.bias",
                        f"{vision_prefix}attn_pool.kv.weight^T -> {vision_prefix}attn_pool.kv.weight",
                        f"{vision_prefix}attn_pool.kv.bias -> {vision_prefix}attn_pool.kv.bias",
                        f"{vision_prefix}attn_pool.proj.weight^T -> {vision_prefix}attn_pool.proj.weight",
                        f"{vision_prefix}attn_pool.proj.bias -> {vision_prefix}attn_pool.proj.bias",
                        f"{vision_prefix}attn_pool.mlp.fc1.weight^T -> {vision_prefix}attn_pool.mlp.fc1.weight",
                        f"{vision_prefix}attn_pool.mlp.fc1.bias -> {vision_prefix}attn_pool.mlp.fc1.bias",
                        f"{vision_prefix}attn_pool.mlp.fc2.weight^T -> {vision_prefix}attn_pool.mlp.fc2.weight",
                        f"{vision_prefix}attn_pool.mlp.fc2.bias -> {vision_prefix}attn_pool.mlp.fc2.bias",
                    ]
                )

        aligner_config = config.aligner_config or {}
        if aligner_config:
            aligner_params = aligner_config.get("params", {})
            projector_type = aligner_params.get("projector_type", "mlp_gelu")
            if projector_type == "linear":
                linear_indices = [0]
            elif projector_type == "mlp_gelu":
                depth = int(aligner_params.get("depth", 1))
                linear_indices = [2 * index for index in range(depth)]
            else:
                linear_indices = []
            for index in linear_indices:
                statements.extend(
                    [
                        f"aligner.layers.{index}.weight^T -> aligner.layers.{index}.weight",
                        f"aligner.layers.{index}.bias -> aligner.layers.{index}.bias",
                    ]
                )

        return {"aoa_statements": statements}


class JanusForCausalLM(JanusPretrainedModel):
    """Janus vision tower, aligner, and PaddleFormers Llama language model."""

    def __init__(self, config: JanusConfig, apply_runtime_compute_dtype: bool = True):
        super().__init__(config)
        _apply_default_bfloat16_policy(config)
        vision_config = config.vision_config
        vision_params = vision_config.get("params", {}) if vision_config else None
        self.vision_model = JanusVisionModel(vision_params) if vision_config else None
        aligner_config = config.aligner_config
        aligner_params = aligner_config.get("params", {}) if aligner_config else None
        vision_high_precision = bool(self.vision_model is not None and self.vision_model.vision_tower.high_precision)
        self.aligner = (
            JanusMlpProjector(aligner_params, high_precision=vision_high_precision) if aligner_config else None
        )
        self.language_model = _make_janus_language_model(config.language_config)
        self.runtime_compute_dtypes = {
            "language": getattr(config, "language_compute_dtype", None) or "checkpoint",
            "vision": getattr(config, "vision_compute_dtype", None) or "checkpoint",
        }
        if apply_runtime_compute_dtype:
            self._apply_runtime_compute_dtypes()

    def _apply_runtime_compute_dtypes(self):
        language_dtype = _requested_runtime_dtype(self.config, "language_compute_dtype")
        vision_dtype = _requested_runtime_dtype(self.config, "vision_compute_dtype")
        if language_dtype is not None:
            self.language_model.to(dtype=language_dtype)
        if vision_dtype is not None:
            if self.vision_model is None or self.aligner is None:
                raise ValueError("vision_compute_dtype requires vision_model and aligner")
            self.vision_model.to(dtype=vision_dtype)
            self.aligner.to(dtype=vision_dtype)

    def _language_embedding_dtype(self):
        return self.language_model.get_input_embeddings().weight.dtype

    def prepare_inputs_embeds(
        self,
        input_ids: paddle.Tensor,
        pixel_values: paddle.Tensor,
        images_seq_mask: paddle.Tensor,
        images_emb_mask: paddle.Tensor,
    ) -> paddle.Tensor:
        """Replace image placeholder token embeddings with aligned features."""

        if self.vision_model is None or self.aligner is None:
            raise ValueError("Janus vision_model and aligner are required for image inputs")
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must have rank 4 or 5, got {pixel_values.shape}")

        batch_size, num_images = pixel_values.shape[:2]
        images = pixel_values.reshape([batch_size * num_images, *pixel_values.shape[2:]])
        image_embeds = self.aligner(self.vision_model(images))
        image_embeds = image_embeds.astype(self._language_embedding_dtype())
        image_embeds = image_embeds.reshape([batch_size, num_images * image_embeds.shape[1], image_embeds.shape[2]])
        seq_mask, image_mask = _validated_image_masks(images_seq_mask, images_emb_mask, input_ids.shape[0])

        safe_input_ids = paddle.where(input_ids < 0, paddle.zeros_like(input_ids), input_ids)
        inputs_embeds = self.language_model.get_input_embeddings()(safe_input_ids).clone()
        hidden_size = inputs_embeds.shape[-1]
        flat_inputs = inputs_embeds.reshape([-1, hidden_size])
        flat_inputs[seq_mask.reshape([-1])] = image_embeds.reshape([-1, hidden_size])[image_mask.reshape([-1])]
        return flat_inputs.reshape(inputs_embeds.shape)

    def forward(
        self,
        input_ids: paddle.Tensor | None = None,
        pixel_values: paddle.Tensor | None = None,
        images_seq_mask: paddle.Tensor | None = None,
        images_emb_mask: paddle.Tensor | None = None,
        inputs_embeds: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        **kwargs: Any,
    ):
        image_kwargs = {
            "pixel_values": pixel_values,
            "images_seq_mask": images_seq_mask,
            "images_emb_mask": images_emb_mask,
            "images": kwargs.pop("images", None),
            "image_embeds": kwargs.pop("image_embeds", None),
        }
        pixel_values = pixel_values if pixel_values is not None else image_kwargs["images"]
        image_embeds = image_kwargs["image_embeds"]
        if pixel_values is not None or image_embeds is not None:
            if input_ids is None or images_seq_mask is None or images_emb_mask is None:
                raise ValueError("image inputs require input_ids, images_seq_mask, and images_emb_mask")
            if image_embeds is None:
                inputs_embeds = self.prepare_inputs_embeds(
                    input_ids,
                    pixel_values,
                    images_seq_mask,
                    images_emb_mask,
                )
            else:
                safe_input_ids = paddle.where(input_ids < 0, paddle.zeros_like(input_ids), input_ids)
                inputs_embeds = self.language_model.get_input_embeddings()(safe_input_ids).clone()
                image_embeds = image_embeds.astype(self._language_embedding_dtype())
                seq_mask, image_mask = _validated_image_masks(images_seq_mask, images_emb_mask, input_ids.shape[0])
                hidden_size = inputs_embeds.shape[-1]
                flat_inputs = inputs_embeds.reshape([-1, hidden_size])
                flat_inputs[seq_mask.reshape([-1])] = image_embeds.reshape([-1, hidden_size])[image_mask.reshape([-1])]
                inputs_embeds = flat_inputs.reshape(inputs_embeds.shape)
            input_ids = None

        # Transformers' causal-LM contract scores the token at position ``t``
        # against the label at ``t + 1``.  The shared Paddle Llama wrapper
        # intentionally retains its historical pre-aligned-label behavior, so
        # apply the shift at the Janus boundary where ms-swift supplies the
        # ordinary, unshifted SFT labels.
        outputs = self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=None,
            **kwargs,
        )
        if labels is None:
            return outputs

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shifted_logits = logits[..., :-1, :]
        shifted_labels = labels[..., 1:]
        shifted_loss_mask = loss_mask[..., 1:] if loss_mask is not None else None
        loss, _ = self.language_model.criterion(
            shifted_logits,
            shifted_labels,
            shifted_loss_mask,
        )

        if isinstance(outputs, tuple):
            return (loss,) + outputs
        outputs.loss = loss
        return outputs

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        cache_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                cache_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, tuple) and past_key_values and past_key_values[0] is not None:
                cache_length = past_key_values[0][0].shape[-2]

        if cache_length > 0:
            # Image features are already represented by the prefill KV cache.
            for name in ("pixel_values", "images", "image_embeds", "images_seq_mask", "images_emb_mask"):
                model_inputs[name] = None
        return model_inputs


JanusModel = JanusForCausalLM


__all__ = ["JanusPretrainedModel", "JanusModel", "JanusForCausalLM"]
