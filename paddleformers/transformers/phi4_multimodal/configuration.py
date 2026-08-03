# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
"""Phi-4-Multimodal configuration."""

import copy
import json
import math
import os
import shutil
from pathlib import Path

from ..configuration_utils import PretrainedConfig


def _convert_phi4mm_config(config_dict, with_lora_adapters=True):
    original_config = copy.deepcopy(config_dict)
    config = dict(config_dict)

    config.pop("_name_or_path", None)
    config.pop("architectures", None)
    config.pop("auto_map", None)
    vision_lora = config.pop("vision_lora", None) or {}
    speech_lora = config.pop("speech_lora", None) or {}
    config.pop("transformers_version", None)
    config.pop("_attn_implementation", None)
    config.pop("model_type", None)

    embd_layer = config.pop("embd_layer")
    audio_embd_layer = embd_layer["audio_embd_layer"]
    vision_embd_layer = embd_layer["image_embd_layer"]

    audio_config = config.pop("audio_processor")["config"]
    audio_config.pop("activation_checkpointing", None)
    audio_config.pop("cnn_layer_norm", None)
    audio_config.pop("input_layer", None)
    audio_config.pop("batch_norm", None)
    audio_config.pop("encoder_embedding_config", None)
    audio_config.pop("ext_pw_kernel_size", None)
    audio_config.pop("bias_in_glu", None)
    audio_config.pop("causal", None)

    audio_config["hidden_size"] = audio_config.pop("attention_dim")
    audio_config["num_attention_heads"] = audio_config.pop("attention_heads")
    audio_config["intermediate_size"] = audio_config.pop("linear_units")
    audio_config["nemo_conv_channels"] = audio_config.pop("nemo_conv_settings")["conv_channels"]
    audio_config["bias_max_distance"] = audio_config.pop("relative_attention_bias_args")["t5_bias_max_distance"]
    audio_config["downsample_rate"] = audio_embd_layer["downsample_rate"]
    audio_config.pop("depthwise_seperable_out_channel", None)

    if "depthwise_separable_out_channel" not in audio_config:
        audio_config["depthwise_separable_out_channel"] = audio_config.get("ext_pw_out_channel")

    config["audio_config"] = audio_config
    config["vision_config"] = {"crop_size": vision_embd_layer["crop_size"]}
    config["eos_token_id"] = [199999, 200020]

    if with_lora_adapters:
        config.update(
            {
                "vision_lora_rank": vision_lora.get("r", 0),
                "vision_lora_alpha": vision_lora.get("lora_alpha", 1),
                "speech_lora_rank": speech_lora.get("r", 0),
                "speech_lora_alpha": speech_lora.get("lora_alpha", 1),
            }
        )
    # Keep the upstream representation so a fine-tuned checkpoint can be
    # exported back to the format consumed by Transformers and vLLM.
    config["_phi4mm_hf_config"] = original_config
    return config


class Phi4MultimodalVisionConfig(PretrainedConfig):
    model_type = "phi4_multimodal_vision"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=448,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        crop_size=448,
        image_token_id=200010,
        feature_layer=-2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.crop_size = crop_size
        self.image_token_id = image_token_id
        self.feature_layer = feature_layer


class Phi4MultimodalAudioConfig(PretrainedConfig):
    model_type = "phi4_multimodal_audio"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=1536,
        num_blocks=24,
        num_attention_heads=16,
        activation="swish",
        chunk_size=-1,
        left_chunk=18,
        dropout_rate=0.0,
        ext_pw_out_channel=1024,
        depthwise_separable_out_channel=1024,
        depthwise_multiplier=1,
        kernel_size=3,
        conv_activation="swish",
        input_size=80,
        conv_glu_type="swish",
        time_reduction=8,
        bias_max_distance=1000,
        bias_symmetric=False,
        nemo_activation="relu",
        nemo_conv_channels=1024,
        downsample_rate=1,
        initializer_range=0.02,
        audio_token_id=200011,
        feature_layer=-2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_blocks = num_blocks
        self.num_attention_heads = num_attention_heads
        self.activation = activation
        self.chunk_size = chunk_size
        self.left_chunk = left_chunk
        self.dropout_rate = dropout_rate
        self.ext_pw_out_channel = ext_pw_out_channel
        self.depthwise_separable_out_channel = depthwise_separable_out_channel
        self.depthwise_multiplier = depthwise_multiplier
        self.kernel_size = kernel_size
        self.conv_activation = conv_activation
        self.input_size = input_size
        self.conv_glu_type = conv_glu_type
        self.time_reduction = time_reduction
        self.bias_max_distance = bias_max_distance
        self.bias_symmetric = bias_symmetric
        self.nemo_activation = nemo_activation
        self.nemo_conv_channels = nemo_conv_channels
        self.downsample_rate = downsample_rate
        self.initializer_range = initializer_range
        self.audio_token_id = audio_token_id
        self.feature_layer = feature_layer

        nemo_final_size = self.input_size
        for _ in range(int(math.log2(self.time_reduction))):
            nemo_final_size = math.floor((nemo_final_size - 1) / 2 + 1)
        self.nemo_final_size = nemo_final_size


class Phi4MultimodalConfig(PretrainedConfig):
    model_type = "phi4_multimodal"

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if config_dict.get("model_type") == "phi4mm":
            config_dict = _convert_phi4mm_config(config_dict)
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        vocab_size=200064,
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=24,
        num_key_value_heads=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        rope_parameters=None,
        bos_token_id=199999,
        eos_token_id=None,
        pad_token_id=199999,
        sliding_window=None,
        partial_rotary_factor=1.0,
        vision_config=None,
        audio_config=None,
        attention_bias=False,
        mlp_bias=False,
        lm_head_bias=False,
        vision_lora_rank=0,
        vision_lora_alpha=1,
        speech_lora_rank=0,
        speech_lora_alpha=1,
        **kwargs,
    ):
        self._phi4mm_hf_config = kwargs.pop("_phi4mm_hf_config", None)
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id if eos_token_id is not None else [199999, 200020],
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.lm_head_bias = lm_head_bias
        self.vision_lora_rank = vision_lora_rank
        self.vision_lora_alpha = vision_lora_alpha
        self.speech_lora_rank = speech_lora_rank
        self.speech_lora_alpha = speech_lora_alpha
        self._active_lora_adapter = None
        self.register_unsavable_keys("_phi4mm_hf_config")

        if isinstance(vision_config, dict):
            self.vision_config = Phi4MultimodalVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Phi4MultimodalVisionConfig()
        else:
            self.vision_config = vision_config

        if isinstance(audio_config, dict):
            self.audio_config = Phi4MultimodalAudioConfig(**audio_config)
        elif audio_config is None:
            self.audio_config = Phi4MultimodalAudioConfig()
        else:
            self.audio_config = audio_config

        # Build rope_parameters dict for compatibility with rope utils
        self.rope_parameters = rope_parameters if rope_parameters is not None else self._build_rope_parameters()

    def to_phi4mm_dict(self):
        """Return the upstream Phi-4-MM config used by Transformers/vLLM."""
        output = copy.deepcopy(self._phi4mm_hf_config) if self._phi4mm_hf_config is not None else {}

        # Training may update these values, so always take them from the live
        # PaddleFormers config instead of the originally loaded JSON.
        for key in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "resid_pdrop",
            "embd_pdrop",
            "attention_dropout",
            "hidden_act",
            "max_position_embeddings",
            "original_max_position_embeddings",
            "initializer_range",
            "rms_norm_eps",
            "use_cache",
            "tie_word_embeddings",
            "rope_theta",
            "rope_scaling",
            "partial_rotary_factor",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "sliding_window",
            "attention_bias",
            "mlp_bias",
            "lm_head_bias",
        ):
            output[key] = copy.deepcopy(getattr(self, key))

        if self.dtype is not None:
            output["torch_dtype"] = str(self.dtype).split(".")[-1]

        audio = self.audio_config
        vision = self.vision_config
        output.update(
            {
                "model_type": "phi4mm",
                "architectures": ["Phi4MMForCausalLM"],
                "auto_map": {"AutoConfig": "configuration_phi4mm.Phi4MMConfig"},
                "embd_layer": {
                    "embedding_cls": "image_audio",
                    "image_embd_layer": {
                        "crop_size": vision.crop_size,
                        "embedding_cls": "tune_image",
                        "enable_gradient_checkpointing": True,
                        "hd_transform_order": "sub_glb",
                        "image_token_compression_cls": "avg_pool_2d",
                        "projection_cls": "mlp",
                        "use_hd_transform": True,
                        "with_learnable_separator": True,
                    },
                    "audio_embd_layer": {
                        "compression_rate": audio.time_reduction,
                        "downsample_rate": audio.downsample_rate,
                        "embedding_cls": "audio",
                        "enable_gradient_checkpointing": True,
                        "projection_cls": "mlp",
                        "use_conv_downsample": False,
                        "use_qformer": False,
                    },
                },
                "img_processor": output.get("img_processor"),
                "audio_processor": {
                    "name": "cascades",
                    "config": {
                        "activation": audio.activation,
                        "activation_checkpointing": {
                            "interval": 1,
                            "module": "transformer",
                            "offload": False,
                        },
                        "attention_dim": audio.hidden_size,
                        "attention_heads": audio.num_attention_heads,
                        "batch_norm": False,
                        "bias_in_glu": True,
                        "causal": True,
                        "chunk_size": audio.chunk_size,
                        "cnn_layer_norm": True,
                        "conv_activation": audio.conv_activation,
                        "conv_glu_type": audio.conv_glu_type,
                        "depthwise_multiplier": audio.depthwise_multiplier,
                        "depthwise_seperable_out_channel": audio.depthwise_separable_out_channel,
                        "dropout_rate": audio.dropout_rate,
                        "encoder_embedding_config": {"input_size": audio.input_size},
                        "ext_pw_kernel_size": 1,
                        "ext_pw_out_channel": audio.ext_pw_out_channel,
                        "input_layer": "nemo_conv",
                        "input_size": audio.input_size,
                        "kernel_size": audio.kernel_size,
                        "left_chunk": audio.left_chunk,
                        "linear_units": audio.intermediate_size,
                        "nemo_conv_settings": {"conv_channels": audio.nemo_conv_channels},
                        "num_blocks": audio.num_blocks,
                        "relative_attention_bias_args": {
                            "t5_bias_max_distance": audio.bias_max_distance,
                            "type": "t5",
                        },
                        "time_reduction": audio.time_reduction,
                    },
                },
                "vision_lora": {
                    "r": self.vision_lora_rank,
                    "lora_alpha": self.vision_lora_alpha,
                },
                "speech_lora": {
                    "r": self.speech_lora_rank,
                    "lora_alpha": self.speech_lora_alpha,
                },
            }
        )
        return output

    def save_pretrained(self, save_directory, **kwargs):
        """Save a checkpoint that remains loadable by upstream serving tools."""
        super().save_pretrained(save_directory, **kwargs)
        output_config = os.path.join(save_directory, "config.json")
        with open(output_config, "w", encoding="utf-8") as writer:
            json.dump(self.to_phi4mm_dict(), writer, indent=2, sort_keys=True, ensure_ascii=False)
            writer.write("\n")

        source = Path(__file__).with_name("configuration_phi4mm_hf.py")
        shutil.copyfile(source, Path(save_directory) / "configuration_phi4mm.py")

    def _build_rope_parameters(self):
        rope_params = {
            "rope_theta": self.rope_theta,
            "partial_rotary_factor": self.partial_rotary_factor,
        }
        if self.rope_scaling is not None:
            rope_params.update(self.rope_scaling)
            if "rope_type" not in rope_params:
                rope_params["rope_type"] = "longrope"
            if rope_params.get("rope_type") in {"longrope", "yarn", "llama3"}:
                rope_params.setdefault("original_max_position_embeddings", self.original_max_position_embeddings)
        else:
            rope_params["rope_type"] = "default"
        return rope_params
