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

import copy
import math
import os
import re
from abc import ABC
from functools import partial
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import requests
from paddle.vision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm
from transformers import TextStreamer

from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.criterion.interface import CriterionLayer
from ...nn.lm_head import LMHead as GeneralLMHead
from ..cache_utils import Cache
from ..deepseek_v3 import DeepseekV3ForCausalLM, DeepseekV3Model
from ..deepseek_v3.modeling import DeepseekV3PretrainedModel
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..qwen2 import Qwen2Config, Qwen2Model
from .configuration import DeepseekOCR2Config
from .conversation import get_conv_template


class MlpProjector(nn.Module):
    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        if cfg["projector_type"] == "identity":
            modules = nn.Identity()

        elif cfg["projector_type"] == "linear":
            modules = nn.Linear(cfg["input_dim"], cfg["n_embed"])

        elif cfg["projector_type"] == "mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = [nn.Linear(cfg["input_dim"], cfg["n_embed"])]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg["n_embed"], cfg["n_embed"]))
            modules = nn.Sequential(*modules)

        elif cfg["projector_type"] == "normlayer_downsample_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            mlp_ratio = cfg.get("mlp_ratio", 1)
            modules = [
                nn.LayerNorm(cfg["input_dim"] * cfg["downsample_ratio"] * cfg["downsample_ratio"]),
                nn.Linear(
                    cfg["input_dim"] * cfg["downsample_ratio"] * cfg["downsample_ratio"], cfg["n_embed"] * mlp_ratio
                ),
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg["n_embed"] * mlp_ratio, cfg["n_embed"] * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg["n_embed"] * mlp_ratio, cfg["n_embed"]))
            modules = nn.Sequential(*modules)

        elif cfg["projector_type"] == "downsample_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            mlp_ratio = cfg.get("mlp_ratio", 1)
            modules = [
                nn.Linear(
                    cfg["input_dim"] * cfg["downsample_ratio"] * cfg["downsample_ratio"], cfg["n_embed"] * mlp_ratio
                )
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg["n_embed"] * mlp_ratio, cfg["n_embed"] * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg["n_embed"] * mlp_ratio, cfg["n_embed"]))
            modules = nn.Sequential(*modules)

        elif cfg["projector_type"] == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            self.high_up_proj = nn.Linear(cfg["input_dim"], cfg["n_embed"] // 2)
            self.low_up_proj = nn.Linear(cfg["input_dim"], cfg["n_embed"] // 2)

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg["n_embed"], cfg["n_embed"]))
            modules = nn.Sequential(*modules)

        elif cfg["projector_type"] == "hybrid_split_feature_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            channel_div = cfg.get("channel_div", 0.5)
            self.high_up_proj = nn.Linear(cfg["input_dim"][0], int(cfg["n_embed"] * channel_div))
            self.low_up_proj = nn.Linear(cfg["input_dim"][1], cfg["n_embed"] - int(cfg["n_embed"] * channel_div))

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg["n_embed"], cfg["n_embed"]))
            modules = nn.Sequential(*modules)

        elif cfg["projector_type"] == "low_high_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg["n_embed"] // 2, cfg["n_embed"] // 2))
            modules = nn.Sequential(*modules)
            self.high_layers = nn.Sequential(*modules)
            self.low_layers = copy.deepcopy(modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg['projector_type']}")

        if cfg.get("token_pooling", False):
            self.token_pooling_layer = nn.Linear(cfg["input_dim"] * 4, cfg["input_dim"])

        if cfg.get("conv_fusion_high_low_features", False):
            self.fusion_layer = nn.Linear(cfg["input_dim"], cfg["input_dim"])
        self.layers = modules

    def forward(self, x):
        if self.cfg.get("token_pooling", False):
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # Concatenate along the channel dimension
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)

            # Pass through the linear layer
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        if self.cfg.get("conv_fusion_high_low_features", False):
            x = self.fusion_layer(x[:, 0]) + x[:, 1]

        if self.cfg["projector_type"] == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x[0], x[1]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = paddle.concat([high_x, low_x], dim=-1)

        if self.cfg["projector_type"] == "hybrid_split_feature_mlp_gelu":
            high_x = x[..., : self.cfg["input_dim"][0]]
            low_x = x[..., self.cfg["input_dim"][0] :]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = paddle.concat([high_x, low_x], dim=-1)

        if self.cfg["projector_type"] == "low_high_split_mlp_gelu":
            high_x, low_x = x[0], x[1]
            high_x = self.high_layers(high_x)
            low_x = self.low_layers(low_x)
            x = paddle.concat([high_x, low_x], dim=-1)
            return x

        if (
            self.cfg["projector_type"] == "downsample_mlp_gelu"
            or self.cfg["projector_type"] == "normlayer_downsample_mlp_gelu"
        ):
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.cfg["downsample_ratio"]:
                pad = self.cfg["downsample_ratio"] - h % self.cfg["downsample_ratio"]
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(
                x, kernel_size=self.cfg["downsample_ratio"], stride=self.cfg["downsample_ratio"], padding=0
            )  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)

    @staticmethod
    def get_flops_per_sample(cfg):
        if cfg["projector_type"] == "linear":
            fwd = 2 * cfg["input_dim"] * cfg["n_embed"]

        elif "mlp_gelu" in cfg["projector_type"]:
            mlp_depth = cfg.get("depth", 1)
            downsample_ratio = cfg.get("downsample_ratio", 1)
            input_dim = sum(cfg["input_dim"]) if isinstance(cfg["input_dim"], list) else cfg["input_dim"]
            input_dim = input_dim * downsample_ratio * downsample_ratio
            fwd = 2 * input_dim * cfg["n_embed"] + (mlp_depth - 1) * 2 * cfg["n_embed"] * cfg["n_embed"]
        else:
            fwd = 0

        return fwd * 3


# ===================qwen2================================


class CustomQwen2Decoder(nn.Module):
    """
    Qwen2 visual encoder
    non-causal attention + causal attention
    token_type_ids: 0=non-causal (bidirectional, for image tokens), 1=causal (for query/text tokens)
    """

    def __init__(
        self,
        decoder_layer: int = 24,
        max_position_embeddings: int = 131072,
        hidden_dimension: int = 896,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
        vocab_size: int = 151936,
        attn_implementation: str = "sdpa",
        rms_norm_eps: float = 1e-06,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        vision_config=None,
    ):
        super().__init__()

        # attn_implementation check
        if attn_implementation == "flash_attention_2":
            raise ValueError(
                "CustomQwen2Decoder do not support flash_attention_2," "new attention mask needs 'sdpa' or 'eager'"
            )

        config = Qwen2Config(
            hidden_size=hidden_dimension,
            num_hidden_layers=decoder_layer,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            _attn_implementation=attn_implementation,
        )

        config.tensor_model_parallel_size = vision_config.tensor_model_parallel_size
        config.sequence_parallel = vision_config.sequence_parallel
        config.recompute_granularity = vision_config.recompute_granularity
        config.recompute_method = vision_config.recompute_method
        config.recompute_num_layers = vision_config.recompute_num_layers

        self.model = Qwen2Model(config)
        del self.model.embed_tokens

    @staticmethod
    def _create_custom_4d_mask(token_type_ids):
        """Create mixed bidirectional (image) + causal (text) boolean attention mask.

        Args:
            token_type_ids: [batch_size, seq_len], 0=image(bidirectional), 1=text(causal)

        Returns:
            mask: [batch_size, 1, seq_len, seq_len] boolean, True=attend, False=block
        """
        batch_size, sequence_length = token_type_ids.shape

        masks = []
        for b in range(batch_size):
            mask = paddle.zeros(
                [sequence_length, sequence_length],
                dtype="bool",
            )

            type_ids = token_type_ids[b]

            image_positions = paddle.nonzero(type_ids == 0).squeeze(-1)
            text_positions = paddle.nonzero(type_ids == 1).squeeze(-1)

            # image tokens: bidirectional (can attend to all image tokens)
            if image_positions.shape[0] > 0:
                mask[image_positions.unsqueeze(-1), image_positions] = True

            # text tokens: causal (attend to all image tokens + previous text tokens)
            for i in range(text_positions.shape[0]):
                text_pos = text_positions[i]
                if image_positions.shape[0] > 0:
                    mask[text_pos, image_positions] = True
                mask[text_pos, text_positions[: i + 1]] = True

            masks.append(mask)

        mask = paddle.stack(masks, axis=0).unsqueeze(1)  # [B, 1, L, L]
        return mask

    def forward(self, inputs_embeds, token_type_ids, attention_mask=None, **kwargs):
        """
        Args:
            inputs_embeds: [batch_size, seq_len, hidden_dim]
            token_type_ids: [batch_size, seq_len], 0=non-causal, 1=causal
            attention_mask: [batch_size, seq_len], optional
        """
        # Pre-compute the custom 4D boolean attention mask
        # True = attend, False = block
        custom_4d_mask = self._create_custom_4d_mask(token_type_ids)

        # Pass the 4D mask directly as attention_mask
        # PaddleFormers _prepare_decoder_attention_mask handles 4D masks directly
        # and converts boolean True->0.0, False->-inf
        return self.model(inputs_embeds=inputs_embeds, attention_mask=custom_4d_mask, **kwargs)


class Qwen2Decoder2Encoder(nn.Module):
    """
    Decoder based on Multilingual BART
    Set the initial weights and configuration with a pretrained multilingual BART model,
    and modify the detailed configurations as a Nougat decoder
    """

    def __init__(
        self,
        decoder_layer: int,
        hidden_dimension: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        max_query: int,
        config=None,
    ):
        super().__init__()

        attn_implementation = getattr(config, "_attn_implementation", "sdpa")

        self.model = CustomQwen2Decoder(
            decoder_layer=decoder_layer,
            hidden_dimension=hidden_dimension,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            attn_implementation=attn_implementation,
            vision_config=config,
        )

        self.query_768 = nn.Embedding(144, hidden_dimension)
        self.query_1024 = nn.Embedding(256, hidden_dimension)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = x.flatten(2).transpose(1, 2)

        bs, n_query, _ = x.shape

        if n_query == 144:
            param_img = self.query_768.weight
        elif n_query == 256:
            param_img = self.query_1024.weight

        batch_query_imgs = param_img.unsqueeze(0).expand(bs, -1, -1)  # (batch_size, num_queries, hidden_size)

        x_combined = paddle.cat([x, batch_query_imgs], dim=1)

        token_type_ids = paddle.cat(
            [
                paddle.zeros(bs, n_query, dtype=paddle.long),
                paddle.ones(bs, n_query, dtype=paddle.long),
            ],
            dim=1,
        )

        y = self.model(x_combined, token_type_ids)[0]

        y = y[:, n_query:, :]  # causal flow query

        return y


def build_qwen2_decoder_as_encoder(
    config=None,
    checkpoint=None,
):
    decoder_layer = getattr(config, "decoder_layer", 24)
    hidden_dimension = getattr(config, "hidden_dimension", 896)
    num_attention_heads = getattr(config, "num_attention_heads", 14)
    num_key_value_heads = getattr(config, "num_key_value_heads", 2)
    intermediate_size = getattr(config, "intermediate_size", 4864)
    max_query = getattr(config, "max_query", 400)

    decoder_as_encoder = Qwen2Decoder2Encoder(
        decoder_layer=decoder_layer,
        hidden_dimension=hidden_dimension,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_query=max_query,
        config=config,
    )

    if checkpoint is not None:
        # with open(checkpoint, "rb") as f:
        state_dict = paddle.load(checkpoint)

        decoder_as_encoder.load_state_dict(state_dict, strict=True)
        # tob
        print(checkpoint)
    return decoder_as_encoder


# =========================Sam-Vary=================================


def get_abs_pos_sam(abs_pos, tgt_size):

    dtype = abs_pos.dtype

    src_size = abs_pos.size(1)

    if src_size != tgt_size:
        old_pos_embed = abs_pos.permute(0, 3, 1, 2)
        old_pos_embed = old_pos_embed.to(paddle.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        return new_pos_embed
    else:
        return abs_pos


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(paddle.ones(num_channels))
        self.bias = nn.Parameter(paddle.zeros(num_channels))
        self.eps = eps

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        hidden_dimension: int = 896,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()

        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(paddle.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.net_2 = nn.Conv2d(out_chans, out_chans * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(out_chans * 2, hidden_dimension, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # x = x + self.pos_embed
            x = x + get_abs_pos_sam(self.pos_embed, x.size(1))

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        x2 = self.net_2(x)
        x3 = self.net_3(x2.clone())

        return x3


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()

        self._attn_implementation = "sdpa"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(paddle.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(paddle.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, H, W, _ = x.shape

        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = add_decomposed_rel_pos(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]
        if self.use_rel_pos:
            rel_h = rel_h.view(B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3))
            rel_w = rel_w.view(B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3))
            attn_bias = (rel_h + rel_w).view(B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4))
            x, _ = attention_interface(self, query=q, key=k, value=v, attention_mask=attn_bias, is_causal=False)
        else:
            x, _ = attention_interface(self, query=q, key=k, value=v, is_causal=False)

        x = x.view(B, H, W, -1)

        x = self.proj(x)

        return x


def window_partition(x: paddle.Tensor, window_size: int) -> Tuple[paddle.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: paddle.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> paddle.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: paddle.Tensor) -> paddle.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        dtype = rel_pos.dtype
        rel_pos = rel_pos.to(paddle.float32)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape([1, rel_pos.shape[0], -1]).permute(0, 2, 1),
            size=(max_rel_dist,),
            mode="linear",
        ).to(dtype)
        rel_pos_resized = rel_pos_resized.reshape([-1, max_rel_dist]).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = paddle.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = paddle.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    q: paddle.Tensor,
    rel_pos_h: paddle.Tensor,
    rel_pos_w: paddle.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> paddle.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = paddle.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = paddle.einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_h = rel_h.unsqueeze(-1)
    rel_w = rel_w.unsqueeze(-2)
    rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
    rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)

    return rel_h, rel_w


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


def build_sam_vit_b(config=None, checkpoint=None):
    encoder_embed_dim = getattr(config, "encoder_embed_dim", 768)
    encoder_depth = getattr(config, "encoder_depth", 12)
    encoder_num_heads = getattr(config, "encoder_num_heads", 12)
    encoder_global_attn_indexes = getattr(config, "encoder_global_attn_indexes", [2, 5, 8, 11])
    return _build_sam(
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        encoder_global_attn_indexes=encoder_global_attn_indexes,
        config=config,
        checkpoint=checkpoint,
    )


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    config=None,
):
    prompt_embed_dim = getattr(config, "prompt_embed_dim", 256)
    hidden_dimension = getattr(config, "hidden_dimension", 896)  # qwen2 hidden dimension
    image_size = getattr(config, "image_size", 1024)
    vit_patch_size = getattr(config, "vit_patch_size", 16)
    mlp_ratio = getattr(config, "mlp_ratio", 4)
    window_size = getattr(config, "window_size", 14)
    layer_norm_eps = getattr(config, "layer_norm_eps", 1e-6)

    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=mlp_ratio,
        norm_layer=partial(paddle.nn.LayerNorm, eps=layer_norm_eps),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=window_size,
        out_chans=prompt_embed_dim,
        hidden_dimension=hidden_dimension,
    )
    image_encoder.eval()
    if checkpoint is not None:
        # with open(checkpoint, "rb") as f:
        state_dict = paddle.load(checkpoint)
        image_encoder.load_state_dict(
            {k[30:]: v for k, v in state_dict.items() if "vision_tower_high" in k}, strict=True
        )
        print(checkpoint)
    return image_encoder


def load_image(image_path):
    try:
        if isinstance(image_path, str) and image_path.startswith(("http://", "https://")):
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        return ImageOps.exif_transpose(image)
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text):
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    # pattern1 = r'<\|ref\|>.*?<\|/ref\|>\n'
    # new_text1 = re.sub(pattern1, '', text, flags=re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):

    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, ouput_path):

    image_width, image_height = image.size

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20,)
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == "image":
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{ouput_path}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    try:
                        if label_type == "title":
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle(
                            [text_x, text_y, text_x + text_width, text_y + text_height], fill=(255, 255, 255, 30)
                        )

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_path):

    result_image = draw_bounding_boxes(image, ref_texts, output_path)

    return result_image


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=6, image_size=768, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.0] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.0] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform


def format_messages(
    conversations: List[Dict[str, str]],
    sft_format: str = "deepseek",
    system_prompt: str = "",
):
    """
    Applies the SFT template to conversation.

    Args:
        conversations (List[Dict]): A List of messages.
        sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
        system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

    Returns:
        sft_prompt (str): The formatted text.
    """

    conv = get_conv_template(sft_format)
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    sft_prompt = conv.get_prompt().strip()

    return sft_prompt


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = 0
    eos_id = 1
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]

    return t


def load_pil_images(conversations: List[Dict[str, str]]) -> List[Image.Image]:
    """

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            if isinstance(image_path, Image.Image):
                pil_img = image_path
            else:
                pil_img = load_image(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images


class BaseTransform(ABC):
    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> paddle.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError


class BasicImageTransform(BaseTransform):
    def __init__(
        self,
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std

        transform_pipelines = [transforms.ToTensor()]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)

    def __call__(self, x):
        x = self.transform(x)
        return x


class NoEOSTextStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):

        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        print(text, flush=True, end="")


class DeepseekOCR2Model(DeepseekV3Model):
    config_class = DeepseekOCR2Config

    def __init__(self, config: DeepseekOCR2Config):
        super(DeepseekOCR2Model, self).__init__(config)

        self.sam_model = build_sam_vit_b(config=config.vision_config)
        self.qwen2_model = build_qwen2_decoder_as_encoder(config=config.vision_config)

        input_dim = getattr(config.vision_config, "hidden_dimension", 896)
        n_embed = getattr(config, "hidden_size", 1280)
        self.projector = MlpProjector({"projector_type": "linear", "input_dim": input_dim, "n_embed": n_embed})
        embed_std = 1 / paddle.sqrt(paddle.tensor(n_embed, dtype=paddle.float32))
        self.view_seperator = nn.Parameter(paddle.randn(n_embed) * embed_std)

    def forward(
        self,
        input_ids: paddle.LongTensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[List[paddle.FloatTensor]] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[paddle.FloatTensor] = None,
        images_seq_mask: Optional[paddle.FloatTensor] = None,
        images_spatial_crop: Optional[paddle.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if inputs_embeds is None:
            # inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = self.get_input_embeddings()(input_ids).astype(self.get_input_embeddings().weight.dtype)

        sam_model = getattr(self, "sam_model", None)
        qwen2_model = getattr(self, "qwen2_model", None)

        if (
            sam_model is not None
            and (input_ids.shape[1] != 1 or self.training)
            and paddle.sum(images[0][1]).item() != 0
        ):

            idx = 0

            for image, crop_shape in zip(images, images_spatial_crop):
                images_in_this_batch = []

                patches = image[0]
                image_ori = image[1]
                patches = patches.astype(inputs_embeds.dtype)
                image_ori = image_ori.astype(inputs_embeds.dtype)

                with paddle.no_grad():

                    if paddle.sum(patches).item() != 0:
                        # P, C, H, W = patches.shape
                        local_features_1 = sam_model(patches)
                        local_features_2 = qwen2_model(local_features_1)
                        local_features = local_features_2
                        local_features = self.projector(local_features)

                        global_features_1 = sam_model(image_ori)
                        global_features_2 = qwen2_model(global_features_1)
                        global_features = global_features_2
                        global_features = self.projector(global_features)

                        _, hw, n_dim = global_features.shape

                        _2, hw2, n_dim2 = local_features.shape

                        global_features = global_features.view(-1, n_dim)

                        local_features = local_features.view(-1, n_dim2)

                        global_local_features = paddle.cat(
                            [local_features, global_features, self.view_seperator[None, :]], dim=0
                        )

                    else:
                        global_features_1 = sam_model(image_ori)
                        global_features_2 = qwen2_model(global_features_1)
                        global_features = global_features_2
                        global_features = self.projector(global_features)

                        _, hw, n_dim = global_features.shape

                        global_features = global_features.view(-1, n_dim)

                        global_local_features = paddle.cat([global_features, self.view_seperator[None, :]], dim=0)

                    images_in_this_batch.append(global_local_features)

                if images_in_this_batch:
                    images_in_this_batch = paddle.cat(images_in_this_batch, dim=0)
                    inputs_embeds[idx].masked_scatter_(
                        images_seq_mask[idx].astype(paddle.bool).unsqueeze(-1), images_in_this_batch
                    )

                idx += 1

        return super(DeepseekOCR2Model, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class DeepseekOCR2ForCausalLM(DeepseekV3ForCausalLM):

    config_class = DeepseekOCR2Config

    # DeepseekV3 LLM keys + SAM + Qwen2 decoder + projector + lm_head
    # Each entry is a regex pattern matched via re.search(rf"\.{key}\.weight$", name)
    # or re.fullmatch(rf"^{key}\.weight$", name). Only 2D tensors are transposed.
    transpose_weight_keys = [
        # ---- DeepseekV3 LLM ----
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate",
        "eh_proj",
        # ---- SAM vision encoder (ImageEncoderViT) ----
        r"attn\.qkv",  # sam_model.blocks.{i}.attn.qkv
        r"attn\.proj",  # sam_model.blocks.{i}.attn.proj
        "lin1",  # sam_model.blocks.{i}.mlp.lin1
        "lin2",  # sam_model.blocks.{i}.mlp.lin2
        # ---- Qwen2 decoder-as-encoder ----
        "qkv_proj",  # qwen2_model.model.model.layers.{i}.self_attn.qkv_proj
        "up_gate_proj",  # qwen2_model.model.model.layers.{i}.mlp.up_gate_proj
        # (o_proj, down_proj already covered by DeepseekV3 keys above)
        # ---- MLP projector (projector_type="linear") ----
        r"projector\.layers",  # projector.layers
        # ---- LM head (GeneralLMHead, weight shape [vocab, hidden], no transpose needed) ----
    ]

    @classmethod
    def _gen_aoa_config(cls, config: DeepseekOCR2Config):
        """Generate AOA config: maps checkpoint weight keys -> model weight keys."""
        # Start from the DeepseekV3 base LLM aoa config
        aoa_config = DeepseekV3PretrainedModel._gen_aoa_config.__func__(cls, config)
        model_prefix = "" if cls == cls.base_model_class else "model."

        # ---- SAM vision encoder (ImageEncoderViT) ----
        sam_src = "model.sam_model"
        sam_tgt = f"{model_prefix}sam_model"

        aoa_config["aoa_statements"] += [
            # patch_embed (Conv2d, no transpose)
            f"{sam_src}.patch_embed.proj.weight -> {sam_tgt}.patch_embed.proj.weight",
            f"{sam_src}.patch_embed.proj.bias -> {sam_tgt}.patch_embed.proj.bias",
            # pos_embed (Parameter)
            f"{sam_src}.pos_embed -> {sam_tgt}.pos_embed",
        ]
        # blocks.$LAYER_ID
        aoa_config["aoa_statements"] += [
            # LayerNorm
            f"{sam_src}.blocks.$LAYER_ID.norm1.weight -> {sam_tgt}.blocks.$LAYER_ID.norm1.weight",
            f"{sam_src}.blocks.$LAYER_ID.norm1.bias -> {sam_tgt}.blocks.$LAYER_ID.norm1.bias",
            f"{sam_src}.blocks.$LAYER_ID.norm2.weight -> {sam_tgt}.blocks.$LAYER_ID.norm2.weight",
            f"{sam_src}.blocks.$LAYER_ID.norm2.bias -> {sam_tgt}.blocks.$LAYER_ID.norm2.bias",
            # Attention linear layers (transpose)
            f"{sam_src}.blocks.$LAYER_ID.attn.qkv.weight^T -> {sam_tgt}.blocks.$LAYER_ID.attn.qkv.weight",
            f"{sam_src}.blocks.$LAYER_ID.attn.qkv.bias -> {sam_tgt}.blocks.$LAYER_ID.attn.qkv.bias",
            f"{sam_src}.blocks.$LAYER_ID.attn.proj.weight^T -> {sam_tgt}.blocks.$LAYER_ID.attn.proj.weight",
            f"{sam_src}.blocks.$LAYER_ID.attn.proj.bias -> {sam_tgt}.blocks.$LAYER_ID.attn.proj.bias",
            # Relative position embeddings
            f"{sam_src}.blocks.$LAYER_ID.attn.rel_pos_h -> {sam_tgt}.blocks.$LAYER_ID.attn.rel_pos_h",
            f"{sam_src}.blocks.$LAYER_ID.attn.rel_pos_w -> {sam_tgt}.blocks.$LAYER_ID.attn.rel_pos_w",
            # MLP linear layers (transpose)
            f"{sam_src}.blocks.$LAYER_ID.mlp.lin1.weight^T -> {sam_tgt}.blocks.$LAYER_ID.mlp.lin1.weight",
            f"{sam_src}.blocks.$LAYER_ID.mlp.lin1.bias -> {sam_tgt}.blocks.$LAYER_ID.mlp.lin1.bias",
            f"{sam_src}.blocks.$LAYER_ID.mlp.lin2.weight^T -> {sam_tgt}.blocks.$LAYER_ID.mlp.lin2.weight",
            f"{sam_src}.blocks.$LAYER_ID.mlp.lin2.bias -> {sam_tgt}.blocks.$LAYER_ID.mlp.lin2.bias",
        ]
        # neck (Conv2d + LayerNorm2d, no transpose for conv)
        aoa_config["aoa_statements"] += [
            f"{sam_src}.neck.0.weight -> {sam_tgt}.neck.0.weight",
            f"{sam_src}.neck.1.weight -> {sam_tgt}.neck.1.weight",
            f"{sam_src}.neck.1.bias -> {sam_tgt}.neck.1.bias",
            f"{sam_src}.neck.2.weight -> {sam_tgt}.neck.2.weight",
            f"{sam_src}.neck.3.weight -> {sam_tgt}.neck.3.weight",
            f"{sam_src}.neck.3.bias -> {sam_tgt}.neck.3.bias",
            # net_2, net_3 (Conv2d)
            f"{sam_src}.net_2.weight -> {sam_tgt}.net_2.weight",
            f"{sam_src}.net_3.weight -> {sam_tgt}.net_3.weight",
        ]

        # ---- Qwen2 decoder-as-encoder ----
        q2_src = "model.qwen2_model"
        q2_tgt = f"{model_prefix}qwen2_model"

        aoa_config["aoa_statements"] += [
            f"{q2_src}.query_768.weight -> {q2_tgt}.query_768.weight",
            f"{q2_src}.query_1024.weight -> {q2_tgt}.query_1024.weight",
            f"{q2_src}.model.model.norm.weight -> {q2_tgt}.model.model.norm.weight",
        ]
        # Qwen2 decoder
        q2_num_heads = config.vision_config.num_attention_heads
        q2_num_kv_heads = config.vision_config.num_key_value_heads
        q2_layers = f"{q2_src}.model.model.layers"
        q2_layers_tgt = f"{q2_tgt}.model.model.layers"

        # layers.$LAYER_ID
        aoa_config["aoa_statements"] += [
            # Attention o_proj
            f"{q2_layers}.$LAYER_ID.self_attn.o_proj.weight^T -> {q2_layers_tgt}.$LAYER_ID.self_attn.o_proj.weight",
            # MLP down_proj
            f"{q2_layers}.$LAYER_ID.mlp.down_proj.weight^T -> {q2_layers_tgt}.$LAYER_ID.mlp.down_proj.weight",
            # Norms
            f"{q2_layers}.$LAYER_ID.input_layernorm.weight -> {q2_layers_tgt}.$LAYER_ID.input_layernorm.weight",
            f"{q2_layers}.$LAYER_ID.post_attention_layernorm.weight -> {q2_layers_tgt}.$LAYER_ID.post_attention_layernorm.weight",
        ]
        # attention qkv (fused_qkv: separate q/k/v in checkpoint -> fused qkv_proj in model)
        aoa_config["aoa_statements"] += [
            f"{q2_layers}.$LAYER_ID.self_attn.q_proj.weight^T, {q2_layers}.$LAYER_ID.self_attn.k_proj.weight^T, {q2_layers}.$LAYER_ID.self_attn.v_proj.weight^T -> {q2_layers_tgt}.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={q2_num_heads}, num_key_value_groups={q2_num_kv_heads}",
        ]
        aoa_config["aoa_statements"] += [
            f"{q2_layers}.$LAYER_ID.self_attn.q_proj.bias, {q2_layers}.$LAYER_ID.self_attn.k_proj.bias, {q2_layers}.$LAYER_ID.self_attn.v_proj.bias -> {q2_layers_tgt}.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={q2_num_heads}, num_key_value_groups={q2_num_kv_heads}, axis=0",
        ]
        # FFN (fused_ffn: separate gate/up in checkpoint -> fused up_gate_proj in model)
        aoa_config["aoa_statements"] += [
            f"{q2_layers}.$LAYER_ID.mlp.gate_proj.weight^T, {q2_layers}.$LAYER_ID.mlp.up_proj.weight^T -> {q2_layers_tgt}.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
        ]

        # ---- MLP projector (linear: single nn.Linear) ----
        aoa_config["aoa_statements"] += [
            f"model.projector.layers.weight^T -> {model_prefix}projector.layers.weight",
            f"model.projector.layers.bias -> {model_prefix}projector.layers.bias",
        ]

        # ---- view_seperator (Parameter) ----
        aoa_config["aoa_statements"] += [
            f"model.view_seperator -> {model_prefix}view_seperator",
        ]

        # ---- lm_head ----
        aoa_config["aoa_statements"] += [
            "lm_head.weight -> lm_head.weight",
        ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: DeepseekOCR2Config):
        """Generate inverse AOA config: maps model weight keys -> checkpoint weight keys."""
        aoa_config = DeepseekV3PretrainedModel._gen_inv_aoa_config.__func__(cls, config)
        model_prefix = "" if cls == cls.base_model_class else "model."

        # ---- SAM vision encoder ----
        sam = f"{model_prefix}sam_model"

        aoa_config["aoa_statements"] += [
            f"{sam}.patch_embed.proj.weight -> model.sam_model.patch_embed.proj.weight",
            f"{sam}.patch_embed.proj.bias -> model.sam_model.patch_embed.proj.bias",
            f"{sam}.pos_embed -> model.sam_model.pos_embed",
        ]
        # blocks.$LAYER_ID
        aoa_config["aoa_statements"] += [
            # LayerNorm
            f"{sam}.blocks.$LAYER_ID.norm1.weight -> model.sam_model.blocks.$LAYER_ID.norm1.weight",
            f"{sam}.blocks.$LAYER_ID.norm1.bias -> model.sam_model.blocks.$LAYER_ID.norm1.bias",
            f"{sam}.blocks.$LAYER_ID.norm2.weight -> model.sam_model.blocks.$LAYER_ID.norm2.weight",
            f"{sam}.blocks.$LAYER_ID.norm2.bias -> model.sam_model.blocks.$LAYER_ID.norm2.bias",
            # Attention (transpose back)
            f"{sam}.blocks.$LAYER_ID.attn.qkv.weight^T -> model.sam_model.blocks.$LAYER_ID.attn.qkv.weight",
            f"{sam}.blocks.$LAYER_ID.attn.qkv.bias -> model.sam_model.blocks.$LAYER_ID.attn.qkv.bias",
            f"{sam}.blocks.$LAYER_ID.attn.proj.weight^T -> model.sam_model.blocks.$LAYER_ID.attn.proj.weight",
            f"{sam}.blocks.$LAYER_ID.attn.proj.bias -> model.sam_model.blocks.$LAYER_ID.attn.proj.bias",
            # Relative position embeddings
            f"{sam}.blocks.$LAYER_ID.attn.rel_pos_h -> model.sam_model.blocks.$LAYER_ID.attn.rel_pos_h",
            f"{sam}.blocks.$LAYER_ID.attn.rel_pos_w -> model.sam_model.blocks.$LAYER_ID.attn.rel_pos_w",
            # MLP (transpose back)
            f"{sam}.blocks.$LAYER_ID.mlp.lin1.weight^T -> model.sam_model.blocks.$LAYER_ID.mlp.lin1.weight",
            f"{sam}.blocks.$LAYER_ID.mlp.lin1.bias -> model.sam_model.blocks.$LAYER_ID.mlp.lin1.bias",
            f"{sam}.blocks.$LAYER_ID.mlp.lin2.weight^T -> model.sam_model.blocks.$LAYER_ID.mlp.lin2.weight",
            f"{sam}.blocks.$LAYER_ID.mlp.lin2.bias -> model.sam_model.blocks.$LAYER_ID.mlp.lin2.bias",
        ]
        # neck + net
        aoa_config["aoa_statements"] += [
            f"{sam}.neck.0.weight -> model.sam_model.neck.0.weight",
            f"{sam}.neck.1.weight -> model.sam_model.neck.1.weight",
            f"{sam}.neck.1.bias -> model.sam_model.neck.1.bias",
            f"{sam}.neck.2.weight -> model.sam_model.neck.2.weight",
            f"{sam}.neck.3.weight -> model.sam_model.neck.3.weight",
            f"{sam}.neck.3.bias -> model.sam_model.neck.3.bias",
            f"{sam}.net_2.weight -> model.sam_model.net_2.weight",
            f"{sam}.net_3.weight -> model.sam_model.net_3.weight",
        ]

        # ---- Qwen2 decoder-as-encoder ----
        q2 = f"{model_prefix}qwen2_model"

        aoa_config["aoa_statements"] += [
            f"{q2}.query_768.weight -> model.qwen2_model.query_768.weight",
            f"{q2}.query_1024.weight -> model.qwen2_model.query_1024.weight",
            f"{q2}.model.model.norm.weight -> model.qwen2_model.model.model.norm.weight",
        ]
        # Qwen2 decoder
        q2_num_heads = config.vision_config.num_attention_heads
        q2_num_kv_heads = config.vision_config.num_key_value_heads
        q2_layers = f"{q2}.model.model.layers"
        q2_ckpt_layers = "model.qwen2_model.model.model.layers"

        # layers.$LAYER_ID
        aoa_config["aoa_statements"] += [
            # Attention o_proj
            f"{q2_layers}.$LAYER_ID.self_attn.o_proj.weight^T -> {q2_ckpt_layers}.$LAYER_ID.self_attn.o_proj.weight",
            # MLP down_proj
            f"{q2_layers}.$LAYER_ID.mlp.down_proj.weight^T -> {q2_ckpt_layers}.$LAYER_ID.mlp.down_proj.weight",
            # Norms
            f"{q2_layers}.$LAYER_ID.input_layernorm.weight -> {q2_ckpt_layers}.$LAYER_ID.input_layernorm.weight",
            f"{q2_layers}.$LAYER_ID.post_attention_layernorm.weight -> {q2_ckpt_layers}.$LAYER_ID.post_attention_layernorm.weight",
        ]
        # attention qkv unfuse (fused qkv_proj -> separate q/k/v in checkpoint)
        aoa_config["aoa_statements"] += [
            f"{q2_layers}.$LAYER_ID.self_attn.qkv_proj.weight -> {q2_ckpt_layers}.$LAYER_ID.self_attn.q_proj.weight, {q2_ckpt_layers}.$LAYER_ID.self_attn.k_proj.weight, {q2_ckpt_layers}.$LAYER_ID.self_attn.v_proj.weight, fused_qkv, num_heads={q2_num_heads}, num_key_value_groups={q2_num_kv_heads}",
        ]
        for lid in range(config.vision_config.decoder_layer):
            aoa_config["aoa_statements"] += [
                f"{q2_ckpt_layers}.{lid}.self_attn.q_proj.weight^T -> {q2_ckpt_layers}.{lid}.self_attn.q_proj.weight",
                f"{q2_ckpt_layers}.{lid}.self_attn.k_proj.weight^T -> {q2_ckpt_layers}.{lid}.self_attn.k_proj.weight",
                f"{q2_ckpt_layers}.{lid}.self_attn.v_proj.weight^T -> {q2_ckpt_layers}.{lid}.self_attn.v_proj.weight",
            ]
        aoa_config["aoa_statements"] += [
            f"{q2_layers}.$LAYER_ID.self_attn.qkv_proj.bias -> {q2_ckpt_layers}.$LAYER_ID.self_attn.q_proj.bias, {q2_ckpt_layers}.$LAYER_ID.self_attn.k_proj.bias, {q2_ckpt_layers}.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={q2_num_heads}, num_key_value_groups={q2_num_kv_heads}, axis=0",
        ]
        # FFN unfuse (fused up_gate_proj -> separate gate/up in checkpoint)
        aoa_config["aoa_statements"] += [
            f"{q2_layers}.$LAYER_ID.mlp.up_gate_proj.weight -> {q2_ckpt_layers}.$LAYER_ID.mlp.gate_proj.weight, {q2_ckpt_layers}.$LAYER_ID.mlp.up_proj.weight, fused_ffn",
        ]
        decoder_layer_count = config.vision_config.decoder_layer
        for lid in range(decoder_layer_count):
            aoa_config["aoa_statements"] += [
                f"{q2_ckpt_layers}.{lid}.mlp.gate_proj.weight^T -> {q2_ckpt_layers}.{lid}.mlp.gate_proj.weight",
                f"{q2_ckpt_layers}.{lid}.mlp.up_proj.weight^T -> {q2_ckpt_layers}.{lid}.mlp.up_proj.weight",
            ]

        # ---- MLP projector ----
        aoa_config["aoa_statements"] += [
            f"{model_prefix}projector.layers.weight^T -> model.projector.layers.weight",
            f"{model_prefix}projector.layers.bias -> model.projector.layers.bias",
        ]

        # ---- view_seperator ----
        aoa_config["aoa_statements"] += [
            f"{model_prefix}view_seperator -> model.view_seperator",
        ]

        # ---- lm_head ----
        aoa_config["aoa_statements"] += [
            "lm_head.weight -> lm_head.weight",
        ]

        return aoa_config

    def __init__(self, config):
        super(DeepseekOCR2ForCausalLM, self).__init__(config)
        self.model = DeepseekOCR2Model(config)

        self.vocab_size = config.vocab_size

        self.lm_head = GeneralLMHead(config)

        self.criterion = CriterionLayer(config)

        # Initialize weights and apply final processing
        # self._post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: paddle.LongTensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.LongTensor] = None,
        past_key_values: Optional[List[paddle.FloatTensor]] = None,
        inputs_embeds: Optional[paddle.FloatTensor] = None,
        labels: Optional[paddle.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[paddle.FloatTensor] = None,
        images_seq_mask: Optional[paddle.FloatTensor] = None,
        images_spatial_crop: Optional[paddle.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:

            loss_mask = labels != -100
            loss, _ = self.criterion(logits, labels, loss_mask)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = cache_length
                max_cache_length = None
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.astype("int64").cumsum(-1) - 1
            position_ids = paddle.where(attention_mask == 0, paddle.to_tensor(1, dtype="int64"), position_ids)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        # cache_position = paddle.arange(past_length, past_length + position_ids.shape[-1])

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "images_seq_mask": kwargs.get("images_seq_mask", None),
                "images_spatial_crop": kwargs.get("images_spatial_crop", None),
            }
        )
        return model_inputs

    def disable_paddle_init(self):
        """
        Disable the redundant paddle default initialization to accelerate model creation.
        """
        import paddle

        setattr(paddle.nn.Linear, "reset_parameters", lambda self: None)
        setattr(paddle.nn.LayerNorm, "reset_parameters", lambda self: None)

    def _build_conversation(self, prompt, image_file=""):
        """Build the conversation list from prompt and optional image_file."""
        if prompt and image_file:
            return [
                {
                    "role": "<|User|>",
                    # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                    "content": f"{prompt}",
                    # "content": "君不见黄河之水天上来的下一句是什么？",
                    # "content": "<image>\nFree OCR. ",
                    # "content": "<image>\nParse the figure. ",
                    # "content": "<image>\nExtract the text in the image. ",
                    "images": [f"{image_file}"],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        elif prompt:
            return [
                {
                    "role": "<|User|>",
                    # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                    "content": f"{prompt}",
                    # "content": "君不见黄河之水天上来的下一句是什么？",
                    # "content": "<image>\nFree OCR. ",
                    # "content": "<image>\nParse the figure. ",
                    # "content": "<image>\nExtract the text in the image. ",
                    # "images": [f'{image_file}'],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        else:
            assert False, "prompt is none!"

    def prepare_inputs_for_infer(self, tokenizer, conversation, base_size=1024, image_size=640, crop_mode=True):
        """
        Tokenize and build image tensors from a conversation list.

        Returns a dict with keys:
            input_ids, images_crop, images_ori, images_seq_mask,
            images_spatial_crop, valid_img_tokens, image_draw, w, h
        """
        prompt = format_messages(conversations=conversation, sft_format="plain", system_prompt="")

        patch_size = 16
        downsample_ratio = 4
        images = load_pil_images(conversation)

        valid_img_tokens = 0
        ratio = 1

        image_draw = images[0].copy()

        w, h = image_draw.size
        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))

        image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        images_seq_mask = []

        image_token = "<image>"
        image_token_id = 128815
        text_splits = prompt.split(image_token)

        images_list, images_crop_list, images_seq_mask = [], [], []
        tokenized_str = []
        images_spatial_crop = []
        for text_sep, image in zip(text_splits, images):

            tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            if crop_mode:

                if image.size[0] <= 768 and image.size[1] <= 768:
                    crop_ratio = [1, 1]

                else:
                    if crop_mode:
                        # best_width, best_height = select_best_resolution(image.size, self.candidate_resolutions)
                        images_crop_raw, crop_ratio = dynamic_preprocess(image)
                    else:
                        # best_width, best_height = self.image_size, self.image_size
                        crop_ratio = [1, 1]

                """process the global view"""
                # image = image.resize((base_size, base_size))
                global_view = ImageOps.pad(
                    image, (base_size, base_size), color=tuple(int(x * 255) for x in image_transform.mean)
                )

                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                # elif base_size == 640:
                #     valid_img_tokens += int(100 * ratio)

                images_list.append(image_transform(global_view))

                width_crop_num, height_crop_num = crop_ratio

                images_spatial_crop.append([width_crop_num, height_crop_num])

                if width_crop_num > 1 or height_crop_num > 1:
                    """process the local views"""

                    for i in range(len(images_crop_raw)):
                        images_crop_list.append(image_transform(images_crop_raw[i]))

                if image_size == 768:
                    valid_img_tokens += len(images_crop_list) * 144

                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

                """add image tokens"""

                tokenized_image = ([image_token_id] * num_queries_base) * num_queries_base
                tokenized_image += [image_token_id]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([image_token_id] * (num_queries * width_crop_num)) * (
                        num_queries * height_crop_num
                    )
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                # num_image_tokens.append(len(tokenized_image))

            else:
                # best_width, best_height = self.image_size, self.image_size
                # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

                """process the global view"""
                if image_size <= 768:
                    print("directly resize")
                    image = image.resize((image_size, image_size))
                # else:
                global_view = ImageOps.pad(
                    image, (image_size, image_size), color=tuple(int(x * 255) for x in image_transform.mean)
                )
                images_list.append(image_transform(global_view))

                if base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                elif base_size == 640:
                    valid_img_tokens += int(100 * 1)
                elif base_size == 512:
                    valid_img_tokens += int(64 * 1)
                elif base_size == 768:
                    valid_img_tokens += int(144 * 1)

                width_crop_num, height_crop_num = 1, 1

                images_spatial_crop.append([width_crop_num, height_crop_num])

                """add image tokens"""
                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)

                tokenized_image = ([image_token_id] * num_queries) * num_queries
                tokenized_image += [image_token_id]
                # tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                #             num_queries * height_crop_num)
                tokenized_str += tokenized_image
                images_seq_mask += [True] * len(tokenized_image)
                # num_image_tokens.append(len(tokenized_image))

        """process the last text split"""
        tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos tokens"""
        bos_id = 0
        tokenized_str = [bos_id] + tokenized_str
        images_seq_mask = [False] + images_seq_mask

        input_ids = paddle.LongTensor(tokenized_str)

        images_seq_mask = paddle.tensor(images_seq_mask, dtype=paddle.bool)

        if len(images_list) == 0:
            images_ori = paddle.zeros((1, 3, image_size, image_size))
            images_spatial_crop = paddle.zeros((1, 2), dtype=paddle.long)
            images_crop = paddle.zeros((1, 3, base_size, base_size))

        else:
            images_ori = paddle.stack(images_list, dim=0)
            images_spatial_crop = paddle.tensor(images_spatial_crop, dtype=paddle.long)
            if images_crop_list:
                images_crop = paddle.stack(images_crop_list, dim=0)
            else:
                images_crop = paddle.zeros((1, 3, base_size, base_size))

        return dict(
            input_ids=input_ids,
            images_crop=images_crop,
            images_ori=images_ori,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            valid_img_tokens=valid_img_tokens,
            image_draw=image_draw,
            w=w,
            h=h,
        )

    def infer(
        self,
        tokenizer,
        prompt="",
        image_file="",
        output_path="",
        base_size=1024,
        image_size=640,
        crop_mode=True,
        test_compress=False,
        save_results=False,
        eval_mode=False,
    ):
        self.disable_paddle_init()

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f"{output_path}/images", exist_ok=True)

        conversation = self._build_conversation(prompt, image_file)
        inputs = self.prepare_inputs_for_infer(
            tokenizer,
            conversation,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
        )
        input_ids = inputs["input_ids"]
        images_crop = inputs["images_crop"]
        images_ori = inputs["images_ori"]
        images_seq_mask = inputs["images_seq_mask"]
        images_spatial_crop = inputs["images_spatial_crop"]
        valid_img_tokens = inputs["valid_img_tokens"]
        image_draw = inputs["image_draw"]
        w = inputs["w"]
        h = inputs["h"]

        if not eval_mode:
            streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            with paddle.autocast("cuda", dtype=paddle.bfloat16):
                with paddle.no_grad():
                    output_ids = self.generate(
                        input_ids.unsqueeze(0),
                        images=[(images_crop, images_ori)],
                        images_seq_mask=images_seq_mask.unsqueeze(0),
                        images_spatial_crop=images_spatial_crop,
                        # do_sample=False,
                        # num_beams = 1,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        streamer=streamer,
                        max_new_tokens=8192,
                        no_repeat_ngram_size=20,
                        use_cache=True,
                    )

        else:
            with paddle.autocast("cuda", dtype=paddle.bfloat16):
                with paddle.no_grad():
                    output_ids = self.generate(
                        input_ids.unsqueeze(0),
                        images=[(images_crop, images_ori)],
                        images_seq_mask=images_seq_mask.unsqueeze(0),
                        images_spatial_crop=images_spatial_crop,
                        # do_sample=False,
                        # num_beams = 1,
                        temperature=0.0,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=8192,
                        no_repeat_ngram_size=35,
                        use_cache=True,
                    )

        if "<image>" in conversation[0]["content"] and eval_mode:
            # outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).shape[1]:])
            outputs = tokenizer.decode(output_ids[0])
            stop_str = "<｜end▁of▁sentence｜>"
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            # re_match
            outputs = outputs.strip()

            return outputs

        if "<image>" in conversation[0]["content"] and test_compress:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).shape[1] :])
            pure_texts_outputs_token_length = len(text_encode(tokenizer, outputs, bos=False, eos=False))
            print("=" * 50)
            print("image size: ", (w, h))
            print("valid image tokens: ", int(valid_img_tokens))
            print("output texts tokens (valid): ", pure_texts_outputs_token_length)
            print("compression ratio: ", round(pure_texts_outputs_token_length / valid_img_tokens, 2))
            print("=" * 50)

        if "<image>" in conversation[0]["content"] and save_results:
            outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).shape[1] :])
            stop_str = "<｜end▁of▁sentence｜>"

            print("=" * 15 + "save results:" + "=" * 15)

            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            matches_ref, matches_images, mathes_other = re_match(outputs)
            result = process_image_with_refs(image_draw, matches_ref, output_path)

            for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
                outputs = outputs.replace(a_match_image, "![](images/" + str(idx) + ".jpg)\n")

            for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
                outputs = outputs.replace(a_match_other, "").replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

            with open(f"{output_path}/result.mmd", "w", encoding="utf-8") as afile:
                afile.write(outputs)

            if "line_type" in outputs:
                import matplotlib.pyplot as plt

                lines = eval(outputs)["Line"]["line"]

                line_type = eval(outputs)["Line"]["line_type"]

                endpoints = eval(outputs)["Line"]["line_endpoint"]

                fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)

                for idx, line in enumerate(lines):
                    try:
                        p0 = eval(line.split(" -- ")[0])
                        p1 = eval(line.split(" -- ")[-1])

                        if line_type[idx] == "--":
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color="k")
                        else:
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color="k")

                        ax.scatter(p0[0], p0[1], s=5, color="k")
                        ax.scatter(p1[0], p1[1], s=5, color="k")
                    except:
                        pass

                for endpoint in endpoints:

                    label = endpoint.split(": ")[0]
                    (x, y) = eval(endpoint.split(": ")[1])
                    ax.annotate(
                        label, (x, y), xytext=(1, 1), textcoords="offset points", fontsize=5, fontweight="light"
                    )

                plt.savefig(f"{output_path}/geo.jpg")
                plt.close()

            result.save(f"{output_path}/result_with_boxes.jpg")


class DeepseekOCR2ForConditionalGeneration(DeepseekOCR2ForCausalLM):
    pass


__all__ = ["DeepseekOCR2Model", "DeepseekOCR2ForCausalLM", "DeepseekOCR2ForConditionalGeneration"]
