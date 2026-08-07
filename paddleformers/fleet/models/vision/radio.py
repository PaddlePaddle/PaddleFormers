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

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddleformers.fleet.config_logger import (
    has_config_logger_enabled,
    log_config_to_disk,
)
from paddleformers.fleet.models.common.vision_layer.vision_layer import VisionLayer
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.layers import ColumnParallelLinear
from paddleformers.fleet.transformer.enums import ModelType
from paddleformers.fleet.transformer.transformer_block import TransformerBlock
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

# RADIO reference code: https://github.com/NVlabs/RADIO

try:
    from einops import rearrange

    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False


class RADIOViTModel(VisionLayer):
    """RADIO ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (LayerSpec): Specifies module to use for transformer layers.
        ln_pre_impl (LayerSpec or type): Specifies the layer norm type to use for ln_pre.
        ln_post_impl (LayerSpec or type): Specifies the layer norm type to use for ln_post.
        use_mask_token (bool, optional): Whether to use RADIO mask token. Default to False.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
        max_img_h (int): Max input image height.
        max_img_w (int): Max input image width.
        pos_dropout (int): Positional encoding dropout value. Defaults to 0.
        has_cpe: (bool): Whether to use conditional positional encoding. Defaults to True.
        embedder_bias: (bool): Bias in embedder linear. Defaults to False.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: LayerSpec,
        ln_pre_impl: LayerSpec | type = None,
        ln_post_impl: LayerSpec | type = None,
        use_mask_token: bool = False,
        add_class_token: bool = True,
        class_token_len: int = 8,
        patch_dim: int = 16,
        img_h: int = 224,
        img_w: int = 224,
        max_img_h: int = 2048,
        max_img_w: int = 2048,
        pos_dropout: int = 0,
        has_cpe: bool = True,
        embedder_bias: bool = False,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(config=transformer_config)

        if has_config_logger_enabled(transformer_config):
            log_config_to_disk(
                transformer_config, locals(), prefix=type(self).__name__
            )

        self.class_token_len = class_token_len
        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0

        self.input_dims = (img_h // patch_dim, img_w // patch_dim)

        # used for positional embedding
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_num_rows = max_img_h // patch_dim
        self.max_num_cols = max_img_w // patch_dim
        self.max_num_patches = self.max_num_rows * self.max_num_cols

        # TODO: are we actually going to use this anywhere?
        self.use_mask_token = use_mask_token
        if self.use_mask_token:
            self.mask_token = nn.Parameter(
                paddle.zeros(1, self.visual_hidden_size)
            )

        self.add_class_token = add_class_token
        self.class_token_len = class_token_len
        if self.add_class_token:
            self.class_token = nn.Parameter(
                paddle.randn(
                    self.class_token_len,
                    self.visual_hidden_size,
                    dtype=transformer_config.params_dtype,
                )
            )

        self.seq_length = (img_h // self.patch_dim) * (
            img_w // self.patch_dim
        ) + (self.class_token_len if self.add_class_token else 0)

        pos_scale = self.visual_hidden_size**-0.5
        self.position_embeddings = nn.Parameter(
            paddle.randn(
                1,
                self.max_num_patches,
                self.visual_hidden_size,
                dtype=transformer_config.params_dtype,
            )
            * pos_scale
        )
        self.pos_dropout = pos_dropout
        self.has_cpe = has_cpe

        # Using non-TE version so we can force gather_output
        self.embedder = ColumnParallelLinear(
            input_size=3 * self.patch_dim * self.patch_dim,
            output_size=self.visual_hidden_size,
            bias=embedder_bias,
            config=transformer_config,
            gather_output=True,
            init_method=lambda tensor: paddle.normal(
                mean=0.0, std=1.0, shape=tensor.shape
            ),
        )

        self.model_type = ModelType.encoder_or_decoder

        self.ln_pre = None
        self.ln_post = None
        self.pg_collection = pg_collection
        self.vp_stage = vp_stage
        if ln_pre_impl is not None:
            self.ln_pre = build_spec_layer(
                ln_pre_impl,
                config=transformer_config,
                hidden_size=self.visual_hidden_size,
                eps=transformer_config.rms_norm_eps,
            )
        if ln_post_impl is not None:
            self.ln_post = build_spec_layer(
                ln_post_impl,
                config=transformer_config,
                hidden_size=self.visual_hidden_size,
                eps=transformer_config.rms_norm_eps,
            )

        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )

    def set_input_tensor(self, input_tensor: paddle.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self, x: paddle.Tensor, attention_mask: paddle.Tensor | None = None
    ) -> paddle.Tensor:
        """Forward function of the RADIO ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (paddle.Tensor): input data of shape [batch, img_h, img_w]
            attention_mask (paddle.Tensor with dtype=bool): Attention mask to use.

        Returns:
            x (paddle.Tensor): output after final transformer block of shape [b, s, h].
        """

        if not HAVE_EINOPS:
            raise ImportError(
                "einops is required for RADIOViTModel, please install it with `pip install einops`"
            )

        input_size = x.shape[2:]
        py = x.shape[-2] // self.patch_dim
        px = x.shape[-1] // self.patch_dim
        x = rearrange(
            x,
            "b c (py yy) (px xx) -> b (py px) (c yy xx)",
            py=py,
            yy=self.patch_dim,
            px=px,
            xx=self.patch_dim,
        )
        x, _ = self.embedder(x)  # [batch, seq_length, hidden_size]

        x, _ = self.apply_pos_enc(x, input_size=input_size)

        if self.add_class_token:
            class_token = self.class_token.expand(
                x.shape[0], -1, -1
            )  # [batch, class_token_len, hidden_size]

            x = paddle.concat(
                [class_token, x], dim=1
            )  # [batch, seq_length + class_token_len, hidden_size]

        assert x.shape[1] == self.seq_length, (
            f"{x.shape[1]} != {self.seq_length}"
        )

        if self.ln_pre:
            x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [b, s, h] -> [s, b, h]
        x = x.contiguous()

        x = self.decoder(x, attention_mask=attention_mask)

        x = x.permute(1, 0, 2)  # [s, b, h] -> [b, s, h]
        x = x.contiguous()

        if self.ln_post:
            x = self.ln_post(x)

        return x

    def apply_pos_enc(
        self,
        patches: paddle.Tensor,
        patch_idxs: paddle.Tensor | None = None,
        input_size: tuple[int, int] | None = None,
    ) -> paddle.Tensor:
        """Apply positional encoding to patches"""
        pos_enc = self.get_pos_enc(patches.shape[0], patch_idxs, input_size)

        if self.training and self.pos_dropout > 0:
            keeps = (
                paddle.rand(
                    patches.shape[0],
                    1,
                    1,
                    dtype=pos_enc.dtype,
                    device=pos_enc.device,
                )
                > self.pos_dropout
            )
            pos_enc_drop = paddle.where(keeps, pos_enc, 0)
        else:
            pos_enc_drop = pos_enc

        return patches + pos_enc_drop, pos_enc

    def get_pos_enc(
        self,
        batch_size: int,
        patch_idxs: paddle.Tensor | None = None,
        input_size: tuple[int, int] | None = None,
    ) -> paddle.Tensor:
        """Get positional encoding for certain input size"""
        if input_size is None:
            input_dims = self.input_dims
        else:
            input_dims = tuple(d // self.patch_dim for d in input_size)

        pos_embed = self._get_pos_embeddings(batch_size, input_dims)

        if patch_idxs is None:
            return pos_embed

        exp_patch_idxs = patch_idxs.unsqueeze(-1).expand(
            -1, -1, pos_embed.shape[-1]
        )

        pos_embed = paddle.gather(
            pos_embed.expand(patch_idxs.shape[0], -1, -1),
            dim=1,
            index=exp_patch_idxs,
        )
        return pos_embed

    def _get_pos_embeddings(self, batch_size: int, input_dims: tuple[int, int]):
        """Get RADIO absolute positional embeddings"""
        if (self.max_num_rows, self.max_num_cols) == input_dims:
            return self.position_embeddings

        pos_embed = self.position_embeddings.reshape(
            1, self.max_num_rows, self.max_num_cols, -1
        ).permute(0, 3, 1, 2)

        def window_select(pos_embed):
            if input_dims[0] < pos_embed.shape[-2]:
                pos_embed = pos_embed[..., : input_dims[0], :]
            if input_dims[1] < pos_embed.shape[-1]:
                pos_embed = pos_embed[..., :, : input_dims[1]]
            return pos_embed

        if self.has_cpe:
            if self.training:
                min_scale = math.sqrt(0.1)
                scale = (
                    paddle.rand(batch_size, 1, 1, device=pos_embed.device)
                    * (1 - min_scale)
                    + min_scale
                )
                aspect_min = math.log(3 / 4)
                aspect_max = -aspect_min
                aspect = paddle.exp(
                    paddle.rand(batch_size, 1, 1, device=pos_embed.device)
                    * (aspect_max - aspect_min)
                    + aspect_min
                )

                scale_x = scale * aspect
                scale_y = scale * (1 / aspect)
                scale_xy = paddle.stack([scale_x, scale_y], dim=-1).clamp_(0, 1)

                pos_xy = paddle.rand(
                    batch_size, 1, 1, 2, device=pos_embed.device
                ) * (1 - scale_xy)

                lin_x = paddle.linspace(
                    0, 1, steps=input_dims[1], device=pos_embed.device
                )[None, None].expand(batch_size, input_dims[0], -1)
                lin_y = paddle.linspace(
                    0, 1, steps=input_dims[0], device=pos_embed.device
                )[None, :, None].expand(batch_size, -1, input_dims[1])

                lin_xy = paddle.stack([lin_x, lin_y], dim=-1)

                grid_xy = lin_xy * scale_xy + pos_xy

                # Convert to [-1, 1] range
                grid_xy.mul_(2).sub_(1)

                pos_embed = F.grid_sample(
                    pos_embed.float().expand(batch_size, -1, -1, -1),
                    grid=grid_xy,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).to(pos_embed.dtype)
            else:
                max_dim = max(input_dims)
                pos_embed = F.interpolate(
                    pos_embed.float(),
                    size=(max_dim, max_dim),
                    align_corners=True,
                    mode="bilinear",
                ).to(pos_embed.dtype)

                pos_embed = window_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(
                pos_embed.float(),
                size=input_dims,
                align_corners=True,
                mode="bilinear",
            ).to(pos_embed.dtype)

        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        return pos_embed
