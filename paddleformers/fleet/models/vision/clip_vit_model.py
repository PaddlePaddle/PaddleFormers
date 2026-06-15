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


import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddleformers.fleet.config_logger import (
    has_config_logger_enabled,
    log_config_to_disk,
)
from paddleformers.fleet.models.common.vision_layer.vision_layer import VisionLayer
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.transformer.enums import ModelType
from paddleformers.fleet.transformer.transformer_block import TransformerBlock
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

NORM_IMPL = paddle.nn.LayerNorm


# Note: This is under development and is missing features like position embedding interpolation.
class CLIPViTModel(VisionLayer):
    """CLIP ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (LayerSpec): Specifies module to use for transformer layers.
        ln_pre_impl (LayerSpec or type): Specifies the layer norm type to use for ln_pre.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
        pg_collection (ProcessGroupCollection): Model communication process groups
        vp_stage (int): Virtual pipeline stage
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: LayerSpec,
        ln_pre_impl: LayerSpec | type = NORM_IMPL,
        ln_post_impl: LayerSpec | type = NORM_IMPL,
        add_class_token: bool = True,
        class_token_len: int = 1,
        patch_dim: int = 14,
        img_h: int = 336,
        img_w: int = 336,
        model_subtype: str = "clip",
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
    ) -> None:
        error_msg = f"CLIPViTModel model subtype {model_subtype} is not supported."
        assert model_subtype in [
            "clip",
            "siglip",
            "internvit",
            "internvit300M",
        ], error_msg

        if model_subtype == "siglip":
            assert class_token_len == 0, "SigLIP does not support class tokens."
            assert not add_class_token, "SigLIP does not support class tokens."

        super().__init__(config=transformer_config)

        if has_config_logger_enabled(transformer_config):
            log_config_to_disk(transformer_config, locals(), prefix=type(self).__name__)

        self.class_token_len = class_token_len
        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.add_class_token = add_class_token
        self.class_token_len = class_token_len

        self.seq_length = self.num_patches + (self.class_token_len if self.add_class_token else 0)

        self.ln_pre = None
        self.ln_post = None
        self.pg_collection = pg_collection
        self.vp_stage = vp_stage
        kwargs = {}
        if isinstance(ln_pre_impl, LayerSpec):
            kwargs["config"] = transformer_config
        if model_subtype == "clip":
            self.ln_pre = build_spec_layer(
                ln_pre_impl,
                normalized_shape=self.visual_hidden_size,
                epsilon=transformer_config.rms_norm_eps,
                **kwargs,
            )
            conv_bias = False
            padding = 0
        elif model_subtype == "siglip":
            self.ln_post = build_spec_layer(
                ln_post_impl,
                normalized_shape=self.visual_hidden_size,
                epsilon=transformer_config.rms_norm_eps,
                **kwargs,
            )
            conv_bias = True
            padding = "valid"
        elif model_subtype.startswith("internvit"):
            conv_bias = True
            padding = 0
        else:
            raise ValueError(f"unsupported vision model type {model_subtype}")

        self.conv1 = paddle.nn.Conv2d(
            in_channels=3,
            out_channels=self.visual_hidden_size,
            kernel_size=self.patch_dim,
            stride=self.patch_dim,
            bias=conv_bias,
            padding=padding,
        )

        self.position_ids = paddle.arange(self.seq_length).expand(1, -1).cuda()

        self.position_embeddings = paddle.nn.Embedding(
            self.seq_length,
            self.visual_hidden_size,
            dtype=transformer_config.params_dtype,
        )

        self.add_class_token = add_class_token
        if self.add_class_token:
            self.class_token = paddle.nn.Parameter(
                paddle.randn(
                    1,
                    self.class_token_len,
                    self.visual_hidden_size,
                    dtype=transformer_config.params_dtype,
                )
            )

        self.model_type = ModelType.encoder_or_decoder

        # Transformer layers.
        # TODO: Make pre_process and post_process configurable.
        # NOTE: a final layer norm and/or linear layer in some implementations are omitted here.
        # They can be added separately where needed.
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

    def forward(self, x: paddle.Tensor, attention_mask: paddle.Tensor | None = None) -> paddle.Tensor:
        """Forward function of the CLIP ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (paddle.Tensor): input data of shape [batch, img_h, img_w]
            attention_mask (paddle.Tensor with dtype=bool): Attention mask to use.

        Returns:
            x (paddle.Tensor): output after final transformer block of shape [b, s, h].
        """
        x = self.conv1(x)  # shape = [batch, hidden_size, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, hidden_size, grid ** 2]
        x = x.permute(0, 2, 1)  # [batch, grid ** 2, hidden_size]

        if self.add_class_token:
            class_token = self.class_token.expand(x.shape[0], -1, -1)  # [batch, class_token_len, hidden_size]
            x = paddle.concat([class_token, x], dim=1)  # [batch, grid ** 2 + class_token_len, hidden_size]

        assert x.shape[1] == self.seq_length, f"{x.shape[1]} != {self.seq_length}"
        x = x + self.position_embeddings(self.position_ids)
        if self.ln_pre:
            x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # [b, s, h] -> [s, b, h]
        # `permute` can make the tensor non-contiguous, breaking pipelining.
        x = x.contiguous()

        x = self.decoder(x, attention_mask)
        x = x.permute(1, 0, 2)  # [s, b, h] -> [b, s, h]
        x = x.contiguous()
        if self.ln_post:
            x = self.ln_post(x)
        return x


def get_num_image_embeddings(
    img_h,
    img_w,
    patch_dim,
    vision_model_type,
    disable_vision_class_token,
    class_token_len,
    pixel_shuffle,
    use_tile_tags=False,
    max_num_tiles=0,
    tokenizer_type=None,
):
    """Get the number of image embeddings per image tile."""
    if vision_model_type == "siglip":
        keep_class_token = False
    elif vision_model_type in ("clip", "internvit", "internvit300M"):
        keep_class_token = not disable_vision_class_token
    elif vision_model_type.startswith("radio"):
        keep_class_token = not disable_vision_class_token
    elif vision_model_type == "cradio-g":
        class_token_len = 8
        keep_class_token = not disable_vision_class_token
    elif vision_model_type.startswith("hf://"):
        from paddleformers.fleet.models.huggingface.module import get_hf_model_type

        model_type = get_hf_model_type(vision_model_type)

        if "siglip" in model_type:
            keep_class_token = False
        else:
            raise NotImplementedError(f"unsupported huggingface vision model: {vision_model_type}")
    else:
        raise NotImplementedError(f"unknown vision model type {vision_model_type}")

    num_patches_per_dim_h = img_h // patch_dim
    num_patches_per_dim_w = img_w // patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    num_image_embeddings_per_tile = num_patches + (class_token_len if keep_class_token else 0)

    if pixel_shuffle:
        num_image_embeddings_per_tile = int(num_image_embeddings_per_tile * (0.5**2))

    if use_tile_tags:
        if tokenizer_type in ("llama3p1", "chatml", "qwen2p0", "qwen2p5"):
            num_image_embeddings_per_tile += 5
        else:
            raise ValueError("tokenizer type not defined")

        if 10 < max_num_tiles < 100:
            if tokenizer_type.startswith("qwen"):
                num_image_embeddings_per_tile += 1  # add padding 0
        elif max_num_tiles > 100:
            raise ValueError(f"max number of tiles {max_num_tiles} not supported")

    return num_image_embeddings_per_tile
