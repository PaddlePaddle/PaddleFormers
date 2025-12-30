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

""" Ernie model configuration"""

from paddleformers.transformers.configuration_utils import PretrainedConfig

__all__ = [
    "DFNRopeVisionTransformerConfig",
]


class DFNRopeVisionTransformerConfig(PretrainedConfig):
    """
    Configuration class for DFNRopeVisionTransformer model.
    This class inherits from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "DFNRope_vision_transformer"

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="quick_gelu",
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        attn_implementation="eager",  # new added
        recompute=False,
        attn_sep=False,
        vit_num_recompute_layers=10000,
        **kwargs,
    ):
        """
        Initialize DFNRopeVisionTransformer model configuration with default or specified parameters.

        Args:
            depth (int): Number of transformer layers in the model.
            embed_dim (int): Dimensionality of the embedding layer.
            hidden_size (int): Dimensionality of the feedforward network.
            hidden_act (str): Activation function for the feedforward network.
            mlp_ratio (float): Ratio between the number of input features and
                the number of output features in the feedforward network.
            num_heads (int): Number of attention heads in each attention layer.
            in_channels (int): Number of channels in the input image.
            patch_size (int):
                Size of patches in the input image. Defaults to 14.
            spatial_merge_size (int):
                Spatial merge size for the spatial transformer module. Defaults to 2.
            attn_implementation (str): Attention implementation type. Defaults to "eager".
            recompute (bool): Whether to use recompute. Defaults to False.
            attn_sep (bool): Whether to separate attention computation into two stages. Defaults to False.
            vit_num_recompute_layers (int): Number of recomputed layers for ViT. Defaults to
        """
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.attn_implementation = attn_implementation
        self.recompute = recompute
        self.attn_sep = attn_sep
        self.vit_num_recompute_layers = vit_num_recompute_layers
