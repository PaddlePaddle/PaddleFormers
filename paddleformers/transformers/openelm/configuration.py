# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# Adapted for PaddlePaddle / paddleformers from the original Apple OpenELM implementation.

"""Implements OpenELMConfig based on PretrainedConfig for paddleformers."""
from numbers import Number
from typing import List, Optional, Union

import numpy as np

from ..configuration_utils import PretrainedConfig


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """Ensure v is divisible by divisor (from MobileNet)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def compute_heads(model_dim: int, head_dim: int) -> int:
    """Compute number of heads. Raises ValueError if not divisible."""
    if model_dim % head_dim == 0:
        return model_dim // head_dim
    else:
        raise ValueError(f"Model dimension should be divisible by head dimension. Got: {model_dim} and {head_dim}.")


OpenELM_CONFIGS = {
    "OpenELM-270M": dict(
        num_transformer_layers=16,
        model_dim=1280,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-450M": dict(
        num_transformer_layers=20,
        model_dim=1536,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-1_1B": dict(
        num_transformer_layers=28,
        model_dim=2048,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-3B": dict(
        num_transformer_layers=36,
        model_dim=3072,
        head_dim=128,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
}


class OpenELMConfig(PretrainedConfig):
    """Configuration class for OpenELM model."""

    model_type = "openelm"

    @property
    def num_hidden_layers(self) -> int:
        """Alias for num_transformer_layers, needed by DynamicCache."""
        return self.num_transformer_layers

    def __init__(
        self,
        vocab_size: int = 32000,
        max_context_length: int = 2048,
        num_transformer_layers: int = 12,
        model_dim: int = 2048,
        head_dim: int = 128,
        qkv_multipliers: Union[Number, List[Number]] = 1.0,
        num_query_heads: Union[int, None] = None,
        num_gqa_groups: int = 1,
        ffn_multipliers: Union[Number, List[Number]] = 4.0,
        ffn_with_glu: bool = True,
        ffn_dim_divisor: int = 256,
        activation_fn_name: str = "swish",
        normalization_layer_name: str = "rms_norm",
        normalize_qk_projections: bool = False,
        share_input_output_layers: bool = False,
        rope_freq_constant: int = 10000,
        rope_max_length: int = 4096,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        self.num_transformer_layers = num_transformer_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.qkv_multipliers = qkv_multipliers
        self.num_query_heads = num_query_heads
        self.num_gqa_groups = num_gqa_groups
        self.ffn_multipliers = ffn_multipliers
        self.ffn_with_glu = ffn_with_glu
        self.ffn_dim_divisor = ffn_dim_divisor
        self.activation_fn_name = activation_fn_name
        self.normalization_layer_name = normalization_layer_name
        self.normalize_qk_projections = normalize_qk_projections
        self.share_input_output_layers = share_input_output_layers
        self.rope_freq_constant = rope_freq_constant
        self.rope_max_length = rope_max_length
        self.num_query_heads = (
            compute_heads(model_dim=model_dim, head_dim=head_dim) if num_query_heads is None else num_query_heads
        )
        self.initializer_range = initializer_range

        self.__post_init__()
        super().__init__(
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def __post_init__(self) -> None:
        if self.num_gqa_groups is not None:
            head_multiple_of = self.num_gqa_groups
        else:
            head_multiple_of = 2

        if isinstance(self.qkv_multipliers, Number):
            qkv_dim = make_divisible(
                self.model_dim * self.qkv_multipliers,
                divisor=self.head_dim * head_multiple_of,
            )
            query_dims = [int(qkv_dim)] * self.num_transformer_layers

        elif isinstance(self.qkv_multipliers, (tuple, list)) and len(self.qkv_multipliers) == 2:
            # Layer-wise scaling (https://arxiv.org/abs/2008.00623)
            qkv_multipliers = [
                round(v, 2)
                for v in np.linspace(
                    self.qkv_multipliers[0],
                    self.qkv_multipliers[1],
                    num=self.num_transformer_layers,
                    dtype=float,
                )
            ]
            query_dims = [
                int(make_divisible(self.model_dim * m, divisor=self.head_dim * head_multiple_of))
                for m in qkv_multipliers
            ]
        else:
            raise NotImplementedError(
                f"QKV multipliers should be a single number or a list containing exactly two numbers. Got: {qkv_multipliers}."
            )

        self.num_query_heads = [int(compute_heads(q_dim, self.head_dim)) for q_dim in query_dims]
        self.num_kv_heads = [q_heads // self.num_gqa_groups for q_heads in self.num_query_heads]

        if isinstance(self.ffn_multipliers, Number):
            self.ffn_multipliers = [self.ffn_multipliers] * self.num_transformer_layers
        elif isinstance(self.ffn_multipliers, (tuple, list)):
            if len(self.ffn_multipliers) == 2:
                self.ffn_multipliers = [
                    round(v, 2)
                    for v in np.linspace(
                        self.ffn_multipliers[0],
                        self.ffn_multipliers[1],
                        num=self.num_transformer_layers,
                        dtype=float,
                    )
                ]
            else:
                assert (
                    len(self.ffn_multipliers) == self.num_transformer_layers
                ), f"{len(self.ffn_multipliers)=}!={self.num_transformer_layers=}"
        else:
            raise NotImplementedError(
                f"FFN multipliers should be a single number or a list containing exactly two numbers. Got: {qkv_multipliers}."
            )

        for layer_idx in range(len(query_dims)):
            assert self.num_query_heads[layer_idx] % self.num_kv_heads[layer_idx] == 0
