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
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddleformers.fleet import tensor_parallel
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.transformer.layer import FleetLayer
from paddleformers.fleet.transformer.paddle_norm import WrappedPaddleNorm
from paddleformers.fleet.transformer.transformer_layer import TransformerLayer
from paddleformers.fleet.utils import WrappedTensor

if TYPE_CHECKING:
    from paddleformers.fleet.packed_seq_params import PackedSeqParams
    from paddleformers.fleet.transformer.transformer_config import (
        TransformerConfig,
    )

LayerNormImpl = WrappedPaddleNorm

logger = logging.getLogger(__name__)


@dataclass
class TransformerBlockSublayersSpec:
    """
    Dataclass for specifying the sublayers_spec of a transformer block.

    This class defines the structure for configuring the layers and normalization
    within a transformer block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (list[LayerSpec] | None): A list of layer specifications for
            the layers within the transformer block. Each specification typically
            defines a complete transformer layer (e.g., self-attention, feed-forward network).
        layer_norm (LayerSpec | paddle.nn.Layer | None): Specification
            or instance of the layer normalization to be applied.
    """

    layer_specs: list[LayerSpec] | None = None
    layer_norm: LayerSpec | None = None


def _get_block_sublayers_spec(
    config: TransformerConfig,
    spec: TransformerBlockSublayersSpec | LayerSpec,
    vp_stage: int | None = None,
    pp_rank: int | None = None,
) -> TransformerBlockSublayersSpec:
    """
    Retrieve or construct TransformerBlockSublayersSpec based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (TransformerBlockSublayersSpec | LayerSpec): Specification for the
            transformer block sublayers_spec. Can be either a TransformerBlockSublayersSpec
            instance or a LayerSpec.
        vp_stage (int | None): Virtual pipeline stage number.

    Returns:
        TransformerBlockSublayersSpec: The sublayers_spec for the transformer block.
    """

    # Transformer block sublayers_spec.
    if isinstance(spec, TransformerBlockSublayersSpec):
        return spec

    # LayerSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `TransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, LayerSpec):
        if issubclass(spec.layer, TransformerBlock):
            return spec.sublayers_spec
        elif issubclass(spec.layer, TransformerLayer):
            return TransformerBlockSublayersSpec(
                layer_specs=[spec] * config.num_hidden_layers,
                layer_norm=LayerNormImpl,
            )
        else:
            raise Exception(f"specialize for {spec.layer.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class TransformerBlock(FleetLayer):
    """Transformer class."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: TransformerBlockSublayersSpec | LayerSpec,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
    ):
        assert vp_stage is None, (
            "pipeline parallel is not supported in TransformerBlock."
        )
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.sublayers_spec = _get_block_sublayers_spec(
            config, spec, vp_stage, pp_rank=None
        )
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process

        # required for pipeline parallel schedules
        self.input_tensor = None

        assert self.config.cpu_offloading is False

        self.config._cpu_offloading_context = None

        self._build_layers()
        self.num_layers_per_pipeline_rank = len(self.layers)

    def _build_layers(self):
        # Transformer layers.
        def _build_layer(layer_spec, layer_number):
            layer_config = self.config

            layer = build_spec_layer(
                layer_spec,
                config=layer_config,
                layer_number=layer_number,
                pg_collection=self.pg_collection,
            )
            return layer

        # offset is implicit in TransformerLayer
        self.layers = paddle.nn.LayerList(
            [
                _build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.sublayers_spec.layer_specs)
            ]
        )

        # In pipeline parallelism, we want to add this LN only to the last stage of the pipeline
        # self.post_process and self.post_layer_norm guide this behavior
        if (
            self.sublayers_spec.layer_norm
            and self.post_process
            and self.post_layer_norm
        ):
            self.norm = build_spec_layer(
                self.sublayers_spec.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.rms_norm_eps,
            )
        else:
            self.norm = None  # Either this or nn.Identity

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states: Tensor | WrappedTensor,
        attention_mask: Tensor,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: Tensor | None = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Tensor | WrappedTensor): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor | None): Context tensor for cross-attention.
            context_mask (Tensor | None): Mask for cross-attention context
            rotary_pos_emb (Tensor | None): Rotary positional embeddings.
            rotary_pos_cos (Tensor | None): Rotary embedding cosine.
            rotary_pos_sin (Tensor | None): Rotary embedding sine.
            attention_bias (Tensor | None): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
            packed_seq_params (PackedSeqParams | None): Parameters for packed sequence
                processing.

        Returns:
            Tensor | Tuple[Tensor, Tensor]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            hidden_states = self.input_tensor

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            # Forward pass.
            dict_args = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "context": context,
                "context_mask": context_mask,
                "rotary_pos_emb": rotary_pos_emb,
                "rotary_pos_cos": rotary_pos_cos,
                "rotary_pos_sin": rotary_pos_sin,
                "attention_bias": attention_bias,
                "packed_seq_params": packed_seq_params,
            }
            for l_no, layer in enumerate(self.layers):
                dict_args = layer(dict_args)
                hidden_states = dict_args["hidden_states"]
                context = dict_args.get("context", None)

        # Final layer norm.
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states
