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

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    ScheduleNode,
    build_spec_layer,
)
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)

from paddleformers.fleet import tensor_parallel
from paddleformers.fleet.context_parallel_utils import ContextParallelScatterOp
from paddleformers.fleet.parallel_state import (
    get_context_parallel_world_size,
)
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from paddleformers.fleet.tensor_parallel.random import get_cuda_rng_tracker
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddleformers.fleet.models.backends import BackendSpecProvider
    from paddleformers.fleet.packed_seq_params import PackedSeqParams
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig

SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
    AttnMaskType.padding_causal,
]


class MTPLossLoggingHelper:
    """Helper class for logging MTP losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: paddle.Tensor,
        layer_number: int,
        num_hidden_layers: int,
        reduce_group: paddle.distributed.communication.group.Group
        | None = None,
        avg_group: paddle.distributed.communication.group.Group | None = None,
    ):
        """Save the mtp loss for logging.
        Args:
            loss (paddle.Tensor): The loss tensor.
            layer_number (int): Layer index of the loss.
            num_hidden_layers (int): The number of total layers.
            reduce_group (paddle.distributed.communication.group.Group): The group for reducing the loss.
            mean_group (paddle.distributed.communication.group.Group): The group for averaging the loss.
        """
        # Skip mtp loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = paddle.zeros(num_hidden_layers)
        tracker["values"][layer_number] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    def clean_loss_in_tracker():
        """Clear the mtp losses."""
        tracker = MTPLossLoggingHelper.tracker
        tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    def reduce_loss_in_tracker():
        """Collect and reduce the mtp losses across ranks."""
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]
        # Reduce mtp losses across ranks.
        if tracker.get("reduce_group") is not None:
            paddle.distributed.all_reduce(
                values, group=tracker.get("reduce_group")
            )
        if tracker.get("avg_group") is not None:
            paddle.distributed.all_reduce(
                values,
                group=tracker["avg_group"],
                op=paddle.distributed.ReduceOp.AVG,
            )

    def track_mtp_metrics(
        loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None
    ):
        """Track the Multi-Token Prediction (MTP) metrics for logging."""
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        mtp_losses = tracker["values"] * loss_scale
        num_nextn_predict_layers = mtp_losses.shape[0]
        for i in range(num_nextn_predict_layers):
            name = f"mtp_{i + 1} loss"
            loss = mtp_losses[i]
            if total_loss_dict is not None:
                if name in total_loss_dict:
                    total_loss_dict[name] += loss
                else:
                    total_loss_dict[name] = loss
            if writer is not None:
                writer.add_scalar(name, loss, iteration)
            if wandb_writer is not None:
                wandb_writer.log({f"{name}": loss}, iteration)

        MTPLossLoggingHelper.clean_loss_in_tracker()


@dataclass
class MultiTokenPredictionLayerSublayersSpec:
    """
    Dataclass for specifying the sublayers_spec of a MultiTokenPrediction layer.

    Args:
        hnorm (Union[LayerSpec, type]): Specification or instance of the
             hidden states normalization to be applied.
        enorm (Union[LayerSpec, type]): Specification or instance of the
            embedding normalization to be applied.
        eh_proj (Union[LayerSpec, type]): Specification or instance of the
            linear projection to be applied (non-mHC mode: [2h] -> [h]).
        e_proj (Union[LayerSpec, type]): Specification or instance of the
            embedding projection (mHC mode: [h] -> [h]).
        h_proj (Union[LayerSpec, type]): Specification or instance of the
            hidden state per-stream projection (mHC mode: [h] -> [h]).
        transformer_layer (Union[LayerSpec, type]): Specification
            or instance of the transformer block to be applied.
    """

    enorm: LayerSpec | type = None
    hnorm: LayerSpec | type = None
    eh_proj: LayerSpec | type = None
    e_proj: LayerSpec | type = None
    h_proj: LayerSpec | type = None
    transformer_layer: LayerSpec | type = None
    layer_norm: LayerSpec | type = None


def get_mtp_layer_spec_for_backend(
    config: TransformerConfig,
    transformer_layer_spec: LayerSpec,
    backend: BackendSpecProvider,
    layer_number: int,
) -> LayerSpec:
    """Get the MTP layer spec.

    Returns:
        LayerSpec: Layer specification with layers from the backend.
    """
    column_parallel_linear_impl: type = backend.column_parallel_linear()
    layer_norm_impl: type = backend.layer_norm()

    submodules_kwargs = {
        "enorm": layer_norm_impl,
        "hnorm": layer_norm_impl,
        "transformer_layer": transformer_layer_spec,
        "layer_norm": layer_norm_impl,
    }

    if config.enable_hyper_connections:
        submodules_kwargs["e_proj"] = column_parallel_linear_impl
        submodules_kwargs["h_proj"] = column_parallel_linear_impl
    else:
        submodules_kwargs["eh_proj"] = column_parallel_linear_impl

    mtp_layer_spec = LayerSpec(
        layer=WeightOnlyMTPLayer
        if config.mtp_load_weight_only
        else MultiTokenPredictionLayer,
        sublayers_spec=MultiTokenPredictionLayerSublayersSpec(
            **submodules_kwargs
        ),
        extra_kwargs={
            "config": config,
            "layer_number": layer_number,
        },
    )
    return mtp_layer_spec


class MTPLossAutoScaler(paddle.autograd.PyLayer):
    """An AutoScaler that triggers the backward pass and scales the grad for mtp loss."""

    main_loss_backward_scale: paddle.Tensor = paddle.tensor(1.0)

    @staticmethod
    def forward(ctx, output: paddle.Tensor, mtp_loss: paddle.Tensor):
        """Preserve the mtp by storing it in the context to avoid garbage collection.

        Args:
            output (paddle.Tensor): The output tensor.
            mtp_loss (paddle.Tensor): The mtp loss tensor.

        Returns:
            paddle.Tensor: The output tensor.
        """
        ctx.save_for_backward(mtp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: paddle.Tensor):
        """Compute and scale the gradient for mtp loss..

        Args:
            grad_output (paddle.Tensor): The gradient of the output.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: The gradient of the output, scaled mtp loss
                                               gradient.
        """
        (mtp_loss,) = ctx.saved_tensor()
        mtp_loss_backward_scale = MTPLossAutoScaler.main_loss_backward_scale
        scaled_mtp_loss_grad = (
            paddle.ones_like(mtp_loss) * mtp_loss_backward_scale
        )
        return grad_output, scaled_mtp_loss_grad

    @staticmethod
    def set_loss_scale(scale: paddle.Tensor):
        """set the scale of the mtp loss.

        Args:
            scale (paddle.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        MTPLossAutoScaler.main_loss_backward_scale = scale


class MultiTokenPredictionLayer(FleetLayer):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential layers to predict
    D additional tokens.

    The k-th MTP layer consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MultiTokenPredictionLayerSublayersSpec,
        layer_number: int = 1,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__(config=config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.sequence_parallel = config.sequence_parallel
        self.tensor_parallel = config.tensor_model_parallel_size
        self.sublayers_spec = sublayers_spec
        self.layer_number = layer_number
        self.cp_group = pg_collection.cp

        self_attention_spec = (
            self.sublayers_spec.transformer_layer.sublayers_spec.self_attn
        )
        attn_mask_type = self_attention_spec.extra_kwargs.get(
            "attn_mask_type", ""
        )
        assert attn_mask_type in SUPPORTED_ATTN_MASK, (
            "Multi-Token Prediction (MTP) is not jet supported with "
            + f"{attn_mask_type} attention mask type."
            + f"The supported attention mask types are {SUPPORTED_ATTN_MASK}."
        )

        self.mhc_enabled = config.enable_hyper_connections

        self.enorm = build_spec_layer(
            self.sublayers_spec.enorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        self.hnorm = build_spec_layer(
            self.sublayers_spec.hnorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        if self.mhc_enabled:
            # mHC mode: separate e_proj and h_proj, operating per-stream.
            # e_proj: [h] -> [h], applied to embedding then broadcast across streams.
            # h_proj: [h] -> [h], applied per-stream on hidden states.
            self.e_proj = build_spec_layer(
                self.sublayers_spec.e_proj,
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )
            self.h_proj = build_spec_layer(
                self.sublayers_spec.h_proj,
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )
            self.eh_proj = None

            # Learned contraction parameters for MTP output
            n = config.num_residual_streams
            hc_dim = config.hidden_size * n
            self.hc_head_fn = self.create_parameter(
                shape=[hc_dim, n],
                dtype=self.config.params_dtype,
                default_initializer=nn.initializer.Constant(0.0),
            )
            # Use model-parallel RNG tracker for Xavier init so that the
            # initialization is independent of pipeline layer_index.
            if paddle.distributed.get_world_size() <= 1:
                nn.initializer.XavierUniform()(self.hc_head_fn)
            else:
                with get_cuda_rng_tracker().fork():
                    nn.initializer.XavierUniform()(self.hc_head_fn)
            self.hc_head_base = self.create_parameter(
                shape=[n],
                dtype=self.config.params_dtype,
                default_initializer=nn.initializer.Constant(0.0),
            )
            self.hc_head_scale = self.create_parameter(
                shape=[1],
                dtype=self.config.params_dtype,
                default_initializer=nn.initializer.Constant(1.0),
            )
            if self.sequence_parallel:
                self.hc_head_fn.is_distributed = False
                self.hc_head_base.is_distributed = False
                self.hc_head_scale.is_distributed = False
        else:
            # Non-mHC mode: eh_proj [2h] -> [h]
            # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
            # of the i-th token's hidden states and the (i + K)-th token's decoder input,
            # so the input's shape is [s, b, 2*h].
            # The output will be sent to the following transformer layer,
            # so the output's shape should be [s, b, h].
            if self.config.gpt_model_use_experimental_version:
                self.eh_proj = paddle.incubate.nn.FusedLinear(
                    self.config.hidden_size * 2,
                    self.config.hidden_size,
                    bias_attr=self.config.use_bias,
                )
                if self.config.tensor_model_parallel_size > 1:
                    mark_as_sequence_parallel_parameter(self.eh_proj.weight)
                    if self.config.use_bias:
                        mark_as_sequence_parallel_parameter(self.eh_proj.bias)
            else:
                self.eh_proj = build_spec_layer(
                    self.sublayers_spec.eh_proj,
                    self.config.hidden_size * 2,
                    self.config.hidden_size,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=False,
                    skip_bias_add=False,
                    is_expert=False,
                )
            self.e_proj = None
            self.h_proj = None

        self.transformer_layer = build_spec_layer(
            self.sublayers_spec.transformer_layer,
            config=self.config,
            is_mtp_layer=True,
        )
        if not self.config.gpt_model_use_experimental_version:
            self.norm = build_spec_layer(
                self.sublayers_spec.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.rms_norm_eps,
            )

        self.offload_context = nullcontext()

    def _concat_embeddings(
        self,
        hidden_states: paddle.Tensor,
        decoder_input: paddle.Tensor,
        mtp_hidden_inputs_mask: paddle.Tensor | None = None,
    ):
        """
        Concatenate the tokens before sending to transformer layer.

        In mHC mode, hidden_states is [s, b, n*h] (multi-stream) and decoder_input
        is [s, b, h] (single-stream embedding). Uses separate e_proj and h_proj.
        In non-mHC mode, concatenates and projects with eh_proj as before.
        """
        decoder_input = self.enorm(decoder_input)

        if self.mhc_enabled:
            # mHC mode: hidden_states is [s, b, n*h]
            n = self.config.num_residual_streams
            h = self.config.hidden_size
            s, b, _ = hidden_states.shape

            hs_streams = hidden_states.reshape([s, b, n, h])
            hs_streams = self.hnorm(hs_streams)

            # Apply mask if needed
            if mtp_hidden_inputs_mask is not None:
                # [B, 1, S] -> [B, S, 1]
                mtp_hidden_inputs_mask = mtp_hidden_inputs_mask.transpose(
                    [0, 2, 1]
                ).astype(hs_streams.dtype)
                if (
                    get_context_parallel_world_size() > 1
                    and self.config.experimental_dataflow
                ):
                    mtp_hidden_inputs_mask = ContextParallelScatterOp.apply(
                        mtp_hidden_inputs_mask,
                        axis=1,
                        mode=self.config.cp_balance_mode,
                    )
                # when sp enable
                if self.sequence_parallel:
                    # [B, S/CP, 1] -> [S/CP, B, 1]
                    mtp_hidden_inputs_mask = mtp_hidden_inputs_mask.transpose(
                        [1, 0, 2]
                    )
                    # [S/CP, B, 1] -> [S/CP/TP, B, 1]
                    mtp_hidden_inputs_mask = (
                        scatter_to_sequence_parallel_region(
                            mtp_hidden_inputs_mask
                        )
                    )
                hs_streams = hs_streams * mtp_hidden_inputs_mask.unsqueeze(-1)

            # e_proj: [.., h] -> [.., h/tp]
            e_out, _ = self.e_proj(decoder_input)
            # h_proj: applied per-stream [.., n, h] -> [.., n, h/tp]
            # 这里hs_streams是4D tensor: [b,s,n,h]会导致算梯度的时候调用.t()报错，必须reshape到更低维度
            orig_shape = list(hs_streams.shape)  # [s/sp, b, n, h]
            if self.tensor_parallel > 1 and self.sequence_parallel:
                # [s/sp, b, n, h] --> [s, b, n, h]
                orig_shape[0] = orig_shape[0] * self.tensor_parallel
            hs_flat = hs_streams.reshape([-1, orig_shape[-1]])  # [s/sp*b*n, h]
            h_out, _ = self.h_proj(hs_flat)  # [s*b*n, h/tp]
            h_out = h_out.reshape([*orig_shape[:-1], -1])  # [s, b, n, h/tp]
            # Broadcast add before gather (saves one all-gather vs gathering separately)
            hidden_states = e_out.unsqueeze(-2) + h_out
            if self.tensor_parallel > 1:
                hidden_states = gather_from_tensor_model_parallel_region(
                    hidden_states
                )
            # Flatten back to [.., n*h]
            *leading, n, h = hidden_states.shape
            hidden_states = hidden_states.reshape([*leading, n * h])

            if self.sequence_parallel:
                hidden_states = scatter_to_sequence_parallel_region(
                    hidden_states
                )
        else:
            hidden_states = self.hnorm(hidden_states)
            # Apply mtp_hidden_inputs_mask to mask out hidden state contributions
            # at specific positions (e.g. EOS boundaries) in MTP.
            # mask shape: [B, 1, S] -> [B, S, 1] to broadcast with hidden_states [B, S, H]
            if mtp_hidden_inputs_mask is not None:
                mtp_hidden_inputs_mask = mtp_hidden_inputs_mask.transpose(
                    [0, 2, 1]
                )
                mtp_hidden_inputs_mask = mtp_hidden_inputs_mask.astype(
                    hidden_states.dtype
                )

                if (
                    get_context_parallel_world_size() > 1
                    and self.config.experimental_dataflow
                ):
                    # In EB dataflow and CP size > 1, mtp_hidden_inputs_mask is [b, s, 1];
                    # we need to scatter it to [b, s/cp, 1] here.
                    mtp_hidden_inputs_mask = ContextParallelScatterOp.apply(
                        mtp_hidden_inputs_mask,
                        axis=1,
                        mode=self.config.cp_balance_mode,
                    )

                # when sp enable
                if self.sequence_parallel:
                    if self.config.gpt_model_use_experimental_version:
                        mtp_hidden_inputs_mask = mtp_hidden_inputs_mask.reshape(
                            [-1, 1]
                        )
                        mtp_hidden_inputs_mask = ScatterOp.apply(
                            mtp_hidden_inputs_mask
                        )
                    else:
                        # [B, S/CP, 1] -> [S/CP, B, 1]
                        mtp_hidden_inputs_mask = (
                            mtp_hidden_inputs_mask.transpose([1, 0, 2])
                        )
                        mtp_hidden_inputs_mask = (
                            scatter_to_sequence_parallel_region(
                                mtp_hidden_inputs_mask
                            )
                        )
                hidden_states = hidden_states * mtp_hidden_inputs_mask
            # At the (k - 1)-th MTP layer, concatenates the i-th token's hidden_states
            # and the (i + K)-th token's embedding, and combine them with linear projection.
            hidden_states = paddle.cat((decoder_input, hidden_states), -1)
            hidden_states = self.eh_proj(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states, _ = hidden_states
            # For tensor parallel we need to gather the tensor across the model-parallel
            # ranks after the linear projection. This used to call
            # `all_gather_last_dim_from_tensor_parallel_region`, but that utility reduces
            # the gradient in backward pass and was therefore incorrect in this context.
            # It has been replaced with the correct `gather_from_tensor_model_parallel_region`.
            if not self.config.gpt_model_use_experimental_version:
                if self.tensor_parallel > 1:
                    hidden_states = gather_from_tensor_model_parallel_region(
                        hidden_states
                    )
                # For sequence parallel, scatter after linear_fc and before transformer layer.
                if self.sequence_parallel:
                    hidden_states = scatter_to_sequence_parallel_region(
                        hidden_states
                    )
        return hidden_states

    def _proj_and_transformer_layer(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: paddle.Tensor | None = None,
        context: paddle.Tensor | None = None,
        context_mask: paddle.Tensor | None = None,
        rotary_pos_emb: paddle.Tensor | None = None,
        rotary_pos_cos: paddle.Tensor | None = None,
        rotary_pos_sin: paddle.Tensor | None = None,
        swa_rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        swa_rotary_pos_cos: Tensor | None = None,
        swa_rotary_pos_sin: Tensor | None = None,
        attention_bias: paddle.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        mtp_hidden_inputs_mask: paddle.Tensor | None = None,
        input_ids: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
        **kwargs,
    ) -> paddle.Tensor:
        """
        Concatenates embeddings with hidden states and then applies transformer layer forward.
        """
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            hidden_states = self._concat_embeddings(
                hidden_states, decoder_input, mtp_hidden_inputs_mask
            )

            input_dict = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "context": context,
                "context_mask": context_mask,
                "rotary_pos_emb": rotary_pos_emb,
                "rotary_pos_cos": rotary_pos_cos,
                "rotary_pos_sin": rotary_pos_sin,
                "swa_rotary_pos_emb": swa_rotary_pos_emb,
                "swa_rotary_pos_cos": swa_rotary_pos_cos,
                "swa_rotary_pos_sin": swa_rotary_pos_sin,
                "attention_bias": attention_bias,
                "packed_seq_params": packed_seq_params,
                "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
                "is_mtp": True,
                "input_ids": input_ids,
                "position_ids": position_ids,
            }
            rst_dict = self.transformer_layer(input_dict)

        hidden_states = rst_dict["hidden_states"]

        # In mHC mode, skip postprocess here - it's deferred to forward()
        # so we can keep multi-stream state for subsequent MTP layers.
        if (
            not self.mhc_enabled
            and not self.config.gpt_model_use_experimental_version
        ):
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def _postprocess(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """Postprocess MTP layer output: learned contraction in mHC mode + layer norm.

        In mHC mode, the hidden_states is multi-stream [s, b, n*h] and needs to be
        contracted to single-stream [s, b, h] before being used for loss computation.
        """
        if self.mhc_enabled:
            from paddleformers.fleet.transformer.hyper_connection import (
                HyperConnectionModule,
            )

            hidden_states = HyperConnectionModule.learned_output_contract(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
                self.config.num_residual_streams,
                self.config.rms_norm_eps,
            )

        # Final layer norm
        if not self.config.gpt_model_use_experimental_version:
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def _checkpointed_forward(self, forward_func, *args, **kwargs):
        def checkpoint_handler():
            """Determines whether to use the `tensor_parallel.checkpoint`"""
            hidden_states = kwargs.get("hidden_states", None)
            decoder_input = kwargs.get("decoder_input", None)
            attention_mask = kwargs.get("attention_mask", None)
            attn_mask_startend_row_indices = kwargs.get(
                "attn_mask_startend_row_indices", None
            )
            context = kwargs.get("context", None)
            context_mask = kwargs.get("context_mask", None)
            rotary_pos_emb = kwargs.get("rotary_pos_emb", None)
            rotary_pos_cos = kwargs.get("rotary_pos_cos", None)
            rotary_pos_sin = kwargs.get("rotary_pos_sin", None)
            swa_rotary_pos_emb = kwargs.get("swa_rotary_pos_emb", None)
            swa_rotary_pos_cos = kwargs.get("swa_rotary_pos_cos", None)
            swa_rotary_pos_sin = kwargs.get("swa_rotary_pos_sin", None)
            attention_bias = kwargs.get("attention_bias", None)
            packed_seq_params = kwargs.get("packed_seq_params", None)
            mtp_hidden_inputs_mask = kwargs.get("mtp_hidden_inputs_mask", None)
            input_ids = kwargs.get("input_ids", None)
            position_ids = None
            if self.config.gpt_model_use_experimental_version:
                position_ids = kwargs.get("position_ids", None)
            return recompute(
                forward_func,
                hidden_states=hidden_states
                if hidden_states is not None
                else None,
                decoder_input=decoder_input
                if decoder_input is not None
                else None,
                attention_mask=attention_mask
                if attention_mask is not None
                else None,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices
                if attn_mask_startend_row_indices is not None
                else None,
                context=context if context is not None else None,
                context_mask=context_mask if context_mask is not None else None,
                rotary_pos_emb=rotary_pos_emb
                if rotary_pos_emb is not None
                else None,
                rotary_pos_cos=rotary_pos_cos
                if rotary_pos_cos is not None
                else None,
                rotary_pos_sin=rotary_pos_sin
                if rotary_pos_sin is not None
                else None,
                swa_rotary_pos_emb=swa_rotary_pos_emb
                if swa_rotary_pos_emb is not None
                else None,
                swa_rotary_pos_cos=swa_rotary_pos_cos
                if swa_rotary_pos_cos is not None
                else None,
                swa_rotary_pos_sin=swa_rotary_pos_sin
                if swa_rotary_pos_sin is not None
                else None,
                attention_bias=attention_bias
                if attention_bias is not None
                else None,
                packed_seq_params=packed_seq_params
                if packed_seq_params is not None
                else None,
                mtp_hidden_inputs_mask=mtp_hidden_inputs_mask
                if mtp_hidden_inputs_mask is not None
                else None,
                input_ids=input_ids if input_ids is not None else None,
                position_ids=position_ids if position_ids is not None else None,
            )

        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            assert self.config.recompute_num_layers == 1, (
                "recompute_num_layers must be 1 for MTP recompute"
            )
            outputs = checkpoint_handler()
        elif self.config.recompute_method == "block":
            warnings.warn(
                "recompute_method == 'block' is not supported for MTP yet."
                " Skipping recompute."
            )
            outputs = forward_func(*args, **kwargs)
        else:
            raise ValueError("Invalid activation recompute method.")

        return outputs

    def forward(self, dict_args: dict):
        if "context" in dict_args:
            assert dict_args["context"] is None, (
                "multi token prediction + cross attention is not yet supported."
            )
        if "packed_seq_params" in dict_args:
            assert dict_args["packed_seq_params"] is None, (
                "multi token prediction + sequence packing is not yet supported."
            )

        # === Magic Send branch ===
        # hidden_states is pure backbone output (not concatenated); mtp_input_embeds provided by MTPEmbeddingLayer
        if self.config.enable_mtp_magic_send:
            hidden_states = dict_args["hidden_states"]
            mhc_multistream = dict_args.pop("mhc_multistream", None)
            # Save backbone output for downstream GPTMainLMHead (main logits computation)
            dict_args["_backbone_hidden_states"] = hidden_states
            mtp_input_embeds = dict_args.get("mtp_input_embeds", None)
            if mtp_input_embeds is None:
                raise RuntimeError(
                    "enable_mtp_magic_send=True but mtp_input_embeds not found in dict_args. "
                    "MTPEmbeddingLayer may not have been executed."
                )

            # mtp_input_embeds: [B, S+num_mtp, H] (full embedding of original input_ids)
            # Extract shifted slice for current depth as decoder_input
            num_mtp = self.config.num_nextn_predict_layers
            # Compute global main sequence length S (before CP/SP scatter).
            # hidden_states arriving here is already CP-local and/or SP-local,
            # so we must recover the full sequence length for correct slicing of
            # mtp_input_embeds (which is always kept at full [B, S+num_mtp, H]).
            cp_world_size = get_context_parallel_world_size()
            if self.config.sequence_parallel:
                # SP format: hidden_states is [S_local/tp, B, H]
                seq_len = (
                    hidden_states.shape[0]
                    * self.config.tensor_model_parallel_size
                )
            else:
                # Non-SP: hidden_states is [B, S_local, H]
                seq_len = hidden_states.shape[1]
            # Recover global seq_len if CP is active
            if cp_world_size > 1 and self.config.experimental_dataflow:
                seq_len = seq_len * cp_world_size

            # shifted embedding for depth k: mtp_input_embeds[:, (k+1):(k+1+seq_len), :]
            depth = self.layer_number
            decoder_input = mtp_input_embeds[
                :, (depth + 1) : (depth + 1 + seq_len), :
            ]

            # Apply CP/SP scatter to decoder_input to match the format of hidden_states.
            # In the non-magic-send path, GPTEmbedding applies these transforms to each
            # shifted MTP embedding before it enters MultiTokenPredictionLayer.
            if (
                get_context_parallel_world_size() > 1
                and self.config.experimental_dataflow
            ):
                decoder_input = ContextParallelScatterOp.apply(
                    decoder_input, axis=1, mode=self.config.cp_balance_mode
                )

            if self.config.sequence_parallel:
                batch_size, local_seq_len, hidden_size = decoder_input.shape
                decoder_input = decoder_input.reshape(
                    [-1, decoder_input.shape[-1]]
                )
                decoder_input = ScatterOp.apply(decoder_input)
                if not self.config.gpt_model_use_experimental_version:
                    decoder_input = (
                        decoder_input.reshape([batch_size, -1, hidden_size])
                        .permute(1, 0, 2)
                        .contiguous()
                    )  # [S/tp, B, H]

            # Pop auxiliary data
            origin_start_row_indices = dict_args.pop(
                "attn_mask_startend_row_indices", None
            )
            mtp_startend_row_indices_all = dict_args.pop(
                "mtp_startend_row_indices_all", None
            )
            mtp_hidden_inputs_mask_all = dict_args.pop(
                "mtp_hidden_inputs_mask_all", None
            )
            mtp_input_ids_for_moe_mask = dict_args.pop(
                "mtp_input_ids_for_moe_mask", None
            )
            origin_input_ids = dict_args.pop("input_ids", None)

            # Trim rotary_pos_emb to main decoder length
            origin_rotary_pos_emb = dict_args.get("rotary_pos_emb", None)
            if origin_rotary_pos_emb is not None:
                if self.config.sequence_parallel:
                    dict_args["rotary_pos_emb"] = origin_rotary_pos_emb[
                        :seq_len
                    ]
                else:
                    dict_args["rotary_pos_emb"] = origin_rotary_pos_emb[
                        :, :seq_len
                    ]
            origin_rotary_pos_cos = dict_args.get("rotary_pos_cos", None)
            if origin_rotary_pos_cos is not None:
                dict_args["rotary_pos_cos"] = origin_rotary_pos_cos[:, :seq_len]
            origin_rotary_pos_sin = dict_args.get("rotary_pos_sin", None)
            if origin_rotary_pos_sin is not None:
                dict_args["rotary_pos_sin"] = origin_rotary_pos_sin[:, :seq_len]

            # Set per-depth mask
            if mtp_startend_row_indices_all is not None:
                if self.config.gpt_model_use_experimental_version:
                    dict_args["attn_mask_startend_row_indices"] = (
                        mtp_startend_row_indices_all[:, depth : depth + 1, :, :]
                    )
                else:
                    dict_args["attn_mask_startend_row_indices"] = (
                        mtp_startend_row_indices_all[
                            :, depth : depth + 1, :, :1
                        ]
                    )
            if mtp_hidden_inputs_mask_all is not None:
                dict_args["mtp_hidden_inputs_mask"] = (
                    mtp_hidden_inputs_mask_all[:, depth : depth + 1, :]
                )
            if mtp_input_ids_for_moe_mask is not None:
                dict_args["input_ids"] = mtp_input_ids_for_moe_mask[
                    :, depth, :
                ].contiguous()
            else:
                dict_args.pop("input_ids", None)

            # Set hidden_states and decoder_input, call _proj_and_transformer_layer
            # mHC: use multi-stream hidden states for MTP computation
            if self.mhc_enabled and mhc_multistream is not None:
                dict_args["hidden_states"] = mhc_multistream
            else:
                dict_args["hidden_states"] = hidden_states
            dict_args["decoder_input"] = decoder_input

            if self.config.recompute_granularity == "full" and self.training:
                hidden_states = self._checkpointed_forward(
                    self._proj_and_transformer_layer,
                    **dict_args,
                )
            else:
                hidden_states = self._proj_and_transformer_layer(
                    **dict_args,
                )

            # mHC: contract multi-stream output to single-stream for loss computation
            if self.mhc_enabled and mhc_multistream is not None:
                hidden_states = self._postprocess(hidden_states)

            # Write back result
            dict_args.pop("decoder_input", None)

            # Concat [backbone_hidden | mtp_hidden] for unified split in downstream LM heads.
            backbone_hs = dict_args.get(
                "_backbone_hidden_states", hidden_states
            )
            dict_args["hidden_states"] = paddle.concat(
                [backbone_hs, hidden_states]
            )

            # Strip auxiliary float tensors (stop_gradient=False) to avoid grad=None crash in P2P backward.
            _keep_keys = {"hidden_states", "labels"}
            for key in list(dict_args.keys()):
                if key not in _keep_keys:
                    dict_args.pop(key)

            return dict_args

        # === Original concat+split logic ===
        hidden_states_concat = dict_args["hidden_states"]
        # mHC: pop multi-stream tensor if available
        mhc_multistream = dict_args.pop("mhc_multistream", None)

        # New dataflow: pop mtp_startend_row_indices_all if present (experimental_dataflow=True)
        # Shape: [B, num_nextn_predict_layers, S, 1]
        origin_start_row_indices = dict_args.pop(
            "attn_mask_startend_row_indices", None
        )
        mtp_startend_row_indices_all = dict_args.pop(
            "mtp_startend_row_indices_all", None
        )
        mtp_hidden_inputs_mask_all = dict_args.pop(
            "mtp_hidden_inputs_mask_all", None
        )
        # Pop per-depth MTP input_ids for MoE routing mask.
        # Shape: [B, num_nextn_predict_layers, max_seq] when present, None otherwise.
        mtp_input_ids_for_moe_mask = dict_args.pop(
            "mtp_input_ids_for_moe_mask", None
        )
        # Save and clear backbone input_ids so it doesn't leak into MTP transformer layers
        origin_input_ids = dict_args.pop("input_ids", None)

        # Trim rotary_pos_emb to main decoder length (remove MTP extra positions)
        # rotary_pos_emb includes extra positions beyond the main decoder length;
        # MTP's internal transformer_layer processes main-length sequences only.
        # Compute main_seq_len from the split hidden_states shape.
        n = self.config.num_nextn_predict_layers
        if self.config.sequence_parallel:
            main_seq_len = (
                hidden_states_concat.shape[0]
                // (n + 1)
                * self.config.tensor_model_parallel_size
            )
        else:
            # Non-SP: MTP parts are concatenated on batch dim (axis=0),
            # so shape[1] is already the per-part sequence length.
            main_seq_len = hidden_states_concat.shape[1]
        origin_rotary_pos_emb = dict_args.get("rotary_pos_emb", None)
        if origin_rotary_pos_emb is not None:
            if self.config.sequence_parallel:
                dict_args["rotary_pos_emb"] = origin_rotary_pos_emb[
                    :main_seq_len
                ]
            else:
                dict_args["rotary_pos_emb"] = origin_rotary_pos_emb[
                    :, :main_seq_len
                ]
        origin_rotary_pos_cos = dict_args.get("rotary_pos_cos", None)
        if origin_rotary_pos_cos is not None:
            dict_args["rotary_pos_cos"] = origin_rotary_pos_cos[
                :, :main_seq_len
            ]
        origin_rotary_pos_sin = dict_args.get("rotary_pos_sin", None)
        if origin_rotary_pos_sin is not None:
            dict_args["rotary_pos_sin"] = origin_rotary_pos_sin[
                :, :main_seq_len
            ]
        # Shape check: mtp_startend_row_indices_all [B, num_nextn, S, 1],
        #              mtp_hidden_inputs_mask_all   [B, num_nextn, S]
        if mtp_startend_row_indices_all is not None:
            num_nextn = self.config.num_nextn_predict_layers
            assert mtp_startend_row_indices_all.shape[1] == num_nextn, (
                f"mtp_startend_row_indices_all.shape[1]={mtp_startend_row_indices_all.shape[1]} "
                f"!= num_nextn_predict_layers={num_nextn}"
            )
        if mtp_hidden_inputs_mask_all is not None:
            num_nextn = self.config.num_nextn_predict_layers
            assert mtp_hidden_inputs_mask_all.shape[1] == num_nextn, (
                f"mtp_hidden_inputs_mask_all.shape[1]={mtp_hidden_inputs_mask_all.shape[1]} "
                f"!= num_nextn_predict_layers={num_nextn}"
            )
        if (
            mtp_startend_row_indices_all is not None
            and mtp_hidden_inputs_mask_all is not None
        ):
            assert mtp_startend_row_indices_all.shape[:3] == [
                mtp_hidden_inputs_mask_all.shape[0],
                mtp_hidden_inputs_mask_all.shape[1],
                mtp_hidden_inputs_mask_all.shape[2],
            ], (
                f"mtp_startend_row_indices_all shape {mtp_startend_row_indices_all.shape} "
                f"and mtp_hidden_inputs_mask_all shape {mtp_hidden_inputs_mask_all.shape} "
                f"mismatch on [B, num_nextn, S] dims"
            )

        # Split mhc_multistream chunks if available
        mhc_chunks = None
        if mhc_multistream is not None:
            mhc_chunks = paddle.split(
                mhc_multistream, self.config.num_nextn_predict_layers + 1
            )

        if self.config.train_mtp_only:
            for i in range(self.config.num_nextn_predict_layers):
                tensor_list = paddle.split(
                    hidden_states_concat,
                    self.config.num_nextn_predict_layers + 1,
                )
                if mhc_chunks is not None:
                    # mHC mode: use multi-stream as MTP input
                    dict_args["hidden_states"] = mhc_chunks[i]
                else:
                    dict_args["hidden_states"] = tensor_list[i]
                dict_args["decoder_input"] = tensor_list[i + 1]

                # New dataflow: get the mask for depth i, shape [B, 1, S, 1]
                mtp_mask_i = None
                if mtp_startend_row_indices_all is not None:
                    mtp_mask_i = mtp_startend_row_indices_all[
                        :, i : i + 1, :, :
                    ]
                    dict_args["attn_mask_startend_row_indices"] = mtp_mask_i

                # New dataflow: get hidden inputs mask for depth i, shape [B, 1, S]
                if mtp_hidden_inputs_mask_all is not None:
                    dict_args["mtp_hidden_inputs_mask"] = (
                        mtp_hidden_inputs_mask_all[:, i : i + 1, :]
                    )

                # Get per-depth input_ids for MoE routing mask
                if mtp_input_ids_for_moe_mask is not None:
                    dict_args["input_ids"] = mtp_input_ids_for_moe_mask[
                        :, i, :
                    ].contiguous()
                else:
                    dict_args.pop("input_ids", None)

                if (
                    self.config.recompute_granularity == "full"
                    and self.training
                ):
                    hidden_states = self._checkpointed_forward(
                        self._proj_and_transformer_layer,
                        **dict_args,
                    )
                else:
                    hidden_states = self._proj_and_transformer_layer(
                        **dict_args,
                    )

                if mhc_chunks is not None:
                    # mHC: hidden_states is multi-stream, store for next depth
                    mhc_chunks[i + 1] = hidden_states
                    # Contract to single-stream for loss computation
                    tensor_list[i + 1] = self._postprocess(hidden_states)
                else:
                    tensor_list[i + 1] = hidden_states

                hidden_states_concat = paddle.concat(tensor_list)
            dict_args["hidden_states"] = hidden_states_concat
            dict_args.pop("decoder_input")
        else:
            tensor_list = paddle.split(
                hidden_states_concat, self.config.num_nextn_predict_layers + 1
            )
            if mhc_chunks is not None:
                # mHC mode: use multi-stream as MTP input
                dict_args["hidden_states"] = mhc_chunks[self.layer_number]
            else:
                dict_args["hidden_states"] = tensor_list[self.layer_number]
            dict_args["decoder_input"] = tensor_list[self.layer_number + 1]

            # New dataflow: get the mask for this layer's depth, shape [B, 1, S, 1]
            mtp_mask = None
            if mtp_startend_row_indices_all is not None:
                if self.config.gpt_model_use_experimental_version:
                    mtp_mask = mtp_startend_row_indices_all[
                        :,
                        self.layer_number : self.layer_number + 1,
                        :,
                        :,
                    ]
                else:
                    mtp_mask = mtp_startend_row_indices_all[
                        :,
                        self.layer_number : self.layer_number + 1,
                        :,
                        :1,
                    ]
                dict_args["attn_mask_startend_row_indices"] = mtp_mask

            # New dataflow: get hidden inputs mask for this layer's depth, shape [B, 1, S]
            if mtp_hidden_inputs_mask_all is not None:
                dict_args["mtp_hidden_inputs_mask"] = (
                    mtp_hidden_inputs_mask_all[
                        :, self.layer_number : self.layer_number + 1, :
                    ]
                )

            # Get per-depth input_ids for MoE routing mask
            if mtp_input_ids_for_moe_mask is not None:
                dict_args["input_ids"] = mtp_input_ids_for_moe_mask[
                    :, self.layer_number, :
                ].contiguous()
            else:
                dict_args.pop("input_ids", None)

            if self.config.recompute_granularity == "full" and self.training:
                hidden_states = self._checkpointed_forward(
                    self._proj_and_transformer_layer,
                    **dict_args,
                )
            else:
                hidden_states = self._proj_and_transformer_layer(
                    **dict_args,
                )

            if mhc_chunks is not None:
                # mHC: hidden_states is multi-stream, store for next depth
                mhc_chunks[self.layer_number + 1] = hidden_states
                # Contract to single-stream for loss computation
                tensor_list[self.layer_number + 1] = self._postprocess(
                    hidden_states
                )
            else:
                tensor_list[self.layer_number + 1] = hidden_states

            hidden_states_concat = paddle.concat(tensor_list)
            dict_args["hidden_states"] = hidden_states_concat
            dict_args.pop("decoder_input")

        # mHC: pass updated multi-stream to subsequent MTP layers
        if (
            mhc_chunks is not None
            and self.layer_number < self.config.num_nextn_predict_layers - 1
        ):
            mhc_multistream = paddle.concat(mhc_chunks)
            dict_args["mhc_multistream"] = mhc_multistream

        # Restore mtp_startend_row_indices_all for subsequent MTP layers (num_nextn > 1)
        if mtp_startend_row_indices_all is not None:
            dict_args["mtp_startend_row_indices_all"] = (
                mtp_startend_row_indices_all
            )
        # Restore mtp_hidden_inputs_mask_all for subsequent MTP layers (num_nextn > 1)
        if mtp_hidden_inputs_mask_all is not None:
            dict_args["mtp_hidden_inputs_mask_all"] = mtp_hidden_inputs_mask_all
        # Restore mtp_input_ids_for_moe_mask for subsequent MTP layers (num_nextn > 1)
        if mtp_input_ids_for_moe_mask is not None:
            dict_args["mtp_input_ids_for_moe_mask"] = mtp_input_ids_for_moe_mask
        # Restore backbone input_ids
        if origin_input_ids is not None:
            dict_args["input_ids"] = origin_input_ids
        else:
            dict_args.pop("input_ids", None)
        # Restore rotary_pos_emb/cos/sin to full length
        if origin_rotary_pos_emb is not None:
            dict_args["rotary_pos_emb"] = origin_rotary_pos_emb
        if origin_rotary_pos_cos is not None:
            dict_args["rotary_pos_cos"] = origin_rotary_pos_cos
        if origin_rotary_pos_sin is not None:
            dict_args["rotary_pos_sin"] = origin_rotary_pos_sin
        # Clean up per-depth slice key
        dict_args.pop("mtp_hidden_inputs_mask", None)
        if origin_start_row_indices is not None:
            dict_args["attn_mask_startend_row_indices"] = (
                origin_start_row_indices
            )
        return dict_args

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="MultiTokenPredictionLayer")


class WeightOnlyMTPLayer(MultiTokenPredictionLayer):
    """MTP layer that only holds weights without participating in forward computation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for _, param in self.state_dict().items():
            param.is_weight_only_mtp = True

    def forward(self, dict_args: dict):
        return dict_args

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="WeightOnlyMTPLayer")
