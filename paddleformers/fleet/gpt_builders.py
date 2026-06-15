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
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from functools import partial

from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddleformers.fleet.models.common.empty_layer import EmptyLayer
from paddleformers.fleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    MainLanguageLoss,
)
from paddleformers.fleet.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_layers_spec,
    get_gpt_layer_local_spec,
    get_gpt_mtp_layers_spec,
    get_gpt_spec,
)


def gpt_builder(config, **kwargs):
    print("building GPT model ...")
    if config.n_routed_experts:
        # Define the decoder block spec
        transformer_layers_spec = get_gpt_decoder_layers_spec(
            config,
            normalization=config.normalization,
        )
    else:
        # Define the decoder layer spec
        transformer_layer_spec_func = _get_transformer_layer_spec_func(config)
        transformer_layers_spec = []
        for layer_number in range(config.num_hidden_layers):
            real_layer_number = layer_number + config.num_empty_layers_add_in_head
            transformer_layers_spec.append(transformer_layer_spec_func(layer_number=real_layer_number))
    mtp_layers_spec = None
    if config.num_nextn_predict_layers is not None:
        if hasattr(transformer_layers_spec, "layer_specs") and len(transformer_layers_spec.layer_specs) == 0:
            transformer_layers_spec_for_mtp_func = _get_transformer_layer_spec_func(config)
            transformer_layers_spec_for_mtp = []
            for layer_number in range(config.num_layers):
                transformer_layers_spec_for_mtp.append(transformer_layers_spec_for_mtp_func(layer_number=layer_number))
        else:
            transformer_layers_spec_for_mtp = transformer_layers_spec
        mtp_layers_spec = get_gpt_mtp_layers_spec(
            config,
            transformer_layers_spec_for_mtp,
        )

    head_empty_layers_spec = []
    for i in range(config.num_empty_layers_add_in_head):
        head_empty_layers_spec.append(LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config}))

    tail_empty_layers_spec = []
    num_empty_layers_add_in_tail = config.num_empty_layers_add_in_tail
    if config.separate_mtp_headloss:
        num_empty_layers_add_in_tail -= 1
    for i in range(num_empty_layers_add_in_tail):
        tail_empty_layers_spec.append(LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config}))

    gpt_spec = get_gpt_spec(
        config=config,
        head_empty_layers_spec=head_empty_layers_spec,
        transformer_layers_spec=transformer_layers_spec,
        tail_empty_layers_spec=tail_empty_layers_spec,
        mtp_layers_spec=mtp_layers_spec,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        max_sequence_length=config.max_sequence_length,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rope_theta,
        swa_rotary_base=config.swa_rope_theta,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
    )

    loss_fn = None
    if "loss_fn" in kwargs:
        loss_fn = kwargs.pop("loss_fn")

    if config.separate_mtp_headloss:
        loss_fn = MainLanguageLoss(config)

    return build_spec_layer(
        gpt_spec,
        loss_fn=LanguageLoss(config) if not loss_fn else loss_fn,
        **kwargs,
    )


def _get_transformer_layer_spec_func(config):
    """Get transformer layer specification based on configuration.

    Args:
        config: Model configuration

    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    return partial(
        get_gpt_layer_local_spec,
        config=config,
        use_qk_norm=config.use_qk_norm,
        num_experts=config.n_routed_experts,
        multi_latent_attention=config.multi_latent_attention,
        normalization=config.normalization,
    )
