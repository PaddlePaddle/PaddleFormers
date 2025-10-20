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
"""
Layerwise dropout callback.
"""

from copy import deepcopy

from paddleformers.trainer.trainer_callback import TrainerCallback

from ernie.modeling_moe import Ernie4_5_DecoderLayer


class LayerwiseDropoutCallback(TrainerCallback):
    """
    Callback for dynamic layer-wise dropout rate adjustment during model training.

    This callback implements progressive dropout rate adjustment based on:
    1. Training warmup phase (linear increase to target dropout rates)
    2. Layer depth (exponentially decreasing rates for deeper layers)

    Attributes:
        Inherits all attributes from the base TrainerCallback class.
    """

    def on_step_begin(self, args, state, control, **kwargs):
        """
        Adjust layer dropout rates at the start of each training step.

        Args:
            args (TrainingArguments): Training configuration containing dropout parameters.
            state (TrainerState): Current training state including step information.
            control (TrainerControl): Training flow control object.
            **kwargs (Any): Additional keyword arguments containing the model instance.

        Raises:
            AssertionError: If the model instance is not provided in kwargs.
        """

        model = kwargs.get('model', None)
        assert model is not None

        dropout_warmup_steps = args.dropout_warmup_steps

        def update_dropout(layer):
            """
            Update dropout rates for a single layer based on warmup and depth.

            Args:
                layer (nn.Layer): The neural network layer to update.

            Note:
                Only affects layers of type Ernie4_5_DecoderLayer.
                Implements two dropout adjustments:
                1. Warmup: Gradually increases dropout to target rate
                2. Layer decay: Reduces dropout rate for deeper layers
            """
            if isinstance(layer, Ernie4_5_DecoderLayer):
                if state.global_step < dropout_warmup_steps:
                    hidden_step_drop_rate = max(
                        0.0,
                        args.hidden_dropout_prob * (state.global_step / float(dropout_warmup_steps)),
                    )
                    attention_step_drop_rate = max(
                        0.0,
                        args.attention_probs_dropout_prob * (state.global_step / float(dropout_warmup_steps)),
                    )
                else:
                    hidden_step_drop_rate = args.hidden_dropout_prob
                    attention_step_drop_rate = args.attention_probs_dropout_prob

                layer_idx = layer.layer_idx
                hidden_layer_drop_rate = max(
                    0.0,
                    hidden_step_drop_rate * (layer_idx / layer.config.num_hidden_layers),
                )
                attention_layer_drop_rate = max(
                    0.0,
                    attention_step_drop_rate * (layer_idx / layer.config.num_hidden_layers),
                )

                if hasattr(layer.residual_add1, "p"):
                    layer.residual_add1.p = hidden_layer_drop_rate
                    layer.residual_add2.p = hidden_layer_drop_rate
                else:
                    layer.residual_add1.dropout.p = hidden_layer_drop_rate
                    layer.residual_add2.dropout.p = hidden_layer_drop_rate
                deep_config = deepcopy(layer.self_attn.config)
                deep_config.attention_probs_dropout_prob = attention_layer_drop_rate
                layer.self_attn.config = deep_config

        model.apply(update_dropout)
