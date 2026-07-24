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


from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddleformers.fleet.transformer.layer import FleetLayer
from paddleformers.fleet.transformer.mlp import MLP, MLPSublayersSpec
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class MultimodalProjector(FleetLayer):
    """
    MultimodalProjector will take the encoded input with input_size hidden state and project
    it into the hidden size of the language model for multimodal training. When projector is
    type affine linear_fc1 from sublayers_spec is used.

    Args:
        transformer_config (TransformerConfig): Transformer config
        sublayers_spec (MLPSublayersSpec): Specifies MLP sublayers_spec for mlp type projector
        projector_type (str): Projector type
        input_size (int): Input size from feature encoder
        tp_group : Tensor parallel group
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLPSublayersSpec,
        projector_type: str,
        input_size: int,
        tp_group=None,
    ):
        super().__init__(config=config)
        self.projector_type = projector_type

        assert sublayers_spec is not None, "MLPSublayersSpec must be provided"

        if self.projector_type == "mlp":
            self.encoder = MLP(
                config=config,
                sublayers_spec=sublayers_spec,
                input_size=input_size,
                tp_group=tp_group,
            )
        elif self.projector_type == "affine":
            self.encoder = build_spec_layer(
                sublayers_spec.linear_fc1,
                input_size,
                config.hidden_size,
                config=config,
                init_method=config.init_method,
                gather_output=True,
                bias=config.add_bias_linear,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name=None,
                tp_group=tp_group,
            )
        else:
            raise Exception(
                f"Unsupported multimodal projection type {self.projector_type}"
            )

    def forward(self, hidden_states):
        """Run multimodal projector.

        Args:
            hidden_states (paddle.Tensor): Input.

        Returns:
            paddle.Tensor: The projected output.
        """

        # Run encoder.
        encoder_output, encoder_output_bias = self.encoder(hidden_states)

        if encoder_output_bias is not None:
            encoder_output = encoder_output + encoder_output_bias

        # the encoder produces "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.

        # TODO(zhangweilong): remove this when we fix the issue
        # encoder_output = make_viewless_tensor(
        #     inp=encoder_output, requires_grad=True, keep_graph=True
        # )

        return encoder_output
