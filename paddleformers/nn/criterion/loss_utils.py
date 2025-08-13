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

from ...distributed import GatherOp, parallel_matmul


def calc_lm_head_logits(
    config, hidden_states, weight, bias, tensor_parallel_output=None, training=True, gather_hidden_states=False
):
    """
    Calculate language model head logits with support for various parallelization strategies.

    This is the core function that computes the final output logits for a language model,
    handling sequence parallelism and tensor parallelism configurations.

    Args:
        config (Ernie4_5_Config): Model configuration.
        hidden_states (Tensor): Hidden states from the transformer layers
        weight (Tensor): Weight matrix for the language model head
        bias (Tensor): Bias vector for the language model head
        tensor_parallel_output (bool, optional): Override for tensor parallel output behavior.
                                               If None, uses config.tensor_parallel_output.
                                               Defaults to None.
        training (bool, optional): Whether in training mode. Defaults to True.

    Returns:
        Tensor: The computed logits for language modeling.
    """
    if config.sequence_parallel and gather_hidden_states:
        hidden_states = GatherOp.apply(hidden_states)
        seq_length = config.max_sequence_length

        hidden_states = hidden_states.reshape([-1, seq_length, hidden_states.shape[-1]])

    if tensor_parallel_output is None:
        tensor_parallel_output = config.tensor_parallel_output

    logits = parallel_matmul(
        hidden_states,
        weight,
        bias=bias,
        transpose_y=config.get("tie_word_embeddings", False),
        tensor_parallel_degree=config.tensor_parallel_degree,
        tensor_parallel_output=tensor_parallel_output,
        fuse_linear=config.get("fuse_linear", False),
        training=training,
    )

    return logits
