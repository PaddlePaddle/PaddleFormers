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
import functools

import numpy as np
import paddle
from paddle.distributed.fleet.utils.sequence_parallel_utils import GatherOp

from ...transformers.tensor_parallel_utils import parallel_matmul


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
        transpose_y=True,
        tensor_parallel_degree=config.tensor_parallel_degree,
        tensor_parallel_output=tensor_parallel_output,
        fuse_linear=config.get("fuse_linear", False),
        training=training,
    )

    return logits


def subbatch(f, arg_idx, axis, bs, out_idx, use_recompute=False, same_arg_idx={}):
    """
    Converts a function to one that applies to subbatch of an input dimension.
    This is useful for processing large tensors in smaller chunks to reduce memory usage.

    Args:
        f (Callable): Original function to be converted to subbatch processing.
        arg_idx ([int]): Indices of the inputs to be subbatched.
        axis ([int]): Indices of the dimensions to be subbatched for each input.
        bs (int): Subbatch size (number of elements to process at once).
        out_idx (int): Index of the output dimension that needs stacking.
        use_recompute (bool, optional): Whether to use recomputation for memory savings. Defaults to False.
        same_arg_idx (dict, optional): Mapping of argument indices that share the same tensor.
                                     e.g. {1: 0} means args[1] == args[0], avoiding duplicate slicing.

    Returns:
        Callable: Converted function that processes inputs in subbatches.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        assert len(arg_idx) == len(axis), "Number of batching args and number of batching dims should match."

        inps = [args[i] for i in arg_idx]
        axis_width = [inp.shape[d] for inp, d in zip(inps, axis)]
        assert len(set(axis_width)) == 1, "Batch sizes should be kept equal."

        inp_axis = {inp: d for inp, d in zip(inps, axis)}

        axis_width = axis_width[0]
        if axis_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in np.arange(0, axis_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in same_arg_idx:
                    assert (
                        i > same_arg_idx[i]
                    ), f"expect i > same_arg_idx[i], but got i: {i} and same_arg_idx[i]: {same_arg_idx[i]}"
                    _args.append(_args[same_arg_idx[i]])
                elif i in arg_idx:
                    inp = inp.slice(
                        [inp_axis[inp]],
                        [slice_at],
                        [min(inp.shape[inp_axis[inp]], slice_at + bs)],
                    )
                    _args.append(inp)
                else:
                    _args.append(inp)
            if use_recompute:
                out = paddle.distributed.fleet.utils.recompute(f, *_args, **kwargs)
            else:
                out = f(*_args, **kwargs)
            outs.append(out)

        return paddle.cat(outs, out_idx)

    return wrapper
