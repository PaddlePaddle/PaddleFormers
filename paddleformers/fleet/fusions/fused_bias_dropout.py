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

import paddle


def _bias_dropout_add_func(x_with_bias, residual, prob, training):
    # type: (Tuple[Tensor, Optional[Tensor]], Tensor, float, bool) -> Tensor
    # NOTE: Previously, the argument `bias` used to be passed as
    # `bias.expand_as(residual)` when the `bias_dropout_func` is called from the
    # transformer layer but broadcasting should automatically take care of that.
    # Also, looking at broadcasting semantics, `expand_as` and broadcasting
    # seem to be identical performance-wise (both just change the view).

    x, bias = x_with_bias  # unpack

    # Run in-place if in eval mode and inputs do not require gradients
    inplace = (
        not training
        and x.stop_gradient
        and not residual.stop_gradient
        and (bias is None or bias.stop_gradient)
    )

    # If we want to train mixed precision, then the output of this function
    # should be half precision. However, in AMP O1, the input (residual) is
    # in fp32, and it will up-cast the result to fp32, causing pipeline parallel
    # GPU communication to hang. Therefore, we need to cast residual to the same
    # dtype as x.
    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)

    # The Dropout operation, Residual Addition and the tensor returning can be
    # done generically outside the if statement, but that stops fusing of Bias
    # Addition-Dropout-Residual Addition operation. So doing it together inside
    # the conditional branch to improve performance
    if bias is not None:
        if inplace:
            x.add_(bias)
        else:
            x = x + bias
        out = paddle.nn.functional.dropout(x, p=prob, training=training)
        if inplace:
            out.add_(residual)
        else:
            out = residual + out
        return out
    else:
        out = paddle.nn.functional.dropout(x, p=prob, training=training)
        if inplace:
            out.add_(residual)
        else:
            out = out + residual
        return out


def bias_dropout_add_unfused(training):
    def _bias_dropout_add(x_with_bias, residual, prob):
        return _bias_dropout_add_func(x_with_bias, residual, prob, training)

    return _bias_dropout_add


def get_bias_dropout_add(training, fused):
    return bias_dropout_add_unfused(training)
