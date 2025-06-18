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

from paddle import nn
from paddle.nn import functional as F


class TransposedLinear(nn.Layer):
    """
    Same as paddle.layer.Linear, except weight matrix is stored as [out_features, in_features] (same as torch),
    instead of [in_features, out_features]
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        bias_attr=None,
        name=None,
    ):
        super(TransposedLinear, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[out_features, in_features],  # regular linear has shape [in_features, out_features]
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self.bias = self.create_parameter(
            shape=[out_features],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )
        self.name = name

    def forward(self, input):
        out = F.linear(x=input, weight=self.weight.T, bias=self.bias, name=self.name)
        return out

    def extra_repr(self):
        name_str = ", name={}".format(self.name) if self.name else ""
        return "in_features={}, out_features={}, dtype={}{}".format(
            self.weight.shape[1], self.weight.shape[0], self._dtype, name_str
        )
