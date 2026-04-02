# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
    shard_weight,
)

from ...nn.experts import MoeExpertsBase
from ...transformers import linear_utils

ColumnSequenceParallelLinear = linear_utils.ColumnSequenceParallelLinear
RowSequenceParallelLinear = linear_utils.RowSequenceParallelLinear

from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    AllGatherOp,
    ReduceScatterOp,
    mark_as_sequence_parallel_parameter,
)

from ...transformers.mc2_parallel_linear import (
    MC2ColumnParallelCoreLinear,
    MC2ColumnSeqParallelCoreLinear,
    MC2RowParallelCoreLinear,
    MC2RowSeqParallelCoreLinear,
)
from ...utils.import_utils import is_paddlefleet_available
from .utils import rng_ctx

# Conditionally import paddlefleet modules
if is_paddlefleet_available():
    from paddlefleet.transformer.moe.moe_expert import BMMFunction, DeepGEMMBMMFunction
else:
    # Define mock objects or alternative implementations when paddlefleet is not available
    class BMMFunction:
        pass

    class DeepGEMMBMMFunction:
        pass


class LoRALinear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        mp_moe: bool = False,
        is_distributed: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # Actual trainable parameters
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            dtype=self._dtype,
            is_bias=False,
            default_initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
        )
        self.lora_B = self.create_parameter(
            shape=[r, out_features],
            dtype=self._dtype,
            is_bias=False,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0),
                learning_rate=lora_plus_scale,
            ),
        )
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self.disable_lora = False
        if mp_moe or is_distributed:
            for p in self.parameters():
                p.is_distributed = is_distributed
                p.mp_moe = mp_moe

    def rope_init(self):
        if self.cos is None or self.sin is None:
            inv_freq = 1.0 / (10000 ** (paddle.arange(0, self.r, 2, dtype=paddle.float32) / self.r))
            t = paddle.arange(self.rb1, dtype=paddle.float32)
            freqs = t.unsqueeze(1) @ inv_freq.unsqueeze(0)
            emb = paddle.cat([freqs, freqs], axis=-1)
            self.cos = paddle.unsqueeze(paddle.cos(emb), axis=0).astype(self._dtype)
            self.sin = paddle.unsqueeze(paddle.sin(emb), axis=0).astype(self._dtype)

    def get_delta_weight(self, lora_A=None, lora_B=None):
        # compute the delta weight，which is used to merge weights
        lora_A = lora_A if lora_A is not None else self.lora_A
        lora_B = lora_B if lora_B is not None else self.lora_B
        delta_weight = lora_A @ lora_B * self.scaling

        return delta_weight

    def merge(self):
        if not self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight + delta_weight
            self.weight.set_value(new_weight)
            self.merged = True

    def unmerge(self):
        if self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight - delta_weight
            self.weight.set_value(new_weight)
            self.merged = False

    def forward(self, input: paddle.Tensor, *args, **kwargs):
        if self.disable_lora or self.merged:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
        else:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
            result += (self.lora_dropout(input) @ self.lora_A @ self.lora_B) * self.scaling
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class FleetLoRALinear(LoRALinear):
    def __init__(self, in_features, out_features, skip_bias_add, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.skip_bias_add = skip_bias_add

    def forward(self, input: paddle.Tensor):
        out_bias = self.bias if self.skip_bias_add else None
        if self.skip_bias_add:
            self.bias = None
        output = super().forward(input)
        return output, out_bias


class RowParallelLoRALinear(RowParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        **kwargs
    ):
        RowParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # compatible
        self.name = self._name

        # Actual trainable parameters
        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
            self.lora_A = self.create_parameter(
                shape=[self.input_size_per_partition, r],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu")
                ),
            )
        self.lora_B = self.create_parameter(
            shape=[r, self.out_features],
            dtype=self._dtype,
            is_bias=False,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0),
                learning_rate=lora_plus_scale,
            ),
        )

        self.lora_A.is_distributed = True
        self.lora_A.split_axis = 0
        self.lora_B.is_distributed = False
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self.disable_lora = False

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        state_dict = self.state_dict(structured_name_prefix="")
        return build_sharded_state_dict(state_dict, {"weight": 0, "lora_A": 0}, structured_name_prefix)

    def get_delta_weight(self, lora_A=None, lora_B=None):
        lora_A = lora_A if lora_A is not None else self.lora_A
        lora_B = lora_B if lora_B is not None else self.lora_B
        delta_weight = lora_A @ lora_B * self.scaling

        return delta_weight

    def unmerge(self):
        if self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight - delta_weight
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight + delta_weight
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, x: paddle.Tensor):
        if not self.input_is_parallel:
            input_mp = mp_ops._c_split(x, group=self.model_parallel_group)
        else:
            input_mp = x
        if self.disable_lora or self.merged:
            # x @ W : [bz, in_f / ws] ===> [bz, out_f]
            if MC2RowParallelCoreLinear is None:
                result_mp = F.linear(x=input_mp, weight=self.weight, name=self.name)
                output = mp_ops._mp_allreduce(
                    result_mp,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                )
            else:
                output = MC2RowParallelCoreLinear.apply(input_mp, self.weight, self.model_parallel_group)
            output = output + self.bias if self.bias is not None else output
        else:
            # x @ W : [bz, in_f / ws] ===> [bz, out_f]
            if MC2RowParallelCoreLinear is None:
                result_mp = F.linear(x=input_mp, weight=self.weight, name=self.name)
                output = mp_ops._mp_allreduce(
                    result_mp,
                    group=self.model_parallel_group,
                    use_calc_stream=True,
                    use_model_parallel=True,
                )
            else:
                output = MC2RowParallelCoreLinear.apply(input_mp, self.weight, self.model_parallel_group)

            # x @ A: [bz, in_f/ ws] ===> [bz, r]
            input_mp = self.lora_dropout(input_mp) @ self.lora_A
            # all reduce to keep Lora B's gradient on different gpu consistent
            input_dup = mp_ops._mp_allreduce(
                input_mp,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
            #  @ B: [bz, r] ===> [bz, out_f]
            delta_mp = (input_dup @ self.lora_B) * self.scaling
            output += delta_mp
            output = output + self.bias if self.bias is not None else output
        return output

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class FleetRowParallelLoRALinear(RowParallelLoRALinear):
    def __init__(self, in_features, out_features, skip_bias_add, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.skip_bias_add = skip_bias_add

    def forward(self, input: paddle.Tensor):
        out_bias = self.bias if self.skip_bias_add else None
        if self.skip_bias_add:
            self.bias = None
        output = super().forward(input)
        return output, out_bias


class RowSequenceParallelLoRALinear(RowSequenceParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        **kwargs
    ):
        RowSequenceParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # compatible
        self.name = self._name

        # Actual trainable parameters
        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
            self.lora_A = self.create_parameter(
                shape=[self.input_size_per_partition, r],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu")
                ),
            )
        self.lora_B = self.create_parameter(
            shape=[r, self.out_features],
            dtype=self._dtype,
            is_bias=False,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0),
                learning_rate=lora_plus_scale,
            ),
        )

        self.lora_A.is_distributed = True
        self.lora_A.split_axis = 0
        self.lora_B.is_distributed = False
        mark_as_sequence_parallel_parameter(self.lora_B)
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self.disable_lora = False

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        state_dict = self.state_dict(structured_name_prefix="")
        return build_sharded_state_dict(state_dict, {"weight": 0, "lora_A": 0}, structured_name_prefix)

    def get_delta_weight(self, lora_A=None, lora_B=None):
        lora_A = lora_A if lora_A is not None else self.lora_A
        lora_B = lora_B if lora_B is not None else self.lora_B
        delta_weight = lora_A @ lora_B * self.scaling

        return delta_weight

    def unmerge(self):
        if self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight - delta_weight
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight + delta_weight
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, x: paddle.Tensor):
        if not self.input_is_parallel:
            input_mp = mp_ops._c_split(x, group=self.model_parallel_group)
        else:
            input_mp = x

        if MC2RowSeqParallelCoreLinear is None:
            output_parallel = self.linear(input_mp, self.weight, name=self._name)
            output_ = ReduceScatterOp.apply(output_parallel)
            result_mp = output_ + self.bias if self.bias is not None else output_
        else:
            output_ = MC2RowSeqParallelCoreLinear.apply(input_mp, self.weight, self.model_parallel_group)
            result_mp = output_ + self.bias if self.bias is not None else output_

        if not self.merged and not self.disable_lora:
            input_mp = self.lora_dropout(input_mp)
            # TODO(@gexiao): temporary workaround for deterministic calculation
            if True or MC2RowSeqParallelCoreLinear is None:
                input_mp = input_mp @ self.lora_A
                input_mp = ReduceScatterOp.apply(input_mp)
            else:
                input_mp = MC2RowSeqParallelCoreLinear.apply(input_mp, self.lora_A, self.model_parallel_group)
            delta_mp = (input_mp @ self.lora_B) * self.scaling
            result_mp += delta_mp
        return result_mp

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class FleetRowSequenceParallelLoRALinear(RowSequenceParallelLoRALinear):
    def __init__(self, in_features, out_features, skip_bias_add, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.skip_bias_add = skip_bias_add

    def forward(self, input: paddle.Tensor):
        out_bias = self.bias if self.skip_bias_add else None
        if self.skip_bias_add:
            self.bias = None
        output = super().forward(input)
        return output, out_bias


class ColumnParallelLoRALinear(ColumnParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        lora_A_weight_attr: Optional[paddle.ParamAttr] = None,
        **kwargs
    ):
        ColumnParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # compatible
        self.name = self._name

        # Actual trainable parameters
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            dtype=self._dtype,
            is_bias=False,
            attr=lora_A_weight_attr,
        )
        self.lora_A.is_distributed = False
        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
            self.lora_B = self.create_parameter(
                shape=[r, self.output_size_per_partition],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                    learning_rate=lora_plus_scale,
                ),
            )

        self.lora_B.is_distributed = True
        self.lora_B.split_axis = 1
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self.disable_lora = False

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        state_dict = self.state_dict(structured_name_prefix="")
        return build_sharded_state_dict(state_dict, {"weight": 1, "bias": 0, "lora_B": 1}, structured_name_prefix)

    def get_delta_weight(self, lora_A=None, lora_B=None):
        lora_A = lora_A if lora_A is not None else self.lora_A
        lora_B = lora_B if lora_B is not None else self.lora_B
        delta_weight = lora_A @ lora_B * self.scaling

        return delta_weight

    def unmerge(self):
        if self.merged:
            # Make sure that the weights are not merged
            delta_weight = self.get_delta_weight()
            new_weight = self.weight - delta_weight
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            # Merge the weights and mark it
            delta_weight = self.get_delta_weight()
            new_weight = self.weight + delta_weight
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor):
        if self.disable_lora or self.merged:
            if MC2ColumnParallelCoreLinear is None:
                input_mp = mp_ops._c_identity(input, group=self.model_parallel_group)
                result_mp = F.linear(x=input_mp, weight=self.weight, bias=self.bias, name=self.name)
            else:
                res_mp = MC2ColumnParallelCoreLinear.apply(input, self.weight, self.model_parallel_group)
                result_mp = (res_mp + self.bias) if self.bias is not None else res_mp
        else:
            if MC2ColumnParallelCoreLinear is None:
                input_mp = mp_ops._c_identity(input, group=self.model_parallel_group)
                result_mp = F.linear(x=input_mp, weight=self.weight, bias=self.bias, name=self.name)
            else:
                res_mp = MC2ColumnParallelCoreLinear.apply(input, self.weight, self.model_parallel_group)
                result_mp = (res_mp + self.bias) if self.bias is not None else res_mp

            input_a = self.lora_dropout(input) @ self.lora_A
            if MC2ColumnParallelCoreLinear is None:
                input_a_mp = mp_ops._c_identity(input_a, group=self.model_parallel_group)
                delta_mp = (input_a_mp @ self.lora_B) * self.scaling
            else:
                tmp = MC2ColumnParallelCoreLinear.apply(input_a, self.lora_B, self.model_parallel_group)
                delta_mp = tmp * self.scaling
            result_mp += delta_mp

        if self.gather_output and self.is_mp:
            result = mp_ops._c_concat(result_mp, group=self.model_parallel_group)
        else:
            result = result_mp
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class FleetColumnParallelLoRALinear(ColumnParallelLoRALinear):
    def __init__(self, in_features, out_features, skip_bias_add, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.skip_bias_add = skip_bias_add

    def forward(self, input: paddle.Tensor):
        out_bias = self.bias if self.skip_bias_add else None
        if self.skip_bias_add:
            self.bias = None
        output = super().forward(input)
        return output, out_bias


class ColumnSequenceParallelLoRALinear(ColumnSequenceParallelLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        lora_A_weight_attr: Optional[paddle.ParamAttr] = None,
        **kwargs
    ):
        ColumnSequenceParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # compatible
        self.name = self._name

        # Actual trainable parameters
        self.lora_A = self.create_parameter(
            shape=[in_features, r],
            dtype=self._dtype,
            is_bias=False,
            attr=lora_A_weight_attr,
        )
        self.lora_A.is_distributed = False
        mark_as_sequence_parallel_parameter(self.lora_A)

        with rng_ctx(self.is_mp, paddle.in_dynamic_mode()):
            self.lora_B = self.create_parameter(
                shape=[r, self.output_size_per_partition],
                dtype=self._dtype,
                is_bias=False,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.0),
                    learning_rate=lora_plus_scale,
                ),
            )

        self.lora_B.is_distributed = True
        self.lora_B.split_axis = 1
        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self.disable_lora = False

    def sharded_state_dict(
        self,
        structured_name_prefix: str = "",
    ):
        state_dict = self.state_dict(structured_name_prefix="")
        return build_sharded_state_dict(state_dict, {"weight": 1, "bias": 0, "lora_B": 1}, structured_name_prefix)

    def get_delta_weight(self, lora_A=None, lora_B=None):
        lora_A = lora_A if lora_A is not None else self.lora_A
        lora_B = lora_B if lora_B is not None else self.lora_B
        delta_weight = lora_A @ lora_B * self.scaling

        return delta_weight

    def unmerge(self):
        if self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight - delta_weight
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            delta_weight = self.get_delta_weight()
            new_weight = self.weight + delta_weight
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, x: paddle.Tensor):
        if MC2ColumnSeqParallelCoreLinear is None:
            if self.is_mp:
                input_parallel = AllGatherOp.apply(x)
            else:
                input_parallel = x
            result_mp = self.linear(input_parallel, self.weight, self.bias, name=self._name)
        else:
            result_mp = MC2ColumnSeqParallelCoreLinear.apply(x, self.weight, self.model_parallel_group)
            if self.bias is not None:
                result_mp += self.bias

        if not self.merged and not self.disable_lora:
            input_a = self.lora_dropout(x) @ self.lora_A
            # TODO(@gexiao): temporary workaround for deterministic calculation
            if True or MC2ColumnSeqParallelCoreLinear is None:
                input_a = AllGatherOp.apply(input_a)
                delta_mp = (input_a @ self.lora_B) * self.scaling
            else:
                input_a = MC2ColumnSeqParallelCoreLinear.apply(input_a, self.lora_B, self.model_parallel_group)
                delta_mp = input_a * self.scaling
            result_mp += delta_mp

        if self.gather_output and self.is_mp:
            result = mp_ops._c_concat(result_mp, group=self.model_parallel_group)
        else:
            result = result_mp
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


class FleetColumnSequenceParallelLoRALinear(ColumnSequenceParallelLoRALinear):
    def __init__(self, in_features, out_features, skip_bias_add, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.skip_bias_add = skip_bias_add

    def forward(self, input: paddle.Tensor):
        out_bias = self.bias if self.skip_bias_add else None
        if self.skip_bias_add:
            self.bias = None
        output = super().forward(input)
        return output, out_bias


class LoRAConv2D(nn.Conv2D):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs
    ):
        nn.Conv2D.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # Actual trainable parameters
        lora_A = nn.Conv2D(
            in_channels,
            r,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            weight_attr=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
            bias_attr=False,
        )
        self.lora_A = lora_A.weight
        self.lora_A_forward = lambda x: nn.Conv2D.__call__(lora_A, x)
        lora_B = nn.Conv2D(
            r,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            weight_attr=nn.initializer.Constant(value=0.0),
            bias_attr=False,
        )
        self.lora_B_forward = lambda x: nn.Conv2D.__call__(lora_B, x)
        self.lora_B = lora_B.weight
        self.scaling = lora_alpha / r

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        if self.bias is not None:
            self.bias.stop_gradient = True
        self.disable_lora = False

    def get_delta_weight(self, lora_A=None, lora_B=None):
        weight_A = (lora_A if lora_A else self.lora_A).cast(dtype=self.weight.dtype)
        weight_B = (lora_B if lora_B else self.lora_B).cast(dtype=self.weight.dtype)

        if self.weight.shape[2:4] == [1, 1]:
            # conv2d 1x1
            delta_weight = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling
        else:
            # conv2d 3x3
            delta_weight = (
                F.conv2d(
                    weight_A.transpose([1, 0, 2, 3]),
                    weight_B,
                ).transpose([1, 0, 2, 3])
                * self.scaling
            )

        return delta_weight

    def unmerge(self):
        if self.merged:
            delta_weight = self.get_delta_weight()
            # Make sure that the weights are not merged
            new_weight = self.weight - delta_weight
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            delta_weight = self.get_delta_weight()
            # Merge the weights and mark it
            new_weight = self.weight + delta_weight
            self.weight.set_value(new_weight)
            self.merged = True

    def forward(self, input: paddle.Tensor, *args, **kwargs):
        previous_dtype = input.dtype
        result = super().forward(input)
        if not self.merged and not self.disable_lora:
            result += (
                self.lora_B_forward(self.lora_A_forward(self.lora_dropout(input.cast(dtype=self.lora_A.dtype))))
                * self.scaling
            )
        result = result.cast(dtype=previous_dtype)
        return result

    def extra_repr(self):
        main_str = "{_in_channels}, {_out_channels}, kernel_size={_kernel_size}"
        if self._stride != [1] * len(self._stride):
            main_str += ", stride={_stride}"
        if self._padding != 0:
            main_str += ", padding={_padding}"
        if self._padding_mode != "zeros":
            main_str += ", padding_mode={_padding_mode}"
        if self.output_padding != 0:
            main_str += ", output_padding={output_padding}"
        if self._dilation != [1] * len(self._dilation):
            main_str += ", dilation={_dilation}"
        if self._groups != 1:
            main_str += ", groups={_groups}"
        main_str += ", data_format={_data_format}, rank={r}, alpha={lora_alpha}"
        return main_str.format(**self.__dict__)


class LoRAMoeExperts(MoeExpertsBase):
    def __init__(
        self,
        base_layer,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.num_experts = base_layer.num_experts
        self.act_fn = base_layer.act_fn
        self.r = r
        self.lora_alpha = lora_alpha
        self.merged = False
        self.disable_lora = False
        self.lora_plus_scale = lora_plus_scale

        self.gate_up_proj, self.gate_up_proj_lora_A, self.gate_up_proj_lora_B = self._init_lora(
            base_layer, "gate_up_proj"
        )
        self.down_proj, self.down_proj_lora_A, self.down_proj_lora_B = self._init_lora(base_layer, "down_proj")

        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

    def _init_lora(self, base_layer, parameter_name: str):
        if not hasattr(base_layer, parameter_name):
            raise ValueError(f"Parameter '{parameter_name}' does not exist in the base layer.")

        parameter = getattr(base_layer, parameter_name)
        parameter.stop_gradient = True
        num_experts, in_features, out_features = parameter.shape
        lora_A = self.create_parameter(
            shape=[num_experts, in_features, self.r],
            dtype=paddle.get_default_dtype(),
            is_bias=False,
            default_initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
        )
        lora_B = self.create_parameter(
            shape=[num_experts, self.r, out_features],
            dtype=paddle.get_default_dtype(),
            is_bias=False,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0),
                learning_rate=self.lora_plus_scale,
            ),
        )

        return (parameter, lora_A, lora_B)

    def get_delta_weight(self, lora_A, lora_B):
        return lora_A @ lora_B * self.scaling

    def merge(self):
        if not self.merged:
            delta_weight = self.get_delta_weight(self.gate_up_proj_lora_A, self.gate_up_proj_lora_B)
            new_parameter = self.gate_up_proj + delta_weight
            self.gate_up_proj.set_value(new_parameter)
            delta_weight = self.get_delta_weight(self.down_proj_lora_A, self.down_proj_lora_B)
            new_parameter = self.down_proj + delta_weight
            self.down_proj.set_value(new_parameter)
            self.merged = True

    def unmerge(self):
        if self.merged:
            delta_weight = self.get_delta_weight(self.gate_up_proj_lora_A, self.gate_up_proj_lora_B)
            new_parameter = self.gate_up_proj - delta_weight
            self.gate_up_proj.set_value(new_parameter)
            delta_weight = self.get_delta_weight(self.down_proj_lora_A, self.down_proj_lora_B)
            new_parameter = self.down_proj - delta_weight
            self.down_proj.set_value(new_parameter)
            self.merged = False

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = paddle.zeros_like(hidden_states)
        with paddle.no_grad():
            expert_mask = paddle.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = paddle.greater(expert_mask.sum(dim=(-1, -2)), paddle.to_tensor(0, dtype="int32")).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = paddle.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            if not (self.disable_lora or self.merged):
                delta_state = (
                    current_state
                    @ self.gate_up_proj_lora_A[expert_idx]
                    @ self.gate_up_proj_lora_B[expert_idx]
                    * self.scaling
                )
                current_state = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]) + delta_state
            else:
                current_state = nn.functional.linear(current_state, self.gate_up_proj[expert_idx])
            gate, up = current_state.chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            if not (self.disable_lora or self.merged):
                delta_states = (
                    current_hidden_states
                    @ self.down_proj_lora_A[expert_idx]
                    @ self.down_proj_lora_B[expert_idx]
                    * self.scaling
                )
                current_hidden_states = (
                    nn.functional.linear(current_hidden_states, self.down_proj[expert_idx]) + delta_states
                )
            else:
                current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class FleetLoRAMoeExperts(MoeExpertsBase):
    def __init__(
        self,
        base_layer,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.config = base_layer.config
        self.activation_func = base_layer.activation_func
        self.expert_parallel = base_layer.expert_parallel
        self.ep_group = base_layer.ep_group
        self.moe_deep_gemm = base_layer.moe_deep_gemm
        self.activation_recompute = base_layer.activation_recompute
        self.r = r
        self.lora_alpha = lora_alpha
        self.merged = False
        self.disable_lora = False
        self.lora_plus_scale = lora_plus_scale

        self.weight1, self.weight1_lora_A, self.weight1_lora_B = self._init_lora(base_layer, "weight1")
        self.weight2, self.weight2_lora_A, self.weight2_lora_B = self._init_lora(base_layer, "weight2")

        if not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

    def _init_lora(self, base_layer, parameter_name: str):
        if not hasattr(base_layer, parameter_name):
            raise ValueError(f"Parameter '{parameter_name}' does not exist in the base layer.")

        parameter = getattr(base_layer, parameter_name)
        parameter.stop_gradient = True
        num_experts, in_features, out_features = parameter.shape
        lora_A = paddle.create_parameter(
            shape=[num_experts, in_features, self.r],
            dtype=parameter.dtype,
            is_bias=False,
            default_initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu"),
        )
        lora_B = paddle.create_parameter(
            shape=[num_experts, self.r, out_features],
            dtype=parameter.dtype,
            is_bias=False,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.0),
                learning_rate=self.lora_plus_scale,
            ),
        )
        lora_A.is_distributed = self.expert_parallel
        lora_B.is_distributed = self.expert_parallel

        return (parameter, lora_A, lora_B)

    def get_delta_weight(self, lora_A, lora_B):
        return lora_A @ lora_B * self.scaling

    def merge(self):
        if not self.merged:
            delta_weight = self.get_delta_weight(self.weight1_lora_A, self.weight1_lora_B)
            new_parameter = self.weight1 + delta_weight
            self.weight1.set_value(new_parameter)
            delta_weight = self.get_delta_weight(self.weight2_lora_A, self.weight2_lora_B)
            new_parameter = self.weight2 + delta_weight
            self.weight2.set_value(new_parameter)
            self.merged = True

    def unmerge(self):
        if self.merged:
            delta_weight = self.get_delta_weight(self.weight1_lora_A, self.weight1_lora_B)
            new_parameter = self.weight1 - delta_weight
            self.weight1.set_value(new_parameter)
            delta_weight = self.get_delta_weight(self.weight2_lora_A, self.weight2_lora_B)
            new_parameter = self.weight2 - delta_weight
            self.weight2.set_value(new_parameter)
            self.merged = False

    def sharded_state_dict(self, structured_name_prefix: str = ""):
        state_dict = self.state_dict(structured_name_prefix="")
        if self.ep_group is None:
            return build_sharded_state_dict(state_dict, None, structured_name_prefix)

        sharded_dict = {}
        lora_keys = [
            "weight1_lora_A",
            "weight1_lora_B",
            "weight2_lora_A",
            "weight2_lora_B",
        ]
        for short_key, tensor in state_dict.items():
            full_key = f"{structured_name_prefix}{short_key}"
            if short_key in lora_keys:
                sharded_dict[full_key] = shard_weight(
                    key=full_key,
                    weight=tensor,
                    axis=0,
                    group=self.ep_group,
                )
            else:
                # weight1/weight2 (base, stop_gradient=True) — replicate as-is
                from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
                    make_replicated_sharded_weight,
                )

                sharded_dict[full_key] = make_replicated_sharded_weight(full_key, tensor)
        return sharded_dict

    def forward(
        self,
        permuted_local_hidden_states: paddle.Tensor,
        tokens_per_expert: paddle.Tensor,
    ):
        """Forward step of the GroupedMLP without TP/DP."""

        def apply_lora(base_weight, lora_A, lora_B):
            if self.disable_lora or self.merged:
                return base_weight
            else:
                return base_weight + self.get_delta_weight(lora_A, lora_B)

        w1 = apply_lora(self.weight1, self.weight1_lora_A, self.weight1_lora_B)
        w2 = apply_lora(self.weight2, self.weight2_lora_A, self.weight2_lora_B)

        if permuted_local_hidden_states.numel() != 0:
            if not isinstance(tokens_per_expert, list):
                tokens_per_expert = tokens_per_expert.cpu().tolist()
            tokens_per_expert = [int(x) for x in tokens_per_expert]
            tokens_per_expert_tensor = paddle.to_tensor(tokens_per_expert, dtype="int32")

            if self.moe_deep_gemm:
                fc1_output = DeepGEMMBMMFunction.apply(
                    permuted_local_hidden_states,
                    w1,
                    tokens_per_expert_tensor,
                )
            else:
                fc1_output = BMMFunction.apply(
                    permuted_local_hidden_states,
                    w1,
                    tokens_per_expert,
                )

            if self.activation_recompute:
                raise NotImplementedError("Recompute in GroupedMLPExpert is not implemented")
            else:
                intermediate_parallel = self.activation_func(fc1_output)
                if self.moe_deep_gemm:
                    fc2_output = DeepGEMMBMMFunction.apply(
                        intermediate_parallel,
                        w2,
                        tokens_per_expert_tensor,
                    )
                else:
                    fc2_output = BMMFunction.apply(intermediate_parallel, w2, tokens_per_expert)
        else:
            # No token is allocated for local experts.
            assert paddle.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = w1.reshape(self.config.hidden_size, -1)
            w2 = w2.reshape(-1, self.config.hidden_size)
            h = paddle.matmul(permuted_local_hidden_states, w1)
            if self.activation_recompute:
                raise NotImplementedError("Recompute in GroupedMLPExpert is not implemented")
            else:
                h = self.activation_func(h)
                fc2_output = paddle.matmul(h, w2)

        return fc2_output, None
