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

from ...transformers import linear_utils

ColumnSequenceParallelLinear = linear_utils.ColumnSequenceParallelLinear
RowSequenceParallelLinear = linear_utils.RowSequenceParallelLinear

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        AllGatherOp,
        ReduceScatterOp,
        mark_as_sequence_parallel_parameter,
    )
except:
    AllGatherOp = None
    ReduceScatterOp = None
    mark_as_sequence_parallel_parameter = None

from ...transformers.mc2_parallel_linear import (
    MC2ColumnParallelCoreLinear,
    MC2ColumnSeqParallelCoreLinear,
    MC2RowParallelCoreLinear,
    MC2RowSeqParallelCoreLinear,
)
from .lora_quick_layers import quick_lora
from .utils import rng_ctx


class LoRALinear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_quick_lora: bool = False,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        pissa: bool = False,
        lora_use_mixer: bool = False,
        use_mora: bool = False,
        lorapro: bool = False,
        mp_moe: bool = False,
        is_distributed: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")
        self.use_mora = use_mora
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.pissa = pissa
        self.lora_use_mixer = lora_use_mixer
        self.lorapro = lorapro

        # Actual trainable parameters
        if use_mora:  # reset the rank and create high rank matrix
            self.in_features = in_features
            self.out_features = out_features
            new_r = int(math.sqrt((in_features + out_features) * r) + 0.5)
            new_r = new_r // 2 * 2
            self.r = new_r
            self.lora_A = self.create_parameter(
                shape=[self.r, self.r],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.Constant(value=0.0),
            )
            self.cos = None
            self.sin = None
            # Count the number of tiles
            self.rb1 = self.in_features // self.r if self.in_features % self.r == 0 else self.in_features // self.r + 1
            self.rb2 = (
                self.out_features // self.r if self.out_features % self.r == 0 else self.out_features // self.r + 1
            )
            self.rope_init()
        else:
            self.lora_A = self.create_parameter(
                shape=[in_features, r],
                dtype=self._dtype,
                is_bias=False,
                default_initializer=nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                ),
            )
            if self.lora_use_mixer:
                self.lora_AB = self.create_parameter(
                    shape=[r, r],
                    dtype=self._dtype,
                    is_bias=False,
                    default_initializer=nn.initializer.KaimingUniform(
                        negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
                    ),
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
        self.apply_pissa = False
        if use_mora or pissa:
            self.scaling = 1.0
        elif not rslora:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling = self.lora_alpha / math.sqrt(self.r)

        # Freezing the pre-trained weight matrix
        self.weight.stop_gradient = True
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False
        if mp_moe or is_distributed:
            for p in self.parameters():
                p.is_distributed = is_distributed
                p.mp_moe = mp_moe

    def pissa_init(self, rank):
        weight = self.weight
        dtype = weight.dtype
        if dtype != paddle.float32:
            weight = weight.astype(paddle.float32)

        U, S, Vh = paddle.linalg.svd(weight.data, full_matrices=False)
        Ur = U[:, :rank]
        Sr = S[:rank]
        Vhr = Vh[:rank]

        lora_A = Ur @ paddle.diag(paddle.sqrt(Sr))
        lora_B = paddle.diag(paddle.sqrt(Sr)) @ Vhr
        self.lora_A.set_value(lora_A.astype(dtype))
        self.lora_B.set_value(lora_B.astype(dtype))
        res = weight.data - lora_A @ lora_B
        weight = res.astype(dtype)
        self.weight.set_value(weight)

    def rope_init(self):
        if self.cos is None or self.sin is None:
            inv_freq = 1.0 / (10000 ** (paddle.arange(0, self.r, 2, dtype=paddle.float32) / self.r))
            t = paddle.arange(self.rb1, dtype=paddle.float32)
            freqs = t.unsqueeze(1) @ inv_freq.unsqueeze(0)
            emb = paddle.concat([freqs, freqs], axis=-1)
            self.cos = paddle.unsqueeze(paddle.cos(emb), axis=0).astype(self._dtype)
            self.sin = paddle.unsqueeze(paddle.sin(emb), axis=0).astype(self._dtype)

    @property
    def use_quick_lora(self):
        return self._use_quick_lora and self.training and not self.merged

    def _apply_mora(self, x):
        r = self.r

        # Calculate grouping
        sum_inter = self.in_features // r

        # padding
        if self.in_features % r != 0:
            pad_size = r - self.in_features % r
            x = paddle.concat([x, x[..., :pad_size]], axis=-1)
            sum_inter += 1

        # reshape the input to apply RoPE
        in_x = x.reshape([*x.shape[:-1], sum_inter, r])

        # apply RoPE rotation
        rh_in_x = paddle.concat([-in_x[..., r // 2 :], in_x[..., : r // 2]], axis=-1)
        in_x = in_x * self.cos + rh_in_x * self.sin

        # matmul with high rank matrix
        out_x = in_x @ self.lora_A

        # reshape the output
        out_x = out_x.reshape([*x.shape[:-1], -1])[..., : self.out_features]
        if out_x.shape[-1] < self.out_features:
            repeat_time = self.out_features // out_x.shape[-1]
            if self.out_features % out_x.shape[-1] != 0:
                repeat_time += 1
            out_x = paddle.concat([out_x] * repeat_time, axis=-1)[..., : self.out_features]

        return out_x

    def get_delta_weight(self, lora_A=None, lora_B=None, lora_AB=None):
        # compute the delta weight，which is used to merge weights
        if self.lora_use_mixer:
            lora_A = lora_A if lora_A is not None else self.lora_A
            lora_B = lora_B if lora_B is not None else self.lora_B
            lora_AB = lora_AB if lora_AB is not None else self.lora_AB
            delta_weight = lora_A @ lora_AB @ lora_B * self.scaling
        elif self.use_mora:
            lora_A = lora_A if lora_A is not None else self.lora_A
            r = self.r
            # compute padding
            pad_size = r - self.in_features % r if self.in_features % r != 0 else 0
            # initialize weights
            w = paddle.zeros([self.in_features + pad_size, self.in_features], dtype=lora_A.dtype)

            # create the weights after rotation
            aw2 = paddle.concat([lora_A[:, r // 2 :], -lora_A[:, : r // 2]], axis=-1)
            # apply RoPE
            for i in range(self.rb1 - 1):
                w[i * r : (i + 1) * r, i * r : (i + 1) * r] = aw2 * self.sin[:, i] + lora_A * self.cos[:, i]
            # Process the last chunk that may be incomplete
            i = self.rb1 - 1
            w[i * r :, i * r :] = (aw2 * self.sin[:, i] + lora_A * self.cos[:, i])[:, : r - pad_size]
            # padding
            if pad_size > 0:
                w[i * r :, :pad_size] = (aw2 * self.sin[:, i] + lora_A * self.cos[:, i])[:, r - pad_size :]
            # reshape the weights
            if self.in_features < self.out_features:
                w = paddle.concat([w] * self.rb2, axis=0)[: self.out_features]
            else:
                w = w[: self.out_features]
            final_weight = w
            delta_weight = final_weight.T
        else:
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
        if not self.apply_pissa and self.pissa:
            self.pissa_init(self.r)
            self.apply_pissa = True
        if self.disable_lora or self.merged:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
        elif self.use_quick_lora:
            # Use the quick lora implementation
            result = quick_lora(input, self.lora_A, self.lora_B, self.weight, self.bias, self.scaling)
        elif self.use_mora:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
            input = self.lora_dropout(input)
            mora_out = self._apply_mora(input)
            result += mora_out
        else:
            result = F.linear(x=input, weight=self.weight, bias=self.bias, name=self.name)
            if self.lora_use_mixer:
                result += (self.lora_dropout(input) @ self.lora_A @ self.lora_AB @ self.lora_B) * self.scaling
            else:
                result += (self.lora_dropout(input) @ self.lora_A @ self.lora_B) * self.scaling
        return result

    def extra_repr(self):
        name = f", name={self.name}" if self.name else ""
        return f"in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]}, rank={self.r}{name}"


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
        use_quick_lora: bool = False,
        pissa: bool = False,
        use_mora: bool = False,
        **kwargs
    ):
        RowParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")

        if pissa or use_mora:
            raise ValueError("Pissa or Mora is not supported in model parallel by now")

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
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        return self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
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
        elif self.use_quick_lora:
            # Use the quick lora implementation
            result_mp = quick_lora(
                input_mp,
                self.lora_A,
                self.lora_B,
                self.weight,
                self.bias,
                self.scaling,
                is_row=True,
                group=self.model_parallel_group,
                world_size=self.world_size,
            )
            output = mp_ops._mp_allreduce(
                result_mp,
                group=self.model_parallel_group,
                use_calc_stream=True,
                use_model_parallel=True,
            )
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
        use_quick_lora: bool = False,
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
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        # TODO(@gexiao): support qlora
        return False  # self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
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
        use_quick_lora: bool = False,
        pissa: bool = False,
        use_mora: bool = False,
        **kwargs
    ):
        ColumnParallelLinear.__init__(self, in_features, out_features, **kwargs)
        if not isinstance(r, int) or r <= 0:
            raise ValueError("Lora rank r should be a positive integer")

        if pissa or use_mora:
            raise ValueError("Pissa or Mora is not supported in model parallel by now")

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
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        return self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            # Make sure that the weights are not merged
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            # Merge the weights and mark it
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
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

        elif self.use_quick_lora:
            # Use the quick lora implementation
            input_mp = mp_ops._c_identity(input, group=self.model_parallel_group) if self.is_mp else input
            result_mp = quick_lora(
                input_mp,
                self.lora_A,
                self.lora_B,
                self.weight,
                self.bias,
                self.scaling,
                is_column=True,
                group=self.model_parallel_group,
                world_size=self.world_size,
            )
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
        use_quick_lora: bool = False,
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
        self._use_quick_lora = use_quick_lora and lora_dropout == 0.0
        self.disable_lora = False

    @property
    def use_quick_lora(self):
        # TODO(@gexiao): support qlora
        return False  # self._use_quick_lora and self.training and not self.merged

    def unmerge(self):
        if self.merged:
            new_weight = self.weight - self.lora_A @ self.lora_B * self.scaling
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            new_weight = self.weight + self.lora_A @ self.lora_B * self.scaling
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

    def unmerge(self):
        if self.merged:
            weight_A = self.lora_A.cast(dtype=self.weight.dtype)
            weight_B = self.lora_B.cast(dtype=self.weight.dtype)
            if self.weight.shape[2:4] == [1, 1]:
                # conv2d 1x1
                delta_weight = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(
                    2
                ).unsqueeze(3) * self.scaling
            else:
                # conv2d 3x3
                delta_weight = (
                    F.conv2d(
                        weight_A.transpose([1, 0, 2, 3]),
                        weight_B,
                    ).transpose([1, 0, 2, 3])
                    * self.scaling
                )
            # Make sure that the weights are not merged
            new_weight = self.weight - delta_weight
            self.weight.set_value(new_weight)
            self.merged = False

    def merge(self):
        if not self.merged:
            weight_A = self.lora_A.cast(dtype=self.weight.dtype)
            weight_B = self.lora_B.cast(dtype=self.weight.dtype)
            if self.weight.shape[2:4] == [1, 1]:
                # conv2d 1x1
                delta_weight = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(
                    2
                ).unsqueeze(3) * self.scaling
            else:
                # conv2d 3x3
                delta_weight = (
                    F.conv2d(
                        weight_A.transpose([1, 0, 2, 3]),
                        weight_B,
                    ).transpose([1, 0, 2, 3])
                    * self.scaling
                )
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
