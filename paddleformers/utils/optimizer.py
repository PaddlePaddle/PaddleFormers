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


import numpy as np
import paddle
from paddle import pir
from paddle.base import core, framework
from paddle.base.framework import Variable, in_dynamic_or_pir_mode, in_pir_mode
from paddle.base.libpaddle import DataType
from paddle.distributed import fleet
from paddle.optimizer.adamw import AdamW
from paddle.pir import Value

try:
    from .adamw_triton import adamw_triton
except:
    adamw_triton = None


from ..quantization.qat_utils import dequantize, quantize
from .accuracy_target import ACCURACY_TARGET_HF as _ACCURACY_TARGET_HF


def _f32(value):
    """Round a python scalar the way a CUDA kernel's ``opmath_t`` cast does."""
    return float(np.float32(value))


def _fma(a, b, c):
    """FP32 fused multiply-add: ``a * b + c`` with a single rounding.

    torch's ``_foreach_addcdiv_`` kernel contracts its multiply-add into an FMA,
    which matters whenever the update nearly cancels the parameter -- exactly the
    elements where a separate multiply and add lands one BF16 ULP away. Paddle
    has no FMA primitive, so the intermediate is carried in FP64: the FP32
    product is exact there (48 mantissa bits) and the addend is FP32, so the sum
    rounds once on the way back to FP32.
    """
    return (a.astype("float64") * b.astype("float64") + c.astype("float64")).astype("float32")


def _hf_bitexact_adamw_step(
    param,
    grad,
    moment1,
    moment2,
    master_weight,
    *,
    lr,
    beta1,
    beta2,
    epsilon,
    weight_decay,
    step,
):
    """One AdamW step matching torch's ``_multi_tensor_adamw`` bit-for-bit.

    For a BF16 parameter torch keeps ``exp_avg`` / ``exp_avg_sq`` in BF16 and every
    ``_foreach_*`` call is "FP32 opmath, one round back to BF16". PaddleFormers
    keeps an FP32 master weight and FP32 moments, so it carries extra precision
    the reference has already discarded and the two trajectories separate after a
    few steps. Writing BF16-rounded values into those FP32 buffers reproduces the
    reference exactly and leaves the buffer dtypes -- and therefore the optimizer
    checkpoint layout -- untouched.

    ``step`` is the 1-based update count. torch derives the bias corrections from
    ``beta ** step`` in python doubles, whereas paddle's ``beta_pow`` accumulators
    reach the same power through repeated FP32 multiplication, which is not the
    same value; the caller therefore passes the count explicitly.
    """
    work = param.dtype
    with paddle.amp.auto_cast(False):
        current = master_weight if master_weight is not None else param
        p = current.astype(work)
        if weight_decay != 0.0:
            # _foreach_mul_(params, 1 - lr * weight_decay)
            p = (p.astype("float32") * _f32(1.0 - lr * weight_decay)).astype(work)

        g = grad.astype("float32")

        # _foreach_lerp_(exp_avg, grad, 1 - beta1); |weight| < 0.5 uses
        # self + weight * (end - self), and nvcc contracts that into an FMA.
        # Without the contraction the moment lands one ULP off wherever
        # ``beta1 * m`` and ``(1 - beta1) * g`` nearly cancel (first seen at step 2
        # on one element of layer 0's post-attention norm, where torch keeps
        # 1.53e-12 and an unfused evaluation collapses to exactly 0).
        m = moment1.astype("float32")
        m = _fma(
            paddle.full_like(m, _f32(1.0 - beta1)),
            g - m,
            m,
        ).astype(work)

        # _foreach_mul_(exp_avg_sq, beta2)
        v = (moment2.astype("float32") * _f32(beta2)).astype(work)
        # _foreach_addcmul_(exp_avg_sq, grad, grad, 1 - beta2), also contracted:
        # the squared gradient is formed first, then a single fused multiply-add
        # folds in the running value. Verified against a real two-step trajectory
        # (``tools/search_adamw_lerp.py``); both unfused groupings miss one element
        # of layer 0's shared-expert down projection.
        v = _fma(
            paddle.full_like(g, _f32(1.0 - beta2)),
            g * g,
            v.astype("float32"),
        ).astype(work)

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        step_size = _f32((lr / bias_correction1) * -1)
        bias_correction2_sqrt = _f32(bias_correction2**0.5)

        # _foreach_sqrt / _foreach_div_ / _foreach_add_ each allocate a tensor of
        # the moment dtype, so each one rounds.
        denom = paddle.sqrt(v.astype("float32")).astype(work)
        denom = (denom.astype("float32") / bias_correction2_sqrt).astype(work)
        denom = (denom.astype("float32") + _f32(epsilon)).astype(work)

        # _foreach_addcdiv_(params, exp_avg, denom, step_size)
        p = _fma(
            paddle.full_like(m, step_size, dtype="float32"),
            m.astype("float32") / denom.astype("float32"),
            p.astype("float32"),
        ).astype(work)
    return p, m, v


class AdamWMini(AdamW):
    def _add_moments_pows(self, p):
        acc_dtype = p.dtype
        if self._is_dtype_fp16_or_bf16(acc_dtype):
            acc_dtype = DataType.FLOAT32 if in_pir_mode() else paddle.float32

        self._add_accumulator(self._moment1_acc_str, p, dtype=acc_dtype)
        # change moment2
        self._add_accumulator(self._moment2_acc_str, p, dtype=acc_dtype, shape=[1])
        try:
            type = core.VarDesc.VarType.DENSE_TENSOR
        except:
            type = core.VarDesc.VarType.LOD_TENSOR
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.9 if isinstance(self._beta1, (Variable, Value)) else self._beta1,
            shape=[1],
            type=type,
            device="cpu",
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=0.999 if isinstance(self._beta2, (Variable, Value)) else self._beta2,
            shape=[1],
            type=type,
            device="cpu",
        )

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param = param_and_grad[0]

        # Whether we should do weight decay for the parameter.
        with_decay = True
        if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param.name):
            with_decay = False

        moment1 = self._get_accumulator_master(self._moment1_acc_str, param_and_grad[0])
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param_and_grad[0])
        beta1_pow_acc = self._get_accumulator_master(self._beta1_pow_acc_str, param_and_grad[0])
        beta2_pow_acc = self._get_accumulator_master(self._beta2_pow_acc_str, param_and_grad[0])
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
        master_weight = self._master_weights[param_and_grad[0].name] if find_master else None
        lr = self._create_param_lr(param_and_grad)
        # create the adamw optimize op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

            _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
            _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)

            found_inf = self._get_auxiliary_var("found_inf") if in_pir_mode() else None
            self.adamw_python(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                moment1,
                moment2,
                beta1_pow_acc,
                beta2_pow_acc,
                master_weight,
                found_inf,
                _beta1,
                _beta2,
                self._epsilon,
                lr_ratio_,
                self._weight_decay,
                with_decay,
                find_master,
            )
            return None
        else:
            raise NotImplementedError("Not implemented yet.")

    def adamw_python(
        self,
        param,
        grad,
        learning_rate,
        moment1,
        moment2,
        beta1_pow,
        beta2_pow,
        master_weight,
        skip_update,
        beta1,
        beta2,
        epsilon,
        lr_ratio,
        coeff,
        with_decay,
        multi_precision,
    ):
        if skip_update:
            return
        if not with_decay:
            coeff = 0.0
        if not multi_precision:
            master_weight = None
        lr = learning_rate * lr_ratio
        if master_weight is not None:
            p = master_weight
        else:
            p = param
        p *= 1.0 - lr * coeff
        mom1 = moment1
        mom2 = moment2

        mom1 = beta1 * mom1 + (1.0 - beta1) * grad
        mom2 = beta2 * mom2 + (1.0 - beta2) * (grad * grad).mean()
        denom = mom2.sqrt() / (1.0 - beta2_pow).sqrt() + epsilon
        p += (mom1 / denom) * (-(lr / (1.0 - beta1_pow)))
        if master_weight is not None:
            master_weight[:] = p
            param[:] = p.astype(param.dtype)
        else:
            param[:] = p
        moment1[:] = mom1
        moment2[:] = mom2
        beta1_pow[:], beta2_pow[:] = beta1 * beta1_pow[:], beta2 * beta2_pow[:]
        return


class AdamWCustom(AdamW):
    def __init__(
        self,
        quantization_config,
        tensorwise_offload_optimizer,
        *args,
        accuracy_target=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_scale_mapping = {}
        for p in self._param_groups:
            if "quantization_linear" in p.name and "w_1" in p.name:
                self.weight_scale_mapping[p.name.replace("w_1", "w_0")] = p
        self.quantization_config = quantization_config
        if paddle.distributed.get_world_size() > 1:
            self._hcg = fleet.get_hybrid_communicate_group()
            self.mp_group = self._hcg.get_model_parallel_group()
        else:
            self.mp_group = None

        self.tensorwise_offload_optimizer = tensorwise_offload_optimizer
        #: Which reference this optimizer reproduces bit-for-bit. Comes from
        #: ``config.use_accuracy_compatible`` via the trainer, so the single
        #: config field stays the only source of truth.
        self.accuracy_target = accuracy_target
        self.hf_bitexact = accuracy_target == _ACCURACY_TARGET_HF

    def _add_moments_pows(self, p, moment_dtype=core.VarDesc.VarType.FP32):
        acc_dtype = p.dtype

        self._add_accumulator(self._moment1_acc_str, p, dtype=moment_dtype)
        self._add_accumulator(self._moment2_acc_str, p, dtype=moment_dtype)
        try:
            type = core.VarDesc.VarType.DENSE_TENSOR
        except:
            type = core.VarDesc.VarType.LOD_TENSOR
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=(0.9 if isinstance(self._beta1, (Variable, Value)) else self._beta1),
            shape=[1],
            type=type,
        )
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            dtype=acc_dtype,
            fill_value=(0.999 if isinstance(self._beta2, (Variable, Value)) else self._beta2),
            shape=[1],
            type=type,
        )

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            if p.name in self._already_create_accumulator:
                continue
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                if self._use_lowprecision_moment:
                    if p.name in self.weight_scale_mapping:
                        p_scale = self.weight_scale_mapping[p.name]
                        if str(p_scale.dtype) == "paddle.float16":
                            moment_dtype = core.VarDesc.VarType.FP16
                        elif str(p_scale.dtype) == "paddle.bfloat16":
                            moment_dtype = core.VarDesc.VarType.BF16
                    else:
                        if str(p.dtype) == "paddle.float16":
                            moment_dtype = core.VarDesc.VarType.FP16
                        elif str(p.dtype) == "paddle.bfloat16":
                            moment_dtype = core.VarDesc.VarType.BF16
                else:
                    moment_dtype = core.VarDesc.VarType.FP32

                self._add_moments_pows(master_p, moment_dtype)
                self._add_hf_step_accumulator(master_p)
                self._already_create_accumulator.add(p.name)

            elif self._is_dtype_fp16_or_bf16(p.dtype) and not self._multi_precision:
                raise NotImplementedError("AdamWCustom only support AMP training")
            else:
                self._add_moments_pows(p)
                self._add_hf_step_accumulator(p)
                self._already_create_accumulator.add(p.name)
            if self.tensorwise_offload_optimizer:
                self.offload_optim(p)

    def _add_hf_step_accumulator(self, target):
        """Accumulator holding the 1-based HF update count of ``target``.

        The count must live in optimizer state -- a plain python dict is lost by
        ``state_dict()``, so after a checkpoint restore the count would restart
        at 1 while the parameters and moments are at step N, and the bias
        correction would jump off the continuous HF trajectory. As an
        accumulator it is saved/restored (including sharded checkpoints) by the
        same machinery as ``beta1_pow``/``beta2_pow``.
        """
        if self.hf_bitexact:
            try:
                acc_type = core.VarDesc.VarType.DENSE_TENSOR
            except AttributeError:
                acc_type = core.VarDesc.VarType.LOD_TENSOR
            self._add_accumulator(
                "hf_bitexact_step",
                target,
                dtype=paddle.float32,
                shape=[1],
                fill_value=0.0,
                type=acc_type,
            )

    def _create_master_weight(self, param):
        if param.name in self._master_weights:
            var = self._master_weights[param.name]
        else:
            var_name = self._gen_master_weight_var_name(param)
            if param.name in self.weight_scale_mapping:
                weight_scale = self.weight_scale_mapping[param.name]
                if self.quantization_config.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
                    var = dequantize(
                        param,
                        weight_scale,
                        "weight",
                        self.quantization_config.weight_quantize_algo,
                        self.quantization_config,
                        apply_hadamard=self.quantization_config.apply_hadamard,
                        side="left",
                    ).astype("float32")
                else:
                    raise NotImplementedError(
                        f"Unknown weight_quantize_algo {self.quantization_config.weight_quantize_algo}"
                    )
            else:
                var = paddle.cast(param, "float32")
            var.name = var_name
            self._master_weights[param.name] = var
        return var

    def _is_dtype_fp16_or_bf16(self, dtype):
        """
        check the dtype is fp16 or the dtype is bf16
        :param dtype: instance of core.VarDesc.VarType
        :return: True if dtype is one of fp16 or bf16, False otherwise
        """
        if dtype == paddle.int8 or dtype == paddle.float8_e4m3fn:
            return True
        assert isinstance(
            dtype, (core.VarDesc.VarType, core.DataType)
        ), "The dtype should be an instance of core.VarDesc.VarType or core.DataType."
        if isinstance(dtype, core.VarDesc.VarType):
            return dtype == core.VarDesc.VarType.FP16 or dtype == core.VarDesc.VarType.BF16
        else:
            return dtype == core.DataType.FLOAT16 or dtype == core.DataType.BFLOAT16

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, (framework.Block, pir.Block))
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)
        param, grad = param_and_grad

        # Whether we should do weight decay for the parameter.
        with_decay = True
        if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param.name):
            with_decay = False

        if self.tensorwise_offload_optimizer:
            self.reload_optim(param)

        moment1 = self._get_accumulator_master(self._moment1_acc_str, param_and_grad[0])
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param_and_grad[0])
        beta1_pow_acc = self._get_accumulator_master(self._beta1_pow_acc_str, param_and_grad[0])
        beta2_pow_acc = self._get_accumulator_master(self._beta2_pow_acc_str, param_and_grad[0])
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param_and_grad[0].dtype)
        master_weight = self._master_weights[param_and_grad[0].name] if find_master else None
        if param.name in self.weight_scale_mapping:
            weight_scale = self.weight_scale_mapping[param.name]
        else:
            weight_scale = None
        lr = self._create_param_lr(param_and_grad)
        # create the adamw optimize op
        if in_dynamic_or_pir_mode():
            lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

            _beta1 = self._beta1 if not isinstance(self._beta1, Variable) else self._beta1.item(0)
            _beta2 = self._beta2 if not isinstance(self._beta2, Variable) else self._beta2.item(0)

            found_inf = self._get_auxiliary_var("found_inf") if in_pir_mode() else None
            skip_update_param = weight_scale is not None
            # ``adamw_triton`` keeps paddle's old FP32-opmath / repeated
            # ``beta_pow`` arithmetic, which diverges from torch's ``_multi_tensor_
            # adamw``; the HF target therefore always routes through
            # ``adamw_custom`` so it cannot silently lose its bit-exact recipe.
            apply_adamw = self.adamw_custom if (adamw_triton is None or self.hf_bitexact) else adamw_triton
            apply_adamw(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                moment1,
                moment2,
                beta1_pow_acc,
                beta2_pow_acc,
                master_weight,
                found_inf,
                _beta1,
                _beta2,
                self._epsilon,
                lr_ratio_,
                self._weight_decay,
                with_decay,
                find_master,
                skip_update_param,
            )
            if skip_update_param:
                if param.weight_quantize_algo in ["a8w8linear", "a8w4linear", "fp8linear"]:
                    if "parallel_quantization_linear" not in param.name:
                        group = None
                    elif param.weight_quantize_algo in ["a8w8linear", "a8w4linear"] and "row" in param.name:
                        group = None
                    else:
                        group = self.mp_group
                    param[:], weight_scale[:] = quantize(
                        x=master_weight.astype(weight_scale.dtype),
                        weight_quantize_algo=self.quantization_config.weight_quantize_algo,
                        tensor_type="weight",
                        quantization_config=self.quantization_config,
                        side="left",
                        apply_hadamard=self.quantization_config.apply_hadamard,
                        group=group,
                    )
                else:
                    raise NotImplementedError(
                        f"Please check your weight_quantize_algo {self.quantization_config.weight_quantize_algo}."
                    )
            if self.tensorwise_offload_optimizer:
                self.offload_optim(param)

            return None
        else:
            raise NotImplementedError("Not implemented yet.")

    def adamw_custom(
        self,
        param,
        grad,
        learning_rate,
        moment1,
        moment2,
        beta1_pow,
        beta2_pow,
        master_weight,
        skip_update,
        beta1,
        beta2,
        epsilon,
        lr_ratio,
        coeff,
        with_decay,
        multi_precision,
        skip_update_param,
    ):
        if skip_update:
            return
        if not with_decay:
            coeff = 0.0
        if not multi_precision:
            master_weight = None
        lr = learning_rate * lr_ratio
        if self.hf_bitexact:
            # torch derives the bias corrections from ``beta ** step``; paddle's
            # ``beta_pow`` accumulators reach that power by repeated FP32
            # multiplication, which rounds differently. Track the count in an
            # accumulator so ``state_dict()`` / ``set_state_dict()`` (and the
            # sharded checkpoint paths) carry it across a resume.
            step_acc = self._get_accumulator_master("hf_bitexact_step", param)
            step = int(step_acc.item()) + 1
            step_acc.set_value(paddle.full(step_acc.shape, step, dtype=step_acc.dtype))
            new_p, new_m, new_v = _hf_bitexact_adamw_step(
                param,
                grad,
                moment1,
                moment2,
                master_weight,
                lr=float(lr),
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                weight_decay=float(coeff),
                step=step,
            )
            if master_weight is not None:
                master_weight[:] = new_p.astype(master_weight.dtype)
                if not skip_update_param:
                    param[:] = new_p.astype(param.dtype)
            else:
                param[:] = new_p.astype(param.dtype)
            moment1[:] = new_m.astype(moment1.dtype)
            moment2[:] = new_v.astype(moment2.dtype)
            beta1_pow[:], beta2_pow[:] = (
                beta1 * beta1_pow[:],
                beta2 * beta2_pow[:],
            )
            return
        if master_weight is not None:
            p = master_weight
        else:
            p = param

        p *= 1.0 - lr * coeff
        moment_dtype = moment1.dtype
        mom1 = moment1.astype("float32")
        mom2 = moment2.astype("float32")

        mom1 = beta1 * mom1 + (1.0 - beta1) * grad
        mom2 = beta2 * mom2 + (1.0 - beta2) * grad * grad
        denom = mom2.sqrt() / (1.0 - beta2_pow).sqrt() + epsilon
        p += (mom1 / denom) * (-(lr / (1.0 - beta1_pow)))

        if master_weight is not None:
            master_weight[:] = p
            if not skip_update_param:
                param[:] = p.astype(param.dtype)
        else:
            param[:] = p
        moment1[:] = mom1.astype(moment_dtype)
        moment2[:] = mom2.astype(moment_dtype)
        beta1_pow[:], beta2_pow[:] = beta1 * beta1_pow[:], beta2 * beta2_pow[:]
        return

    def offload_optim(self, p):
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype)
        if find_master:
            self._master_weights[p.name] = self._master_weights[p.name].pin_memory()
            target_name = self._master_weights[p.name].name
        else:
            target_name = p.name
        for name in [self._moment1_acc_str, self._moment2_acc_str]:
            if self._name is not None:
                name = self._name + "_" + name
            self._accumulators[name][target_name] = self._accumulators[name][target_name].pin_memory()

    def reload_optim(self, p):
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype)
        if find_master:
            self._master_weights[p.name] = self._master_weights[p.name].cuda()
            target_name = self._master_weights[p.name].name
        else:
            target_name = p.name
        for name in [self._moment1_acc_str, self._moment2_acc_str]:
            if self._name is not None:
                name = self._name + "_" + name
            self._accumulators[name][target_name] = self._accumulators[name][target_name].cuda()
