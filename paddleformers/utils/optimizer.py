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

import re

import paddle
from paddle import pir
from paddle.base import core, framework
from paddle.base.dygraph import base as imperative_base
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
    def __init__(self, quantization_config, tensorwise_offload_optimizer, *args, **kwargs):
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
                self._already_create_accumulator.add(p.name)

            elif self._is_dtype_fp16_or_bf16(p.dtype) and not self._multi_precision:
                raise NotImplementedError("AdamWCustom only support AMP training")
            else:
                self._add_moments_pows(p)
                self._already_create_accumulator.add(p.name)
            if self.tensorwise_offload_optimizer:
                self.offload_optim(p)

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
            apply_adamw = self.adamw_custom if adamw_triton is None else adamw_triton
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


class AdamWLoRAPro(AdamW):
    def __init__(self, scaling_factor=2.0, x_mode="zero", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert scaling_factor is not None
        if x_mode not in ["zero", "sylvester", "symmetry"]:
            raise ValueError(
                f"Invalid x_mode value: {x_mode}, " f"mode should be in ['zero', 'sylvester', 'symmetry']"
            )
        self.scaling_factor = scaling_factor
        self.x_mode = x_mode

    def _solve_sylvester(self, A, B, C, X=None):
        if A.dtype in [paddle.bfloat16, paddle.float16]:
            A = A.to("float32")
            B = B.to("float32")
            C = C.to("float32")
        B = -B
        m = tuple(B.shape)[-1]
        n = tuple(A.shape)[-1]
        R, U = paddle.linalg.eig(x=A)
        S, V = paddle.linalg.eig(x=B)

        CV = C @ V

        U_real, U_imag = paddle.real(U), paddle.imag(U)
        CV_real, CV_imag = paddle.real(CV), paddle.imag(CV)

        n_dim = U_real.shape[0]

        block_top = paddle.concat([U_real, -U_imag], axis=1)  # (n, 2n)
        block_bot = paddle.concat([U_imag, U_real], axis=1)  # (n, 2n)
        A_block = paddle.concat([block_top, block_bot], axis=0)  # (2n, 2n)
        B_block = paddle.concat([CV_real, CV_imag], axis=0)  # (2n, m)

        F_block = paddle.linalg.solve(A_block, B_block)  # [F_real; F_imag]

        F_real = F_block[:n_dim, :]
        F_imag = F_block[n_dim:, :]
        F = paddle.complex(F_real, F_imag)

        W = R[..., :, None] - S[..., None, :]
        Y = F / W
        try:
            V_inv = paddle.linalg.inv(V)
        except RuntimeError:
            # Add regularization to handle singular matrices
            epsilon = 1e-6 * paddle.mean(paddle.abs(V))
            V_reg = V + epsilon * paddle.eye(V.shape[-1])
            V_inv = paddle.linalg.inv(V_reg)
        X = U[..., :n, :n] @ Y[..., :n, :m] @ V_inv[..., :m, :m]

        if all(paddle.isreal(x.flatten()[0]) for x in [A, B, C]):
            return paddle.real(X)
        else:
            return X

    @imperative_base.no_grad
    @framework.non_static_only
    def step(self) -> None:
        """
        Execute the optimizer and update parameters once.

        Returns:
            None

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> a = paddle.rand([2,13], dtype="float32")
                >>> linear = paddle.nn.Linear(13, 5)
                >>> # This can be any optimizer supported by dygraph.
                >>> opt = paddle.optimizer.AdamW(learning_rate = 0.01,
                ...                             parameters = linear.parameters())
                >>> out = linear(a)
                >>> out.backward()
                >>> opt.step()
                >>> opt.clear_grad()
        """
        if paddle.base.dygraph.base.in_to_static_mode():
            self._declarative_step()
            return

        if not isinstance(self._parameter_list[0], dict):
            param_id_to_idx = {id(param): idx for idx, param in enumerate(self._parameter_list)}

            lora_params = {}
            for idx, param in enumerate(self._parameter_list):
                name = getattr(param, "name", f"param_{idx}")
                match = re.match(r"lo_ra_linear_(\d+)\.w_(\d+)", name)
                if match:
                    layer_num = int(match.group(1))
                    weight_type = match.group(2)
                    if layer_num not in lora_params:
                        lora_params[layer_num] = {}
                    lora_params[layer_num][weight_type] = param

            for layer_num, weights in lora_params.items():
                if "1" in weights and "2" in weights:
                    param_B = weights["1"]
                    param_A = weights["2"]

                    idx_B = param_id_to_idx[id(param_B)]
                    idx_A = param_id_to_idx[id(param_A)]

                    if param_A._grad_ivar() is not None and param_B._grad_ivar() is not None:
                        A = param_A.detach()
                        B = param_B.detach()
                        grad_A = param_A._grad_ivar()
                        grad_B = param_B._grad_ivar()

                        delta = 1e-08
                        AA_T = A @ A.T
                        B_TB = B.T @ B
                        AA_T_inv = paddle.linalg.pinv(AA_T + delta * paddle.eye(num_rows=AA_T.shape[0]))
                        B_TB_inv = paddle.linalg.pinv(B_TB + delta * paddle.eye(num_rows=B_TB.shape[0]))

                        if self.x_mode == "sylvester":
                            X = self._solve_sylvester(
                                B_TB, AA_T, -(1 / self.scaling_factor**2) * B_TB_inv @ grad_A @ A.T
                            )
                        elif self.x_mode == "symmetry":
                            X = -0.5 * (1 / self.scaling_factor**2) * B_TB_inv @ B.T @ grad_B @ AA_T
                        else:  # zero mode
                            X = paddle.zeros(shape=(B_TB_inv.shape[0], B_TB_inv.shape[0]))

                        X = X.clone().detach().cast(A.dtype)

                        new_grad_A = (1 / self.scaling_factor**2) * B_TB_inv @ grad_A + X @ A
                        new_grad_B = (1 / self.scaling_factor**2) * (
                            (paddle.eye(num_rows=B.shape[0]) - B @ B_TB_inv @ B.T) @ grad_B @ AA_T_inv
                        ) - B @ X

                        self._parameter_list[idx_A]._grad_ivar()[:] = new_grad_A
                        self._parameter_list[idx_B]._grad_ivar()[:] = new_grad_B

            params_grads = []
            for param in self._parameter_list:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    if framework.in_dygraph_mode():
                        if (
                            hasattr(grad_var, "is_selected_rows")
                            and grad_var.is_selected_rows()
                            and self.regularization is not None
                        ):
                            raise RuntimeError(
                                "AdamW don't support weight_decay with sparse parameters, please set it to None."
                            )
                    else:
                        if (
                            hasattr(grad_var, "_is_sparse")
                            and grad_var._is_sparse()
                            and self.regularization is not None
                        ):
                            raise RuntimeError(
                                "AdamW don't support weight_decay with sparse parameters, please set it to None."
                            )
                    params_grads.append((param, grad_var))

                    self._apply_optimize(loss=None, startup_program=None, params_grads=params_grads)
        else:
            raise NotImplementedError("AdamWLoRAPro does not support parameter groups")
