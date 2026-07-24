# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle
from parameterized import parameterized

from paddlefleet_ops.sonicmoe import KernelBackendMoE, MoE, enable_quack_gemm
from paddlefleet_ops.sonicmoe.enums import ActivationType

from .commons_test import TestCommons

_SEED = 42


_gpu_capability = paddle.device.cuda.get_device_capability()
_IS_SM90A = _gpu_capability == (9, 0)
_SKIP_REASON = (
    f"MmaF16BF16Op requires sm_90a, "
    f"but current GPU capability is sm_{_gpu_capability[0]}{_gpu_capability[1]}a"
)
# torch._dynamo.config.cache_size_limit = 1024
# torch._dynamo.config.accumulated_cache_size_limit = 1024
# torch._funcpaddle.config.donated_buffer = False

import os

RUN_IN_PADDLE_CI = paddle.utils.strtobool(os.getenv("RUN_IN_PADDLE_CI", "0"))


problem_shapes = [
    (8192, 768, 256, 128, 8),
    (8192, 768, 512, 64, 4),
    (8192, 768, 1024, 32, 2),
    (8192, 1536, 256, 128, 8),
    (8192, 1536, 512, 64, 4),
    (8192, 1536, 1024, 32, 2),
    (8192, 4096, 256, 256, 16),
    (8192, 4096, 512, 128, 8),
    (8192, 4096, 1024, 64, 4),
    (8192, 4096, 512, 256, 16),
    (8192, 4096, 1024, 128, 8),
    (8192, 4096, 2048, 64, 4),
]

if RUN_IN_PADDLE_CI:
    problem_shapes = problem_shapes[:1]


@unittest.skipUnless(_IS_SM90A, _SKIP_REASON)
class MoETest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [paddle.device("cuda")],
            [paddle.bfloat16],
            problem_shapes,
            [KernelBackendMoE.sonicmoe],  # kernel_backend_moe
            [False],  # is_compiling
            [False, True],  # add_bias
            [False, True],  # use_quack_gemm
        )
    )
    def test_moe(
        self,
        device: paddle.device,
        dtype: paddle.dtype,
        problem_shape: tuple[int, int, int, int, int],
        kernel_backend_moe: KernelBackendMoE,
        is_compiling: bool,
        add_bias: bool,
        use_quack_gemm: bool,
    ) -> None:
        if use_quack_gemm and (is_compiling or add_bias):
            self.skipTest("unsupported test")

        self.set_seed(_SEED)

        T, H, I, E, K = problem_shape
        with paddle.device(device):
            moe = MoE(
                num_experts=E,
                num_experts_per_tok=K,
                hidden_size=H,
                intermediate_size=I,
                activation_function=ActivationType.SWIGLU,
                add_bias=add_bias,
                std=0.02,
            ).to(dtype=dtype)

        if add_bias:
            b1, b2 = moe.c_fc.bias, moe.c_proj.bias
            paddle.nn.init.normal_(b1, 0, 0.01)
            paddle.nn.init.normal_(b2, 0, 0.01)

        moe_kernel = moe
        moe_paddle = moe

        if is_compiling:
            moe_kernel = paddle.compile(moe_kernel, fullgraph=True)

        paddle.cuda.empty_cache()
        x_paddle = 0.02 * paddle.randn(
            T, H, device=device, dtype=dtype, requires_grad=True
        )
        x_kernel = x_paddle.clone().detach().requires_grad_()

        # with torch.autocast(x_paddle.device.type, paddle.float32):
        if True:
            with enable_quack_gemm(use_quack_gemm):
                y_kernel = moe_kernel(
                    x_kernel, kernel_backend_moe=kernel_backend_moe
                )[0]

            y_paddle = moe_paddle(
                x_paddle, kernel_backend_moe=KernelBackendMoE.sonicmoe
            )[0]
            self.assert_equal_tensors(
                y_kernel.float(),
                y_paddle.float(),
                False,
                atol_bfloat16=1.4e-2,
                rtol_bfloat16=2e-2,
                dtype=dtype,
            )

        dy_paddle = 0.02 * paddle.randn(
            T, H, device=device, dtype=dtype, requires_grad=True
        )
        dy_kernel = dy_paddle.clone().detach().requires_grad_()

        W = list(moe.parameters())

        # with torch.autocast(x_paddle.device.type, paddle.float32):
        if paddle.__version__ != "3.3.0":
            kernel_grads = paddle.autograd.grad(
                y_kernel,
                [x_kernel, *W],
                grad_outputs=dy_kernel,
                retain_graph=True,
            )
            paddle_grads = paddle.autograd.grad(
                y_paddle,
                [x_paddle, *W],
                grad_outputs=dy_paddle,
                retain_graph=True,
            )

            for _paddle_grad, _kernel_grad in zip(paddle_grads, kernel_grads):
                self.assert_equal_tensors(
                    _kernel_grad.float(),
                    _paddle_grad.float(),
                    False,
                    atol_bfloat16=2e-2,
                    rtol_bfloat16=2e-2,
                    dtype=dtype,
                )

            for w in W:
                w.grad = None

        paddle_grads = kernel_grads = None
        paddle.cuda.empty_cache()
