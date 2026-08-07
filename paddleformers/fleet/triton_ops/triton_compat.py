#!/usr/bin/env python3

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

"""
Triton kernel 兼容工具。

用于在 torch 兼容模式下正确切换 Triton driver，以便 Paddle 张量可以
传给 Triton kernel。若当前环境未启用 torch 兼容模式，则保持原样。
"""

from functools import cache
from importlib.metadata import PackageNotFoundError, distribution

import paddle


@cache
def _is_package_installed(dist_name: str) -> bool:
    try:
        distribution(dist_name)
        return True
    except PackageNotFoundError:
        return False


if _is_package_installed("torch") and paddle.is_compiled_with_cuda():
    with paddle.use_compat_guard(enable=True, silent=True):
        from triton.runtime.driver import _create_driver

        paddle_driver = _create_driver()
else:
    paddle.enable_compat(scope={"triton"})


def _swap_driver_guard(fn):
    from triton.runtime.driver import driver

    def wrapped_fn(*args, **kwargs):
        driver.set_active(paddle_driver)
        try:
            return fn(*args, **kwargs)
        finally:
            driver.reset_active()

    return wrapped_fn


def enable_compat_on_triton_kernel(triton_kernel):
    if not _is_package_installed("torch"):
        return triton_kernel

    if not paddle.is_compiled_with_cuda():
        return triton_kernel

    class WrappedTritonKernel:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, index):
            return _swap_driver_guard(self.kernel[index])

    return WrappedTritonKernel(triton_kernel)
