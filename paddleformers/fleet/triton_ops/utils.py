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

"""Utilities for Triton ops: torch compat check and conditional dispatch."""

from functools import cache
from importlib.metadata import PackageNotFoundError, distribution

import paddle


def is_torch_compat_available() -> bool:
    """Return True if paddle provides torch-compat mode."""
    return hasattr(paddle, "enable_compat")


def dispatch_to(dispatch_fn, *, cond=None):
    """Decorator: call dispatch_fn when cond is True, else fall back to fn.

    Args:
        dispatch_fn: high-performance implementation.
        cond: predicate deciding whether to use dispatch_fn.
    """
    if cond is None:
        cond = lambda self, *args, **kwargs: True

    def decorator(fn):
        def wrapper(*args, **kwargs):
            if cond(*args, **kwargs) and is_torch_compat_available():
                return dispatch_fn(*args, **kwargs)
            return fn(*args, **kwargs)

        wrapper.__original_fn__ = fn
        return wrapper

    return decorator


@cache
def _is_package_installed(dist_name: str) -> bool:
    """Check whether a package is installed."""
    try:
        distribution(dist_name)
        return True
    except PackageNotFoundError:
        return False


# Initialize the Paddle Triton driver (only when supported).
_paddle_driver = None
if _is_package_installed("torch") and paddle.is_compiled_with_cuda():
    try:
        with paddle.use_compat_guard(enable=True, silent=True):
            from triton.runtime.driver import _create_driver

            _paddle_driver = _create_driver()
    except Exception:
        pass


def swap_driver_guard(fn):
    """
    Driver-switch guard: ensure the Triton kernel uses the correct Paddle
    driver.
    """
    from triton.runtime.driver import driver

    def wrapped_fn(*args, **kwargs):
        if _paddle_driver is not None:
            driver.set_active(_paddle_driver)
        try:
            return fn(*args, **kwargs)
        finally:
            if _paddle_driver is not None:
                driver.reset_active()

    return wrapped_fn


def enable_compat_on_triton_kernel(triton_kernel):
    """
    Triton kernel compatibility decorator.
    Automatically handles the driver switch between Paddle and Triton so the
    Triton kernel runs correctly within a PaddlePaddle environment.
    """
    if not paddle.is_compiled_with_cuda():
        return triton_kernel

    class WrappedTritonKernel:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, index):
            return swap_driver_guard(self.kernel[index])

    return WrappedTritonKernel(triton_kernel)
