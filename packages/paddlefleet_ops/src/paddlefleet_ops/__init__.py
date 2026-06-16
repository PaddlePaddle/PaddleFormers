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

import ctypes
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

import paddle

from .utils import (
    HardwareIncompatibleBlocker,
    ModuleContext,
    clean_module_namespace,
    get_cuda_version,
    get_nvshmem_host_lib_path,
    import_custom_ops,
    patch_module_namespace,
)
from .version import __version__ as __version__

# paddle.enable_compat(scope={"triton"}) enables the torch proxy
# specifically for the 'triton' module. This means `import torch` inside 'triton'
# will actually import paddle's compatibility layer (acting as torch).
#
# 'scope' acts as an allowlist. To add other modules, you can do:
# paddle.enable_compat(scope={"triton", "new_module"})
#
# Note: Ensure that any torch APIs used in 'new_module' are already implemented in Paddle.

if "torch" in sys.modules and sys.modules["torch"] is None:
    # Note: In paddleformer's __init__.py, it will set sys.modules["torch"] to None
    # We should restore or delete it before enable_compat
    del sys.modules["torch"]
    if "torch_save" in sys.modules and sys.modules["torch_save"] is not None:
        sys.modules["torch"] = sys.modules["torch_save"]

logger = logging.getLogger(__name__)
ops_dir = Path(__file__).parent
cuda_capability = (
    paddle.cuda.get_device_capability()
    if paddle.is_compiled_with_cuda()
    else None
)
if paddle.is_compiled_with_cuda():
    _python_version = sys.version_info
    _python_version_str = ".".join(map(str, _python_version[:3]))
    _cuda_version = get_cuda_version()
    _cuda_version_str = ".".join(map(str, _cuda_version))
    _capability_str = (
        f"{cuda_capability[0]}.{cuda_capability[1]}"
        if cuda_capability
        else "unavailable"
    )

    DEEP_GEMM_HINT = (
        "For developers: guard imports with `is_deep_gemm_available()` and only call `paddlefleet_ops.deep_gemm` when flag branch enabled.\n"
        "For users: set `use_deep_gemm=False` if you want to skip it, or use a GPU with compute capability equal to 9.x to enable."
    )

    DEEP_EP_HINT = (
        "For developers: guard imports with `is_deep_ep_available()` and only call `paddlefleet_ops.deep_ep` when flag branch enabled.\n"
        "For users: avoid `moe_token_dispatcher_type='deepep'` or use a GPU with compute capability equal to 9.x to enable."
    )

    HYBRID_EP_HINT = (
        "For developers: guard imports with `is_hybrid_ep_available()` and only call `paddlefleet_ops.hybrid_ep` when HybridEP branch enabled.\n"
        "For users: avoid `moe_token_dispatcher_type='hybridep'` or use a GPU with compute capability equal to 9.x to enable."
    )

    SONIC_MOE_HINT = (
        "For developers: guard imports with `is_sonicmoe_available()` and only call `paddlefleet_ops.sonicmoe` when flag branch enabled.\n"
        "For users: set `using_sonic_moe=False` or upgrade to Python >= 3.12, CUDA >= 12.9, and a GPU with compute capability >= 10.0 (Blackwell) to enable."
    )

    CUDNN_FRONTEND_HINT = "For developers: guard imports with `is_cudnn_frontend_available()` and only call `paddlefleet_ops.cudnn` when flag branch enabled."

    FLASH_MLA_HINT = (
        "For developers: guard imports with `is_flash_mla_available()` and only call `paddlefleet_ops.flash_mla` when flag branch enabled.\n"
        "For users: use a GPU with compute capability >= 9.0 (Hopper or Blackwell) to enable."
    )
else:
    DEEP_GEMM_HINT = "deep_gemm is not supported on XPU backend."
    DEEP_EP_HINT = "deep_ep is not supported on XPU backend."
    HYBRID_EP_HINT = "hybrid_ep is not supported on XPU backend."
    SONIC_MOE_HINT = "sonicmoe is not supported on XPU backend."
    CUDNN_FRONTEND_HINT = "cudnn frontend is not supported on XPU backend."
    FLASH_MLA_HINT = "flash_mla is not supported on XPU backend."

FLASH_MASK_HINT = (
    "For developers: guard imports with `is_flash_mask_available()` and only call `paddlefleet_ops.flash_mask` when flag branch enabled.\n"
    "For users: use a GPU with compute capability >= 10.0 (Blackwell) to enable."
)


def _build_notice(
    lib_module: str, reason: str, hint_for_error: str | None = None
) -> tuple[str, str]:
    """Compose warning/error messages; only errors carry extra hints."""
    warning = f"{lib_module} not supported: {reason}"
    error_reason = f"{reason} \n{hint_for_error}" if hint_for_error else reason
    error = f"{lib_module} not supported: {error_reason}"
    return warning, error


def _hopper_requirement(
    lib_module: str, hint: str | None = None
) -> tuple[str, str]:
    reason = (
        f"{lib_module} requires GPU compute capability >= 9.0 (Hopper). "
        f"Current capability: {_capability_str}."
    )
    return _build_notice(lib_module, reason, hint_for_error=hint)


def _cutedsl_requirement(
    lib_module: str, hint: str | None = None
) -> tuple[str, str]:
    reason = f"{lib_module} requires Python 3.12. Current Python version: {_python_version_str}."
    return _build_notice(lib_module, reason, hint_for_error=hint)


def _blackwell_requirement(
    lib_module: str, hint: str | None = None
) -> tuple[str, str]:
    reason = (
        f"{lib_module} requires GPU compute capability >= 10.0 (Blackwell). "
        f"Current capability: {_capability_str}."
    )
    return _build_notice(lib_module, reason, hint_for_error=hint)


def _sonic_moe_requirement(
    lib_module: str, hint: str | None = None
) -> tuple[str, str]:
    reasons = []
    if sys.version_info < (3, 12):
        reasons.append(
            f"Python >= 3.12 required (current {_python_version_str})"
        )
    if _cuda_version < (12, 9):
        reasons.append(f"CUDA >= 12.9 required (current {_cuda_version_str})")
    if not cuda_capability or cuda_capability[0] < 10:
        reasons.append(
            f"GPU compute capability >= 10.0 (Blackwell) required (current {_capability_str})"
        )
    reason = "; ".join(reasons) if reasons else "Runtime requirements not met."
    return _build_notice(lib_module, reason, hint_for_error=hint)


_DEEP_GEMM_AVAILABLE = False
_DEEP_EP_AVAILABLE = False
_HYBRID_EP_AVAILABLE = False
_SONIC_MOE_AVAILABLE = False
_FLASH_MLA_AVAILABLE = False
_FLASH_MASK_AVAILABLE = False
_CUDNN_FRONTEND_AVAILABLE = False

if paddle.is_compiled_with_cuda():
    if paddle.cuda.get_device_capability()[0] >= 9:
        _DEEP_GEMM_AVAILABLE = True
        _DEEP_EP_AVAILABLE = True
        _FLASH_MLA_AVAILABLE = True
        if _cuda_version >= (12, 9):
            _HYBRID_EP_AVAILABLE = True
    if paddle.cuda.get_device_capability()[0] >= 10:
        _FLASH_MASK_AVAILABLE = True
    if (
        sys.version_info >= (3, 12)
        and paddle.cuda.get_device_capability()[0] >= 10
        and _cuda_version >= (12, 9)
    ):
        _SONIC_MOE_AVAILABLE = True
    if sys.version_info >= (3, 12):
        _CUDNN_FRONTEND_AVAILABLE = True

if paddle.is_compiled_with_xpu():
    _DEEP_EP_AVAILABLE = True


def is_deep_gemm_available():
    return _DEEP_GEMM_AVAILABLE


def is_deep_ep_available():
    return _DEEP_EP_AVAILABLE


def is_hybrid_ep_available():
    return _HYBRID_EP_AVAILABLE


def is_sonic_moe_available():
    return _SONIC_MOE_AVAILABLE


def is_flash_mla_available():
    return _FLASH_MLA_AVAILABLE


def is_flash_mask_available():
    return _FLASH_MASK_AVAILABLE


def is_cudnn_frontend_available():
    return _CUDNN_FRONTEND_AVAILABLE


def _try_load_nvshmem(ops_dir: Path):
    third_party_temp_dir = ops_dir.parent / "_third_party_install_temp"
    if third_party_temp_dir.exists():
        try:
            nvshmem_spec = importlib.util.find_spec("nvidia.nvshmem")
            if nvshmem_spec and nvshmem_spec.submodule_search_locations:
                nvshmem_dir = nvshmem_spec.submodule_search_locations[0]
                nvshmem_host_lib_path = get_nvshmem_host_lib_path(nvshmem_dir)
                logger.info(
                    f"Pre-loading NVSHMEM library from: {nvshmem_host_lib_path}"
                )
                ctypes.CDLL(str(nvshmem_host_lib_path), mode=ctypes.RTLD_GLOBAL)
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during NVSHMEM pre-loading: {e}"
            ) from e


def _safe_load_ecosystem_lib(
    lib_name: str,
    ops_dir: Path,
    module_globals: dict[str, Any],
    extra_libs_name: list[str] | None = None,
):
    lib_names = [lib_name]
    if extra_libs_name:
        lib_names += extra_libs_name
    with ModuleContext(lib_names, ops_dir):
        try:
            module = importlib.import_module(lib_name)
            patch_module_namespace(lib_name, "paddlefleet_ops.")
            if extra_libs_name:
                for extra_lib in extra_libs_name:
                    if extra_lib != "quack":
                        clean_module_namespace(extra_lib)
            module_globals[lib_name] = module
            logger.info(f"Successfully loaded ecosystem library: {lib_name}")
        except ImportError as e:
            raise ImportError(
                f"Failed to import ecosystem library '{lib_name}': {e}"
            ) from e


import_custom_ops(
    package="paddlefleet_ops._extensions",
    module_name=".ops",
    global_ns=globals(),
)

# 别名：算子注册名改为 paddlefleet_fused_swiglu_probs_bwd 以避免与 FusedQuantOps 冲突，
# 但对外仍保持 fused_swiglu_probs_bwd 的导入接口。
if "paddlefleet_fused_swiglu_probs_bwd" in globals():
    globals()["fused_swiglu_probs_bwd"] = globals()[
        "paddlefleet_fused_swiglu_probs_bwd"
    ]

blocked_import_messages: dict[str, str] = {}

if paddle.is_compiled_with_cuda():
    if is_deep_gemm_available():
        paddle.enable_compat(scope={"deep_gemm", "triton"}, silent=True)
        _safe_load_ecosystem_lib("deep_gemm", ops_dir, globals())
    else:
        warning, error = _hopper_requirement(
            "paddlefleet_ops.deep_gemm", hint=DEEP_GEMM_HINT
        )
        logger.warning(warning)
        blocked_import_messages["paddlefleet_ops.deep_gemm"] = error
    if is_deep_ep_available():
        paddle.enable_compat(scope={"deep_ep"}, silent=True)
        # Loading libnvshmem_host.so.* first when use editable install
        _try_load_nvshmem(ops_dir)
        _safe_load_ecosystem_lib("deep_ep", ops_dir, globals())
    else:
        warning, error = _hopper_requirement(
            "paddlefleet_ops.deep_ep", hint=DEEP_EP_HINT
        )
        logger.warning(warning)
        blocked_import_messages["paddlefleet_ops.deep_ep"] = error

    if is_hybrid_ep_available():
        paddle.enable_compat(scope={"hybrid_ep"}, silent=True)
        os.environ["HYBRID_EP_SKIP_DEEP_EP"] = "1"
        _safe_load_ecosystem_lib("hybrid_ep", ops_dir, globals())
    else:
        warning, error = _hopper_requirement(
            "paddlefleet_ops.hybrid_ep", hint=HYBRID_EP_HINT
        )
        logger.warning(warning)
        blocked_import_messages["paddlefleet_ops.hybrid_ep"] = error

    if is_sonic_moe_available():
        paddle.enable_compat(
            scope={"sonicmoe", "quack", "triton"},
            silent=True,
        )
        _safe_load_ecosystem_lib("sonicmoe", ops_dir, globals(), ["quack"])
    else:
        warning, error = _sonic_moe_requirement(
            "paddlefleet_ops.sonicmoe", hint=SONIC_MOE_HINT
        )
        logger.warning(warning)
        blocked_import_messages["paddlefleet_ops.sonicmoe"] = error

    if is_flash_mla_available():
        paddle.enable_compat(scope={"flash_mla"}, silent=True)
        _safe_load_ecosystem_lib("flash_mla", ops_dir, globals())
    else:
        warning, error = _hopper_requirement(
            "paddlefleet_ops.flash_mla", hint=FLASH_MLA_HINT
        )
        logger.warning(warning)
        blocked_import_messages["paddlefleet_ops.flash_mla"] = error

    if is_flash_mask_available():
        _safe_load_ecosystem_lib("flash_mask", ops_dir, globals())
    else:
        warning, error = _blackwell_requirement(
            "paddlefleet_ops.flash_mask", hint=FLASH_MASK_HINT
        )
        logger.warning(warning)
        blocked_import_messages["paddlefleet_ops.flash_mask"] = error

    if is_cudnn_frontend_available():
        paddle.enable_compat(scope={"cudnn"}, silent=True)
        _safe_load_ecosystem_lib("cudnn", ops_dir, globals())
    else:
        warning, error = _cutedsl_requirement(
            "paddlefleet_ops.cudnn", hint=CUDNN_FRONTEND_HINT
        )
        logger.warning(warning)
        blocked_import_messages["paddlefleet_ops.cudnn"] = error

    if blocked_import_messages:
        sys.meta_path.insert(
            0, HardwareIncompatibleBlocker(blocked_import_messages)
        )

    try:
        paddle.enable_compat(scope={"triton"}, silent=True)
        from ._extensions.flashmask import (
            rr_attn_estimate_triton_func as rr_attn_estimate_triton_func,
        )
    finally:
        paddle.disable_compat()


def __getattr__(name):
    module_name = f"paddlefleet_ops.{name}"

    if module_name in blocked_import_messages:
        raise RuntimeError(blocked_import_messages[module_name])

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
