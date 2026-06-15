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

import logging
import os
import shutil
from pathlib import Path

# backends.py and build_utils.py live alongside this file in packages/paddlefleet_ops/
import backends
from build_utils import get_special_build_deps
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


def change_pwd():
    """change_pwd"""
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)


common_dependencies: list[str] = []


def get_special_setup_deps():
    if backends.IS_NVIDIA:
        deps = [
            "triton",  # for deep_gemm, flashmask
            "nvidia-cutlass-dsl[cu13]==4.4.1",  # for sonic_moe and flash_attention
            "filelock",  # for sonic_moe
            "apache-tvm-ffi>=0.1.3,<0.1.12",  # for supersonic_moe
        ]
        return deps
    elif backends.IS_XPU:
        return []
    else:
        return []


class CustomBdistWheel(_bdist_wheel):
    """Custom bdist_wheel that removes .o files from wheel before packaging."""

    def _is_all_o_files(self, dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if not file.endswith(".o"):
                    return False
        return True

    def _clean_build_dir(self, wheel_dir):
        build_dir = os.path.join(wheel_dir, "build")
        if not os.path.exists(build_dir):
            return
        if not self._is_all_o_files(build_dir):
            return
        try:
            shutil.rmtree(build_dir)
        except Exception as e:
            logging.warning(f"Failed to remove directory {build_dir}: {e}")

    def write_wheelfile(self, wheelfile_base, generator=None):
        # Only strip source files from the wheel bdist staging dir.
        # NOTE: we intentionally keep the build/ directory (.o files) on disk
        # so that setuptools can use incremental compilation on the next build.
        if hasattr(self, "bdist_dir") and self.bdist_dir:
            extensions_path = Path(self.bdist_dir) / "paddlefleet_ops" / "_extensions"
            for ext in (".cu", ".h", ".txt"):
                for file in extensions_path.glob(f"*{ext}"):
                    try:
                        os.remove(file)
                    except Exception:
                        pass

        if generator is not None:
            super().write_wheelfile(wheelfile_base, generator=generator)
        else:
            super().write_wheelfile(wheelfile_base)


def _detect_local_gpu_arch():
    """Auto-detect the compute capability of the first visible GPU via nvidia-smi.
    Returns a dot-separated string like '9.0', or None if detection fails.
    """
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        caps = {line.strip() for line in out.decode().splitlines() if line.strip()}
        return ";".join(sorted(caps)) if caps else None
    except Exception:
        return None


# Map "X.Y" arch strings to -gencode flags.
# sm_90a is used for H100/H800 (full feature set including wgmma/tma).
_ARCH_TO_GENCODE = {
    "8.0": "-gencode=arch=compute_80,code=sm_80",
    "9.0": "-gencode=arch=compute_90a,code=sm_90a",
    "10.0": "-gencode=arch=compute_100,code=sm_100",
    "10.3": "-gencode=arch=compute_103,code=sm_103",
}


def _build_gencode_flags(cuda_major: int, cuda_minor: int) -> list[str]:
    """Return the -gencode flags for the current build.

    Priority (highest first):
    1. PADDLE_CUDA_ARCH_LIST env var (semicolon- or comma-separated, e.g. "9.0;10.0")
    2. Auto-detect from the local GPU via nvidia-smi
    3. Conservative default based on CUDA toolkit version:
       - CUDA < 12.8  → sm_90 only
       - CUDA ≥ 12.8  → sm_90 + sm_100 + sm_103
    """
    import re

    raw = os.environ.get("PADDLE_CUDA_ARCH_LIST", "").strip()

    if not raw:
        raw = _detect_local_gpu_arch() or ""

    if not raw:
        # Fallback: version-based default
        raw = "9.0" if (cuda_major == 12 and cuda_minor < 8) else "9.0;10.0;10.3"

    archs = [a.strip() for a in re.split(r"[;,]", raw) if a.strip()]
    flags = [_ARCH_TO_GENCODE[arch] for arch in archs if arch in _ARCH_TO_GENCODE]

    logging.info(f"CUDA gencode flags: {flags}")
    return flags


def setup_ops_extension():
    from build_utils import get_cuda_version

    # import paddle.core
    from paddle.utils.cpp_extension import CUDAExtension, setup

    # paddle_compiled_with_onednn = is_compiled_with_onednn()
    paddle_compiled_with_onednn = False

    cuda_major, cuda_minor = get_cuda_version()
    gencode_flags = _build_gencode_flags(cuda_major, cuda_minor)

    nvcc_args = [
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-maxrregcount=32",
        "-lineinfo",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        *gencode_flags,
        "-DNDEBUG",
    ]

    if paddle_compiled_with_onednn:
        nvcc_args.append("-DPADDLE_WITH_DNNL")

    # change_pwd() MUST be called before CUDAExtension() so that the
    # os.path.abspath() calls Paddle makes internally use _pkg_dir as the
    # base.  After that, a plain relpath() (no explicit start=) is enough
    # because os.getcwd() is already _pkg_dir.
    change_pwd()
    _pkg_dir = os.getcwd()

    # setup() requires paths relative to the setup.py directory
    _ext_rel = "src/paddlefleet_ops/_extensions"

    ext_module = CUDAExtension(
        sources=[
            f"{_ext_rel}/fuse_transpose_split_fp8_quant.cu",
            f"{_ext_rel}/tokens_stable_unzip.cu",
            f"{_ext_rel}/tokens_unzip_gather.cu",
            f"{_ext_rel}/tokens_zip_unique_add.cu",
            f"{_ext_rel}/tokens_zip_prob.cu",
            f"{_ext_rel}/merge_subbatch_cast.cu",
            f"{_ext_rel}/tokens_unzip_slice.cu",
            f"{_ext_rel}/fuse_swiglu_scale.cu",
            f"{_ext_rel}/fuse_weighted_swiglu_fp8_quant.cu",
            f"{_ext_rel}/router_metadata.cu",
            f"{_ext_rel}/count_cumsum.cu",
            f"{_ext_rel}/filter_scores.cu",
            f"{_ext_rel}/fuse_stack_transpose_fp8_quant.cu",
            f"{_ext_rel}/fuse_apply_rotary_pos_emb_vision.cu",
            f"{_ext_rel}/fused_swiglu_probs_bwd.cu",
        ],
        include_dirs=[str(Path(__file__).parent / _ext_rel)],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-w",
                "-Wno-abi",
                "-fPIC",
                "-std=c++17",
            ]
            + (["-DPADDLE_WITH_DNNL"] if paddle_compiled_with_onednn else []),
            "nvcc": nvcc_args,
        },
    )

    # Paddle's CUDAExtension (and its setup()) re-converts sources to absolute
    # paths internally via os.path.abspath(), which setuptools then rejects.
    # We wrap the extension in a subclass that intercepts the `sources`
    # attribute: writes always store the raw value (absolute paths are fine for
    # the compiler), but reads always return paths relative to _pkg_dir so
    # setuptools' validation passes.
    class _RelativeSourcesExt(ext_module.__class__):
        @property
        def sources(self):
            return [os.path.relpath(s, _pkg_dir) if os.path.isabs(s) else s for s in self._sources]

        @sources.setter
        def sources(self, value):
            self._sources = value if value is not None else []

    # Read sources via the old class (plain list attribute) BEFORE switching,
    # then hand them to the new setter so _sources is initialised correctly.
    _saved_sources = list(ext_module.sources)
    ext_module.__class__ = _RelativeSourcesExt
    ext_module.sources = _saved_sources

    setup(
        name="paddlefleet_ops._extensions.ops",
        ext_modules=[ext_module],
        cmdclass={"bdist_wheel": CustomBdistWheel},
        install_requires=dependencies,
    )


def setup_install_no_extension():
    from setuptools import setup

    setup(
        name="paddlefleet-ops",
        install_requires=dependencies,
    )


try:
    dependencies = common_dependencies + get_special_build_deps() + get_special_setup_deps()
except Exception as e:
    raise Exception(f"Failed to resolve special dependencies: {e}, using common dependencies only") from e

if backends.IS_NVIDIA:
    setup_ops_extension()
elif backends.IS_XPU:
    setup_install_no_extension()
else:
    logging.error("\033[31m Error: Do not support this backend now.\033[0m\n")
