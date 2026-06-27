#!/usr/bin/env bash

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

set -e

DIFF_REF="${1:-${PADDLEFORMERS_CI_DIFF_REF:-${AGILE_COMPILE_BRANCH:-}}}"
if [[ $# -gt 0 ]]; then
    shift
fi
PIP_INSTALL_ARGS=("$@")

packages_changed() {
    if [[ -z "${DIFF_REF}" ]]; then
        echo "No diff ref provided, install sparse-mapped paddlefleet_ops wheel by packages commit"
        return 1
    fi

    if git diff --name-only "${DIFF_REF}" -- packages/ | grep -q .; then
        return 0
    fi
    return 1
}

detect_cuda_version() {
    local cuda_ver=""
    if command -v nvcc &>/dev/null; then
        cuda_ver=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    elif command -v nvidia-smi &>/dev/null; then
        cuda_ver=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    elif [[ -n "${CUDA_HOME:-}" ]]; then
        cuda_ver=$("${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')
    elif [[ -n "${CUDA_PATH:-}" ]]; then
        cuda_ver=$("${CUDA_PATH}/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')
    fi
    echo "${cuda_ver}"
}

cuda_info() {
    local cuda_ver cuda_major cuda_minor cuda_suffix nvshmem_dep
    cuda_ver=$(detect_cuda_version)
    cuda_major=${cuda_ver%%.*}
    cuda_minor=${cuda_ver#*.}

    if [[ -z "${cuda_ver}" ]]; then
        echo "Cannot detect CUDA version for local paddlefleet_ops build" >&2
        exit 1
    fi

    case "${cuda_ver}" in
        "13.2")
            cuda_suffix="cu132"
            ;;
        "13.0")
            cuda_suffix="cu130"
            ;;
        "12.9")
            cuda_suffix="cu129"
            ;;
        *)
            echo "Unsupported CUDA version for local paddlefleet_ops build: ${cuda_ver}" >&2
            exit 1
            ;;
    esac

    if [[ "${cuda_major}" == "13" ]]; then
        nvshmem_dep="paddle-nvidia-nvshmem-cu13>=3.3.9,<3.5"
    elif [[ "${cuda_major}" == "12" && "${cuda_minor}" -gt 6 ]]; then
        nvshmem_dep="paddle-nvidia-nvshmem-cu12>=3.3.9,<3.5"
    else
        nvshmem_dep="nvidia-nvshmem-cu12>=3.3.9,<3.5"
    fi

    echo "${cuda_suffix} ${nvshmem_dep}"
}

install_build_deps() {
    local cuda_suffix nvshmem_dep
    read -r cuda_suffix nvshmem_dep < <(cuda_info)
    local cuda_index="https://www.paddlepaddle.org.cn/packages/nightly/${cuda_suffix}/"

    uv pip install --system --group paddlefleet-ops-build \
        "${PIP_INSTALL_ARGS[@]}"
    python -m pip install \
        "${nvshmem_dep}" \
        --extra-index-url "${cuda_index}" \
        "${PIP_INSTALL_ARGS[@]}"
}

install_local_ops_wheel() {
    local cuda_suffix nvshmem_dep
    read -r cuda_suffix nvshmem_dep < <(cuda_info)
    local cuda_index="https://www.paddlepaddle.org.cn/packages/nightly/${cuda_suffix}/"
    python -m pip install dist/paddlefleet_ops-*.whl \
        --extra-index-url "${cuda_index}" \
        "${PIP_INSTALL_ARGS[@]}"
}

build_and_install_local_ops() {
    echo "packages/ changed, build paddlefleet_ops wheel from current workspace"
    export UV_SKIP_WHEEL_FILENAME_CHECK=${UV_SKIP_WHEEL_FILENAME_CHECK:-1}
    export IS_NVIDIA=${IS_NVIDIA:-True}

    if ! command -v uv &>/dev/null; then
        python -m pip install uv "${PIP_INSTALL_ARGS[@]}"
    fi

    install_build_deps
    git submodule update --init --recursive
    rm -f dist/paddlefleet_ops-*.whl
    uv build --package paddlefleet-ops --wheel --out-dir dist --no-build-isolation -vv
    install_local_ops_wheel
}

if packages_changed; then
    build_and_install_local_ops
else
    echo "packages/ unchanged, install sparse-mapped paddlefleet_ops wheel by packages commit"
    bash scripts/install_ops_wheel.sh
fi
