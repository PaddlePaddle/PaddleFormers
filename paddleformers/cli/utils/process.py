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

import os
import platform
import re
import subprocess

import paddle
import psutil


def terminate_process_tree(pid: int) -> None:
    """
    Terminate the process tree of the given process ID

    Args:
        pid (int): The process ID that needs to be terminated

    Returns:
        None
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(children, timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass

    try:
        parent.terminate()
    except psutil.NoSuchProcess:
        pass


def is_env_enabled(env_var: str, default: str = "0") -> bool:
    r"""Check if the environment variable is enabled."""
    return os.getenv(env_var, default).lower() in ["true", "y", "1"]


def is_valid_model_dir(directory: str) -> bool:
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            if item.lower().endswith((".safetensors", ".pdparams")):
                return True
    return False


def detect_device() -> str:
    """
    Detect the current device type (GPU/NPU/XPU).

    Returns:
        str: Device type ('gpu', 'npu', 'xpu')
    """
    try:
        place = paddle.get_device()
        place_lower = place.lower()

        if "npu" in place_lower:
            return "npu"
        elif "xpu" in place_lower:
            return "xpu"
        elif "iluvatar" in place_lower:
            return "iluvatar_gpu"
        else:
            return "gpu"
    except Exception as e:
        print(f"Error detecting device: {e}")


def set_ascend_environment():
    """Configure environment variables for Huawei Ascend AI accelerator."""

    # Get system architecture (e.g., x86_64, arm64)
    arch = platform.machine()

    # Base path for Ascend Toolkit
    ascend_toolkit_home = "/usr/local/Ascend/ascend-toolkit/latest"

    # Construct LD_LIBRARY_PATH components
    ld_library_path_parts = [
        # Driver libraries
        "/usr/local/Ascend/driver/lib64",
        "/usr/local/Ascend/driver/lib64/common",
        "/usr/local/Ascend/driver/lib64/driver",
        # Toolkit libraries
        f"{ascend_toolkit_home}/lib64",
        f"{ascend_toolkit_home}/lib64/plugin/opskernel",
        f"{ascend_toolkit_home}/lib64/plugin/nnengine",
        # Architecture-specific TBE operator tiling libraries
        f"{ascend_toolkit_home}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/{arch}",
        # AML (Ascend Machine Learning) libraries
        f"{ascend_toolkit_home}/tools/aml/lib64",
        f"{ascend_toolkit_home}/tools/aml/lib64/plugin",
    ]

    # Preserve existing LD_LIBRARY_PATH and prepend new paths
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if current_ld_library_path:
        ld_library_path_parts.insert(0, current_ld_library_path)

    # Construct PYTHONPATH components
    pythonpath_parts = [
        # Python site-packages from toolkit
        f"{ascend_toolkit_home}/python/site-packages",
        # TBE (Tensor Boost Engine) operator implementation
        f"{ascend_toolkit_home}/opp/built-in/op_impl/ai_core/tbe",
        # Preserve existing PYTHONPATH
        os.environ.get("PYTHONPATH", ""),
    ]

    # Construct PATH components
    path_parts = [
        # Toolkit binaries
        f"{ascend_toolkit_home}/bin",
        # Compiler binaries
        f"{ascend_toolkit_home}/compiler/ccec_compiler/bin",
        f"{ascend_toolkit_home}/tools/ccec_compiler/bin",
        # Preserve existing PATH
        os.environ.get("PATH", ""),
    ]

    # Set all environment variables
    os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, ld_library_path_parts))
    os.environ["ASCEND_TOOLKIT_HOME"] = ascend_toolkit_home
    os.environ["PYTHONPATH"] = ":".join(filter(None, pythonpath_parts))
    os.environ["PATH"] = ":".join(filter(None, path_parts))

    # Additional Ascend-specific environment variables
    os.environ["ASCEND_AICPU_PATH"] = ascend_toolkit_home
    os.environ["ASCEND_OPP_PATH"] = f"{ascend_toolkit_home}/opp"  # Operator package path
    os.environ["TOOLCHAIN_HOME"] = f"{ascend_toolkit_home}/toolkit"
    os.environ["ASCEND_HOME_PATH"] = ascend_toolkit_home


def remove_paddle_shm_files():
    try:
        subprocess.run(
            r'find /dev/shm/ -type f -name "paddle_*" -print0 | xargs -0 rm -f',
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"error while deleting : {e}")


def set_cuda_environment():
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.PIPE, text=True)

        cuda_version_match = re.search(r"CUDA Version:\s+(\d+)", nvidia_smi_output)
        if cuda_version_match:
            cuda_version = cuda_version_match.group(1)
            print(f"cuda version checked: {cuda_version}")

            if cuda_version != "12":
                ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
                new_ld_path = f"/usr/local/cuda/compat:{ld_library_path}"
                os.environ["LD_LIBRARY_PATH"] = new_ld_path
                print(f"set LD_LIBRARY_PATH to: {new_ld_path}")
        else:
            print("cannot detect cuda version from nvidia-smi")

    except subprocess.CalledProcessError as e:
        print(f"run nvidia-smi error: {e}")
    except Exception as e:
        print(f"process cuda version error: {e}")
