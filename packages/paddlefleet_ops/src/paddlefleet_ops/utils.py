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
from __future__ import annotations

import importlib
import inspect
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types


def import_custom_ops(package, module_name, global_ns):
    """
    Imports custom operations from a specified module within a package and adds them to a global namespace.

    Args:
        package (str): The name of the package containing the module.
        module_name (str): The name of the module within the package.
        global_ns (dict): The global namespace to add the imported functions to.
    """
    try:
        module = importlib.import_module(module_name, package=package)
        functions = inspect.getmembers(module, inspect.isfunction)
        for func_name, func in functions:
            if func_name.startswith("__") or func_name == "_C_ops":
                continue
            logging.debug(f"Import {func_name} from {package}")
            try:
                global_ns[func_name] = func
            except Exception as e:
                logging.warning(f"Failed to import op {func_name}: {e}")

    except Exception:
        logging.warning(f"Ops of {package} import failed, it may be not compiled.")


class ModuleContext:
    """
    Manages the context for loading a module, including:
    1. Stashing existing modules to prevent conflicts.
    2. Managing sys.path.
    3. Restoring stashed modules upon exit.
    """

    def __init__(self, module_names: list[str], path: Path):
        self.module_names = module_names
        self.path = str(path)
        self._stash: dict[str, types.ModuleType] = {}

    def _stash_modules(self):
        """Moves modules matching module_name from sys.modules to stash."""
        for name in list(sys.modules.keys()):
            for module_name in self.module_names:
                if name == module_name or name.startswith(module_name + "."):
                    self._stash[name] = sys.modules.pop(name)

    def _restore_modules(self):
        """Restores stashed modules to sys.modules."""
        sys.modules.update(self._stash)
        self._stash.clear()

    def __enter__(self):
        self._stash_modules()
        sys.path.insert(0, self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path in sys.path:
            sys.path.remove(self.path)
        self._restore_modules()


def patch_module_namespace(source_name: str, target_prefix: str):
    """
    Moves loaded modules from source_name to target_prefix + source_name.
    Effectively 'installs' the module into the new namespace.
    """
    for name in list(sys.modules.keys()):
        if name == source_name or name.startswith(source_name + "."):
            module = sys.modules.pop(name)
            new_name = target_prefix + name
            sys.modules[new_name] = module


def clean_module_namespace(module_name: str):
    for name in list(sys.modules.keys()):
        if name == module_name:
            sys.modules.pop(name)


class HardwareIncompatibleBlocker(importlib.abc.MetaPathFinder):
    """Blocks imports for modules that do not meet runtime requirements."""

    def __init__(self, error_messages: dict[str, str]):
        self.error_messages = error_messages

    def find_spec(self, fullname, path, target=None):
        for module_name, error_msg in self.error_messages.items():
            if fullname == module_name or fullname.startswith(module_name + "."):
                raise RuntimeError(error_msg)


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_path(base_dir):
    path = Path(base_dir).joinpath("lib")
    for file in path.rglob("libnvshmem_host.so.*"):
        return file.resolve()
    raise FileNotFoundError(
        f"Could not locate 'libnvshmem_host.so.*' within the expected directory: {path}. "
        "Please ensure nvidia-nvshmem is installed correctly."
    )


def get_cuda_version():
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise FileNotFoundError(
            "nvcc command not found. Please make sure CUDA toolkit is installed and nvcc is in PATH."
        )

    result = subprocess.run(
        ["nvcc", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    version_output = result.stdout

    match = re.search(r"release (\d+)\.(\d+)", version_output)
    if not match:
        raise ValueError(f"Cannot parse CUDA version from nvcc output:\n{version_output}")
    cuda_major = int(match.group(1))
    cuda_minor = int(match.group(2))
    return cuda_major, cuda_minor
