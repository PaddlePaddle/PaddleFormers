# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import builtins
import functools
import importlib.util
import os
import shutil
import site
import sys
from contextlib import contextmanager
from typing import Optional, Tuple, Type, Union

import pip

from paddleformers.utils.log import logger

_original_import = builtins.__import__
_imported_modules = {}
_paddlenlp_ops_updated = False
_original_attributes = {}
pybind_ops_list = [
    "update_inputs_v2",
    "save_output",
    "set_preids_token_penalty_multi_scores",
    "rebuild_padding_v2",
    "append_attention",
    "save_output_dygraph",
    "per_token_group_quant",
    "per_tensor_quant_fp8",
]


def custom_import(name, *args, **kwargs):
    global _paddlenlp_ops_updated, _imported_modules, _original_attributes
    global pybind_ops_list

    if _paddlenlp_ops_updated:
        if name in _imported_modules:
            return _imported_modules[name]

    module = _original_import(name, *args, **kwargs)

    if not _paddlenlp_ops_updated and os.getenv("DYNAMIC_INFERENCE_MODE", "1").lower() in [
        "1",
        "true",
        "t",
        "yes",
        "y",
    ]:
        if name == "paddlenlp_ops":
            # logger.debug("Using Pybind paddlenlp_ops!")
            if name not in _original_attributes:
                bak_dict = {}
                for ops_name in pybind_ops_list:
                    bak_dict[ops_name] = getattr(module, ops_name, None)
                _original_attributes[name] = bak_dict

            for ops_name in pybind_ops_list:
                pybind_ops_name = f"f_{ops_name}"
                if hasattr(module, pybind_ops_name):
                    setattr(module, ops_name, getattr(module, pybind_ops_name))

            _paddlenlp_ops_updated = True

    _imported_modules[name] = module
    return module


@contextmanager
def dynamic_graph_pybind_context():
    global _original_import, _paddlenlp_ops_updated
    original_import = builtins.__import__

    try:
        builtins.__import__ = custom_import
        yield
    finally:
        builtins.__import__ = original_import

        if "paddlenlp_ops" in _original_attributes:
            paddlenlp_ops_module = sys.modules.get("paddlenlp_ops")
            if paddlenlp_ops_module:
                for attr, value in _original_attributes["paddlenlp_ops"].items():
                    setattr(paddlenlp_ops_module, attr, value)
                _paddlenlp_ops_updated = False


def auto_dynamic_graph_pybind(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with dynamic_graph_pybind_context():
            return func(self, *args, **kwargs)

    return wrapper


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_g2p_en_available = _is_package_available("g2p_en")
_sentencepiece_available = _is_package_available("sentencepiece")
_sklearn_available = importlib.util.find_spec("sklearn") is not None
if _sklearn_available:
    try:
        importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:
        _sklearn_available = False


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_g2p_en_available = _is_package_available("g2p_en")
_sentencepiece_available = _is_package_available("sentencepiece")
_sklearn_available = importlib.util.find_spec("sklearn") is not None
if _sklearn_available:
    try:
        importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError:
        _sklearn_available = False


def is_datasets_available():
    import importlib

    return importlib.util.find_spec("datasets") is not None


def is_protobuf_available():
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


def is_paddle_cuda_available() -> bool:
    if is_paddle_available():
        import paddle

        return paddle.device.cuda.device_count() > 0
    else:
        return False


def is_g2p_en_available():
    return _g2p_en_available


def is_sentencepiece_available():
    return _sentencepiece_available


def is_paddle_available() -> bool:
    """check if `torch` package is installed
    Returns:
        bool: if `torch` is available
    """
    return is_package_available("paddle")


def is_tiktoken_available():
    return importlib.util.find_spec("tiktoken") is not None


def is_psutil_available():
    return importlib.util.find_spec("psutil") is not None


def is_torch_available() -> bool:
    """check if `torch` package is installed
    Returns:
        bool: if `torch` is available
    """
    return is_package_available("torch")


def is_package_available(package_name: str) -> bool:
    """check if the package is available
    Args:
        package_name (str): the installed package name
    Returns:
        bool: the existence of installed package
    """
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None and package_spec.has_location


def is_fast_tokenizer_available() -> bool:
    """check if `fast_tokenizer` ia available
    Returns:
        bool: if `fast_tokenizer` is available
    """
    return is_package_available("fast_tokenizer")


def is_tokenizers_available() -> bool:
    """check if `tokenizers` ia available
    Returns:
        bool: if `tokenizers` is available
    """
    return is_package_available("tokenizers")


def is_paddlenlp_ops_available() -> bool:
    """check if `paddlenlp_ops` ia available
    Returns:
        bool: if `paddlenlp_ops` is available
    """
    return is_package_available("paddlenlp_ops")


def is_transformers_available() -> bool:
    """check if `transformers` package is installed
    Returns:
        bool: if `transformers` is available
    """
    return is_package_available("transformers")


def install_package(
    package_name: str,
    version: Optional[str] = None,
    module_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """install the specific version of package

    Args:
        package_name (str): the name of package
        version (str): the version of package
        module_name (str): the imported name of package
        cache_dir (str): cache dir
    """
    module_name = module_name or package_name

    # 1. remove the existing version of package
    uninstall_package(package_name, module_name)

    # 2. install the package
    if version:
        package_name += f"=={version}"

    arguments = ["install"]
    if cache_dir:
        arguments += ["-t", cache_dir]
        sys.path.insert(0, cache_dir)

    # 3. load the pypi mirror to speedup of installing packages
    mirror_key = "PYPI_MIRROR"
    mirror_source = os.environ.get(mirror_key, None)
    if mirror_source is not None:
        logger.info(f"loading <{mirror_source}> from as the final mirror source to install package.")
        arguments += ["-i", mirror_source]

    arguments += [package_name]
    pip.main(arguments)

    # 4. add site-package to the top of package
    for site_package_dir in site.getsitepackages():
        sys.path.insert(0, site_package_dir)


def uninstall_package(package_name: str, module_name: Optional[str] = None):
    """uninstall the package from site-packages.

    To remove the cache of source package module & class & method, it should:
        1. remove the source files of packages under the `site-packages` dir.
        2. remove the cache under the `locals()`
        3. remove the cache under the `sys.modules`

    Args:
        package_name (str): the name of package
    """
    module_name = module_name or package_name
    for site_package_dir in site.getsitepackages():
        if os.path.exists(site_package_dir):
            for file in os.listdir(site_package_dir):
                package_dir = os.path.join(site_package_dir, file)
                if file.startswith(package_name) and os.path.isdir(package_dir):
                    shutil.rmtree(package_dir)

    for site_package_dir in site.getsitepackages():
        while sys.path[0] == site_package_dir:
            sys.path.pop(0)

    for key in list(locals().keys()):
        if module_name in key:
            del locals()[key]

    for key in list(sys.modules.keys()):
        if module_name in key:
            del sys.modules[key]


def import_module(module_name: str) -> Optional[Type]:
    """import module base on the model
    Args:
        module_name (str): the name of target module
    """
    # 1. prepare the name
    assert "." in module_name, "`.` must be in the module_name"
    index = module_name.rindex(".")
    module = module_name[:index]
    target_module_name = module_name[index + 1 :]

    # 2. get the target module name
    try:
        module = importlib.import_module(module)
        target_module = getattr(module, target_module_name, None)
        return target_module
    except ModuleNotFoundError:
        return None
