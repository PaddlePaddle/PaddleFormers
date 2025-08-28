# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import importlib
import importlib.machinery
import importlib.util
import os
from itertools import chain
from types import ModuleType
from typing import Any, Dict, List, Optional


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py

    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: dict[str, set[str]],
        module_spec: Optional[importlib.machinery.ModuleSpec] = None,
        extra_objects: Optional[Dict[str, object]] = None,
        explicit_import_shortcut: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(name)

        self._object_missing_backend = {}
        self._explicit_import_shortcut = explicit_import_shortcut if explicit_import_shortcut else {}

        self._modules = set(import_structure.keys())
        self._class_to_module = {value: key for key, values in import_structure.items() for value in values}
        # Needed for autocompletion in an IDE
        self.__all__ = [*self._modules, *chain.from_iterable(import_structure.values())]
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    def __dir__(self) -> List[str]:
        """Custom dir() implementation for better IDE support."""
        result = list(super().__dir__())
        result.extend(attr for attr in self.__all__ if attr not in result)
        return sorted(result)

    def __getattr__(self, name: str) -> Any:
        """Lazy import mechanism for module attributes."""
        # Check cached objects first
        if name in self._objects:
            return self._objects[name]

        # Handle regular imports from import_structure
        if name in self._class_to_module:
            try:
                module = self._get_module(self._class_to_module[name])
                value = getattr(module, name)
            except (ModuleNotFoundError, RuntimeError) as e:
                raise ModuleNotFoundError(
                    f"Could not import module '{name}'. Are this object's requirements defined correctly?"
                ) from e
        elif name in self._modules:
            try:
                value = self._get_module(name)
            except (ModuleNotFoundError, RuntimeError) as e:
                raise ModuleNotFoundError(
                    f"Could not import module '{name}'. Are this object's requirements defined correctly?"
                ) from e
        else:
            # Handle explicit import shortcuts
            value = None
            for key, values in self._explicit_import_shortcut.items():
                if name in values:
                    value = self._get_module(key)

            if value is None:
                raise AttributeError(f"module {self.__name__} has no attribute {name}")

        # Cache the resolved value
        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        """Internal helper for safely importing submodules."""
        try:
            return importlib.import_module(f".{module_name}", self.__name__)
        except Exception as e:
            raise e

    def __reduce__(self):
        """Support for pickle protocol."""
        return (self.__class__, (self._name, self.__file__, self._import_structure))
