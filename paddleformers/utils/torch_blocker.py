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
# See the License for the specific l

import builtins
import importlib.util
import sys
import traceback


class TorchBlocker:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._stack = []
        self._torch_blocked = False
        self._torch_backup = {}
        self._transformers_by_paddleformers = False
        self._in_paddleformers_init = True

        self._original_import = builtins.__import__
        self._original_find_spec = importlib.util.find_spec
        self._block_torch()

    def _fake_find_spec(self, name, package=None):
        if self._torch_blocked and (name == "torch" or name.startswith("torch.")):
            return None
        return self._original_find_spec(name, package)

    def _block_torch(self):
        if self._torch_blocked:
            return

        self._torch_backup = {}
        for name in list(sys.modules.keys()):
            if name == "torch" or name.startswith("torch."):
                self._torch_backup[name] = sys.modules.pop(name)

        importlib.util.find_spec = self._fake_find_spec
        self._torch_blocked = True

    def _unblock_torch(self):
        if not self._torch_blocked:
            return

        importlib.util.find_spec = self._original_find_spec

        for name, mod in self._torch_backup.items():
            sys.modules[name] = mod
        self._torch_backup = {}

        self._torch_blocked = False

    def _clear_transformers_cache(self):
        to_remove = [name for name in sys.modules.keys() if name == "transformers" or name.startswith("transformers.")]
        for name in to_remove:
            del sys.modules[name]

    def _is_called_from_paddleformers(self):
        if any("paddleformers" in s for s in self._stack):
            return True

        for frame_info in traceback.extract_stack():
            if "paddleformers" in frame_info.filename and "torch_blocker" not in frame_info.filename:
                return True

        return False

    def _custom_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level > 0 and globals:
            pkg = globals.get("__package__") or globals.get("__name__", "")
            if pkg:
                parts = pkg.split(".")
                base = ".".join(parts[: len(parts) - level + 1]) if level <= len(parts) else ""
                full_name = f"{base}.{name}" if name and base else (name or base)
            else:
                full_name = name
        else:
            full_name = name

        if self._torch_blocked and (full_name == "torch" or full_name.startswith("torch.")):
            raise ImportError("torch is blocked (paddleformers mode)")

        from_paddleformers = self._is_called_from_paddleformers()

        if full_name.startswith("paddleformers") and not self._torch_blocked:
            self._block_torch()

        if full_name == "transformers":
            if from_paddleformers:
                self._transformers_by_paddleformers = True
            else:
                if self._transformers_by_paddleformers:
                    self._unblock_torch()
                    self._clear_transformers_cache()
                    self._transformers_by_paddleformers = False
        self._stack.append(full_name)
        try:
            return self._original_import(name, globals, locals, fromlist, level)
        finally:
            self._stack.pop()

    def _start(self):
        builtins.__import__ = self._custom_import

    def stop(self):
        builtins.__import__ = self._original_import
        self._unblock_torch()

    def reset(self):
        self._stack = []
        self._transformers_by_paddleformers = False
        self._unblock_torch()

    def finish_paddleformers_init(self):
        self._in_paddleformers_init = False

    @property
    def is_torch_blocked(self) -> bool:
        return self._torch_blocked

    @property
    def is_transformers_loaded_by_paddleformers(self) -> bool:
        return self._transformers_by_paddleformers
