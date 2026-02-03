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

import builtins
import importlib.util
import sys


class TorchBlocker:
    def __init__(self, verbose: bool = True):
        self.torch_module = {}
        self.PF = False
        self.TF = False
        self.PF_RESR = True
        self.block_torch = False
        self.torch_transformers_pf = False
        self._original_import = builtins.__import__
        self._original_find_spec = importlib.util.find_spec

        builtins.__import__ = self._custom_import
        importlib.util.find_spec = self._fake_find_spec

    def _fake_find_spec(self, name, package=None):
        # frame = sys._getframe(1)
        # while frame:
        #     filename = frame.f_code.co_filename or ""
        #     if "PaddleFormers/tests/" in filename:
        #         return self._original_find_spec(name, package)
        #     frame = frame.f_back
        if self.block_torch and (name == "torch" or name.startswith("torch.") or name.startswith("torchvision")):
            return None
        return self._original_find_spec(name, package)

    def _is_called_from_paddleformers(self, current_globals) -> bool:
        if current_globals is not None:
            caller_mod = current_globals.get("__package__") or current_globals.get("__name__", "") or ""
            if caller_mod.startswith("paddleformers"):
                return True
        frame = sys._getframe(1)
        while frame:
            filename = frame.f_code.co_filename or ""
            if "paddleformers" in filename and "torch_blocker" not in filename:
                return True
            frame = frame.f_back
        return False

    def _custom_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        frame = sys._getframe(1)
        while frame:
            filename = frame.f_code.co_filename or ""
            if "PaddleFormers/tests/data" in filename:
                return self._original_import(name, globals, locals, fromlist, level)
            frame = frame.f_back
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

        top_level = (full_name or "").split(".")[0]

        if top_level == "paddleformers" and self.torch_transformers_pf is False:
            for module in [
                i for i in sys.modules.keys() if i.startswith("transformers") or i.startswith("paddleformers")
            ]:
                sys.modules.pop(module)
            for module_name in [i for i in sys.modules.keys() if i.startswith("torch")]:
                module = sys.modules.pop(module_name)
                self.torch_module[module_name] = module
            self.torch_transformers_pf = True

        if top_level not in ("transformers", "torch"):
            return self._original_import(name, globals, locals, fromlist, level)
        # print(">>",top_level)
        # if top_level == "paddleformers" and self.TF == True:
        #     for module in [i for i in sys.modules.keys() if i.startswith("transformers")]:
        #         sys.modules.pop(module)
        #     self.TF = False
        from_paddleformers = self._is_called_from_paddleformers(globals)
        if from_paddleformers is False:
            if self.PF is False:
                self.block_torch = False
            else:
                for module in [i for i in sys.modules.keys() if i.startswith("transformers")]:
                    sys.modules.pop(module)
                for module_name, module in self.torch_module.items():
                    sys.modules[module_name] = module
                self.torch_transformers_pf = False
                self.torch_module = {}
                self.PF = False
                self.PF_RESR = True
                self.block_torch = False
        else:
            self.PF = True
            self.torch_transformers_pf = True
            if self.PF_RESR is True:
                for module in [i for i in sys.modules.keys() if i.startswith("transformers")]:
                    sys.modules.pop(module)
                for module_name in [i for i in sys.modules.keys() if i.startswith("torch")]:
                    module = sys.modules.pop(module_name)
                    self.torch_module[module_name] = module
                self.PF_RESR = False
                self.block_torch = True
            else:
                self.block_torch = True

        return self._original_import(name, globals, locals, fromlist, level)
