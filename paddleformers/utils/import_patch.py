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

import importlib.abc
import inspect
import sys
import traceback


class Restorer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if hasattr(self, "path_save"):
            sys.path = self.path_save
            del self.path_save
            sys.path_importer_cache.clear()
        return None


class TorchBlocker(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.restorer = Restorer()
        self.MAX_DEPTH = 1e10
        self.paddleformers_import_stack_depth = self.MAX_DEPTH

    def find_spec(self, fullname, path, target=None):
        stack = traceback.extract_stack()
        stack_depth = len(stack)
        if self.paddleformers_import_stack_depth > stack_depth:
            self.paddleformers_import_stack_depth = self.MAX_DEPTH
        if fullname == "paddleformers" or fullname.startswith("paddleformers."):
            self.paddleformers_import_stack_depth = min(self.paddleformers_import_stack_depth, stack_depth)
        if fullname != "torch" and not fullname.startswith("torch."):
            return None
        stack = inspect.stack()
        for frame_info in stack[1:][:10]:
            filename = frame_info.filename.lower()
            if "transformers" in filename and stack_depth > self.paddleformers_import_stack_depth:
                self.restorer.path_save = sys.path[:]
                sys.path = []
                sys.path_importer_cache.clear()
                return None
        return None
