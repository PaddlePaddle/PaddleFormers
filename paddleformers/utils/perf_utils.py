# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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

import contextlib

import paddle

_PROFILER_ENABLED = False
_DEBUG_INFO = None


def memory_info():
    def _to_giga_bytes(nbytes):
        return nbytes / (1024 * 1024 * 1024)

    max_memory_allocated = _to_giga_bytes(paddle.device.cuda.max_memory_allocated())
    max_memory_reserved = _to_giga_bytes(paddle.device.cuda.max_memory_reserved())
    memory_allocated = _to_giga_bytes(paddle.device.cuda.memory_allocated())
    memory_reserved = _to_giga_bytes(paddle.device.cuda.memory_reserved())
    return f"memory_allocated={memory_allocated:.3f} GB, memory_reserved={memory_reserved:.3f}; max_memory_allocated={max_memory_allocated:.3f} GB, max_memory_reserved={max_memory_reserved:.3f} GB"


def _forward_pre_hook(layer, inputs):
    global _DEBUG_INFO
    if _DEBUG_INFO is not None:
        if _DEBUG_INFO == "memory":
            print(f"++ [Enter {layer.__class__.__name__} forward] {memory_info()}")
        else:
            print(f"++ [Enter {layer.__class__.__name__} forward] ")

    if _PROFILER_ENABLED:
        paddle.base.core.nvprof_nvtx_push(layer.__class__.__name__ + "_fwd")
    return None


def _forward_post_hook(layer, inputs, outputs):
    global _DEBUG_INFO
    if _DEBUG_INFO is not None:
        if _DEBUG_INFO == "memory":
            print(f"-- [Leave {layer.__class__.__name__} forward] {memory_info()}")
        else:
            print(f"-- [Leave {layer.__class__.__name__} forward] ")

    if _PROFILER_ENABLED:
        paddle.base.core.nvprof_nvtx_pop()


def _register_hook_recursively(layer, pre_hook, post_hook):
    if not isinstance(layer, paddle.nn.Layer):
        return

    for sublayer in layer.children():
        _register_hook_recursively(sublayer, pre_hook, post_hook)

    if pre_hook is not None:
        layer.register_forward_pre_hook(hook=pre_hook)
    if post_hook is not None:
        layer.register_forward_post_hook(hook=post_hook)


def register_profile_hook(model, debug=None):
    if debug is not None:
        global _DEBUG_INFO
        _DEBUG_INFO = debug

    if isinstance(model, paddle.nn.Layer):
        _register_hook_recursively(model, _forward_pre_hook, _forward_post_hook)
    elif isinstance(model, list):
        for layer in model:
            _register_hook_recursively(layer, _forward_pre_hook, _forward_post_hook)


def switch_profile(iter_id, start, end, event_name=None, enable_layerwise_event=False):
    global _PROFILER_ENABLED
    if event_name is None:
        event_name = "iter_{}".format(iter_id)
    if iter_id == start:
        paddle.device.cuda.synchronize()
        paddle.base.core.nvprof_start()
        _PROFILER_ENABLED = True
        if enable_layerwise_event:
            paddle.profiler.utils._is_profiler_used = True
            paddle.base.core.nvprof_enable_record_event()
        paddle.base.core.nvprof_nvtx_push(event_name)
    elif iter_id == end:
        paddle.base.core.nvprof_nvtx_pop()
        _PROFILER_ENABLED = False
        paddle.base.core.nvprof_stop()
        if enable_layerwise_event:
            paddle.profiler.utils._is_profiler_used = False
    elif iter_id > start and iter_id < end:
        paddle.base.core.nvprof_nvtx_pop()
        paddle.base.core.nvprof_nvtx_push(event_name)


@contextlib.contextmanager
def add_record_event(event_name):
    if _PROFILER_ENABLED:
        paddle.base.core.nvprof_nvtx_push(event_name)
        yield
        paddle.base.core.nvprof_nvtx_pop()
    else:
        yield


def push_record_event(event_name):
    if _PROFILER_ENABLED:
        paddle.base.core.nvprof_nvtx_push(event_name)


def pop_record_event():
    if _PROFILER_ENABLED:
        paddle.base.core.nvprof_nvtx_pop()
