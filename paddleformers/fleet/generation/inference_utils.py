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

"""Inference utilities for PaddleFleet models."""

from __future__ import annotations

import os

import paddle


def init_inference_fleet(ep_degree: int = 1) -> None:
    """Initialize PaddlePaddle distributed environment for inference.

    This is a convenience wrapper that handles both multi-GPU launch
    (via ``paddle.distributed.launch``) and single-GPU cases.

    Args:
        ep_degree: Expert parallelism degree. When > 1, initializes
            the distributed environment for MoE expert sharding via DeepEP.
    """
    if ep_degree > 1:
        if not paddle.distributed.is_initialized():
            paddle.distributed.init_parallel_env()
    else:
        # Single GPU: still set device
        # Handle multi-GPU CUDA_VISIBLE_DEVICES like "0,1,2,3" by taking first device.
        # Also tolerate empty / unset env (e.g. cleared in container scripts).
        cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").strip() or "0"
        gpu_id = int(cvd.split(",")[0])
        paddle.set_device(f"cuda:{gpu_id}")
