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
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import subprocess

IS_NVIDIA = False
IS_XPU = False
IS_ILUVATAR_GPU = False
IS_METAX_GPU = False
_initialized = False


def init_backend_type():
    global IS_NVIDIA, IS_XPU, IS_ILUVATAR_GPU, IS_METAX_GPU, _initialized
    if _initialized:
        return
    _initialized = True
    typelist = {"on", "yes", "1", "true"}
    IS_NVIDIA = os.environ.get("IS_NVIDIA", "0").lower() in typelist
    IS_XPU = os.environ.get("IS_XPU", "0").lower() in typelist
    IS_ILUVATAR_GPU = os.environ.get("IS_ILUVATAR_GPU", "0").lower() in typelist
    IS_METAX_GPU = os.environ.get("IS_METAX_GPU", "0").lower() in typelist
    if IS_NVIDIA or IS_XPU or IS_ILUVATAR_GPU or IS_METAX_GPU:
        return

    try:
        subprocess.check_output(["nvidia-smi"])
        IS_NVIDIA = True
        print("Backend is NVIDIA. IS_NVIDIA =", IS_NVIDIA)
        return
    except Exception:
        # print("Backend is not NVIDIA")
        pass
    try:
        subprocess.check_output(["xpu-smi"])
        IS_XPU = True
        print("Backend is XPU. IS_XPU =", IS_XPU)
        return
    except Exception:
        # print("Backend is not XPU")
        pass

    try:
        subprocess.check_output(["ixsmi"])
        IS_ILUVATAR_GPU = True
        print("Backend is ILUVATAR-GPU. IS_ILUVATAR_GPU =", IS_ILUVATAR_GPU)
        return
    except Exception:
        # print("Backend is not ILUVATAR-GPU")
        pass

    try:
        subprocess.check_output(["mx-smi"])
        IS_METAX_GPU = True
        print("Backend is Metax-GPU. IS_METAX_GPU =", IS_METAX_GPU)
        return
    except Exception:
        # print("Backend is not Metax-GPU")
        pass
    if not (IS_NVIDIA or IS_XPU or IS_ILUVATAR_GPU or IS_METAX_GPU):
        logging.getLogger(
            "Please verify your environment and ensure that device information retrieval commands are functional. NVIDIA(nvidia-smi), XPU(xpu-smi), MetaX-GPU(mx-smi), and Iluvatar-GPU(ixsmi) are supported! You may also configure environment variables manually: IS_NVIDIA/IS_XPU/IS_ILUVATAR_GPU/IS_METAX_GPU"
        )


init_backend_type()
