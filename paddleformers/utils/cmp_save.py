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

import hashlib
import os
import traceback

import numpy as np
import paddle

FILE_DIR = "./saved_tensors/npy"


def compare_and_save(data, name: str, to_save: bool = False, print_tensor: bool = False):
    if print_tensor:
        print(name, type(data), data.shape if data is not None else None, data)
    try:
        if isinstance(data, paddle.Tensor):
            if data.dtype == paddle.complex64:
                data_np = data.detach().cpu().numpy()
            else:
                data_float = data.astype("float32")
                data_np = data_float.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            data_np = data
        else:
            data_float = data.float().contiguous()
            data_np = data_float.detach().cpu().numpy()

        array_bytes = data_np.tobytes()
        data_md5 = hashlib.md5(array_bytes).hexdigest()
        print(
            f"{name} md5: {data_md5}, dtype: {data.dtype}, shape: {data.shape if data is not None else None}, device: {data.device}"
        )
        if to_save:
            os.makedirs(FILE_DIR, exist_ok=True)
            file = FILE_DIR + name + ".npy"
            np.save(file, data_np)
    except:
        print(name, traceback.format_exc())
