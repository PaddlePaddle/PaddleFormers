# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import json
import mmap
from collections import OrderedDict

import ml_dtypes
import numpy as np

__all__ = [
    "fast_safe_open",
    "fast_load_file",
]

np.bfloat16 = ml_dtypes.bfloat16
np.float8_e5m2 = ml_dtypes.float8_e5m2
np.float8_e4m3fn = ml_dtypes.float8_e4m3fn


MAX_HEADER_SIZE = 100 * 1000 * 1000

dtype_size = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}

numpy_dtype = {
    "BOOL": np.bool_,
    "U8": np.uint8,
    "I8": np.int8,
    "F8_E5M2": np.float8_e5m2,  # no fp8
    "F8_E4M3": np.float8_e4m3fn,  # no fp8
    "I16": np.int16,
    "U16": np.uint16,
    "I32": np.int32,
    "U32": np.uint32,
    "I64": np.int64,
    "U64": np.uint64,
    "F16": np.float16,
    "BF16": np.bfloat16,  # no bf16
    "F32": np.float32,
    "F64": np.float64,
}


def getSize(fileobject):
    fileobject.seek(0, 2)  # move the cursor to the end of the file
    size = fileobject.tell()
    fileobject.seek(0)  # move the cursor to the start of the file
    return size


def metadata_validate(metadata):
    start = 0
    for key, info in metadata.items():
        s, e = info["data_offsets"]
        if s != start or e < s:
            raise ValueError(f"SafeTensorError::InvalidOffset({key})")
        start = e
        nelements = np.prod(info["shape"])
        nbytes = nelements * dtype_size[info["dtype"]]
        if (e - s) != nbytes:
            raise ValueError("SafeTensorError::TensorInvalidInfo")
    return start


def read_metadata(buffer):
    buffer_len = getSize(buffer)
    if buffer_len < 8:
        raise ValueError("SafeTensorError::HeaderTooSmall")

    n = np.frombuffer(buffer.read(8), dtype=np.uint64).item()
    if n > MAX_HEADER_SIZE:
        raise ValueError("SafeTensorError::HeaderTooLarge")

    stop = n + 8
    if stop > buffer_len:
        raise ValueError("SafeTensorError::InvalidHeaderLength")

    tensors = json.loads(buffer.read(n), object_pairs_hook=OrderedDict)
    metadata = tensors.pop("__metadata__", None)
    buffer_end = metadata_validate(tensors)

    if buffer_end + 8 + n != buffer_len:
        raise ValueError("SafeTensorError::MetadataIncompleteBuffer")

    return stop, tensors, metadata


def readinto_numpy(meta, buffer, base_ptr):
    def create_empty(info):
        return np.empty(shape=info["shape"], dtype=numpy_dtype[info["dtype"]])

    ret = {}
    for k, v in meta.items():
        t = create_empty(v)
        buffer.seek(base_ptr + v["data_offsets"][0])
        buffer.readinto(memoryview(t))
        ret[k] = t
    return ret


class PySafeSlice:
    def __init__(self, info, bufferfile, base_ptr, buffermmap):
        self.info = info
        self.bufferfile = bufferfile
        self.buffermmap = buffermmap
        self.base_ptr = base_ptr

        self.start = [0 for dim in self.shape]
        self.stop = [dim for dim in self.shape]
        self.step = [1 for dim in self.shape]

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, index):
        # https://github.com/numpy/numpy/blob/4d652465cea38e9504f954ac708d91e4954bd13a/numpy/lib/_arrayterator_impl.py#L96-L126
        # Fix index, handling ellipsis and incomplete slices.
        if not isinstance(index, tuple):
            index = (index,)
        fixed = []
        length, dims = len(index), self.ndim
        for slice_ in index:
            if slice_ is Ellipsis:
                fixed.extend([slice(None)] * (dims - length + 1))
                length = len(fixed)
            elif isinstance(slice_, int):
                fixed.append(slice(slice_, slice_ + 1, 1))
            else:
                fixed.append(slice_)
        index = tuple(fixed)
        if len(index) < dims:
            index += (slice(None),) * (dims - len(index))

        out_start, out_stop, out_step = copy.deepcopy((self.start, self.stop, self.step))
        for i, (start, stop, step, slice_) in enumerate(zip(self.start, self.stop, self.step, index)):
            out_start[i] = slice_.start if slice_.start is not None else 0
            out_step[i] = slice_.step if slice_.step is not None else 1
            out_stop[i] = slice_.stop if slice_.stop is not None else stop - start
            out_stop[i] = min(stop, out_stop[i])

        target_shape = []
        for x, y, z, sli in zip(out_start, out_stop, out_step, index):
            assert z == 1, "only support step = 1"
            if y - x > 1 or sli.step is None:
                target_shape.append(max(int(y - x), 0))

        if len(target_shape) == 0:
            if self.shape == [1]:
                target_shape = self.shape

        # https://github.com/huggingface/safetensors/blob/b947b59079a6197d7930dfb535818ac4896113e8/safetensors/src/slice.rs#L297-L315
        indices = []
        span = self.bits
        for i, (start, stop, step) in enumerate(zip(out_start[::-1], out_stop[::-1], out_step[::-1])):
            if len(indices) == 0:
                if start == 0 and stop == self.shape[::-1][i]:
                    pass
                    #  We haven't started to slice yet, just increase the span
                else:
                    offset = start * span
                    small_span = stop * span - offset
                    indices.append((offset, offset + small_span))

            else:
                capacity = (stop - start) * len(indices)
                newindices = []
                for n in range(start, stop):
                    offset = n * span
                    for (old_start, old_stop) in indices:
                        newindices.append((old_start + offset, old_stop + offset))
                indices = newindices
                assert len(indices) == capacity, f"error {capacity} {len(indices)}"
            span *= self.shape[::-1][i]

        if len(indices) == 0:
            indices.append((0, self.nbytes))

        merge_indices = []
        last_end = -1
        last_start = -1
        for start, end in indices:
            if start == last_end:
                last_end = end
                continue
            else:
                if last_start != -1:
                    merge_indices.append((last_start, last_end))
                last_start = start
                last_end = end
        if last_start != -1:
            merge_indices.append((last_start, last_end))
        tensor = np.empty(shape=[1] if len(target_shape) == 0 else np.prod(target_shape), dtype=self.dtype)

        tensor_view = memoryview(tensor.view(np.uint8).reshape(-1))
        curr_data_ptr = 0
        # if to many slice and each slice < 1M
        if len(merge_indices) > 128 and (merge_indices[0][1] - merge_indices[0][0] < 1024 * 1024):
            # Use mmap for random access
            for start, end in merge_indices:
                data_len = end - start
                tensor_view[curr_data_ptr : curr_data_ptr + data_len] = self.buffermmap[
                    self.start_offset + start : self.start_offset + end
                ]
                curr_data_ptr += data_len
        else:
            # Use file read for sequence access
            for start, end in merge_indices:
                data_len = end - start
                self.bufferfile.seek(self.start_offset + start)
                view = tensor_view[curr_data_ptr : curr_data_ptr + data_len]
                self.bufferfile.readinto(view)
                curr_data_ptr += data_len

        return tensor.reshape(target_shape)

    def get(self, *args, **kwargs):
        # int fix for empty shape []
        nbytes = int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize
        buffer = bytearray(nbytes)
        self.bufferfile.seek(self.start_offset)
        self.bufferfile.readinto(buffer)
        tensor = np.frombuffer(buffer, dtype=self.dtype).reshape(self.shape)
        return tensor

    @property
    def start_offset(self):
        return self.base_ptr + self.info["data_offsets"][0]

    def get_shape(self):
        return self.shape

    @property
    def shape(self):
        return self.info["shape"]

    @property
    def dtype(self):
        return numpy_dtype[self.info["dtype"]]

    @property
    def nelements(self):
        return np.prod(self.info["shape"])

    @property
    def bits(self):
        return dtype_size[self.info["dtype"]]

    @property
    def nbytes(self):
        return self.nelements * dtype_size[self.info["dtype"]]


# a simple file writer object
class fast_safe_open:
    def __init__(self, filename, framework=None, device="cpu"):
        self.filename = filename
        self.framework = framework
        self.file = open(self.filename, "rb")
        self.file_mmap = mmap.mmap(self.file.fileno(), 0, flags=mmap.MAP_PRIVATE)
        self.base, self.tensors_decs, self.__metadata__ = read_metadata(self.file)
        self.tensors = OrderedDict()
        for key, info in self.tensors_decs.items():
            self.tensors[key] = PySafeSlice(info, self.file, self.base, self.file_mmap)
            self.tensors[key].key = key

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file_mmap.close()
        self.file.close()

    def metadata(self):
        return self.__metadata__

    def keys(self):
        return list(self.tensors.keys())

    def get_tensor(self, name):
        return self.tensors[name].get()

    def get_slice(self, name):
        return self.tensors[name]


def fast_load_file(filename):
    result = {}
    with fast_safe_open(filename, framework="np") as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result
