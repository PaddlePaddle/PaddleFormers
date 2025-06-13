# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# https://github.com/NVIDIA/Megatron-LM/blob/060415572f4365a2e895f8036c4e37dad0efbdf5/megatron/data/indexed_dataset.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# copied from fairseq/fairseq/data/indexed_dataset.py
# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

import os
import shutil
import struct
import time
from dataclasses import fields
from functools import lru_cache
from itertools import accumulate

import numpy as np
import paddle


def print_rank_0(*args, **kwargs):
    if paddle.distributed.get_rank() == 0:
        print(*args, **kwargs)


def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def get_available_dataset_impl():
    return ["lazy", "mmap"]


def make_dataset(path, impl, skip_warmup=False):
    if CompatibleIndexedDataset.exists(path):
        print("Using old dataset (.npy & .npz)")
        return CompatibleIndexedDataset(path)
    elif not IndexedDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None
    elif impl == "lazy" and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == "mmap" and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None


def make_sft_dataset(path, dataclass, skip_warmup=False, impl="mmap"):
    if impl != "mmap":
        raise ValueError("SFT Indexed Dataset only support mmap memory-mapped method temporarily")

    print_rank_0(" > building dataset index ...")
    start_time = time.time()
    sft_indexed_dataset = SFTMMapIndexedDataset(path, dataclass, skip_warmup)
    print_rank_0(" > finished creating SFT indexed dataset in {:4f} " "seconds".format(time.time() - start_time))
    print_rank_0("    number of samples: {}".format(len(sft_indexed_dataset.doc_idx) - 1))

    return sft_indexed_dataset


def dataset_exists(path, impl):
    if impl == "mmap":
        return MMapIndexedDataset.exists(path)
    else:
        return IndexedDataset.exists(path)


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


def read_shorts(f, n):
    a = np.empty(n, dtype=np.int32)
    f.readinto(a)
    return a


def write_shorts(f, a):
    f.write(np.array(a, dtype=np.int32))


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
    9: np.uint32,
    10: np.uint64,
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def sft_index_file_path(prefix_path):
    return os.path.join(prefix_path, "index.idx")


def sft_data_file_path(prefix_path, dataclass):
    file_path_list = []
    for field in fields(dataclass):
        file_path = os.path.join(prefix_path, f"{field.name}.bin")
        file_path_list.append(file_path)
    return file_path_list


def data_file_path(prefix_path):
    return prefix_path + ".bin"


def loss_mask_file_path(prefix_path):
    return prefix_path + ".lsm"


def create_doc_idx(sizes):
    doc_idx = [0]
    for i, s in enumerate(sizes):
        if s == 0:
            doc_idx.append(i + 1)
    return doc_idx


class IndexedDataset(paddle.io.Dataset):
    """Loader for IndexedDataset"""

    _HDR_MAGIC = b"TNTIDX\x00\x00"

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                "Index file doesn't match expected format. " "Make sure that --dataset-impl is configured properly."
            )
            version = f.read(8)
            assert struct.unpack("<Q", version) == (1,)
            code, self.element_size = struct.unpack("<QQ", f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack("<QQ", f.read(16))
            self.doc_count = struct.unpack("<Q", f.read(8))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_shorts(f, self.s)
            self._doc_idx = read_longs(f, self.doc_count)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), "rb", buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError("index out of range")

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if not self.data_file:
            self.read_data(self.path)
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i] : self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            return a
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sizes = self.sizes[self.dim_offsets[start] : self.dim_offsets[stop]]
            size = sum(sizes)
            a = np.empty(size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[start] * self.element_size)
            self.data_file.readinto(a)
            offsets = list(accumulate(sizes))
            sents = np.split(a, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        if not self.data_file:
            self.read_data(self.path)
        size = self.sizes[idx]
        ptr = self.data_offsets[idx]
        if length is None:
            length = size - offset
        ptr += offset
        a = np.empty(length, dtype=self.dtype)
        self.data_file.seek(ptr * self.element_size)
        self.data_file.readinto(a)
        return a

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory

    @property
    def doc_idx(self):
        return self._doc_idx

    def get_doc_idx(self):
        return self._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._doc_idx = doc_idx_


class IndexedDatasetBuilder(object):
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.uint16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float32: 4,
        np.float64: 8,
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, "wb")
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]
        self.doc_idx = [0]

    def add_item(self, tensor):
        tensor = np.array(tensor, dtype=self.dtype)
        bytes = self.out_file.write(tensor)
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.shape:
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.shape))
        del bytes

    def end_document(self):
        self.doc_idx.append(len(self.sizes))

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        doc_offset = len(self.sizes)

        begin = self.data_offsets[-1]
        for data_offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + data_offset)
        self.sizes.extend(index.sizes)

        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)

        self.doc_idx.extend((doc_offset + index.doc_idx)[1:])

        with open(data_file_path(another_file), "rb") as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, "wb")
        index.write(b"TNTIDX\x00\x00")
        index.write(struct.pack("<Q", 1))
        index.write(struct.pack("<QQ", code(self.dtype), self.element_size))
        index.write(struct.pack("<QQ", len(self.data_offsets) - 1, len(self.sizes)))
        index.write(struct.pack("<Q", len(self.doc_idx)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_shorts(index, self.sizes)
        write_longs(index, self.doc_idx)
        index.close()

        print("Total sentences num: %d" % len(self.sizes))
        print("Total documents num: %d" % (len(self.doc_idx) - 1))
        print("Total tokens num: %d" % sum(self.sizes))
        print("Average tokens per sentence: %.2f" % (sum(self.sizes) / len(self.sizes)))
        print("Average tokens per document: %.2f" % (sum(self.sizes) / (len(self.doc_idx) - 1)))


def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(paddle.io.Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._buffer_mmap = np.memmap(path, mode="r", order="C")
            self._buffer = memoryview(self._buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(self._buffer, dtype=np.int32, count=self._len, offset=offset)
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(
                self._buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._buffer_mmap._mmap.close()
            del self._buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None
        self._loss_mask_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state, skip_warmup=True)

    def _do_init(self, path, skip_warmup):
        self._path = path

        if not self.exists(path):
            raise ValueError("Missing file, %s" % (path))

        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode="r", order="C")
        if os.path.exists(loss_mask_file_path(self._path)):
            self._loss_mask_buffer_mmap = np.memmap(loss_mask_file_path(self._path), mode="r", order="C")
            self._loss_mask_buffer = memoryview(self._loss_mask_buffer_mmap)
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        if hasattr(self, "_loss_mask_buffer_mmap"):
            self._loss_mask_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._loss_mask_buffer
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    def get(self, idx, offset=0, length=None):
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        mask_ptr = ptr // np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)
        mask_array = None
        if self._loss_mask_buffer is not None:
            mask_array = np.frombuffer(self._loss_mask_buffer, dtype=np.uint8, count=length, offset=mask_ptr)
        return np_array, mask_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))


class SFTMMapIndexedDataset(paddle.io.Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")
                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []
                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size
                    return pointers

                def write(self, sizes, doc_idx):

                    pointers = self._get_pointers(sizes)
                    self._file.write(struct.pack("<Q", len(sizes)))
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._buffer_mmap = np.memmap(path, mode="r", order="C")
            self._buffer = memoryview(self._buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(self._buffer, dtype=np.int32, count=self._len, offset=offset)
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(
                self._buffer, dtype=np.int64, count=self._len, offset=offset + self._sizes.nbytes
            )
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._buffer_mmap._mmap.close()
            del self._buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._doc_count - 1

    def __init__(self, path, dataclass, skip_warmup=False):
        super().__init__()
        self._dataclass = dataclass
        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state, skip_warmup=True)

    def _do_init(self, path, skip_warmup):
        self._path = path
        if not self.exists(path, self._dataclass):
            raise ValueError("Missing file, %s" % (path))

        self._index = self.Index(sft_index_file_path(self._path), skip_warmup)
        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            for data_file in sft_data_file_path(self._path, self._dataclass):
                _warmup_mmap_file(data_file)
        print_rank_0("    creating numpy buffer of mmap...")

        self._bin_buffer_mmap_dict = {}
        self._bin_buffer_dict = {}
        for data_file in sft_data_file_path(self._path, self._dataclass):
            self._bin_buffer_mmap_dict[data_file] = np.memmap(data_file, mode="r", order="C")
            self._bin_buffer_dict[data_file] = memoryview(self._bin_buffer_mmap_dict[data_file])
        print_rank_0("    creating memory view of numpy buffer...")

    def __del__(self):
        for key, value in self._bin_buffer_mmap_dict.items():
            value._mmap.close()
        for key, value in self._bin_buffer_dict.items():
            del value
        del self._index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        def get_index(idx):
            doc_idx = self._index.doc_idx
            start_sentence, end_sentence = doc_idx[idx], doc_idx[idx + 1]
            start_pointers, _ = self._index[start_sentence]
            length_list = self._index._sizes[start_sentence:end_sentence]

            dataclass_fields = fields(self._dataclass)
            dataclass_list = []
            sequence_offset = start_pointers
            scalar_offset = doc_idx[idx] * np.dtype(self._index.dtype).itemsize

            for length in length_list:
                field_data = {field.name: [] for field in dataclass_fields}
                for field in dataclass_fields:
                    bin_buffer = self._bin_buffer_dict[os.path.join(self._path, f"{field.name}.bin")]
                    if field.type != int:
                        data = np.frombuffer(bin_buffer, dtype=self._index.dtype, count=length, offset=sequence_offset)
                        field_data[field.name] = data.tolist()
                    else:
                        data = np.frombuffer(bin_buffer, dtype=self._index.dtype, count=1, offset=scalar_offset)
                        field_data[field.name] = int(data[0])

                dataclass_list.append(self._dataclass(**field_data))

                sequence_offset += length * np.dtype(self._index.dtype).itemsize
                scalar_offset += np.dtype(self._index.dtype).itemsize
            return dataclass_list

        if isinstance(idx, (int, np.integer)):
            return get_index(idx)
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            return [get_index(idx) for idx in range(start, stop)]

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path, dataclass):
        file_path_list = sft_data_file_path(path, dataclass)
        file_path_list.append(sft_index_file_path(path))
        for file_path in file_path_list:
            if not os.path.exists(file_path):
                return False
        return True


def make_builder(out_file, impl, save_dtype, loss_mask_file=None):
    if impl == "mmap":
        return MMapIndexedDatasetBuilder(out_file, dtype=save_dtype, loss_mask_file=loss_mask_file)
    else:
        return IndexedDatasetBuilder(out_file, dtype=save_dtype)


class SFTMMapIndexedDatasetBuilder(object):
    def __init__(self, output_file_dict, dtype):
        self._data_file_dict = {}
        for key, filename in output_file_dict.items():
            self._data_file_dict[key] = open(filename, "wb")
        self.output_file_dict = output_file_dict
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, sequence):
        add_sequence_len = False
        for key in self._data_file_dict.keys():
            tensor = np.array(getattr(sequence, key), dtype=self._dtype)
            if tensor.size > 1 and not add_sequence_len:
                self._sizes.append(tensor.size)
                add_sequence_len = True
            self._data_file_dict[key].write(tensor.tobytes(order="C"))

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def finalize(self, index_file):
        for key, filename in self._data_file_dict.items():
            filename.close()
        with SFTMMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype, loss_mask_file=None):
        self._data_file = open(out_file, "wb")
        self._loss_mask_file = None
        if loss_mask_file is not None:
            self._loss_mask_file = open(loss_mask_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def flush_loss_mask_item(self, loss_mask_lst):
        for loss_mask in loss_mask_lst:
            tensor = np.array(loss_mask, dtype=np.uint8)
            self._loss_mask_file.write(tensor.tobytes(order="C"))

    def add_item(self, tensor):
        tensor = np.array(tensor, dtype=self._dtype)
        self._data_file.write(tensor.tobytes(order="C"))
        self._sizes.append(tensor.size)

    def add_doc(self, tensor, sizes):
        np_array = np.array(tensor, dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.extend(sizes)
        self._doc_idx.append(len(self._sizes))

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend((offset + index.doc_idx)[1:])

        # Concatenate data
        with open(data_file_path(another_file), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
        print("Total sentences num: %d" % len(self._sizes))
        print("Total documents num: %d" % (len(self._doc_idx) - 1))
        print("Total tokens num: %d" % sum(self._sizes))
        print("Average tokens per sentence: %.2f" % (sum(self._sizes) / len(self._sizes)))
        print("Average tokens per document: %.2f" % (sum(self._sizes) / (len(self._doc_idx) - 1)))


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):

    print_rank_0(" > building dataset index ...")

    start_time = time.time()
    indexed_dataset = make_dataset(data_prefix, data_impl, skip_warmup)
    assert indexed_dataset.sizes.shape[0] == indexed_dataset.doc_idx[-1]
    print_rank_0(" > finished creating indexed dataset in {:4f} " "seconds".format(time.time() - start_time))

    print_rank_0(" > indexed dataset stats:")
    print_rank_0("    number of documents: {}".format(indexed_dataset.doc_idx.shape[0] - 1))
    print_rank_0("    number of sentences: {}".format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class CompatibleIndexedDataset(paddle.io.Dataset):
    def __init__(self, path):
        super().__init__()

        self._path = path

        # All document ids, extend as 1-D array.
        self._token_ids = np.load(path + "_ids.npy", mmap_mode="r", allow_pickle=True)
        process_data = np.load(path + "_idx.npz")
        self._sizes = process_data["lens"]
        self._pointers = np.empty(len(self._sizes) + 1, dtype=np.int64)
        self._pointers[0] = 0
        np.cumsum(self._sizes, out=self._pointers[1:])
        self._doc_idx = process_data["docs"]

    def __getstate__(self):
        return self._path

    def __len__(self):
        return len(self._sizes)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            size = self._sizes[idx]
            ptr = self._pointers[idx]
            np_array = self._token_ids[ptr : ptr + size]
            return np_array

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._pointers[start]
            sizes = self._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = self._token_ids[ptr : ptr + total_size]
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        size = self._sizes[idx]
        ptr = self._pointers[idx]

        if length is None:
            length = size - offset
        ptr += offset
        np_array = self._token_ids[ptr : ptr + length]
        return np_array, None

    @property
    def sizes(self):
        return self._sizes

    @property
    def doc_idx(self):
        return self._doc_idx

    def get_doc_idx(self):
        return self._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._doc_idx = doc_idx_

    @staticmethod
    def exists(path):
        return os.path.isfile(path + "_ids.npy") and os.path.isfile(path + "_idx.npz")
