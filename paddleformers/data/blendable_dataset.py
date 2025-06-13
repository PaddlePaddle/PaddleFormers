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

import hashlib
import importlib.metadata
import os
import time

import numpy as np
import paddle

local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))


def print_rank_0(*args, **kwargs):
    if paddle.distributed.get_rank() == 0:
        print(*args, **kwargs)


class BlendableDataset(paddle.io.Dataset):
    def __init__(self, datasets, weights, size, share_folder, *, data_cache_path=None):

        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = size

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # Build indices.
        def _build_indices():
            start_time = time.time()

            fast_dataindex_version = importlib.metadata.version("fast_dataindex")
            if fast_dataindex_version > "0.1.1":
                assert (
                    num_datasets < 32767
                ), f"Detect num_datasets({num_datasets})>=32767. Currently, num_datasets should be less than 32767."
                dataset_index = np.zeros(self.size, dtype=np.int16)
            else:
                assert (
                    num_datasets < 255
                ), f"Detect num_datasets:({num_datasets})>=255. When 'fast_dataindex<=0.1.1', num_datasets should be less than 255. To support num_datasets greater than 255, please upgrade `fast_dataindex>=0.1.2`."
                dataset_index = np.zeros(self.size, dtype=np.uint8)
            dataset_sample_index = np.zeros(self.size, dtype=np.int64)

            from fast_dataindex import helpers

            helpers.build_blending_indices(
                dataset_index,
                dataset_sample_index,
                weights,
                num_datasets,
                self.size,
                local_rank == 0,
                #    paddle.distributed.get_rank() == 0,
            )
            print_rank_0(
                "> elapsed time for building blendable dataset indices: "
                "{:.2f} (sec)".format(time.time() - start_time)
            )
            return dataset_index, dataset_sample_index

        desc = "Blendable dataset\n\n"
        desc += "Datasets:\n"
        for dataset in datasets:
            desc += dataset.desc + "\n\n"
        desc += f"Weights: {weights}\n"
        desc += f"Size: {size}\n"
        self.desc = desc

        if data_cache_path:
            desc_hash = hashlib.md5(desc.encode("utf-8")).hexdigest()
            desc_path = os.path.join(data_cache_path, desc_hash + ".dsc")
            index_path = os.path.join(data_cache_path, desc_hash + "_index.npy")
            sample_index_path = os.path.join(data_cache_path, desc_hash + "_sample_index.npy")
            cache_hit = os.path.isfile(index_path) and os.path.isfile(sample_index_path)
            # cache_success = True
            # if paddle.distributed.get_rank() == 0 and not cache_hit:
            check_rank_flag = not cache_hit and local_rank == 0
            if share_folder:
                check_rank_flag = not cache_hit and paddle.distributed.get_rank() == 0

            print(
                f"searching for blendable dataset, cache_hit={cache_hit}, share_folder {share_folder}, check_rank_flag {check_rank_flag}",
                flush=True,
            )
            if check_rank_flag:
                print(
                    " > WARNING: could not find index map files for blendable"
                    " dataset, building indices on rank 0 ...",
                    flush=True,
                )
                dataset_index, dataset_sample_index = _build_indices()
                try:
                    os.makedirs(os.path.dirname(index_path), exist_ok=True)
                    with open(desc_path, "wt") as fd:
                        fd.write(desc)
                        np.save(index_path, dataset_index, allow_pickle=True)
                        np.save(sample_index_path, dataset_sample_index, allow_pickle=True)
                except OSError:
                    print(f"There was an error trying to create the data cache directory ({data_cache_path})")
                    print("or a file in it. This is set with the --data-cache-path argument. Please")
                    print("ensure you have write access to this directory or specify one that you do have")
                    print("write access to.")
                    # cache_success = False

            # hcg = paddle.distributed.fleet.get_hybrid_communicate_group()

            # counts = paddle.to_tensor([cache_success], dtype="int64")
            # paddle.distributed.all_reduce(counts, group=hcg.get_data_parallel_group())
            # paddle.distributed.all_reduce(counts, group=hcg.get_pipeline_model_parallel_group())
            # if counts[0].item() != (
            #     paddle.distributed.get_world_size()
            #     // paddle.distributed.get_world_size(group=hcg.get_tensor_model_parallel_group())
            # ):
            #     print_rank_0("Data index creation unsuccessful, exiting.")
            #     exit()

            else:
                while True:
                    if (not os.path.isfile(index_path)) or (not os.path.isfile(sample_index_path)):
                        print("building indices on rank 0 ...", flush=True)
                        time.sleep(3)
                    else:
                        try:
                            np.load(index_path, allow_pickle=True, mmap_mode="r")
                            print("build success", flush=True)
                            break
                        except Exception:
                            print("%s file is still writing or damaged, please wait for a moment." % index_path)
                            time.sleep(3)

            # paddle.distributed.barrier()
            # Load on all ranks.
            print_rank_0(f"> loading blendable dataset index: {index_path}")
            self.dataset_index = np.load(index_path, allow_pickle=True, mmap_mode="r")
            assert self.dataset_index.size == self.size

            print_rank_0(f"> loading blendable dataset sample index: {sample_index_path}")
            self.dataset_sample_index = np.load(sample_index_path, allow_pickle=True, mmap_mode="r")
            assert self.dataset_sample_index.size == self.size
        else:
            print_rank_0(
                "building indices for the blendable dataset, Since --data_cache is not specified, the index file will not be stored.",
                flush=True,
            )
            self.dataset_index, self.dataset_sample_index = _build_indices()

        # Check size
        _ = self.__getitem__(self.size - 1)
        try:
            _ = self.__getitem__(self.size)
            raise RuntimeError("BlendedDataset size is improperly bounded")
        except IndexError:
            pass
        print_rank_0("> size of blendable dataset: " "{} samples".format(self.size))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return {
            "dataset_idx": dataset_idx,
            **self.datasets[dataset_idx][sample_idx],
        }
