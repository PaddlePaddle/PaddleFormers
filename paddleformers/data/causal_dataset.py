# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT style dataset."""
import hashlib
import math
import os
import time

import numpy as np
import paddle

from .blendable_dataset import BlendableDataset
from .indexed_dataset import make_dataset as make_indexed_dataset

local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))


# class FakeHCG:
#     def get_data_parallel_group(self):
#         return None

#     def get_pipe_parallel_group(self):
#         return None

#     def get_model_parallel_group(self):
#         return None


def check_data_split(splits_string, do_train, do_eval, do_predict):
    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    data_flag = True
    assert splits_sum > 0.0, "sum of splits should larger than 0.0!"
    if (do_train and splits[0] == 0) or (do_eval and splits[1] == 0) or (do_predict and splits[2] == 0):
        data_flag = False
    if not data_flag:
        raise ValueError("If do_train/do_eval/do_predict is True, the corresponding dataset split should not be 0!")


def get_train_valid_test_split_(splits_string, size):
    """Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def get_datasets_weights_and_num_samples(data_prefix, train_val_test_num_samples):

    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [0] * num_datasets
    prefixes = [0] * num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2 * i])
        prefixes[i] = (data_prefix[2 * i + 1]).strip()
    # Normalize weights
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]

    # Add 0.5% (the 1.005 factor) so in case the blending dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    # (NOTE, yujun06): This is a workaround to avoid issues with indexing in the blending dataset. Therefore, we need to add 20 samples to each dataset.
    datasets_train_valid_test_num_samples = []
    for weight in weights:
        datasets_train_valid_test_num_samples.append(
            [int(math.ceil(val * weight * 1.005)) + 20 for val in train_val_test_num_samples]
        )

    return prefixes, weights, datasets_train_valid_test_num_samples


def print_rank_0(*args, **kwargs):
    if paddle.distributed.get_rank() == 0:
        print(*args, **kwargs)


def build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    train_val_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    train_data_prefix=None,
    valid_data_prefix=None,
    test_data_prefix=None,
    return_doc_ids=False,
    share_folder=False,
    *,
    data_cache_path=None,
    need_data=True,
):
    """Build train, valid, and test datasets."""

    # Single dataset.
    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(
            data_prefix[0],
            data_impl,
            splits_string,
            train_val_test_num_samples,
            seq_length,
            seed,
            skip_warmup,
            share_folder=share_folder,
            data_cache_path=data_cache_path,
            need_data=need_data,
        )

    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix, train_val_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output
    # NOTE: megatron/gpt_dataset.py has been updated. When creating BlendableDataset, we will use the raw train_val_test_num_samples instead of the expanded ones.
    # Please refer to https://github.com/NVIDIA/NeMo/blob/72f630d087d45655b1a069dc72debf01dfdbdb2d/nemo/collections/nlp/data/language_modeling/megatron/gpt_dataset.py#L74-L80 for more information
    train_num_samples, valid_num_samples, test_num_samples = train_val_test_num_samples

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            prefixes[i],
            data_impl,
            splits_string,
            datasets_train_valid_test_num_samples[i],
            seq_length,
            seed,
            skip_warmup,
            return_doc_ids,
            share_folder=share_folder,
            data_cache_path=data_cache_path,
            need_data=need_data,
        )
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(
            train_datasets, weights, train_num_samples, share_folder, data_cache_path=data_cache_path
        )
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(
            valid_datasets, weights, valid_num_samples, share_folder, data_cache_path=data_cache_path
        )
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(
            test_datasets,
            weights,
            test_num_samples,
            share_folder,
            data_cache_path=data_cache_path,
        )

    return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    train_val_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    return_doc_ids=False,
    share_folder=False,
    *,
    data_cache_path=None,
    need_data=True,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    if need_data:
        indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)

        total_num_of_documents = indexed_dataset.sizes.shape[0]
        splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

        # Print stats about the splits.
        print_rank_0(" > dataset split:")

        def print_split_stats(name, index):
            print_rank_0("    {}:".format(name))
            print_rank_0(
                "     document indices in [{}, {}) total of {} "
                "documents".format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
            )

        print_split_stats("train", 0)
        print_split_stats("validation", 1)
        print_split_stats("test", 2)

    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.barrier()

    def build_dataset(index, name):
        documents = np.arange(splits[index], splits[index + 1], 1, np.int32) if need_data else None
        dataset = GPTDataset(
            name,
            data_prefix,
            documents,
            indexed_dataset if need_data else None,
            splits_string,
            train_val_test_num_samples[index],
            seq_length,
            seed,
            return_doc_ids,
            share_folder,
            data_cache_path=data_cache_path,
            need_data=need_data,
        )
        if need_data:
            return dataset if splits[index + 1] > splits[index] else None
        else:
            return None

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(" > building dataset index ...")

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
    print_rank_0(" > finished creating indexed dataset in {:4f} " "seconds".format(time.time() - start_time))
    print_rank_0("    number of documents: {}".format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class GPTDataset(paddle.io.Dataset):
    def __init__(
        self,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        splits_string,
        num_samples,
        seq_length,
        seed,
        return_doc_ids=False,
        share_folder=False,
        *,
        data_cache_path=None,
        need_data=True,
    ):

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.return_doc_ids = return_doc_ids

        # Build index mappings.
        if need_data and len(documents) > 0:
            assert np.min(documents) >= 0
            assert np.max(documents) < indexed_dataset.sizes.shape[0]

            (
                doc_idx_filename,
                sample_idx_filename,
                shuffle_idx_filename,
                self.desc,
                self.desc_hash,
                num_epochs,
            ) = _build_index_mappings(
                self.name,
                data_prefix,
                documents,
                self.indexed_dataset.sizes,
                splits_string,
                num_samples,
                seq_length,
                seed,
                share_folder,
                data_cache_path=data_cache_path,
            )

        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()

        # Load mappings.
        if need_data and len(documents) > 0:
            start_time = time.time()
            print_rank_0(f" > loading doc-idx mapping from {doc_idx_filename}")
            self.doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")

            print_rank_0(f" > loading sample-idx mapping from {sample_idx_filename}")
            self.sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")

            print_rank_0(f" > loading shuffle-idx mapping from {shuffle_idx_filename}")
            self.shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")

            print_rank_0("    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time))
            print_rank_0("    total number of samples: {}".format(self.sample_idx.shape[0]))
            print_rank_0("    total number of epochs: {}".format(num_epochs))

        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.barrier()

    def __len__(self):
        # -1 is due to data structure used to retrieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1

    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        doc_ids = []
        if doc_index_f == doc_index_l:
            doc_ids.append(self.doc_idx[doc_index_f])

            sample, mask = self.indexed_dataset.get(
                self.doc_idx[doc_index_f], offset=offset_f, length=offset_l - offset_f + 1
            )
        else:
            # Otherwise, get the rest of the initial document.
            doc_ids.append(self.doc_idx[doc_index_f])
            sample, mask = self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)
            append_mask = True
            if mask is None:
                append_mask = False

            sample_list = [sample]
            mask_list = []
            mask_list = [mask]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                doc_ids.append(self.doc_idx[i])
                sample, mask = self.indexed_dataset.get(self.doc_idx[i])
                sample_list.append(sample)
                if append_mask:
                    mask_list.append(mask)

            # And finally add the relevant portion of last document.
            doc_ids.append(self.doc_idx[doc_index_l])
            sample, mask = self.indexed_dataset.get(self.doc_idx[doc_index_l], length=offset_l + 1)
            sample_list.append(sample)
            if append_mask:
                mask_list.append(mask)
            sample = np.concatenate(sample_list)
            if append_mask:
                mask = np.concatenate(mask_list)
        # print(sample)
        if self.return_doc_ids:  # for retro preprocessing
            if mask is None:
                return {"text": np.array(sample, dtype=np.int64), "doc_ids": np.array(doc_ids, dtype=np.int64)}
            else:
                return {
                    "text": np.array(sample, dtype=np.int64),
                    "doc_ids": np.array(doc_ids, dtype=np.int64),
                    "mask": np.array(mask, dtype=np.int64),
                }
        else:
            if mask is None:
                return {"text": np.array(sample, dtype=np.int64)}
            else:
                return {"text": np.array(sample, dtype=np.int64), "mask": np.array(mask, dtype=np.int64)}


def _build_index_mappings(
    name, data_prefix, documents, sizes, splits_string, num_samples, seq_length, seed, share_folder, *, data_cache_path
):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """

    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)

    # rng state
    np_rng = np.random.RandomState(seed=seed)
    # Filename of the index mappings.
    desc = "GPT Dataset\n\n"
    desc += f"Data prefix {data_prefix}\n"
    desc += f"Dataset name {name}\n"
    desc += f"Number of samples {num_samples}\n"
    desc += f"Sequence length {seq_length}\n"
    desc += f"Random seed {seed}\n"
    desc += f"Split {splits_string}\n"
    desc_hash = hashlib.md5(desc.encode("utf-8")).hexdigest()
    desc_filename = desc_hash + ".dsc"
    doc_idx_filename = desc_hash + "_doc_idx.npy"
    sample_idx_filename = desc_hash + "_sample_idx.npy"
    shuffle_idx_filename = desc_hash + "_shuffle_idx.npy"

    # Look for cache in main data dir first to avoid unnecessary
    # duplication, then look in data-cache-path if specified,
    # If nothing is found, use the last path looked in
    build_indices = True
    prefixes = [os.path.join(os.path.dirname(data_prefix), "index-cache")]
    if data_cache_path is not None:
        prefixes.append(data_cache_path)
    for prefix in prefixes:
        idx_path = {
            "desc": os.path.join(prefix, desc_filename),
            "doc": os.path.join(prefix, doc_idx_filename),
            "sample": os.path.join(prefix, sample_idx_filename),
            "shuffle": os.path.join(prefix, shuffle_idx_filename),
        }
        for f in idx_path.values():
            if not os.path.isfile(f):
                break
            else:
                # Found our files!
                build_indices = False
                break
    data_cache_dir = os.path.dirname(idx_path["desc"])
    # data_cache_success = True
    # Build the indexed mapping if not exist.
    check_rank_flag = build_indices and local_rank == 0
    if share_folder:
        check_rank_flag = build_indices and paddle.distributed.get_rank() == 0

    # if build_indices and paddle.distributed.get_rank() == 0:

    print(
        f"searching for causal dataset, build_indices={build_indices}, share_folder {share_folder}, check_rank_flag {check_rank_flag}",
        flush=True,
    )
    if check_rank_flag:
        print_rank_0(" > WARNING: could not find index map files, building " "the indices on rank 0 ...")

        # For the last epoch, decide whether include the entire epoch
        # in the global shuffle or not.

        # If we need only one epoch, then separating last epoch  does
        # not mean anything.
        if num_epochs == 1:
            separate_last_epoch = False
            print(" > only one epoch required, setting " "separate_last_epoch to False", flush=True)

        else:
            # Get the number of samples for the last epoch
            num_samples_from_epochs_minus_one = ((num_epochs - 1) * tokens_per_epoch - 1) // seq_length
            last_epoch_num_samples = num_samples - num_samples_from_epochs_minus_one
            assert last_epoch_num_samples >= 0, "last epoch number of samples should be non-negative."
            num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
            assert last_epoch_num_samples <= (
                num_samples_per_epoch + 1
            ), "last epoch number of samples exceeded max value."
            # If we have less than 80% of the samples for the last epoch,
            # separate out the epoch and treat it differently.
            # Note: the 80% number is just based on common sense and can
            # be adjusted if needed.
            separate_last_epoch = last_epoch_num_samples < int(0.80 * num_samples_per_epoch)
            if separate_last_epoch:
                string = (
                    " > last epoch number of samples ({}) is smaller "
                    "than 80% of number of samples per epoch ({}), "
                    "setting separate_last_epoch to True"
                )
            else:
                string = (
                    " > last epoch number of samples ({}) is larger "
                    "than 80% of number of samples per epoch ({}), "
                    "setting separate_last_epoch to False"
                )
            print(string.format(last_epoch_num_samples, num_samples_per_epoch), flush=True)

        try:
            os.makedirs(data_cache_dir, exist_ok=True)

            # description
            with open(idx_path["desc"], "wt") as fd:
                fd.write(desc)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch)
            np.save(idx_path["doc"], doc_idx, allow_pickle=True)
            print_rank_0(
                " > elapsed time to build and save doc-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            # from megatron.data import helpers
            from fast_dataindex import helpers

            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch)
            np.save(idx_path["sample"], sample_idx, allow_pickle=True)
            print_rank_0(
                " > elapsed time to build and save sample-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retrieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_, sample_idx.shape[0] - 1, np_rng)
            np.save(idx_path["shuffle"], shuffle_idx, allow_pickle=True)
            print_rank_0(
                " > elapsed time to build and save shuffle-idx mapping"
                " (seconds): {:4f}".format(time.time() - start_time)
            )
        except OSError:
            print(f"There was an error trying to create the data cache directory ({data_cache_dir})")
            print('or a file in it. This defaults to a directory "index-cache" within the directory')
            print("the data files are in and can be set with the --data-cache-path argument. Please")
            print("ensure you have write access to this directory or specify one that you do have")
            print("write access to.")
            # data_cache_success = False
    else:
        while True:
            if (
                (not os.path.isfile(idx_path["doc"]))
                or (not os.path.isfile(idx_path["sample"]))
                or (not os.path.isfile(idx_path["shuffle"]))
            ):
                print("building indices on rank 0 ...", flush=True)
                time.sleep(3)
            else:
                try:
                    np.load(idx_path["shuffle"], allow_pickle=True, mmap_mode="r")
                    print("build success", flush=True)
                    break
                except Exception:
                    print("%s file is still writing or damaged, please wait for a moment." % idx_path["shuffle"])
                    time.sleep(3)
    # try:
    #     hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
    # except:
    #     hcg = FakeHCG()

    # counts = paddle.to_tensor([data_cache_success], dtype="int64")
    # paddle.distributed.all_reduce(counts, group=hcg.get_data_parallel_group())
    # paddle.distributed.all_reduce(counts, group=hcg.get_pipe_parallel_group())
    # if counts[0].item() != (
    #     paddle.distributed.get_world_size() // paddle.distributed.get_world_size(group=hcg.get_model_parallel_group())
    # ):
    #     print_rank_0("Data index creation unsuccessful, exiting.")
    #     exit()
    # paddle.distributed.barrier()

    return idx_path["doc"], idx_path["sample"], idx_path["shuffle"], desc, desc_hash, num_epochs


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence length, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-documents.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs - 1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Beginning offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += remaining_seq_length + doc_length - 1
                remaining_seq_length = 0
            else:
                # Otherwise, start from the beginning of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print(
        " > building shuffle index with split [0, {}) and [{}, {}) "
        "...".format(num_samples, num_samples, total_size),
        flush=True,
    )

    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))
