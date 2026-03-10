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

import argparse
import os
import shutil
from datetime import datetime

import numpy as np

from paddleformers.data import indexed_dataset


def print_datetime(string):
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[" + string + "] datetime: {} ".format(time_str))


def merge_sft_datasets(input_dirs, output_dir):
    """
    merge SFTMMapIndexedDataset bin (index.idx + several .bin files)
    """
    os.makedirs(output_dir, exist_ok=True)

    # get all common .bin file names
    bin_files_set = None

    print_datetime("Validating input directories...")

    for input_dir in input_dirs:

        index_path = os.path.join(input_dir, "index.idx")
        if not os.path.exists(index_path):
            raise ValueError(f"index.idx not found in {input_dir}")

        current_bin_files = set()
        for filename in os.listdir(input_dir):
            if filename.endswith(".bin"):
                current_bin_files.add(filename)

        if not current_bin_files:
            raise ValueError(f"No .bin files found in {input_dir}")

        if bin_files_set is None:
            bin_files_set = current_bin_files
        else:
            bin_files_set = bin_files_set.intersection(current_bin_files)

    if not bin_files_set:
        raise ValueError("No common .bin files found across input directories")

    bin_files = sorted(bin_files_set)
    print_datetime(f"Found {len(bin_files)} common bin files: {bin_files}")

    print_datetime("Reading index files...")
    all_indices = []
    dtype = None

    for input_dir in input_dirs:
        index_path = os.path.join(input_dir, "index.idx")
        index = indexed_dataset.SFTMMapIndexedDataset.Index(index_path)

        if dtype is None:
            dtype = index.dtype
        else:
            assert index.dtype == dtype, f"Dtype mismatch in {index_path}"

        all_indices.append(index)

    print_datetime("Merging index data...")
    merged_sizes = []
    merged_doc_idx = [0]

    for idx in all_indices:
        merged_sizes.extend(idx.sizes.tolist())
        offset = merged_doc_idx[-1]
        merged_doc_idx.extend((offset + idx.doc_idx)[1:].tolist())

    merged_sizes = np.array(merged_sizes, dtype=np.int32)
    merged_doc_idx = np.array(merged_doc_idx, dtype=np.int64)

    print_datetime(f"Total samples: {len(merged_sizes)}, Total docs: {len(merged_doc_idx) - 1}")

    for bin_file in bin_files:
        print_datetime(f"Merging {bin_file}...")
        output_bin_path = os.path.join(output_dir, bin_file)

        with open(output_bin_path, "wb") as out_f:
            for input_dir in input_dirs:
                input_bin_path = os.path.join(input_dir, bin_file)
                with open(input_bin_path, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f)

        print_datetime(f"Finished merging {bin_file}")

    print_datetime("Writing merged index.idx...")
    output_index_path = os.path.join(output_dir, "index.idx")

    with indexed_dataset.SFTMMapIndexedDataset.Index.writer(output_index_path, dtype) as writer:
        writer.write(merged_sizes.tolist(), merged_doc_idx.tolist())

    print_datetime("Merge completed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(merged_sizes)}")
    print(f"Total documents: {len(merged_doc_idx) - 1}")
    print(f"Total tokens: {merged_sizes.sum()}")


def main(args):

    if os.path.isdir(args.input):
        subdirs = []
        for name in sorted(os.listdir(args.input)):
            path = os.path.join(args.input, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "index.idx")):
                subdirs.append(path)

        if subdirs:
            print_datetime(f"Detected SFT format with {len(subdirs)} subdirectories")
            merge_sft_datasets(subdirs, args.output)
            return

        if os.path.exists(os.path.join(args.input, "index.idx")):
            print(
                "Error: Single SFT directory detected. Use the parent directory containing multiple SFT subdirectories."
            )
            return

    prefixes = set()
    for basename in os.listdir(args.input):
        prefix, ext = os.path.splitext(basename)

        if prefix in prefixes:
            continue

        if not os.path.isfile(os.path.join(args.input, basename)):
            continue

        ext_pair = ".bin" if ext == ".idx" else ".idx"
        assert os.path.isfile(
            os.path.join(args.input, prefix) + ext_pair
        ), f"ERROR: {ext_pair} file not provided for {os.path.join(args.input, prefix)}"

        prefixes.add(prefix)

    builder = None

    for prefix in sorted(prefixes):
        print_datetime(f"start processing file {prefix}")
        if builder is None:
            dataset = indexed_dataset.make_dataset(os.path.join(args.input, prefix), args.data_impl)

            if isinstance(dataset, indexed_dataset.MMapIndexedDataset):
                builder = indexed_dataset.MMapIndexedDatasetBuilder(
                    args.output_prefix + ".bin", dtype=dataset._index.dtype
                )
            else:
                builder = indexed_dataset.IndexedDatasetBuilder(args.output_prefix + ".bin", dtype=dataset.dtype)

            del dataset
        print_datetime(f"start merge file {prefix}")
        builder.merge_file_(os.path.join(args.input, prefix))
        print_datetime(f"end merge file {prefix}")

    print_datetime("start finalize")
    builder.finalize(args.output_prefix + ".idx")
    print_datetime("end finalize")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge indexed datasets. Supports two formats:\n"
        "1. Traditional: prefix.idx + prefix.bin pairs in one directory\n"
        "2. SFT format: Multiple subdirectories, each containing index.idx and .bin files"
    )

    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to directory. For traditional format: directory containing .idx/.bin pairs. "
        "For SFT format: parent directory containing multiple SFT subdirectories",
    )
    group.add_argument("--data_impl", type=str, default="mmap", help="data_impl for traditional format (mmap or lazy)")

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output", type=str, default=None, help="Output directory path for SFT format")
    group.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file prefix for traditional format (without .idx/.bin suffix)",
    )

    args = parser.parse_args()

    assert os.path.isdir(args.input), f"ERROR: {args.input} is not a directory or does not exist"

    subdirs = []
    for name in sorted(os.listdir(args.input)):
        path = os.path.join(args.input, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "index.idx")):
            subdirs.append(path)

    if subdirs:
        is_sft_format = True
    elif os.path.exists(os.path.join(args.input, "index.idx")):
        print("Error: Single SFT directory detected. Expected a parent directory with multiple SFT subdirectories.")
        print("  Or use --output-prefix for traditional format merging.")
        exit(1)

    if is_sft_format:
        if args.output is None:
            parser.error("--output is required for SFT format merging")
    else:
        if args.output_prefix is None:
            parser.error("--output-prefix is required for traditional format merging")
        assert os.path.isdir(
            os.path.dirname(args.output_prefix)
        ), f"ERROR: {os.path.dirname(args.output_prefix)} is not a directory or does not exist"

    main(args)
