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
    # Parse input_dirs from comma-separated string
    input_dirs = [d.strip() for d in args.input_dirs.split(",")]

    # Build actual paths with split subdirectory
    actual_dirs = []
    for input_dir in input_dirs:
        actual_path = os.path.join(input_dir, args.split)
        if not os.path.isdir(actual_path):
            raise ValueError(f"Directory not found: {actual_path}")
        actual_dirs.append(actual_path)

    print_datetime(f"Merging {len(actual_dirs)} directories in order:")
    for i, d in enumerate(actual_dirs):
        print(f"  [{i}] {d}")

    merge_sft_datasets(actual_dirs, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge SFT indexed datasets. Each input directory should contain train/eval subdirectories "
        "with index.idx and .bin files."
    )

    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input_dirs",
        type=str,
        required=True,
        help="Comma-separated list of input directories to merge in order, e.g., 'A,B'. "
        "Each directory should contain train/eval subdirectories.",
    )
    group.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "eval"],
        help="Which split to merge: 'train' or 'eval'. Will look for <input_dir>/train or <input_dir>/eval.",
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output",
        type=str,
        default="merge",
        help="Output directory path. Default: 'merge'. The split name will be appended, e.g., 'merge/train'.",
    )

    args = parser.parse_args()

    # Append split name to output directory
    args.output = os.path.join(args.output, args.split)

    main(args)
