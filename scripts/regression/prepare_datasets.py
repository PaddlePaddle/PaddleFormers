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

"""
Dataset preparation utilities for model training regression tests.

This module handles downloading and preparing datasets required for
various training types (VL, text, etc.).
"""

import argparse
import os
import subprocess

VL_DATA_DIR = "tests/fixtures/dummy/sft-vl"

VL_DATASETS = {
    "DoclingMatix": {
        "url": "https://paddleformers.bj.bcebos.com/datasets/DoclingMatix.tar.gz",
        "archive_name": "DoclingMatix.tar.gz",
        "extract_cmd": "tar zxf {archive_path} -C {target_dir}/",
        "check_path": "DoclingMatix",
        "is_dir": True,
    },
    "thinksafe_vl_data": {
        "url": "https://paddleformers.bj.bcebos.com/datasets/thinksafe_vl_data.tar",
        "archive_name": "thinksafe_vl_data.tar",
        "extract_cmd": "tar xf {archive_path} -C {target_dir}/",
        "check_path": "thinksafe_vl_data",
        "is_dir": True,
    },
    "ocr_vl_sft-test_Bengali": {
        "url": "https://paddleformers.bj.bcebos.com/datasets/ocr-vl/ocr_vl_sft-test_Bengali.jsonl",
        "extract_cmd": None,
        "check_path": "ocr_vl_sft-test_Bengali.jsonl",
        "is_dir": False,
    },
    "ocr_vl_sft-train_Bengali": {
        "url": "https://paddleformers.bj.bcebos.com/datasets/ocr-vl/ocr_vl_sft-train_Bengali.jsonl",
        "extract_cmd": None,
        "check_path": "ocr_vl_sft-train_Bengali.jsonl",
        "is_dir": False,
    },
}

DPO_VL_IMAGES_URL = "https://paddle-qa.bj.bcebos.com/paddleformers/images.tar"
DPO_VL_DATA_DIR = "tests/fixtures/dummy/dpo-vl"


def download_file(url: str, target_dir: str, filename: str = None) -> str:
    """Download a file from URL to target directory.

    Args:
        url: URL to download from.
        target_dir: Directory to save the file.
        filename: Optional filename, if None will use URL's filename.

    Returns:
        Path to the downloaded file.
    """
    if filename is None:
        filename = os.path.basename(url)

    target_path = os.path.join(target_dir, filename)

    print(f"[INFO] Downloading {url}...")
    subprocess.run(f"wget -q {url} -O {target_path}", shell=True, check=True)
    return target_path


def download_vl_datasets(force: bool = False) -> None:
    """Download VL (Vision-Language) training datasets.

    Downloads datasets required for sft-vl and dpo-vl training types.

    Args:
        force: If True, re-download even if files exist.
    """
    os.makedirs(VL_DATA_DIR, exist_ok=True)

    for name, config in VL_DATASETS.items():
        check_path = os.path.join(VL_DATA_DIR, config["check_path"])

        # Check if already exists
        if not force:
            if config["is_dir"] and os.path.isdir(check_path):
                print(f"[INFO] {name} already exists, skipping...")
                continue
            if not config["is_dir"] and os.path.isfile(check_path):
                print(f"[INFO] {name} already exists, skipping...")
                continue

        # Download
        print(f"[INFO] Downloading {name}...")

        if config["extract_cmd"]:
            # Download archive to target directory and extract with full path
            archive_name = config.get("archive_name", os.path.basename(config["url"]))
            archive_path = os.path.join(VL_DATA_DIR, archive_name)
            subprocess.run(f"wget -q {config['url']} -O {archive_path}", shell=True, check=True)
            subprocess.run(
                config["extract_cmd"].format(archive_path=archive_path, target_dir=VL_DATA_DIR), shell=True, check=True
            )
            # Cleanup archive
            if os.path.exists(archive_path):
                os.remove(archive_path)
        else:
            # Direct download to target directory
            subprocess.run(f"wget -q {config['url']} -P {VL_DATA_DIR}/", shell=True, check=True)

        print(f"[INFO] {name} downloaded successfully.")


def prepare_datasets_for_train_type(train_type: str, force: bool = False) -> None:
    """Prepare datasets based on training type.

    Args:
        train_type: Training type (e.g., 'sft', 'sft-vl', 'dpo-vl').
        force: If True, re-download even if files exist.
    """
    if train_type.endswith("-vl"):
        print(f"[INFO] Preparing VL datasets for {train_type}...")
        download_vl_datasets(force=force)
        # Also download DPO VL images for dpo-vl training
        if train_type == "dpo-vl":
            download_dpo_vl_images(force=force)
    else:
        print(f"[INFO] No additional datasets needed for {train_type}")


def download_dpo_vl_images(force: bool = False) -> None:
    """Download DPO-VL images to tests/fixtures/dummy/dpo-vl"""
    if not force and os.path.isdir(f"{DPO_VL_DATA_DIR}/images"):
        print("[INFO] DPO VL images exists, skipping...")
        return

    print(f"[INFO] Downloading {DPO_VL_IMAGES_URL}...")
    subprocess.run(
        f"wget -q {DPO_VL_IMAGES_URL} -O /tmp/images.tar && "
        f"mkdir -p {DPO_VL_DATA_DIR} && "
        f"tar xf /tmp/images.tar -C {DPO_VL_DATA_DIR} && "
        f"rm /tmp/images.tar",
        shell=True,
        check=True,
    )


def prepare_all_datasets(force: bool = False) -> None:
    """Download all datasets before running tests.

    Args:
        force: If True, re-download even if files exist.
    """
    print("[INFO] Preparing all datasets for regression tests...")
    download_vl_datasets(force=force)
    # Also download DPO VL images as they may be needed for dpo-vl tests
    download_dpo_vl_images(force=force)
    print("[INFO] All datasets prepared successfully.")


def main():
    """Main entry point for dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare datasets for model training regression tests")
    parser.add_argument(
        "--train-type",
        type=str,
        default=None,
        help="Training type (e.g., 'sft-vl', 'dpo-vl'). If not specified, downloads all.",
    )
    parser.add_argument("--vl", action="store_true", help="Download VL datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")

    args = parser.parse_args()

    if args.train_type:
        prepare_datasets_for_train_type(args.train_type, force=args.force)
    elif args.vl or args.all:
        download_vl_datasets(force=args.force)
    else:
        print("Usage:")
        print("  python prepare_datasets.py --vl           # Download VL datasets")
        print("  python prepare_datasets.py --all          # Download all datasets")
        print("  python prepare_datasets.py --train-type sft-vl  # Download datasets for specific train type")
        print("  python prepare_datasets.py --force        # Force re-download")


if __name__ == "__main__":
    main()
