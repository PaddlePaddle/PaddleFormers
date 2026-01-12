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
import glob
import io
import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from paddleformers.utils.log import logger

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def parse_args():
    parser = argparse.ArgumentParser(description="COCO Dataset Preparation for Qwen2.5-VL Grounding")
    parser.add_argument("--dataset_repo", type=str, default="detection-datasets/coco", help="dataset repository ID")
    parser.add_argument(
        "--output_dir", type=str, default="./data/coco_grounding", help="Output directory for processed data"
    )
    parser.add_argument("--total_samples", type=int, default=15000, help="Total number of samples to process")
    parser.add_argument("--val_ratio", type=float, default=0.01, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# Target resolution following Qwen2.5-VL dynamic resolution strategy
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
) -> Tuple[int, int]:
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# Mapping bbox in absolute pixel values (Only for Qwen2.5-VL)
def convert_to_qwen25vl_format(bbox: List[float], orig_height: int, orig_width: int) -> List[int]:
    new_height, new_width = smart_resize(orig_height, orig_width)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height

    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)

    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))

    return [x1_new, y1_new, x2_new, y2_new]


def get_data_path(dataset_repo: str) -> str:
    download_hub = os.environ.get("DOWNLOAD_SOURCE", "huggingface")

    if download_hub == "huggingface":
        logger.info(f"Checking dataset {dataset_repo} (HuggingFace)...")
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(repo_id=dataset_repo, repo_type="dataset", allow_patterns="data/*.parquet")

    elif download_hub == "modelscope":
        from modelscope.msdatasets import MsDataset

        dataset_repo_ms = dataset_repo.replace("detection-datasets", "AI-ModelScope")
        logger.info(f"Checking dataset {dataset_repo_ms} (ModelScope)...")
        local_dir = MsDataset.load(dataset_repo_ms, subset_name="detection-datasets--coco", use_streaming=True)

    else:
        raise ValueError(f"Invalid download hub: {download_hub}")

    data_path = os.path.join(local_dir, "data")
    if not os.path.exists(data_path) and os.path.exists(local_dir):
        return local_dir
    return data_path


def scan_dataset_metadata(files: List[str], desc: str) -> List[dict]:
    candidates = []
    for f in tqdm(files, desc=desc):
        try:
            df = pq.read_table(f, columns=["objects"]).to_pandas()
            for idx, row in df.iterrows():
                cats = row["objects"].get("category", [])
                if any(0 <= c < len(COCO_CLASSES) for c in cats):
                    candidates.append({"file": f, "idx": idx})
        except Exception as e:
            logger.warning(f"Skipping corrupt file {f}: {e}")
    return candidates


def process_row(row, img_save_dir: str) -> Optional[Dict]:
    img_id = row["image_id"]
    fname = f"{img_id:012d}.jpg"
    save_path = os.path.join(img_save_dir, fname)

    try:
        if os.path.exists(save_path):
            img = Image.open(save_path).convert("RGB")
        else:
            image_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.save(save_path)
    except Exception as e:
        logger.error(f"Error processing image {img_id}: {e}")
        return None

    objects = row["objects"]
    refs, bboxes = [], []

    category_list = objects.get("category", [])
    bbox_list = objects.get("bbox", [])

    if len(category_list) != len(bbox_list):
        return None

    for cat, bbox in zip(category_list, bbox_list):
        if 0 <= cat < len(COCO_CLASSES):
            refs.append(COCO_CLASSES[cat])
            new_bbox = convert_to_qwen25vl_format(bbox, img.height, img.width)
            bboxes.append(new_bbox)

    if not refs:
        return None

    text_label = ", ".join(["<ref-object><bbox>"] * len(refs))

    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "<image>Task: Object Detection"},
            {"role": "assistant", "content": text_label},
        ],
        "images": [os.path.join("images", fname)],
        "objects": {"ref": refs, "bbox": bboxes},
    }


def main():
    args = parse_args()

    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    logger.info(f"Starting processing, Output Dir: {args.output_dir}")

    data_path = get_data_path(args.dataset_repo)
    all_files = glob.glob(os.path.join(data_path, "*.parquet"))

    train_files = [f for f in all_files if "train" in os.path.basename(f)]
    val_files = [f for f in all_files if "val" in os.path.basename(f)]

    if not train_files:
        logger.error(f"No parquet training files found in {data_path}")
        return

    logger.info("Scanning metadata (Phase 1)...")
    train_pool = scan_dataset_metadata(train_files, "Scanning Train")
    val_pool = scan_dataset_metadata(val_files, "Scanning Val")

    n_val = int(args.total_samples * args.val_ratio)
    n_train = args.total_samples - n_val

    if len(train_pool) < n_train:
        logger.warning(f"Requested {n_train} train samples, but only found {len(train_pool)}. Using all available.")
        n_train = len(train_pool)

    if len(val_pool) < n_val:
        logger.warning(f"Requested {n_val} val samples, but only found {len(val_pool)}. Using all available.")
        n_val = len(val_pool)

    logger.info(f"Sampling Plan: Train={n_train}, Val={n_val} (Target Total={args.total_samples})")

    random.seed(args.seed)
    random.shuffle(train_pool)
    random.shuffle(val_pool)

    selected_train = train_pool[:n_train]
    selected_val = val_pool[:n_val]

    tasks = defaultdict(list)
    for item in selected_train:
        tasks[item["file"]].append((item["idx"], "train"))
    for item in selected_val:
        tasks[item["file"]].append((item["idx"], "val"))

    logger.info(f"Processing images (Phase 2) - Reading from {len(tasks)} parquet files...")

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as train_f, open(val_path, "w", encoding="utf-8") as val_f:

        for p_file, task_list in tqdm(tasks.items(), desc="Processing Parquet"):
            try:
                df = pq.read_table(p_file).to_pandas()
                for row_idx, split in task_list:
                    entry = process_row(df.iloc[row_idx], img_dir)
                    if entry:
                        line = json.dumps(entry, ensure_ascii=False) + "\n"
                        if split == "train":
                            train_f.write(line)
                        else:
                            val_f.write(line)
            except Exception as e:
                logger.error(f"Failed to process file {p_file}: {e}")
                continue

    logger.info(f"Output images and jsonl saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
