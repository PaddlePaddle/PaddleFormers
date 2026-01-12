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

import json
import math
import os
import re
from typing import Dict, List, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

MODEL_RESULT = {
    "image_path": "images/000000299887.jpg",
    "ground_truth": {
        "ref": ["motorcycle", "person", "person", "truck"],
        "bbox": [[5, 129, 322, 471], [212, 105, 447, 475], [373, 121, 522, 475], [0, 178, 48, 213]],
    },
    "prediction": "motorcycle(0,132),(379,475), person(212,106),(442,475), person(362,118),(522,475), truck(0,189),(48,213)",
}
ROOT_DIR = "data/coco_grounding"


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
        raise ValueError("absolute aspect ratio must be smaller than 200")

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


def parse_prediction_string(pred_str: str) -> List[Dict]:
    if not pred_str:
        return []

    # Matches format: label(x1,y1),(x2,y2)
    pattern = r"([a-zA-Z0-9_ ]+)\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)\s*,\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)"
    matches = re.findall(pattern, pred_str)
    results = []
    for m in matches:
        label = m[0].strip()
        bbox = [float(x) for x in m[1:]]
        results.append({"label": label, "bbox": bbox})
    return results


def get_color_by_label(label: str) -> str:
    palette = [
        "#FF0000",
        "#00AA00",
        "#0000FF",
        "#FF00FF",
        "#800080",
        "#008080",
        "#FFA500",
        "#8B4513",
        "#DC143C",
        "#2E8B57",
        "#4B0082",
        "#FF4500",
        "#2F4F4F",
        "#8B0000",
        "#191970",
    ]
    color_index = hash(label) % len(palette)
    return palette[color_index]


def visualize_sample(
    sample_data: Union[Dict, str],
    image_root: str = "",
    save_path: str = "output_vis.jpg",
    show_gt: bool = True,
    show_pred: bool = True,
    random_colors: bool = True,
):
    """
    Visualizes Ground Truth and Prediction for a single sample.
    """

    if isinstance(sample_data, str):
        item = json.loads(sample_data)
    else:
        item = sample_data

    rel_path = item.get("image_path", "")
    gt_data = item.get("ground_truth", {})
    pred_str = item.get("prediction", "")

    full_image_path = os.path.join(image_root, rel_path)
    try:
        img = Image.open(full_image_path).convert("RGB")
    except FileNotFoundError:
        print(f"[Warning] Image not found: {full_image_path}, creating blank placeholder.")
        img = Image.new("RGB", (640, 640), color=(200, 200, 200))

    orig_w, orig_h = img.size

    try:
        new_h, new_w = smart_resize(orig_h, orig_w)
        resized_img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
    except Exception as e:
        print(f"[Error] Resize failed: {e}")
        return

    draw = ImageDraw.Draw(resized_img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        try:
            font = ImageFont.truetype("arialbd.ttf", 16)
        except OSError:
            font = ImageFont.load_default()

    def _draw_box(bbox, text, color, offset_y=0):
        x1, y1, x2, y2 = bbox
        nx1, ny1 = x1 * scale_x, y1 * scale_y
        nx2, ny2 = x2 * scale_x, y2 * scale_y

        draw.rectangle([nx1, ny1, nx2, ny2], outline=color, width=3)

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        text_bg_x1 = nx1
        text_bg_y1 = ny1 - text_h - 4 + offset_y
        if text_bg_y1 < 0:
            text_bg_y1 = ny1 + 4

        text_bg_x2 = text_bg_x1 + text_w + 8
        text_bg_y2 = text_bg_y1 + text_h + 4

        draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=color)
        draw.text((text_bg_x1 + 4, text_bg_y1 + 2), text, font=font, fill=(255, 255, 255))

    if show_gt and gt_data:
        refs = gt_data.get("ref", [])
        bboxes = gt_data.get("bbox", [])
        for label, bbox in zip(refs, bboxes):
            label_text = f"GT: {label}"
            color = get_color_by_label(label) if random_colors else "#00AA00"
            _draw_box(bbox, label_text, color, offset_y=0)

    if show_pred and pred_str:
        preds = parse_prediction_string(pred_str)
        for p in preds:
            label = p["label"]
            label_text = f"Pred: {label}"
            color = get_color_by_label(label) if random_colors else "#FF0000"
            offset = 25 if (show_gt and not random_colors) else 0
            _draw_box(p["bbox"], label_text, color, offset_y=offset)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    resized_img.save(save_path, quality=95)
    print(f"Visualization saved to: {save_path}")


if __name__ == "__main__":
    visualize_sample(
        sample_data=MODEL_RESULT,
        image_root=ROOT_DIR,
        save_path="vis_gt.jpg",
        show_gt=True,
        show_pred=False,
        random_colors=True,
    )

    visualize_sample(
        sample_data=MODEL_RESULT,
        image_root=ROOT_DIR,
        save_path="vis_pred.jpg",
        show_gt=True,
        show_pred=True,
        random_colors=False,
    )
