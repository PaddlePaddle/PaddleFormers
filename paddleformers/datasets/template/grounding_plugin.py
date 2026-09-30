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

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BaseGroundingPlugin:
    # Coordinate space the model expects for bounding boxes. ``"none"`` keeps the
    # raw pixel coordinates; ``"norm1000"`` rescales them to 0-1000 relative to
    # the owning image, which is what the Qwen grounding tokens encode.
    norm_bbox: str = "none"

    def normalize_bbox(
        self,
        bbox: List[float],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> List[int]:
        if self.norm_bbox == "none" or image_size is None:
            return [int(coord) for coord in bbox]
        if self.norm_bbox != "norm1000":
            raise ValueError(f"Unsupported bbox normalization mode: {self.norm_bbox}")

        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError(f"Image dimensions must be positive, got width={width}, height={height}")
        # bbox is [x1, y1, x2, y2]: even indices scale by width, odd by height.
        return [int(round(coord / (width if index % 2 == 0 else height) * 1000)) for index, coord in enumerate(bbox)]

    def format_ref_object(self, obj_name: str) -> str:
        return f"<|object_ref_start|>{obj_name}<|object_ref_end|>"

    def format_bbox(
        self,
        bbox: List[float],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> str:
        normalized = self.normalize_bbox(bbox, image_size=image_size)
        return f"<|box_start|>({normalized[0]},{normalized[1]}),({normalized[2]},{normalized[3]})<|box_end|>"

    def process_messages(self, messages, objects):
        ref_objects = objects.get("ref", [])
        bboxes = objects.get("bbox", [])
        # Filled in by the dataset when normalization is required. ``image_id``
        # selects the owning image for multi-image samples and defaults to 0.
        widths = objects.get("width", [])
        heights = objects.get("height", [])
        image_ids = objects.get("image_id", [])

        ref_idx = 0
        bbox_idx = 0

        for message in messages:
            content = message.get("content", "")
            ref_count = content.count("<ref-object>")
            bbox_count = content.count("<bbox>")
            current_refs = ref_objects[ref_idx : ref_idx + ref_count]
            current_bboxes = bboxes[bbox_idx : bbox_idx + bbox_count]

            for ref in current_refs:
                message["content"] = message["content"].replace("<ref-object>", self.format_ref_object(ref), 1)
            for local_index, bbox in enumerate(current_bboxes):
                image_index = 0
                bbox_index = bbox_idx + local_index
                if bbox_index < len(image_ids):
                    image_index = image_ids[bbox_index]
                image_size = None
                if image_index < len(widths) and image_index < len(heights):
                    image_size = (widths[image_index], heights[image_index])
                message["content"] = message["content"].replace(
                    "<bbox>", self.format_bbox(bbox, image_size=image_size), 1
                )

            ref_idx += ref_count
            bbox_idx += bbox_count

        return messages


PLUGINS = {
    "base": BaseGroundingPlugin,
}


def register_grounding_plugin(name, plugin_class):
    if name in PLUGINS:
        raise ValueError(f"Grounding plugin {name} already exists.")

    PLUGINS[name] = plugin_class


def get_grounding_plugin(
    name: str,
    **kwargs,
):
    if name not in PLUGINS:
        raise ValueError(f"Grounding plugin `{name}` not found.")

    return PLUGINS[name](**kwargs)
