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
from typing import List


@dataclass
class BaseGroundingPlugin:
    def normalize_bbox(self, bbox: List[float]) -> List[int]:
        return [int(coord) for coord in bbox]

    def format_ref_object(self, obj_name: str) -> str:
        return f"<|object_ref_start|>{obj_name}<|object_ref_end|>"

    def format_bbox(self, bbox: List[float]) -> str:
        normalized = self.normalize_bbox(bbox)
        return f"<|box_start|>({normalized[0]},{normalized[1]}),({normalized[2]},{normalized[3]})<|box_end|>"

    def process_messages(self, messages, objects):

        ref_objects = objects.get("ref", [])
        bboxes = objects.get("bbox", [])

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
            for bbox in current_bboxes:
                message["content"] = message["content"].replace("<bbox>", self.format_bbox(bbox), 1)

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
