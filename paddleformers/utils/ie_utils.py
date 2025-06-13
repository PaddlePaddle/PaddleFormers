# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import re
from io import BytesIO

import numpy as np
import paddle
from PIL import Image

from ..metrics import SpanEvaluator
from .image_utils import NormalizeImage, Permute, ResizeImage

resize_func = ResizeImage(target_size=224, interp=1)
norm_func = NormalizeImage(is_channel_first=False, mean=[123.675, 116.280, 103.530], std=[58.395, 57.120, 57.375])
permute_func = Permute(to_bgr=False)


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def pad_image_data(image_data):
    if not image_data:
        image = np.zeros([3, 224, 224])
        return image
    # decode image
    data = np.frombuffer(bytearray(image_data), dtype="uint8")
    image = np.array(Image.open(BytesIO(data)).convert("RGB"))
    sample = {"image": image}
    # resize image
    sample = resize_func(sample)
    # norm image
    sample = norm_func(sample)
    # permute
    sample = permute_func(sample)
    return sample["image"]


def unify_prompt_name(prompt):
    # The classification labels are shuffled during finetuning, so they need
    # to be unified during evaluation.
    if re.search(r"\[.*?\]$", prompt):
        prompt_prefix = prompt[: prompt.find("[", 1)]
        cls_options = re.search(r"\[.*?\]$", prompt).group()[1:-1].split(",")
        cls_options = sorted(list(set(cls_options)))
        cls_options = ",".join(cls_options)
        prompt = prompt_prefix + "[" + cls_options + "]"
        return prompt
    return prompt


def get_relation_type_dict(relation_data, schema_lang="ch"):
    def compare(a, b, schema_lang="ch"):
        if schema_lang == "ch":
            a = a[::-1]
            b = b[::-1]

        res = ""
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]:
                res += a[i]
            else:
                break
        if res == "":
            return res
        if schema_lang == "ch" and res[::-1][0] == "的":
            return res[::-1][1:]
        elif schema_lang == "en" and res[-3:] == " of":
            return res[:-3]
        return ""

    relation_type_dict = {}
    added_list = []
    for i in range(len(relation_data)):
        added = False
        if relation_data[i][0] not in added_list:
            for j in range(i + 1, len(relation_data)):
                match = compare(relation_data[i][0], relation_data[j][0], schema_lang=schema_lang)
                if match != "":
                    match = unify_prompt_name(match)
                    if relation_data[i][0] not in added_list:
                        added_list.append(relation_data[i][0])
                        relation_type_dict.setdefault(match, []).append(relation_data[i][1])
                    added_list.append(relation_data[j][0])
                    relation_type_dict.setdefault(match, []).append(relation_data[j][1])
                    added = True
            if not added:
                added_list.append(relation_data[i][0])
                if schema_lang == "ch":
                    suffix = relation_data[i][0].rsplit("的", 1)[1]
                    suffix = unify_prompt_name(suffix)
                    relation_type = suffix
                else:
                    prefix = relation_data[i][0].split(" of ", 1)[0]
                    prefix = unify_prompt_name(prefix)
                    relation_type = prefix
                relation_type_dict.setdefault(relation_type, []).append(relation_data[i][1])
    return relation_type_dict


def uie_loss_func(outputs, labels):
    criterion = paddle.nn.BCELoss()
    start_ids, end_ids = labels
    start_prob, end_prob = outputs
    start_ids = paddle.cast(start_ids, "float32")
    end_ids = paddle.cast(end_ids, "float32")
    loss_start = criterion(start_prob, start_ids)
    loss_end = criterion(end_prob, end_ids)
    loss = (loss_start + loss_end) / 2.0
    return loss


def compute_metrics(p):
    metric = SpanEvaluator()
    start_prob, end_prob = p.predictions
    start_ids, end_ids = p.label_ids
    metric.reset()

    num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
    metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    metric.reset()

    return {"precision": precision, "recall": recall, "f1": f1}
