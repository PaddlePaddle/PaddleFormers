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


def convert_txt_data(item):
    if isinstance(item["src"], str):
        item["src"] = [item["src"]]
    if isinstance(item["tgt"], str):
        item["tgt"] = [item["tgt"]]

    # data check
    if len(item["src"]) == 0 or len(item["tgt"]) == 0:
        # raise ValueError("Ignore example with empty src or empty tgt.")
        return None

    for item_str in item["src"] + item["tgt"]:
        if len(item_str.strip()) == 0:
            # raise ValueError("Ignore example with empty string in str / tgt field.")
            return None

    if "label" not in item:
        item["label"] = [1] * len(item["src"])

    if not (len(item["src"]) == len(item["tgt"]) == len(item["label"])):
        # raise ValueError(
        #     f"The length of src & tgt & label must be equal, but get len(item['src']) : {len(item['src'])}, ' len(item['tgt']) : {len(item['tgt'])}, ' len(item['label']) : {len(item['label'])}"
        # )
        return None

    if "is_system" not in item:
        # If is_system is 1, it indicates that the sample includes system settings
        # and no other sample should be concatenated before it.
        item["is_system"] = 0

    if item["is_system"] == 1:
        item["system"] = item["src"][0]
        item["src"] = item["src"][1:]
        item["tgt"] = item["tgt"][1:]
        item["label"] = item["label"][1:]

    # update "system"
    if "system" in item:
        if not isinstance(item["system"], str):
            raise ValueError("System field must be a string.")
        item["is_system"] = 1

    res = {}
    # convert to OpenAI format
    res["messages"] = []
    if len(item.get("system", "")) > 0:
        res["messages"].append({"role": "system", "content": item["system"]})
    for q, a in zip(item["src"], item["tgt"]):
        res["messages"].append({"role": "user", "content": q})
        res["messages"].append({"role": "assistant", "content": a})
    return res


def convert_mm_data(item):
    if len(item.get("image_info", [])) > 0 and len(item.get("video_info", [])) > 0:
        assert "order" in item, "when image and video both exist, data must contain order"
        order = item["order"]
        order_type = order["type"]
        order_index = order["index"]
    else:
        if len(item.get("image_info", [])) > 0:
            mm_info = item.get("image_info", [])
            mm_type = "image"
        else:
            mm_info = item.get("video_info", [])
            mm_type = "video"
        order_type = ["text"] * len(item.get("text_info", []))
        order_index = list(range(len(order_type)))

        matched_text_index_list = []
        for i, info in enumerate(mm_info):
            matched_text_index_list.append((info["matched_text_index"], i))
        matched_text_index_list.sort()
        idx_shift = 0
        for idx, i in matched_text_index_list:
            order_type.insert(idx + idx_shift, mm_type)
            order_index.insert(idx + idx_shift, i)
            idx_shift += 1

    data_info = {
        "text_info": item.get("text_info", []),
        "image_info": item.get("image_info", []),
        "video_info": item.get("video_info", []),
    }

    messages = []
    images = []
    videos = []

    if len(item.get("system", "")) > 0:
        messages.append({"role": "system", "content": item["system"]})

    content = ""
    tag = ""
    for data_type, data_idx in zip(order_type, order_index):
        if data_type == "text":
            new_tag = data_info["text_info"][data_idx]["tag"]
        else:
            new_tag = "mask"
        if tag != new_tag:
            if tag == "mask":
                messages.append({"role": "user", "content": content})
            elif tag == "no_mask":
                messages.append({"role": "assistant", "content": content})
            tag = new_tag
            content = ""
        if data_type == "text":
            content += data_info["text_info"][data_idx]["text"]
        elif data_type == "image":
            content += "<image>"
            images.append(data_info["image_info"][data_idx]["image_url"])
        elif data_type == "video":
            content += "<video>"
            videos.append(data_info["video_info"][data_idx]["image_url"])
    if tag == "mask":
        messages.append({"role": "user", "content": content})
    elif tag == "no_mask":
        messages.append({"role": "assistant", "content": content})
    res = {"messages": messages}
    if len(images) > 0:
        res["images"] = images
    if len(videos) > 0:
        res["videos"] = videos
    return res


def erniekit_convertor(item):
    if "src" in item and "tgt" in item:
        res = convert_txt_data(item)
    else:
        res = convert_mm_data(item)
    return res


def query_response_convertor(item):
    res = {}
    # convert to OpenAI format
    res["messages"] = []
    if len(item.get("system", "")) > 0:
        res["messages"].append({"role": "system", "content": item["system"]})
    for q, a in item.get("history", []):
        res["messages"].append({"role": "user", "content": q})
        res["messages"].append({"role": "assistant", "content": a})
    res["messages"].append({"role": "user", "content": item.get("query", "")})
    res["messages"].append({"role": "assistant", "content": item.get("response", "")})

    images = item.get("images", [])
    videos = item.get("videos", [])
    if len(images) > 0:
        res["images"] = images
    if len(videos) > 0:
        res["videos"] = videos

    return res
