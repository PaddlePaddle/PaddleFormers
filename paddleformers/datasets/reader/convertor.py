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

import json


def convert_dpo_txt_data(data):
    """Convert raw format example to Example."""
    if isinstance(data["src"], str):
        data["src"] = [data["src"]]
    if isinstance(data["tgt"], str):
        data["tgt"] = [data["tgt"]]
    if len(data["src"]) != len(data["tgt"]) + 1:
        raise ValueError(
            f"Data format error. src length must be tgt length + 1. "
            f"But got src_length:{len(data['src'])} tgt_length:{len(data['tgt'])}"
        )
    if (len(data["response"]) != 2) or (len(data["response"]) != len(data["sort"])):
        raise ValueError(
            f"Response and sort length must be 2. "
            f"But got response_length:{len(data['response'])} sort_length:{len(data['sort'])}."
        )
    if data["sort"][0] == data["sort"][1]:
        raise ValueError(f"Sort field must be different." f" But got 'sort':{data['sort']}")
    if isinstance(data["response"][0], str) and isinstance(data["response"][1], str):
        data["response"] = [[data["response"][0]], [data["response"][1]]]
    for response in data["response"]:
        if not isinstance(response, list):
            raise ValueError(f"Session level response should be List[List[str]], but got List of {type(response)}")
        if len(response) % 2 != 1:
            raise ValueError("The number of responses should be even, but an odd number of responses were obtained.")
        for r in response:
            if len(r.strip()) < 1:
                raise ValueError(f"Response field must be longer than 1." f" But got 'response':{data['response']}.")

    if len(data["response"][0]) < 1 or len(data["response"][1]) < 1:
        raise ValueError(f"Ignore empty response." f" But got 'response':{data['response']}.")
    if data["sort"][0] > data["sort"][1]:
        chosen = data["response"][0]
        rejected = data["response"][1]
    else:
        chosen = data["response"][1]
        rejected = data["response"][0]

    if "is_system" not in data:
        # If is_system is 1, it indicates that the sample includes system settings
        # and no other sample should be concatenated before it.
        data["is_system"] = 0

    if data["is_system"] == 1:
        data["system"] = data["src"][0]
        data["src"] = data["src"][1:]
        data["tgt"] = data["tgt"][1:]

    if "system" in data:
        if not isinstance(data["system"], str):
            raise ValueError("System field must be a string.")

    # convert to OpenAI format
    data["messages"] = []
    if "system" in data:
        data["messages"].append({"role": "system", "content": data["system"]})
    for idx in range(len(data["src"])):
        data["messages"].append({"role": "user", "content": data["src"][idx]})
        if idx != len(data["src"]) - 1:
            data["messages"].append({"role": "assistant", "content": data["tgt"][idx]})

    chosen_response, rejected_response = [], []
    for idx in range(len(chosen)):
        if idx % 2 == 0:
            # assistant
            chosen_response.append({"role": "assistant", "content": chosen[idx]})
            rejected_response.append({"role": "assistant", "content": rejected[idx]})
        else:
            # user
            chosen_response.append({"role": "user", "content": chosen[idx]})
            rejected_response.append({"role": "user", "content": rejected[idx]})

    data["chosen_response"] = chosen_response
    data["rejected_response"] = rejected_response
    return data


def convert_txt_data(item):
    if isinstance(item["src"], str):
        item["src"] = [item["src"]]
    if isinstance(item["tgt"], str):
        item["tgt"] = [item["tgt"]]

    # data check
    if len(item["src"]) == 0 or len(item["tgt"]) == 0:
        raise ValueError("Ignore example with empty src or empty tgt.")
        return None

    for item_str in item["src"] + item["tgt"]:
        if len(item_str.strip()) == 0:
            raise ValueError("Ignore example with empty string in str / tgt field.")
            return None

    if "label" not in item:
        item["label"] = [1] * len(item["src"])

    if not (len(item["src"]) == len(item["tgt"]) == len(item["label"])):
        raise ValueError(
            f"The length of src & tgt & label must be equal, but get len(item['src']) : {len(item['src'])}, ' len(item['tgt']) : {len(item['tgt'])}, ' len(item['label']) : {len(item['label'])}"
        )
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
        "tools": item.get("tools", []),
    }

    messages = []
    images = []
    videos = []

    if len(item.get("system", "")) > 0:
        messages.append({"role": "system", "content": item["system"]})

    content = ""
    tool_calls_str = ""
    tag = ""
    for data_type, data_idx in zip(order_type, order_index):
        if data_type == "text":
            new_tag = data_info["text_info"][data_idx]["tag"]
        else:
            new_tag = "mask"
        if tag != new_tag:
            if tag == "mask":
                tool_response = data_info["text_info"][data_idx - 1].get("tool_response", False)
                if tool_response:
                    role = "observation"
                else:
                    role = "user"
                messages.append({"role": role, "content": content})
            elif tag == "no_mask":
                if len(tool_calls_str) > 0:
                    messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls_str})
                else:
                    messages.append({"role": "assistant", "content": content})
            tag = new_tag
            content = ""
            tool_calls_str = ""
        if data_type == "text":
            content += data_info["text_info"][data_idx]["text"]
            tool_calls = data_info["text_info"][data_idx].get("tool_calls", "")
            if isinstance(tool_calls, list):
                tool_calls = json.dumps(tool_calls)
            tool_calls_str += tool_calls
        elif data_type == "image":
            content += "<image>"
            images.append(data_info["image_info"][data_idx]["image_url"])
        elif data_type == "video":
            content += "<video>"
            videos.append(data_info["video_info"][data_idx]["image_url"])
    if tag == "mask":
        tool_response = data_info["text_info"][data_idx].get("tool_response", False)
        if tool_response:
            role = "observation"
        else:
            role = "user"
        messages.append({"role": role, "content": content})
    elif tag == "no_mask":
        if len(tool_calls_str) > 0:
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls_str})
        else:
            messages.append({"role": "assistant", "content": content})
    res = {"messages": messages}
    if len(images) > 0:
        res["images"] = images
    if len(videos) > 0:
        res["videos"] = videos
    if len(data_info["tools"]) > 0:
        res["tools"] = data_info["tools"]
    return res


def convert_pretraining_data(data):
    # convert to messages format
    if isinstance(data["text"], list):
        data["text"] = data["text"][0]
    assert isinstance(data["text"], str)

    if len(data["text"].strip()) == 0:
        raise ValueError("Ignore example with empty string.")

    res = {"messages": [{"role": "assistant", "content": data["text"]}]}

    return res


def erniekit_convertor(item):
    # erniekit dpo data
    if "src" in item and "tgt" in item and "response" in item:
        res = convert_dpo_txt_data(item)
    # erniekit sft data
    elif "src" in item and "tgt" in item:
        res = convert_txt_data(item)
    # erniekit pretraining data
    elif "text" in item:
        res = convert_pretraining_data(item)
    # erniekit multi modal data
    else:
        res = convert_mm_data(item)
    return res


def messages_convertor(item):
    return item
