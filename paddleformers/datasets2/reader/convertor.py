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


def erniekit_convertor(data):
    all_data = []
    for item in data:

        if isinstance(item["src"], str):
            item["src"] = [item["src"]]
        if isinstance(item["tgt"], str):
            item["tgt"] = [item["tgt"]]

        # data check
        if len(item["src"]) == 0 or len(item["tgt"]) == 0:
            # raise ValueError("Ignore example with empty src or empty tgt.")
            continue

        for item_str in item["src"] + item["tgt"]:
            if len(item_str.strip()) == 0:
                # raise ValueError("Ignore example with empty string in str / tgt field.")
                continue

        if "label" not in item:
            item["label"] = [1] * len(item["src"])

        if not (len(item["src"]) == len(item["tgt"]) == len(item["label"])):
            # raise ValueError(
            #     f"The length of src & tgt & label must be equal, but get len(item['src']) : {len(item['src'])}, ' len(item['tgt']) : {len(item['tgt'])}, ' len(item['label']) : {len(item['label'])}"
            # )
            continue

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
        if "system" in item:
            res["messages"].append({"role": "system", "content": item["system"]})
        for q, a in zip(item["src"], item["tgt"]):
            res["messages"].append({"role": "user", "content": q.strip()})
            res["messages"].append({"role": "assistant", "content": a.strip()})
        all_data.append(res)

    return all_data
