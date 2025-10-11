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
        each_line = dict()
        each_line["messages"] = []
        if "system" in item:
            each_line["messages"].append({"role": "system", "content": item["system"]})
        for q, a in zip(item["src"], item["tgt"]):
            each_line["messages"].append({"role": "user", "content": q.strip()})
            each_line["messages"].append({"role": "assistant", "content": a.strip()})
        all_data.append(each_line)

    return all_data
