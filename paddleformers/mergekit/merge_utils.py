# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


def divide_positions(m, n):
    if n == 0:
        raise ValueError("n should be greater than zero")
    if m < n:
        raise ValueError("tensor number should be greater than or equal to processor number")
    base_value = m // n
    remainder = m % n
    positions = [0]
    for i in range(1, n):
        if remainder > 0:
            positions.append(positions[-1] + base_value + 1)
            remainder -= 1
        else:
            positions.append(positions[-1] + base_value)
    positions.append(m)
    return positions


def divide_lora_key_list(key_list, n, lora_config):
    lora_key = []
    other_key = []
    for module_name in key_list:
        if (
            any(re.fullmatch(target_module, module_name) for target_module in lora_config.target_modules)
            and "weight" in module_name
        ):
            lora_key.append(module_name)
        else:
            other_key.append(module_name)
    lora_positions = divide_positions(len(lora_key), n)
    other_positions = divide_positions(len(other_key), n)
    divided_key_list = []
    for i in range(len(lora_positions) - 1):
        divided_key = (
            lora_key[lora_positions[i] : lora_positions[i + 1]]
            + other_key[other_positions[i] : other_positions[i + 1]]
        )
        divided_key_list.append(divided_key)
    return divided_key_list


def divide_safetensor_key_list(weight_map, n):
    file_map = {}
    for key in weight_map:
        if weight_map[key] in file_map:
            file_map[weight_map[key]].append(key)
        else:
            file_map[weight_map[key]] = [key]
    file_list = list(file_map.keys())
    p = divide_positions(len(file_list), n)
    key_list = []
    positions = [0]
    for i in range(n):
        for file in file_list[p[i] : p[i + 1]]:
            key_list += file_map[file]
        positions.append(len(key_list))
    return key_list, positions
