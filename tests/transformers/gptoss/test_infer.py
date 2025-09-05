# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from paddleformers.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_path = "openai/gpt-oss-20b-bf16"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    convert_from_hf=True,
)

inputs = tokenizer.encode("Hello world", return_tensors='pd')

outputs = model.generate(**inputs,max_new_tokens=128, return_dict=True)
print(tokenizer.decode(outputs[0][0].tolist()))

# 模型保存
# 如果需要将模型保存为适配HF-torch模型，新增一个参数save_to_hf
model.save_pretrained(save_dir=save_dir, save_to_hf=True)