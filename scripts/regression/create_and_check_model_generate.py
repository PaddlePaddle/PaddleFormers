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

import sys

import paddle

MODEL_MAPPING = {
    "glm_moe": {
        "import_path": "paddleformers.transformers.glm4_moe.modeling",
        "class_name": "Glm4MoeForCausalLMDeprecated",
        "dtype": "bfloat16",
    },
    "qwen3_next": {
        "import_path": "paddleformers.transformers.qwen3_next.modeling",
        "class_name": "Qwen3NextForCausalLM",
        "dtype": "float32",
    },
    "gemma3_text": {
        "import_path": "paddleformers.transformers.gemma3_text.modeling",
        "class_name": "Gemma3ForCausalLM",
        "dtype": "bfloat16",
    },
    "gpt_oss": {
        "import_path": "paddleformers.transformers.gpt_oss.modeling",
        "class_name": "GptOssForCausalLM",
        "dtype": "bfloat16",
    },
    "qwen3moe": {
        "import_path": "paddleformers.transformers",
        "class_name": "Qwen3MoeForCausalLMDeprecated",
        "dtype": "float32",
    },
    "phi3": {
        "import_path": "paddleformers.transformers.phi3.modeling",
        "class_name": "Phi3ForCausalLM",
        "dtype": "bfloat16",
    },
    "llama": {
        "import_path": "paddleformers.transformers.llama.modeling",
        "class_name": "LlamaForCausalLM",
        "dtype": "bfloat16",
    },
    "deepseek_v3": {
        "import_path": "paddleformers.transformers.deepseek_v3.modeling",
        "class_name": "DeepseekV3ForCausalLM",
        "dtype": "bfloat16",
    },
    "qwen2moe": {
        "import_path": "paddleformers.transformers",
        "class_name": "Qwen2MoeForCausalLMDeprecated",
        "dtype": "float32",
    },
    "qwen3": {
        "import_path": "paddleformers.transformers.qwen3.modeling",
        "class_name": "Qwen3ForCausalLMDeprecated",
        "dtype": "bfloat16",
    },
    "qwen2": {
        "import_path": "paddleformers.transformers.qwen2.modeling",
        "class_name": "Qwen2ForCausalLM",
        "dtype": "bfloat16",
    },
    "ernie4_5_moe": {
        "import_path": "paddleformers.transformers.ernie4_5_moe.modeling",
        "class_name": "Ernie4_5_MoeForCausalLM",
        "dtype": "bfloat16",
    },
    "ernie4_5": {
        "import_path": "paddleformers.transformers.ernie4_5.modeling",
        "class_name": "Ernie4_5ForCausalLM",
        "dtype": "bfloat16",
    },
    "ernie4_5_moe_vl": {
        "import_path": "paddleformers.transformers.ernie4_5_moe_vl.model.modeling_moe",
        "class_name": "Ernie4_5_MoeForCausalLM",
        "dtype": "bfloat16",
    },
    "paddleocr_vl": {
        "import_path": "paddleformers.transformers",
        "class_name": "PaddleOCRVLForConditionalGeneration",
        "dtype": "float32",
    },
    "qwen2_5_vl": {
        "import_path": "paddleformers.transformers",
        "class_name": "Qwen2_5_VLForConditionalGeneration",
        "dtype": "float32",
    },
    "qwen3_vl_moe": {
        "import_path": "paddleformers.transformers",
        "class_name": "Qwen3VLMoeForConditionalGenerationDeprecated",
        "dtype": "bfloat16",
    },
    "qwen3_vl": {
        "import_path": "paddleformers.transformers",
        "class_name": "Qwen3VLForConditionalGenerationDeprecatedn",
        "dtype": "bfloat16",
    },
}


def create_and_check_model_generate(
    model_key,
    model_path,
    excepted_result,
):
    model_config = MODEL_MAPPING.get(model_key)
    if not model_config:
        raise ValueError(f"Unsupported model key: {model_key}")

    module = __import__(model_config["import_path"], fromlist=[model_config["class_name"]])
    model_class = getattr(module, model_config["class_name"])
    model_dtype = model_config["dtype"]
    model = model_class.from_pretrained(
        model_path, dtype=model_dtype, convert_from_hf=True, num_nextn_predict_layers=0
    )

    input_ids = paddle.to_tensor([[1, 306, 4658, 278, 6593, 310, 2834, 338]])
    attention_mask = paddle.ones_like(input_ids)
    with paddle.no_grad():
        result = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10)

    excepted_result = paddle.to_tensor(excepted_result)
    print(f"excepted_result is : {excepted_result}")
    print(f"result[0] is : {result[0]}")
    assert paddle.allclose(result[0], excepted_result), f"Result {result[0]} does not match expected {excepted_result}"

    return [tensor.numpy().tolist() for tensor in result]


if __name__ == "__main__":

    create_and_check_model_generate(
        model_key=sys.argv[1],
        model_path=sys.argv[2],
        excepted_result=sys.argv[3],
    )
