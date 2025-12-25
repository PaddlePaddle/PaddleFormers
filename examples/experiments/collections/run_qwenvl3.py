import copy
import os
import re

import paddle

from paddleformers.transformers import AutoTokenizer, AutoConfig, AutoProcessor, process_vision_info
from paddleformers.transformers.qwen3_vl.modeling import Qwen3VLForConditionalGeneration

from vlm.qwen3vl.model import Qwen3VLModel, Qwen3VLProvider2B

model_path = f"{os.environ['HOME']}/Qwen3VL-2B-Instruct/weights"

formars_config = AutoConfig.from_pretrained(model_path)
formars_config.fuse_attention_qkv = True
formers_model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, convert_from_hf=True, fuse_attention_qkv=True, fuse_attention_ffn=True,
).eval()


fleet_model = Qwen3VLModel(
    Qwen3VLProvider2B(), model_version="qwen3-vl"
)
fleet_model.provide()
fleet_model = fleet_model.module


def map_layer_name(original_name):
    mapping_rules = [
        ("model.language_model.layers", "language_model"),
        ("model.language_model.embed_tokens.weight", "language_model.0.embedding.embed_tokens.weight"),
        ("model.language_model.norm", "language_model.{}.norm"),
        ("model.visual.patch_embed.proj", "vision_model.conv1"),
        (".mlp.linear_fc1.", ".mlp.up_gate_proj."),
        (".mlp.linear_fc2.", ".mlp.down_proj."),
        # ("model.visual.merger.mlp.0", "vision_projection.encoder.up_gate_proj"),
        # ("model.visual.merger.mlp.2", "vision_projection.encoder.down_proj"),
        # ("model.visual.merger.ln_q", "vision_model.decoder.norm"),
        # ("model.language_model.norm", "language_model.decoder.norm"),
        ("model.visual.merger", "vision_model.decoder.merger"),
        ("model.visual.deepstack_merger_list", "vision_model.decoder.deepstack_merger_list"),
        ("model.visual.", "vision_model."),
        (".blocks.",".decoder.layers."),
        (".norm1", ".input_layernorm"),
        (".norm2", ".post_attention_layernorm"),
        (".attn", ".self_attn"),
        (".qkv.",".qkv_proj."),
        (".proj.", ".o_proj."),
        (".q_norm.", ".q_layernorm."),
        (".k_norm.", ".k_layernorm."),
    ]
    
    layer_index = -1
    if "language_model" in original_name:
        # 使用正则表达式匹配 layers.X. 的模式
        match = re.search(r'layers\.(\d+)\.', original_name)
        if match:
            layer_index = int(match.group(1))
        def increment_layer(match):
            layer_num = int(match.group(1))
            return f'layers.{layer_num + 1}.'
        
        # 替换所有layer编号
        updated_name = re.sub(r'layers\.(\d+)\.', increment_layer, original_name)

        mapped_name = copy.deepcopy(updated_name)
    else:
        mapped_name = copy.deepcopy(original_name)
    # 按顺序应用替换规则
    for old, new in mapping_rules:
        mapped_name = mapped_name.replace(old, new)
    
    return mapped_name, layer_index


fleet_state_dict = {}
for name, param in fleet_model.named_parameters():
    print("fleet ", name, param.name, param.shape, sep="\t")
    fleet_state_dict[name]=param


language_depth = 0
for name, param in formers_model.named_parameters():
    print("formers ", name, param.name, param.shape)
    fleet_name, layer_index = map_layer_name(name)
    language_depth = max(language_depth, layer_index)
    if "language_model.norm" in name:
        fleet_name = fleet_name.format(language_depth + 2)
    if fleet_name not in fleet_state_dict:
        print(f"name {name} correspond {fleet_name} not found")
        raise ValueError("fuck my life")
    paddle.assign(param, fleet_state_dict[fleet_name])

for name, param in fleet_model.named_parameters():
    # print(name,param.name,param.shape)
    fleet_state_dict[name]=param
       
for name, param in formers_model.named_parameters():
    fleet_name, _ = map_layer_name(name)
    if "language_model.norm" in name:
        fleet_name = fleet_name.format(language_depth + 2)
    if not paddle.equal_all(param.cast('float32'),fleet_state_dict[fleet_name].cast('float32')):
        print(f"name {name} correspond {fleet_name} not equal")

processor = AutoProcessor.from_pretrained(model_path)
print("hehe")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    video=video_inputs,
    padding=True,
    return_tensors="pd",
)
print(inputs)

# paddle.seed(42)
# with paddle.no_grad():
#     formers_output = formers_model(**inputs)
# print(formers_output)

paddle.seed(42)
with paddle.no_grad():
    fleet_output = fleet_model(**inputs)
print(fleet_output)
