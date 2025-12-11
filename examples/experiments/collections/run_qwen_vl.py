import copy
import re
import paddle
from paddleformers.transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", convert_from_hf=True,
).eval()

# change the implementation of attention(default is "eager")
# model.language_model.config._attn_implementation = "flashmask"
# model.visual.config._attn_implementation = "flashmask"

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
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
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)
print(inputs)
# print("model ",model)
# generated_ids = model.generate(**inputs, max_new_tokens=128)

# output_text = processor.tokenizer.batch_decode(
#     generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
# )




# from paddleformers.transformers import Qwen2_5_VLModel,Qwen2_5_VLConfig,Qwen2_5_VLVisionConfig,Qwen2_5_VLTextConfig
# qwen_25vl_3b_text_kwargs={
#     "vocab_size": 151936,
#     "hidden_size": 2048,
#     "intermediate_size": 11008,
#     "fuse_attention_ffn": True,
#     "num_hidden_layers": 36,
#     "num_attention_heads": 16,
#     "num_key_value_heads": 2,
#     "hidden_act": "silu",
#     "max_position_embeddings": 32768,
#     "initializer_range": 0.02,
#     "rms_norm_eps": 1e-6,
#     "use_cache": True,
#     "tie_word_embeddings": False,
#     "rope_theta": 10000.0,
#     "use_sliding_window": False,
#     "sliding_window": 4096,
#     "max_window_layers": 80,
#     "layer_types": None,
#     "attention_dropout": 0.0,
#     "rope_scaling": None
# }
# qwen_25vl_3b_vis_kwargs={
#     "depth": 32,
#     "hidden_size": 1280,
#     "hidden_act": "silu",
#     "intermediate_size": 3420,
#     "num_heads": 16,
#     "in_channels": 3,
#     "patch_size": 14,
#     "spatial_merge_size": 2,
#     "temporal_patch_size": 2,
#     "tokens_per_second": 4,
#     "window_size": 112,
#     "out_hidden_size": 1280,
#     "fullatt_block_indexes": [7, 15, 23, 31],
#     "initializer_range": 0.02
# }

# qwen25vltext_3b_config = Qwen2_5_VLTextConfig(*qwen_25vl_3b_text_kwargs)
# qwen25vlvis_3b_config = Qwen2_5_VLVisionConfig(*qwen_25vl_3b_vis_kwargs)
# qwen25vl_3b_config = Qwen2_5_VLConfig(text_config=qwen25vltext_3b_config,vision_config=qwen25vlvis_3b_config)
# qwen25vl_3b_model = Qwen2_5_VLModel(qwen25vl_3b_config)
# print("qwen25vl_3b_model ",qwen25vl_3b_model)

def map_layer_name(original_name):
    """映射层名称，处理可变的数字索引"""
    # 定义替换规则
    mapping_rules = [
        ("model.language_model.layers","language_model"),
        ("model.language_model.embed_tokens.weight","language_model.0.embedding.embed_tokens.weight"),
        ("model.language_model.norm","language_model.37.norm_cls"),
        ("model.visual.patch_embed.proj","vision_model.conv1"),
        ("model.visual.merger.mlp.0","vision_projection.encoder.up_gate_proj"),
        ("model.visual.merger.mlp.2","vision_projection.encoder.down_proj"),
        ("model.visual.merger.ln_q","vision_model.decoder.norm"),
        ("model.language_model.norm","language_model.decoder.norm"),
        ("model.visual", "vision_model"),
        (".blocks.",".decoder.layers."),
        (".norm1", ".input_layernorm"),
        (".norm2", ".post_attention_layernorm"),
        (".attn", ".self_attn"),
        (".qkv.",".qkv_proj."),
        (".proj.", ".o_proj."),
    ]

    if "language_model" in original_name:
        # 使用正则表达式匹配 layers.X. 的模式
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
    
    return mapped_name


from experiments.collections.vlm.qwen2vl.model import Qwen2VLModel, Qwen25VLProvider3B

fleet_model = Qwen2VLModel(Qwen25VLProvider3B(), model_version="qwen25-vl")
fleet_model.provide()
fleet_model = fleet_model.module
print("fleet_model ")
fleet_state_dict = {}

for name, param in fleet_model.named_parameters():
    print("fleet ",name,param.name,param.shape)
    fleet_state_dict[name]=param


# print(model.language_model)
for name, param in model.named_parameters():
    print("formers ",name,param.name,param.shape)
    fleet_name = map_layer_name(name)
    if fleet_name not in fleet_state_dict:
        print(f"name {name} duiying {fleet_name} not found")
        raise ValueError("fuck my life")
    paddle.assign(param,fleet_state_dict[fleet_name])

for name, param in fleet_model.named_parameters():
    # print(name,param.name,param.shape)
    fleet_state_dict[name]=param
       
for name, param in model.named_parameters():
    fleet_name = map_layer_name(name)
    if not paddle.equal_all(param.cast('float32'),fleet_state_dict[fleet_name].cast('float32')):
        print(f"name {name} duiying {fleet_name} not equal")

paddle.seed(42)
with paddle.no_grad():
    output = model(**inputs)
print(output)
paddle.seed(42)
with paddle.no_grad():
    output = fleet_model(**inputs)
print(output)
