# from PIL import Image
# import paddle
# import numpy as np


# def glm_ocr_recognize(
#     model,
#     processor,
#     image_path: str,
#     text_prompt: str = "Text Recognition:",
#     max_new_tokens: int = 64,
# ):
#     img = Image.open(image_path).convert("RGB")
#     tok = processor.tokenizer

#     prompt = (
#         "[gMASK]<sop>"
#         "<|user|>\n"
#         "<|begin_of_image|><|image|><|end_of_image|>\n"
#         f"{text_prompt}\n"
#         "<|assistant|>\n"
#     )

#     # processor 返回 pt tensors，转成 paddle tensor
#     inputs = processor(
#         images=img,
#         text=prompt,
#         return_tensors="pd",
#     )
#     print(inputs["input_ids"].shape)
#     print(inputs["pixel_values"].shape)
#     print(inputs["input_ids"])
#     print(inputs["pixel_values"])

#     inputs.pop("token_type_ids", None)

#     # pt -> numpy -> paddle
#     paddle_inputs = {}
#     for k, v in inputs.items():
#         arr = v.numpy()
#         # pixel_values 通常是 float16/float32，保持原始 dtype
#         paddle_inputs[k] = paddle.to_tensor(arr)

#     eos = tok.eos_token_id
#     pad = tok.pad_token_id if tok.pad_token_id is not None else eos

#     with paddle.no_grad():
#         generated = model.generate(
#             **paddle_inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             eos_token_id=eos,
#             pad_token_id=pad,
#         )

#     new_tokens = generated[0][0]
#     print(new_tokens)
#     # paddle tensor -> python list
#     new_tokens_list = new_tokens.numpy().tolist()
#     out = tok.decode(new_tokens_list, skip_special_tokens=True)

#     return out.strip()


# # ── 加载 ──────────────────────────────────────────────────────────────────────
# # processor 直接复用 transformers 的（只做 tokenize / image preprocess，无推理）
# from .processor import GlmOcrProcessor
# from .modeling import GlmOcrForConditionalGeneration

# MODEL_PATH = "/home/work/zkx_test/glm_ocr_hf"
# IMG_PATH   = "/home/work/zkx_test/glmocr/handwritten.png"

# processor = GlmOcrProcessor.from_pretrained(MODEL_PATH, use_fast=False)

# # dtype 自动推断；若显存紧张可改 dtype="float16"
# model = GlmOcrForConditionalGeneration.from_pretrained(
#     MODEL_PATH,
#     dtype="float16",   # paddle 用 dtype= 而非 torch_dtype=
# ).eval()

# # paddle 默认在 CPU；有 GPU 时：
# # paddle.device.set_device("gpu:0")
# # model.to(paddle.float16)   # 或在 from_pretrained 里指定

# print(glm_ocr_recognize(model, processor, IMG_PATH))





# #auto版
# from PIL import Image
# import paddle
# from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
# from paddleformers.transformers import AutoTokenizer

# MODEL_PATH = "/home/work/zkx_test/glm_ocr_hf_paddle"
# # MODEL_PATH = "/home/work/zkx_test/PaddleFormers/GlmOCR-SFT-Bengali-lora"
# IMG_PATH   = "/home/work/zkx_test/glmocr/handwritten.png"

# processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
# tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)  # 单独加载
# model = AutoModelForConditionalGeneration.from_pretrained(
#     MODEL_PATH,
#     dtype="float16",
# ).eval()


# def glm_ocr_recognize(
#     model,
#     processor,
#     tok,                          # 作为参数传入
#     image_path: str,
#     text_prompt: str = "Text Recognition:",
#     max_new_tokens: int = 2048,
# ):
#     img = Image.open(image_path).convert("RGB")

#     prompt = (
#         "[gMASK]<sop>"
#         "<|user|>\n"
#         "<|begin_of_image|><|image|><|end_of_image|>\n"
#         f"{text_prompt}\n"
#         "<|assistant|>\n"
#     )

#     inputs = processor(
#         images=img,
#         text=prompt,
#         return_tensors="pd",
#     )
#     inputs.pop("token_type_ids", None)
#     print("input: ", inputs)
#     # 调试信息
#     print("=== processor type ===")
#     print(type(processor))
#     print("=== processor attrs ===")
#     print(dir(processor))
    
#     print("=== tok type ===")
#     print(type(tok))
    
#     print("=== model type ===")
#     print(type(model))
#     print("=== model config ===")
#     print(model.config)
    

#     eos = tok.eos_token_id
#     pad = tok.pad_token_id if tok.pad_token_id is not None else eos

#     with paddle.no_grad():
#         generated = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             eos_token_id=eos,
#             pad_token_id=pad,
#         )

#     new_tokens = generated[0][0]
#     print("new_tokens", new_tokens)
#     out = tok.decode(new_tokens.numpy().tolist(), skip_special_tokens=True)
#     return out.strip()


# print(glm_ocr_recognize(model, processor, tok, IMG_PATH))

import requests
from io import BytesIO
import time
import paddle
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor, AutoTokenizer

# model_path = "/home/work/zkx_test/PaddleFormers/PaddleOCR-VL-SFT-Bengali_full"
# model_path = "/home/work/zkx_test/PaddleFormers/GlmOCR-SFT-Bengali-lora/export"
model_path = "/home/work/zkx_test/PaddleFormers/GlmOCR-SFT-Bengali-lora-1/export"
processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForConditionalGeneration.from_pretrained(
    model_path, dtype="float16",
).eval()
# print(type(model))
# print(type(tok))
# print(type(processor))

image_url = "https://paddle-model-ecology.bj.bcebos.com/PPOCRVL/dataset/bengali_sft/5b/7a/5b7a5c1c-207a-4924-b5f3-82890dc7b94a.png"
image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
task = "ocr"

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": PROMPTS[task]},
]}]

prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pd")

inputs.pop("token_type_ids", None)
print(inputs)
print(inputs["input_ids"].shape)
eos = [59246, 59253]
pad = tok.eos_token_id
# pad = tok.pad_token_id if tok.pad_token_id is not None else eos
# print("eos_token_id:", tok.eos_token_id)
# print("pad_token_id:", tok.pad_token_id)



paddle.device.synchronize()
t0 = time.time()
with paddle.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        eos_token_id=eos,
        pad_token_id=pad,    )
paddle.device.synchronize()
print(f"耗时: {time.time() - t0:.2f}s")

output_ids = outputs[0][0].numpy().tolist()
print(output_ids)
output_text = tok.decode(output_ids, skip_special_tokens=True)
print(output_text)


# # GT = নট চলল রফযনর পঠ সওযর\nহয গলয গলয ভব এখন দটত, মঝ মঝ খবর নয যদও লগ যয\nঝগড\nদরগর কছ চল এল
# # Excepted Answer = নট চলল রফযনর পঠ সওযর\nহয গলয গলয ভব এখন দটত, মঝ মঝ খবর নয যদও লগ যয\nঝগড\nদরগর কছ চল এল、
#                     নট চলল রফযনর পঠ সওযর\nহয গলয গলয ভব এখন দটত, মঝ মঝ খবর নয যদও লগ যয\nঝগড\nদরগর কছ চল এল