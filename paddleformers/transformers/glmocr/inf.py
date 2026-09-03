from .processor import Qwen2VLProcessor
from .image_processor import Glm46VImageProcessor
from paddleformers.transformers import Qwen2Tokenizer
from PIL import Image

image_processor = Glm46VImageProcessor()

# Load the tokenizer object first, don't pass the path directly
tokenizer = Qwen2Tokenizer.from_pretrained('/home/work/zkx_test/glm_ocr_hf/')

processor = Qwen2VLProcessor(image_processor=image_processor, tokenizer=tokenizer)

img = Image.open("/home/work/zkx_test/glmocr/handwritten.png")

result = processor(
    images=[img],
    text=["描述这张图片：<|image|>"],
    return_tensors="pd",
)
print(result["input_ids"])
print(result["pixel_values"])
print(result["input_ids"].shape)
print(result["pixel_values"].shape)