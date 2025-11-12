### 🏞️ Image Processor

`Image Processor` is an image preprocessing tool responsible for preparing input features for vision or multimodal models. It provides various transformations, such as resizing and normalization, and supports returning outpus in Paddle Tensor.

It acts as a bridge between raw image data and the model, ensuring that input features are optimized for VLMs (Vision-Language Models.

Using the `[~BaseImageProcessor.from_pretrained]` method, you can easily load the processor configuration associated with a pretrained model (e.g., target image size, normalization settings). When loading, the Image Processor automatically reads the `preprocessor_config.json` file from the model directory to ensure the preprocessing steps are identical to those used during model trainging or inference.

The method supports loading from a **local directory** or **multiple download sources**:
- [huggingface](https://huggingface.co) (**Default**)
- [modelscope](https://modelscope.cn/home)
- [aistudio](https://aistudio.baidu.com/overview)


### 💻 Usage Example

Here’s how to load an `Image Processor` and process image data with [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

```python
from paddleformers.transformers import AutoImageProcessor
from PIL import Image
import requests

image_processor = AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

processed_image = image_processor(image, return_tensors="pd")   # return Paddle Tensor
```


> **How to change the download source?**
>
> You can specify the model download source in two ways:
>
> - Via the `download_hub` parameter, passed directly in the `from_pretrained` method.。
>
> ```python
> image_processor = AutoImageProcessor.from_pretrained(
>     "Qwen/Qwen2.5-VL-3B-Instruct",
>     download_hub="modelscope"
> )
> ```
>
> - Via the `DOWNLOAD_SOURCE` environment variable, to change the default download source.。
> ```bash
> export DOWNLOAD_SOURCE=aistudio
> ```
