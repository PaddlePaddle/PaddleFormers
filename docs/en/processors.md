### ⚙️ Processors

`Processor` is a multimodal preprocessing tool responsible for preparing inputs that combine more than one modality (like text, images). It provides a unified interface for different transformations, such as tokenizing text and resizing/normalizing images, and supports returning outpus in Paddle Tensor.

For example, [Qwen2.5-VL](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/paddleformers/transformers/qwen2_5_vl/processor.py) is a vision-language model that uses the [Qwen2-VL](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/paddleformers/transformers/qwen2_vl/image_processor.py) image processor and the [Qwen2](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/paddleformers/transformers/qwen2/tokenizer.py) tokenizer. A `ProcessorMixin` class wraps both of these, providing a single class for the model.


Using the `[~ProcessorMixin.from_pretrained]` method, you can easily load the processor configuration associated with a pretrained model (e.g., target image size, tokenization vocabulary). The Processor automatically loads all necessary configuration files (like `processor_config.json`, `preprocessor_config.json`, `tokenizer_config.json`, etc.) from the model directory to ensure the preprocessing steps are identical to those used during model training or inference.

The method supports loading from a **local directory** or **multiple download sources**:
- [huggingface](https://huggingface.co) (**Default**)
- [modelscope](https://modelscope.cn/home)
- [aistudio](https://aistudio.baidu.com/overview)


### 💻 Usage Example

Here’s how to load an `Processor` and process image/video data with [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).


- Case 1: Processing image and text directly:

```python
from paddleformers.transformers import AutoProcessor
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

text = "Describe this image."
url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

inputs = processor(text=text, images=image, return_tensors="pd")   # return Paddle Tensor
```

- Case 2: Handling conversational inputs (chat-formatted messages[image]):

```python

from paddleformers.transformers import AutoProcessor
from paddleformers.transformers import process_vision_info  # Processing functions for QwenVL models

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
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
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)
```

- Case 3: Handling conversational inputs (chat-formatted messages[video]):

```python

from paddleformers.transformers import AutoProcessor
from paddleformers.transformers import process_vision_info  # Processing functions for QwenVL models

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "http://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_video/example_video.mp4",
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages, video_backend="paddlecodec")    # load_video backend support: ["paddlecodec", "decord"], default: "paddlecodec"
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)
```


> **How to change the download source?**
>
> You can specify the model download source in two ways:
>
> - Via the `download_hub` parameter, passed directly in the `from_pretrained` method.
>
> ```python
> processor = AutoProcessor.from_pretrained(
>     "Qwen/Qwen2.5-VL-3B-Instruct",
>     download_hub="modelscope"
> )
> ```
>
> - Via the `DOWNLOAD_SOURCE` environment variable, to change the default download source.
> ```bash
> export DOWNLOAD_SOURCE=aistudio
> ```
