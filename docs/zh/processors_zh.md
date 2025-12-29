### ⚙️ Processors

`Processor`是一种多模态预处理工具，用于为包含多种模态（如文本、图像等）的模型准备输入。它提供了一个统一的接口来执行不同类型的转换操作，例如文本分词、图像的调整大小与归一化，同时支持输出 Paddle 张量。

例如，[Qwen2.5-VL](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/paddleformers/transformers/qwen2_5_vl/processor.py)是一个视觉语言模型，它使用了[Qwen2-VL](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/paddleformers/transformers/qwen2_vl/image_processor.py) image processor 和[Qwen2](https://github.com/PaddlePaddle/PaddleFormers/blob/develop/paddleformers/transformers/qwen2/tokenizer.py) tokenizer. `ProcessorMixin` 类将这两者封装在一起，为模型提供一个统一的处理接口。

通过`[~ProcessorMixin.from_pretrained]`方法，您可以轻松加载与预训练模型关联的处理器配置（如目标图像尺寸、分词词表等）。会自动从模型目录中加载所有必要的配置文件（例如`processor_config.json`, `preprocessor_config.json`, `tokenizer_config.json`等），以确保预处理步骤与模型训练或推理阶段保持一致。

该方法支持从**本地目录**或**多个下载源**加载：
- [huggingface](https://huggingface.co) (**默认**)
- [modelscope](https://modelscope.cn/home)
- [aistudio](https://aistudio.baidu.com/overview)


### 💻 使用示例

下面是一个示例，展示如何加载 `Processor` 并处理图像/视频数据（以[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)模型为例）。

- 示例 1：直接处理图像与文本:

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

- 示例 2：处理对话式输入（聊天格式的多模态消息[image]）:

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

- 示例 3：处理对话式输入（聊天格式的多模态消息[video]）:

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
image_inputs, video_inputs = process_vision_info(messages, video_backend="paddlecodec")    # load_video backend支持: ["paddlecodec"、"decord"], 默认"paddlecodec"
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)
```


> **如何更改下载源？**
>
> 通过两种方式指定模型下载源：
>
> - 通过`download_hub`参数，在`from_pretrained`方法中指定下载源。
>
> ```python
> processor = AutoProcessor.from_pretrained(
>     "Qwen/Qwen2.5-VL-3B-Instruct",
>     download_hub="modelscope"
> )
> ```
>
> - 通过`DOWNLOAD_SOURCE`环境变量，更改默认下载源。
> ```bash
> export DOWNLOAD_SOURCE=aistudio
> ```
