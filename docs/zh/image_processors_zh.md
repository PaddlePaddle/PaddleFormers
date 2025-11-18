### 🏞️ Image Processor

`Image Processor`是一个图像预处理工具，负责为视觉或多模态模型准备输入特征。它提供多种变换操作，例如调整大小和归一化，并支持输出 Paddle 张量。

Image Processor 充当原始图像数据与模型之间的桥梁，确保输入特征针对 VLM（Vision-Language Model）进行了优化。

使用 `[~BaseImageProcessor.from_pretrained]` 方法，可以轻松加载与预训练模型配套的处理器配置（如图像目标大小、是否归一化等）。加载时，Image Processor 会自动读取模型目录下的 `preprocessor_config.json`配置文件，以确保预处理步骤与模型训练或推理时时完全一致。

该方法支持从**本地目录**或**多个下载源**加载：
- [huggingface](https://huggingface.co) (**默认**)
- [modelscope](https://modelscope.cn/home)
- [aistudio](https://aistudio.baidu.com/overview)


### 💻 使用示例

下面是一个示例，展示如何加载 `Image Processor` 并处理图像数据（以[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)模型为例）。

```python
from paddleformers.transformers import AutoImageProcessor
from PIL import Image
import requests

image_processor = AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

url = "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

processed_image = image_processor(image, return_tensors="pd")   # 返回Paddle张量
```


> **如何更改下载源？**
>
> 通过两种方式指定模型下载源：
>
> - 通过`download_hub`参数，在`from_pretrained`方法中指定下载源。
>
> ```python
> image_processor = AutoImageProcessor.from_pretrained(
>     "Qwen/Qwen2.5-VL-3B-Instruct",
>     download_hub="modelscope"
> )
> ```
>
> - 通过`DOWNLOAD_SOURCE`环境变量，更改默认下载源。
> ```bash
> export DOWNLOAD_SOURCE=aistudio
> ```
