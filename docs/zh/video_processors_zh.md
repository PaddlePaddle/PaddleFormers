### 🎬 Video Processor

`Video Processor`是一个视频预处理工具，负责为多模模型准备输入特征，并处理其输出。它提供各种转换，例如调整大小、归一化等功能，同时支持输出 Paddle 张量。

Video Processor 是在图像处理器基础上扩展视频输入处理的功能，允许模型使用一套与图像不同的参数来处理视频。它充当原始视频数据和模型之间的桥梁，确保输入特征针对 VLM（Vision-Language Model）进行了优化。

使用 `[~BaseVideoProcessor.from_pretrained]` 方法，可以轻松加载与预训练模型配套的处理器配置（如视频帧目标大小、是否归一化等）。加载时，Video Processor 会自动读取模型目录下的 `video_preprocessor_config.json` 或 `preprocessor_config.json` 配置文件，以确保预处理步骤与模型调用时完全一致。

该方法支持从**本地目录**或**多个下载源**加载：
- [huggingface](https://huggingface.co) (**默认**)
- [modelscope](https://modelscope.cn/home)
- [aistudio](https://aistudio.baidu.com/overview)


### 💻 使用示例

下面是一个示例，展示如何加载 `Video Processor` 并处理视频数据（以[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)模型为例）。

```python
from paddleformers.transformers import AutoVideoProcessor
from paddleformers.transformers.video_utils import load_video

video_processor = AutoVideoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

video = load_video("your_video.mp4")    # 或者视频URL
processed_video = video_processor(video[0], return_tensors="pd")    # 返回Paddle张量
```


> **如何更改下载源？**
>
> 通过两种方式指定模型下载源：
>
> - 通过`download_hub`参数，在`from_pretrained`方法中指定下载源。
>
> ```python
> video_processor = AutoVideoProcessor.from_pretrained(
>     "Qwen/Qwen2.5-VL-3B-Instruct",
>     download_hub="modelscope"
> )
> ```
>
> - 通过`DOWNLOAD_SOURCE`环境变量，更改默认下载源。
> ```bash
> export DOWNLOAD_SOURCE=aistudio
> ```
