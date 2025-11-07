### 🎬 Video Processor

`Video Processor` is a video preprocessing tool responsible for preparing input features for multimodal models and processing their outputs. It provides various transformations, such as resizing and normalization, and supports returning outpus in Paddle Tensor.

The Video Processor extends the functionality of an image processor to handle video inputs, allowing models to process videos using a different set of parameters than images. It acts as a bridge between raw video data and the model, ensuring that input features are optimized for VLM (Vision-Language Model).

Using the `[~BaseVideoProcessor.from_pretrained]` method, you can easily load the processor configuration associated with a pretrained model (e.g., target image size, normalization settings). When loading, the Video Processor automatically reads the `video_preprocessor_config.json` or `preprocessor_config.json` file from the model directory to ensure the preprocessing steps are identical to those used during model inference.

The method supports loading from a **local directory** or **multiple download sources**:
- [huggingface](https://huggingface.co) (**Default**)
- [modelscope](https://modelscope.cn/home)
- [aistudio](https://aistudio.baidu.com/overview)


### 💻 Usage Example

Here’s how to load a`Video Processor` and process video data with [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

```python
from paddleformers.transformers import AutoVideoProcessor
from paddleformers.transformers.video_utils import load_video

video_processor = AutoVideoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

video = load_video("your_video.mp4")    # or video URL
processed_video = video_processor(video[0], return_tensors="pd")    # return Paddle Tensor
```


> **How to change the download source?**
>
> You can specify the model download source in two ways:
>
> - Via the `download_hub` parameter, passed directly in the `from_pretrained` method.。
>
> ```python
> video_processor = AutoVideoProcessor.from_pretrained(
>     "Qwen/Qwen2.5-VL-3B-Instruct",
>     download_hub="modelscope"
> )
> ```
>
> - Via the `DOWNLOAD_SOURCE` environment variable, to change the default download source.。
> ```bash
> export DOWNLOAD_SOURCE=aistudio
> ```
