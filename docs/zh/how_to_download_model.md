# 1. PaddleFormers 自动下载

当使用 PaddleFormers 训练时，无论是使用 API 接口，还是通过命令行工具中的配置文件，在指定了正确的`repo_id/model_id`后，都会自动下载模型文件到本地并缓存。

**配置文件**

```yaml
model_name_or_path: Qwen/Qwen3-0.6B-Base
```

**API 使用**

```python
from paddleformers.transformers import AutoTokenizer, AutoModelForCausalLM
# 将会自动下载Qwen3-0.6B-Base的tokenizer物料
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
# 将会自动下载Qwen3-0.6B-Base的模型物料
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype="bfloat16", convert_from_hf=True).eval()
```

考虑到不同用户网络连接的问题，PaddleFormers 支持以下三种下载源：

* [AIStudio](https://aistudio.baidu.com/modelsoverview)
* [ModelScope](https://www.modelscope.cn/models)
* [HuggingFace](https://huggingface.co/models)

使用自动下载时，默认从 HuggingFace 下载，用户可以通过配置环境变量 `DOWNLOAD_SOURCE` 修改下载源，可取值为 "aistudio", "modelscope", "huggingface"。例如：

```shell
export DOWNLOAD_SOURCE=aistudio # "aistudio", "modelscope" or "huggingface"
```

# 2. 手动下载

如果网络不稳定，自动下载可能会失败，您可以选择手动下载模型保存到本地文件夹中，在配置文件或者 API 中指定对应的路径即可

**配置文件**

```yaml
model_name_or_path: {模型保存路径}
```

**API 使用**

```python
from paddleformers.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("{模型保存路径}")
model = AutoModelForCausalLM.from_pretrained("{模型保存路径}", dtype="bfloat16", convert_from_hf=True).eval()
```

## 2.1. HuggingFace

使用命令行工具下载

```shell
# 首先请先安装huggingface_hub库
python -m pip install huggingface_hub

# 下载整个repo到指定目录
# 填写要下载的模型repo_id，在local_dir后指定下载路径，以下示例为下载到当前文件夹
hf download {repo_id} --local-dir ./

# 下载单个文件到指定目录（以下载README.md为例）
hf download {repo_id} README.md --local-dir ./
```

更多下载方式请参考 [HuggingFace 模型下载](https://huggingface.co/docs/hub/models-downloading)

## 2.2. AIStudio

使用命令行工具下载

注意：在 AIstudio 上，`repo_id` 前缀可能因模型发布者不同，和 HuggingFace 有所差异，在加载或引用模型时，请务必使用完整的、准确的 `repo_id`，以确保正确访问目标模型。您可以在模型详情页或分享链接中找到该标识符。

```shell
# 首先请先安装aistudio-sdk库
python -m pip install --upgrade aistudio-sdk

# 下载整个repo到指定目录
# 填写要下载的模型repo_id，在local_dir后指定下载路径，以下示例为下载到当前文件夹
aistudio download --model {repo_id} --local_dir ./

# 下载单个文件到指定目录（以下载README.md为例）
aistudio download README.md --model {repo_id} --local_dir ./
```

更多下载方式请参考 [AIStudio 模型下载](https://ai.baidu.com/ai-doc/AISTUDIO/zlisofwng)

## 2.3. ModelScope

使用命令行工具下载

注意：在 ModelScope 上，`repo_id` 前缀可能因模型发布者不同，和 HuggingFace 有所差异，在加载或引用模型时，请务必使用完整的、准确的 `repo_id`，以确保正确访问目标模型。您可以在模型详情页或分享链接中找到该标识符。

```shell
# 首先请先安装modelscope库
python -m pip install modelscope

# 下载整个repo到指定目录
# 填写要下载的模型repo_id，在local_dir后指定下载路径，以下示例为下载到当前文件夹
modelscope download --model {repo_id} --local_dir ./

# 下载单个文件到指定目录（以下载README.md为例）
modelscope download --model {repo_id} README.md --local_dir ./
```

更多下载方式请参考 [ModelScope 模型下载](https://www.modelscope.cn/docs/models/download)
