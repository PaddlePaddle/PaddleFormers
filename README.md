<p align="center">
  <img src="https://github.com/user-attachments/assets/9d1c1937-7fac-48f8-9d61-f7ac67b61b18" align="middle"  width="500" />
</p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win-pink.svg"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleFormers/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleFormers?color=ccf"></a>
</p>

<h4 align="center">
    <a href=#最新更新> 最新更新 </a> |
    <a href=#特性> 特性 </a> |
    <a href=#安装> 安装 </a> |
    <a href=#快速体验> 快速体验 </a> |
    <a href=#社区交流> 社区交流 </a>
</h4>

# PaddleFormers
## 📝简介
PaddleFormers 是基于百度深度学习框架 PaddlePaddle 搭建的 Transformers 库，旨在为 PaddlePaddle 生态提供与 Hugging Face Transformers 项目对等的模型接口与功能体验，支持大语言模型（LLM）与视觉语言模型（VLM）的训练能力。PaddleFormers 充分发挥 PaddlePaddle 在高性能训练方面的内置优势，全面支持包括张量并行、流水线并行和专家并行在内的主流大模型分布式训练策略，以及自动混合精度等加速技术，在 DeepSeek-V3、GLM-4.5-Air 等重点模型上，训练性能明显超越 Megatron-LM ，实现了高效的预训练与后训练性能。

结合业界主流优化方法与飞桨在业务实践中积累的高效特性，PaddleFormers 致力于打造**高性能、低资源占用**的训练体验，帮助用户高效便捷地完成大模型训练，而无需关注底层复杂的优化细节。

## 🆕最新更新
* 2026.01.21 - PaddleFomers v1.0版本发布啦！我们提供了针对 LLM 和 VLM 等模型的训练能力，针对 DeepSeek-V3模型和 GLM-4.5-Air 等重点模型，我们实现了极致性能优化（训练性能明显超越 Megatron-LM ）。针对 PaddleOCR-VL，我们在昆仑芯 P800、天数天垓150等国产计算芯片上进行了适配，更好的满足国内用户需求。

## ✨特性
* **丰富的模型支持：** PaddleFormers 实现了对于 100+ 主流的大语言模型和视觉语言模型的训练能力支持，涵盖了 DeepSeek-V3、GLM-4.5系列、Qwen2和 Qwen3系列、Qwen3-VL 等前沿模型。同时提供了对 ERNIE-4.5、ERNIE-4.5-VL、PaddleOCR-VL 等文心系列模型完备的训练能力。
* **高性能组网实现：** 实现了 FP8低精度训练与高性能算子优化、通信计算重叠优化、精细化存算均衡等策略，大幅提升大模型训练的计算、通信和存储效率。在 DeepSeek-V3、GLM-4.5-Air 等模型上，训练性能明显超越 Megatron-LM。
* **全流程能力支持：** PaddleFormers 实现了从预训练到后训练的全流程训练能力支持，其中后训练支持 CPT / SFT / SFT-LoRA / DPO / DPO-LoRA 等主流能力，帮助用户高效、便捷地完成大模型的迭代与优化。PaddleFormers 还实现了对 Safetensors 格式的 **全面支持** ，训练完成的模型，其存储格式与 Hugging Face 上托管的权重格式一致，可以在任意支持该格式的框架或工具中使用（如 FastDeploy / vLLM / SGLang 等）。
* **完备的训练能力支持：** PaddleFormers 实现了对于 **Function Call** 、 **Thinking**​ 等大模型前沿能力的训练支持，并通过 **Data Packing** 、 **Padding Free**​ 等数据流技术显著优化训练性能。
* **国产芯片深度适配：** 支持昆仑芯 P800、天数天垓150、沐曦 C550等国产计算平台，基于128卡昆仑芯 P800支持 DeepSeek V3的 SFT，成为最少国产算力资源后训练方案。

## 📋模型列表

<table border="1" cellpadding="8" cellspacing="0" style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="text-align: left;">模型类型</th>
      <th style="text-align: left;">模型系列</th>
      <th style="text-align: left;">模型名称</th>
      <th style="text-align: left;">Chat Template</th>
    </tr>
  </thead>
  <tbody>
    <!-- LLM 分类 - 跨行合并开始 -->
    <tr>
      <td rowspan="10" style="vertical-align: top;">LLM</td>
      <td>DeepSeekv3</td>
      <td>deepseek-ai/DeepSeek-V3-Base、deepseek-ai/DeepSeek-V3、deepseek-ai/DeepSeek-V3-0324</td>
      <td>deepseek3</td>
    </tr>
    <tr>
      <td>🏛️ERNIE-4.5</td>
      <td>baidu/ERNIE-4.5-0.3B-Base-PT、baidu/ERNIE-4.5-0.3B-PT、baidu/ERNIE-4.5-21B-A3B-Base-PT、baidu/ERNIE-4.5-21B-A3B-PT、baidu/ERNIE-4.5-300B-A47B-Base-PT、baidu/ERNIE-4.5-300B-A47B-PT、baidu/ERNIE-4.5-21B-A3B-Thinking</td>
      <td>ernie、ernie_nothink</td>
    </tr>
    <tr>
      <td>gemma3</td>
      <td>google/gemma-3-270m、google/gemma-3-270m-it、google/gemma-3-1b-pt、google/gemma-3-1b-it、google/gemma-3-4b-pt、google/gemma-3-4b-it、google/gemma-3-12b-pt、google/gemma-3-12b-it、google/gemma-3-27b-pt、google/gemma-3-27b-it</td>
      <td>gemma</td>
    </tr>
    <tr>
      <td>GLM-4.5</td>
      <td>zai-org/GLM-4.5-Air-Base、zai-org/GLM-4.5-Air、zai-org/GLM-4.5-Base、zai-org/GLM-4.5</td>
      <td>glm4_moe</td>
    </tr>
    <tr>
      <td>gpt-oss</td>
      <td>openai/gpt-oss-20b、openai/gpt-oss-120b</td>
      <td>gpt</td>
    </tr>
    <tr>
      <td>Llama-3</td>
      <td>meta-llama/Meta-Llama-3-8B、meta-llama/Meta-Llama-3-8B-Instruct、meta-llama/Meta-Llama-3-70B、meta-llama/Meta-Llama-3-70B-Instruct、meta-llama/Llama-3.1-8B、meta-llama/Llama-3.1-8B-Instruct、meta-llama/Llama-3.1-70B、meta-llama/Llama-3.1-70B-Instruct、meta-llama/Llama-3.1-405B、meta-llama/Llama-3.1-405B-Instruct、meta-llama/Llama-3.2-1B、meta-llama/Llama-3.2-1B-Instruct、meta-llama/Llama-3.2-3B、meta-llama/Llama-3.2-3B-Instruct、meta-llama/Llama-3.3-70B-Instruct</td>
      <td>llama3</td>
    </tr>
    <tr>
      <td>phi-4</td>
      <td>microsoft/phi-4</td>
      <td>phi4</td>
    </tr>
    <tr>
      <td>Qwen2</td>
      <td>Qwen/Qwen2-0.5B、Qwen/Qwen2-0.5B-Instruct、Qwen/Qwen2-1.5B、Qwen/Qwen2-1.5B-Instruct、Qwen/Qwen2-7B、Qwen/Qwen2-7B-Instruct、Qwen/Qwen2-57B-A14B、Qwen/Qwen2-57B-A14B-Instruct、Qwen/Qwen2-72B、Qwen/Qwen2-0.5B-Instruct</td>
      <td>qwen</td>
    </tr>
    <tr>
      <td>Qwen3</td>
      <td>Qwen/Qwen3-0.6B-Base、Qwen/Qwen3-0.6B、Qwen/Qwen3-1.7B-Base、Qwen/Qwen3-1.7B、Qwen/Qwen3-4B-Base、Qwen/Qwen3-4B、Qwen/Qwen3-4B-Instruct-2507、Qwen/Qwen3-4B-Thinking-2507、Qwen/Qwen3-8B-Base、Qwen/Qwen3-8B、Qwen/Qwen3-14B-Base、Qwen/Qwen3-14B、Qwen/Qwen3-32B、Qwen/Qwen3-30B-A3B-Base、Qwen/Qwen3-30B-A3B、Qwen/Qwen3-30B-A3B-Instruct-2507、Qwen/Qwen3-30B-A3B-Thinking-2507、Qwen/Qwen3-235B-A22B、Qwen/Qwen3-235B-A22B-Instruct-2507、Qwen/Qwen3-235B-A22B-Thinking-2507</td>
      <td>qwen3、qwen3_nothink</td>
    </tr>
    <tr>
      <td>Qwen3-Next</td>
      <td>Qwen/Qwen3-Next-80B-A3B-Instruct、Qwen/Qwen3-Next-80B-A3B-Thinking</td>
      <td>qwen3、qwen3_nothink</td>
    </tr>
    <!-- VLM 分类 - 跨行合并开始 -->
    <tr>
      <td rowspan="4" style="vertical-align: top;">VLM</td>
      <td>🏛️ERNIE-4.5-VL</td>
      <td>baidu/ERNIE-4.5-VL-28B-A3B-Base-PT、baidu/ERNIE-4.5-VL-28B-A3B-PT、baidu/ERNIE-4.5-VL-424B-A47B-Base-PT、baidu/ERNIE-4.5-VL-424B-A47B-PT、baidu/ERNIE-4.5-VL-28B-A3B-Thinking</td>
      <td>ernie_vl、ernie_vl_nothink</td>
    </tr>
    <tr>
      <td>🏛️PaddleOCR-VL</td>
      <td>PaddlePaddle/PaddleOCR-VL</td>
      <td>paddleocr_vl</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL</td>
      <td>Qwen/Qwen2.5-VL-3B-Instruct、Qwen/Qwen2.5-VL-7B-Instruct、Qwen/Qwen2.5-VL-32B-Instruct、Qwen/Qwen2.5-VL-72B-Instruct</td>
      <td>qwen2_vl</td>
    </tr>
    <tr>
      <td>Qwen3-VL</td>
      <td>Qwen/Qwen3-VL-2B-Instruct、Qwen/Qwen3-VL-2B-Thinking、Qwen/Qwen3-VL-4B-Instruct、Qwen/Qwen3-VL-4B-Thinking、Qwen/Qwen3-VL-8B-Instruct、Qwen/Qwen3-VL-8B-Thinking、Qwen/Qwen3-VL-32B-Instruct、Qwen/Qwen3-VL-32B-Thinking、Qwen/Qwen3-VL-30B-A3B-Instruct、Qwen/Qwen3-VL-30B-A3B-Thinking、Qwen/Qwen3-VL-235B-A22B-Instruct、Qwen/Qwen3-VL-235B-A22B-Thinking</td>
      <td>qwen3_vl、qwen3_vl_nothink</td>
    </tr>
  </tbody>
</table>

* 更多关于模型训练能力的支持细节，请参考：[PaddleFormers 模型能力矩阵](./docs/zh/model_capability.md)
* 带有🏛️标签的模型是 PaddleFormers 官方维护的模型

## 💾安装
**环境依赖**

* python ≥ 3.10
* CUDA ≥ 12.0
* PaddleFleet ≥ 0.1（仅为 GPU 训练功能依赖）

**安装依赖（GPU）**

<details>
  <summary>基于 Docker 容器的方式（<b>推荐</b>）</summary>

------
> 为了避免本地环境存在较多冲突，我们建议使用 PaddleFormers 的预置镜像来准备环境，容器中已经拉取了 PaddleFormers 仓库并完成了安装：
>
> ```shell
> # 以cuda12.6为例
> docker run --gpus all --name paddleformers-work -v $(pwd):/work  \
>     -w=/work --shm-size=512G --network=host -it \
>     ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5 /bin/bash
>
> # cuda12.9镜像：ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.3.0-gpu-cuda12.9-cudnn9.9
> # cuda13.0镜像：ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.3.0-gpu-cuda13.0-cudnn9.13
> ```
------

</details>

<details>
  <summary>基于 pip/源码的安装方式</summary>

------
> 我们推荐使用 `conda` / `venv` / `uv` 等虚拟环境工具管理 python 环境。
>
> ```shell
> # conda
> conda create -n paddleformers-work python=3.10 #支持python3.10～3.13
> conda activate paddleformers-work
> # venv
> python -m venv .paddleformers-work
> source .paddleformers-work/bin/activate
> # uv
> uv venv .paddleformers-work
> source .paddleformers-work/bin/activate
> ```
------
> **安装方案一：** 拉取源码安装
>
> ```shell
> # Install development version
> git clone https://github.com/PaddlePaddle/PaddleFormers.git
> cd PaddleFormers
> # cuda12.6
> python -m pip install -e '.[paddlefleet]' --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
> # cuda12.9
> # python -m pip install -e '.[paddlefleet]' --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
> # cuda13.0
> # python -m pip install -e '.[paddlefleet]' --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/
> ```
------
> **安装方案二：** 如果您不想拉取源码，可以基于下面的命令安装 PaddleFormers 和 PaddleFleet。
>
> ```shell
> # Install via pip
> # cuda12.6
> python -m pip install paddleformers[paddlefleet] --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
> # cuda12.9
> # python -m pip install paddleformers[paddlefleet] --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
> # cuda13.0
> # python -m pip install paddleformers[paddlefleet] --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/
> ```
------
> **安装方案三：** 如果您只需使用 tokenizer 或者 processor，可以通过以下命令安装，这种情况下不会安装训练相关的依赖，安装速度更加快。
>
> ```shell
> python -m pip install paddleformers
> ```
------

</details>

 **安装依赖（XPU & ILUVATAR-GPU & Metax GPU）**

* [昆仑芯安装说明文档](./docs/zh/XPU_installation_guide.md)
* [天数智芯安装说明文档](./docs/zh/ILUVATAR-GPU_installation_guide.md)
* [沐曦安装说明文档](./docs/zh/Metax-GPU_installation_guide.md)

# ⚡快速体验

PaddleFormers 在 API 设计上与 Hugging Face Transformers 保持了高度一致，使用示例如下：

**使用 tokenizer**

```python
from paddleformers.transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
print(tokenizer.encode("中华人民共和国"))
# 中华人民共和国将会被编码为两个token：
# [105492, 104773]
```

**文本生成**

```python
from paddleformers.transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", dtype="bfloat16").eval()

input_features = tokenizer("请给我一段大模型的简短介绍：", return_tensors="pd")
outputs = model.generate(**input_features, max_new_tokens=128)

print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
```

**模型训练**

```shell
paddleformers-cli train ./examples/config/sft/full.yaml
```

## 📊数据处理
* [数据集格式说明](./docs/zh/dataset_format.md)
* [Chat Template 说明](./docs/zh/chat_template_guide.md)
* [数据流参数说明](./docs/zh/data_processing_guide.md)

## 🚀模型训练 & 部署
* [PaddleFormers 命令行工具](./docs/zh/cli_usage.md)
* [训练参数配置说明](./docs/zh/training_arguments.md)
* [基于 PaddleFormers 进行模型预训练/后预训练](./docs/zh/pt_and_cpt_guide.md)
* [基于 PaddleFormers 进行指令微调（SFT & LoRA）](./docs/zh/sft_and_lora_guide.md)
* [基于 PaddleFormers 进行偏好对齐（DPO & LoRA）](./docs/zh/dpo_and_lora_guide.md)
* [基于 FastDeploy / vLLM 部署模型](./docs/zh/deployment_guide.md)

## 💻多硬件使用
* [昆仑芯使用说明文档](./docs/zh/XPU_usage_guide.md)
* [天数智芯使用说明文档](./docs/zh/ILUVATAR-GPU_usage_guide.md)
* [沐曦使用说明文档](./docs/zh/Metax-GPU_usage_guide.md)

## 🔍最佳实践
* [基于 DeepSeekv3的高效预训练](./examples/best_practices/DeepSeek-V3/)
* [基于 ERNIE-4.5的高效预训练](./examples/best_practices/ERNIE-4.5/)
* [训练一个偏好 Emoji 输出的对齐模型](./examples/best_practices/tutorials/how_to_train_an_emoji_model.md)
* [训练一个支持思考能力的模型](./examples/best_practices/tutorials/how_to_train_a_reasoning_model.md)
* [训练一个支持 Function Call 能力的模型](./examples/best_practices/tutorials/how_to_train_a_function_call_model.md)
* [基于 PaddleOCR-VL 微调实现孟加拉语识别能力](./examples/best_practices/PaddleOCR-VL/)
* [训练一个支持 Grounding 的模型](./examples/best_practices/tutorials/how_to_train_a_visual_grounding_model.md)

## ➕其他
* [如何下载模型](./docs/zh/how_to_download_model.md)
* [常见问题处理](https://github.com/PaddlePaddle/PaddleFormers/issues/3699)

## 💬社区相关

**贡献代码**

* 欢迎社区用户为 PaddleFormers 贡献代码，详情请参考 [贡献指南](CONTRIBUTING.md)。

**和我们交流**

* 微信扫描二维码并填写问卷，即可加入交流群与众多社区开发者以及官方团队深度交流.

<div align="center">
  <img src="https://github.com/user-attachments/assets/9f0a736c-b047-4912-a70f-8b1ea772c3eb" width="300" alt="qrcode">
</div>

## 🙏致谢
我们借鉴了 Hugging Face 的[Transformers](https://github.com/huggingface/transformers)🤗关于预训练模型使用的优秀设计，在此对 Hugging Face 作者及其开源社区表示感谢。

## 📜许可证
PaddleFormers 遵循[Apache-2.0开源协议](LICENSE)。
