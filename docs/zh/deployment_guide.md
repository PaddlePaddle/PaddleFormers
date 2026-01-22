# 1. 引言

当模型训练完成，用于推理时，需要基于高效的推理引擎进行部署，以满足 低时延 / 高吞吐 等需求。基于 PaddleFormers 训练完成的模型可以直接使用 vLLM 和 FastDeploy 等工具推理。本文档介绍如何使用 vLLM 和 FastDeploy 部署大语言模型，实现高性能、低延迟的在线推理服务。

# 2. vLLM 使用指南

**vLLM** 是一个快速且易于使用的大语言模型推理部署库。有以下优点：

* 显著提升吞吐和显存效率
* 通过 PagedAttention 高效管理 key 和 value 内存占用
* 支持连续批处理（continuous batching）

## 2.1. 环境准备

### 2.1.1 环境要求

请参考 [vllm 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/) 中的环境要求

### 2.1.2 依赖安装

|Mandatory|Minimum|Recommend|
|-|-|-|
|vllm|0.13.0|0.13.0|

```shell
pip install vllm>=0.13.0
```

更多安装方式请参考 [vllm 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/)

## 2.2. 启动服务

当使用 PaddleFormers 训练完成模型后，模型权重会保存于训练配置中的 output_dir 中，请将该目录作为 `--model` 指定的路径。（LoRA 训练需要指定为模型合并参数后保存的路径）

```shell
python -m vllm.entrypoints.openai.api_server \
       --model {模型保存地址} \
       --port 8180 \
       --trust-remote-code \
       --max-model-len 4096 \
       --max-num-seqs 32 \
       --served-model-name {自定义名称}
```

## 2.3. 发起服务请求

通过如下命令发起服务请求。其中 model 需要传入启动服务时设置的名称 {served-model-name}

```shell
curl http://localhost:8180/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "{served-model-name}",
        "messages": [
            {"role": "user", "content": "把李白的静夜思改写为现代诗"}
        ]
    }'
```

vLLM 服务接口兼容 OpenAI 协议，可以通过如下 Python 代码发起服务请求。

```python
from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8180/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="{served-model-name}",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "把李白的静夜思改写为现代诗"},
    ],
)
print("Chat response:", chat_response)
```

# 3. FastDeploy 使用指南

**FastDeploy** 是基于飞桨（PaddlePaddle）的大语言模型与视觉语言模型推理部署工具包。有以下优点：

* **负载均衡式 PD 分解**：工业级解决方案，支持上下文缓存与动态实例角色切换，在保障 SLO 达标和吞吐量的同时优化资源利用率
* **统一 KV 缓存传输**：轻量级高性能传输库，支持智能 NVLink/RDMA 选择
* **OpenAI API 协议兼容**：服务化部署支持 OpenAI 协议调用
* **全量化格式支持**：W8A16、W8A8、W4A16、W4A8、W2A16、FP8等
* **丰富的加速策略**：推测解码、多令牌预测（MTP）及分块预填充
* **多硬件支持**：NVIDIA GPU、昆仑芯 XPU、海光 DCU、天数智芯 GPU、燧原 GCU、沐曦 GPU、英特尔 Gaudi 等

## 3.1. 环境准备

### 3.1.1. 环境要求

请参考 [FastDeploy 安装文档](https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs/get_started/installation) 的环境要求

### 3.1.2. 依赖安装

|Mandatory|Minimum|Recommend|
|-|-|-|
|FastDeploy|2.3.2|2.4.0|

为避免环境配置问题并简化安装流程，推荐使用官方镜像创建容器使用

```shell
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-cuda-12.6:2.3.2
```

更多安装方式请参考 [FastDeploy 安装文档](https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs/get_started/installation)

## 3.2. 启动服务

当使用 PaddleFormers 训练完成模型后，模型权重会保存于训练配置中的 output_dir 中，请将该目录作为 `--model` 指定的路径。（LoRA 训练需要指定为模型合并参数后保存的路径）

```shell
python -m fastdeploy.entrypoints.openai.api_server \
       --model {模型保存地址} \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 4096 \
       --max-num-seqs 32
```

## 3.2. 发起服务请求

执行启动服务指令后，当终端打印如下信息，说明服务已经启动成功。

```log
api_server.py[line:91] Launching metrics service at http://0.0.0.0:8181/metrics
api_server.py[line:94] Launching chat completion service at http://0.0.0.0:8180/v1/chat/completions
api_server.py[line:97] Launching completion service at http://0.0.0.0:8180/v1/completions
INFO:     Started server process [13909]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8180 (Press CTRL+C to quit)
```

FastDeploy 提供服务探活接口，用以判断服务的启动状态，执行如下命令返回 `HTTP/1.1 200 OK` 即表示服务启动成功。

```shell
curl -i http://0.0.0.0:8180/health
```

通过如下命令发起服务请求。

```shell
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "把李白的静夜思改写为现代诗"}
  ]
}'
```

FastDeploy 服务接口兼容 OpenAI 协议，可以通过如下 Python 代码发起服务请求。

```python
import openai
host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "把李白的静夜思改写为现代诗"},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```
