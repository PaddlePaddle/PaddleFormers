# 1. 安装

**环境依赖**

|Chip type|Driver version|
|-|-|
|BI150|4.3.8|

* **机器：** BI150/BI150s 64GB 8-card machine
* **镜像：** ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:3.3.0
* **GCC path：** /usr/bin/gcc (9.4)
* **python version：** 3.10

要验证 Iluvatar GPU 是否正常，可以使用`ixsmi`命令

```shell
ixsmi
#example：$ ixsmi
Timestamp    Thu Jul 10 16:59:37 2025
+-----------------------------------------------------------------------------+
|  IX-ML: 4.3.0       Driver Version: 4.3.0       CUDA Version: 10.2          |
|-------------------------------+----------------------+----------------------|
| GPU  Name                     | Bus-Id               | Clock-SM  Clock-Mem  |
| Fan  Temp  Perf  Pwr:Usage/Cap|      Memory-Usage    | GPU-Util  Compute M. |
|===============================+======================+======================|
| 0    Iluvatar BI-V150         | 00000000:13:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 1    Iluvatar BI-V150         | 00000000:16:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    103W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 2    Iluvatar BI-V150         | 00000000:1C:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 3    Iluvatar BI-V150         | 00000000:1F:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    106W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 4    Iluvatar BI-V150         | 00000000:27:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 5    Iluvatar BI-V150         | 00000000:2A:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    105W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 6    Iluvatar BI-V150         | 00000000:34:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 7    Iluvatar BI-V150         | 00000000:37:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    106W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 8    Iluvatar BI-V150         | 00000000:3D:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 9    Iluvatar BI-V150         | 00000000:40:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    107W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 10   Iluvatar BI-V150         | 00000000:48:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 11   Iluvatar BI-V150         | 00000000:4B:00.0     | 1500MHz   1600MHz    |
| N/A  33C   P0    103W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 12   Iluvatar BI-V150         | 00000000:54:00.0     | 1500MHz   1600MHz    |
| N/A  34C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 13   Iluvatar BI-V150         | 00000000:57:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    104W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 14   Iluvatar BI-V150         | 00000000:64:00.0     | 1500MHz   1600MHz    |
| N/A  35C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 15   Iluvatar BI-V150         | 00000000:67:00.0     | 1500MHz   1600MHz    |
| N/A  36C   P0    107W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU        PID      Process name                                Usage(MiB) |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

**安装依赖**

1. 拉取镜像

```shell
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:3.3.0
```

2. 启动 docker

```shell
docker run -itd --name paddleformers-ixuca --network host -v /usr/src:/usr/src -v /lib/modules:/lib/modules -v /dev:/dev -v /home:/home -v /data:/data --privileged --cap-add=ALL --pid=host ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:3.3.0
docker exec -it paddleformers-ixuca bash
```

3. 安装 PaddlePaddle

```shell
# 推荐使用 Paddle 3.3.0 版本
python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install paddle-iluvatar-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/ixuca/
# 如果想用 nightly 的 PaddlePaddle：
python -m pip install  --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -m pip install --pre paddle-iluvatar-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/ixuca/
```

4. 安装 PaddleFormers

```shell
git clone https://github.com/PaddlePaddle/PaddleFormers.git
cd PaddleFormers
python -m pip install -e .
```

# 2. 开始训练

所有示例脚本都位于 examples/config/iluvatar 下。下面以 ERNIE-4.5-21B-A3B-PT 的 SFT 为例。首先需要通过 Huggingface 下载模型：

```shell
hf download baidu/ERNIE-4.5-21B-A3B-PT --local-dir baidu/ERNIE-4.5-21B-A3B-PT
```

如果需要对 ERNIE-4.5-21B-A3B-PT 进行全参数 SFT：

```shell
bash examples/config/iluvatar/ERNIE-4.5-21B-A3B-PT/sft/run_full_8k.sh
```
如果需要对 ERNIE-4.5-21B-A3B-PT 进行基于 LoRA 的 SFT：

```shell
bash examples/config/iluvatar/ERNIE-4.5-21B-A3B-PT/sft/run_lora_8k.sh
```

训练产物位于 checkpoints 下，如果要使用 FastDeploy 进行推理，请先参考[https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/iluvatar_gpu.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/iluvatar_gpu.md) 安装 FastDeploy。

加载 ERNIE-4.5-21B-A3B-PT 全参数 SFT 后的权重进行推理：

```shell
python3 -m fastdeploy.entrypoints.openai.api_server --model checkpoints/ernie-21B-sft-full-tp-pp/ --port 8180 --tensor-parallel-size 1 --quantization wint8 --max-model-len 32768 --max-num-seqs 8 --block-size 16
```
加载 ERNIE-4.5-21B-A3B-PT SFT-LoRA 后的权重进行推理：

```shell
# 导出权重
bash examples/config/iluvatar/ERNIE-4.5-21B-A3B-PT/sft/run_full_export.sh
# 进行推理
python3 -m fastdeploy.entrypoints.openai.api_server --model checkpoints/ernie-sft-lora-tp-pp/export/ --port 8180 --tensor-parallel-size 1 --quantization wint8 --max-model-len 32768 --max-num-seqs 8 --block-size 16
```
