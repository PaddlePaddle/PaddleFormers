# 1. 安装

**环境依赖**

|Chip type|Driver version|
|-|-|
|KunlunxinP800|5.0.21.21|

* **机器：** KunlunxinP800 96GB 8-card machine
* **镜像：** ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310
* **GCC path：**  /usr/bin/gcc (8.4)
* **python version：** 3.10

要验证 XPU 是否正常，可以使用`xpu_smi`命令

```shell
xpu_smi
#example：$ xpu_smi
Wed Jun 25 19:45:10 2025
+-----------------------------------------------------------------------------+
| XPU-SMI               Driver Version: 5.0.21.21    XPU-RT Version: 5.0.21   |
|-------------------------------+----------------------+----------------------+
| XPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | XPU-Util  Compute M. |
|                               |             L3-Usage |            SR-IOV M. |
|===============================+======================+======================|
|   0  P800 OAM           N/A   | 00000000:03:00.0 N/A |                    0 |
| N/A   37C  N/A     88W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  P800 OAM           N/A   | 00000000:05:00.0 N/A |                    0 |
| N/A   41C  N/A     90W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  P800 OAM           N/A   | 00000000:63:00.0 N/A |                    0 |
| N/A   36C  N/A     89W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  P800 OAM           N/A   | 00000000:65:00.0 N/A |                    0 |
| N/A   36C  N/A     89W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  P800 OAM           N/A   | 00000000:83:00.0 N/A |                    0 |
| N/A   40C  N/A     88W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  P800 OAM           N/A   | 00000000:85:00.0 N/A |                    0 |
| N/A   40C  N/A     90W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  P800 OAM           N/A   | 00000000:A3:00.0 N/A |                    0 |
| N/A   39C  N/A     90W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  P800 OAM           N/A   | 00000000:A5:00.0 N/A |                    0 |
| N/A   40C  N/A     87W / 400W |      0MiB / 98304MiB |      0%      Default |
|                               |      0MiB /    96MiB |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  XPU   XI   CI        PID   Type   Process name                  XPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

**安装依赖**

1. 拉取镜像

```shell
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310
```

2. 启动 docker

```shell
# Recommended: Map your project directory and a dataset directory
# Replace pwd with the actual path on your host machine
docker run -it --privileged=true  --net host --shm-size '256gb' --device=/dev/xpu0:/dev/xpu0 --device=/dev/xpu1:/dev/xpu1 --device=/dev/xpu2:/dev/xpu2 --device=/dev/xpu3:/dev/xpu3 --device=/dev/xpu4:/dev/xpu4 --device=/dev/xpu5:/dev/xpu5 --device=/dev/xpu6:/dev/xpu6 --device=/dev/xpu7:/dev/xpu7 --device=/dev/xpuctrl:/dev/xpuctrl --name paddle-xpu-dev -v $(pwd):/work -w=/work -v xxx ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 /bin/bash
```

3. 安装 PaddlePaddle

```shell
# 推荐使用 Paddle 3.3.0 版本
python -m pip install https://paddle-qa.bj.bcebos.com/PaddleFormers/xpu_whl/paddlepaddle_xpu-3.3.0.dev20260122-cp310-cp310-linux_x86_64.whl
# 如果想用 nightly 的 PaddlePaddle：
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

4. 安装 PaddleFormers

```shell
git clone https://github.com/PaddlePaddle/PaddleFormers.git
cd PaddleFormers
python -m pip install -e .
```

# 2. 开始训练

所有示例脚本都位于 examples/config/xpu 下。下面以 ERNIE-4.5-21B-A3B-PT 的 SFT 为例。首先需要通过 Huggingface 下载模型：

```shell
hf download baidu/ERNIE-4.5-21B-A3B-PT --local-dir baidu/ERNIE-4.5-21B-A3B-PT
```

如果需要对 ERNIE-4.5-21B-A3B-PT 进行全参数 SFT：

```shell
paddleformers-cli train examples/config/xpu/ERNIE-4.5-21B-A3B/sft/full_32k.yaml
```
如果需要对 ERNIE-4.5-21B-A3B-PT 进行基于 LoRA 的 SFT：

```shell
bash train examples/config/xpu/ERNIE-4.5-21B-A3B/sft/run_lora_32k.sh
```

训练产物位于 checkpoints 下，如果要使用 FastDeploy 进行推理，请先参考[https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/kunlunxin_xpu.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/kunlunxin_xpu.md)
