# 1. 安装

**环境依赖**

|Chip type|Driver version|
|-|-|
|MetaX C550|2.15.9|

* **机器：** MetaX C550 64GB 8-card machine
* **镜像：** cr.metax-tech.com/public-ai-release/maca/paddle-metax:3.3.0-maca.ai3.3.0.10-py310-ubuntu22.04-amd64
* **GCC path：**  /usr/bin/gcc (9.4)
* **python version：** 3.10

要验证 Metax GPU 是否正常，可以使用`mx-smi`命令

```shell
mx-smi
#example：$ mx-smi
mx-smi  version: 2.2.4

=================== MetaX System Management Interface Log ===================
Timestamp                                         : Tue Jul 29 12:16:08 2025

Attached GPUs                                     : 8
+---------------------------------------------------------------------------------+
| MX-SMI 2.2.4                        Kernel Mode Driver Version: 2.15.9          |
| MACA Version: 2.32.0.6              BIOS Version: 1.25.1.0                      |
|------------------------------------+---------------------+----------------------+
| GPU         NAME                   | Bus-id              | GPU-Util             |
| Temp        Pwr:Usage/Cap          | Memory-Usage        | GPU-State            |
|====================================+=====================+======================|
| 0           MetaX C550             | 0000:0f:00.0        | 0%                   |
| 29C         53W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 1           MetaX C550             | 0000:34:00.0        | 0%                   |
| 29C         53W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 2           MetaX C550             | 0000:48:00.0        | 0%                   |
| 30C         54W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 3           MetaX C550             | 0000:5a:00.0        | 0%                   |
| 29C         54W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 4           MetaX C550             | 0000:87:00.0        | 0%                   |
| 30C         53W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 5           MetaX C550             | 0000:ae:00.0        | 0%                   |
| 32C         55W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 6           MetaX C550             | 0000:c2:00.0        | 0%                   |
| 32C         56W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+
| 7           MetaX C550             | 0000:d7:00.0        | 0%                   |
| 31C         56W / 450W             | 858/65536 MiB       | Available            |
+------------------------------------+---------------------+----------------------+

+---------------------------------------------------------------------------------+
| Process:                                                                        |
|  GPU                    PID         Process Name                 GPU Memory     |
|                                                                  Usage(MiB)     |
|=================================================================================|
|  no process found                                                               |
+---------------------------------------------------------------------------------+
```
**安装依赖**

1. 拉取镜像（以 release3.3为例）

```shell
docker pull cr.metax-tech.com/public-ai-release/maca/paddle-metax:3.3.0-maca.ai3.3.0.10-py310-ubuntu22.04-amd64
```

2. 启动 docker

```shell
docker run \
    -it \
    --device=/dev/dri \
    --device=/dev/mxcd \
    --device=/dev/infiniband \
    --group-add video \
    --name <container_name> \
    --network=host \
    --uts=host \
    --ipc=host \
    --privileged=true \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --shm-size '500gb' \
    --ulimit memlock=-1 \
    -v /sw_home/:/sw_home/ \
    -v /pde_ai/:/pde_ai/ \
    -v /mxstorage/:/mxstorage/ \
  mxcr.io/ai-release/maca/paddle-metax:3.3.0-maca.ai3.3.0.0-py310-ubuntu22.04-amd64 \
  /bin/bash
```

3. 进入容器运行环境

```shell
docker exec -it <container_name>
```

4. 初始化容器

```shell
apt update
apt install -y libglib2.0-dev
apt install -y unzip
apt install -y vim git #unzip libgl1-mesa-glx libsm6 libxext6 ffmpeg
apt-get install -y libssl-dev
```

5. 拉取代码分支

```shell
git clone  https://github.com/PaddlePaddle/PaddleFormers.git
cd PaddleFormers
python -m pip install -e .
```


# 2. 开始训练

所有示例脚本都位于 examples/config/metax 下。下面以 ERNIE-4.5-21B-A3B-PT 的 SFT 为例。首先需要通过 Huggingface 下载模型：

```shell
hf download baidu/ERNIE-4.5-21B-A3B-PT --local-dir baidu/ERNIE-4.5-21B-A3B-PT
```

如果需要对 ERNIE-4.5-21B-A3B-PT 进行全参数 SFT：

```shell
bash ./examples/config/metax/ERNIE-4.5-21B-A3B/sft/run_sft.sh
```

如果需要对 ERNIE-4.5-21B-A3B-PT 进行基于 LoRA 的 SFT：

```shell
bash ./examples/config/metax/ERNIE-4.5-21B-A3B/sft/run_lora.sh
```

训练产物位于 checkpoints 下，如果要使用 FastDeploy 进行推理，请参考[https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/metax_gpu.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/metax_gpu.md)
