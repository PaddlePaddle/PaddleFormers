# 1. 安装

**环境依赖**

|Chip type|Driver version|
|-|-|
|MUSA S5000|5.1.0-server|

* **机器：** MUSA S5000 80GB 8-card machine
* **GCC path：**  /usr/bin/gcc (12.3.0)
* **python version：** 3.10

要验证 MUSA GPU 是否正常，可以使用`mthreads-gmi`命令

```shell
mthreads-gmi
#example：$ mthreads-gmi

Tue Jun 23 17:48:04 2026
---------------------------------------------------------------------
    mthreads-gmi:2.4.1           Driver Version:5.1.0-server
---------------------------------------------------------------------
ID   Name                 |PCIe                |%GPU  Mem
     Device Type   Perf   |Pcie Lane Width     |Temp  MPC Capable
                                               |      ECC Mode
+-------------------------------------------------------------------+
0    MTT S5000            |00000000:18:00.0    |0%    0MiB(81920MiB)
     Physical      P0     |16x(16x)            |48C   YES
                                               |      On-die, EDC
+-------------------------------------------------------------------+
1    MTT S5000            |00000000:3a:00.0    |0%    0MiB(81920MiB)
     Physical      P0     |16x(16x)            |48C   YES
                                               |      On-die, EDC
+-------------------------------------------------------------------+
2    MTT S5000            |00000000:4b:00.0    |0%    0MiB(81920MiB)
     Physical      P0     |16x(16x)            |49C   YES
                                               |      On-die, EDC
+-------------------------------------------------------------------+
3    MTT S5000            |00000000:5c:00.0    |0%    0MiB(81920MiB)
     Physical      P0     |16x(16x)            |48C   YES
                                               |      On-die, EDC
+-------------------------------------------------------------------+
4    MTT S5000            |00000000:9a:00.0    |0%    0MiB(81920MiB)
     Physical      P0     |16x(16x)            |48C   YES
                                               |      On-die, EDC
+-------------------------------------------------------------------+
5    MTT S5000            |00000000:ba:00.0    |0%    0MiB(81920MiB)
     Physical      P0     |16x(16x)            |50C   YES
                                               |      On-die, EDC
+-------------------------------------------------------------------+
6    MTT S5000            |00000000:ca:00.0    |0%    0MiB(81920MiB)
     Physical      P0     |16x(16x)            |48C   YES
                                               |      On-die, EDC
+-------------------------------------------------------------------+
7    MTT S5000            |00000000:da:00.0    |0%    0MiB(81920MiB)
     Physical      P0     |16x(16x)            |49C   YES
                                               |      On-die, EDC
---------------------------------------------------------------------

---------------------------------------------------------------------
Processes:
ID   PID           Process name                           GPU Memory
                                                               Usage
+-------------------------------------------------------------------+
   No running processes found
---------------------------------------------------------------------
```
**安装依赖**

1. 拉取镜像

```shell
docker pull registry.mthreads.com/public/paddle_musa:musa-5.1.0-py310-ubuntu22.04
```

2. 启动 docker

```shell
docker run \
    -it \
    --network=host \
    --privileged \
    --env MTHREADS_VISIBLE_DEVICES=all \
    --shm-size=80g \
    --name <container_name> \
    -v <mnt>:<mnt> \
  registry.mthreads.com/public/paddle_musa:musa-5.1.0-py310-ubuntu22.04 \
  /bin/bash
```

3. 进入容器运行环境

```shell
docker exec -it <container_name>
```

4. 拉取代码分支

```shell
git clone  https://github.com/PaddlePaddle/PaddleFormers.git
cd PaddleFormers
python -m pip install -e .
```


# 2. 开始训练

所有示例脚本都位于 examples/config/musa 下。
下面以 ERNIE-4.5-21B-A3B-PT 的 SFT 为例，可以参考 examples/best_practices/ERNIE-4.5/README.md 进行数据和模型准备。

如果需要对 ERNIE-4.5-21B-A3B-PT 进行全参数 SFT：

```shell
bash ./examples/config/musa/ERNIE-4.5-21B-A3B/sft/run_sft.sh
```
