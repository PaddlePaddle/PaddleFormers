[English](README.md) | 简体中文

# ERNIE-4.5-300B-A47B 预训练
本文档介绍如何进行 ERNIE-4.5-300B-A47B 预训练，运行训练至少需要96卡 NVIDIA H800 80G。

## 数据准备
本 repo 为您准备了 demo 数据集以方便您进行测试，demo 数据放在了 `./demo_data` 路径下。如果您想使用其他数据集或使用自定义数据集，
请参考 [Pretrain 数据集](https://paddlenlp.readthedocs.io/zh/latest/llm/dataset.html) 中的内容。

## 镜像准备
您的机器需要安装 CUDA 驱动（>= 525.60.13），并安装 CUDA toolkit 12.9。您可以使用镜像 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda12.9-cudnn9.9` 来进行预训练任务，同时
请确保您的集群中有 `mpi` 环境。

## 环境准备
`mpirun python -m pip install -r requirements.txt --force-reinstall`

## 开始训练
在准备好环境后。您可以通过执行以下命令来进行2016卡预训练：
`mpirun bash scripts/train_2016_gpus.sh`，
或执行以下命令来进行96卡预训练：
`mpirun bash scripts/train_96_gpus.sh`

- 注意，您需要将 `train_2016_gpus.sh` 或 `train_96_gpus.sh` 中的 `master_ip` 与 `port` 根据您的环境进行替换。

该工具包提供了 ERNIE-4.5-300B-A47B 预训练的高性能实现，包括多维混合并行训练策略和 FP8 混合精度优化，更多的优化点和功能会基于此版本持续更新。
