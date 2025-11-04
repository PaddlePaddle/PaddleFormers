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

## 性能相关开关
通用性能开关：
- `use_quant_before_a2a`：在 all to all 通讯前进行 FP8 量化，降低通讯量。
- `use_combine_before_a2a`：gate prob 乘法迁移，降低 MOE 模块的显存开销。
- `use_async_a2a`：异步 all to all，将通讯与计算进行 overlap。
- `use_rms_qkv_recompute`：将 rms norm 计算与 qkv 计算进行 fusion。
- `use_ep_comm_overlap`：将 all to all 与专家计算进行 overlap。当开启`use_quant_before_a2a`或
`use_async_a2a`时，`use_ep_comm_overlap`需要关闭。

FP8 node 旨在将 MOE 模块中的 matmul 采用 FP8 精度进行计算，同时进行一些显存优化。FP8相关开关：
- `use_fp8_mlp`：MLP 结构开启 FP8 计算。
- `use_fp8_fuse_node`：MOE 模块开启 FP8 计算。
- `fp8_mem_configs`：FP8 显存相关优化策略。
  - `recompute_fwd_gate_up`：gate up 结构 recompute，节省显存。
  - `dequant_input`：对 MLP 的 input 仅存储 FP8 量化版本，节省显存。
  - `shared_expert`：share expert 是否使用 FP8 计算。
- `fp8_fused_ops_configs`：FP8算子融合相关优化策略。
  - `stack_quant`：stack 与 fp8 quant 融合。
  - `swiglu_probs_bwd`：swiglu 与 gate prob 乘法反向的融合。
  - `split_group_gemm`：MOE 各 experts 的计算是否采用 group gemm。
  - `spaq`：swiglu 与 gate prob 乘法和 FP8 quant 的融合。
  - `transpose_split_quant`：transpose、split 与 quant 的融合。
