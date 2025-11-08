English | [简体中文](./README_zh.md)

# ERNIE-4.5-300B-A47B Pre-Training

This document introduce how to pre-train the ERNIE-4.5-300B-A47B model, the pre-training requires at least 96 NVIDIA H800 80G GPUs.

## Data Preparation
This repository provide a demo dataset on the path `./demo_data` for quick start. If other dataset or user defined dataset are needed,
please reference this document [Pretrain dataset](https://paddlenlp.readthedocs.io/en/latest/llm/dataset.html).

## Docker Image Preparation
The CUDA driver on your machine should be ‌≥525.60.13, and the CUDA toolkit 12.9 image is needed. You can use `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda12.9-cudnn9.9` for training. And `mpi` environment should be deployed on the cluster.

## Runtime Environment Preparation
`mpirun python -m pip install -r requirements.txt --force-reinstall`

## Start Pre-Training
After the environment is ready, pre-training on 2016 GPUs can be launched by:
`mpirun bash scripts/train_2016_gpus.sh`,
pre-training on 96 GPUs can be launched by:
`mpirun bash scripts/train_96_gpus.sh`

- Note that, the `master_ip` and `port` in `train_2016_gpus.sh` or `train_96_gpus.sh`
should be replaced according to the real environment.


The toolkit provides a high-performance implementation of ERNIE-4.5-300B-A47B pre-training, including the hybrid parallelism training strategy and FP8 mixed precision optimization. More advanced optimizations are on the way.
