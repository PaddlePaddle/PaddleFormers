[English](README.md) | 简体中文

# 预训练权重转换工具
这篇文档介绍如何将我们发布的预训练权重转换为当前模型可以加载的权重格式。

## 下载预训练权重
下载已发布的预训练权重。

## 保存当前模型的 checkpoint
运行预训练模型，并得到一份当前模型的 checkpoint。

## 转换权重
`python  tools/uc_to_sharded/convert_uc_to_sharded.py --org <path_to_pretrained_weights> --cur <path_to_checkpoint> --dst <path_to_converted_weights>`
