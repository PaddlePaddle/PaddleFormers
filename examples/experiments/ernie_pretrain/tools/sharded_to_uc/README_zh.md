# 预训练权重转换工具
这篇文档介绍如何将预训练产出的`sharded model`格式的多机权重转化为单卡视角的`unified checkpoint`格式的权重。当前转化脚本仅针对`ERNIE-4.5-21B-A3B`使用`pure sharding + ep`策略训练产生的权重。

## 进行多机预训练并保存当前模型的 checkpoint
运行预训练模型，并得到一份当前模型的多机 checkpoint。

## 多机权重聚合
将多机权重聚合到一台机器上：
`python tools/sharded_to_uc/gather_all_ckpt.py --org_path <path_to_multi_nodes_weights> --tgt_path <path_to_gathered_ckpt> --hostfile_path <path_to_hostfile>`

## 聚合 sharding、ep 切分的参数
将 sharding stage1 v2切分的参数进行还原，将 ep 切分的参数进行还原：
`python tools/sharded_to_uc/merge_sharding_ep.py --base_path <path_to_gathered_ckpt> --output_dir_path <path_to_single_card_ckpt>`

## 将 sharded ckpt 转化为 unified checkpoint
转化为 unified checkpoint，用来给 post training 使用：
`python tools/sharded_to_uc/convert_sharded_to_uc.py --sharded_path <path_to_single_card_ckpt> --uc_path <path_to_unified_ckpt>`
