---
name: hf-to-paddle-logits-aligner
description: 验证 PaddleFormers 模型与 transformers 模型的 logits 能否对齐
---

## Your Task

PaddleFormers中，已有$0的模型代码，在`paddleformers/transformers/$0`中，已可以正确加载模型权重，现在需要生成一个输入，验证paddle模型前向计算的结果与transformers模型是否一致。

1. 使用`from_pretrained`分别加载PaddleFormers和transformers模型，加载paddle模型时需加上参数`load_checkpoint_format="flex_checkpoint"`
2. 生成一个符合格式的输入，分别使用两个框架计算前向，得到`logits`
3. 比对两个`logits`是否一致，比对标准为二者完全相等，将结果卸载到numpy中计算
4. 如果`logits`不一致，需要检查有什么问题：
   - 可以使用`register_forward_hook`逐层检查精度，扫描出精度差异大的部分，定位具体问题
   - 如果是因计算方式或接口调用错误导致的精度差异，可以修复问题
   - 如果计算方式完全一致，是算子差异带来的精度差异，可以在`hook`里面继续写验证逻辑，例如使用transformers中对应算子的输出mock确认，确认后告诉我是哪些算子导致的差异

## Note

1. 模型配置及权重在`./$0`目录下

2. 运行paddle时，需要加载环境变量：

   ```bash
   unset PADDLE_ELASTIC_JOB_ID
   unset PADDLE_TRAINER_ENDPOINTS
   unset DISTRIBUTED_TRAINER_ENDPOINTS
   unset FLAGS_START_PORT
   unset PADDLE_ELASTIC_TIMEOUT
   
   export NNODES=1
   export PADDLE_TRAINERS_NUM=1
   # export CUDA_VISIBLE_DEVICES='0'
   export FLAGS_selected_gpus=0
   export FLAGS_use_accuracy_compatible_kernel=1
   export FLAGS_cudnn_deterministic=1
   ```

   

