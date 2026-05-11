---
name: hf-to-paddle-weight-aligner
description: 验证 PaddleFormers 模型热启，并修复出现的问题
---

## Your Task

PaddleFormers中，已有$0的模型代码，需要验证能否正确加载`safetensors`。你需要：

1. 使用`from_pretrained`分别加载PaddleFormers和transformers模型，提取二者的网络权重，加载paddle模型时需加上参数`load_checkpoint_format="flex_checkpoint"`
2. 对比二者网络权重是否一致，可以使用`numpy`直接比对二者是否相等，需注意，paddle的Linear算子权重是转置的，维度与torch相反，比对的时候需要先转置对齐权重维度，再比对具体值
3. 如果paddle的权重没有正确加载，查找原因

## Note

1. 模型检查点在$0目录下
2. 如果权重加载不对，有可能是AOA配置的问题，也有可能是组网实现的问题，重点检查这两个地方
3. 所有测试代码和中间文件生成在`claude_workspace`目录中，可以生成多个测试代码文件，transformers和PaddleFormers分别使用不同的独立文件加载、输出，最后使用独立文件去比对
4. PaddleFormers和transformers的权重key应该是严格对齐的，比对的时候不需要设置专门的映射规则

## Additional resources

- AOA配置的写法可以参考文档[Flex Checkpoints 用户文档](references/aoa_doc.md)