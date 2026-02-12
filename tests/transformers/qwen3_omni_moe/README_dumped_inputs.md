# Dumped Inputs Testing Guide

## 概述

这个功能允许在训练过程中保存模型的输入数据，然后在单测中加载并重放这些真实的训练数据进行调试和验证。

## 工作流程

### 1. 训练时自动保存输入数据

在 `trainer.py` 的 `compute_loss` 方法中，每次前向传播前会自动保存 `**inputs` 到：
```
{output_dir}/dumped_inputs/inputs_step_{global_step}.npz
```

保存的数据包括所有可以转换为 numpy 的 tensor 和数组。

### 2. 单测中加载并测试

运行单测有两种方式：

#### 方式 1：自动加载最新的 dumped input
```bash
cd PaddleFormers/tests/transformers/qwen3_omni_moe
python test_modeling.py
```

这会：
1. 运行随机输入测试（原有功能）
2. 自动查找并加载 `output_paddle/dumped_inputs/` 中最新的输入文件
3. 使用真实训练数据进行测试

#### 方式 2：指定特定的 dumped input 文件
```bash
python test_modeling.py /path/to/inputs_step_100.npz
```

## 输出示例

```
============================================================
Test 1: Random input test
============================================================
output_ids:  <class 'paddle.Tensor'> Tensor(shape=[1, 20], dtype=float32, ...)

============================================================
Test 2: Dumped input test
============================================================
Loading dumped inputs from: /root/.../dumped_inputs/inputs_step_100.npz
Loaded input keys: ['input_ids', 'attention_mask', 'position_ids', ...]
  input_ids: shape=[2, 512], dtype=int64
  attention_mask: shape=[2, 512], dtype=int64
  position_ids: shape=[2, 512], dtype=int64
output_ids:  <class 'paddle.Tensor'> Tensor(shape=[2, 512], dtype=float32, ...)
```

## 注意事项

1. **存储空间**：每个 step 的输入都会被保存，注意磁盘空间占用
2. **调试用途**：这个功能主要用于调试，生产环境建议关闭或控制保存频率
3. **路径配置**：默认保存到 `output_dir/dumped_inputs/`，可根据需要修改路径

## 自定义

### 控制保存频率

如果想只保存特定 step 的数据，可以在 `trainer.py` 中添加条件：

```python
# 只在特定 step 保存
if self.state.global_step % 100 == 0:  # 每 100 步保存一次
    # dump logic here
```

### 保存额外信息

可以在 `dump_dict` 中添加更多信息：

```python
dump_dict['global_step'] = self.state.global_step
dump_dict['labels'] = labels.numpy() if labels is not None else None
```

## 故障排查

1. **找不到 dumped_inputs 目录**
   - 确保训练已经运行并生成了输入文件
   - 检查 `output_dir` 路径配置是否正确

2. **加载数据出错**
   - 检查 numpy 版本兼容性
   - 确认 paddle 版本与保存时一致

3. **模型输入不匹配**
   - 确认模型配置与训练时一致
   - 检查输入 key 是否完整
