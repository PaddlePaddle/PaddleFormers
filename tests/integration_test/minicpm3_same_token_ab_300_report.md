# MiniCPM3 迁移验证报告（Paddle vs Transformers/Torch）

## 1. 目标

参考 Mistral 迁移验收口径，验证 MiniCPM3 迁移到 PaddleFormers 后的单卡前向、生成和 GSM8K SFT 训练 loss 曲线一致性。

## 2. 实验环境

- 日期：2026-05-27
- 设备：单卡 GPU
- 模型：MiniCPM3-4B，本地 HuggingFace 单文件 `pytorch_model.bin`
- 前向/生成：真实 4B 模型
- loss 曲线：`/tmp/minicpm3-4B-l2-dev`（2 层 slim 版）
- 对照组 A：Transformers / Torch
- 对照组 B：Paddle / PaddleFormers

说明：本机 ms-swift 环境使用的 Transformers 5.8 与 MiniCPM3 remote code 存在兼容问题，修复导入 shim 后仍出现 NaN；因此本次使用可稳定运行的 Transformers/Torch 环境作为 baseline。训练对照固定同一批 `input_ids/labels`，剥离模板和数据管线差异。

## 3. 单卡前向精度

真实 4B 模型 logits 对齐结果：

- shape = `[1, 39, 73448]`
- input_ids_equal = true
- max_abs_diff = `1.71661376953125e-05`
- mean_abs_diff = `1.451915522920899e-06`
- last_token_max_abs_diff = `1.0251998901367188e-05`
- last_token_mean_abs_diff = `1.7648510493017966e-06`

结论：logits diff 远小于 `1e-2`，满足验收要求。

## 4. 模型生成验证

相同 prompt 下，真实 4B 模型 greedy 生成前 10 个 token：

- Torch: `[59320, 5, 59399, 59353, 59400, 72, 59320, 59349, 59370, 1709]`
- Paddle: `[59320, 5, 59399, 59353, 59400, 72, 59320, 59349, 59370, 1709]`
- first_10_equal = true

结论：生成输出正常，前 10 token 与 Transformers 一致。

## 5. GSM8K 300-step loss 曲线

### 5.1 对齐设置

- 数据集：GSM8K 纯文 SFT
- `max_seq_len`: 8192
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 4
- `max_steps`: 300
- `learning_rate`: 1e-5
- `warmup_steps`: 20
- schedule：线性 warmup + 线性衰减到 0
- optimizer：AdamW
- betas：`(0.9, 0.999)`
- eps：`1e-8`
- weight_decay：0
- seed：23

### 5.2 FP32 统计

定义：`diff = loss_paddle - loss_torch`

- from step 1:
  - n = 300
  - mean_diff = `-4.172325134277344e-07`
  - std_diff = `9.291444580626426e-07`
  - max_abs_diff = `3.814697265625e-06`
- from step 21:
  - n = 280
  - mean_diff = `-6.003039223807199e-07`
  - std_diff = `5.532849961071569e-07`
  - max_abs_diff = `2.384185791015625e-06`
- from step 101:
  - n = 200
  - mean_diff = `-7.212162017822265e-07`
  - std_diff = `4.986526149327275e-07`
  - max_abs_diff = `1.9073486328125e-06`

### 5.3 BF16 统计

- from step 1:
  - n = 300
  - mean_diff = `3.2245318094889324e-05`
  - std_diff = `0.0016769665466711426`
  - max_abs_diff = `0.0054013729095458984`
- from step 21:
  - n = 280
  - mean_diff = `5.795104163033622e-05`
  - std_diff = `0.001721116854577658`
  - max_abs_diff = `0.0054013729095458984`
- from step 101:
  - n = 200
  - mean_diff = `0.00017192959785461426`
  - std_diff = `0.0018608560569672616`
  - max_abs_diff = `0.0054013729095458984`

结论：FP32 基本数值级重合；BF16 loss 差异围绕 0 小幅波动，满足 300-step loss 曲线验收要求。

## 6. 关键修复

- 支持 HuggingFace 单文件 `pytorch_model.bin`：`from_pretrained(convert_from_hf=True)` 检测到本地 PyTorch bin 时自动使用 legacy checkpoint 加载链路，避免误入 flex checkpoint/AOA loader。
- 修复 CausalLM loss 归一化：显式按非 `-100` label 数量归一，与 Torch `ignore_index=-100` 的 CrossEntropyLoss 行为对齐。

## 7. 验收结论

- 单卡前向精度对齐：通过。
- 生成前 10 token 对齐：通过。
- GSM8K 300-step loss 曲线：通过。
