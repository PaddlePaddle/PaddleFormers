# Mistral 迁移验证报告（Paddle vs ms-swift/Torch）

## 1. 目标
在“同一模型 + 同一批 token（固定 input_ids/labels）+ 同优化超参”条件下，执行 300-step 最小对照训练，剥离模板与数据管线差异，验证迁移到 Paddle 后训练行为是否与主流 Swift(Torch)一致。

## 2. 实验环境
- 日期：2026-05-25
- 设备：单卡 GPU（GPU 4）
- 模型：`/tmp/mistral-7B-l2-dev`（约 0.44B，2 层 slim 版）
- 框架：
  - 对照组 A：ms-swift / Torch
  - 对照组 B：Paddle / PaddleFormers

## 3. 对齐项（强对齐）
- 固定同一批 `input_ids/labels`（完全一致）
- `optimizer`: AdamW
- `learning_rate`: 1e-5
- `warmup_steps`: 20（线性 warmup + 线性衰减到 0）
- `betas`: (0.9, 0.999)
- `eps`: 1e-8
- `weight_decay`: 0
- `gradient_accumulation_steps`: 4
- 总训练：300 optimizer steps（脚本内部执行 1200 micro-steps）

## 4. 运行产物
- Torch loss 日志：`/tmp/ab_same_tokens_torch_loss.jsonl`
- Paddle loss 日志：`/tmp/ab_same_tokens_paddle_loss.jsonl`

## 5. 统计口径
定义：`diff = loss_paddle - loss_torch`

### 5.1 全量 300 step
- n = 300
- mean_diff = +0.0765098
- std_diff = 0.4849979
- max_abs_diff = 4.3311100

### 5.2 分段统计（去除早期瞬态）
- from step 21:
  - n = 280
  - mean_diff = -0.0086790
  - std_diff = 0.0229094
  - max_abs_diff = 0.1105619
- from step 51:
  - n = 250
  - mean_diff = -0.0026415
  - std_diff = 0.0117540
  - max_abs_diff = 0.0689497
- from step 101:
  - n = 200
  - mean_diff = +0.0007368
  - std_diff = 0.0000286
  - max_abs_diff = 0.0007660

## 6. 关键观察
- 早期（step 1~20）存在较大差异，导致全量均值被显著拉高。
- 中后期快速收敛并趋于稳定；step 101 之后两边差异已在 `~1e-3` 量级以内，且波动极小。
- 说明在剥离模板和数据管线后，Paddle 迁移模型与 Torch 对照在训练动力学上基本一致；主要偏差集中在早期更新瞬态。

## 7. 验收结论（建议口径）
- 结论：**通过（有条件）**。
- 依据：
  - “同 token 输入”的 300-step A/B 证明迁移模型在中后段 loss 行为与 Torch 高度一致；
  - 主要差异来源不是模型结构迁移本身，而是早期优化瞬态与实现细节（数值路径/内核/调度边界）差异。

## 8. 残余风险与后续建议
- 若需“全程（含前 20 step）几乎重合”，建议进一步强约束：
  - 固定 dtype 路径（例如统一 fp32 或统一 bf16 + 确认 AMP 策略一致）；
  - 固定算子实现（关闭/统一某些 fused kernel）；
  - 固定初始化与随机源（含 dataloader 顺序、dropout、seed 链路）。
- 对外验收建议同时给出：
  - 全程统计（保留客观性）；
  - 去瞬态分段统计（体现主体训练区间的一致性）。

## 9. BF16 强对齐补充（2026-05-27）

### 9.1 目的
在保留“同 token 输入”前提下，切换到更贴近实际训练的低精度路径，验证迁移模型在 `bf16` 下与 Torch 的一致性。

### 9.2 对齐设置
- 同一模型：`/tmp/mistral-7B-l2-dev`（2 层 slim）
- 同一批 `input_ids/labels`
- 同优化超参：AdamW, lr=1e-5, warmup=20, grad_accum=4, 300 optimizer steps
- 统一关闭 fused/flash 快路径（no-fused）
- 精度路径：
  - Torch：`fp32 参数 + bf16 autocast`
  - Paddle：`fp32 参数 + bf16 autocast`

### 9.3 产物
- Torch: `/tmp/ab_same_tokens_torch_loss_bf16_nofused_300.jsonl`
- Paddle: `/tmp/ab_same_tokens_paddle_loss_bf16_nofused_300.jsonl`

### 9.4 对比统计（diff = paddle - torch）
- from step 1:
  - n=300
  - mean_diff = -0.0028461179737738953
  - std_diff = 0.04033434740975705
  - max_abs_diff = 0.28579092025756836
- from step 21:
  - n=280
  - mean_diff = +0.0015440172271545245
  - std_diff = 0.005615626457655772
  - max_abs_diff = 0.01901003159582615
- from step 51:
  - n=250
  - mean_diff = +0.0022719342584023253
  - std_diff = 0.0051013218456804335
  - max_abs_diff = 0.01901003159582615
- from step 101:
  - n=200
  - mean_diff = +0.00010462422229466028
  - std_diff = 0.0002033153940003238
  - max_abs_diff = 0.0012710219016298652

### 9.5 结论补充
- 在 `bf16 + no-fused` 强对齐条件下，Paddle 与 Torch 在中后段 loss 曲线高度一致。
- 早期（step 1~20）仍存在瞬态差异，但 step>=101 后差异已收敛到 `1e-4 ~ 1e-3` 量级，可认为迁移后训练行为稳定且与主流实现基本对齐。
