# Molmo 图文 CINN 复现

2026-11-11 在 40885ae 上重新执行入口（Paddle 3.5.0.dev20260722）：两端各 20 步和同权重独立推理通过；loss 最大差 4.77e-7，72 项首步梯度均保留且有限、最大差 1.46e-3，推理 logits 最大差 2.83e-5。动态图从 step20 续跑到22通过，并生成分段首步梯度。该回归不代表重新完成 30000 步收敛训练。

用于框架测试的随机 tiny，FP32、关闭 TF32、单卡 batch 1。固定真实图片子集：20 张训练、4 张留出，学习类别 ID 数字 token，不是完整模型任务质量评测。两端均启用 molmo_attention_boundary.py 的 attention 边界适配；CINN 编译视觉残差层、文本 MLP 与 attention 核心，RoPE 系数生成与图文组装留在动态侧。没有将整个文本 attention 排除编译。

## 环境与资产

本地环境：V100S 32GB，Python 3.10.20，Paddle 3.5.0.dev20260722（08de6be508389d5277b02e78fbc8742f1ed78160，编译 CUDA 12.6），numpy 1.26.4。安装匹配的 Paddle GPU 包及 PaddleFormers requirements。无 AMP、无分布式。

从当前 checkout 导入 PaddleFormers，实际 commit 写入结果元数据。模型配置及预处理保留在 run_model.py。

权重暂不上传。CHECKPOINT_DIR 放 initial.pdparams 与 inference.pdparams，分别为历史共同随机起点与动态图 30000 步权重。asset_manifest.json 校验权重、数据和 tokenizer 的大小及 SHA256，缺失或不匹配即失败。data/images_manifest.json 记录真实 ImageNet 图片来源，使用须遵守原数据集条款。初始权重根据原 seed 重建，并验证全部参数哈希与历史初始记录完全一致。

## 运行

```bash
export CINN_RESOURCE_DIR=/path/to/molmo-resources
export CHECKPOINT_DIR=/path/to/local-weights
export PYTHON_BIN=/path/to/python
bash tests/cinn/molmo/run_model.sh
bash tests/cinn/molmo/run_model_cinn.sh
"$PYTHON_BIN" tests/cinn/molmo/compare_results.py tests/cinn/molmo/results
MODE=dynamic STEPS=30000 RESULTS_DIR=tests/cinn/molmo/results/full bash tests/cinn/molmo/train.sh
MODE=cinn STEPS=30000 RESULTS_DIR=tests/cinn/molmo/results/full bash tests/cinn/molmo/train.sh
MODE=dynamic RESULTS_DIR=tests/cinn/molmo/results/infer1 bash tests/cinn/molmo/infer.sh
MODE=cinn RESULTS_DIR=tests/cinn/molmo/results/infer1 bash tests/cinn/molmo/infer.sh
```

默认每端 20 步真实 AdamW 更新加独立进程固定前向推理；学习率 2e-5。推理不是 generate，预热 10 次、计时 30 次，GPU 同步，排除预处理、H2D 和首次编译。GPU_ID 默认为 0，重复运行必须指定新的 RESULTS_DIR。CINN 日志缺少编译标志则入口失败。

## 验证边界

训练输出 loss.jsonl、validation.jsonl、首步梯度及权重；独立推理输出 logits.npy、inference.json。两端使用相同 inference.pdparams。吞吐提升=(动态耗时/CINN 耗时-1)×100%，与延迟下降不同。运行产物不上传。

历史长跑用于固定子集拟合；本次交付短测仅验证脚本移植及数值、梯度路径，不宣称重新完成长程训练，不代表原始完整模型质量或泛化能力。

短程比较阈值：--loss-atol 1e-5、--grad-atol 2e-3、--logits-atol 1e-4，rtol 均为 0。超限或非有限值直接失败，阈值记录在输出中。合成负例单测验证 loss/梯度偏差 0.1 时必定失败，包含 python -O；不将负例当作模型精度数据。Molmo 的梯度阈值依据历史实测 1.46e-3 留余量设为 2e-3，没有照搬其他模型的 1e-4。上述阈值用于短程回归，不能替代长程收敛判断。

按交付约定保留 run_model.sh / run_model_cinn.sh，底层调用同一 run_model_common.sh；该目录是显式 GPU 手工入口，未声称接入默认 CI。续跑会保存当前分段首步的梯度，而不是遗漏此文件。


外部资源暂不上传：CINN_RESOURCE_DIR 下需包含 assets/tokenizer.json、data/images24.npz 和 data/images_manifest.json，均按 asset_manifest.json 校验。图片使用遵守 ImageNet 数据条款。资源和权重请联系贡献者获取，公开下载入口待后续维护。上述历史结果不代表当前 checkout 已重新完成长跑。
