# MiniMax tiny CINN 复现

2026-11-11 在 fa2ffb50c974a72b3d715e35846b8136149a5425 上移除测试侧 MoE 替换后，正式实现双端各 20 步和同权重独立推理通过：loss 最大差 9.54e-7、41 项首步梯度最大差 5.73e-6、logits 最大差 8.82e-6。环境为下述 Paddle 3.5.0.dev20260722；这是短程数值回归，不是重新完成长程收敛或 Stage3 验收。

随机 tiny 框架测试，FP32、TF32关闭、batch1、seq64。数据是 GSM8K 前24条，question+换行+answer，经 assets 测试 tokenizer 构造下一 token 标签；前20训练、4留出。不代表完整 MiniMax 任务质量。

## 环境

V100S 32GB、Python3.10.20、Paddle3.5.0.dev20260722（08de6be508389d5277b02e78fbc8742f1ed78160）、编译 CUDA12.6；transformers5.3.0、tokenizers0.22.2、numpy1.26.4及 PaddleFormers requirements。无 AMP/分布式。

从当前 checkout 导入 PaddleFormers，源码 commit 写入结果元数据，不要求外部源码快照。CINN 编译完整 forward wrapper，loss/optimizer 在动态侧；config.get 保留 Python。动态图与 CINN 都直接运行正式 MiniMaxSparseMoeBlock：专家累加使用无条件的函数式 index_add，不再依赖 token_idx.shape[0] 分支，也不安装测试侧 forward 替换补丁。所有专家仍按固定顺序执行，包括没有本地 token 的专家；没有移除专家或冻结参数。模型单测覆盖空专家场景下的输出、输入梯度及全部参数梯度，单卡验证不代表已完成 Stage3 多卡验收。

## 运行

权重不上传。CHECKPOINT_DIR 包含 initial.pdparams（共同随机起点）及 inference.pdparams（动态图30000步权重）；严格校验 asset_manifest.json，不匹配即失败。data/gsm8k_24.jsonl 来源 https://github.com/openai/grade-school-math （MIT），assets 为原测试配置/tokenizer。

```bash
export CINN_ASSET_DIR=/path/to/minimax/assets
export CHECKPOINT_DIR=/path/to/local-weights
export PYTHON_BIN=/path/to/python
bash tests/cinn/minimax/run_model.sh
bash tests/cinn/minimax/run_model_cinn.sh
"$PYTHON_BIN" tests/cinn/minimax/compare_results.py tests/cinn/minimax/results
MODE=dynamic STEPS=30000 RESULTS_DIR=tests/cinn/minimax/results/full bash tests/cinn/minimax/train.sh
MODE=cinn STEPS=30000 RESULTS_DIR=tests/cinn/minimax/results/full bash tests/cinn/minimax/train.sh
MODE=dynamic RESULTS_DIR=tests/cinn/minimax/results/infer1 bash tests/cinn/minimax/infer.sh
MODE=cinn RESULTS_DIR=tests/cinn/minimax/results/infer1 bash tests/cinn/minimax/infer.sh
```

快速入口20步真实更新训练加独立推理，AdamW lr2e-5/weight_decay0.01。推理固定文本前向，非 generate，10预热/30计时，排除预处理/H2D/编译；每步前后 GPU 同步。GPU_ID 默认0。需用新 RESULTS_DIR，拒绝覆盖。RESUME 可指定可信完整状态文件，两端状态不可互换；模型/优化器/RNG 每500步及结束保存。

## 验证

results/<mode>-train 包含 loss.jsonl、validation.jsonl、first_gradients.pdparams 等；<mode>-infer 保存 logits.npy 及 inference.json。两端独立推理必须使用同一个 inference.pdparams，不能比较各自训练产生的不同权重。入口检查 CINN 编译日志标志，不满足即失败。吞吐提升=(动态 ms/CINNms-1)×100%，不能与延迟下降混用。运行产物与权重不提交。

历史30000步仅证明固定子集充分拟合；本次新入口回归是短程训推和梯度检查，不宣称重新完成长跑。

compare_results.py 的短程回归阈值为 --loss-atol 1e-5、--grad-atol 1e-4、--logits-atol 1e-4，rtol 均为 0；超限、非有限值或缺失梯度均失败，实际阈值打印在结果中。test_compare_results.py 用合成负例验证 loss/梯度偏差 0.1 时失败（含 python -O）；这些负例不是模型精度数据。长程训练轨迹是否收敛应单独判断，不能直接套用短程逐点阈值。

按交付约定保留 run_model.sh / run_model_cinn.sh 两个便捷入口，它们共用 run_model_common.sh，不维护重复的运行逻辑。tests/cinn 为显式 GPU 手工回归入口，不属于默认 CI；生产 MoE 的回归放在 tests/transformers/minimax/test_modeling.py。

2026-10-07 本地入口回归通过：两端各20步，loss 最大绝对差9.54e-7；41项首步梯度均保留，最大绝对差5.73e-6；共同 checkpoint 独立推理 logits 最大绝对差8.83e-6。该短测只用于验证交付入口，不替代长程收敛材料。

Tokenizer/config assets are external and checked against asset_manifest.json. Obtain weights and assets from the contributor; public download hosting is pending. Historical results above refer to the previous source snapshot, not this checkout.

2026-11-05 当前 develop f7fa77f 入口回归通过：动态图和 CINN 各 20 步，loss 最大绝对差 9.54e-7；41 项首步梯度完整且有限，最大绝对差 4.77e-6；同权重推理 logits 最大绝对差 8.82e-6。推理中位耗时分别为 9.945 ms 和 2.265 ms；这是单次短程回归测量，不替代历史长程收敛结果或多次性能复核。若自定义 RESULTS_DIR，比较命令的结果目录应相应替换。
