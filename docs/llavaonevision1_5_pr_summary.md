# LLaVA-OneVision-1.5 PR Summary

This note is the PR-facing summary for the PaddleFormers migration of
`lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`.

## Scope

- Add LLaVA-OneVision-1.5 model definition, including Rice vision tower, text
  decoder, multimodal projector, conditional generation model, and HF-to-Paddle
  checkpoint mapping.
- Add AutoConfig and AutoModelForConditionalGeneration registration.
- Add reduced-depth validation scripts, GSM8K SFT configs, CI smoke configs,
  and benchmark configs.
- Add unit tests for configuration, model forward, image-token forward, and
  AutoModel reload paths.

Due to local RTX 3090 24 GB memory limits, formal local acceptance evidence uses
the corrected reduced-depth full-width checkpoint:

```text
.cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
```

This checkpoint keeps the original text/vision widths and vocabulary while
reducing only model depth. It is the only validation path summarized in this
PR-facing note.

## Acceptance Evidence

### Forward and Generation Alignment

Reduced-depth full-width FP32 text-only comparison:

```text
max_diff: 0.00355536
mean_diff: 0.00055093
last_max_diff: 0.00296354
last_mean_diff: 0.00053563
first_10_tokens_match: True
```

Reduced-depth full-width FP32 image+text comparison:

```text
max_diff: 0.00360462
mean_diff: 0.00043145
last_max_diff: 0.00234246
last_mean_diff: 0.00036684
first_10_tokens_match: True
```

These pass the `1e-2` order forward-logits target and the first-10-token greedy
generation target on the reduced-depth full-width path.

### GSM8K 300-Step SFT

Run:

```text
script: scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_300.sh
config: examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_300.yaml
GPUs: 4 x RTX 3090 24 GB
parallelism: sharding stage2
max_steps: 300
```

Final metrics:

```text
train_loss: 4.131325361728668
final_eval_loss: 2.947042942047119
final_eval_ppl: 19.04953976659703
train_steps_per_second: 0.1968
```

Eval loss decreased from `4.722357273101807` at step 50 to
`2.947042942047119` at step 300.

### Compiler Performance

Reduced-depth full-width inference comparison:

```text
dynamic tokens_per_second:   9729.92
to_static tokens_per_second: 13766.42
speedup: 41.49%
```

Training compiler comparison was measured through the local single-card LoRA
fallback because full fine-tuning OOMs on a 24 GB card and static multi-card
sharding is rejected by the current PaddleFormers trainer. The static path was
applied successfully, but local LoRA training did not show a speedup:

```text
300-step LoRA, recompute_granularity=full:
dynamic train_steps_per_second: 7.4357
to_static train_steps_per_second: 7.4175
training speedup: -0.24%
```

This should be called out as an acceptance risk and rerun in the official
acceptance environment.

### Unit and CI Smoke

Unit tests:

```bash
python -m unittest \
  tests.transformers.llavaonevision1_5.test_configuration \
  tests.transformers.llavaonevision1_5.test_modeling \
  -v
```

Recorded result: `Ran 11 tests ... OK`.

Local CI smoke commands:

```bash
SKIP_PRECISION_CHECK=1 bash tests/integration_test/llavaonevision1_5_sft_single_card.sh single
SKIP_PRECISION_CHECK=1 bash tests/integration_test/llavaonevision1_5_sft_single_card.sh lora_single
```

Recorded result: both passed locally. Official CE still needs remote precision
baseline upload.

## PR File Groups

- Model code: `paddleformers/transformers/llavaonevision1_5/`
- Auto registration:
  `paddleformers/transformers/__init__.py`,
  `paddleformers/transformers/auto/configuration.py`,
  `paddleformers/transformers/auto/modeling.py`
- Tests: `tests/transformers/llavaonevision1_5/`
- CI smoke:
  `tests/integration_test/llavaonevision1_5_sft_single_card.sh`,
  `tests/config/ci/llavaonevision1_5_sft_single.yaml`,
  `tests/config/ci/llavaonevision1_5_lora_single.yaml`
- Reduced-depth full-width training and benchmark configs:
  `examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_300.yaml`,
  `examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_300.yaml`,
  `tests/config/benchmark/config/sft/LLaVA-OneVision-1.5-Reduced-Depth-FullWidth.yaml`
- Validation helpers: `scripts/llavaonevision1_5/`
- PR evidence:
  `docs/llavaonevision1_5_reduced_acceptance.md`,
  `docs/llavaonevision1_5_pr_summary.md`,
  `llavaonevision1_5_gsm8k_reduced_depth_fullwidth_300_loss.csv`,
  `llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_300_loss.csv`

## Remaining Risks

- Matching ms-swift loss baseline for the same reduced-depth full-width model is
  still required for formal loss-curve comparison.
- Official CE needs the tiny checkpoint and precision baselines uploaded.
- Training-side compiler speedup does not meet the 20% target in the local LoRA
  fallback; rerun in the official acceptance environment.

## Suggested PR Description

```text
### PR 类型
新增模型

### 改动内容
- 新增 LLaVA-OneVision-1.5 PaddleFormers 模型实现和 Auto 注册。
- 新增 HF-to-Paddle 权重映射、reduced-depth full-width 资产生成、前向/生成对齐、GSM8K SFT 和 compiler benchmark 辅助脚本。
- 新增模型单测、单卡 SFT/LoRA CI smoke 配置和 benchmark 配置。
- 补充 reduced-depth full-width 验收记录。

### 验证结果
- reduced-depth full-width FP32 text-only logits max_diff=0.00355536，first_10_tokens_match=True。
- reduced-depth full-width FP32 image+text logits max_diff=0.00360462，first_10_tokens_match=True。
- reduced-depth full-width GSM8K 300-step SFT 完成，eval loss 从 4.7223 降到 2.9470。
- reduced-depth full-width 推理 compiler speedup=41.49%。
- 本地单测通过：Ran 11 tests ... OK。
- 本地 CI smoke 通过：single 与 lora_single 均通过，官方 CE baseline 待上传。

### 风险说明
- 本地 RTX 3090 24 GB 环境下，PR 验收证据统一采用 reduced-depth full-width 路径。
- ms-swift 对照曲线、官方 CE precision baseline、训练侧 compiler 20% speedup 仍需在正式验收环境补齐。
```
