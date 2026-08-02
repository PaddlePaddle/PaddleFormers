# LLaVA-OneVision-1.5 Reduced-Depth Acceptance

This document records the current PR acceptance path for
`lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`.

Only the current best local validation path is included here:
**reduced depth, full width**. Historical debug paths and non-PR validation
experiments are intentionally excluded from this PR-facing acceptance note.

## Validation Scope

Primary checkpoint:

```text
.cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
```

Reduction policy:

```text
text hidden_size: 4096
text intermediate_size: 12288
text num_hidden_layers: 4
text num_attention_heads: 32
text num_key_value_heads: 8
text vocab_size: 151936
vision hidden_size: 1024
vision intermediate_size: 4096
vision depth: 4
vision num_heads: 16
```

This keeps the original model width, attention shape, and vocabulary. Only the
text and vision depths are reduced so the model can be validated on the local
RTX 3090 24 GB environment.

## Forward And Generation Alignment

Paddle checkpoint:

```text
.cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
```

HF-compatible export:

```text
.cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2-hf
```

Text-only FP32 comparison:

```text
reference_dir: .cache/llavaonevision1_5/reduced_depth_fullwidth_text_reference_fp32
max_diff: 0.00355536
mean_diff: 0.00055093
last_max_diff: 0.00296354
last_mean_diff: 0.00053563
first_10_tokens_match: True
```

Multimodal image+text FP32 comparison:

```text
reference_dir: .cache/llavaonevision1_5/reduced_depth_fullwidth_mm_reference_fp32
max_pixels: 200704
max_diff: 0.00360462
mean_diff: 0.00043145
last_max_diff: 0.00234246
last_mean_diff: 0.00036684
first_10_tokens_match: True
```

Result:

- Forward logits pass the `1e-2` order target on the reduced-depth full-width
  checkpoint.
- The first 10 generated tokens match Transformers exactly for both text-only
  and image+text FP32 validation.

Reproduction helpers:

```text
scripts/llavaonevision1_5/dump_hf_text_reference.py
scripts/llavaonevision1_5/dump_hf_reference.py
scripts/llavaonevision1_5/compare_paddle_reference.py
```

## GSM8K 300-Step SFT

Run setup:

```text
checkpoint: .cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
config: examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_300.yaml
script: scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_300.sh
dataset: data/gsm8k_erniekit/train.jsonl
eval_dataset: data/gsm8k_erniekit/test.jsonl
GPUs: 4 cards
parallelism: sharding stage2
max_steps: 300
```

Final metrics:

```text
exit_code: 0
train_runtime: 0:25:24.03
train_loss: 4.131325361728668
train_samples_per_second: 3.1495
train_steps_per_second: 0.1968
final_eval_loss: 2.947042942047119
final_eval_ppl: 19.04953976659703
max_memory_allocated_per_rank: 17.407671213150024 GB
max_memory_reserved_per_rank: 22.237423181533813 GB
```

Eval loss checkpoints:

```text
step 50:  4.722357273101807
step 100: 3.807039737701416
step 150: 3.3450706005096436
step 200: 3.1214351654052734
step 250: 2.994046211242676
step 300: 2.947042942047119
```

Artifacts:

```text
log: llavaonevision1_5_gsm8k_reduced_depth_fullwidth_300.log
loss_csv: llavaonevision1_5_gsm8k_reduced_depth_fullwidth_300_loss.csv
checkpoint: checkpoints/llavaonevision1_5-gsm8k-reduced-depth-fullwidth-sft-300
```

Result: the loss curve decreases normally throughout the 300-step run. Formal
loss-curve acceptance still needs a matching ms-swift baseline generated with
the same reduced-depth full-width model definition.

## Single-Card LoRA SFT

Single-card LoRA is the local fallback for training checks on 24 GB cards.

Run setup:

```text
checkpoint: .cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
config: examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_300.yaml
script: scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_lora_300.sh
GPUs: 1 card
fine_tuning: lora
lora_rank: 8
freeze_config: freeze_vision freeze_aligner
trainable_parameters_per_device: 2,031,616
max_steps: 300
```

Final metrics:

```text
exit_code: 0
train_runtime: 0:05:47.16
train_loss: 6.198721834818522
train_samples_per_second: 3.4566
train_steps_per_second: 0.8641
final_eval_loss: 4.883253574371338
final_eval_ppl: 132.05963150093888
max_memory_reserved: 8.986 GB
```

Eval loss checkpoints:

```text
step 50:  6.7881317138671875
step 100: 5.820539951324463
step 150: 5.516609191894531
step 200: 5.231670379638672
step 250: 4.969984531402588
step 300: 4.883253574371338
```

Artifacts:

```text
log: llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_300.log
loss_csv: llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_300_loss.csv
checkpoint: checkpoints/llavaonevision1_5-gsm8k-reduced-depth-fullwidth-lora-300
```

Result: the single-card LoRA loss curve decreases normally.

## Compiler Performance

Reduced-depth full-width inference comparison:

```text
script: scripts/llavaonevision1_5/compare_inference_compile_reduced.sh
model: .cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
seq_len: 64
steps: 50
warmup_steps: 10
dtype: bfloat16
benchmark_mode: manual-lm-head
attn_implementation: eager

dynamic tokens_per_second:   9729.92
to_static tokens_per_second: 13766.42
speedup: 41.49%
```

Reduced-depth full-width LoRA training compiler comparison:

```text
script: scripts/llavaonevision1_5/compare_training_compile_lora_reduced_depth_fullwidth.sh

300-step LoRA, recompute_granularity=full:
dynamic train_steps_per_second: 7.4357
to_static train_steps_per_second: 7.4175
training speedup: -0.24%

300-step LoRA, recompute_granularity=None:
dynamic train_steps_per_second: 7.6282
to_static train_steps_per_second: 7.3416
training speedup: -3.76%
```

Result:

- Inference compiler comparison passes the 20% improvement target.
- Local LoRA training compiler comparison applies the static path but does not
  meet the 20% training-side target. This should be rerun in the official
  acceptance environment.

## Unit And CI Smoke

Unit test command:

```bash
python -m unittest \
  tests.transformers.llavaonevision1_5.test_configuration \
  tests.transformers.llavaonevision1_5.test_modeling \
  -v
```

Recorded result:

```text
Ran 11 tests ... OK
```

Local CI smoke commands:

```bash
SKIP_PRECISION_CHECK=1 bash tests/integration_test/llavaonevision1_5_sft_single_card.sh single
SKIP_PRECISION_CHECK=1 bash tests/integration_test/llavaonevision1_5_sft_single_card.sh lora_single
```

Recorded result:

```text
single: passed
lora_single: passed
```

Official CE still needs the tiny checkpoint and precision baselines uploaded.

## Reproduction Commands

Run reduced-depth full-width 300-step SFT:

```bash
GPUS=0,1,2,3 bash scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_300.sh
```

Run single-card LoRA 300-step SFT:

```bash
bash scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_lora_300.sh
```

Run inference compiler comparison:

```bash
bash scripts/llavaonevision1_5/compare_inference_compile_reduced.sh
```

Run LoRA training compiler comparison:

```bash
bash scripts/llavaonevision1_5/compare_training_compile_lora_reduced_depth_fullwidth.sh
```

Run asset check:

```bash
python scripts/llavaonevision1_5/check_acceptance_assets.py
```

## Remaining PR Acceptance Items

- Provide or regenerate the ms-swift loss baseline for the same reduced-depth
  full-width checkpoint, then compare against
  `llavaonevision1_5_gsm8k_reduced_depth_fullwidth_300_loss.csv`.
- Upload the official CE tiny checkpoint and precision baselines.
- Rerun training-side compiler comparison in the official acceptance
  environment if the 20% training speedup is required for closure.
