# LLaVA-OneVision-1.5 Reduced Acceptance Progress

This document tracks the reduced-model validation path for the PaddleFormers
migration. The reduced path is used because the available RTX 3090 hardware
cannot reliably run full 8B training. The reduction must be clearly labeled in
all reports.

## Reduced Model Scope

Current reduced checkpoint:

- Path: `.cache/llavaonevision1_5/reduced-random-llavaonevision1_5-4l-512h`
- Text decoder: 4 layers, hidden size 512, intermediate size 1536
- Vision tower: 4 layers, hidden size 256, intermediate size 768
- Max position embeddings: 1024
- Tokenizer: copied from the local LLaVA/Qwen-compatible tokenizer assets

Important note: this is a reduced-layer/width random checkpoint for local
validation. It is not the full 8B checkpoint and should not be reported as full
model acceptance.

## Completed Reduced Validation

### GSM8K Data

Converted HuggingFace `datasets.save_to_disk()` GSM8K data into erniekit JSONL:

- Train: `data/gsm8k_erniekit/train.jsonl`
- Test: `data/gsm8k_erniekit/test.jsonl`

### 300-Step SFT

Config:

- `examples/config/sft/llavaonevision1_5_gsm8k_reduced_300.yaml`

Result on RTX 3090 GPU 1:

```text
train_runtime: 0:02:30.32
train_loss: 8.2981
train_samples_per_second: 7.9828
train_steps_per_second: 1.9957
max_memory_allocated: 3.537 GB
max_memory_reserved: 7.364 GB
final_eval_loss: 7.064287185668945
```

Artifacts:

- Log: `llavaonevision1_5_gsm8k_reduced_300.log`
- Loss CSV: `llavaonevision1_5_gsm8k_reduced_300_loss.csv`
- Checkpoint: `checkpoints/llavaonevision1_5-gsm8k-reduced-sft-300`

### Compiler Performance

Inference reduced benchmark:

```text
Dynamic tokens_per_sec: 11721.84
To_static tokens_per_sec: 51104.16
Speedup: 335.96%
```

Training reduced short benchmark, 30 steps:

```text
Dynamic train_steps_per_second: 7.6398
To_static train_steps_per_second: 7.6492
Speedup: 0.12%
```

Simple train/infer average speedup:

```text
168.04%
```

Training `to_static` was confirmed by the log message:

```text
Successfully to apply @to_static to the whole model.
```

## Reproduction Commands

Run 300-step reduced SFT:

```bash
bash scripts/llavaonevision1_5/run_gsm8k_sft_reduced_300.sh
```

Run reduced inference compiler comparison:

```bash
bash scripts/llavaonevision1_5/compare_inference_compile_reduced.sh
```

Run reduced training compiler comparison:

```bash
bash scripts/llavaonevision1_5/compare_training_compile_reduced.sh
```

## Reduced Text-Only Forward/Generation Alignment

A reduced text-only Transformers reference was generated from the HF-compatible
exported reduced checkpoint:

- HF-compatible reduced checkpoint:
  `.cache/llavaonevision1_5/reduced-random-llavaonevision1_5-4l-512h-hf`
- Paddle reduced checkpoint:
  `.cache/llavaonevision1_5/reduced-random-llavaonevision1_5-4l-512h`
- Reference prompt: `Describe this image briefly.`

FP32 comparison result:

```text
max_diff: 0.00496596
mean_diff: 0.00073955
last_max_diff: 0.00390404
last_mean_diff: 0.00072535
first_10_tokens_match: True
```

BF16 comparison result:

```text
max_diff: 0.08203125
mean_diff: 0.01197283
last_max_diff: 0.06250000
last_mean_diff: 0.01155120
first_10_tokens_match: False
```

Interpretation: reduced FP32 forward logits are within the `1e-2` acceptance
order and reduced FP32 generation first-10 tokens match Transformers exactly.
The BF16 run is close but slightly above `1e-2` mean diff and diverges at the
10th generated token, which should be reported as BF16 numerical drift on the
reduced random checkpoint.

Reproduction helpers:

- `scripts/llavaonevision1_5/dump_hf_text_reference.py`
- `scripts/llavaonevision1_5/compare_paddle_reference.py`

## Reduced CI/CE and Benchmark Entrypoints

Reduced-path helper scripts and configs have been added so the validation flow
can be repeated without falling back to the full 8B paths:

- Asset check: `scripts/llavaonevision1_5/check_acceptance_assets.py`
- CE asset preparation: `scripts/llavaonevision1_5/prepare_ce_assets.sh`
- Reduced 300-step SFT: `scripts/llavaonevision1_5/run_gsm8k_sft_reduced_300.sh`
- Reduced inference compiler comparison:
  `scripts/llavaonevision1_5/compare_inference_compile_reduced.sh`
- Reduced training compiler comparison:
  `scripts/llavaonevision1_5/compare_training_compile_reduced.sh`
- LoRA single-card CI config:
  `tests/config/ci/llavaonevision1_5_lora_single.yaml`
- Reduced benchmark config:
  `tests/config/benchmark/config/sft/LLaVA-OneVision-1.5-Reduced-4L-512H.yaml`

Local asset check result:

```text
OK: tiny checkpoint directory
OK: reduced checkpoint directory
OK: HF-compatible reduced checkpoint directory
OK: reduced text reference directory
OK: GSM8K erniekit train data
OK: GSM8K erniekit eval data
optional-missing: VL dummy image directory
```

The optional VL dummy image directory is not required for the current reduced
GSM8K text-only validation path.

Local CI single-card smoke result:

```text
Command: SKIP_PRECISION_CHECK=1 bash PaddleFormers/tests/integration_test/llavaonevision1_5_sft_single_card.sh single
Result: Test passed.
train_runtime: 0:00:01.90
train_loss: 12.9211
train_samples_per_second: 20.9772
train_steps_per_second: 5.2443
```

This verifies that the model-specific CI training entrypoint can complete in
the local Paddle Docker environment. The remote precision baseline download was
skipped only for local smoke validation.

Local CI LoRA single-card smoke result:

```text
Command: SKIP_PRECISION_CHECK=1 bash PaddleFormers/tests/integration_test/llavaonevision1_5_sft_single_card.sh lora_single
Result: Test passed.
train_runtime: 0:00:01.65
train_loss: 12.9267
train_samples_per_second: 24.132
train_steps_per_second: 6.033
trainable_parameters_per_device: 6,656
```

This adds local smoke coverage for the LoRA variant expected by the CE flow.
The workflow now includes a separate `LLaVA-OneVision-1.5-LoRA-single-card`
entry. As with the SFT smoke, official CI still needs the corresponding remote
precision baseline file to be uploaded.


## Reduced-Depth Full-Width Scope Update

The client-facing reduced validation scope has been corrected to **reduce depth
only**. The earlier `4L-512H` checkpoint reduced both layer count and hidden
width, so it is now retained only as a local debugging reference and should not
be used as the primary acceptance artifact.

Primary reduced-depth full-width checkpoint:

```text
.cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
```

This checkpoint keeps the original model widths and vocabulary while reducing
only the text and vision depths:

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

Local memory findings on RTX 3090 24 GB cards:

- Single-card full SFT OOMs during AdamW FP32 master/optimizer-state creation.
- Two-card TP does not materially reduce per-card memory for this custom model.
- Two-card sharding stage2 still OOMs while creating AdamW moments.
- Four-card sharding stage2 completes a 1-step full SFT smoke successfully.

Successful 4-card smoke result:

```text
GPUs: 1,2,3,4 on host, visible as 0,1,2,3 in container
sharding: stage2
max_steps: 1
train_loss: 13.0345
train_runtime: 0:00:04.17
current_memory_allocated: 13.63 GB
max_memory_reserved: 19.92 GB
```

New primary reduced-depth full-width entrypoints:

- `examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_300.yaml`
- `scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_300.sh`
- `tests/config/benchmark/config/sft/LLaVA-OneVision-1.5-Reduced-Depth-FullWidth.yaml`

Recommended local run command:

```bash
GPUS=0,1,2,3 bash scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_300.sh
```

The ms-swift loss baseline still needs to be provided or regenerated under the
same reduced-depth full-width model definition. The old `4L-512H` loss curve is
not directly comparable to this corrected acceptance scope.

## Remaining Reduced-Path Work

- Prepare or obtain an ms-swift baseline for the same reduced model definition,
  then compare the 300-step GSM8K loss curve against
  `llavaonevision1_5_gsm8k_reduced_300_loss.csv`.
- Treat the old reduced-layer/width checkpoint as a debugging-only artifact.
- Continue validation on the reduced-depth full-width checkpoint, which matches
  the client-approved reduction direction.
- Keep the full 8B forward-logits gap documented separately as out of scope for
  the current reduced-hardware validation path.

## Reduced-Depth Full-Width Forward Alignment

Single-card forward/generation alignment has been rerun on the corrected
reduced-depth full-width checkpoint:

```text
Paddle checkpoint: .cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
HF-compatible export: .cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2-hf
```

Text-only FP32 result:

```text
reference_dir: .cache/llavaonevision1_5/reduced_depth_fullwidth_text_reference_fp32
max_diff: 0.00355536
mean_diff: 0.00055093
last_max_diff: 0.00296354
last_mean_diff: 0.00053563
first_10_tokens_match: True
```

Text-only BF16 result:

```text
reference_dir: .cache/llavaonevision1_5/reduced_depth_fullwidth_text_reference_bf16
attn_implementation: sdpa
max_diff: 0.07031250
mean_diff: 0.01065192
last_max_diff: 0.06250000
last_mean_diff: 0.01061600
first_10_tokens_match: True
```

Multimodal image+text FP32 result:

```text
reference_dir: .cache/llavaonevision1_5/reduced_depth_fullwidth_mm_reference_fp32
max_pixels: 200704
max_diff: 0.00360462
mean_diff: 0.00043145
last_max_diff: 0.00234246
last_mean_diff: 0.00036684
first_10_tokens_match: True
```

Interpretation:

- The corrected reduced-depth full-width checkpoint passes the `1e-2` order
  forward logits target in FP32 for both text-only and multimodal image+text
  validation.
- The first 10 generated tokens match Transformers exactly in both text-only
  and multimodal FP32 validation.
- BF16 is also generation-aligned and has mean diff around `1e-2`; the larger
  BF16 max diff is treated as cross-framework BF16 numerical variance.
- The earlier full 8B multimodal logits gap remains documented separately and is
  not representative of the reduced-depth full-width acceptance path.

## Reduced-Depth Full-Width GSM8K 300-Step SFT

The corrected reduced-depth full-width model has completed the requested
GSM8K 300-step SFT run with four RTX 3090 24 GB cards using sharding stage2.

Run setup:

```text
checkpoint: .cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
config: examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_300.yaml
script: scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_300.sh
dataset: data/gsm8k_erniekit/train.jsonl
eval_dataset: data/gsm8k_erniekit/test.jsonl
GPUs: 4 cards, sharding stage2
max_steps: 300
```

Final training metrics:

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

The loss curve decreases normally throughout the 300-step run. The formal
loss-curve acceptance comparison still requires an ms-swift baseline generated
with the same reduced-depth full-width model definition.

## Compiler Performance Comparison

Inference compiler comparison has been run on the corrected reduced-depth
full-width model using one RTX 3090 card. The benchmark uses the static-friendly
`manual-lm-head` path with eager attention and a text-only sequence of length
64.

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

This passes the requested inference-side compiler improvement target.

Full fine-tuning training compiler comparison remains blocked by current
PaddleFormers static training and local-memory limitations:

```text
dynamic 4-card sharding stage2 training:
  max_steps: 30
  train_runtime: 0:01:34.60
  train_steps_per_second: 0.3171
  train_samples_per_second: 5.0735

to_static=true + 4-card sharding stage2:
  AssertionError: static training is only supported when world_size == 1 or enable_auto_parallel is set.

to_static=true + enable_auto_parallel=true:
  AssertionError: Auto parallel only support dynamic parallel now. Static parallel will be supported later.
```

Single-card training is not a viable fallback for this full-width reduced-depth
model on local RTX 3090 24 GB cards because AdamW optimizer-state creation OOMs.
A more aggressive `1L text + 1L vision` full-width checkpoint also OOMs on a
single 24 GB card while creating AdamW moment tensors, so further layer
reduction is not the right fix for full fine-tuning. The bottleneck is the
full-width vocabulary embedding/lm-head plus optimizer states.

Single-card LoRA training was added as the practical compiler-training fallback.
It keeps the corrected reduced-depth full-width base model and trains only LoRA
adapters on the language model, with `freeze_vision freeze_aligner`.

```text
dynamic config:
  examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_compile_dynamic.yaml
static config:
  examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_compile_static.yaml
log_dynamic:
  llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_compile_train_dynamic.log
log_static:
  llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_compile_train_static.log
trainable_parameters_per_device: 2,031,616
max_memory_reserved: 8.986 GB

dynamic train_steps_per_second: 5.9549
to_static train_steps_per_second: 5.9007
training speedup: -0.91%
```

The static LoRA run does apply the compiler path:

```text
to_static: True
Successfully to apply @to_static to the whole model.
```

Longer 300-step LoRA compiler comparisons were also run to reduce warmup and
logging noise:

```text
script:
  scripts/llavaonevision1_5/compare_training_compile_lora_reduced_depth_fullwidth.sh

300-step LoRA, recompute_granularity=full:
  dynamic train_steps_per_second: 7.4357
  to_static train_steps_per_second: 7.4175
  training speedup: -0.24%

300-step LoRA, recompute_granularity=None:
  dynamic train_steps_per_second: 7.6282
  to_static train_steps_per_second: 7.3416
  training speedup: -3.76%
```

Conclusion: the reduced-depth full-width path now has valid single-card
training compiler comparisons, and the static graph path is applied correctly,
but the observed LoRA training speed does not meet the 20% training-side speedup
target on local RTX 3090 hardware. The inference compiler benchmark remains
strongly positive at `41.49%`.

Acceptance risk note:

- Full fine-tuning training compiler comparison cannot currently be measured in
  the local reduced-depth full-width setup because static multi-card sharding is
  rejected by PaddleFormers, while single-card full fine-tuning OOMs during
  AdamW optimizer-state creation.
- LoRA is the only practical local single-card training compiler route. It
  updates only 2,031,616 trainable parameters, so the compiler has little
  trainable-graph work to optimize compared with the frozen full-width base
  model forward/backward and trainer/data overhead.
- The 20% training-speedup target should be rerun in the official acceptance
  environment, or reported as a known local limitation while keeping the passing
  41.49% inference speedup as the compiler evidence available locally.

## Reduced-Depth Full-Width LoRA GSM8K 300-Step SFT

The single-card LoRA 300-step GSM8K run has completed successfully on the
corrected reduced-depth full-width checkpoint. This is the practical local
single-card training path because full fine-tuning OOMs on RTX 3090 24 GB even
after reducing depth.

Run setup:

```text
checkpoint: .cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
config: examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_300.yaml
script: scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_lora_300.sh
dataset: data/gsm8k_erniekit/train.jsonl
eval_dataset: data/gsm8k_erniekit/test.jsonl
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
checkpoint: checkpoints/llavaonevision1_5-gsm8k-reduced-depth-fullwidth-lora-300
adapter_weights: peft_model-00001-of-00001.safetensors
```

The LoRA loss curve decreases normally and confirms that the corrected
reduced-depth full-width model can be trained on one local 24 GB card when the
trainable parameter set is restricted. This does not replace the formal
ms-swift baseline comparison, which still needs a matching reduced-depth
full-width ms-swift run from the client side or a separately reproduced
baseline.
