# MiMo PaddleFormers Migration Notes

## Scope

This migration adds XiaomiMiMo/MiMo-7B-Base as a Qwen2-compatible decoder model with MiMo-specific configuration and MTP layer registration.

## Implemented

- `MiMoConfig` with `model_type = "mimo"` and MiMo-7B-Base defaults.
- `MiMoModel`, `MiMoForCausalLM`, and `MiMoForCausalLMPipe` Auto registration.
- MTP layers are represented in the model state tree so full HF checkpoints can map the additional weights.
- Tiny model unit tests under `tests/transformers/mimo`.
- Reduced-depth full-width asset and benchmark helpers under `scripts/mimo`.
- GSM8K 300-step SFT configs:
  - `examples/config/sft/mimo_gsm8k_300.yaml`
  - `examples/config/sft/mimo_gsm8k_reduced_depth_fullwidth_300.yaml`
  - `tests/config/benchmark/config/sft/MiMo-7B-Base.yaml`
  - `tests/config/benchmark/config/sft/MiMo-7B-Base-Reduced-Depth-FullWidth.yaml`

## Useful Commands

Create a tiny CI smoke checkpoint:

```bash
python scripts/mimo/create_tiny_random.py --output-dir ./.cache/mimo/tiny-random-mimo
```

Prepare both tiny and reduced CE assets, optionally copying a Qwen2/MiMo tokenizer:

```bash
TOKENIZER_DIR=/path/to/mimo-or-qwen2-tokenizer bash scripts/mimo/prepare_ce_assets.sh
```

Create a same-width reduced-depth checkpoint for acceptance fallback:

```bash
LAYERS=4 OUTPUT_DIR=./.cache/mimo/reduced-depth-4l-fullwidth-random bash scripts/mimo/prepare_reduced_assets.sh
```

Local Paddle native assets are loaded in training configs with:

```yaml
convert_from_hf: false
load_checkpoint_format: sharding_io
save_checkpoint_format: flex_checkpoint
```

Compare full HF and Paddle logits/generation:

```bash
python scripts/mimo/compare_forward.py --model XiaomiMiMo/MiMo-7B-Base --dtype bfloat16
```

For the official checkpoint, the current validated local path is to convert HF
safetensors to Paddle native first:

```bash
python scripts/mimo/convert_hf_to_paddle_native.py \
  --hf-dir /path/to/MiMo-7B-Base \
  --output-dir /path/to/MiMo-7B-Base-paddle-bf16 \
  --dtype bfloat16
```

Create a reduced-depth full-width checkpoint from that converted checkpoint:

```bash
python scripts/mimo/create_reduced_from_paddle_checkpoint.py \
  --source-dir /path/to/MiMo-7B-Base-paddle-bf16 \
  --output-dir ./.cache/mimo/reduced-depth-4l-fullwidth \
  --num-hidden-layers 4
```

Run 300-step GSM8K SFT on the reduced-depth checkpoint:

```bash
CUDA_VISIBLE_DEVICES=2 \
PATH=/path/to/paddle-env/bin:$PATH \
PADDLEFORMERS_DIST_LOG=/tmp/mimo_assets/dist_log \
paddleformers-cli train /tmp/mimo_reduced_real_sft_300.yaml \
  2>&1 | tee /tmp/mimo_assets/logs/mimo_reduced_real_sft_300.log
```

Validated local result for the true-weight 4-layer full-width checkpoint:
`eval_loss=2.16945743560791`, `train_loss=3.152836615641912`, and
`Total_Tokens_per_second_per_gpu=666.939347597846`.

Create the matching reduced-depth HF checkpoint and run the ms-swift baseline:

```bash
python scripts/mimo/create_reduced_from_hf_checkpoint.py \
  --source-dir /path/to/MiMo-7B-Base \
  --output-dir /path/to/MiMo-7B-Base-reduced-4l-hf-bf16 \
  --num-hidden-layers 4

CUDA_VISIBLE_DEVICES=2 swift sft \
  --model /path/to/MiMo-7B-Base-reduced-4l-hf-bf16 \
  --template qwen \
  --system 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' \
  --dataset /tmp/mimo_assets/ms_swift/gsm8k_train.jsonl \
  --val_dataset /tmp/mimo_assets/ms_swift/gsm8k_test.jsonl \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --warmup_steps 20 \
  --weight_decay 0.0 \
  --adam_beta2 0.999 \
  --max_steps 300 \
  --eval_steps 50 \
  --save_steps 100 \
  --logging_steps 1 \
  --output_dir /tmp/mimo_assets/ms_swift/output-reduced-4l-300-paddle-aligned \
  --report_to none \
  --save_total_limit 1 \
  --seed 23 \
  --data_seed 23
```

Validated local ms-swift result for the same 4-layer full-width checkpoint:
`eval_loss=3.2244072`, `train_loss=4.205`. The curve decreases but remains
higher than Paddle's curve, so this acceptance item is not fully closed yet.

A second ms-swift run with `--lr_scheduler_type linear` was also completed to
match Paddle's printed LR schedule. It ended at `eval_loss=3.267` and
`train_loss=4.266`, so scheduler mismatch is not the primary source of the
remaining loss gap. Tokenized sample inputs/labels and sampled mapped weights
match between the reduced HF and Paddle checkpoints. Follow-up controls also
ruled out `adamw_torch_fused` versus `adamw_torch` and train-sample shuffle
order: the non-fused run ended at `eval_loss=3.280`, and the no-shuffle run
ended at `eval_loss=3.265`. A single-sample initial-loss check gave HF shifted
loss `14.7959` and Paddle shifted loss `14.6266`, so the remaining gap appears
after optimization starts; the leading suspect is framework-level training
semantics, especially mixed precision/master weights and gradient/loss
normalization during gradient accumulation. A Paddle control with
`fp16_opt_level: O1` was attempted, but it OOMed at the first optimizer step
while initializing AdamW accumulators, so this cannot be isolated locally
without a larger card or further memory reductions.

Compare compiler on/off inference and training:

```bash
bash scripts/mimo/compare_inference_compile_reduced.sh
bash scripts/mimo/compare_training_compile_reduced.sh
```

Validated local inference compiler result for the true-weight reduced-depth
checkpoint: dynamic `10840.92 tokens/s`, to_static `17253.67 tokens/s`,
speedup `59.15%`.

Training compiler inference passed locally with a `59.15%` speedup. For
training, full-parameter static SFT reached the optimizer step and then hit
local GPU memory pressure while creating optimizer states. The LoRA fallback
completed dynamic and static 30-step runs with the same final loss and a
`5.85%` speedup; it is recorded as a resource-constrained static-path
validation, not as the formal full-training 20% target.

## Acceptance Items To Run With Full Assets

1. Single-card forward alignment against Transformers: target logits diff at `1e-2`.
2. Greedy generation alignment: first 10 generated tokens match Transformers.
3. GSM8K SFT for 300 steps with the hyperparameters in `examples/config/sft/mimo_gsm8k_300.yaml`. Reduced-depth Paddle and ms-swift runs are complete, but the loss curves are not numerically aligned yet.
4. CI/CE tiny model upload and CE config wiring after a PaddleFormers/tiny-random-mimo checkpoint is available.
5. Compiler on/off train and inference benchmark: inference exceeds the 20% target locally; training static mode passes with the LoRA fallback, while full-parameter static training needs a freer/larger GPU for the formal speedup target.

## Notes

MiMo's HF remote code subclasses Qwen2 and does not call the MTP layers during normal causal LM forward. PaddleFormers follows the same behavior: base logits and generation use the Qwen2-compatible main decoder, while the MTP layers are present for checkpoint compatibility and future speculative decoding work.
