# MiMo Acceptance Status

## Done In Code

- MiMo model/config registration in PaddleFormers.
- MTP layer state tree and HF-to-Paddle AOA mapping.
- Tiny unit test skeleton for `MiMoModel` and `MiMoForCausalLM`.
- Full and reduced-depth GSM8K SFT configs.
- CI single-card smoke config and integration script.
- CE/reduced asset generation helpers.
- HF/Paddle logits and generation comparison helpers.
- Split-env tiny reference comparison helpers:
  - `scripts/mimo/dump_tiny_qwen2_reference.py`
  - `scripts/mimo/compare_paddle_reference.py`
- Compiler on/off inference and training benchmark helpers.
- Model capability matrix entry.
- MiMo LoRA target-module mapping for SFT LoRA runs.

## Verified Locally

Environment:

- Paddle side: `<paddle-env>`, Python 3.12, `paddlepaddle-gpu==3.4.0.post20260424+267502364e4`, `paddlefleet==0.3.0.dev20260425`.
- Torch side: `<torch-env>`, torch CUDA available.
- GPU used for the latest reduced-depth SFT pass: `CUDA_VISIBLE_DEVICES=2`.

Results:

- `paddleformers-cli` starts successfully with `PADDLEFORMERS_DIST_LOG=/tmp/mimo_assets/dist_log`.
- MiMo unit test passes:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unittest tests.transformers.mimo.test_modeling -v
```

Result: `Ran 22 tests`, `OK (skipped=3)`.

- Tiny and reduced-depth full-width acceptance assets can be generated locally:

```bash
PYTHON=python \
DTYPE=bfloat16 \
TOKENIZER_DIR=/path/to/MiMo-7B-Base \
OUTPUT_DIR=/tmp/mimo_assets/reduced-depth-4l-fullwidth-random \
bash scripts/mimo/prepare_reduced_assets.sh

python3 scripts/mimo/check_acceptance_assets.py \
  --tiny-dir /tmp/mimo_assets/tiny-random-mimo \
  --reduced-dir /tmp/mimo_assets/reduced-depth-4l-fullwidth-random
```

Result: tiny and reduced checkpoint directories, weights, configs, tokenizer assets, and local GSM8K data are present. The reduced checkpoint keeps original hidden width/vocab and reduces only model depth.

- Tiny and reduced-depth full-width Paddle checkpoints load and run forward on GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import paddle; from paddleformers.transformers import MiMoForCausalLM; paddle.set_device('gpu:0'); m=MiMoForCausalLM.from_pretrained('/tmp/mimo_assets/tiny-random-mimo', dtype='float32', load_checkpoint_format='sharding_io', convert_from_hf=False); m.eval(); ids=paddle.randint(0, 100, [1,8], dtype='int64'); out=m(input_ids=ids, return_dict=True); print(out.logits.shape)"

CUDA_VISIBLE_DEVICES=1 python -c "import paddle; from paddleformers.transformers import MiMoForCausalLM; paddle.set_device('gpu:0'); m=MiMoForCausalLM.from_pretrained('/tmp/mimo_assets/reduced-depth-4l-fullwidth-random', dtype='bfloat16', load_checkpoint_format='sharding_io', convert_from_hf=False); m.eval(); ids=paddle.randint(0, 100, [1,4], dtype='int64'); out=m(input_ids=ids, return_dict=True); print(out.logits.shape)"
```

Results: tiny `paddle.Size([1, 8, 151665])`; reduced-depth full-width `paddle.Size([1, 4, 151680])`.

- 10-step SFT smoke passes using local tiny checkpoint:

```bash
PATH=/path/to/paddle-env/bin:$PATH \
PADDLEFORMERS_DIST_LOG=/tmp/mimo_assets/dist_log \
CUDA_VISIBLE_DEVICES=0 SKIP_PRECISION_CHECK=1 \
config_yaml=/tmp/mimo_sft_single_smoke.yaml \
data_dir=$PWD/tests/fixtures/dummy/sft \
model_name_or_path=/tmp/mimo_assets/tiny-random-mimo \
output_dir=/tmp/mimo_assets/checkpoints/smoke \
bash tests/integration_test/mimo_sft_single_card.sh single
```

Result: `train_loss=12.9117`, checkpoint saved to `/tmp/mimo_assets/checkpoints/smoke`.

- Tiny HF-Qwen2-compatible reference vs Paddle MiMo passes:

```bash
CUDA_VISIBLE_DEVICES=2 python \
  scripts/mimo/dump_tiny_qwen2_reference.py \
  --model /tmp/mimo/tiny-random-mimo-tokenizer \
  --output-dir /tmp/mimo/reference-tiny-qwen2 \
  --dtype float32 --device cuda:0 --max-new-tokens 10 --topk 10

CUDA_VISIBLE_DEVICES=2 python \
  scripts/mimo/compare_paddle_reference.py \
  --model /tmp/mimo/tiny-random-mimo-tokenizer \
  --reference /tmp/mimo/reference-tiny-qwen2/reference.npz \
  --dtype float32 --device gpu --max-new-tokens 10 --topk 10 \
  --load-checkpoint-format sharding_io --no-convert-from-hf
```

Result: `max_diff=3.3974647521972656e-06`, `mean_diff=3.872926015446865e-07`, first 10 greedy tokens match.

- Full official MiMo FP32 forward/generation alignment passes after converting HF safetensors to Paddle native bf16 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4 python \
  scripts/mimo/convert_hf_to_paddle_native.py \
  --hf-dir /path/to/MiMo-7B-Base \
  --output-dir /path/to/MiMo-7B-Base-paddle-bf16 \
  --dtype bfloat16

CUDA_VISIBLE_DEVICES=5 python \
  scripts/mimo/dump_hf_reference.py \
  --model /path/to/MiMo-7B-Base \
  --output-dir /tmp/mimo_assets/reference-full-fp32 \
  --dtype float32 --device cuda:0 --max-new-tokens 10 --topk 10

CUDA_VISIBLE_DEVICES=6 python \
  scripts/mimo/compare_paddle_reference.py \
  --model /path/to/MiMo-7B-Base-paddle-bf16 \
  --reference /tmp/mimo_assets/reference-full-fp32/reference.npz \
  --dtype float32 --device gpu --max-new-tokens 10 --topk 10 \
  --load-checkpoint-format sharding_io --no-convert-from-hf --load-via-cpu
```

Result: `max_diff=0.003246307373046875`, `mean_diff=4.875975355389528e-05`, first 10 greedy tokens match. HF generated text: `The speed of the train is 30 miles`.

- A true-weight reduced-depth full-width checkpoint can be created from the converted official Paddle checkpoint:

```bash
python scripts/mimo/create_reduced_from_paddle_checkpoint.py \
  --source-dir /path/to/MiMo-7B-Base-paddle-bf16 \
  --output-dir /path/to/MiMo-7B-Base-reduced-4l-paddle-bf16 \
  --num-hidden-layers 4
```

Result: reduced checkpoint saved under `/path/to/MiMo-7B-Base-reduced-4l-paddle-bf16`, keeping original vocab/hidden width and reducing only decoder depth to 4 layers.

- Reduced-depth full-width true-weight GSM8K SFT completed for 300 steps on one GPU:

```bash
CUDA_VISIBLE_DEVICES=2 \
PATH=/path/to/paddle-env/bin:$PATH \
PADDLEFORMERS_DIST_LOG=/tmp/mimo_assets/dist_log \
paddleformers-cli train /tmp/mimo_reduced_real_sft_300.yaml \
  2>&1 | tee /tmp/mimo_assets/logs/mimo_reduced_real_sft_300.log
```

Result: `Exit code 0`, final `eval_loss=2.16945743560791`, `train_loss=3.152836615641912`, `Total_Tokens_per_second_per_gpu=666.939347597846`, `max_memory_reserved=35.45309829711914` GB. Checkpoint saved to `/tmp/mimo_assets/checkpoints/reduced-real-4l-300/checkpoint-300`.

- Matching reduced-depth full-width HF checkpoint for ms-swift baseline was created:

```bash
python scripts/mimo/create_reduced_from_hf_checkpoint.py \
  --source-dir /path/to/MiMo-7B-Base \
  --output-dir /path/to/MiMo-7B-Base-reduced-4l-hf-bf16 \
  --num-hidden-layers 4
```

Result: 4-layer full-width HF checkpoint saved under `/path/to/MiMo-7B-Base-reduced-4l-hf-bf16`. A second 4-layer HF checkpoint with `num_nextn_predict_layers=0` was also prepared to match PaddleFormers SFT's effective training config.

- ms-swift 300-step GSM8K baseline completed with Paddle-aligned visible hyperparameters:

```bash
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

Result: final `eval_loss=3.2244072`, `train_loss=4.205`, checkpoint saved to `/tmp/mimo_assets/ms_swift/output-reduced-4l-300-paddle-aligned/v0-20260531-014418/checkpoint-300`.

- An additional ms-swift 300-step run was completed with `--lr_scheduler_type linear` to match the LR values printed by PaddleFormers:

```bash
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
  --lr_scheduler_type linear \
  --warmup_steps 20 \
  --weight_decay 0.0 \
  --adam_beta2 0.999 \
  --max_steps 300 \
  --eval_steps 50 \
  --save_steps 100 \
  --logging_steps 1 \
  --output_dir /tmp/mimo_assets/ms_swift/output-reduced-4l-300-paddle-linear \
  --report_to none \
  --save_total_limit 1 \
  --seed 23 \
  --data_seed 23
```

Result: final `eval_loss=3.267`, `train_loss=4.266`, checkpoint saved to `/tmp/mimo_assets/ms_swift/output-reduced-4l-300-paddle-linear/v0-20260531-173801/checkpoint-300`.

- Three additional ms-swift controls were completed:

  - `--optim adamw_torch` with the same linear schedule ended at `eval_loss=3.280`, `train_loss=4.272`, so the default `adamw_torch_fused` optimizer is not the source of the gap.
  - `--dataset_shuffle false` with the same linear schedule used the original GSM8K order; the printed first sample changed to the first ERNIEKit row (`Natalia...`) and the run ended at `eval_loss=3.265`, `train_loss=4.297`, so sample shuffle order is not the source of the gap.
  - The MTP-disabled HF checkpoint (`num_nextn_predict_layers=0`) with the same linear schedule ended at `eval_loss=3.267`, `train_loss=4.266`, matching the linear run above. This confirms the remaining loss gap is not caused by MTP checkpoint structure.

Reduced-depth full-width eval loss comparison:

| Step | Paddle eval loss | ms-swift cosine | Delta | ms-swift linear | Delta | ms-swift linear no-shuffle | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 3.298718 | 4.469769 | +1.171052 | 4.521 | +1.222282 | 4.661 | +1.362282 |
| 100 | 2.797493 | 3.558206 | +0.760713 | 3.657 | +0.859507 | 3.696 | +0.898507 |
| 150 | 2.528211 | 3.311581 | +0.783370 | 3.402 | +0.873789 | 3.397 | +0.868789 |
| 200 | 2.345575 | 3.238986 | +0.893411 | 3.304 | +0.958425 | 3.294 | +0.948425 |
| 250 | 2.223719 | 3.225060 | +1.001341 | 3.271 | +1.047281 | 3.268 | +1.044281 |
| 300 | 2.169457 | 3.224407 | +1.054950 | 3.267 | +1.097543 | 3.265 | +1.095543 |

The loss curves both decrease, and Paddle reaches a lower eval loss, but the numeric curve is not yet acceptance-aligned with ms-swift. Confirmed differences already corrected or ruled out in the ms-swift runs: system prompt, `weight_decay`, `adam_beta2`, scheduler type, fused vs non-fused AdamW, sample shuffle order, and MTP vs no-MTP checkpoint structure. Additional diagnostics confirmed sampled mapped weights from the reduced HF and Paddle checkpoints match exactly. A single-sample initial-loss check using the same tokenized ms-swift sample gave HF shifted loss `14.7959` and Paddle shifted loss `14.6266`, so the initial forward/loss path is close and the larger gap appears after optimization starts.

Current explanation: ms-swift/HF normalizes the accumulated training loss by the total number of non-ignored label tokens across the gradient accumulation window, while PaddleFormers computes micro-batch mean loss and averages gradients over `gradient_accumulation_steps`. On variable-length GSM8K samples this changes per-sample weighting during optimization, so the curves can both improve while not matching numerically. A more exact full-precision/master-weight control would require a larger or freer GPU; it is not included in the local acceptance package.

- Compiler inference benchmark passes on the true-weight reduced-depth checkpoint:

```bash
CUDA_VISIBLE_DEVICES=2 \
MODEL_NAME_OR_PATH=/path/to/MiMo-7B-Base-reduced-4l-paddle-bf16 \
LOAD_CHECKPOINT_FORMAT=sharding_io \
CONVERT_FROM_HF=false \
OUTPUT_DIR=/tmp/mimo_assets/benchmarks/inference_compile_reduced_real \
bash scripts/mimo/compare_inference_compile_reduced.sh
```

Result: dynamic `10840.92 tokens/s`, to_static `17253.67 tokens/s`, speedup `59.15%`.

- Compiler full-parameter training benchmark partially completed. Dynamic full SFT passed with `Total_Tokens_per_second_per_gpu=1447.6772132213646`. Static full SFT reached forward/backward/optimizer step and then failed from local GPU memory pressure while creating AdamW accumulators:

```text
Cannot allocate 2.314453GB memory on GPU 0, 45.618591GB memory has been allocated and available memory is only 1.752563GB.
```

This indicates the remaining local full-SFT static blocker is memory availability. Because the local validation GPUs are shared and one usable card did not have enough free memory for full-parameter static optimizer states, the training compiler validation below uses LoRA as the resource-constrained fallback path.

- Compiler LoRA training benchmark passes on the true-weight reduced-depth checkpoint with `fine_tuning=lora`, `lora_rank=8`, `recompute_granularity=null`, and `to_static` toggled:

```bash
CUDA_VISIBLE_DEVICES=2 \
PATH=/path/to/paddle-env/bin:$PATH \
PADDLEFORMERS_DIST_LOG=/tmp/mimo_assets/dist_log_lora_dynamic_30 \
paddleformers-cli train /tmp/mimo_train_dynamic_lora_30.yaml \
  2>&1 | tee /tmp/mimo_assets/logs/mimo_train_compile_dynamic_lora_30.log

CUDA_VISIBLE_DEVICES=2 \
PATH=/path/to/paddle-env/bin:$PATH \
PADDLEFORMERS_DIST_LOG=/tmp/mimo_assets/dist_log_lora_static_30 \
paddleformers-cli train /tmp/mimo_train_static_lora_30.yaml \
  2>&1 | tee /tmp/mimo_assets/logs/mimo_train_compile_static_lora_30.log
```

Result: dynamic `7906.42 tokens/s`, static `8369.32 tokens/s`, speedup `5.85%`. Both runs completed 30 steps with the same final `train_loss=11.528600597381592` and peak reserved memory about `5.95GB`.

This LoRA result is used only as a local fallback validation for training-side compiler availability under the current memory constraints. It verifies that MiMo SFT can run through the static training path and produce the same loss as dynamic mode. The measured speedup is below the 20% target, which is reasonable for this fallback setting because LoRA trains only about `1.91M` parameters (`0.10%` of the model) and shifts much less work into optimizer-state updates than full-parameter SFT. The formal training compiler speedup should be rerun with full-parameter training on a freer/larger acceptance GPU; locally, that path is blocked by the OOM above.

- Static checks pass:

```bash
python3 -m py_compile scripts/mimo/*.py paddleformers/transformers/mimo/*.py tests/transformers/mimo/*.py

bash -n scripts/mimo/*.sh tests/integration_test/mimo_sft_single_card.sh
```

- MiMo scripts/configs no longer depend on helper scripts from another model migration.

## Remaining Acceptance Work

1. Run full 300-step GSM8K SFT where resources allow:

```bash
bash scripts/mimo/run_gsm8k_sft_300.sh
```

2. Investigate the remaining Paddle vs ms-swift loss gap after aligning the visible SFT hyperparameters. The current reduced-depth baseline is useful evidence but does not satisfy the formal loss-curve requirement yet.

3. Re-run full-parameter static training on a freer/larger GPU, or document the local full-parameter static OOM. Inference compiler already exceeds the 20% target locally. LoRA training is recorded as the resource-constrained fallback: it validates the static training path, but its `5.85%` speedup is not treated as satisfying the formal full-training 20% target.

4. Upload CE tiny checkpoint to an approved repo, then update the CE baseline losses and generation tokens in `scripts/regression/config.yaml`.

## Current Local Blocker

Full official HF weights are available locally under `/path/to/MiMo-7B-Base`, the Paddle native converted checkpoint is under `/path/to/MiMo-7B-Base-paddle-bf16`, the true-weight 4-layer Paddle reduced checkpoint is under `/path/to/MiMo-7B-Base-reduced-4l-paddle-bf16`, and the matching 4-layer HF reduced checkpoint for ms-swift is under `/path/to/MiMo-7B-Base-reduced-4l-hf-bf16`.

Current blockers are the reduced-depth Paddle vs ms-swift loss gap, full-parameter static training local OOM, and CE asset upload. The LoRA training compiler run is documented as a memory-constrained fallback validation rather than the formal training-speed target.
