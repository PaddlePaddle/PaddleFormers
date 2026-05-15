# LLaVA-OneVision-1.5 Migration Notes

This note tracks the PaddleFormers migration work for `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`.

## Source References

- PyTorch/HuggingFace reference repo: `/sda/data/Lichenyang/LLaVA-OneVision/LLaVA-OneVision-1.5`
- HF model id: `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`
- PaddleFormers target repo: `/sda/data/Lichenyang/PaddleFormers`
- Local requirement docs:
  - `PaddleFormers高热模型合入CheckList.pdf`
  - `新增模型添加CE流程.pdf`

## Reduced-Layer Scope Decision

As of 2026-05-10, follow-up validation is scoped to the reduced-layer model
because the available RTX 3090 hardware cannot reliably run the full 8B training
workload. This reduction is allowed by the client when the full model cannot run
locally. All reduced-layer results must be clearly labeled as reduced-layer
results and should not be presented as full 8B acceptance numbers.

## Acceptance Checklist

- Single-GPU forward precision:
  - Compare Paddle logits against Transformers logits.
  - Target diff: `1e-2` order.
- Generation:
  - Greedy output must be normal.
  - First 10 generated tokens should match Transformers for the selected case.
- Training loss:
  - Dataset: GSM8K.
  - Config: `examples/config/sft/full.yaml`.
  - Steps: 300.
  - Compare against ms-swift loss curve; error should fluctuate around 0.
- Unit tests:
  - Add PaddleFormers transformer tests for the new model.
- CI/CE:
  - Add reduced-layer model CI or CE coverage.
  - Add full-weight CE coverage where hardware permits.
- Compiler performance:
  - Compare training and inference speed with Paddle compiler on/off.
  - Target average improvement: over 10%; project close target: 20%.

## CE Flow Notes From PDF

- Tiny/random model should be uploaded to an AiStudio repo.
- Suggested model size: no more than 300 MB.
- CE environment:
  - A100 80G
  - Python 3.10
  - CUDA 12.6
- Required CE configurations should cover PT/SFT/DPO and LoRA variants where applicable.
- For non-MoE models, EP coverage is not required.
- Recommended deterministic environment variables:

```bash
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
```

## Implementation Phases

1. Add configuration and AutoConfig registration.
2. Port the text backbone and Rice vision tower.
3. Add conditional generation model and HF weight conversion mapping.
4. Add processor support or reuse compatible Qwen VL processor behavior.
5. Add forward-logits comparison script against Transformers.
6. Add greedy generation comparison script.
7. Add reduced-size unit tests.
8. Add CE configs and tiny/random model flow.
9. Run GSM8K SFT 300-step loss comparison.
10. Run compiler on/off performance comparison.

## Forward/Generation Comparison Command

Run inside a PaddleFormers GPU environment. For example:

```bash
cd /sda/data/Lichenyang/PaddleFormers

docker run --gpus all --name paddleformers-llava-ov15 --rm -it \
  -v "$PWD":/work \
  -v /sda/data/Lichenyang/hf-cache:/root/.cache/huggingface \
  -w=/work --shm-size=512G --network=host \
  -e HF_ENDPOINT=https://hf-mirror.com \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5 \
  /bin/bash
```

Then install the editable checkout inside the container:

```bash
python -m pip install -e '.[paddlefleet]' \
  --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/ \
  --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Run the comparison:

```bash
python scripts/llavaonevision1_5/compare_forward.py \
  --model lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --prompt "Describe this image briefly." \
  --max-new-tokens 16 \
  --topk 10
```

Expected acceptance targets:

- `max_diff` should be in the `1e-2` order.
- `first_10_tokens_match` should be `True` for the selected greedy case.
- The HF model's `generation_config.json` uses `repetition_penalty=1.05`. The Paddle reference-comparison script mirrors that by default; pass `--repetition-penalty 1.0` only when intentionally comparing raw greedy decoding without the HF generation config.

## Tiny/Random Checkpoint Command

```bash
python scripts/llavaonevision1_5/create_tiny_random.py \
  --output-dir ./tiny-random-llavaonevision1_5
```

The resulting directory can be used as the starting point for the tiny model repo required by the CE flow.

## Local Smoke Test Commands

The following commands were run in the Paddle Docker image:

```bash
python -m unittest \
  tests.transformers.llavaonevision1_5.test_configuration \
  tests.transformers.llavaonevision1_5.test_modeling \
  -v
```

Result: `Ran 11 tests ... OK`.

The added unit coverage includes:

- `AutoModelForConditionalGeneration.from_config(...)` resolving the base `LLaVAOneVision1_5Model` alias.
- Saving a tiny `LLaVAOneVision1_5ForConditionalGeneration` checkpoint and reloading it through `AutoModelForConditionalGeneration.from_pretrained(...)`.

```bash
python scripts/llavaonevision1_5/create_tiny_random.py \
  --output-dir /work/tiny-random-llavaonevision1_5
```

Result: tiny checkpoint saved successfully.

```bash
python -c "import paddle; from paddleformers.transformers import LLaVAOneVision1_5ForConditionalGeneration; m=LLaVAOneVision1_5ForConditionalGeneration.from_pretrained('/work/tiny-random-llavaonevision1_5'); m.eval(); ids=paddle.randint(0,90,[1,5], dtype='int64'); out=m(input_ids=ids); print(out.logits.shape)"
```

Result: `paddle.Size([1, 5, 99])`.

```bash
python -c "import paddle; from paddleformers.transformers import AutoModelForConditionalGeneration; m=AutoModelForConditionalGeneration.from_pretrained('/work/tiny-random-llavaonevision1_5'); m.eval(); ids=paddle.randint(0,90,[1,5], dtype='int64'); print(m(input_ids=ids).logits.shape)"
```

Result: AutoModel registration resolves to `LLaVAOneVision1_5ForConditionalGeneration` and returns `paddle.Size([1, 5, 99])`.

## Real Checkpoint Alignment Notes

The Transformers reference bundle was generated successfully with the PyTorch/NVIDIA container:

```bash
python scripts/llavaonevision1_5/dump_hf_reference.py \
  --model lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --prompt "Describe this image briefly." \
  --max-new-tokens 16 \
  --topk 10 \
  --output-dir /sda-work/llavaonevision1_5_reference
```

Result:

- Reference directory: `/sda/data/Lichenyang/llavaonevision1_5_reference`
- First 10 generated token ids: `[[641, 279, 6802, 358, 646, 1490, 264, 5220, 11699, 389]]`
- Decoded prefix: `In the picture I can see a woman sitting on the sand and there is a`

Paddle real checkpoint loading observations:

- Direct GPU flex checkpoint loading from the HF checkpoint reaches the AOA load stage but OOMs on a 24 GB RTX 3090:
  - Fails while allocating an extra `96 MB`.
  - GPU memory at failure: about `23.46 GB` allocated, about `95 MB` available.
- CPU flex checkpoint loading can complete the HF -> Paddle weight load, which indicates the AOA rules are at least structurally loadable.
- CPU forward with `bfloat16` fails because CPU `conv2d` does not support `bfloat16`.
- CPU forward with `float32` can proceed past vision and RMSNorm after disabling fused RMSNorm in the validation script, but full 8B CPU forward is too slow for routine iteration.
- A CPU conversion can produce a native Paddle checkpoint when saving with:
  - `save_checkpoint_format="naive"`
  - `save_to_hf=False`
  - `safe_serialization=True`
- The usable converted checkpoint is currently:
  - `/sda/data/Lichenyang/llavaonevision1_5_paddle_native_naive_bf16_v2`
  - It has native keys such as `model.language_model.*` and `model.visual.*`.
- A config bug was fixed during alignment:
  - The top-level `tie_word_embeddings` must follow `text_config.tie_word_embeddings`.
  - The HF config has `text_config.tie_word_embeddings=False`, and an accidental top-level `True` incorrectly tied `lm_head` to token embeddings during initialization.
- A BF16 comparison bug was fixed in `scripts/llavaonevision1_5/compare_paddle_reference.py`:
  - Paddle BF16 tensors must be cast to float32 before calling `.numpy()`.
  - Calling `.numpy().astype("float32")` reads BF16 storage as uint16 bit patterns and creates fake `~1e4` logits.

Latest real-checkpoint validation:

- Native checkpoint load on GPU succeeds and uses all weights.
- Local unit tests still pass after the default `sdpa` and generation-cache changes:
  - `Ran 9 tests ... OK`
- Pure text BF16 validation:
  - `max_diff: 0.40625`
  - `mean_diff: 0.053196684`
  - Last-token top ids are nearly identical, with the same top-5 tokens.
- Full image BF16 validation:
  - After matching the HF remote-code 1D `cache_position` behavior and defaulting text/vision attention to `sdpa`: `max_diff: 30.25`, `mean_diff: 0.48766083`.
  - Before that fix, the Paddle port incorrectly used Qwen-VL-style 3D mRoPE positions and had `mean_diff: 2.27211094`.
  - Disabling fused RMSNorm does not materially change the result: `max_diff: 29.9375`, `mean_diff: 0.48681486`.
  - Manually computing the final `lm_head` matmul in FP32 also does not improve alignment:
    - default lm head: `max_diff: 30.25`, `mean_diff: 0.48766083`, `last_mean: 0.05362026`
    - FP32 lm head: `max_diff: 30.20581436`, `mean_diff: 0.48778811`, `last_mean: 0.05385083`
  - The first prefill token already matches HF raw logits: both choose token `641`.
  - HF raw full-recompute for `prompt + 641` chooses token `419`, while HF `generate()` chooses token `279` because `generation_config.json` applies `repetition_penalty=1.05`.
  - Paddle raw greedy generation without the HF generation config chooses: `[[641, 419, 2168, 11, 582, 646, 1490, 264, 5220, 11699]]`.
  - Paddle generation with `repetition_penalty=1.05` matches the HF first 10 tokens exactly: `[[641, 279, 6802, 358, 646, 1490, 264, 5220, 11699, 389]]`.
  - HF first generated tokens: `[[641, 279, 6802, 358, 646, 1490, 264, 5220, 11699, 389]]`.
- HF attention implementation notes:
  - Loading the HF model without forcing `attn_implementation` reports `text_config._attn_implementation == "eager"` and `vision_config._attn_implementation == "eager"`.
  - A second HF reference bundle forced with `attn_implementation="sdpa"` produced the same first 10 generated tokens.
  - Comparing Paddle against the forced HF SDPA reference still gives `max_diff: 30.25`, `mean_diff: 0.48766083`, so the remaining gap is not explained only by HF eager vs SDPA reference selection.
- Attention implementation probe on the full-image reference:
  - `text=sdpa, vision=sdpa`: `max_diff: 30.25`, `mean_diff: 0.48766083`, `last_mean: 0.05362026`, `last_max: 0.5`.
  - `text=eager, vision=sdpa`: `max_diff: 29.875`, `mean_diff: 0.49618867`, `last_mean: 0.06097588`, `last_max: 0.53125`.
  - `text=sdpa, vision=eager`: `max_diff: 30.78125`, `mean_diff: 0.50233340`, `last_mean: 0.04101007`, `last_max: 0.28125`.
  - `text=eager, vision=eager`: `max_diff: 30.90625`, `mean_diff: 0.51293838`, `last_mean: 0.10198803`, `last_max: 0.609375`.
- Visual-path probe results:
  - `position_ids` and `rope_delta` match exactly when explicitly using the old 3D probe path.
  - HF visual embeddings vs Paddle visual embeddings: `max_diff: 2.7480469`, `mean_diff: 0.021572137`.
  - Forcing vision `sdpa` gives a slightly better visual embedding mean diff: `mean_diff: 0.020837113`.
  - Patch embedding and class-token insertion match exactly; pre-layernorm is effectively identical.
  - Visual tower drift accumulates through the 24 Rice blocks: block 0 `mean_diff ~= 2.74e-4`, block 23 `mean_diff ~= 1.96e-2`.
- Rice vision block-0 submodule probe:
  - `patch`: `max=0`, `mean=0`
  - `pre_ln`: `max=0.00024414`, `mean ~= 3.42e-10`
  - `norm1`: `max=0.00390625`, `mean ~= 3.60e-09`
  - `attn`: `max=0.00390625`, `mean ~= 5.83e-05`
  - `after_attn`: `max=0.0078125`, `mean ~= 5.83e-05`
  - `norm2`: `max=0.03125`, `mean ~= 1.76e-04`
  - `fc1`: `max=0.03125`, `mean ~= 8.21e-04`
  - `act`: `max=0.015625`, `mean ~= 8.89e-05`
  - `mlp`: `max=0.0078125`, `mean ~= 2.17e-04`
  - `block_out`: `max=0.0078125`, `mean ~= 2.44e-04`
  - Interpretation: Rice block 0 starts close; the full visual gap comes from small BF16/operator differences accumulating across all 24 vision blocks rather than an early structural mismatch.
- Rice vision block-23 same-input probe:
  - Feeding HF block-23 input directly into Paddle block 23 gives:
    - `norm1 mean ~= 5.38e-09`
    - `attn mean ~= 2.67e-04`
    - `after_attn mean ~= 2.63e-04`
    - `norm2 mean ~= 1.49e-04`
    - `fc1 mean ~= 9.32e-04`
    - `act mean ~= 1.33e-04`
    - `mlp mean ~= 0.001263`
    - `block_out mean ~= 0.001401`
  - Interpretation: even the last Rice block is close when given the same input. The larger final visual embedding gap is cumulative, not a single-block structural error.
- Precision/attention variant probes:
  - Casting only the Paddle visual tower to FP32 made the full-logits diff worse against the BF16 HF reference:
    - `visual_fp32 max_diff: 32.90625`, `mean_diff: 0.66678005`, `last_mean: 0.07826272`
  - A Rice-specific eager branch matching the HF formula was added for fidelity, but the real full-model eager path still does not improve over SDPA:
    - all eager: `max_diff: 30.90625`, `mean_diff: 0.51293838`
    - text sdpa + vision eager: `max_diff: 30.78125`, `mean_diff: 0.50233340`, `last_mean: 0.04101007`
  - Current best full-image setting remains text/vision `sdpa`: `max_diff: 30.25`, `mean_diff: 0.48766083`.
- Last-token/logits-to-keep probe:
  - `scripts/llavaonevision1_5/compare_paddle_reference.py` now supports `--logits-to-keep` and prints last-token diff.
  - Current best generation-preserving setting:
    - `--logits-to-keep 1`: `max_diff: 0.5`, `mean_diff: 0.05361148`
    - `last_max_diff: 0.5`, `last_mean_diff: 0.05361148`
    - first 10 generated tokens still match exactly with `repetition_penalty=1.05`.
  - Disabling fused RMSNorm improves last-token diff but breaks the first-10-token generation match:
    - `last_mean_diff ~= 0.04438`
    - Paddle first 10 tokens become `[[641, 279, 2168, 582, 646, 1490, 1052, 374, 264, 5220]]`
  - Therefore `fuse_rms_norm=True` remains the default for now to preserve the already-passing generation acceptance item; `--disable-fused-rms-norm` is kept as a diagnostic option.
- Text RoPE alignment update:
  - The Paddle text path now supports the HF-style 2D `position_ids` path and uses plain `apply_rotary_pos_emb` when the rotary embedding is 2D.
  - This removes the unnecessary Qwen-VL-style mRoPE expansion for the default LLaVA-OneVision-1.5 forward path.
  - Validation result is unchanged but generation remains aligned:
    - `--logits-to-keep 1`: `last_mean_diff: 0.05361148`
    - `first_10_tokens_match: True`
- Decoder probe with HF image embeddings and 1D positions:
  - `max_diff: 17.8125`, `mean_diff: 0.14942159`.
  - Last-token diff is much smaller: `max ~= 0.484375`, `mean ~= 0.0581506`, with nearly identical top tokens.
  - With language `sdpa`, decoder-only alignment improves to `max_diff: 15.4296875`, `mean_diff: 0.12983358`, `last_mean: 0.035441652`, `last_max: 0.2421875`.
- Text layer-0 weight/layout probe:
  - Native Paddle fused `qkv_proj.weight` exactly matches the expected grouped layout:
    - `qkv grouped expected: max=0`, `mean=0`
    - flat `[q, k, v]` layout is intentionally different: `mean ~= 0.02662`
  - Native Paddle fused `up_gate_proj.weight` exactly matches HF `[gate, up]` after transpose:
    - `mlp gate+up expected: max=0`, `mean=0`
    - `[up, gate]` order is intentionally different: `mean ~= 0.02776`
  - This rules out checkpoint conversion order errors for decoder layer 0 fused qkv and fused MLP.
- Text layer-0 submodule probe on the reference prompt, using token embeddings only:
  - The manual probe must use the checkpoint `rope_theta=1000000.0`; using the class default by mistake produced misleading `~0.05385` layer-output mean diff.
  - `hidden0`: `max=0`, `mean=0`
  - first RMSNorm output `ln1`: `max=0`, `mean=0`
  - `q_pre`: `max=0.00390625`, `mean=0.00002287`
  - `k_pre`: `max=0.00390625`, `mean=0.00002945`
  - `v_pre`: `max=0.00097656`, `mean=0.00002277`
  - `q_norm`: `max=0.125`, `mean=0.00231563`
  - `k_norm`: `max=2.0`, `mean=0.00371573`
  - `attn_out`: `max=0.015625`, `mean=0.00011993`
  - `after_attn`: `max=0.015625`, `mean=0.00011992`
  - `ln2`: `max=0.01025391`, `mean=0.00043710`
  - `gate`: `max=0.046875`, `mean=0.00116169`
  - `up`: `max=0.03125`, `mean=0.00100414`
  - `mlp_out`: `max=0.0625`, `mean=0.00102616`
  - `layer_out`: `max=0.125`, `mean=0.00106182`
  - Same-input attention probe with HF `q_norm/k_norm/v_pre` and correct RoPE theta gives `attn_out max=0.001953125`, `mean=5.2e-08`.
  - Same-input MLP probe with HF `ln2` gives exact `gate/up` and `mlp_out mean=0.00035856`.
  - Interpretation: decoder layer 0 is close when RoPE parameters are correct. The current full-model gap is more likely dominated by Rice visual-output drift and downstream decoder amplification, not by decoder layer-0 weight layout or basic attention/MLP formulas.
- Current remaining alignment gap is concentrated in accumulated Rice vision BF16 drift plus long multimodal decoder amplification, not in checkpoint loading, image token placement, or decoder layer-0 fused weight layout.

Next engineering target:

- Forward-logits alignment remains the main blocker for the `1e-2` acceptance target.
- Preserve HF generation config during checkpoint conversion/export, especially `repetition_penalty=1.05`, because it is required for first-10-token generation parity.
- Continue layer-by-layer vision checks and test precision/attention variants for the Rice tower, because the decoder layer-0 and fused-weight layouts now look close.
- After forward diff is reduced, run GSM8K SFT/loss-curve and compiler-performance validations.

## Current Status

- Configuration scaffold has been added:
  - `paddleformers/transformers/llavaonevision1_5/configuration.py`
  - `paddleformers/transformers/llavaonevision1_5/__init__.py`
- Paddle-native Rice vision tower scaffold has been added:
  - `paddleformers/transformers/llavaonevision1_5/modeling.py`
- Paddle-native text model and conditional generation scaffolds have been added:
  - `LLaVAOneVision1_5TextModel`
  - `LLaVAOneVision1_5Model`
  - `LLaVAOneVision1_5ForConditionalGeneration`
- AutoConfig registration has been added:
  - `paddleformers/transformers/__init__.py`
  - `paddleformers/transformers/auto/configuration.py`
- AutoModel mapping has been added:
  - `paddleformers/transformers/auto/modeling.py`
- Forward and generation comparison script has been added:
  - `scripts/llavaonevision1_5/compare_forward.py`
  - This script intentionally imports both Transformers/PyTorch and PaddleFormers. It is a validation tool, not model implementation code.
- Tiny/random checkpoint helper has been added:
  - `scripts/llavaonevision1_5/create_tiny_random.py`
  - This is intended for reduced-size CI/CE smoke coverage before uploading a tiny model repo.
- Minimal configuration test scaffold has been added:
  - `tests/transformers/llavaonevision1_5/test_configuration.py`
- Minimal Rice vision modeling test scaffold has been added:
  - `tests/transformers/llavaonevision1_5/test_modeling.py`
- Syntax check passed for the new Python files.
- Paddle Docker runtime smoke passed:
  - Configuration tests.
  - Text-only conditional generation shape test.
  - Image-token conditional generation shape test.
  - Rice vision forward shape tests.
  - Tiny/random checkpoint save and reload.
  - AutoModelForConditionalGeneration tiny checkpoint load.
- The native converted checkpoint now loads on GPU; the next blocker for full acceptance is reducing the real multimodal forward/generation gap in the vision path.

## CI/CE and Benchmark Entries

The migration now includes the engineering entries needed for follow-up
acceptance runs:

- `tests/config/ci/llavaonevision1_5_sft_single.yaml`
- `tests/integration_test/llavaonevision1_5_sft_single_card.sh`
- `.github/workflows/fleet-model-test.yml`
- `tests/config/benchmark/config/sft/LLaVA-OneVision-1.5-8B-Instruct.yaml`
- `tests/config/benchmark/config/sft/LLaVA-OneVision-1.5-Reduced-4L-512H.yaml`
- `examples/config/sft/llavaonevision1_5_gsm8k_300.yaml`
- `scripts/llavaonevision1_5/benchmark_inference.py`
- `scripts/llavaonevision1_5/compare_inference_compile.sh`
- `scripts/llavaonevision1_5/check_acceptance_assets.py`
- `scripts/llavaonevision1_5/convert_gsm8k_to_erniekit.py`
- `scripts/llavaonevision1_5/prepare_ce_assets.sh`
- `scripts/llavaonevision1_5/run_gsm8k_sft_300.sh`
- `scripts/llavaonevision1_5/run_gsm8k_sft_reduced_300.sh`
- `scripts/llavaonevision1_5/compare_inference_compile_reduced.sh`
- `scripts/llavaonevision1_5/compare_training_compile_reduced.sh`

The CI path mirrors the existing Qwen3-VL single-card CE flow and expects a
tiny checkpoint at:

```text
$CACHE_DIR/llavaonevision1_5/tiny-random-llavaonevision1_5
```

The benchmark config is set to `max_steps: 300`, matching the requested SFT
step count. Formal loss-curve and compiler performance acceptance still require
the official benchmark dataset, ms-swift baseline logs, and the CI precision
baseline file to be available in the target environment.

Local asset check:

- Tiny model weights/config exist under `tiny-random-llavaonevision1_5`.
- Tiny tokenizer assets have been copied from a local Qwen3-VL tokenizer, and
  the tiny config was regenerated with the matching tokenizer vocab size.
- The reduced CE config uses text-only SFT data under
  `tests/fixtures/dummy/sft` and `template: default`, so it does not require
  the VL image fixture or an image processor.
- Local reduced CE smoke passed on RTX 3090 GPU 1 with:
  `SKIP_PRECISION_CHECK=1 bash -x PaddleFormers/tests/integration_test/llavaonevision1_5_sft_single_card.sh single`.
- The run completed 10 SFT steps, saved a checkpoint, and printed
  `***** train metrics *****` with `train_loss: 12.9211`.
- Local reduced LoRA CE smoke passed on RTX 3090 GPU 2 with:
  `SKIP_PRECISION_CHECK=1 bash -x PaddleFormers/tests/integration_test/llavaonevision1_5_sft_single_card.sh lora_single`.
- The LoRA smoke completed 10 SFT steps, saved adapter weights, and printed
  `***** train metrics *****` with `train_loss: 12.9267`.
- The workflow now includes a separate
  `LLaVA-OneVision-1.5-LoRA-single-card` entry in addition to the base SFT
  single-card entry.
- Official CI still needs the precision baseline files uploaded:
  `llavaonevision1_5_sft_single_card_gt_loss.txt` and
  `llavaonevision1_5_lora_single_card_gt_loss.txt`. Without them, the scripts
  correctly reach the BOS precision download step and receive 404.
- GSM8K was downloaded locally with HuggingFace `datasets.save_to_disk()` at
  `/sda/data/Lichenyang/datasets/gsm8k`.
- `scripts/llavaonevision1_5/convert_gsm8k_to_erniekit.py` now supports both
  JSONL input and HuggingFace `save_to_disk` directories via `--split`.
- GSM8K has been converted to erniekit SFT JSONL:
  - `/sda/data/Lichenyang/PaddleFormers/data/gsm8k_erniekit/train.jsonl`: 7473 rows
  - `/sda/data/Lichenyang/PaddleFormers/data/gsm8k_erniekit/test.jsonl`: 1319 rows
- `scripts/llavaonevision1_5/check_acceptance_assets.py` passes for the tiny CE
  checkpoint and converted GSM8K train/test files.
- The formal 300-step GSM8K loss-curve comparison still needs the ms-swift
  baseline log for the same dataset and hyperparameters.
- A 1-step GSM8K SFT preflight was attempted on RTX 3090 GPU 1 with
  `max_steps=1`, `max_seq_len=512`, and the converted 8B checkpoint. The data
  paths and training argument parsing are now valid, but full-model training is
  currently blocked at checkpoint loading:
  - `flex_checkpoint` loading reaches AOA but fails on
    `model.embed_tokens.weight -> model.language_model.embed_tokens.weight` shape
    propagation.
  - `naive` loading is now allowed by `TrainingArguments`, but this path exposes
    expected transposed weight mismatches for Rice visual block weights and text
    `down_proj` weights.
  - Interpretation: the converted checkpoint is sufficient for the dedicated
    inference/alignment scripts, but the SFT training CLI still needs either a
    fully training-compatible flex checkpoint export or a trainer-side loading
    path that applies the same HF-to-Paddle transpose rules.

## Reduced-Layer GSM8K Validation

Because full 8B SFT is not practical on a 24 GB RTX 3090, a reduced-layer local
validation path was added for training-chain coverage. This follows the client
side allowance that layer count can be reduced when the full model cannot fit
local hardware.

Reduced checkpoint:

- Path: `/sda/data/Lichenyang/PaddleFormers/.cache/llavaonevision1_5/reduced-random-llavaonevision1_5-4l-512h`
- Text decoder: 4 layers, hidden size 512, intermediate size 1536
- Vision tower: 4 layers, hidden size 256, intermediate size 768
- Tokenizer: copied from the local LLaVA/Qwen-compatible tokenizer assets
- Size: about 669 MB; this is for local validation, not the <=300 MB CE tiny repo.

Reduced GSM8K config:

- `examples/config/sft/llavaonevision1_5_gsm8k_reduced_300.yaml`
- Uses converted GSM8K erniekit JSONL data under `data/gsm8k_erniekit`
- Uses `template: default` for text-only GSM8K. The multimodal
  `llavaonevision1_5` template expects an image processor and is not appropriate
  for this pure-text local SFT smoke.
- `max_seq_len: 512`, `max_steps: 300`.

Validation result on RTX 3090 GPU 1:

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

- Training log: `llavaonevision1_5_gsm8k_reduced_300.log`
- Loss CSV: `llavaonevision1_5_gsm8k_reduced_300_loss.csv`
- Output checkpoint: `checkpoints/llavaonevision1_5-gsm8k-reduced-sft-300`

A few GSM8K examples are skipped by the `max_seq_len=512` truncation strategy.
This is acceptable for the reduced local training-chain validation. Use
`max_seq_len=1024` if a less truncated reduced run is needed.

Reduced compiler performance probes on RTX 3090 GPU 1:

```text
Inference dynamic tokens_per_sec: 11721.84
Inference to_static tokens_per_sec: 51104.16
Inference speedup: 335.96%

Training dynamic train_steps_per_second: 7.6398
Training to_static train_steps_per_second: 7.6492
Training speedup: 0.12%

Simple average train/infer speedup: 168.04%
```

Notes:

- Inference uses `scripts/llavaonevision1_5/compare_inference_compile_reduced.sh`.
- Training uses `scripts/llavaonevision1_5/compare_training_compile_reduced.sh` with `MAX_STEPS=30`.
- Training `to_static` is confirmed by the log message
  `Successfully to apply @to_static to the whole model.`
- The reduced training compiler probe is effectively flat on this short run;
  the average exceeds 20% because reduced inference benefits strongly from
  `to_static`.

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

Short text-only inference benchmark result on RTX 3090 GPU 0 using
`--benchmark-mode manual-lm-head` and `--attn-implementation eager`:

```text
seq_len: 64
steps: 5
dynamic tokens_per_sec:   1627.63
to_static tokens_per_sec: 2103.48
speedup: 29.24%
```

The static-friendly benchmark path includes the vocabulary projection via a
manual lm-head matmul and avoids dynamic config access inside `LMHead.forward`.
The full default model path still fails before producing a speed number:

- SDPA path: `attn_mask_startend_row_indices` becomes `UndefinedVar` during
  static conversion.
- Eager attention path: static conversion reaches `lm_head.forward` and fails
  with a recursion error through `configuration_utils.__getattribute__`.

## Reduced-Depth Full-Width Scope Correction

The reduced validation scope has been corrected after confirming that the client
allowed reducing layer count but did not explicitly allow reducing hidden width.
The old `4L-512H` reduced checkpoint remains useful only as a local debugging
artifact and should not be treated as the primary acceptance model.

Primary reduced-depth full-width checkpoint:

```text
.cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
```

This checkpoint keeps the original text/vision widths and vocabulary while
reducing depth only:

```text
text hidden_size: 4096
text intermediate_size: 12288
text num_hidden_layers: 4
vision hidden_size: 1024
vision intermediate_size: 4096
vision depth: 4
vocab_size: 151936
```

A 4-card sharding stage2 SFT smoke has passed locally on RTX 3090 cards:

```text
max_steps: 1
train_loss: 13.0345
train_runtime: 0:00:04.17
current_memory_allocated: 13.63 GB
max_memory_reserved: 19.92 GB
```

Single-card full SFT and two-card variants still OOM during AdamW optimizer-state
creation. Therefore the corrected reduced-depth full-width 300-step local SFT
should use four 24 GB GPUs with sharding stage2:

```bash
GPUS=0,1,2,3 bash scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_300.sh
```

New primary entrypoints:

- `examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_300.yaml`
- `scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_300.sh`
- `tests/config/benchmark/config/sft/LLaVA-OneVision-1.5-Reduced-Depth-FullWidth.yaml`

The ms-swift baseline should be generated under this same reduced-depth
full-width definition; the old `4L-512H` baseline is not directly comparable.

## Reduced-Depth Full-Width Precision Result

The corrected reduced-depth full-width checkpoint has passed single-card
forward/generation validation against Transformers.

Text-only FP32:

```text
max_diff: 0.00355536
mean_diff: 0.00055093
last_max_diff: 0.00296354
last_mean_diff: 0.00053563
first_10_tokens_match: True
```

Multimodal image+text FP32:

```text
max_diff: 0.00360462
mean_diff: 0.00043145
last_max_diff: 0.00234246
last_mean_diff: 0.00036684
first_10_tokens_match: True
```

BF16 text-only:

```text
max_diff: 0.07031250
mean_diff: 0.01065192
last_max_diff: 0.06250000
last_mean_diff: 0.01061600
first_10_tokens_match: True
```

This resolves the previous forward-logits blocker for the reduced-depth
full-width validation path. The primary remaining reduced-path item is the
GSM8K 300-step loss comparison against an ms-swift baseline generated with the
same reduced-depth full-width definition.

## Reduced-Depth Full-Width GSM8K 300-Step Result

The corrected reduced-depth full-width GSM8K SFT run has completed successfully
on four RTX 3090 24 GB cards with sharding stage2.

```text
exit_code: 0
max_steps: 300
train_runtime: 0:25:24.03
train_loss: 4.131325361728668
train_samples_per_second: 3.1495
train_steps_per_second: 0.1968
final_eval_loss: 2.947042942047119
final_eval_ppl: 19.04953976659703
max_memory_allocated_per_rank: 17.407671213150024 GB
max_memory_reserved_per_rank: 22.237423181533813 GB
```

Eval loss decreased across the run:

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

This satisfies the local reduced-depth full-width 300-step run requirement. The
only missing loss-curve item is the external ms-swift baseline under the same
reduced-depth full-width configuration.

## Reduced-Depth Full-Width Compiler Result

Inference compiler comparison has passed on the corrected reduced-depth
full-width checkpoint:

```text
script: scripts/llavaonevision1_5/compare_inference_compile_reduced.sh
model: .cache/llavaonevision1_5/reduced-depth-4l-fullwidth-random-v2
seq_len: 64
steps: 50
warmup_steps: 10
dtype: bfloat16
benchmark_mode: manual-lm-head
attn_implementation: eager
dynamic tokens_per_second: 9729.92
to_static tokens_per_second: 13766.42
speedup: 41.49%
```

Full fine-tuning training compiler comparison could not be completed in the
local reduced-depth full-width setup because PaddleFormers currently rejects
static training with the required multi-card sharding path:

```text
dynamic 4-card sharding stage2:
  max_steps: 30
  train_runtime: 0:01:34.60
  train_steps_per_second: 0.3171
  train_samples_per_second: 5.0735

to_static=true + 4-card sharding stage2:
  AssertionError: static training is only supported when world_size == 1 or enable_auto_parallel is set.

to_static=true + enable_auto_parallel=true:
  AssertionError: Auto parallel only support dynamic parallel now. Static parallel will be supported later.
```

Single-card full fine-tuning is not a practical fallback on the local RTX 3090
24 GB cards. The 4-layer full-width model OOMs during AdamW optimizer-state
creation, and even a `1L text + 1L vision` full-width checkpoint OOMs at the
same optimizer-state stage. Further layer reduction is therefore not the right
lever for full fine-tuning; the remaining single-card bottleneck is the
full-width embedding/lm-head plus AdamW moment tensors.

As a practical single-card training compiler fallback, LoRA was run on the
corrected reduced-depth full-width checkpoint with `freeze_vision freeze_aligner`
and language-model LoRA target modules. A short 30-step probe showed that static
compilation is valid for this path, but did not produce a speedup:

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

The static run confirms compiler application:

```text
to_static: True
Successfully to apply @to_static to the whole model.
```

Longer 300-step LoRA compiler comparisons were also run to avoid warmup and
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

This gives us valid single-card training compiler comparisons for the
reduced-depth full-width LoRA setting, and they consistently show no positive
training speedup on local RTX 3090 hardware even though `to_static` is applied.
Inference compiler speedup remains passing at `41.49%`.

Current interpretation for acceptance discussion:

- The inference compiler metric is satisfied on the corrected reduced-depth
  full-width checkpoint.
- Full fine-tuning training compiler measurement is blocked locally by
  PaddleFormers' current static multi-card limitation and single-card AdamW OOM.
- LoRA training is the only local single-card training compiler path, but LoRA
  updates only about 2.03M trainable parameters while most of the workload is the
  frozen full-width base model forward/backward and trainer/data overhead; this
  workload does not show a 20% static-graph gain on RTX 3090.
- A formal 20% training-speedup claim should therefore be rerun in the official
  acceptance hardware/software environment, or the training compiler item should
  be recorded as a known local limitation while accepting the passing inference
  compiler result.

Reduced-depth full-width LoRA 300-step GSM8K SFT has also completed on a single
RTX 3090 24 GB card:

```text
config: examples/config/sft/llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_300.yaml
script: scripts/llavaonevision1_5/run_gsm8k_sft_reduced_depth_fullwidth_lora_300.sh
log: llavaonevision1_5_gsm8k_reduced_depth_fullwidth_lora_300.log
checkpoint: checkpoints/llavaonevision1_5-gsm8k-reduced-depth-fullwidth-lora-300
trainable_parameters_per_device: 2,031,616
train_runtime: 0:05:47.16
train_loss: 6.198721834818522
train_steps_per_second: 0.8641
final_eval_loss: 4.883253574371338
final_eval_ppl: 132.05963150093888
max_memory_reserved: 8.986 GB
```

The LoRA eval loss decreases across the 300-step run:

```text
step 50:  6.7881317138671875
step 100: 5.820539951324463
step 150: 5.516609191894531
step 200: 5.231670379638672
step 250: 4.969984531402588
step 300: 4.883253574371338
```

This is the practical local single-card training route for the corrected
reduced-depth full-width model. Full fine-tuning still requires multi-card
sharding on the available RTX 3090 hardware because AdamW optimizer-state
creation OOMs on one 24 GB card.
