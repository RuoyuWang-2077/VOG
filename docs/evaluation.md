# Evaluation protocol

The benchmark evaluates one DVOC adapter. Each prompt index has one fixed noise seed and one PNG. Every judge consumes the same verified PNG bytes.

## 1. Generate

```bash
torchrun --standalone --nproc_per_node=8 -m dvoc generate \
  --config configs/endpoint_j2.json \
  --adapter /path/to/checkpoint-20/lora \
  --output /path/to/eval/generated
```

Defaults: the first 1,000 test prompts, seed `20260923 + index`, 1024², 9 deterministic Euler steps, guidance 0. Supply the checkpoint's EMA `lora/` directory. Generation records the adapter hashes, prompt-list hash, model settings, initial-latent hash, and PNG hash. Pixel encoding uses `clip(image * 255, 0, 255).astype(uint8)`.

`manifest.json` stores relative image paths so the generated directory can be moved as a unit. `--resume` reuses samples with matching metadata and image hashes; conflicting protocols are rejected.

## 2. Score fixed images

Run regular metrics in the training environment:

```bash
torchrun --standalone --nproc_per_node=8 -m dvoc score \
  --config configs/endpoint_j2.json --judge regular \
  --manifest /path/to/eval/generated/manifest.json \
  --output /path/to/eval/scores
```

Run each quality judge in a separate process using the quality environment:

```bash
/path/to/quality-env/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m dvoc score \
  --config configs/endpoint_j2.json --judge qalign \
  --manifest /path/to/eval/generated/manifest.json \
  --output /path/to/eval/scores

/path/to/quality-env/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m dvoc score \
  --config configs/endpoint_j2.json --judge deqa \
  --manifest /path/to/eval/generated/manifest.json \
  --output /path/to/eval/scores
```

Set `QALIGN_MODEL_PATH` to the released `q-future/one-align` model and `DEQA_MODEL_PATH` to `zhiyuanyou/DeQA-Score-Mix3`. The reference model revisions are `dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb` and `246e219b7e3bc9b5dff5269c1a8564087aefca3f`, respectively. The supplied quality requirements preserve the older Transformers environment used by these releases.

For a local Q-Align release that hardcodes its Hub ID in tokenizer/processor construction, the scorer creates a local adapter directory with those two paths redirected to the supplied model directory. Model weights and scoring logic are unchanged; the original source directory is preserved. The released `model.score` API supplies both quality and aesthetics. DeQA uses its released `model.score` API. Both run FP16, eager attention, batch size 1, and TF32 disabled.

The regular judge uses HPSv3++ condition 0.3, CLIP, AES, and ImageReward. Each sample has an isolated reward random stream so resuming or reassigning completed samples cannot silently change dropout state. Q-Align/DeQA values must lie in [1, 5]. All metrics must be finite; missing metrics and invalid sentinels cause failure.

Add `--resume` to scorer commands. The manifest, rank count, metric configuration, and image hashes must match. JSONL is flushed per sample. A partial final JSONL line is reported explicitly rather than silently discarded.

## 3. Summarize

Use the training environment for W&B integration:

```bash
dvoc summarize --config configs/endpoint_j2.json \
  --manifest /path/to/eval/generated/manifest.json \
  --scores /path/to/eval/scores --output /path/to/eval/report
```

The report is created only after all judges cover the same complete sample set. It includes:

- `summary.json`: count, mean, sample standard deviation, and standard error per metric, plus protocols.
- `per_image.csv`: prompt index, prompt, seed, and all seven scores.
- W&B scalar summaries, a metric table, a per-image table, and sample images, when enabled.

Standard error describes variation across evaluation prompts; it is not a training-seed confidence interval. Raw scores from different metrics are not added together.

## Online training evaluation

`training/evaluation.py` performs EMA evaluation during training using the five configured online metrics, including auxiliary PickScore. It defaults to HPS condition 0.0 and records `eval/*` metrics. Formal fixed-PNG evaluation uses `benchmark/*` metrics and condition 0.3. Keep the two protocols separate when reporting results.
