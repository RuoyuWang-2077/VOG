# DVOC

**Reward-guided diffusion fine-tuning with latent targets and sparse Euler gradients.**

[中文说明](README.zh-CN.md) · [Algorithm](docs/algorithm.md) · [Evaluation](docs/evaluation.md) · [Validation](docs/validation.md)

DVOC improves a sampled clean latent with a bounded reward-gradient update, then fits the diffusion model to that target through selected states of its cached trajectory. This repository provides the Z-Image-Turbo implementation, LoRA training, complete checkpoint continuation, seven-metric evaluation, and Weights & Biases integration.

## Method

For a sampled endpoint `y`, DVOC constructs a detached displacement `delta` using normalized reward ascent inside a ball of radius `rho * ||y||₂`. It recomputes velocities at `K` cached states and composes an Euler endpoint `y_hat`:

```python
connected = y_hat + (y - y_hat).detach()
target = (y + delta).detach()
normalizer = mean_abs(connected - target).detach().clamp(min=1e-5)
loss = mean((connected - target) ** 2 / normalizer)
```

The forward value of `connected` is the sampled endpoint. Its gradient passes through the selected velocity predictions. The cached states and target are detached. There is no `N/K` rescaling.

The default recipe uses **K=3, J=2**, 9 deterministic Euler steps, 1024×1024 images, guidance 0, and LoRA rank 32 / alpha 64. The training reward is `0.03846 * HPSv3++ + CLIP`, with HPS condition 0.0.

## Installation

Use Python 3.10+ and a CUDA environment for model training. Model weights are downloaded separately or supplied from local storage.

```bash
cd dvoc
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hps,evaluation]'
python -m pip install --no-deps image-reward
python -m pip check
```

HPSv3++ requires its model source distribution, inference configuration, reward checkpoint, and Qwen3-VL backbone. Set their paths using the example below:

```bash
# Edit the placeholders before sourcing this file.
source configs/environment.example.sh
```

Set `model.path` in your JSON configuration to a local Z-Image-Turbo directory or a Hugging Face model ID. Reward loaders also recognize `DVOC_MODEL_ROOT/<organization>/<model>`.

## Data and configuration

`data/pickapic/train.txt` contains the 25,415-prompt training list; `data/pickapic/test.txt` contains the fixed 1,000-prompt evaluation list. The paired plan records each prompt index, initial noise seed, active states, and target reward seeds. See [data provenance and hashes](data/README.md).

```bash
dvoc check --config configs/endpoint_j2.json --world-size 8 --models
```

Configurations contain only overrides; defaults and validation live in `src/dvoc/config.py`. Relative file paths in files under `configs/` resolve against the project root. Use a fresh output directory for each training run.

| Configuration | Target anchor | Ascent steps J | Active states K |
|---|---|---:|---:|
| `endpoint_j2.json` | Sampled endpoint | 2 | 3 |
| `endpoint_j1.json` | Sampled endpoint | 1 | 3 |
| `proxy_j2.json` | Late-state clean-latent prediction | 2 | 3 |

All three configurations implement DVOC. The proxy variant transports its optimized displacement to the actual sampled endpoint and uses the same endpoint-based radius.

## Training

```bash
torchrun --standalone --nproc_per_node=8 -m dvoc train \
  --config configs/endpoint_j2.json
```

Each rank samples 3×24=72 trajectories per update and accumulates gradients one sample at a time. Eight ranks produce one global AdamW update from 576 trajectories. Defaults are 20 updates, learning rate 3e-4, weight decay 1e-4, gradient clipping at 1, and EMA decay `min((1+step)/(10+step), 0.9)`.

To change the number of ranks or sampling settings, create a matching paired plan and set `data.paired_plan` in the configuration:

```bash
dvoc plan --config configs/endpoint_j2.json --world-size 1 \
  --images-per-prompt 3 --output data/one_gpu_plan.json
```

The supplied plan targets eight ranks. A newly generated plan defines a new experiment. For a smaller smoke run, reduce `total_updates` and `data.batches_per_update` in a separate configuration before creating its plan.

## Checkpoints and continuation

```bash
torchrun --standalone --nproc_per_node=8 -m dvoc train \
  --config configs/endpoint_j2.json \
  --resume /path/to/checkpoint-10 \
  --output /path/to/new-run
```

`total_updates` is the total target count, so 20 continues checkpoint 10 for ten additional updates. Each complete checkpoint contains:

```text
checkpoint-20/
├── raw_lora/                 # Live training adapter, used for continuation
├── lora/                     # EMA adapter, used for evaluation
├── optimizer.pt
├── scaler.pt
├── ema.pt
├── trainer_state.json
├── rank-N/rng_state.pt       # Separate state for every rank
└── checkpoint-complete.json
```

Checkpoints are published after all ranks finish saving. Continuation verifies rank count, paired-plan identity, required state, and the configuration signature. To initialize a fresh run from a chosen adapter, set `model.initial_adapter`; this is separate from full continuation.

## Seven-metric evaluation

Evaluation first freezes a PNG set, then scores the same images with three independent judge groups:

| Group | Metrics | Environment |
|---|---|---|
| `regular` | HPSv3++, CLIP, AES, ImageReward | Training environment |
| `qalign` | Q-Align quality, Q-Align aesthetics | Quality-evaluation environment |
| `deqa` | DeQA | Quality-evaluation environment |

The benchmark uses 1,000 prompts, seed `20260923 + index`, HPS condition **0.3**, and the EMA adapter. Online training evaluation uses condition **0.0** by default and is logged separately.

The released Q-Align and DeQA model code uses an older Transformers stack. Create a separate environment:

```bash
python -m venv .venv-quality
.venv-quality/bin/python -m pip install -r requirements/quality-evaluation.txt
.venv-quality/bin/python -m pip install --no-deps -e .

export QALIGN_MODEL_PATH=/path/to/one-align
export DEQA_MODEL_PATH=/path/to/DeQA-Score-Mix3
```

Run the complete evaluation from the **training environment**, without wrapping this command in `torchrun`:

```bash
dvoc evaluate --config configs/endpoint_j2.json \
  --adapter /path/to/checkpoint-20/lora \
  --output /path/to/evaluation \
  --gpus 8 --qa-python "$PWD/.venv-quality/bin/python"
```

Add `--resume` to reuse verified PNGs and completed scores. The runner launches generation and each scorer in separate processes. Individual `generate`, `score`, and `summarize` commands are documented in [Evaluation](docs/evaluation.md).

Outputs include a portable image manifest, image/latent hashes, per-rank JSONL scores, `report/per_image.csv`, and `report/summary.json` with mean, standard deviation, standard error, and count for each metric. Aggregation requires complete coverage of the same images by all seven metrics.

## Weights & Biases

Only rank zero creates a W&B run. The default mode is **offline**; JSONL logs are always written.

```bash
# Online tracking, after configuring your account locally:
wandb login
WANDB_MODE=online WANDB_PROJECT=dvoc \
  torchrun --standalone --nproc_per_node=8 -m dvoc train \
  --config configs/endpoint_j2.json

# Upload a completed offline run when desired:
wandb sync /path/to/run/wandb/offline-run-*
```

The optional `tracking` section supports:

```json
{
  "tracking": {
    "backend": "wandb",
    "project": "dvoc",
    "entity": "",
    "group": "zimage-dvoc",
    "mode": "offline",
    "tags": ["endpoint", "k3", "j2"],
    "upload_checkpoints": false
  }
}
```

Set `backend` to `jsonl` to disable W&B. Training records reward, loss, gradient norm, parameter displacement, phase timing, probes, evaluation metrics, and sample images. Final evaluation creates a report run with seven-metric summaries, a per-image table, and previews. The same `tracking.group` associates related runs. Checkpoint paths and update numbers are recorded; enabling `upload_checkpoints` also creates versioned W&B model artifacts.

## Repository structure

```text
src/dvoc/
├── algorithm/               # Bounded target ascent and sparse regression
├── models/zimage.py         # Text encoding, Euler sampling, VAE decoding
├── rewards/                 # Frozen training and online-evaluation scorers
├── data/plan.py             # Paired prompts, seeds, and active-state plans
├── training/
│   ├── trainer.py           # Top-level training loop
│   ├── phases.py            # Sampling → targets → fitting → optimizer
│   ├── checkpoint.py        # Complete save and continuation
│   ├── evaluation.py        # Online EMA evaluation
│   ├── diagnostics.py       # Timing and paired reward probes
│   └── logging.py           # JSONL and W&B integration
├── evaluation/              # PNG generation, judges, report, process runner
├── config.py                # Typed defaults and validation
└── cli.py                   # Command-line interface
```

Start reading with `training/trainer.py`, `training/phases.py`, and `algorithm/`. No alternative-method trainer is imported by DVOC.

## Validation and reproducibility

```bash
python -m pip install -e '.[dev]'
ruff check src
ruff format --check src
```

Before packaging, validation covered target bounds, sparse gradients, RNG isolation, complete tiny-model training, real PEFT continuation, two-process gradient averaging, resumable evaluation, report integrity, and W&B offline logging. CPU comparisons against the reference implementation gave zero error for the tested target, composition, loss, and gradient cases. Test sources are not included in this distribution.

These checks do not replace full CUDA/NCCL validation with Z-Image and the external reward checkpoints. See [Validation](docs/validation.md) for the exact scope and [Reproducibility](docs/reproducibility.md) for evaluation and initialization details.

## License

DVOC is released under the [MIT License](LICENSE). External libraries, model weights, and datasets retain their own licenses. Existing reference comments are listed in [NOTICE.md](NOTICE.md).
