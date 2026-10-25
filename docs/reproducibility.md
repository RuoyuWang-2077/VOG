# Reproducibility

## Training invariants

DVOC uses cached, detached Euler states; the stored scheduler timestep for velocity recomputation; a negative raw transformer output for the Euler velocity; a target radius measured from the actual sampled endpoint; per-sample active states; and a detached mean-absolute-displacement loss normalizer. The tested training arithmetic is documented in `algorithm.md`.

The paired plan fixes prompts, initial noise, active states, and reward seeds. Endpoint J1 and J2 share their first target seed. The supplied plan requires eight ranks, 3 samples per rollout batch, 24 batches per update, and K=3. Changing these settings requires a new plan.

## Initialization and continuation

The source distribution contains no pretrained model weights, initial adapters, or trained checkpoints. A fresh run initializes an adapter from the configured seed. Set `model.initial_adapter` to reuse a particular initialization. A recorded hash alone cannot reconstruct its weights.

Full continuation restores the live adapter, optimizer, scaler, EMA, and each rank's RNG. The data-plan hash and configuration signature must match. Checkpoints with the earlier complete-directory layout can also be read, provided their raw adapter, optimizer, EMA, scaler, trainer state, and per-rank RNG files are present. Such checkpoints may lack a configuration signature; verify the matching model and method settings when using them.

Checkpoint saving and evaluation temporarily swap EMA weights with exception-safe restoration. Evaluation/probe random streams are isolated from training. LoRA gradients are averaged once after accumulation.

## Evaluation behavior

Formal evaluation fixes each sample's noise and reward randomness by index. Images are saved once as PNG and reused by all scorers. Online evaluation shards unique indices without padding duplicate prompts. Its aggregation and explicit per-index seed scheme need not reproduce an older evaluation run that used rank-local random streams or padded samplers.

Model and reward checkpoint identity, software versions, and GPU kernels also affect results. CPU arithmetic parity and state-restoration tests do not establish complete cross-device or pretrained-model reproducibility. Full model training and the seven real reward models still require validation in the target CUDA environment.
