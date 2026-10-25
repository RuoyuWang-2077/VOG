# External dependencies and data

This package uses Diffusers for Z-Image, PEFT for LoRA, PyTorch for training, and separate pretrained reward models. Model weights and reward-model source distributions are not bundled.

The CLIP and aesthetic adapters retain their reference links to [imscore](https://github.com/RE-N-Y/imscore/blob/main/src/imscore/preference/model.py) and [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor), respectively. These references concern the corresponding reward-model integrations, not the DVOC algorithm or training method.

ImageReward, PickScore, HPSv3++, Q-Align, DeQA, and pretrained backbones are external dependencies with their own licenses. Their weights and source distributions are not included in this project.

Prompt-list provenance is documented in [data/README.md](data/README.md). The project's MIT license does not replace licenses for external libraries, model weights, or datasets.
