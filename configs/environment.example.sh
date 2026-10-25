# Point these variables to your existing local models before training.
export DVOC_MODEL_ROOT="/path/to/models"
export REWARD_CKPT_PATH="/path/to/reward_ckpts"
export HPSV3PP_SRC="/path/to/HPSv3-PlusPlus-src"
export HPSV3PP_CHECKPOINT="/path/to/hpsv3++.pth"
export HPSV3PP_CONFIG="/path/to/HPSv3-PlusPlus-src/hpsv3/config/train_stage2.yaml"
export HPSV3PP_BACKBONE="/path/to/Qwen3-VL-8B-Instruct"
export IMAGEREWARD_ROOT="/path/to/ImageReward"
export OMP_NUM_THREADS=4
export QALIGN_MODEL_PATH="/path/to/one-align"
export DEQA_MODEL_PATH="/path/to/DeQA-Score-Mix3"
export WANDB_PROJECT="dvoc"
export WANDB_MODE="offline"
