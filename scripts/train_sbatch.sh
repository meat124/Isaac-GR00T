#!/bin/bash
#SBATCH --job-name=PuttingCupintotheDish_demo100
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/meat124/Isaac-GR00T/logs/%x_%j.out
#SBATCH --error=/lustre/meat124/Isaac-GR00T/logs/%x_%j.err

# ---------------------------------------------------------------
# Usage:
#   sbatch scripts/train_sbatch.sh                              # default args
#   sbatch scripts/train_sbatch.sh --num-episodes 50            # limit episodes
#   sbatch scripts/train_sbatch.sh --max-steps 5000             # override steps
#   sbatch --gres=gpu:2 scripts/train_sbatch.sh --num-gpus 2   # multi-GPU
# ---------------------------------------------------------------

set -euo pipefail

# --- configurable paths ---
GR00T_DIR=/lustre/meat124/Isaac-GR00T
DATASET_PATH=/lustre/meat124/rby1_demo/LeRobotDataset_v2/PuttingCupintotheDishV2

# Output dir is automatically derived from the SLURM job name.
# wandb run name will also match since experiment.py uses basename(output_dir).
OUTPUT_DIR=examples/rby1/checkpoints/${SLURM_JOB_NAME}

cd "$GR00T_DIR"

# create log dir if needed
mkdir -p "$GR00T_DIR/logs"

echo "=============================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Job Name     : $SLURM_JOB_NAME"
echo "Output Dir   : $OUTPUT_DIR"
echo "Node         : $(hostname)"
echo "Start        : $(date)"
echo "=============================="

# --- train ---
# All extra arguments passed to sbatch are forwarded to launch_finetune.py.
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/rby1/rby1_config.py \
    --num-gpus 1 \
    --output-dir "$OUTPUT_DIR" \
    --save-total-limit 5 \
    --save-steps 2000 \
    --max-steps 10000 \
    --use-wandb \
    --global-batch-size 8 \
    --gradient-accumulation-steps 32 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4 \
    --num-episodes 100 \
    "$@"

echo "=============================="
echo "End          : $(date)"
echo "=============================="
