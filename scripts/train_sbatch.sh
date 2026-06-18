#!/bin/bash
#SBATCH --job-name=gr00t_n1d7_rby1
#SBATCH --partition=gigabyte_a6000
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/meat124/Isaac-GR00T/logs/%x_%j.out
#SBATCH --error=/lustre/meat124/Isaac-GR00T/logs/%x_%j.err

# ---------------------------------------------------------------
# GR00T N1.7 finetune on RB-Y1 LeRobot datasets (single A6000).
#
# Select the dataset via the DATASET env var. The job name (and thus
# the output/wandb run name) is taken from --job-name.
#
# Usage:
#   sbatch --job-name=put_carrot_on_plate \
#          --export=ALL,DATASET=put_carrot_on_plate scripts/train_sbatch.sh
#   sbatch --job-name=pick_and_hang_mug \
#          --export=ALL,DATASET=pick_and_hang_mug   scripts/train_sbatch.sh
#
#   # quick smoke test (50 steps):
#   sbatch --job-name=carrot_smoke \
#          --export=ALL,DATASET=put_carrot_on_plate scripts/train_sbatch.sh --max-steps 50
#
#   # other partition / multi-GPU:
#   sbatch -p gigabyte_pro6000 --gres=gpu:2 --export=ALL,DATASET=... scripts/train_sbatch.sh --num-gpus 2
#
# Any extra args after the script name are forwarded to launch_finetune.py.
# ---------------------------------------------------------------

set -euo pipefail

# --- configurable paths ---
GR00T_DIR=/lustre/meat124/Isaac-GR00T
DATASET="${DATASET:-put_carrot_on_plate}"
DATASET_PATH=/lustre/meat124/rby1_demo/${DATASET}
# Modality config: rby1_config.py (3 cams) by default; pass rby1_head_config.py for head-only.
MODALITY_CONFIG="${MODALITY_CONFIG:-examples/rby1/rby1_config.py}"

# Output dir is derived from the SLURM job name.
# wandb run name will also match since experiment.py uses basename(output_dir).
OUTPUT_DIR=examples/rby1/checkpoints/${SLURM_JOB_NAME}

cd "$GR00T_DIR"
mkdir -p "$GR00T_DIR/logs"

# --- model loading ---
# Load models from the HF cache (pre-downloaded on the login node) for fast,
# reliable startup. GROOT_PATCH_MISTRAL neutralizes transformers 4.57.3's
# offline-unsafe is_base_mistral() model_info call during tokenizer loading.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GROOT_HF_LOCAL_FIRST=1   # gr00t/__init__.py: resolve repo ids to local cache
export GROOT_PATCH_MISTRAL=1    # gr00t/__init__.py: make _patch_mistral_regex offline-safe
# Compute nodes have outbound internet, so log to wandb live (override with WANDB_MODE).
export WANDB_MODE="${WANDB_MODE:-online}"

echo "=============================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Job Name     : $SLURM_JOB_NAME"
echo "Dataset      : $DATASET_PATH"
echo "Modality cfg : $MODALITY_CONFIG"
echo "Output Dir   : $OUTPUT_DIR"
echo "Node         : $(hostname)"
echo "Start        : $(date)"
echo "=============================="

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: dataset path does not exist: $DATASET_PATH" >&2
    exit 1
fi

# --- train ---
# All extra arguments passed to sbatch are forwarded to launch_finetune.py.
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CONFIG" \
    --num-gpus 1 \
    --output-dir "$OUTPUT_DIR" \
    --save-total-limit 2 \
    --save-only-model \
    --save-steps 2000 \
    --max-steps 20000 \
    --use-wandb \
    --global-batch-size 32 \
    --gradient-accumulation-steps 1 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4 \
    "$@"

echo "=============================="
echo "End          : $(date)"
echo "=============================="
