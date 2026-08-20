#!/bin/bash
# Stage C — AV-ALOHA finetune of GR00T-N1.7 (29-D [manip;head] eef). Parameterized for the
# ablation matrix via env vars. Logs to the latent-head wandb team.
#
#   STAGE_C_BASE=<stageB ckpt | nvidia/GR00T-N1.7-3B>  STAGE_C_EXP=<run name> \
#   STAGE_C_OUT=<output dir>  STAGE_C_DATASET=<lerobot dir>  [STAGE_C_GBS=16] \
#   sbatch scripts/slurm/stage_c_finetune.sbatch
#
# NO WATCHDOG (2026-07-05, user policy): kill -9 on wedged GPU procs leaves D-state
# zombies that DRAIN the node ("Kill task failed" — lost node57/58; the group-kill
# variant here also orphaned launch_finetune.py and killed B3 at 24k). Plain foreground
# run; resume flag auto-added when checkpoints exist. Never kill/monitor from the job.
#
# NO scheduling micromanagement (2026-07-05, user policy): no --exclude, no
# --cpus-per-task/--mem/--gres-flags — partition JobDefaults handle placement.
#
# 3B model: pro6000 (96 GB) fits batch 16; a6000 (48 GB) needs STAGE_C_GBS=8.
#SBATCH --job-name=stageC_ft
#SBATCH --partition=asus_pro6000,gigabyte_pro6000
#SBATCH --qos=pro6000_qos
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=2-00:00:00
#SBATCH --output=/lustre/meat124/lara_ws/slurm_logs/stageC_%j_%x.out
#SBATCH --error=/lustre/meat124/lara_ws/slurm_logs/stageC_%j_%x.err

set -uo pipefail

REPO=/lustre/meat124/lara_ws/Isaac-GR00T
VENV=/scratch2/meat124/.venvs/lara-ws
BASE="${STAGE_C_BASE:?set STAGE_C_BASE}"
EXP="${STAGE_C_EXP:?set STAGE_C_EXP}"
OUT="${STAGE_C_OUT:?set STAGE_C_OUT}"
DATASET="${STAGE_C_DATASET:?set STAGE_C_DATASET}"
GBS="${STAGE_C_GBS:-16}"
STEPS="${STAGE_C_STEPS:-100000}"
# Optional LR override (scale-up: sqrt-scale with GBS, e.g. 1.4e-4 @ GBS=32).
LR_ARGS=()
[ -n "${STAGE_C_LR:-}" ] && LR_ARGS+=(--learning-rate "$STAGE_C_LR")

export HF_HOME=/scratch2/meat124/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;1"
# ffmpeg BINARY (converter/video) + ffmpeg7 LIBS (torchcodec decoder) from the RoboTwin env.
export PATH="/home/meat124/anaconda3/envs/RoboTwin/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/meat124/anaconda3/envs/RoboTwin/lib"
export WANDB_DIR=/lustre/meat124/lara_ws/runs/wandb
mkdir -p "$WANDB_DIR"
export WANDB_MODE=online
# Pin the team explicitly. The account's default entity is mhbench_baselines -- an
# unrelated project -- so leaving this unset files these runs under it.
export WANDB_ENTITY=latent-head
# Stage in the wandb display name + a single stage tag. EXP itself must stay bare —
# it doubles as the output subdir ($OUT/$EXP); experiment.py reads these env fallbacks.
export WANDB_NAME="c-${EXP}"
export WANDB_TAGS="stage-c"
# Route HF-trainer metrics to the stage_c_train/ wandb section at the real
# global step (experiment.py SectionedWandbCallback).
export WANDB_SECTION=stage_c

echo "[$(date)] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES exp=$EXP base=$BASE gbs=$GBS steps=$STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$REPO"

# Auto-resume: Stage C nests its output as <OUT>/<EXP>/checkpoint-N (gotcha §7). If
# checkpoints already exist (SLURM --requeue after a node failure, or a manual
# resubmit), pass --resume-from-checkpoint through to launch_finetune.
resume=""
if compgen -G "$OUT/$EXP/checkpoint-*" > /dev/null; then
  resume="--resume-from-checkpoint"
  echo "[$(date)] existing checkpoints in $OUT/$EXP -> resume"
fi

# Plain foreground run (no watchdog — see header). train_avaloha.py is a wrapper that
# runs launch_finetune.py via subprocess; run in-cgroup so SLURM's own cleanup at
# scancel/timelimit reaps the whole tree.
# STAGE_C_EXTRA passes launch_finetune flags straight through (--lora-llm-rank 16 for the
# anchor recipe, --use-lara ... for the aligned arm). Word-split on purpose.
"$VENV/bin/python" examples/AVAloha/train_avaloha.py \
  --dataset-path "$DATASET" --base-model-path "$BASE" --output-dir "$OUT" \
  --global-batch-size "$GBS" --max-steps "$STEPS" "${LR_ARGS[@]}" --gpus 0 --use-wandb \
  -- --experiment-name "$EXP" $resume ${STAGE_C_EXTRA:-}
rc=$?
echo "[$(date)] stage C finetune exit rc=$rc: $EXP -> $OUT"
exit $rc
