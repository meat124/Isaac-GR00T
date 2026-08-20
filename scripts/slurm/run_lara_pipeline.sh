#!/bin/bash
# Submit one arm's finetune and its evaluation, chained with afterok.
#
#   scripts/slurm/run_lara_pipeline.sh base            # anchor recipe, no alignment
#   scripts/slurm/run_lara_pipeline.sh lara            # + LARA, paper settings
#   scripts/slurm/run_lara_pipeline.sh guard           # + the anti-collapse guards
#
# The arms differ ONLY in the alignment terms; everything else is the recipe behind the
# 41.0% (n=200) AV-ALOHA anchor, so a difference in success rate is attributable.
#
# Env overrides: STEPS, GBS, LR, EVAL_TASK, EVAL_EPISODES, EVAL_CKPTS, TIME_TRAIN.
#
# Plain sbatch + afterok only: no watchdogs, no monitors, no requeue loops (user policy,
# 2026-07-05 -- kill -9 on a wedged GPU process drains the node).
set -euo pipefail

ARM="${1:?usage: run_lara_pipeline.sh <base|lara|guard>}"
REPO=/lustre/meat124/lara_ws/Isaac-GR00T
RUNS=/lustre/meat124/lara_ws/runs
DATASET="${DATASET:-/lustre/meat124/avaloha_lerobot/multitask5_3arms}"
LAM="${LAM:-/scratch2/meat124/lara_runs/lam_egodex/checkpoint-100000}"
VITMAE="${VITMAE:-/scratch2/meat124/.cache/huggingface/hub/models--facebook--vit-mae-large/snapshots/142cb8c25e1b1bc1769997a919aa1b5a2345a6b8}"

STEPS="${STEPS:-40000}"
GBS="${GBS:-32}"
LR="${LR:-1.4e-4}"
EVAL_TASK="${EVAL_TASK:-slot_insertion}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"

# The anchor recipe. LoRA on the LLM with the backbone frozen is what the 41.0% was
# measured with; dropping it is a different experiment, not a simplification.
EXTRA="--lora-llm-rank 16"
LARA_FLAGS="--use-lara --lara-tokenizer-path $LAM --lara-image-encoder-path $VITMAE"

case "$ARM" in
  base)
    EXP="ws_base_vlmlora_mt5"
    # Measured 5 h 10 m for 40k steps on a pro6000.
    TIME_TRAIN="${TIME_TRAIN:-12:00:00}"
    EVAL_CKPTS="${EVAL_CKPTS:-$STEPS}"
    ;;
  lara)
    EXP="ws_lara_paper_mt5"
    EXTRA="$EXTRA $LARA_FLAGS"
    # Measured 0.6 s/step against the baseline's 0.4 (the tokenizer forward and its
    # reconstruction loss ride on every step), so 40k steps is ~6.7 h. Asking for 48 h
    # would only push the job behind backfill.
    TIME_TRAIN="${TIME_TRAIN:-18:00:00}"
    EVAL_CKPTS="${EVAL_CKPTS:-10000 20000 30000 40000}"
    ;;
  guard)
    EXP="ws_lara_guard_mt5"
    # The bare cosine has a degenerate optimum: predict the target's batch mean. These
    # block the three escapes (DC shortcut, amplitude shrink, rank-1 concentration).
    EXTRA="$EXTRA $LARA_FLAGS --lara-align-center --lara-w-var 4.0 --lara-var-floor 0.4"
    EXTRA="$EXTRA --lara-w-cov 1.0 --lara-proj-layers 3 --lara-pool action"
    TIME_TRAIN="${TIME_TRAIN:-18:00:00}"
    EVAL_CKPTS="${EVAL_CKPTS:-10000 20000 30000 40000}"
    ;;
  *)
    echo "unknown arm: $ARM (expected base, lara or guard)" >&2
    exit 2
    ;;
esac

cd "$REPO"
mkdir -p "$RUNS/stage_c" /lustre/meat124/lara_ws/slurm_logs

train_id=$(STAGE_C_BASE=nvidia/GR00T-N1.7-3B \
  STAGE_C_EXP="$EXP" \
  STAGE_C_OUT="$RUNS/stage_c" \
  STAGE_C_DATASET="$DATASET" \
  STAGE_C_GBS="$GBS" \
  STAGE_C_STEPS="$STEPS" \
  STAGE_C_LR="$LR" \
  STAGE_C_EXTRA="$EXTRA" \
  sbatch --parsable --job-name="ft_$EXP" --time="$TIME_TRAIN" \
    scripts/slurm/stage_c_finetune.sbatch)
echo "[submit] finetune $EXP -> job $train_id (time=$TIME_TRAIN)"

# kill-on-invalid-dep so a failed finetune does not leave an eval pending forever.
eval_id=$(STAGE_C_MODEL_DIR="$RUNS/stage_c/$EXP" \
  STAGE_C_CKPTS="$EVAL_CKPTS" \
  STAGE_C_TASK="$EVAL_TASK" \
  STAGE_C_EPISODES="$EVAL_EPISODES" \
  STAGE_C_EVAL_OUT="$RUNS/eval/${EXP}_${EVAL_TASK}" \
  sbatch --parsable --job-name="eval_$EXP" \
    --dependency=afterok:"$train_id" --kill-on-invalid-dep=yes \
    scripts/slurm/stage_c_eval.sbatch)
echo "[submit] eval $EXP ($EVAL_TASK x$EVAL_EPISODES, ckpts: $EVAL_CKPTS) -> job $eval_id"
echo "[submit] arm=$ARM extra: $EXTRA"
