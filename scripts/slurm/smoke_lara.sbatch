#!/bin/bash
# Smoke test for the lara_ws stack: a short baseline run, a short aligned run, and a
# deployment load of the aligned checkpoint.
#
#   sbatch scripts/slurm/smoke_lara.sbatch
#
# What each step is checking:
#   1. the ported AV-ALOHA harness trains at all on this clone/venv;
#   2. the alignment terms appear in the loss and the diagnostics are finite -- and the
#      step time and memory the full run should be sized from;
#   3. an aligned checkpoint still loads through the ordinary Gr00tPolicy path, which is
#      what lets the eval harness stay unmodified.
#SBATCH --job-name=smoke_lara
#SBATCH --partition=asus_pro6000,gigabyte_pro6000
#SBATCH --qos=pro6000_qos
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/meat124/lara_ws/slurm_logs/smoke_%j.out
#SBATCH --error=/lustre/meat124/lara_ws/slurm_logs/smoke_%j.err

set -uo pipefail

REPO=/lustre/meat124/lara_ws/Isaac-GR00T
VENV=/scratch2/meat124/.venvs/lara-ws
OUT=/lustre/meat124/lara_ws/runs/smoke
DATASET=/lustre/meat124/avaloha_lerobot/multitask5_3arms
LAM=/scratch2/meat124/lara_runs/lam_egodex/checkpoint-100000
VITMAE=/scratch2/meat124/.cache/huggingface/hub/models--facebook--vit-mae-large/snapshots/142cb8c25e1b1bc1769997a919aa1b5a2345a6b8

export HF_HOME=/scratch2/meat124/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PATH="/home/meat124/anaconda3/envs/RoboTwin/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/meat124/anaconda3/envs/RoboTwin/lib"

echo "[$(date)] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$REPO"
rm -rf "$OUT"

echo "=== 1/3 baseline, 20 steps (anchor recipe) ==="
"$VENV/bin/python" examples/AVAloha/train_avaloha.py \
  --dataset-path "$DATASET" --base-model-path nvidia/GR00T-N1.7-3B --output-dir "$OUT" \
  --global-batch-size 32 --max-steps 20 --learning-rate 1.4e-4 --save-steps 20 --gpus 0 \
  -- --experiment-name smoke_base --lora-llm-rank 16
rc_base=$?
echo "[smoke] baseline rc=$rc_base"

echo "=== 2/3 aligned, 30 steps (paper settings, guards off) ==="
"$VENV/bin/python" examples/AVAloha/train_avaloha.py \
  --dataset-path "$DATASET" --base-model-path nvidia/GR00T-N1.7-3B --output-dir "$OUT" \
  --global-batch-size 32 --max-steps 30 --learning-rate 1.4e-4 --save-steps 30 --gpus 0 \
  -- --experiment-name smoke_lara --lora-llm-rank 16 \
  --use-lara --lara-tokenizer-path "$LAM" --lara-image-encoder-path "$VITMAE"
rc_lara=$?
echo "[smoke] aligned rc=$rc_lara"

echo "=== 3/3 deployment load of the aligned checkpoint ==="
"$VENV/bin/python" - "$OUT/smoke_lara/checkpoint-30" <<'PY'
import sys

from gr00t.policy.gr00t_policy import Gr00tPolicy

path = sys.argv[1]
policy = Gr00tPolicy(embodiment_tag="NEW_EMBODIMENT", model_path=path, device="cuda")
head = policy.model.action_head
print("loaded  :", path)
print("use_lara:", getattr(policy.model.config, "use_lara", None))
# The tokenizer must be absent at inference: it is the whole reason the aligned
# checkpoint can be deployed by the stock harness.
print("tokenizer attached:", head.latent_motion_tokenizer is not None)
assert head.latent_motion_tokenizer is None, "tokenizer leaked into the deployed model"
print("SMOKE 3/3 OK")
PY
rc_load=$?

echo "[$(date)] smoke done: base=$rc_base lara=$rc_lara load=$rc_load"
exit $(( rc_base | rc_lara | rc_load ))
