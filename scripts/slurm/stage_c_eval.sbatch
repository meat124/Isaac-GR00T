#!/bin/bash
# Stage C — AV-ALOHA closed-loop evaluation of a finetuned GR00T-N1.7 checkpoint.
# Runs deploy_avaloha with EGL (GPU) rendering — the code default and ~10-50x faster than
# the docs' osmesa (CPU software rasterization). Saves one mp4 per episode (h264/yuv420p).
#
# Single checkpoint:
#   STAGE_C_MODEL=<finetune_out>/checkpoint-N  STAGE_C_TASK=slot_insertion \
#   STAGE_C_EVAL_OUT=<dir>  [STAGE_C_EPISODES=100] \
#   sbatch scripts/slurm/stage_c_eval.sbatch
# Multi checkpoint (scale-up matrix: sample-efficiency curve in one job, ~30 min/ckpt):
#   STAGE_C_MODEL_DIR=<finetune_out>/<exp>  STAGE_C_CKPTS="10000 20000 30000 40000" \
#   STAGE_C_EVAL_OUT=<dir>  sbatch scripts/slurm/stage_c_eval.sbatch
# Each checkpoint's deploy log is tee'd to <EVAL_OUT>/ckpt<N>/deploy.log for
# scripts/collect_results.sh.
# NO scheduling micromanagement (2026-07-05, user policy): no --exclude, no
# --cpus-per-task/--mem — partition JobDefaults handle placement.
#SBATCH --job-name=stageC_eval
# Eval is CPU/sim-bound (~50 min/100ep) and the bf16 3B policy at batch=1 needs ~8-10 GB,
# so RTX 3090 (24 GB) fits with headroom and its DefCpuPerGPU (16-20 vs pro6000's 6) helps
# the IK/sim bottleneck. Keeping evals here leaves the pro6000 pool free for training.
#
# RTX3090 ONLY. Success rate is measured through an EGL-rendered observation, so the GPU
# model is part of the measurement: every eval behind the 41.0% anchor landed on an
# RTX3090. big_qos is required because big_suma_rtx3090 does not accept base_qos.
#SBATCH --partition=base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/meat124/lara_ws/slurm_logs/stageCeval_%j_%x.out
#SBATCH --error=/lustre/meat124/lara_ws/slurm_logs/stageCeval_%j_%x.err

set -uo pipefail
REPO=/lustre/meat124/lara_ws/Isaac-GR00T
VENV=/scratch2/meat124/.venvs/lara-ws
TASK="${STAGE_C_TASK:-slot_insertion}"
EVAL_OUT="${STAGE_C_EVAL_OUT:?set STAGE_C_EVAL_OUT}"
EPISODES="${STAGE_C_EPISODES:-100}"
# First episode seed; episode i runs seed SEED+i. Non-zero SEED extends an existing
# eval (e.g. SEED=100 -> seeds 100..199) — use a SEPARATE EVAL_OUT: mp4 names are
# episode-index-based (deploy_avaloha.py), so same-dir reruns collide/overwrite.
SEED="${STAGE_C_SEED:-0}"
# Build a list of "MODEL_PATH|OUT_SUBDIR" jobs: STAGE_C_MODEL_DIR+STAGE_C_CKPTS loops
# the nested checkpoint-<N> dirs (out = <EVAL_OUT>/ckpt<N>); plain STAGE_C_MODEL keeps
# the single-model interface (out = EVAL_OUT). One list -> no parallel-array desync.
JOBS=()
if [ -n "${STAGE_C_MODEL_DIR:-}" ]; then
  for ck in ${STAGE_C_CKPTS:-40000}; do
    JOBS+=("$STAGE_C_MODEL_DIR/checkpoint-$ck|$EVAL_OUT/ckpt$ck")
  done
else
  JOBS+=("${STAGE_C_MODEL:?set STAGE_C_MODEL or STAGE_C_MODEL_DIR}|$EVAL_OUT")
fi

export HF_HOME=/scratch2/meat124/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="/home/meat124/anaconda3/envs/RoboTwin/bin:$PATH"          # ffmpeg (mp4)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/meat124/anaconda3/envs/RoboTwin/lib"  # torchcodec
export MUJOCO_GL=egl                                                    # GPU rendering (fast)
export PYOPENGL_PLATFORM=egl
cd "$REPO"
echo "[$(date)] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES jobs=${JOBS[*]} task=$TASK episodes=$EPISODES render=egl"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- render micro-benchmark: egl (GPU) vs osmesa (CPU), 2 cameras, 15 steps each ---
echo "=== render backend benchmark (egl vs osmesa) ==="
"$VENV/bin/python" - "$TASK" <<'PYEOF' || echo "(benchmark skipped)"
import os, sys, time
task = sys.argv[1]
def bench(backend):
    os.environ["MUJOCO_GL"] = backend; os.environ["PYOPENGL_PLATFORM"] = backend
    import importlib, gym_guided_vision, gymnasium as gym
    from examples.AVAloha.avaloha_layout import gym_env_id
    env = gym.make(gym_env_id(task, 3), disable_env_checker=True)
    env.unwrapped.cameras = env.unwrapped.cameras[:2]  # 2-camera restriction (like deploy)
    env.reset(seed=0); a = env.action_space.sample()*0
    t0 = time.time(); N = 15
    for _ in range(N): env.step(a)
    dt = time.time()-t0; env.close()
    print(f"  {backend}: {dt/N*1000:.0f} ms/step -> ~{400*dt/N:.0f}s/episode (2 cam)")
try: bench("egl")
except Exception as e: print("  egl FAILED:", repr(e)[:120])
PYEOF

rc_all=0
for job in "${JOBS[@]}"; do
  MODEL="${job%%|*}"; OUT="${job##*|}"
  if [ ! -d "$MODEL" ]; then
    echo "[eval] SKIP missing checkpoint: $MODEL"; rc_all=1; continue
  fi
  mkdir -p "$OUT"
  echo "=== eval: $MODEL -> $OUT ($EPISODES episodes, seeds $SEED..$((SEED + EPISODES - 1)), egl, 2-cam) ==="
  "$VENV/bin/python" examples/AVAloha/deploy_avaloha.py \
    --model-path "$MODEL" --task "$TASK" --num-arms 3 \
    --episodes "$EPISODES" --seed "$SEED" --output-dir "$OUT" 2>&1 | tee "$OUT/deploy.log"
  [ "${PIPESTATUS[0]}" != "0" ] && rc_all=1
done
echo "[$(date)] stage C eval done: $EVAL_OUT"
exit $rc_all
