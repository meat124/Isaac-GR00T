#!/bin/bash
# Build the lara_ws venv. Runs on dell_cpu because uv has deadlocked on the login nodes.
#
#   sbatch scripts/slurm/setup_venv.sbatch
#
# Package versions are pinned to the ones the 41.0% AV-ALOHA anchor was measured on, so
# the baseline reproduction here is comparable to it. transformers 4.57 is newer than the
# LARA release's 4.51, which is why moto_lam ships a ViTEncoder kwarg shim.
#SBATCH --job-name=lara_ws_venv
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/meat124/lara_ws/slurm_logs/venv_%j.out
#SBATCH --error=/lustre/meat124/lara_ws/slurm_logs/venv_%j.err

set -euo pipefail

REPO=/lustre/meat124/lara_ws/Isaac-GR00T
VENV=/scratch2/meat124/.venvs/lara-ws
UV=/home/meat124/.local/bin/uv

export HF_HOME=/scratch2/meat124/.cache/huggingface
export XDG_CACHE_HOME=/scratch2/meat124/.cache
export UV_CACHE_DIR=/scratch2/meat124/.cache/uv
export PIP_CACHE_DIR=/scratch2/meat124/.cache/pip

echo "[$(date)] node=$(hostname) building $VENV"
cd "$REPO"

"$UV" venv "$VENV" --python 3.10 --clear
export VIRTUAL_ENV="$VENV"

"$UV" pip install --python "$VENV/bin/python" \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# Prebuilt wheel: flash-attn compiles for ~1 h from source otherwise.
"$UV" pip install --python "$VENV/bin/python" --no-build-isolation \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# unsafe-best-match: the extra indexes carry their own (older) copies of ordinary PyPI
# packages -- requests 2.28.1 sits on the torch index and uv's default first-index-wins
# rule pins to it, which makes gr00t's own datasets/wandb requirements unsatisfiable.
"$UV" pip install --python "$VENV/bin/python" -e . \
  --index-strategy unsafe-best-match \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128

"$UV" pip install --python "$VENV/bin/python" \
  -e external_dependencies/av-aloha/gym_guided_vision

# peft: LLM LoRA (the anchor recipe). lpips/hydra-core/omegaconf: moto's tokenizer.
# transformers/torch are re-pinned last because the installs above can move them.
"$UV" pip install --python "$VENV/bin/python" \
  peft==0.17.1 transformers==4.57.3 lpips hydra-core==1.3.2 omegaconf==2.3.0

echo "=== import check ==="
"$VENV/bin/python" - <<'PY'
import gr00t, gym_guided_vision, flash_attn, peft, lpips, transformers, torch
import moto.latent_motion_tokenizer.loading as ml
from gr00t.model.gr00t_n1d7.lara_align import latent_alignment_loss, pool_dit_tokens
from gr00t.data.dataset.lara_frame_dataset import LaraFrameDataset
print("gr00t   :", gr00t.__file__)
print("moto    :", ml.__file__)
print("torch   :", torch.__version__, "cuda", torch.version.cuda)
print("tfmrs   :", transformers.__version__)
PY
echo "[$(date)] venv build done: $VENV"
