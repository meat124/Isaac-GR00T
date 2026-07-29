#!/bin/bash
# Create the unified `gr00t` conda env holding BOTH Isaac GR00T N1.7 and the
# AV-ALOHA simulator (gym_guided_vision). The simulator only adds mujoco +
# dm_control, which coexist with GR00T's pinned dependencies, so a single env
# serves training, inference, and simulator rollout/visualization.
#
# Usage:  bash examples/AVAloha/setup_env.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-gr00t}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Resolve conda and enable `conda activate` in this non-interactive shell.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[setup] creating conda env '${ENV_NAME}' (python 3.10)"
conda create -y -n "${ENV_NAME}" python=3.10
conda activate "${ENV_NAME}"

# ffmpeg (conda-forge) ships libavcodec + dav1d (AV1 decode) + libx264 (H.264
# encode): needed by torchcodec at train time and by the AV1->H.264 converter.
echo "[setup] installing ffmpeg (conda-forge)"
conda install -y -c conda-forge "ffmpeg=6.*"

cd "${REPO_ROOT}"

echo "[setup] installing torch 2.7.1 (cu128)"
python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

echo "[setup] installing flash-attn (prebuilt wheel)"
python -m pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "[setup] installing Isaac GR00T (+ deps; tensorrt from NVIDIA index)"
python -m pip install --no-cache-dir -e . \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128

echo "[setup] installing AV-ALOHA simulator (gym_guided_vision: mujoco + dm_control)"
python -m pip install --no-cache-dir -e external_dependencies/av-aloha/gym_guided_vision

echo "[setup] done. Activate with: conda activate ${ENV_NAME}"
