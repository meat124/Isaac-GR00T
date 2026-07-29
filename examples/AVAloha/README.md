# AV-ALOHA × Isaac GR00T N1.7

This is **Stage C** of the latent-head fork: robot finetune + closed-loop eval. On this
cluster you almost never call these scripts by hand — the SLURM pipeline drives them
(`scripts/slurm/stage_c_{finetune,eval}.sbatch`); see
[`docs/pipeline_runbook.md`](../../docs/pipeline_runbook.md). The commands below are the
underlying entry points, for a standalone / debug run.

## 1. Environment

**On this cluster the live pipeline uses the uv venv `/scratch2/meat124/.venvs/groot-latent-head`**
(the repo's `.venv` symlink; hard-coded in all four sbatch scripts). Build/refresh it with
`uv sync --all-extras` from the repo root and you are done — nothing below is needed.

`setup_env.sh` builds a **separate, self-contained conda env** (`gr00t`) holding GR00T +
the simulator. It is the portable path for a machine without the uv venv; it is **not**
what the sbatch scripts run. Pick one — do not mix them.

```bash
bash examples/AVAloha/setup_env.sh           # creates+populates the `gr00t` conda env
```

Equivalently, by hand:

```bash
conda create -y -n gr00t python=3.10
conda activate gr00t
conda install -y -c conda-forge "ffmpeg=6.*"           # AV1 decode + H.264 encode + torchcodec
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install -e . --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e external_dependencies/av-aloha/gym_guided_vision
```

Either way the simulator needs the submodule: `git submodule update --init external_dependencies/av-aloha`.

---

## 2. Train & Eval

Tasks: `slot_insertion`, `insert_peg`, `sew_needle`, `tube_transfer`, `hook_package`.
The converted datasets already exist on this cluster at
**`/lustre/meat124/avaloha_lerobot/<task>_3arms`** — step (1) is only for rebuilding them.

```bash
# (1) Convert a dataset (all cameras so any selection works later). One-time; already done.
python examples/AVAloha/convert_avaloha_to_gr00t.py --task slot_insertion --num-arms 3 \
    --cameras all --output-dir /lustre/meat124/avaloha_lerobot/slot_insertion_3arms

# (2) Finetune. Cameras + hyperparameters come from avaloha_config.yaml; --base-model-path
#     is either a Stage B checkpoint (ours) or nvidia/GR00T-N1.7-3B (the B3 baseline).
#     The pipeline runs 40k steps @ global-batch 32; save_steps=10000 -> checkpoint-{10000..40000}.
#     NOTE: the trainer NESTS its output as <output-dir>/<experiment-name>/checkpoint-N.
python examples/AVAloha/train_avaloha.py --gpus 0 \
    --dataset-path /lustre/meat124/avaloha_lerobot/slot_insertion_3arms \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --output-dir /scratch2/meat124/groot_runs/stage_c \
    --global-batch-size 32 --max-steps 40000 --learning-rate 1.4e-4 --use-wandb \
    -- --experiment-name B3_slot_insertion

# (3) Deploy in the simulator + save mp4 rollouts. Rendering defaults to EGL (GPU) — the
#     osmesa (CPU software) fallback is ~10-50x slower; only use it where there is no GPU.
#     Writes results.json (success counts, per-episode) next to the mp4s; deploy.log +
#     results.json are what scripts/collect_results.sh reads.
python examples/AVAloha/deploy_avaloha.py \
    --model-path /scratch2/meat124/groot_runs/stage_c/B3_slot_insertion/checkpoint-40000 \
    --task slot_insertion --num-arms 3 --episodes 100 --max-steps 400 \
    --exec-horizon 8 --output-dir /lustre/meat124/groot_runs/stage_c/eval_B3_slot_insertion/ckpt40000
```

Eval knobs that change the number: `--success-metric` (`seated`, default — geometry-aware;
vs `contact`, looser) and `--success-hold` (default 15 steps ≈ 0.6 s, so a momentary latch
does not count as a success). Episode *i* runs seed `--seed + i`.
