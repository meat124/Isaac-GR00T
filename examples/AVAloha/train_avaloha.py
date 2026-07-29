# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finetune GR00T N1.7 on a converted AV-ALOHA dataset, driven by avaloha_config.yaml.

Reads the camera selection + hyperparameters from the YAML, points the finetuner
at the matching modality config, and launches ``gr00t/experiment/launch_finetune.py``
(single-GPU via ``python``; multi-GPU via ``torchrun``).

Example
-------
    python examples/AVAloha/train_avaloha.py --gpus 0
    python examples/AVAloha/train_avaloha.py --gpus 0,1,2,3 --num-gpus 4
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml


THIS = Path(__file__).resolve()
HERE = THIS.parent
REPO = THIS.parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=str(HERE / "avaloha_config.yaml"))
    p.add_argument("--dataset-path", default=None, help="Default: datasets/avaloha_<task>_<N>arms")
    p.add_argument("--base-model-path", default="nvidia/GR00T-N1.7-3B")
    p.add_argument("--output-dir", default=None, help="Default: outputs/avaloha_<task>_<N>arms")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--gpus", default="0", help="Value for CUDA_VISIBLE_DEVICES.")
    p.add_argument("--master-port", default="29500")
    # Optional overrides (otherwise taken from YAML).
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--global-batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", default=None, help="Override the YAML wandb_project.")
    p.add_argument("extra", nargs="*", help="Extra args passed through to launch_finetune.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    task = cfg.get("task", "slot_insertion")
    num_arms = int(cfg.get("num_arms", 3))

    dataset = args.dataset_path or str(REPO / "datasets" / f"avaloha_{task}_{num_arms}arms")
    output_dir = args.output_dir or str(REPO / "outputs" / f"avaloha_{task}_{num_arms}arms")
    max_steps = args.max_steps if args.max_steps is not None else int(cfg.get("max_steps", 10000))
    global_batch = args.global_batch_size or int(cfg.get("global_batch_size", 16))
    save_steps = (
        args.save_steps if args.save_steps is not None else int(cfg.get("save_steps", 1000))
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(cfg.get("learning_rate", 1e-4))
    )

    stats = Path(dataset) / "meta" / "stats.json"
    if not stats.exists():
        sys.exit(
            f"[train] stats.json not found at {stats}.\n"
            f"        Convert the dataset first:\n"
            f"        python examples/AVAloha/convert_avaloha_to_gr00t.py "
            f"--task {task} --num-arms {num_arms} --cameras all"
        )

    env = {
        **os.environ,
        # Make the modality config + finetuner read the SAME camera selection.
        "AVALOHA_CONFIG": str(Path(args.config).resolve()),
        "CUDA_VISIBLE_DEVICES": args.gpus,
    }

    finetune = str(REPO / "gr00t" / "experiment" / "launch_finetune.py")
    if args.num_gpus > 1:
        launcher = [
            "torchrun",
            f"--nproc_per_node={args.num_gpus}",
            f"--master_port={args.master_port}",
        ]
    else:
        launcher = [sys.executable]

    cmd = launcher + [
        finetune,
        "--base-model-path",
        args.base_model_path,
        "--dataset-path",
        dataset,
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(HERE / "avaloha_config.py"),
        "--num-gpus",
        str(args.num_gpus),
        "--output-dir",
        output_dir,
        "--max-steps",
        str(max_steps),
        "--global-batch-size",
        str(global_batch),
        "--learning-rate",
        str(learning_rate),
        "--state-dropout-prob",
        str(float(cfg.get("state_dropout_prob", 0.0))),
        "--dataloader-num-workers",
        str(int(cfg.get("dataloader_num_workers", 4))),
        "--save-steps",
        str(save_steps),
    ]
    if args.use_wandb or bool(cfg.get("use_wandb", False)):
        cmd += [
            "--use-wandb",
            "--wandb-project",
            args.wandb_project or str(cfg.get("wandb_project", "avaloha-gr00t-n1d7")),
        ]
    cmd += list(args.extra)

    print(f"[train] task={task} num_arms={num_arms} cameras={cfg.get('cameras')}")
    print(f"[train] dataset={dataset}")
    print(f"[train] output={output_dir}  max_steps={max_steps}  global_batch={global_batch}")
    print("[train] " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO))


if __name__ == "__main__":
    main()
