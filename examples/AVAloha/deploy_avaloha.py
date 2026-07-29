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

"""Deploy a finetuned GR00T N1.7 policy in the AV-ALOHA simulator and save mp4s.

Runs closed-loop: at each control step the simulator observation (selected
camera views + joint state + task instruction) is fed to the in-process
``Gr00tPolicy``; the policy returns an absolute joint-target action chunk
(GR00T reconstructs absolute targets from its relative-action outputs), which is
executed in the MuJoCo env. Every step is rendered and written to an mp4.

The set of cameras the policy consumes is read from the checkpoint itself (it is
baked into the processor at finetuning time), so this script needs no modality
config — just point it at the checkpoint.

Example
-------
    MUJOCO_GL=egl python examples/AVAloha/deploy_avaloha.py \
        --model-path outputs/avaloha_slot_insertion_3arms/checkpoint-10000 \
        --task slot_insertion --num-arms 3 \
        --episodes 5 --max-steps 400 \
        --output-dir outputs/avaloha_rollouts
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


# MuJoCo needs a GL backend; default to headless EGL on GPU nodes.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

# Must be set BEFORE torch/CUDA init for deterministic cuBLAS matmuls. Without
# it the flow-matching policy diverges run-to-run (GPU float nondeterminism) and
# success rates swing even with a fixed seed.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avaloha_layout import (  # noqa: E402
    EEF_POSE_GROUPS,
    default_instruction,
    eef_state_action_groups,
    gym_env_id,
    state_action_groups,
)
from avaloha_success import (
    disable_marker_collisions,  # noqa: E402
    evaluate as eval_success,  # noqa: E402
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-path", required=True, help="Path to the finetuned GR00T checkpoint directory."
    )
    p.add_argument("--task", default="slot_insertion")
    p.add_argument("--num-arms", type=int, default=3, choices=[2, 3])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=400, help="Max control steps per episode.")
    p.add_argument(
        "--exec-horizon",
        type=int,
        default=8,
        help="Action steps executed per inference (<= model action horizon).",
    )
    p.add_argument("--instruction", default=None, help="Override the task instruction.")
    p.add_argument(
        "--record-cameras",
        nargs="+",
        default=None,
        help="Cameras shown in the mp4 (default: the policy's input cameras + overhead_cam).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--success-metric",
        choices=["seated", "contact"],
        default="seated",
        help=(
            "'seated' (default): geometry-aware success -- the object must be "
            "deeply/concentrically seated (e.g. peg actually inserted, ball "
            "inside an upright tube2), not just grazing the goal marker. "
            "'contact': raw env is_success (marker touch), looser. See "
            "avaloha_success.py."
        ),
    )
    p.add_argument(
        "--success-hold",
        type=int,
        default=15,
        help=(
            "An episode counts as success only if the goal condition holds for "
            "this many CONSECUTIVE control steps (25Hz; 15 ~= 0.6s). Genuine "
            "settles last long (peg 30-70, slot ~200, tube 40-50 steps) while a "
            "ball merely poured THROUGH the tube lingers only ~3-5 steps, so 15 "
            "cleanly separates them. Set to 1 for single-frame / latched behavior."
        ),
    )
    p.add_argument(
        "--nondeterministic",
        action="store_true",
        help=(
            "Disable the deterministic-GPU settings (cuDNN/cuBLAS). By default "
            "they are on so a fixed (seed, ep) reproduces the same rollout."
        ),
    )
    p.add_argument(
        "--keep-marker-collisions",
        action="store_true",
        help=(
            "Keep the goal marker (pin) geoms colliding. By default they are made "
            "non-colliding to restore the authors' intended ghost behavior — in "
            "MuJoCo 3.10 the gap=100 markers become SOLID and block insertion "
            "(peg can't reach center; ball can't fall into the tube), diverging "
            "from the training data. See avaloha_success.disable_marker_collisions."
        ),
    )
    p.add_argument("--fps", type=int, default=25, help="Output video fps.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def add_bt_dims(x: np.ndarray) -> np.ndarray:
    """Add leading batch and time dims -> (1, 1, ...)."""
    return x[np.newaxis, np.newaxis, ...]


def compose_frame(pixels: dict, cams: list[str], caption: str):
    """Stack the requested camera views horizontally with labels + a caption bar."""
    import cv2

    panels = []
    target_h = 360
    for cam in cams:
        img = pixels[cam]
        h, w = img.shape[:2]
        scale = target_h / h
        panel = cv2.resize(img, (int(w * scale), target_h))
        panel = np.ascontiguousarray(panel)
        cv2.putText(panel, cam, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            panel, cam, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        panels.append(panel)
    row = np.concatenate(panels, axis=1)

    bar_h = 40
    bar = np.zeros((bar_h, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar, caption, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    frame = np.concatenate([bar, row], axis=0)
    # H.264 yuv420p needs even dimensions.
    H, W = frame.shape[:2]
    if H % 2 or W % 2:
        frame = frame[: H - (H % 2), : W - (W % 2)]
    return frame


def write_mp4(path: Path, frames: list, fps: int) -> None:
    """Pipe RGB frames to ffmpeg -> H.264 mp4."""
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def main() -> None:
    args = parse_args()
    instruction = args.instruction or default_instruction(args.task)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (Path(__file__).resolve().parents[2] / "outputs" / f"avaloha_rollouts_{args.task}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Policy (reads its modality config — incl. cameras — from the checkpoint) ----
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag="NEW_EMBODIMENT",
        model_path=args.model_path,
        device=args.device,
    )
    mod = policy.get_modality_config()
    policy_cams = list(mod["video"].modality_keys)
    state_groups = list(mod["state"].modality_keys)
    action_groups = list(mod["action"].modality_keys)
    lang_key = mod["language"].modality_keys[0]
    model_horizon = len(mod["action"].delta_indices)
    exec_h = min(args.exec_horizon, model_horizon)
    print(
        f"[deploy] policy cameras={policy_cams} | model horizon={model_horizon} | exec/step={exec_h}"
    )

    # The sim is always driven by JOINT targets. In eef mode the policy outputs
    # end-effector poses, which we convert to joint targets with IK.
    eef_mode = any(g in EEF_POSE_GROUPS for g in action_groups)
    joint_groups = state_action_groups(args.num_arms)
    state_slice = eef_state_action_groups(args.num_arms) if eef_mode else joint_groups
    kin = None
    if eef_mode:
        from avaloha_kinematics import AvalohaKinematics

        kin = AvalohaKinematics(num_arms=args.num_arms, task=args.task)
        print("[deploy] action space = EEF (task space): FK for state, IK to drive the sim")
    else:
        print("[deploy] action space = JOINT")

    record_cams = args.record_cameras or list(dict.fromkeys(policy_cams + ["overhead_cam"]))

    # ---- Simulator ----
    import gym_guided_vision  # noqa: F401  (registers the gym envs)
    import gymnasium as gym

    env = gym.make(gym_env_id(args.task, args.num_arms), disable_env_checker=True)
    if not args.keep_marker_collisions:
        disabled = disable_marker_collisions(env.unwrapped)
        print(f"[deploy] marker geoms set non-colliding (ghost): {disabled}")
    # Render only the cameras we actually use (policy input + video panels). The
    # env otherwise renders all 6 cameras every step, which dominates runtime
    # under CPU (osmesa) rendering used for reproducible eval.
    used_cams = list(dict.fromkeys(policy_cams + record_cams))
    env.unwrapped.cameras = [c for c in env.unwrapped.cameras if c in used_cams]
    print(f"[deploy] rendering cameras: {env.unwrapped.cameras}")
    tag = "eef" if eef_mode else "joint"

    import torch

    if not args.nondeterministic:
        # Make GPU ops deterministic so a fixed (seed, ep) reproduces exactly.
        # warn_only=True: ops lacking a deterministic kernel warn instead of
        # erroring (they stay nondeterministic, but cuBLAS/cuDNN become exact).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    n_success = 0
    episode_results: list[dict] = []
    # Machine-readable mirror of the deploy.log DONE line (collect_results.sh reads it
    # first, falling back to the log for old evals). Rewritten after every episode so a
    # time-limited job still leaves a parseable partial result.
    results_path = out_dir / "results.json"

    def write_results() -> None:
        results_path.write_text(
            json.dumps(
                {
                    "task": args.task,
                    "model_path": str(args.model_path),
                    "seed": args.seed,
                    "episodes": args.episodes,
                    "episodes_run": len(episode_results),
                    "successes": n_success,
                    "success_metric": args.success_metric,
                    "success_hold": args.success_hold,
                    "max_steps": args.max_steps,
                    "num_arms": args.num_arms,
                    "action_space": tag,
                    "per_episode": episode_results,
                },
                indent=1,
            )
        )

    for ep in range(args.episodes):
        # Full reproducibility: the env's reset() places objects with the GLOBAL
        # np.random (not self.np_random, so env.reset(seed=) alone doesn't fix
        # it), and the flow-matching policy samples its initial noise with an
        # unseeded torch.randn. Seed both per episode so a given (seed, ep) is
        # deterministic across runs.
        ep_seed = args.seed + ep
        np.random.seed(ep_seed)
        torch.manual_seed(ep_seed)
        torch.cuda.manual_seed_all(ep_seed)
        obs, info = env.reset(seed=ep_seed)
        policy.reset()
        frames = []
        ep_success = False  # goal held for >= success_hold consecutive steps
        hold = 0  # current run of consecutive goal-satisfied steps
        step = 0
        while step < args.max_steps:
            qpos = np.asarray(obs["agent_pos"], dtype=np.float32)
            state_feat = kin.fk(qpos) if eef_mode else qpos  # joints -> eef_9d if eef
            model_obs = {
                "video": {cam: add_bt_dims(obs["pixels"][cam]) for cam in policy_cams},
                "state": {
                    g: add_bt_dims(state_feat[state_slice[g][0] : state_slice[g][1]])
                    for g in state_groups
                },
                "language": {lang_key: [[instruction]]},
            }
            action_chunk, _ = policy.get_action(model_obs)

            for h in range(exec_h):
                if step >= args.max_steps:
                    break
                if eef_mode:
                    pred = np.zeros(state_feat.shape[0], dtype=np.float32)
                    for g in action_groups:
                        s, e = state_slice[g]
                        pred[s:e] = np.asarray(action_chunk[g][0][h], dtype=np.float32)
                    # IK eef pose -> joint targets, seeded at the current joints.
                    action = kin.ik(
                        pred, seed_joints=np.asarray(obs["agent_pos"], dtype=np.float32)
                    )
                else:
                    action = np.zeros(qpos.shape[0], dtype=np.float32)
                    for g in action_groups:
                        s, e = joint_groups[g]
                        action[s:e] = np.asarray(action_chunk[g][0][h], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                # Geometry-aware success: the env marker is wide, so is_success
                # trips on a shallow graze. `eval_success` adds a deep-seating
                # (and, for tube, upright) check; `contact` mode falls back to
                # the raw marker touch.
                if args.success_metric == "seated":
                    seated_now, detail = eval_success(args.task, env.unwrapped, info)
                else:
                    seated_now, detail = bool(info.get("is_success", False)), ""
                # Require the goal held for `success_hold` consecutive steps so a
                # transient graze does not count. Once reached, the episode stays
                # a success even if the object later falls out.
                hold = hold + 1 if seated_now else 0
                if hold >= args.success_hold:
                    ep_success = True
                caption = (
                    f"{args.task}[{tag}] | ep {ep} | step {step} | reward {reward:.0f} "
                    f"| {detail} hold {hold}/{args.success_hold} | success {ep_success}"
                )
                frames.append(compose_frame(obs["pixels"], record_cams, caption))
                step += 1
                if terminated or truncated:
                    break
            if terminated or truncated:
                break

        n_success += int(ep_success)
        mp4_path = out_dir / f"episode_{ep:03d}_{'success' if ep_success else 'fail'}.mp4"
        write_mp4(mp4_path, frames, args.fps)
        print(
            f"[deploy] ep {ep}: {step} steps, success={ep_success} "
            f"(held>={args.success_hold}) -> {mp4_path}"
        )
        episode_results.append({"ep": ep, "seed": ep_seed, "success": ep_success, "steps": step})
        write_results()

    env.close()
    print(f"[deploy] DONE [{tag}]: {n_success}/{args.episodes} successful. Videos in {out_dir}")


if __name__ == "__main__":
    main()
