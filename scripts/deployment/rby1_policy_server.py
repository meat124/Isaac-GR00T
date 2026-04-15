from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d6.rtc_groot import GR00TRTCConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer

# Registers RBY1 modality config as a side effect.
from examples.rby1 import rby1_config as _rby1_config  # noqa: F401

REQUIRED_PROCESSOR_FILES = (
    "processor_config.json",
    "statistics.json",
    "embodiment_id.json",
)


@dataclass
class Args:
    """Run GR00T policy server tuned for RBY1 real-robot orchestration."""

    model_path: str
    host: str = "0.0.0.0"
    port: int = 5555
    device: str = "cuda"
    strict: bool = False
    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    ensure_processor_symlinks: bool = True
    denoising_steps: int | None = None

    # RTC (Real-Time Chunking) options
    use_rtc: bool = False
    """Enable Real-Time Chunking guidance during denoising."""
    rtc_max_guidance_weight: float = 1.0
    """Maximum guidance weight for RTC blending (clamped to 1.0 for direct blending)."""
    rtc_execution_horizon: int = 8
    """Number of actions executed per chunk; determines the left-over tail size. Must match EXECUTE_CHUNK_SIZE in the notebook."""
    rtc_schedule: str = "linear"
    """Weight schedule along the action prefix: 'linear', 'exp', 'ones', or 'zeros'."""


def _maybe_create_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    # Some checkpoints keep processor artifacts under model_path/processor/.
    # Gr00tPolicy expects them at model_path root, so create lightweight links.
    dst.symlink_to(src)
    logging.info("Created symlink: %s -> %s", dst, src)


def _ensure_processor_files(model_path: Path) -> None:
    processor_dir = model_path / "processor"
    if not processor_dir.exists():
        return

    for name in REQUIRED_PROCESSOR_FILES:
        src = processor_dir / name
        dst = model_path / name
        if src.exists():
            _maybe_create_symlink(src, dst)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if args.ensure_processor_symlinks:
        _ensure_processor_files(model_path)

    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=str(model_path),
        device=args.device,
        strict=args.strict,
        rtc_config=GR00TRTCConfig(
            enabled=args.use_rtc,
            max_guidance_weight=args.rtc_max_guidance_weight,
            execution_horizon=args.rtc_execution_horizon,
            schedule=args.rtc_schedule,
        ) if args.use_rtc else None,
        execute_chunk_size=args.rtc_execution_horizon if args.use_rtc else None,
    )

    if args.denoising_steps is not None:
        if args.denoising_steps <= 0:
            raise ValueError(
                f"denoising_steps must be > 0, got {args.denoising_steps}"
            )
        policy.model.action_head.num_inference_timesteps = int(args.denoising_steps)
        logging.info(
            "Overriding denoising steps: %d",
            policy.model.action_head.num_inference_timesteps,
        )

    modality_cfg = policy.get_modality_config()
    action_cfg = modality_cfg["action"]
    action_keys = list(action_cfg.modality_keys)
    action_horizon = len(action_cfg.delta_indices)

    # PolicyServer metadata endpoint reads this field when present.
    setattr(policy, "metadata", {
        "served_env": "rby1",
        "policy_loader": "gr00t_checkpoint",
        "checkpoint_dir": str(model_path),
        "embodiment_tag": args.embodiment_tag.value,
        "action_keys": action_keys,
        "action_horizon": action_horizon,
        "num_inference_timesteps": int(policy.model.action_head.num_inference_timesteps),
        "rtc_enabled": args.use_rtc,
        "rtc_execution_horizon": args.rtc_execution_horizon if args.use_rtc else None,
        "rtc_schedule": args.rtc_schedule if args.use_rtc else None,
        "rtc_max_guidance_weight": args.rtc_max_guidance_weight if args.use_rtc else None,
    })

    logging.info(
        "Starting RBY1 GR00T policy server | host=%s port=%d model=%s horizon=%d denoising_steps=%d rtc=%s",
        args.host,
        args.port,
        model_path,
        action_horizon,
        int(policy.model.action_head.num_inference_timesteps),
        f"on(exec_horizon={args.rtc_execution_horizon}, weight={args.rtc_max_guidance_weight}, sched={args.rtc_schedule})" if args.use_rtc else "off",
    )

    server = PolicyServer(policy=policy, host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main(tyro.cli(Args))
