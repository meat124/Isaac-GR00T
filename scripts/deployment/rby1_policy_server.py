from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil

import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
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

    # RTC (Real-Time Chunking) options — drives the native inpainting in the GR00T action head.
    use_rtc: bool = False
    """Enable native Real-Time Chunking (action-head inpainting) for continuous chunking."""
    rtc_execute_chunk_size: int = 8
    """Actions executed per chunk before re-querying. Overlap = action_horizon - this value.
    Must match EXECUTE_CHUNK_SIZE in the notebook."""
    rtc_frozen_steps: int = 1
    """Leading steps held fixed (velocity frozen) to absorb inference latency. Must be <= overlap."""
    rtc_ramp_rate: float | None = None
    """Exponential ramp rate over the unfrozen overlap. None -> use the model config's rtc_ramp_rate."""


def _maybe_create_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    # Some checkpoints keep processor artifacts under model_path/processor/.
    # Gr00tPolicy expects them at model_path root, so create lightweight links.
    try:
        dst.symlink_to(src)
        logging.info("Created symlink: %s -> %s", dst, src)
    except OSError:
        # Filesystems such as exFAT have no symlinks.
        shutil.copyfile(src, dst)
        logging.info("Copied %s to %s", src, dst)


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
        rtc_enabled=args.use_rtc,
        execute_chunk_size=args.rtc_execute_chunk_size if args.use_rtc else None,
        rtc_frozen_steps=args.rtc_frozen_steps,
        rtc_ramp_rate=args.rtc_ramp_rate,
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
        "rtc_execute_chunk_size": args.rtc_execute_chunk_size if args.use_rtc else None,
        "rtc_frozen_steps": args.rtc_frozen_steps if args.use_rtc else None,
        "rtc_ramp_rate": policy._rtc_ramp_rate if args.use_rtc else None,
    })

    logging.info(
        "Starting RBY1 GR00T policy server | host=%s port=%d model=%s horizon=%d denoising_steps=%d rtc=%s",
        args.host,
        args.port,
        model_path,
        action_horizon,
        int(policy.model.action_head.num_inference_timesteps),
        f"on(execute_chunk_size={args.rtc_execute_chunk_size}, frozen_steps={args.rtc_frozen_steps}, ramp_rate={policy._rtc_ramp_rate})" if args.use_rtc else "off",
    )

    server = PolicyServer(policy=policy, host=args.host, port=args.port)

    def _set_rtc(
        enabled: bool, execute_chunk_size: int | None = None, frozen_steps: int | None = None
    ) -> dict:
        """Switch Real-Time Chunking on or off while serving, without reloading the model."""
        if enabled and not execute_chunk_size:
            raise ValueError("execute_chunk_size is required to enable RTC")
        policy._rtc_enabled = bool(enabled)
        if execute_chunk_size:
            policy._execute_chunk_size = int(execute_chunk_size)
        if frozen_steps is not None:
            policy._rtc_frozen_steps = int(frozen_steps)
        policy._prev_raw_action_pred = None  # the next chunk starts afresh
        policy.metadata.update(
            rtc_enabled=policy._rtc_enabled,
            rtc_execute_chunk_size=policy._execute_chunk_size if policy._rtc_enabled else None,
            rtc_frozen_steps=policy._rtc_frozen_steps if policy._rtc_enabled else None,
        )
        logging.info(
            "RTC %s (execute_chunk_size=%s, frozen_steps=%s)",
            "on" if policy._rtc_enabled else "off",
            policy._execute_chunk_size,
            policy._rtc_frozen_steps,
        )
        return {k: policy.metadata[k] for k in ("rtc_enabled", "rtc_execute_chunk_size", "rtc_frozen_steps")}

    server.register_endpoint("set_rtc", _set_rtc)
    server.run()


if __name__ == "__main__":
    main(tyro.cli(Args))
