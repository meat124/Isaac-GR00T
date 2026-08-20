# SPDX-License-Identifier: Apache-2.0
#
# Latent Active Perception — research extension on top of Isaac GR00T N1.7.
"""Build the EgoDex episode manifest (parquet) once, for fast train-time loading.

Run from the repo root::

    python gr00t_ext/scripts/egodex_build_manifest.py \
        --egodex-root /lustre/dataset/EgoDex \
        --out /lustre/meat124/runs/groot_runs/egodex_manifest.parquet
"""

from __future__ import annotations

import argparse

from gr00t_ext.data.egodex_paths import TRAIN_SPLITS, VAL_SPLIT, build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EgoDex manifest parquet")
    parser.add_argument("--egodex-root", default="/lustre/dataset/EgoDex")
    parser.add_argument("--out", default="/lustre/meat124/runs/groot_runs/egodex_manifest.parquet")
    parser.add_argument("--splits", nargs="+", default=[*TRAIN_SPLITS, VAL_SPLIT])
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    df = build_manifest(args.egodex_root, tuple(args.splits), args.out, args.num_workers)
    bad = int((df["n_frames"] < 0).sum())
    print(f"wrote {len(df)} episodes ({bad} unreadable) -> {args.out}")
    print(df.groupby("split").size().to_string())
    print("n_frames:", df["n_frames"].describe().to_string())


if __name__ == "__main__":
    main()
