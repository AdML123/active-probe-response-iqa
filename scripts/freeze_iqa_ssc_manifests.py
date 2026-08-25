"""Freeze the IQA-SSC calibration, evaluation, and disjoint pilot manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from iqa_ssc.manifests import freeze_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=124)
    parser.add_argument("--pilot-count", type=int, default=20)
    args = parser.parse_args()
    result = freeze_manifests(
        asset_root=args.asset_root,
        test_manifest=args.test_manifest,
        historical_manifest=args.historical_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        pilot_count=args.pilot_count,
    )
    print(f"calibration={result.calibration}")
    print(f"evaluation={result.evaluation}")
    print(f"pilot={result.pilot}")


if __name__ == "__main__":
    main()

