#!/usr/bin/env python
"""Battery Degradation Prediction - CLI Entrypoint.

Usage:
    python main.py --data-path path/to/data.csv --target capacity

This script is a thin wrapper around ``battery_degradation.pipeline.run_baseline``
so that users can quickly execute the baseline experiment from the project root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from battery_degradation.pipeline import run_baseline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run battery degradation baseline pipeline.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/battery_degradation_data.csv"),
        help="Path to the CSV dataset file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="capacity",
        help="Target column to predict (e.g. 'capacity' or 'internal_resistance').",
    )
    return parser.parse_args()


def main() -> None:  # noqa: D401
    """Parse CLI arguments and run the baseline pipeline."""
    args = _parse_args()
    run_baseline(data_path=args.data_path, target_column=args.target)


if __name__ == "__main__":
    main()
