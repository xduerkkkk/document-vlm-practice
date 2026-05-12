#!/usr/bin/env python
"""
Preprocess processed FUNSD JSON files and generate dataset statistics.

Usage (run from repo root)::

    python funsd_experiment/scripts/02_preprocess_and_stats.py \\
        --in_dir funsd_experiment/data/processed \\
        --out_dir funsd_experiment/data/prepared \\
        --metrics_out funsd_experiment/outputs/metrics/data_stats.json \\
        --table_out funsd_experiment/report/tables/data_stats.md \\
        --sort_strategy yx
"""

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]  # funsd_experiment/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.preprocess import (
    load_processed_json,
    preprocess_samples,
    save_prepared_json,
)
from src.data.sort_boxes import SortStrategy
from src.data.statistics import (
    compute_split_stats,
    compute_statistics,
    print_stats_summary,
)

logger = logging.getLogger("funsd.preprocess")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, normalise, sort FUNSD data and compute statistics."
    )
    parser.add_argument(
        "--in_dir",
        type=Path,
        default=Path("funsd_experiment/data/processed"),
        help="Directory containing train.json / val.json / test.json "
             "(default: funsd_experiment/data/processed)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("funsd_experiment/data/prepared"),
        help="Directory for prepared JSON output. (default: funsd_experiment/data/prepared)",
    )
    parser.add_argument(
        "--metrics_out",
        type=Path,
        default=Path("funsd_experiment/outputs/metrics/data_stats.json"),
        help="Path for JSON statistics output.",
    )
    parser.add_argument(
        "--table_out",
        type=Path,
        default=Path("funsd_experiment/report/tables/data_stats.md"),
        help="Path for Markdown statistics report.",
    )
    parser.add_argument(
        "--sort_strategy",
        type=str,
        choices=["original", "yx"],
        default="yx",
        help="Text-box sort strategy. (default: yx)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    sort_strategy: SortStrategy = args.sort_strategy  # type: ignore[assignment]
    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    metrics_out: Path = args.metrics_out
    table_out: Path = args.table_out

    # ── Load ──────────────────────────────────────────────────────
    splits_in: dict[str, list[dict]] = {}
    for split_name in ("train", "val", "test"):
        path = in_dir / f"{split_name}.json"
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue
        splits_in[split_name] = load_processed_json(path)
        logger.info("Loaded %s: %d samples", split_name, len(splits_in[split_name]))

    # ── Preprocess ────────────────────────────────────────────────
    splits_out: dict[str, list[dict]] = {}
    for split_name, samples in splits_in.items():
        logger.info("Preprocessing %s (%d samples) …", split_name, len(samples))
        prepared = preprocess_samples(samples, sort_strategy=sort_strategy)
        splits_out[split_name] = prepared

        out_path = out_dir / f"{split_name}.json"
        save_prepared_json(prepared, out_path)

    # ── Statistics ────────────────────────────────────────────────
    compute_statistics(splits_out, metrics_out, table_out)

    stats_dicts = {
        name: compute_split_stats(samples, name)
        for name, samples in splits_out.items()
    }
    print_stats_summary(stats_dicts)

    logger.info("Done.")


if __name__ == "__main__":
    main()
