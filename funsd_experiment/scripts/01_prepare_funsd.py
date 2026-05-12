#!/usr/bin/env python
"""
Preprocess raw FUNSD dataset into unified processed JSON files.

Usage (run from repo root)::

    python funsd_experiment/scripts/01_prepare_funsd.py \\
        --raw_dir data/funsd_raw \\
        --out_dir funsd_experiment/data/processed \\
        --val_ratio 0.15 \\
        --seed 42

Input layout under ``--raw_dir`` is expected to follow the official
FUNSD distribution::

    {raw_dir}/
      dataset/
        training_data/
          annotations/   *.json
          images/        *.png
        testing_data/
          annotations/   *.json
          images/        *.png

Output (three JSON files)::

    {out_dir}/
      train.json
      val.json
      test.json
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Make the funsd_experiment package importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]  # funsd_experiment/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.funsd_reader import read_dataset  # noqa: E402

logger = logging.getLogger("funsd.prepare")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess FUNSD dataset into unified JSON format."
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="Path to the root of the extracted FUNSD dataset.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory where processed JSON files will be written.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of training data to hold out as validation. (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split. (default: 42)",
    )
    return parser.parse_args()


def save_json(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d samples → %s", len(samples), path)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    raw = args.raw_dir / "dataset"
    if not raw.is_dir():
        logger.error("Dataset directory not found: %s (expected %s)", raw, args.raw_dir / "dataset")
        sys.exit(1)

    # ── Read training data ────────────────────────────────────────
    train_ann = raw / "training_data" / "annotations"
    train_img = raw / "training_data" / "images"
    train_samples = read_dataset(train_ann, train_img)
    logger.info("Read %d training samples", len(train_samples))

    # ── Read testing data ─────────────────────────────────────────
    test_ann = raw / "testing_data" / "annotations"
    test_img = raw / "testing_data" / "images"
    test_samples = read_dataset(test_ann, test_img)
    logger.info("Read %d testing samples", len(test_samples))

    # ── Train / val split ─────────────────────────────────────────
    rng = random.Random(args.seed)
    shuffled = list(train_samples)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * args.val_ratio))
    val_samples = shuffled[:val_count]
    train_split = shuffled[val_count:]

    logger.info(
        "After split: train=%d  val=%d  test=%d",
        len(train_split),
        len(val_samples),
        len(test_samples),
    )

    # ── Summary statistics ────────────────────────────────────────
    empty_token_count = sum(
        1 for s in train_samples if not s["tokens"]
    )
    illegal_bbox_count = 0
    mismatch_count = 0
    for s in train_samples + test_samples:
        n = len(s["tokens"])
        if len(s["bboxes"]) != n or len(s["labels"]) != n:
            mismatch_count += 1
        for x1, y1, x2, y2 in s["bboxes"]:
            if x2 < x1 or y2 < y1:
                illegal_bbox_count += 1

    if empty_token_count:
        logger.warning("Found %d empty-token samples (excluded from output)", empty_token_count)
    else:
        logger.info("No empty-token samples found.")
    if illegal_bbox_count:
        logger.warning("Found %d illegal bboxes", illegal_bbox_count)
    else:
        logger.info("No illegal bboxes found.")
    if mismatch_count:
        logger.warning("Found %d samples with length mismatch (excluded from output)", mismatch_count)
    else:
        logger.info("No length-mismatch samples found.")

    # ── Save ──────────────────────────────────────────────────────
    out = args.out_dir
    save_json(train_split, out / "train.json")
    save_json(val_samples, out / "val.json")
    save_json(test_samples, out / "test.json")

    logger.info("Done. Output written to %s/", out.resolve())


if __name__ == "__main__":
    main()
