"""
Preprocessing pipeline: clean, clip, normalise and sort processed samples
into the final *prepared* format.
"""

import json
import logging
from pathlib import Path

from .normalize_bbox import normalize_bbox
from .sort_boxes import SortStrategy, sort_sample

logger = logging.getLogger(__name__)


def _clean_sample(sample: dict) -> dict:
    """Remove empty / illegal entries from a single sample.

    Returns a dict with extra keys ``_empty_tokens``, ``_illegal_bboxes``,
    ``_bboxes_clipped`` for statistics tracking.
    """
    tokens = sample.get("tokens", [])
    bboxes = sample.get("bboxes", [])
    labels = sample.get("labels", [])
    width = sample.get("width", 0)
    height = sample.get("height", 0)
    sid = sample.get("id", "?")

    empty_removed = 0
    illegal_removed = 0
    clipped = 0

    clean_tokens: list[str] = []
    clean_bboxes: list[list[int]] = []
    clean_labels: list[str] = []

    for i, (tok, bb, lb) in enumerate(zip(tokens, bboxes, labels)):
        # --- empty / whitespace-only token ---
        if not tok or not tok.strip():
            empty_removed += 1
            continue

        # --- illegal bbox ---
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            illegal_removed += 1
            continue

        x0, y0, x1, y1 = bb
        if x1 < x0 or y1 < y0:
            illegal_removed += 1
            continue

        # --- clip to image bounds ---
        need_clip = False
        if x0 < 0:
            x0 = 0
            need_clip = True
        if y0 < 0:
            y0 = 0
            need_clip = True
        if x1 > width and width > 0:
            x1 = width
            need_clip = True
        if y1 > height and height > 0:
            y1 = height
            need_clip = True
        if need_clip:
            clipped += 1

        clean_tokens.append(tok.strip())
        clean_bboxes.append([x0, y0, x1, y1])
        clean_labels.append(lb)

    n = len(clean_tokens)
    if len(clean_bboxes) != n or len(clean_labels) != n:
        logger.error("Internal inconsistency in sample %s after cleaning — skipping", sid)
        return {"_skip": True}

    sample["tokens"] = clean_tokens
    sample["bboxes"] = clean_bboxes
    sample["labels"] = clean_labels
    sample["_empty_tokens"] = empty_removed
    sample["_illegal_bboxes"] = illegal_removed
    sample["_bboxes_clipped"] = clipped

    return sample


def _add_normalised_bboxes(sample: dict, scale: int = 1000) -> dict:
    """Add ``bboxes_norm`` field computed from ``bboxes``."""
    width = sample.get("width", 0)
    height = sample.get("height", 0)
    norm_failed = 0

    bboxes_norm: list[list[int]] = []
    for bb in sample.get("bboxes", []):
        nb = normalize_bbox(bb, width, height, scale=scale)
        if nb is None:
            # Fallback: keep a zero-area bbox so lengths stay aligned
            bboxes_norm.append([0, 0, 0, 0])
            norm_failed += 1
        else:
            bboxes_norm.append(nb)

    sample["bboxes_norm"] = bboxes_norm
    sample["_norm_failed"] = norm_failed
    return sample


def preprocess_samples(
    samples: list[dict],
    sort_strategy: SortStrategy = "yx",
    scale: int = 1000,
) -> list[dict]:
    """Run the full preprocessing pipeline on a list of samples.

    1. Clean: remove empty tokens, illegal bboxes; clip out-of-bounds bboxes.
    2. Normalise: add ``bboxes_norm`` scaled to [0, *scale*].
    3. Sort: reorder text boxes according to *sort_strategy*.

    Args:
        samples: List of processed sample dicts.
        sort_strategy: ``"yx"`` or ``"original"``.
        scale: Target coordinate range for normalisation (default 1000).

    Returns:
        List of prepared sample dicts.  Samples that become empty after
        cleaning are dropped with a warning.
    """
    prepared: list[dict] = []
    total_empty = 0
    total_illegal = 0
    total_clipped = 0
    total_norm_failed = 0

    for sample in samples:
        sample = _clean_sample(sample)
        if sample.pop("_skip", False):
            continue

        total_empty += sample.pop("_empty_tokens", 0)
        total_illegal += sample.pop("_illegal_bboxes", 0)
        total_clipped += sample.pop("_bboxes_clipped", 0)

        sample = _add_normalised_bboxes(sample, scale=scale)
        total_norm_failed += sample.pop("_norm_failed", 0)

        if not sample.get("tokens"):
            logger.warning("Sample %s has zero tokens after cleaning — dropping", sample.get("id"))
            continue

        sample = sort_sample(sample, strategy=sort_strategy)
        prepared.append(sample)

    logger.info(
        "Preprocessing summary: empty_tokens=%d  illegal_bboxes=%d  "
        "clipped=%d  norm_failed=%d  samples_before=%d  samples_after=%d",
        total_empty,
        total_illegal,
        total_clipped,
        total_norm_failed,
        len(samples),
        len(prepared),
    )
    return prepared


def load_processed_json(path: Path) -> list[dict]:
    """Load a processed JSON file (list of sample dicts)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def save_prepared_json(samples: list[dict], path: Path) -> None:
    """Save prepared samples to JSON, stripping internal ``_`` keys."""
    clean_samples = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in samples
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d samples → %s", len(clean_samples), path)
