"""
Text-box ordering strategies.

FUNSD annotations are already in reading order for most samples, but
re-sorting by (y0, x0) provides a deterministic layout-consistent
order that downstream models can rely on.
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

SortStrategy = Literal["original", "yx"]


def sort_sample(
    sample: dict,
    strategy: SortStrategy = "yx",
) -> dict:
    """Reorder *tokens*, *bboxes*, *bboxes_norm* (if present) and *labels*
    inside *sample* according to *strategy*.

    Args:
        sample: A single processed/prepared sample dict.
        strategy: ``"original"`` (identity) or ``"yx"`` (top-to-bottom,
                  left-to-right).

    Returns:
        The same dict with reordered lists (mutated in place).
    """
    if strategy == "original":
        return sample

    if strategy != "yx":
        raise ValueError(f"Unknown sort strategy: {strategy}")

    tokens = sample.get("tokens", [])
    bboxes = sample.get("bboxes", [])
    labels = sample.get("labels", [])
    bboxes_norm = sample.get("bboxes_norm")  # may be missing

    n = len(tokens)
    if len(bboxes) != n or len(labels) != n:
        logger.warning("Length mismatch in sample %s — skipping sort", sample.get("id"))
        return sample

    has_norm = bboxes_norm is not None and len(bboxes_norm) == n

    # Build sort keys: (y0, x0) from the bbox the caller wants us to
    # sort by.  Prefer bboxes_norm when available so the order is
    # resolution-independent; fall back to raw bboxes.
    bbox_ref = bboxes_norm if has_norm else bboxes

    # Pack, sort, unpack
    triples = list(zip(bbox_ref, tokens, bboxes, labels))
    triples.sort(key=lambda t: (t[0][1], t[0][0]))  # (y0, x0)

    sorted_tokens = [t[1] for t in triples]
    sorted_bboxes = [t[2] for t in triples]
    sorted_labels = [t[3] for t in triples]

    sample["tokens"] = sorted_tokens
    sample["bboxes"] = sorted_bboxes
    sample["labels"] = sorted_labels

    if has_norm:
        sorted_norm = [t[0] for t in triples]
        sample["bboxes_norm"] = sorted_norm

    return sample
