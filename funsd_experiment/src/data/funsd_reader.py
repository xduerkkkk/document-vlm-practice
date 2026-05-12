"""
FUNSD dataset reader. Converts raw FUNSD annotations into a unified
word-level BIO-tagged format usable by text-only, text+layout, and
LayoutLM-family models.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Mapping from FUNSD entity labels to BIO prefix labels.
# "other" words are all tagged "O" (Outside).
LABEL_MAP: dict[str, str] = {
    "question": "QUESTION",
    "answer": "ANSWER",
    "header": "HEADER",
    "other": "O",
}


def _convert_entity_words_to_bio(
    words: list[dict],
    label: str,
) -> tuple[list[str], list[list[int]], list[str]]:
    """Convert one entity's word list into tokens, bboxes, and BIO tags.

    Args:
        words: List of word dicts, each with ``text`` and ``box``.
        label: Original FUNSD label (question / answer / header / other).

    Returns:
        (tokens, bboxes, bio_labels) — three parallel lists.
    """
    bio_prefix = LABEL_MAP[label]

    tokens: list[str] = []
    bboxes: list[list[int]] = []
    tags: list[str] = []

    for i, w in enumerate(words):
        text = (w.get("text") or "").strip()
        box = w.get("box")

        if not text:
            logger.debug("Skipping empty word text in entity label=%s", label)
            continue

        if not isinstance(box, (list, tuple)) or len(box) != 4:
            logger.warning(
                "Skipping word with invalid bbox: text=%r box=%s", text, box
            )
            continue

        x1, y1, x2, y2 = box
        if x2 < x1 or y2 < y1:
            logger.warning(
                "Skipping word with illegal bbox (x2<x1 or y2<y1): text=%r box=%s",
                text,
                box,
            )
            continue

        tokens.append(text)
        bboxes.append([x1, y1, x2, y2])

        if bio_prefix == "O":
            tags.append("O")
        elif i == 0:
            tags.append(f"B-{bio_prefix}")
        else:
            tags.append(f"I-{bio_prefix}")

    return tokens, bboxes, tags


def read_annotation(
    ann_path: Path,
    image_dir: Path,
) -> Optional[dict]:
    """Read a single FUNSD annotation file and its corresponding image.

    Returns *None* when the annotation or image cannot be read, or when
    the sample is empty after validation.
    """

    sample_id = ann_path.stem  # e.g. "000000"

    # Locate image (png / jpg / jpeg)
    image_path: Optional[Path] = None
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        candidate = image_dir / f"{sample_id}{ext}"
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        logger.warning(
            "Image not found for sample %s in %s", sample_id, image_dir
        )
        return None

    # Read annotation JSON
    try:
        ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read annotation %s: %s", ann_path, exc)
        return None

    # Read image dimensions
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except (OSError, IOError) as exc:
        logger.warning("Failed to open image %s: %s", image_path, exc)
        return None

    # Parse form entities
    form_entities = ann_data.get("form", [])
    if not isinstance(form_entities, list):
        logger.warning("Annotation %s has no valid 'form' list", sample_id)
        return None

    all_tokens: list[str] = []
    all_bboxes: list[list[int]] = []
    all_labels: list[str] = []

    for entity in form_entities:
        words = entity.get("words") or []
        label = entity.get("label", "other")

        if label not in LABEL_MAP:
            logger.debug(
                "Unknown label %r in sample %s — treating as 'other'", label, sample_id
            )
            label = "other"

        tokens, bboxes, tags = _convert_entity_words_to_bio(words, label)
        if tokens:
            all_tokens.extend(tokens)
            all_bboxes.extend(bboxes)
            all_labels.extend(tags)

    if not all_tokens:
        logger.warning("Sample %s has zero valid tokens after parsing", sample_id)
        return None

    # Consistency check
    n = len(all_tokens)
    if len(all_bboxes) != n or len(all_labels) != n:
        logger.error(
            "Length mismatch in sample %s: tokens=%d bboxes=%d labels=%d — skipping",
            sample_id,
            n,
            len(all_bboxes),
            len(all_labels),
        )
        return None

    return {
        "id": sample_id,
        "image_path": str(image_path.resolve()),
        "width": width,
        "height": height,
        "tokens": all_tokens,
        "bboxes": all_bboxes,
        "labels": all_labels,
    }


def read_dataset(
    ann_dir: Path,
    image_dir: Path,
) -> list[dict]:
    """Read all FUNSD samples from a single split directory.

    Args:
        ann_dir: Directory containing ``*.json`` annotation files.
        image_dir: Directory containing ``*.png`` image files.

    Returns:
        List of processed sample dicts. Invalid / empty samples are
        skipped with warnings.
    """
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    ann_files = sorted(ann_dir.glob("*.json"))
    samples: list[dict] = []

    for af in ann_files:
        sample = read_annotation(af, image_dir)
        if sample is not None:
            samples.append(sample)

    return samples
