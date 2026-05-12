"""
Dataset statistics for FUNSD processed / prepared splits.

Produces both a JSON metrics file and a Markdown table report suitable
for inclusion in experiment write-ups.
"""

import json
import logging
import math
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

logger = logging.getLogger(__name__)

# Characters considered "printable" for OCR-noise detection.
_PRINTABLE = set(string.printable)


def _percent(value: float, total: float) -> str:
    if total == 0:
        return "0.00%"
    return f"{value / total * 100:.2f}%"


def _is_non_printable(token: str) -> bool:
    """Return True when *token* contains characters outside ASCII printable range."""
    return any(ch not in _PRINTABLE for ch in token)


def _unusual_symbol_ratio(token: str) -> float:
    """Ratio of non-alphanumeric, non-whitespace characters in *token*."""
    if not token:
        return 0.0
    unusual = sum(1 for ch in token if not ch.isalnum() and not ch.isspace())
    return unusual / len(token)


# ---- per-split statistics ---------------------------------------------------

def compute_split_stats(
    samples: list[dict],
    split_name: str,
    max_seq_len: int = 512,
) -> dict[str, Any]:
    """Compute statistics for a single data split.

    Args:
        samples: List of prepared sample dicts.
        split_name: e.g. ``"train"``.
        max_seq_len: Threshold for the "pages over N tokens" stat (default 512).

    Returns:
        Nested dict with all statistics for this split.
    """
    num_samples = len(samples)
    if num_samples == 0:
        return {"split": split_name, "num_samples": 0}

    # Per-page lengths
    token_counts = [len(s["tokens"]) for s in samples]
    bbox_counts = [len(s["bboxes"]) for s in samples]

    # Label distributions
    label_counter: Counter[str] = Counter()
    bio_counter: Counter[str] = Counter()
    entity_counter: Counter[str] = Counter()  # QUESTION / ANSWER / HEADER / O

    for s in samples:
        for lb in s.get("labels", []):
            label_counter[lb] += 1
            bio_counter[lb] += 1
            # Derive entity category from BIO tag
            if lb == "O":
                entity_counter["O"] += 1
            elif lb.startswith("B-") or lb.startswith("I-"):
                entity = lb[2:]  # strip B- / I- prefix
                entity_counter[entity] += 1
            else:
                entity_counter["UNKNOWN"] += 1

    # Pages over max_seq_len
    over_count = sum(1 for c in token_counts if c > max_seq_len)

    # Cleaning stats (these are added by preprocess.py as _ prefixed keys;
    # they will be 0 if the data was run through cleaning but had no issues.)
    # We look for any sample that still has the temporary keys — they should
    # have been stripped by save_prepared_json, so we estimate from the data.
    # Real counts come from the preprocessing logs.

    # OCR noise
    non_printable_examples: list[str] = []
    high_symbol_examples: list[dict] = []
    long_token_examples: list[str] = []

    for s in samples:
        for tok in s.get("tokens", []):
            if len(non_printable_examples) < 20 and _is_non_printable(tok):
                non_printable_examples.append(tok)
            ratio = _unusual_symbol_ratio(tok)
            if len(high_symbol_examples) < 20 and ratio > 0.5 and len(tok) > 1:
                high_symbol_examples.append({"token": tok, "symbol_ratio": round(ratio, 3)})
            if len(long_token_examples) < 20 and len(tok) > 30:
                long_token_examples.append(tok)

    # Non-printable count across all tokens
    total_tokens = sum(token_counts)
    np_count = sum(
        1 for s in samples for tok in s["tokens"] if _is_non_printable(tok)
    )

    return {
        "split": split_name,
        "num_samples": num_samples,
        "num_tokens_total": total_tokens,
        "tokens_per_page": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": round(mean(token_counts), 1),
            "median": round(median(token_counts), 1),
        },
        "bboxes_per_page": {
            "min": min(bbox_counts),
            "max": max(bbox_counts),
            "mean": round(mean(bbox_counts), 1),
            "median": round(median(bbox_counts), 1),
        },
        "label_distribution": dict(
            sorted(entity_counter.items(), key=lambda x: -x[1])
        ),
        "bio_label_distribution": dict(
            sorted(bio_counter.items(), key=lambda x: -x[1])
        ),
        "pages_over_max_seq_len": {
            "threshold": max_seq_len,
            "count": over_count,
            "ratio": _percent(over_count, num_samples),
        },
        "ocr_noise_summary": {
            "non_printable_tokens_total": np_count,
            "non_printable_ratio": _percent(np_count, total_tokens),
            "non_printable_examples": non_printable_examples[:10],
            "high_symbol_ratio_examples": high_symbol_examples[:10],
            "long_token_examples": long_token_examples[:10],
        },
    }


# ---- combined statistics ----------------------------------------------------

def compute_statistics(
    splits: dict[str, list[dict]],
    json_out: Path,
    md_out: Path,
) -> None:
    """Compute statistics across splits and write JSON + Markdown.

    Args:
        splits: Mapping from split name to sample list, e.g.
                ``{"train": [...], "val": [...], "test": [...]}``.
        json_out: Path for JSON metrics output.
        md_out: Path for Markdown table output.
    """
    all_stats = {
        name: compute_split_stats(samples, name)
        for name, samples in splits.items()
    }

    # Write JSON
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Statistics JSON saved → %s", json_out)

    # Write Markdown
    md_lines = _build_markdown(all_stats)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Statistics Markdown saved → %s", md_out)


def _build_markdown(all_stats: dict[str, dict]) -> list[str]:
    """Build Markdown report lines from stats dict."""
    lines: list[str] = [
        "# FUNSD Dataset Statistics",
        "",
    ]

    # ---- Table 1: Overview ----
    lines += [
        "## 1. Data Overview",
        "",
        "| Split   | Samples | Total Tokens | Tokens/Page (min) | Tokens/Page (max) | Tokens/Page (mean) | Tokens/Page (median) |",
        "|---------|---------|-------------|-------------------|-------------------|--------------------|----------------------|",
    ]
    for name, st in all_stats.items():
        if st["num_samples"] == 0:
            lines.append(f"| {name} | 0 | 0 | - | - | - | - |")
            continue
        tp = st["tokens_per_page"]
        lines.append(
            f"| {name} | {st['num_samples']} | {st['num_tokens_total']} "
            f"| {tp['min']} | {tp['max']} | {tp['mean']} | {tp['median']} |"
        )
    lines.append("")

    # ---- Table 2: Bbox stats ----
    lines += [
        "## 2. Bounding Boxes per Page",
        "",
        "| Split   | Min | Max | Mean | Median |",
        "|---------|-----|-----|------|--------|",
    ]
    for name, st in all_stats.items():
        if st["num_samples"] == 0:
            lines.append(f"| {name} | - | - | - | - |")
            continue
        bp = st["bboxes_per_page"]
        lines.append(
            f"| {name} | {bp['min']} | {bp['max']} | {bp['mean']} | {bp['median']} |"
        )
    lines.append("")

    # ---- Table 3: Entity label distribution ----
    lines += [
        "## 3. Entity Category Distribution",
        "",
        "| Split   | QUESTION | ANSWER | HEADER | O (other) |",
        "|---------|----------|--------|--------|-----------|",
    ]
    for name, st in all_stats.items():
        ld = st.get("label_distribution", {})
        lines.append(
            f"| {name} | {ld.get('QUESTION', 0)} | {ld.get('ANSWER', 0)} "
            f"| {ld.get('HEADER', 0)} | {ld.get('O', 0)} |"
        )
    lines.append("")

    # ---- Table 4: BIO label distribution ----
    lines += [
        "## 4. BIO Label Distribution",
        "",
        "| Split   | B-QUESTION | I-QUESTION | B-ANSWER | I-ANSWER | B-HEADER | I-HEADER | O |",
        "|---------|------------|------------|----------|----------|----------|----------|---|",
    ]
    bio_keys = [
        "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER",
        "B-HEADER", "I-HEADER", "O",
    ]
    for name, st in all_stats.items():
        bd = st.get("bio_label_distribution", {})
        vals = " | ".join(str(bd.get(k, 0)) for k in bio_keys)
        lines.append(f"| {name} | {vals} |")
    lines.append("")

    # ---- Table 5: Pages over max_seq_len ----
    lines += [
        "## 5. Pages Exceeding Max Sequence Length",
        "",
        "| Split   | Threshold | Count | Ratio |",
        "|---------|-----------|-------|-------|",
    ]
    for name, st in all_stats.items():
        ov = st.get("pages_over_max_seq_len", {})
        if ov:
            lines.append(
                f"| {name} | {ov.get('threshold', 512)} "
                f"| {ov.get('count', 0)} | {ov.get('ratio', '0%')} |"
            )
    lines.append("")

    # ---- Table 6: OCR noise summary ----
    lines += [
        "## 6. OCR Noise Summary",
        "",
        "| Split   | Non-printable Tokens | Non-printable Ratio | High-symbol Examples | Long Token Examples |",
        "|---------|----------------------|--------------------|----------------------|---------------------|",
    ]
    for name, st in all_stats.items():
        ns = st.get("ocr_noise_summary", {})
        hp_examples = ", ".join(
            e["token"] for e in ns.get("high_symbol_ratio_examples", [])[:3]
        ) or "-"
        lt_examples = ", ".join(ns.get("long_token_examples", [])[:3]) or "-"
        lines.append(
            f"| {name} | {ns.get('non_printable_tokens_total', 0)} "
            f"| {ns.get('non_printable_ratio', '0%')} "
            f"| {hp_examples} | {lt_examples} |"
        )
    lines.append("")

    return lines


def print_stats_summary(all_stats: dict[str, dict]) -> None:
    """Print a compact summary to the console."""
    for name, st in all_stats.items():
        ns = st["num_samples"]
        tt = st.get("num_tokens_total", 0)
        tp = st.get("tokens_per_page", {})
        ov = st.get("pages_over_max_seq_len", {})
        ld = st.get("label_distribution", {})

        logger.info(
            "%-5s | samples=%3d  tokens=%5d  "
            "tok/page: min=%3d max=%3d mean=%6.1f median=%5.1f  "
            ">512: %s (%s)  "
            "labels: Q=%d A=%d H=%d O=%d",
            name,
            ns,
            tt,
            tp.get("min", 0),
            tp.get("max", 0),
            tp.get("mean", 0.0),
            tp.get("median", 0.0),
            ov.get("count", 0),
            ov.get("ratio", "0%"),
            ld.get("QUESTION", 0),
            ld.get("ANSWER", 0),
            ld.get("HEADER", 0),
            ld.get("O", 0),
        )
