"""Prediction error analysis for FUNSD token classification.

Reads per-model prediction JSON files and classifies each token-level
error into one of six types, then exports summary tables and error-case
samples.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# error classification helpers
# ---------------------------------------------------------------------------

def _is_entity(tag: str) -> bool:
    """True if *tag* is a BIO entity (not 'O')."""
    return tag != "O"


def _entity_type(tag: str) -> str:
    """Return the entity category (QUESTION / ANSWER / HEADER) or 'O'."""
    if tag == "O" or not tag.startswith(("B-", "I-")):
        return "O"
    return tag[2:]  # strip B- or I- prefix


def _bio_prefix(tag: str) -> str:
    """Return 'B', 'I', or 'O'."""
    if tag == "O":
        return "O"
    return tag[0]


def _classify_error(
    gold: str,
    pred: str,
    prev_pred: str | None,
) -> str | None:
    """Classify a single token-level error.

    Returns one of:
        O_to_entity, entity_to_O,
        question_answer_confusion, header_confusion,
        boundary_error, other_label_confusion,
    or None if the token is correct.
    """
    if gold == pred:
        return None

    gold_et = _entity_type(gold)
    pred_et = _entity_type(pred)

    # 1. O ↔ entity
    if gold == "O" and pred != "O":
        return "O_to_entity"
    if gold != "O" and pred == "O":
        return "entity_to_O"

    # 2. QUESTION ↔ ANSWER confusion
    qa_types = {"QUESTION", "ANSWER"}
    if gold_et in qa_types and pred_et in qa_types and gold_et != pred_et:
        return "question_answer_confusion"

    # 3. HEADER confusion
    if (gold_et == "HEADER" and pred_et != "HEADER") or (
        gold_et != "HEADER" and pred_et == "HEADER"
    ):
        return "header_confusion"

    # 4. boundary / BIO format error
    if _entity_type(gold) == _entity_type(pred) and gold != pred:
        # Same entity type but different B/I — boundary error.
        return "boundary_error"

    # 5. catch-all
    return "other_label_confusion"


def _detect_boundary_errors(pred_tags: list[str]) -> set[int]:
    """Return indices where predicted tags violate BIO consistency."""
    errors: set[int] = set()
    for i, tag in enumerate(pred_tags):
        if tag.startswith("I-"):
            if i == 0:
                errors.add(i)
                continue
            prev_type = _entity_type(pred_tags[i - 1])
            curr_type = _entity_type(tag)
            if prev_type != curr_type:
                errors.add(i)
    return errors


# ---------------------------------------------------------------------------
# context extraction
# ---------------------------------------------------------------------------

def _context_tokens(tokens: list[str], idx: int, window: int = 5) -> list[str]:
    """Return a window of *tokens* around position *idx*."""
    lo = max(0, idx - window)
    hi = min(len(tokens), idx + window + 1)
    return tokens[lo:hi]


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def analyze_prediction_errors(
    prediction_dir: str = "funsd_experiment/outputs/predictions",
    analysis_out: str = "funsd_experiment/outputs/analysis",
    report_table_dir: str = "funsd_experiment/report/tables",
    model_names: list[str] | None = None,
):
    """Run error analysis across model predictions.

    Args:
        prediction_dir: Directory containing prediction JSON files.
        analysis_out: Directory for JSON analysis output.
        report_table_dir: Directory for Markdown summary tables.
        model_names: Model names to analyse (default: standard three).
    """
    if model_names is None:
        model_names = ["text_only", "text_layout", "layoutlmv3"]

    prediction_dir = Path(prediction_dir)
    analysis_out = Path(analysis_out)
    report_table_dir = Path(report_table_dir)
    analysis_out.mkdir(parents=True, exist_ok=True)
    report_table_dir.mkdir(parents=True, exist_ok=True)

    all_error_cases: dict[str, list[dict]] = {}
    error_counts: dict[str, dict[str, int]] = {}

    error_type_order = [
        "O_to_entity",
        "entity_to_O",
        "question_answer_confusion",
        "header_confusion",
        "boundary_error",
        "other_label_confusion",
    ]

    for model_name in model_names:
        pred_path = prediction_dir / f"{model_name}_test_predictions.json"
        if not pred_path.exists():
            logger.warning("Predictions not found for %s: %s", model_name, pred_path)
            error_counts[model_name] = {t: 0 for t in error_type_order}
            error_counts[model_name]["_status"] = "missing"
            all_error_cases[model_name] = []
            continue

        with open(pred_path, encoding="utf-8") as f:
            samples = json.load(f)

        counts: dict[str, int] = {t: 0 for t in error_type_order}
        # Per-error-type sample accumulator (max 5 per type).
        case_collectors: dict[str, list[dict]] = {t: [] for t in error_type_order}

        for sample in samples:
            tokens = sample["tokens"]
            gold_tags = sample["gold_labels"]
            pred_tags = sample["pred_labels"]
            bboxes = sample.get("bboxes_norm", None)

            n = min(len(tokens), len(gold_tags), len(pred_tags))
            if len(tokens) != n or len(gold_tags) != n or len(pred_tags) != n:
                logger.warning(
                    "Length mismatch for sample %s: tokens=%d gold=%d pred=%d",
                    sample["id"], len(tokens), len(gold_tags), len(pred_tags),
                )

            boundary_errs = _detect_boundary_errors(pred_tags)

            for i in range(n):
                gold = gold_tags[i]
                pred = pred_tags[i]
                err_type = _classify_error(
                    gold, pred,
                    prev_pred=pred_tags[i - 1] if i > 0 else None,
                )
                if err_type is None:
                    continue

                # If _classify_error missed a boundary error caught by consistency check.
                if err_type not in ("O_to_entity", "entity_to_O") and i in boundary_errs:
                    if _entity_type(gold) == _entity_type(pred):
                        err_type = "boundary_error"

                counts[err_type] = counts.get(err_type, 0) + 1

                # Collect error case (up to 5 per type per model).
                collector = case_collectors.get(err_type, None)
                if collector is not None and len(collector) < 5:
                    collector.append(
                        {
                            "sample_id": sample["id"],
                            "token_index": i,
                            "token": tokens[i],
                            "gold_label": gold,
                            "pred_label": pred,
                            "error_type": err_type,
                            "context_tokens": _context_tokens(tokens, i, 5),
                            "bbox": bboxes[i] if bboxes and i < len(bboxes) else None,
                        }
                    )

        error_counts[model_name] = counts
        all_error_cases[model_name] = [
            {"error_type": t, "cases": case_collectors[t]}
            for t in error_type_order
        ]

    # ---- summary Markdown table ----
    lines = [
        "| Error Type | " + " | ".join(model_names) + " |",
        "|------------|" + "|".join(["------------|"] * len(model_names)),
    ]
    for etype in error_type_order:
        vals = []
        for m in model_names:
            c = error_counts[m].get(etype, "—")
            vals.append(str(c) if isinstance(c, int) else c)
        lines.append(f"| {etype} | " + " | ".join(vals) + " |")
    summary_md = "\n".join(lines)

    summary_path = report_table_dir / "error_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Error Analysis Summary\n\n")
        f.write(summary_md)
        f.write("\n\n*Error types explained:*\n")
        f.write("- **O_to_entity**: gold=O, predicted entity\n")
        f.write("- **entity_to_O**: gold=entity, predicted O\n")
        f.write("- **question_answer_confusion**: QUESTION ↔ ANSWER mix-up\n")
        f.write("- **header_confusion**: HEADER confused with other entity types\n")
        f.write("- **boundary_error**: B/I tag boundary inconsistency\n")
        f.write("- **other_label_confusion**: all other label errors\n")

    # ---- error cases JSON ----
    cases_path = analysis_out / "error_cases.json"
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(all_error_cases, f, indent=2, ensure_ascii=False)

    logger.info("Error analysis complete. Summary: %s", summary_path)
    return all_error_cases, error_counts
