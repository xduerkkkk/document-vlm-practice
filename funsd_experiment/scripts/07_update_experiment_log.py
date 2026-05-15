"""Update experiment_log.md with formal metrics from completed training runs.

Reads per-model metrics JSON files and writes / appends a structured
experiment log suitable for the final report.

Usage:
    python scripts/07_update_experiment_log.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths relative to the repository root (this script lives in funsd_experiment/scripts/).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_METRIC_DIR = _REPO_ROOT / "funsd_experiment" / "outputs" / "metrics"
_LOG_PATH = _REPO_ROOT / "funsd_experiment" / "report" / "experiment_log.md"

_MODEL_CONFIGS = {
    "text_only": {
        "display": "OCR-Only Baseline",
        "input": "tokens",
        "model": "distilbert-base-uncased",
        "epochs": 5,
        "batch_size": 4,
        "lr": 5e-5,
        "max_length": 512,
        "seed": 42,
    },
    "text_layout": {
        "display": "Text + Layout Baseline",
        "input": "tokens + bboxes_norm",
        "model": "distilbert-base-uncased + bbox MLP",
        "epochs": 5,
        "batch_size": 4,
        "lr": 5e-5,
        "max_length": 512,
        "seed": 42,
    },
    "layoutlmv3": {
        "display": "LayoutLMv3 Baseline",
        "input": "tokens + bboxes_norm + image",
        "model": "microsoft/layoutlmv3-base",
        "epochs": 5,
        "batch_size": 2,
        "lr": 5e-5,
        "max_length": 512,
        "seed": 42,
    },
}

_OVERALL_KEYS = [
    ("token_accuracy", "Token Accuracy"),
    ("token_macro_f1", "Token Macro F1"),
    ("entity_precision", "Entity Precision"),
    ("entity_recall", "Entity Recall"),
    ("entity_f1", "Entity F1"),
]

_CLASS_KEYS = ["QUESTION", "ANSWER", "HEADER"]


def _load_metrics(model_key: str) -> dict | None:
    path = _METRIC_DIR / f"{model_key}_test_metrics.json"
    if not path.exists():
        logger.warning("Metrics not found: %s", path)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(val) -> str:
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _build_overall_table(metrics_by_model: dict[str, dict | None]) -> str:
    available = [k for k, v in metrics_by_model.items() if v is not None]
    if not available:
        return "*(no metrics available)*"

    header = (
        "| Metric | "
        + " | ".join(_MODEL_CONFIGS[m]["display"] for m in available)
        + " |"
    )
    sep = "|" + "|".join(["--------"] * (len(available) + 1)) + "|"
    rows = [header, sep]
    for key, label in _OVERALL_KEYS:
        vals = [_fmt(metrics_by_model[m].get(key)) for m in available]
        rows.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def _build_class_table(metrics_by_model: dict[str, dict | None]) -> str:
    available = [k for k, v in metrics_by_model.items() if v is not None]
    if not available:
        return "*(no metrics available)*"

    header = (
        "| Category | "
        + " | ".join(_MODEL_CONFIGS[m]["display"] for m in available)
        + " |"
    )
    sep = "|" + "|".join(["----------"] * (len(available) + 1)) + "|"
    rows = [header, sep]
    for cls in _CLASS_KEYS:
        vals = []
        for m in available:
            pc = metrics_by_model[m].get("per_class", {})
            vals.append(_fmt(pc.get(cls)) if isinstance(pc, dict) else "N/A")
        rows.append(f"| {cls} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main():
    # Load all metrics.
    metrics_by_model: dict[str, dict | None] = {}
    for key in _MODEL_CONFIGS:
        metrics_by_model[key] = _load_metrics(key)

    # Build content.
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# FUNSD Experiment Log",
        "",
        f"*Last updated: {now}*",
        "",
        "## Project Phases",
        "",
        "| Phase | Description | Status |",
        "|-------|-------------|--------|",
        "| 1 | Project structure initialisation | Complete |",
        "| 2 | FUNSD data reader, prepare, preprocess & stats | Complete |",
        "| 3 | OCR-only baseline | Complete |",
        "| 4 | Text + Layout baseline | Complete |",
        "| 5 | LayoutLMv3 baseline | Complete |",
        "| 6 | Unified evaluation & error analysis | Complete |",
        "",
        "## Experimental Setup",
        "",
        "- **Dataset**: FUNSD prepared (train=127, val=22, test=50)",
        "- **Labels**: BIO tagging (O, B/I-QUESTION, B/I-ANSWER, B/I-HEADER)",
        "- **Evaluation**: token accuracy, macro F1, entity-level P/R/F1 (seqeval)",
        "",
        "### Model Configurations",
        "",
    ]

    for key, cfg in _MODEL_CONFIGS.items():
        status = "**Metrics available**" if metrics_by_model[key] else "Metrics pending (TODO)"
        lines.append(f"#### {cfg['display']}")
        lines.append(f"- **Model**: {cfg['model']}")
        lines.append(f"- **Input**: {cfg['input']}")
        lines.append(
            f"- **Hyper-params**: epochs={cfg['epochs']}, "
            f"batch_size={cfg['batch_size']}, "
            f"lr={cfg['lr']}, "
            f"max_length={cfg['max_length']}, "
            f"seed={cfg['seed']}"
        )
        lines.append(f"- **Status**: {status}")
        lines.append("")

    # ---- Results section ----
    lines.append("## Results")
    lines.append("")
    lines.append("### Overall Metrics")
    lines.append("")
    lines.append(_build_overall_table(metrics_by_model))
    lines.append("")
    lines.append("### Per-Class Entity F1")
    lines.append("")
    lines.append(_build_class_table(metrics_by_model))
    lines.append("")
    lines.append(
        "*Metrics sourced from `funsd_experiment/outputs/metrics/*_test_metrics.json`*"
    )

    # Write.
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Experiment log written to %s", _LOG_PATH)

    # Print summary to stdout.
    for key, cfg in _MODEL_CONFIGS.items():
        m = metrics_by_model[key]
        if m:
            print(f"  {cfg['display']:25s}  Entity F1 = {m.get('entity_f1', 'N/A')}")
        else:
            print(f"  {cfg['display']:25s}  [MISSING]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
