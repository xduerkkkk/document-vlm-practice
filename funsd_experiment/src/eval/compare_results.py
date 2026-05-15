"""Cross-model metrics comparison.

Reads per-model metrics JSON files and produces:
- overall comparison table  (Markdown)
- class-level comparison table (Markdown)
- combined JSON summary
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Overall metrics keys to include in the comparison table.
_OVERALL_KEYS = [
    "token_accuracy",
    "token_macro_f1",
    "entity_precision",
    "entity_recall",
    "entity_f1",
]

_OVERALL_LABELS = {
    "token_accuracy": "Token Accuracy",
    "token_macro_f1": "Token Macro F1",
    "entity_precision": "Entity Precision",
    "entity_recall": "Entity Recall",
    "entity_f1": "Entity F1",
}

# Per-class keys (expected inside metrics["per_class"]).
_CLASS_LABELS = ["QUESTION", "ANSWER", "HEADER"]


def _load_metrics(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        logger.warning("Metrics file not found: %s", p)
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_model_comparison(
    model_metrics: dict[str, str],
    overall_out: str,
    class_out: str,
    json_out: str,
) -> dict[str, Any]:
    """Read metrics for each model and write comparison artefacts.

    Args:
        model_metrics: ``{model_name: metrics_json_path}`` mapping.
        overall_out: Path for the overall Markdown comparison table.
        class_out: Path for the per-class Markdown comparison table.
        json_out: Path for the combined JSON summary.

    Returns:
        dict with keys ``available`` (list of models with metrics),
        ``missing`` (list without), ``overall``, ``per_class``.
    """
    results: dict[str, dict | None] = {}
    for model_name, path in model_metrics.items():
        results[model_name] = _load_metrics(path)

    available = [m for m, r in results.items() if r is not None]
    missing = [m for m, r in results.items() if r is None]

    # ---- overall table ----
    overall_lines = [
        "| Metric | "
        + " | ".join(available)
        + (" |" if available else ""),
        "|--------|"
        + "|".join(["--------|"] * len(available)),
    ]
    for key in _OVERALL_KEYS:
        vals = []
        for m in available:
            v = results[m].get(key, None)
            vals.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
        label = _OVERALL_LABELS.get(key, key)
        overall_lines.append(f"| {label} | " + " | ".join(vals) + " |")
    overall_md = "\n".join(overall_lines)

    Path(overall_out).parent.mkdir(parents=True, exist_ok=True)
    with open(overall_out, "w", encoding="utf-8") as f:
        f.write("# Model Comparison — Overall Metrics\n\n")
        f.write(overall_md)
        f.write("\n\n*Missing:* " + (", ".join(missing) if missing else "None") + "\n")

    # ---- class-level table ----
    class_lines = [
        "| Category | "
        + " | ".join(available)
        + (" |" if available else ""),
        "|----------|"
        + "|".join(["----------|"] * len(available)),
    ]
    for cls in _CLASS_LABELS:
        vals = []
        for m in available:
            pc = results[m].get("per_class", {})
            v = pc.get(cls, None) if isinstance(pc, dict) else None
            vals.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
        class_lines.append(f"| {cls} | " + " | ".join(vals) + " |")
    class_md = "\n".join(class_lines)

    Path(class_out).parent.mkdir(parents=True, exist_ok=True)
    with open(class_out, "w", encoding="utf-8") as f:
        f.write("# Model Comparison — Per-Class Entity F1\n\n")
        f.write(class_md)
        f.write("\n\n*Missing:* " + (", ".join(missing) if missing else "None") + "\n")

    # ---- JSON summary ----
    summary = {
        "available": available,
        "missing": missing,
        "overall": {
            m: {k: results[m].get(k) for k in _OVERALL_KEYS}
            for m in available
        },
        "per_class": {
            m: (results[m].get("per_class", {}) if results[m] else None)
            for m in available
        },
    }
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "Comparison done. Available: %s, Missing: %s",
        available, missing,
    )
    return summary


# ---------------------------------------------------------------------------
# Convenience runner for the standard three-model config
# ---------------------------------------------------------------------------

def run_default_comparison(
    metric_dir: str = "funsd_experiment/outputs/metrics",
    report_table_dir: str = "funsd_experiment/report/tables",
    analysis_out: str = "funsd_experiment/outputs/analysis",
):
    """Compare the three standard FUNSD baselines."""
    model_metrics = {
        "text_only": str(Path(metric_dir) / "text_only_test_metrics.json"),
        "text_layout": str(Path(metric_dir) / "text_layout_test_metrics.json"),
        "layoutlmv3": str(Path(metric_dir) / "layoutlmv3_test_metrics.json"),
    }
    return build_model_comparison(
        model_metrics,
        overall_out=str(Path(report_table_dir) / "model_comparison.md"),
        class_out=str(Path(report_table_dir) / "class_level_comparison.md"),
        json_out=str(Path(analysis_out) / "metrics_comparison.json"),
    )
