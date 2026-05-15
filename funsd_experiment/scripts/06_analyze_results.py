"""Orchestration script: run model comparison and error analysis.

Usage:
    python scripts/06_analyze_results.py \
        --metric_dir funsd_experiment/outputs/metrics \
        --prediction_dir funsd_experiment/outputs/predictions \
        --analysis_out funsd_experiment/outputs/analysis \
        --report_table_dir funsd_experiment/report/tables \
        --experiment_log funsd_experiment/report/experiment_log.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure funsd_experiment/src is on the import path.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run model comparison and error analysis for FUNSD baselines"
    )
    p.add_argument("--metric_dir", default="funsd_experiment/outputs/metrics")
    p.add_argument("--prediction_dir", default="funsd_experiment/outputs/predictions")
    p.add_argument("--analysis_out", default="funsd_experiment/outputs/analysis")
    p.add_argument("--report_table_dir", default="funsd_experiment/report/tables")
    p.add_argument("--experiment_log", default="funsd_experiment/report/experiment_log.md")
    return p.parse_args()


def _check_files(metric_dir: Path, prediction_dir: Path):
    """Print which metrics / prediction files exist and which are missing."""
    models = ["text_only", "text_layout", "layoutlmv3"]
    print("\n=== File Status ===")
    for m in models:
        mf = metric_dir / f"{m}_test_metrics.json"
        pf = prediction_dir / f"{m}_test_predictions.json"
        m_status = "EXISTS" if mf.exists() else "MISSING"
        p_status = "EXISTS" if pf.exists() else "MISSING"
        print(f"  {m:12s}  metrics: {m_status:7s}  predictions: {p_status:7s}")
    print()


def _ensure_experiment_log(path: Path):
    """Create or update experiment_log.md with current project phase status."""
    models = ["text_only", "text_layout", "layoutlmv3"]
    metric_dir = path.parent.parent / "outputs" / "metrics"

    lines = [
        "# FUNSD Experiment Log",
        "",
        "## Project Phases",
        "",
        "| Phase | Description | Status | Metrics |",
        "|-------|-------------|--------|---------|",
        "| 1 | Project structure initialisation | Complete | — |",
        "| 2 | FUNSD data reader, prepare, preprocess & stats | Complete | `data_stats.json` |",
    ]

    # Phase 3: OCR-only baseline
    m_path = metric_dir / "text_only_test_metrics.json"
    if m_path.exists():
        lines.append(
            "| 3 | OCR-only baseline (distilbert-base-uncased) | Smoketest done | "
            "`text_only_test_metrics.json` |"
        )
    else:
        lines.append(
            "| 3 | OCR-only baseline (distilbert-base-uncased) | Code ready, smoketest pending | "
            "TODO |"
        )

    # Phase 4: Text + Layout baseline
    m_path = metric_dir / "text_layout_test_metrics.json"
    if m_path.exists():
        lines.append(
            "| 4 | Text+Layout baseline (distilbert + bbox MLP) | Smoketest done | "
            "`text_layout_test_metrics.json` |"
        )
    else:
        lines.append(
            "| 4 | Text+Layout baseline (distilbert + bbox MLP) | Code ready, smoketest pending | "
            "TODO |"
        )

    # Phase 5: LayoutLMv3
    m_path = metric_dir / "layoutlmv3_test_metrics.json"
    if m_path.exists():
        lines.append(
            "| 5 | LayoutLMv3 baseline | Smoketest done | "
            "`layoutlmv3_test_metrics.json` |"
        )
    else:
        lines.append(
            "| 5 | LayoutLMv3 baseline | Code ready, smoketest pending | "
            "TODO |"
        )

    # Phase 6
    lines.extend(
        [
            "| 6 | Unified evaluation & error analysis | Code ready | See `report/tables/` |",
            "",
            "## Notes",
            "",
            "Metrics with actual numeric values reflect completed training runs.",
            'Entries marked "Code ready, smoketest pending" indicate the training script',
            "is implemented but has not yet been run to convergence.",
            "",
            f"*Last updated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Experiment log written to %s", path)


def main():
    args = _parse_args()

    metric_dir = Path(args.metric_dir)
    prediction_dir = Path(args.prediction_dir)
    analysis_out = Path(args.analysis_out)
    report_table_dir = Path(args.report_table_dir)
    experiment_log = Path(args.experiment_log)

    for d in [analysis_out, report_table_dir, experiment_log.parent]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1. file status ----
    _check_files(metric_dir, prediction_dir)

    # ---- 2. model comparison ----
    from eval.compare_results import run_default_comparison

    logger.info("Running model comparison …")
    run_default_comparison(
        metric_dir=str(metric_dir),
        report_table_dir=str(report_table_dir),
        analysis_out=str(analysis_out),
    )

    # ---- 3. error analysis ----
    from eval.error_analysis import analyze_prediction_errors

    logger.info("Running error analysis …")
    analyze_prediction_errors(
        prediction_dir=str(prediction_dir),
        analysis_out=str(analysis_out),
        report_table_dir=str(report_table_dir),
    )

    # ---- 4. experiment log ----
    _ensure_experiment_log(experiment_log)

    logger.info("All done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
