"""Evaluation metrics for FUNSD token classification.

Provides token-level accuracy, token-level macro F1, and entity-level
precision / recall / F1 via seqeval, with per-class breakdowns.
"""

import numpy as np
from seqeval.metrics import (
    classification_report as seq_classification_report,
    f1_score as seq_f1_score,
    precision_score as seq_precision_score,
    recall_score as seq_recall_score,
)
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics_from_tags(y_true, y_pred):
    """Compute all metrics from lists of tag strings (per-sample lists).

    Args:
        y_true: list of list of gold BIO tag strings, e.g.
            [["O", "B-QUESTION", "I-QUESTION"], ["O", "O"]]
        y_pred: list of list of predicted BIO tag strings, same shape.

    Returns:
        dict with keys:
            token_accuracy, token_macro_f1,
            entity_precision, entity_recall, entity_f1,
            per-class F1 for QUESTION, ANSWER, HEADER (B/I each),
            classification_report (dict).
    """
    # ----- token-level metrics (flattened) -----
    flat_true = [t for seq in y_true for t in seq]
    flat_pred = [t for seq in y_pred for t in seq]

    token_acc = float(accuracy_score(flat_true, flat_pred))
    token_macro_f1 = float(f1_score(flat_true, flat_pred, average="macro"))

    # ----- entity-level metrics via seqeval -----
    entity_precision = float(seq_precision_score(y_true, y_pred))
    entity_recall = float(seq_recall_score(y_true, y_pred))
    entity_f1 = float(seq_f1_score(y_true, y_pred))

    # ----- per-class F1 -----
    report = seq_classification_report(y_true, y_pred, output_dict=True)
    per_class = {}
    for entity in ["QUESTION", "ANSWER", "HEADER"]:
        per_class[entity] = report.get(entity, {}).get("f1-score", 0.0)

    return {
        "token_accuracy": token_acc,
        "token_macro_f1": token_macro_f1,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "per_class": per_class,
        "classification_report": report,
    }


def format_metrics_table(metrics):
    """Render metrics dict as a Markdown table string."""
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Token Accuracy | {metrics['token_accuracy']:.4f} |",
        f"| Token Macro F1 | {metrics['token_macro_f1']:.4f} |",
        f"| Entity Precision | {metrics['entity_precision']:.4f} |",
        f"| Entity Recall | {metrics['entity_recall']:.4f} |",
        f"| Entity F1 | {metrics['entity_f1']:.4f} |",
    ]
    for cls_name, f1_val in metrics["per_class"].items():
        lines.append(f"| {cls_name} F1 | {f1_val:.4f} |")
    return "\n".join(lines)
