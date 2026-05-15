"""Train a LayoutLMv3 token classification model on FUNSD.

Uses page images, OCR tokens, and normalised bounding-box coordinates
to predict BIO tags.  This is the full visual-language-model baseline.

Usage:
    python src/train/train_layoutlmv3.py \
        --data_dir funsd_experiment/data/prepared \
        --model_name microsoft/layoutlmv3-base \
        --output_dir funsd_experiment/outputs/checkpoints/layoutlmv3 \
        --epochs 5 --batch_size 2 --learning_rate 5e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments

# Ensure funsd_experiment/src is on the import path.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.layoutlmv3_model import load_layoutlmv3_model
from eval.metrics import compute_metrics_from_tags, format_metrics_table

logger = logging.getLogger(__name__)

# Project root for resolving relative image paths.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train a LayoutLMv3 baseline on FUNSD")
    p.add_argument("--data_dir", default="funsd_experiment/data/prepared")
    p.add_argument("--model_name", default="microsoft/layoutlmv3-base")
    p.add_argument("--output_dir", default="funsd_experiment/outputs/checkpoints/layoutlmv3")
    p.add_argument("--prediction_dir", default="funsd_experiment/outputs/predictions")
    p.add_argument("--metric_dir", default="funsd_experiment/outputs/metrics")
    p.add_argument("--report_table", default="funsd_experiment/report/tables/layoutlmv3_metrics.md")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_json_samples(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_image_path(image_path: str) -> Path:
    """Resolve *image_path* to an absolute, existing path.

    Raises FileNotFoundError if the image does not exist.
    """
    p = Path(image_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return p


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class FUNSDLayoutDataset(TorchDataset):
    """Dataset that loads a page image and encodes it with the LayoutLMv3
    processor on the fly.

    Each ``__getitem__`` call returns a dict of tensors (all squeezed to
    remove the implicit batch dimension added by the processor).
    """

    def __init__(self, samples, processor, label2id, max_length):
        self.samples = samples
        self.processor = processor
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 1. Resolve and load image
        img_path = _resolve_image_path(s["image_path"])
        image = Image.open(img_path).convert("RGB")

        # 2. Convert word labels to ids
        word_labels = [self.label2id[l] for l in s["labels"]]

        # 3. Encode through processor
        enc = self.processor(
            images=image,
            text=s["tokens"],
            boxes=s["bboxes_norm"],
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove the extra batch dimension added by return_tensors="pt".
        for key in list(enc.keys()):
            enc[key] = enc[key].squeeze(0)

        return enc


# ---------------------------------------------------------------------------
# compute_metrics for Trainer
# ---------------------------------------------------------------------------

def _make_compute_metrics(id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        y_true_tags, y_pred_tags = _gather_tags(predictions, labels, id2label)
        return compute_metrics_from_tags(y_true_tags, y_pred_tags)
    return compute_metrics


def _gather_tags(pred_ids, label_ids, id2label):
    """Convert subword-level predictions to per-sample tag lists."""
    y_true, y_pred = [], []
    for p_seq, l_seq in zip(pred_ids, label_ids):
        gold, pred = [], []
        for p, l in zip(p_seq, l_seq):
            if l != -100:
                gold.append(id2label[l])
                pred.append(id2label[p])
        y_true.append(gold)
        y_pred.append(pred)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# prediction output
# ---------------------------------------------------------------------------

def _build_prediction_output(raw_samples, pred_ids, label_ids, id2label):
    """Build per-sample prediction dicts from subword-level model output."""
    results = []
    for i, s in enumerate(raw_samples):
        gold_tags, pred_tags = [], []
        for p, l in zip(pred_ids[i], label_ids[i]):
            if l != -100:
                gold_tags.append(id2label[l])
                pred_tags.append(id2label[p])

        n_kept = len(gold_tags)
        results.append(
            {
                "id": s["id"],
                "tokens": s["tokens"][:n_kept],
                "bboxes_norm": s["bboxes_norm"][:n_kept],
                "gold_labels": gold_tags,
                "pred_labels": pred_tags,
            }
        )
    return results


def _count_truncated(label_ids_batch, raw_samples):
    """Count how many test samples were truncated."""
    n_trunc = 0
    for label_seq, s in zip(label_ids_batch, raw_samples):
        kept = (label_seq != -100).sum()
        if kept < len(s["tokens"]):
            n_trunc += 1
    return n_trunc


def _to_native(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # ---- seed ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- paths ----
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    prediction_dir = Path(args.prediction_dir)
    metric_dir = Path(args.metric_dir)
    report_table = Path(args.report_table)

    for d in [prediction_dir, metric_dir, report_table.parent]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- load raw samples ----
    logger.info("Loading data from %s", data_dir)
    train_samples = _load_json_samples(data_dir / "train.json")
    val_samples = _load_json_samples(data_dir / "val.json")
    test_samples = _load_json_samples(data_dir / "test.json")

    # ---- model, processor, label maps ----
    logger.info("Building model: %s", args.model_name)
    model, processor, label2id, id2label = load_layoutlmv3_model(args.model_name)

    # ---- datasets ----
    logger.info("Creating datasets (max_length=%d) ...", args.max_length)
    train_ds = FUNSDLayoutDataset(train_samples, processor, label2id, args.max_length)
    val_ds = FUNSDLayoutDataset(val_samples, processor, label2id, args.max_length)
    test_ds = FUNSDLayoutDataset(test_samples, processor, label2id, args.max_length)

    # ---- truncation statistics ----
    # Collect labels from test set to count truncation.
    test_labels_all = []
    for i in range(len(test_ds)):
        test_labels_all.append(test_ds[i]["labels"].numpy())
    trunc_count = _count_truncated(test_labels_all, test_samples)
    logger.info("Truncated test samples: %d / %d", trunc_count, len(test_samples))

    # ---- training args ----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_make_compute_metrics(id2label),
    )

    # ---- train ----
    logger.info("Starting training ...")
    trainer.train()

    # ---- evaluate on test set ----
    logger.info("Evaluating on test set ...")
    test_pred_raw = trainer.predict(test_ds)
    test_logits = test_pred_raw.predictions
    test_label_ids = test_pred_raw.label_ids
    test_pred_ids = np.argmax(test_logits, axis=-1)

    test_tags_true, test_tags_pred = _gather_tags(test_pred_ids, test_label_ids, id2label)
    full_metrics = compute_metrics_from_tags(test_tags_true, test_tags_pred)
    full_metrics["num_test_samples"] = len(test_samples)
    full_metrics["num_truncated_samples"] = trunc_count

    # ---- save outputs ----
    # 1. model & processor
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # 2. metrics JSON
    metrics_path = metric_dir / "layoutlmv3_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_to_native(full_metrics), f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", metrics_path)

    # 3. predictions JSON
    predictions = _build_prediction_output(test_samples, test_pred_ids, test_label_ids, id2label)
    preds_path = prediction_dir / "layoutlmv3_test_predictions.json"
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info("Predictions saved to %s", preds_path)

    # 4. Markdown table
    table_md = format_metrics_table(full_metrics)
    with open(report_table, "w", encoding="utf-8") as f:
        f.write("# LayoutLMv3 Baseline — Test Metrics\n\n")
        f.write(table_md)
        f.write(f"\n\nTruncated samples: {trunc_count} / {len(test_samples)}\n")
    logger.info("Report table saved to %s", report_table)

    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
