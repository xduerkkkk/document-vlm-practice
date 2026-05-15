"""Train an OCR-only token classification baseline on FUNSD.

Uses a BERT-family encoder (default: distilbert-base-uncased) to predict
BIO tags from word-level tokens only.  No layout / image features are used.

Usage:
    python src/train/train_text_only.py \
        --data_dir funsd_experiment/data/prepared \
        --model_name distilbert-base-uncased \
        --output_dir funsd_experiment/outputs/checkpoints/text_only \
        --epochs 5 --batch_size 4 --learning_rate 5e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
import torch
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# Ensure funsd_experiment/src is on the import path.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eval.metrics import compute_metrics_from_tags, format_metrics_table
from models.text_only_bert import build_model_and_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train a text-only BERT baseline on FUNSD")
    p.add_argument("--data_dir", default="funsd_experiment/data/prepared")
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--output_dir", default="funsd_experiment/outputs/checkpoints/text_only")
    p.add_argument("--prediction_dir", default="funsd_experiment/outputs/predictions")
    p.add_argument("--metric_dir", default="funsd_experiment/outputs/metrics")
    p.add_argument("--report_table", default="funsd_experiment/report/tables/text_only_metrics.md")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_json_samples(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_dataset(samples: list[dict]) -> Dataset:
    """Keep only fields needed for text-only training."""
    return Dataset.from_list(
        [
            {"id": s["id"], "tokens": s["tokens"], "labels": s["labels"]}
            for s in samples
        ]
    )


def tokenize_and_align(examples, tokenizer, label2id: dict, max_length: int):
    """Tokenize pre-split words and align BIO labels to subwords.

    Alignment rules:
      - special tokens (CLS, SEP, PAD)  -> label = -100
      - first subword of each word       -> original label id
      - subsequent subwords of same word -> -100
    """
    tok = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    all_labels = []
    for i, labels_per_sample in enumerate(examples["labels"]):
        word_ids = tok.word_ids(batch_index=i)
        aligned = []
        prev = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev:
                aligned.append(label2id[labels_per_sample[wid]])
            else:
                aligned.append(-100)
            prev = wid
        all_labels.append(aligned)

    tok["labels"] = all_labels
    return tok


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


def _count_truncated(tokenized_ds, orig_word_counts):
    """Count how many samples got truncated during tokenization.

    We compare the number of non-(-100) labels per sample against the
    original word count.
    """
    n_trunc = 0
    for i, label_seq in enumerate(tokenized_ds["labels"]):
        kept = sum(1 for l in label_seq if l != -100)
        if kept < orig_word_counts[i]:
            n_trunc += 1
    return n_trunc


# ---------------------------------------------------------------------------
# compute_metrics for Trainer
# ---------------------------------------------------------------------------

def _make_compute_metrics(id2label):
    """Return a ``compute_metrics`` callable for HuggingFace Trainer."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        y_true_tags, y_pred_tags = _gather_tags(predictions, labels, id2label)
        return compute_metrics_from_tags(y_true_tags, y_pred_tags)

    return compute_metrics


def _gather_tags(pred_ids, label_ids, id2label):
    """Convert model outputs (subword-level) to per-sample tag lists."""
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

def _build_prediction_output(raw_ds, pred_ids, label_ids, id2label):
    """Build per-sample prediction dicts for JSON export.

    Uses the raw dataset for ``id`` and original ``tokens``, and aligns
    subword predictions back to word-level using the ``-100`` mask in labels.
    """
    samples = []
    for i in range(len(raw_ds)):
        rec = raw_ds[i]
        gold_tags, pred_tags = [], []
        for p, l in zip(pred_ids[i], label_ids[i]):
            if l != -100:
                gold_tags.append(id2label[l])
                pred_tags.append(id2label[p])

        n_kept = len(gold_tags)
        samples.append(
            {
                "id": rec["id"],
                "tokens": rec["tokens"][:n_kept],
                "gold_labels": gold_tags,
                "pred_labels": pred_tags,
            }
        )
    return samples


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

    # ---- load data ----
    logger.info("Loading data from %s", data_dir)
    train_samples = _load_json_samples(data_dir / "train.json")
    val_samples = _load_json_samples(data_dir / "val.json")
    test_samples = _load_json_samples(data_dir / "test.json")

    train_raw = _to_dataset(train_samples)
    val_raw = _to_dataset(val_samples)
    test_raw = _to_dataset(test_samples)

    # ---- model & tokenizer ----
    logger.info("Building model: %s", args.model_name)
    model, tokenizer, label2id, id2label = build_model_and_tokenizer(args.model_name)

    # ---- tokenize ----
    logger.info("Tokenizing (max_length=%d) ...", args.max_length)
    tok_fn = lambda ex: tokenize_and_align(ex, tokenizer, label2id, args.max_length)

    train_ds = train_raw.map(tok_fn, batched=True, batch_size=64)
    val_ds = val_raw.map(tok_fn, batched=True, batch_size=64)
    test_ds = test_raw.map(tok_fn, batched=True, batch_size=64)

    # ---- truncation statistics (before removing columns) ----
    orig_word_counts = [len(s["tokens"]) for s in test_samples]
    trunc_count = _count_truncated(test_ds, orig_word_counts)
    logger.info("Truncated test samples: %d / %d", trunc_count, len(test_samples))

    # Strip columns not consumed by the model / collator.
    keep_cols = {"input_ids", "attention_mask", "labels"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])

    # ---- collator ----
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # ---- training args ----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
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
    # 1. model & tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 2. metrics JSON
    metrics_path = metric_dir / "text_only_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_to_native(full_metrics), f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", metrics_path)

    # 3. predictions JSON
    predictions = _build_prediction_output(test_raw, test_pred_ids, test_label_ids, id2label)
    preds_path = prediction_dir / "text_only_test_predictions.json"
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info("Predictions saved to %s", preds_path)

    # 4. Markdown table
    table_md = format_metrics_table(full_metrics)
    with open(report_table, "w", encoding="utf-8") as f:
        f.write("# Text-Only Baseline — Test Metrics\n\n")
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
