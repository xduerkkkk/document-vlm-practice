"""Train a Text+Layout token classification baseline on FUNSD.

Uses a BERT-family encoder for text and a small MLP for bounding-box
coordinates.  The two representations are concatenated and fed into a
linear BIO classification head.

Usage:
    python src/train/train_text_layout.py \
        --data_dir funsd_experiment/data/prepared \
        --model_name distilbert-base-uncased \
        --output_dir funsd_experiment/outputs/checkpoints/text_layout \
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
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorMixin

# Ensure funsd_experiment/src is on the import path.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.text_only_bert import build_label_maps
from models.text_layout_bert import TextLayoutTokenClassifier
from eval.metrics import compute_metrics_from_tags, format_metrics_table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train a Text+Layout baseline on FUNSD"
    )
    p.add_argument("--data_dir", default="funsd_experiment/data/prepared")
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--output_dir", default="funsd_experiment/outputs/checkpoints/text_layout")
    p.add_argument("--prediction_dir", default="funsd_experiment/outputs/predictions")
    p.add_argument("--metric_dir", default="funsd_experiment/outputs/metrics")
    p.add_argument("--report_table", default="funsd_experiment/report/tables/text_layout_metrics.md")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--bbox_hidden_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_json_samples(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_dataset(samples: list[dict]) -> Dataset:
    """Keep fields needed for text+layout training."""
    return Dataset.from_list(
        [
            {
                "id": s["id"],
                "tokens": s["tokens"],
                "labels": s["labels"],
                "bboxes_norm": s["bboxes_norm"],
            }
            for s in samples
        ]
    )


def tokenize_and_align_with_bbox(examples, tokenizer, label2id, max_length):
    """Tokenize pre-split words and align both labels and bboxes to subwords.

    Label alignment (same as text-only):
      - special tokens           -> -100
      - first subword of a word  -> original label id
      - subsequent subwords      -> -100

    Bbox alignment:
      - special tokens / padding -> [0, 0, 0, 0]
      - every subword of a word  -> the word's bboxes_norm coordinate
    """
    tok = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    all_labels = []
    all_bboxes = []
    for i in range(len(examples["tokens"])):
        word_ids = tok.word_ids(batch_index=i)
        labels_per_sample = examples["labels"][i]
        bboxes_per_sample = examples["bboxes_norm"][i]

        aligned_labels = []
        aligned_bboxes = []
        prev = None
        for wid in word_ids:
            if wid is None:
                # special token ([CLS], [SEP]) or padding
                aligned_labels.append(-100)
                aligned_bboxes.append([0, 0, 0, 0])
            else:
                if wid != prev:
                    aligned_labels.append(label2id[labels_per_sample[wid]])
                else:
                    aligned_labels.append(-100)
                aligned_bboxes.append(bboxes_per_sample[wid])
            prev = wid

        all_labels.append(aligned_labels)
        all_bboxes.append(aligned_bboxes)

    tok["labels"] = all_labels
    tok["bbox"] = all_bboxes
    return tok


def _count_truncated(tokenized_ds, orig_word_counts):
    n_trunc = 0
    for i, label_seq in enumerate(tokenized_ds["labels"]):
        kept = sum(1 for l in label_seq if l != -100)
        if kept < orig_word_counts[i]:
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
# data collator  (pads input_ids, attention_mask, labels, and bbox)
# ---------------------------------------------------------------------------

class LayoutDataCollator(DataCollatorMixin):
    """Data collator that pads *bbox* alongside the usual token-classification fields.

    Returns a dict ready for ``TextLayoutTokenClassifier.forward``.
    """

    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def torch_call(self, features):
        # Separate bbox before handing the rest to the tokenizer pad logic.
        bboxes = [f.pop("bbox") for f in features]

        batch = self.tokenizer.pad(
            features,
            return_tensors="pt",
        )
        # tokenizer.pad uses "label" (singular) for the labels key.
        if "label" in batch:
            batch["labels"] = batch.pop("label")

        # Pad bboxes to the length of input_ids.
        max_len = batch["input_ids"].shape[1]
        padded_bbox = []
        for b in bboxes:
            pad_len = max_len - len(b)
            padded_bbox.append(b + [[0, 0, 0, 0]] * pad_len)
        batch["bbox"] = torch.tensor(padded_bbox, dtype=torch.float)

        return batch


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
    """Build per-sample prediction dicts.

    Includes id, tokens, bboxes_norm, gold_labels, and pred_labels.
    Tokens and bboxes are truncated to the number of kept tokens after tokenization.
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
                "bboxes_norm": rec["bboxes_norm"][:n_kept],
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

    # ---- label maps ----
    label2id, id2label = build_label_maps()
    num_labels = len(label2id)

    # ---- model & tokenizer ----
    logger.info("Building model: %s (bbox_hidden_size=%d)", args.model_name, args.bbox_hidden_size)
    model = TextLayoutTokenClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        bbox_hidden_size=args.bbox_hidden_size,
        label2id=label2id,
        id2label=id2label,
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ---- tokenize ----
    logger.info("Tokenizing (max_length=%d) ...", args.max_length)
    tok_fn = lambda ex: tokenize_and_align_with_bbox(ex, tokenizer, label2id, args.max_length)

    train_ds = train_raw.map(tok_fn, batched=True, batch_size=64)
    val_ds = val_raw.map(tok_fn, batched=True, batch_size=64)
    test_ds = test_raw.map(tok_fn, batched=True, batch_size=64)

    # ---- truncation statistics ----
    orig_word_counts = [len(s["tokens"]) for s in test_samples]
    trunc_count = _count_truncated(test_ds, orig_word_counts)
    logger.info("Truncated test samples: %d / %d", trunc_count, len(test_samples))

    # Strip columns not consumed by the model / collator.
    keep_cols = {"input_ids", "attention_mask", "labels", "bbox"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])

    # ---- collator ----
    data_collator = LayoutDataCollator(tokenizer=tokenizer)

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
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 2. metrics JSON
    metrics_path = metric_dir / "text_layout_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_to_native(full_metrics), f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", metrics_path)

    # 3. predictions JSON
    predictions = _build_prediction_output(test_raw, test_pred_ids, test_label_ids, id2label)
    preds_path = prediction_dir / "text_layout_test_predictions.json"
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    logger.info("Predictions saved to %s", preds_path)

    # 4. Markdown table
    table_md = format_metrics_table(full_metrics)
    with open(report_table, "w", encoding="utf-8") as f:
        f.write("# Text+Layout Baseline — Test Metrics\n\n")
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
