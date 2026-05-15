# FUNSD Experiment Log

*Last updated: 2026-05-15 20:54*

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project structure initialisation | Complete |
| 2 | FUNSD data reader, prepare, preprocess & stats | Complete |
| 3 | OCR-only baseline | Complete |
| 4 | Text + Layout baseline | Complete |
| 5 | LayoutLMv3 baseline | Complete |
| 6 | Unified evaluation & error analysis | Complete |

## Experimental Setup

- **Dataset**: FUNSD prepared (train=127, val=22, test=50)
- **Labels**: BIO tagging (O, B/I-QUESTION, B/I-ANSWER, B/I-HEADER)
- **Evaluation**: token accuracy, macro F1, entity-level P/R/F1 (seqeval)

### Model Configurations

#### OCR-Only Baseline
- **Model**: distilbert-base-uncased
- **Input**: tokens
- **Hyper-params**: epochs=5, batch_size=4, lr=5e-05, max_length=512, seed=42
- **Status**: **Metrics available**

#### Text + Layout Baseline
- **Model**: distilbert-base-uncased + bbox MLP
- **Input**: tokens + bboxes_norm
- **Hyper-params**: epochs=5, batch_size=4, lr=5e-05, max_length=512, seed=42
- **Status**: Metrics pending (TODO)

#### LayoutLMv3 Baseline
- **Model**: microsoft/layoutlmv3-base
- **Input**: tokens + bboxes_norm + image
- **Hyper-params**: epochs=5, batch_size=2, lr=5e-05, max_length=512, seed=42
- **Status**: **Metrics available**

## Results

### Overall Metrics

| Metric | OCR-Only Baseline | LayoutLMv3 Baseline |
|--------|--------|--------|
| Token Accuracy | 0.4303 | 0.6679 |
| Token Macro F1 | 0.3297 | 0.5971 |
| Entity Precision | 0.2900 | 0.5360 |
| Entity Recall | 0.3045 | 0.5745 |
| Entity F1 | 0.2971 | 0.5546 |

### Per-Class Entity F1

| Category | OCR-Only Baseline | LayoutLMv3 Baseline |
|----------|----------|----------|
| QUESTION | 0.3815 | 0.6010 |
| ANSWER | 0.2162 | 0.5514 |
| HEADER | 0.0828 | 0.2609 |

*Metrics sourced from `funsd_experiment/outputs/metrics/*_test_metrics.json`*
