# FUNSD Dataset Statistics

## 1. Data Overview

| Split   | Samples | Total Tokens | Tokens/Page (min) | Tokens/Page (max) | Tokens/Page (mean) | Tokens/Page (median) |
|---------|---------|-------------|-------------------|-------------------|--------------------|----------------------|
| train | 127 | 18668 | 34 | 325 | 147.0 | 139 |
| val | 22 | 3220 | 58 | 282 | 146.4 | 137.5 |
| test | 50 | 8707 | 25 | 433 | 174.1 | 177.0 |

## 2. Bounding Boxes per Page

| Split   | Min | Max | Mean | Median |
|---------|-----|-----|------|--------|
| train | 34 | 325 | 147.0 | 139 |
| val | 58 | 282 | 146.4 | 137.5 |
| test | 25 | 433 | 174.1 | 177.0 |

## 3. Entity Category Distribution

| Split   | QUESTION | ANSWER | HEADER | O (other) |
|---------|----------|--------|--------|-----------|
| train | 6231 | 8183 | 1219 | 3035 |
| val | 1030 | 1336 | 257 | 597 |
| test | 2654 | 3294 | 374 | 2385 |

## 4. BIO Label Distribution

| Split   | B-QUESTION | I-QUESTION | B-ANSWER | I-ANSWER | B-HEADER | I-HEADER | O |
|---------|------------|------------|----------|----------|----------|----------|---|
| train | 2771 | 3460 | 2350 | 5833 | 371 | 848 | 3035 |
| val | 472 | 558 | 363 | 973 | 69 | 188 | 597 |
| test | 1067 | 1587 | 807 | 2487 | 119 | 255 | 2385 |

## 5. Pages Exceeding Max Sequence Length

| Split   | Threshold | Count | Ratio |
|---------|-----------|-------|-------|
| train | 512 | 0 | 0.00% |
| val | 512 | 0 | 0.00% |
| test | 512 | 0 | 0.00% |

## 6. OCR Noise Summary

| Split   | Non-printable Tokens | Non-printable Ratio | High-symbol Examples | Long Token Examples |
|---------|----------------------|--------------------|----------------------|---------------------|
| train | 120 | 0.64% | (s), (s), ($) | - |
| val | 18 | 0.56% | (=, (3., (N, | - |
| test | 57 | 0.65% | (S), "B", S.: | - |
