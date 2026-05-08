# G2PInsight Genomic Analysis Tool

G2PInsight is a command-line toolkit for genotype-to-phenotype association analysis. It provides an end-to-end workflow covering data preprocessing, model training, prediction, and visualization, with optional GWAS/LD-based feature selection during preprocessing.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
  - [1. preprocess (data preprocessing)](#1-preprocess-data-preprocessing)
  - [2. train (single-model training)](#2-train-single-model-training)
  - [3. train-all (all-model training)](#3-train-all-all-model-training)
  - [4. predict (model inference)](#4-predict-model-inference)
  - [5. visualize (result visualization)](#5-visualize-result-visualization)
- [Output Files](#output-files)
- [FAQ](#faq)
- [Developer Guide](#developer-guide)
- [License](#license)

---

## Project Overview

G2PInsight standardizes the following workflow:

1. Align and clean genotype and phenotype data.
2. Optionally apply GWAS/LD-based feature selection during preprocessing.
3. Train classification or regression models.
4. Export model metrics, feature importance, and plotting assets.
5. Run prediction on new samples using trained models.

This tool is suitable for bioinformatics and agricultural genomics applications where reproducible GWAS-oriented ML workflows are required.

---

## Core Features

### 1) Data Preprocessing
- Supported genotype inputs: `VCF (.vcf/.vcf.gz)`, `PLINK binary (.bed/.bim/.fam)`, `PLINK text (.ped/.map)`.
- Automatic sample matching and phenotype cleaning (missing/abnormal/non-numeric handling).
- Automatic task-type inference (`classification` / `regression`).
- Optional SNP quality filtering (`MAF`, `GENO`).
- Feature-selection modes: no selection / GWAS / LD / GWAS+LD.

### 2) Model Training
- Supported models: `LightGBM`, `RandomForest`, `XGBoost`, `SVM`, `CatBoost`, `Logistic`.
- Randomized hyperparameter search + cross-validation.
- Saves model artifacts, metrics, feature importance, SHAP outputs, and plotting data.
- `train` and `train-all` require preprocess-generated `*_metadata.json` as input.

### 3) Model Inference
- Input support: training-matrix format (`.txt/.txt.gz`) or VCF (temporary conversion is handled automatically).
- Prediction feature alignment is enforced against training features.

### 4) Visualization
- Two visualization input modes:
  - Feature-importance file (genome-wide scatter plot).
  - `plotting_data.npz` (performance and CV training curves).
- Outputs static PNG and interactive HTML (for feature-importance plotting).

---

## System Requirements

- Python: `3.8` - `3.12`
- OS: Linux / macOS (recommended)
- Shell: Bash

Check Python version:

```bash
python --version
# or
python3 --version
```

---

## Installation

### Option 1: Install with project script (recommended on Linux/macOS)

```bash
chmod +x tools.sh
./tools.sh
```

### Option 2: Install from wheel

```bash
pip install G2PInsight-1.0.0-py3-none-any.whl
```

### Option 3: Install from source

```bash
pip install .
```

### Verify installation

```bash
G2PInsight --version
G2PInsight -h
```

---

## Quick Start

```bash
# 1) Preprocess
G2PInsight preprocess -g genotype.vcf -p phenotype.txt -o preprocessed/

# 2) Train (metadata-driven)
G2PInsight train -j preprocessed/preprocess/preprocessed_metadata.json -m LightGBM -o results/

# 3) Visualize feature importance
G2PInsight visualize -i results/train/LightGBM/LightGBM_feature_importance.txt -o result_plot
```

> Note: `train` and `train-all` must use preprocess-generated `*_metadata.json`.

---

## Command Reference

## 1. preprocess (data preprocessing)

Purpose: convert genotype + phenotype inputs into model-ready training matrix, with optional GWAS/LD feature selection.

### Usage

```bash
G2PInsight preprocess \
  -g <genotype_input> \
  -p <phenotype.txt> \
  -o <output_path> \
  [-f <1|2|3|4>] \
  [--gwas_pvalue <float>] \
  [--ld-config "<window_kb>,<window_variants>,<r2_threshold>"] \
  [--no-filter-snps]
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `-g, --genotype` | Yes | - | Genotype input path (VCF/PLINK) |
| `-p, --phenotype` | Yes | - | Phenotype file path (at least two columns: sample, phenotype) |
| `-o, --output` | Yes | - | Output directory or output prefix |
| `-f, --feature_selection_mode` | No | `1` | 1=no selection, 2=GWAS, 3=LD, 4=GWAS+LD |
| `--gwas_pvalue` | No | `0.01` | GWAS significance threshold (effective for mode 2/4) |
| `--ld-config` | No | `"50,5,0.2"` | LD config: window_kb, window_variants, r² threshold (effective for mode 3/4) |
| `--no-filter-snps` | No | `False` | Disable SNP quality filtering |

### Example

```bash
G2PInsight preprocess -g data.vcf -p pheno.txt -o out/ -f 1
G2PInsight preprocess -g data.vcf -p pheno.txt -o out/ -f 4 --gwas_pvalue 0.01 --ld-config "50,5,0.2"
```

---

## 2. train (single-model training)

Purpose: train one selected model and export model artifacts, metrics, and plotting data.

### Usage

```bash
G2PInsight train \
  -j <preprocess_metadata.json> \
  -m <LightGBM|RandomForest|XGBoost|SVM|CatBoost|Logistic> \
  -o <output_dir> \
  [--task_type <classification|regression>] \
  [--n_folds <int>] \
  [--random_state <int>] \
  [--feature_importance]
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `-j, --json` | Yes | - | Preprocess-generated `*_metadata.json` |
| `-m, --model` | Yes | - | Model name |
| `-o, --output_dir` | Yes | - | Output directory |
| `--task_type` | No | Auto | Optional explicit task type |
| `--n_folds` | No | `5` | Number of CV folds |
| `--random_state` | No | `42` | Random seed |
| `--feature_importance` | No | `False` | Trigger feature-importance output flow |

### Example

```bash
G2PInsight train -j out/preprocess/out_metadata.json -m LightGBM -o results/
```

---

## 3. train-all (all-model training)

Purpose: train all supported models in parallel and produce comparison outputs.

### Usage

```bash
G2PInsight train-all \
  -j <preprocess_metadata.json> \
  -o <output_dir> \
  [--task_type <classification|regression>] \
  [--n_folds <int>] \
  [--random_state <int>] \
  [--feature_importance]
```

### Important behavior

Current implementation keeps only the best-performing model directory after all-model training and removes the others. It also exports `best_model_info.json`.

### Example

```bash
G2PInsight train-all -j out/preprocess/out_metadata.json -o results/
```

---

## 4. predict (model inference)

Purpose: predict phenotypes using a trained `.pkl` model.

### Usage

```bash
G2PInsight predict \
  -i <input_data.txt|input_data.vcf|input_data.vcf.gz> \
  -m <model.pkl> \
  -o <output_dir> \
  [--task_type <classification|regression>]
```

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `-i, --input` | Yes | - | Prediction input (training matrix or VCF) |
| `-m, --model` | Yes | - | Path to trained model file (`.pkl`) |
| `-o, --output_dir` | Yes | - | Output directory (used for temp conversion when input is VCF) |
| `--task_type` | No | - | Optional task type |

### Output location

Prediction results are written to the model directory:

`{model_dir}/{model_type}_predictions.tsv`

### Example

```bash
G2PInsight predict -i new_data.txt -m results/train/LightGBM/LightGBM_model.pkl -o pred/
```

---

## 5. visualize (result visualization)

Purpose: generate feature-importance plots or model-performance plots.

### Usage

```bash
G2PInsight visualize \
  [-i <feature_importance.txt>] \
  [-I <plotting_data.npz>] \
  -o <output_prefix>
```

### Parameters

| Parameter | Required | Description |
|---|---|---|
| `-i, --importance` | No | Feature-importance file |
| `-I, --indicator` | No | `plotting_data.npz` file from training outputs |
| `-o, --output` | Yes | Output prefix |

### Feature-importance format requirements

Recommended input: `<Model>_feature_importance.txt` generated by training.

Required columns:
1. `feature` (e.g., `1_12345` or `chr1_12345`)
2. `importance_abs` (or `importance`)
3. `effect` (`1` or `-1`)

### Example

```bash
G2PInsight visualize -i results/train/LightGBM/LightGBM_feature_importance.txt -o plot
G2PInsight visualize -I results/train/LightGBM/LightGBM_plotting_data.npz -o plot
```

---

## Output Files

### preprocess
Typical location: `<output>/preprocess/`

- `<prefix>_train_data.txt`
- `<prefix>_metadata.json`
- phenotype distribution plot(s), depending on task type

### train
Typical location: `<output>/train/<Model>/`

- `<Model>_model.pkl`
- `<Model>_metrics.json`
- `<Model>_cv_results.json`
- `<Model>_training_features.json`
- `<Model>_feature_importance.txt`
- `<Model>_shap_values.txt`
- `<Model>_plotting_data.npz`

### train-all
Typical location: `<output>/train/`

- best model directory (other model directories may be removed by current implementation)
- `best_model_info.json`
- `model_comparison_report.json`

### predict
Typical location: model directory

- `<Model>_predictions.tsv`

### visualize
Typical location: `<output_parent>/visualize/`

- `<prefix>_importance_static.png`
- `<prefix>_importance_interactive.html`
- `<prefix>_performance_curves.png`
- `<prefix>_cv_training_curves.png`

---

## FAQ

### 1) `train` requires metadata input

`train` and `train-all` require preprocess-generated `*_metadata.json` via `-j`.

### 2) Cannot find preprocess outputs

Check `<output>/preprocess/` for `<prefix>_train_data.txt` and `<prefix>_metadata.json`.

### 3) `visualize` complains about missing `effect`

The input file does not meet the required 3-column schema. Use training-generated `<Model>_feature_importance.txt`.

### 4) Why does `train-all` keep only one model directory?

This is the current behavior: it selects the best model and removes the rest.

### 5) Why are prediction results not under `-o`?

Prediction outputs are saved in the model directory by current implementation. `-o` is mainly used for temporary conversion workflow management.

---

## Developer Guide

### Project Structure

```text
G2PInsight/
├── G2PInsight/
│   ├── main.py
│   └── bin/
│       ├── preprocess.py
│       ├── modeltraining.py
│       ├── gemma_gwas.py
│       ├── plink_ld.py
│       ├── visualization.py
│       └── font_utils.py
├── pyproject.toml
├── setup.py
└── README.md
```

### Local Development Setup

```bash
git clone <your-repo-url>
cd G2PInsight
pip install -e .
```

### Style Recommendations

- Follow PEP 8.
- Document parameters and return values for new public interfaces.
- Keep README synchronized with CLI behavior whenever arguments or outputs change.

---

## License

This project is distributed under the MIT License. See `LICENSE` for details.
