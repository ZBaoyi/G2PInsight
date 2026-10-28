# G2PInsight

**What it is:** G2PInsight is a command-line pipeline for **genotype-to-phenotype (G2P)** analysis. You provide genotype data (VCF or PLINK) and a phenotype file; the tool prepares a training matrix, fits machine-learning models, explains SNP importance with SHAP, and can score new samples — no coding required.

**What it is for:** breeding and genetics studies where you want to predict traits, compare ML methods, find important genomic regions, and generate publication-ready figures.

**How to use it:** run four stages in order (predict is optional):

```text
preprocess  →  train or train-all  →  visualize  →  predict (optional)
```

Supports **classification** and **regression**, optional **GWAS/LD** SNP filtering in preprocess, and bundled **PLINK** / **GEMMA** binaries.

---

## Install

**Audience:** end users who want to run the pipeline (no coding). Developers: see [Other install options](#other-install-options).

**Need:** Python 3.8–3.13, Linux (recommended) or macOS. PLINK and GEMMA ship with the package. Python 3.14+ is not supported yet: its pandas/numpy wheels need a newer glibc than many HPC systems provide, so `pip install` would fail for those users.

**Get G2PInsight** (recommended — from [PyPI](https://pypi.org/p/G2PInsight)):

```bash
pip install G2PInsight
G2PInsight --version
```

Then go to [Quick Start](#quick-start). Source repository: [github.com/ZBaoyi/G2PInsight](https://github.com/ZBaoyi/G2PInsight).

---

## Other install options

Use these only if the one-liner in [Install](#install) is not enough.

### From a personal conda channel

**Who this is for:** users who prefer conda over pip.

```bash
conda install -c zbaoyi -c conda-forge g2pinsight
G2PInsight --version
```

**Compatibility notes:**

- This package is **noarch**, but conda-forge deps (e.g. `catboost`, `lightgbm`, recent Python builds) often need **glibc ≥ 2.28**. On **CentOS 7 (glibc 2.17)**, `conda install` may fail with `UnsatisfiableError` / `__glibc` — use **pip** ([Install](#install)) or a **git clone** instead.
- Conda run pins match the **Python &lt; 3.13** ranges in `pyproject.toml` (e.g. `xgboost >=1.6,<2.1`). For **Python 3.13**, prefer **pip**.

### From a local clone (editable install)

**Who this is for:** people who already `git clone` the repo, want to read or change the source, need edits without reinstalling, **or** whose system is too old for the conda package above.

**What `-e` means:** editable mode — Python imports the code from this folder. After you edit files under `G2PInsight/`, the `G2PInsight` command uses your changes immediately.

```bash
git clone https://github.com/ZBaoyi/G2PInsight.git
cd G2PInsight
pip install -e .
G2PInsight --version
```

Only want to use the tool (no source edits) and pip works? Prefer `pip install G2PInsight` in [Install](#install) instead.

`bed-reader` and `psutil` are included by default. If `bed-reader` is missing at runtime, preprocess falls back to slower PLINK `--recodeA`.

---

## Table of Contents

- [Install](#install)
- [Other install options](#other-install-options)
- [What you get](#what-you-get)
- [Quick Start](#quick-start)
- [Pipeline](#pipeline)
- [Commands](#commands)
  - [preprocess](#1-preprocess)
  - [train](#2-train)
  - [train-all](#3-train-all)
  - [predict](#4-predict)
  - [visualize](#5-visualize)
- [Output layout](#output-layout)
- [FAQ](#faq)
- [Developer guide](#developer-guide)
- [License](#license)

---

## What you get

| Stage | What it does | Main outputs |
|-------|----------------|--------------|
| **preprocess** | Match samples, QC SNPs, optional GWAS/LD, build training matrix | `*_train_data.txt` (tab-delimited), `*_metadata.json`, phenotype plot |
| **train** | Fit **one** model: tuning, CV, test metrics, SHAP | `{Model}_model.pkl`, metrics, SHAP files |
| **train-all** | Fit **all** applicable models, pick the best on the test set | Best model folder + comparison JSON |
| **visualize** | Genome-wide importance, performance curves, SHAP plots | PNG / HTML under `visualize/` |
| **predict** | Score new samples with a saved model | `*_predictions.tsv` |

**Models:** LightGBM, RandomForest, XGBoost, SVM, CatBoost, Logistic (classification only). **train-all** skips Logistic for regression.

**Important conventions**

- **train** / **train-all** need preprocess `*_metadata.json` (`-j`) — not a raw matrix path alone.
- GWAS/LD runs in **preprocess only**; training does not re-filter SNPs.
- **Large SNP counts:** use preprocess `-f 2`–`4`; for **train-all**, prefer `--parallel_models 1` unless job memory is very large (each parallel worker copies the full matrix).

---

## Quick Start

Example data is **not** shipped in the PyPI or conda package. Get the two files under `data/` first, then run the commands below from the directory that contains that folder.

| File | Content |
|------|---------|
| `data/genotype.vcf` | VCF genotypes (~33k SNPs, ~53 MB, 413 samples) |
| `data/pheno.txt` | Two columns: sample ID + binary trait `Leaf_pubescence` (0/1); optional header OK |

**Recommended — download from the mirror** (works for pip / conda / clone users):

```bash
mkdir -p data
wget -O data/genotype.vcf https://pr-nxy.ynu.edu.cn/G2PInsight/data/genotype.vcf
wget -O data/pheno.txt https://pr-nxy.ynu.edu.cn/G2PInsight/data/pheno.txt
```

Or pull the whole folder:

```bash
wget -r -np -nH --cut-dirs=1 -R "index.html*" https://pr-nxy.ynu.edu.cn/G2PInsight/data/
```

**Alternative — `git clone`:** the same files are in the GitHub repo under `data/` (~53 MB VCF; clone may be slow on some networks).

```bash
# 1. Preprocess  →  test/preprocess/preprocess/preprocess_metadata.json
G2PInsight preprocess --vcf data/genotype.vcf -p data/pheno.txt -o test/preprocess
```
```bash
# 2. Train  →  test/train/train/LightGBM/
G2PInsight train -j test/preprocess/preprocess/preprocess_metadata.json -m LightGBM -o test/train
```
```bash
# 3. Visualize  →  test/plot/visualize/lightgbm_*
G2PInsight visualize \
  -i test/train/train/LightGBM/LightGBM_shap_values.txt \
  -I test/train/train/LightGBM/LightGBM_plotting_data.json \
  -o test/plot/lightgbm
```

**Larger SNP sets** — use preprocess feature selection (`-f 4`), then train conservatively:

```bash
G2PInsight preprocess --vcf data/genotype.vcf -p data/pheno.txt -o test/preprocess -f 4
G2PInsight train-all -j test/preprocess/preprocess/preprocess_metadata.json -o test/train \
  --parallel_models 1 --threads 4
```

---

## Pipeline

```text
  genotype + phenotype
         │
         ▼
   preprocess  ──►  *_train_data.txt + *_metadata.json
         │
         ▼
   train / train-all  ──►  model.pkl, metrics, SHAP, plotting JSON
         │
         ├──► visualize  ──►  PNG / HTML
         └──► predict    ──►  predictions.tsv
```

Use `-o my_project/` as the project folder. Preprocess writes `my_project/preprocess/my_project_metadata.json` — that is the path for **train `-j`**.

---

## Commands

### 1. preprocess

#### What it is

Turns VCF / PLINK genotypes plus a two-column phenotype file into a **sample × SNP training matrix** and a **metadata JSON** that downstream stages require.

#### What you get

| File | Purpose |
|------|---------|
| `{prefix}_train_data.txt` | Tab-delimited matrix (`sample` index, SNP columns, `phenotype`); may be `.txt.gz` if large |
| `{prefix}_metadata.json` | Paths, sample list, SNP count, task type — **required for train** |
| `{prefix}_train_ids.txt` / `{prefix}_test_ids.txt` | Fixed train/test sample IDs |
| `phenotype_distribution_pie.png` | Classification: class proportions |
| `phenotype_distribution_histogram.png` | Regression: trait distribution (after sample matching) |

SNP columns are named `chr_position` (e.g. `1_123456`).

#### How to run

```bash
G2PInsight preprocess (--bfile|--file|--vcf) <genotype> -p <phenotype> -o <output> [options]
```

**Typical run** (bundled example + GWAS→LD):

```bash
G2PInsight preprocess --vcf data/genotype.vcf -p data/pheno.txt -o results/ -f 4
```

**Large genome / PLINK binary** (SNP filtering recommended):

```bash
G2PInsight preprocess --bfile genotype -p data/pheno.txt -o results/ -f 4
```

#### Key options

| Option | Default | What it controls |
|--------|---------|------------------|
| `-f` | `1` | `1` none · `2` GWAS · `3` LD · `4` GWAS→LD |
| `--no-filter-snps` | off | Skip MAF / missingness QC (not for whole-genome) |
| `--train-ids-file` / `--test-ids-file` | — | Fixed train/test IDs (must be paired). GWAS/LD use **train only** |
| `--random_state` | `42` | Seed for default reproducible 80/20 split (IDs sorted first; ignored if ID files given) |

**Sample split:** After MAF/geno QC, preprocess always fixes train/test (custom IDs or sorted-ID 80/20). Feature selection uses train only; test is held out for later evaluation. IDs are written to `*_train_ids.txt` / `*_test_ids.txt` and `sample_split` in metadata.

**Matrix backend (fixed):** `auto` — use `bed-reader` when available, else PLINK `--recodeA`; in-process cache on; chromosome recode serial.

**Inputs:** exactly one of `--bfile`, `--file`, `--vcf`. Phenotype: two columns (sample ID + value); an optional header row is fine (non-numeric values are dropped). See `data/pheno.txt` in the repo.

---

### 2. train

#### What it is

Trains **one** ML model from `*_metadata.json`: load matrix → use preprocess `sample_split` → optional hyperparameter search → K-fold CV on training split → final fit → test metrics → SHAP.

#### What you get

| File | Purpose |
|------|---------|
| `{Model}_model.pkl` | Saved model for **predict** |
| `{Model}_metrics.json` | **Held-out test set** metrics (main result) |
| `{Model}_cv_results.json`, `{Model}_cv_oof.tsv` | CV on **training split** only |
| `{Model}_shap_values.txt` | Genome-wide importance (tab-delimited, `.txt` extension) |
| `{Model}_shap_dependence.tsv` | Per-sample SHAP for dependence plots |
| `{Model}_plotting_data.json` | Index for **visualize -I** |

#### How to run

```bash
G2PInsight train -j <metadata.json> -m <model> -o <output_dir> [options]
```

```bash
# Default (no hyperparameter search)
G2PInsight train -j results/preprocess/results_metadata.json -m LightGBM -o results/

# Large SNP set
G2PInsight train -j results/preprocess/results_metadata.json -m LightGBM -o results/ \
  --threads 4

# Enable hyperparameter search
G2PInsight train -j results/preprocess/results_metadata.json -m LightGBM -o results/ \
  --hyperparameter_search
```

**Evaluation:** Uses the train/test split fixed in preprocess (default 80/20 or custom ID files). Fixed **5-fold** CV and optional tuning use the training split only; `{Model}_metrics.json` reports the held-out test set.

**Flow (default):** 5-fold CV on train split (model defaults) → final fit → SHAP. With `--hyperparameter_search`: hold-out tuning on the train split → CV → final fit → SHAP.

#### Key options

| Option | Default | Notes |
|--------|---------|-------|
| `-m` | — | `LightGBM`, `RandomForest`, `XGBoost`, `SVM`, `CatBoost`, `Logistic` |
| `--hyperparameter_search` | off | Opt-in RandomizedSearchCV; off by default for speed |
| `--threads` | `1` | Per-model CPU threads |
| `--shap_dependence_top` | `50` | Cap SNPs in dependence file when features > 1000 (`0` = all) |
| `--random_state` | `42` | Seed for CV / model fitting (not for train/test split; that comes from preprocess) |

Before loading data, **train** validates preprocess metadata (e.g. rejects a truncated gzip training matrix).

---

### 3. train-all

#### What it is

Trains **every model** for your task on the **same** data split. The parent reads the matrix once; each **worker process** runs the full pipeline for one model. After all finish, the **best model** (by held-out test performance) is kept; other model folders are removed.

#### What you get

| File | Purpose |
|------|---------|
| `{BestModel}/` | Winning model: `.pkl`, metrics, SHAP, plotting files |
| `best_model_info.json` | Which model won and the metric value |
| `model_comparison_report.json` | Success/fail, timings, settings |
| `all_models_cv_results.json` | All models’ CV results (saved before cleanup) |

**Best model rule:** classification → AUC (else accuracy); regression → Pearson *r* on test set.

#### How to run

```bash
G2PInsight train-all -j <metadata.json> -o <output_dir> [options]
```

```bash
# Recommended default
G2PInsight train-all -j results/preprocess/results_metadata.json -o results/ \
  --parallel_models 1 --threads 4

# Millions of SNPs
G2PInsight train-all -j results/preprocess/results_metadata.json -o results/ \
  --parallel_models 1 --threads 4

# Opt-in hyperparameter search
G2PInsight train-all -j results/preprocess/results_metadata.json -o results/ \
  --hyperparameter_search --parallel_models 1 --threads 4
```

Shares **train** options except `-m`. Extra: `--parallel_models` (default `1`). On very wide matrices, high `--parallel_models` needs substantial RAM (each worker copies the full matrix).

---

### 4. predict

#### What it is

Applies a trained `.pkl` model to **new samples**. Aligns features to training SNPs (missing → zero-filled).

#### What you get

`<output_dir>/predict/{Model}_predictions.tsv` — columns `sample`, `prediction` (plus `prob_class_*` for classification).

#### How to run

```bash
G2PInsight predict -i <input> -m <model.pkl> -o <output_dir> [--task_type <type>]
```

| Input (`-i`) | Notes |
|--------------|-------|
| Tab-delimited matrix `.txt` / `.txt.gz` | Same format as preprocess output |
| PLINK binary prefix | `.bed/.bim/.fam` — **training SNPs extracted first** |
| VCF `.vcf` / `.vcf.gz` | Converted internally; **training SNPs extracted first** |

```bash
G2PInsight predict -i new_genotype -m results/train/LightGBM/LightGBM_model.pkl -o results/ --task_type regression
```

Use the same genome build and SNP naming as training (`*_training_features.json`).

---

### 5. visualize

#### What it is

Renders PNG/HTML from training artifacts. Training writes data; **visualize** only plots.

#### Inputs

| Flag | Input | Plots |
|------|--------|--------|
| `-i` | `*_shap_values.txt` (`feature`, `importance_abs`, `effect`) | Genome-wide importance, top-SNP bar; if filename contains `shap` and `*_shap_dependence.tsv` exists → dependence + summary |
| `-I` | `*_plotting_data.json` (+ sibling predictions / CV JSON) | Performance curves, test metrics, CV diagnostics; classification also confusion matrix (+ CV mean bars) |

At least one of `-i` / `-I` is required. `-o PREFIX` is required. Outputs go under `<parent_of_PREFIX>/visualize/`.

#### Output files

**`-i`**

| File | Content |
|------|---------|
| `{prefix}_importance_static.png` | Genome-wide importance (static) |
| `{prefix}_importance_interactive.html` | Genome-wide importance (interactive) |
| `{prefix}_shap_top_snps_bar.png` | Top-SNP bar (non-SHAP name → `*_top_snps_bar.png`) |
| `{prefix}_dependence.png` | SHAP dependence panels |
| `{prefix}_shap_summary.png` | SHAP summary |

**`-I`**

| File | Content |
|------|---------|
| `{prefix}_performance_curves.png` | Regression: pred/true, residuals, hist, Q-Q (also `.pdf` when publication mode); classification: ROC / threshold / PR |
| `{prefix}_confusion_matrix.png` | Classification only |
| `{prefix}_test_metrics.png` | Held-out test metric bars |
| `{prefix}_cv_training_curves.png` | Per-fold CV bars |
| `{prefix}_cv_average_metrics.png` | CV mean bars (classification only) |

#### How to run

```bash
G2PInsight visualize [-i <shap_values.txt>] [-I <plotting_data.json>] -o <prefix> [--top_snps N]
```

```bash
G2PInsight visualize -I results/train/LightGBM/LightGBM_plotting_data.json -o results/plot/perf
G2PInsight visualize -i results/train/LightGBM/LightGBM_shap_values.txt -o results/plot/shap --top_snps 20
```

Training defaults to `--shap_dependence_top 50`; keep visualize `--top_snps` ≤ that cap (or raise the train cap / use `0` for all).

**Code map:** `G2PInsight/bin/visualization.py` — (1) shared style helpers, (2) `EnhancedGenomeVisualizer` for `-i`, (3) SHAP dependence/summary, (4) performance/CV for `-I`. Entry: `main.run_unified_visualization`.

---

## Output layout

```text
results/
├── preprocess/
│   ├── results_train_data.txt[.gz]
│   ├── results_metadata.json       ← train -j
│   ├── results_train_ids.txt
│   ├── results_test_ids.txt
│   └── phenotype_distribution_*.png
├── train/
│   ├── LightGBM/                   ← or only BestModel/ after train-all
│   │   ├── LightGBM_model.pkl
│   │   ├── LightGBM_metrics.json
│   │   ├── LightGBM_shap_values.txt
│   │   ├── LightGBM_plotting_data.json
│   │   └── ...
│   ├── best_model_info.json        ← train-all
│   ├── model_comparison_report.json
│   └── all_models_cv_results.json
├── predict/
│   └── LightGBM_predictions.tsv
└── plot/visualize/
    └── *.png / *.html
```

Temp files under `{output}/tmp/` are removed after successful runs.

---

## FAQ

### train requires metadata — why?

**train** and **train-all** need `*_metadata.json` from preprocess. It records the matrix path, samples, SNP count, and task type.

### Where is the metadata file?

`<output>/preprocess/{prefix}_metadata.json`. Example: `-o results/` → `results/preprocess/results_metadata.json`.

### train-all keeps only one model folder

By design: best model by **test set** metrics. See `best_model_info.json` and `all_models_cv_results.json` for the full comparison.

### train-all fails or workers crash

**Symptom:** `Worker process crashed`, all models `failed`, or OOM under high `--parallel_models`.

**What to do:**

1. **≥ 500k SNPs:** prefer `--parallel_models 1` unless memory is abundant.
2. Re-run preprocess with **`-f 2` or `-f 4`** if too many SNPs were kept.
3. Use `--parallel_models 1 --threads 4` (omit `--hyperparameter_search`).
4. Increase job memory; or test one model: `G2PInsight train -m LightGBM ...`.

### Training or load is very slow

1. Fewer SNPs in preprocess (`-f 2`–`4`).
2. Uncompressed `*_train_data.txt` loads faster than `.gz`.
3. Keep default (no `--hyperparameter_search`).

### SHAP dependence file is huge

Default is `--shap_dependence_top 50`. Use `0` only if you need all SNPs (can be very large). Genome-wide `*_shap_values.txt` is unchanged.

### Preprocess slow or killed

1. Do not use `--no-filter-snps` on whole-genome data; use `-f 2`–`4`.
2. Confirm `bed-reader` is installed (it is a default dependency of `G2PInsight`); without it, preprocess falls back to slower PLINK `--recodeA`.

### Preprocess: `unexpected keyword argument 'fam_location'`

Your env has an old `bed-reader` (0.2.x). Upgrade: `pip install -U 'bed-reader>=1.0'`. Current G2PInsight also accepts the older `fam_filepath` API as a fallback.

### train / train-all: test IDs from metadata missing in matrix

Usually from an older preprocess with **`-f 3` / `-f 4`**: LD used train-only `--keep` and the final `.bed` dropped test samples while metadata still listed them. Fixed in current source — **re-run preprocess**, then train again.

### train-all / CatBoost: `DataFrame` has no attribute `iteritems`

Old CatBoost + **pandas 2.x** (removed `iteritems`). Current G2PInsight passes numpy arrays into CatBoost. Temporary workaround: `pip install -U 'catboost>=1.2,<2'` or `pip install 'pandas>=1.3,<2'`.

### train: XGBoost “Mismatched version … Python package … Shared object”

Usually **conda + pip mixed installs**: e.g. conda left `libxgboost.so` 2.0.x while pip upgraded the Python package to 3.x. Fix in the same env:

```bash
pip uninstall -y xgboost
conda remove -y xgboost 2>/dev/null || true
pip install 'xgboost>=1.6,<2.1'   # Python <3.13; or conda-forge with the same range
```

Prefer one package manager for ML libs; do not `pip install -U xgboost` on top of a conda install.

### predict: constant values or no feature match

1. Same genome build and `chr_pos` naming as training.
2. Check sample IDs match.
3. For VCF/PLINK, training SNPs are extracted automatically — ensure those SNPs exist in the new data.

### visualize: missing `effect` column

Use training-generated `*_shap_values.txt` with `feature`, `importance_abs`, `effect`.

### Which preprocess genotype backend is used?

Preprocess uses **`auto`**: `bed-reader` when the package is installed and a `.bed` is available (default dependency), otherwise PLINK `--recodeA`. There is no separate backend log line.

---

## Developer guide

```text
G2PInsight/
├── main.py
└── bin/
    ├── preprocess.py
    ├── modeltraining.py
    ├── visualization.py
    ├── gemma_gwas.py
    └── plink_ld.py
```

```bash
git clone https://github.com/ZBaoyi/G2PInsight.git
cd G2PInsight
pip install -e .
G2PInsight preprocess -h
```

Bug reports and feature requests: [GitHub Issues](https://github.com/ZBaoyi/G2PInsight/issues).

---

## License

[MIT License](LICENSE)
