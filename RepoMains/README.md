# FedOmics

FedOmics is a reproducible federated-learning pipeline for multi-center omics data. It supports both a **generic, disease-agnostic simulation workflow** and a **real TCGA PRAD workflow** built around Gleason-based prostate cancer stratification.

The repository is designed to answer two practical questions:

1. **Can a federated pipeline learn structured omics phenotypes without sharing raw center data?**
2. **When does a deep model hold a meaningful advantage over logistic regression?**

To answer the second question, FedOmics includes an **interaction-sensitivity ablation experiment** that compares logistic regression and deep learning under linear, interaction-driven, and mixed signal-generation settings.

---

## Core capabilities

- Generic coexpression-aware simulation for any disease or phenotype
- Optional PRAD-informed simulation using TCGA-derived covariance structure
- Real TCGA PRAD download, expression matrix construction, and Gleason label extraction
- Plug-and-play deep-learning backend selection: **PyTorch** or **TensorFlow**
- Logistic-regression baseline for every training run
- Optional train-only synthetic augmentation for small real cohorts
- QC plots for simulated structure and model predictions
- Interaction-sensitivity ablation experiment to justify when deep learning is worth the added complexity

---

## Repository layout

```text
FedOmics/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── config.yaml
├── scripts/
│   ├── run_pipeline.py
│   ├── run_interaction_ablation.py
│   ├── clean_pipeline.py
│   ├── generate_sim_data.py
│   ├── download_tcga.py
│   ├── preprocess_data.py
│   ├── feature_selection.py
│   ├── train_federated.py
│   ├── plot_sim_qc.py
│   ├── plot_predictions.py
│   ├── qc_dataset.py
│   └── report_run.py
├── src/
│   ├── model.py
│   ├── federated.py
│   ├── utils.py
│   └── models/
│       ├── model_pytorch.py
│       └── model_tensorflow.py
└── data/
    ├── demo_dataset/
    ├── processed/
    └── raw/
        └── tcga_prad/
```

---

## Installation

Recommended: **Python 3.10+**

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to use the TensorFlow backend, make sure TensorFlow is installed in your environment. If TensorFlow is unavailable, switch `model_backend` to `pytorch` in `configs/config.yaml` or pass `--model-backend pytorch` at runtime.

---

## All main pipeline run combinations

### 1) Generic simulation, clean rerun
Best starting point for a new user.

```bash
python -m scripts.run_pipeline --clean --mode sim
```

What it does:
1. cleans prior outputs
2. generates synthetic center datasets
3. creates simulation QC plots
4. preprocesses train/validation splits
5. performs feature selection
6. trains the selected deep-learning backend plus logistic regression baseline
7. writes metrics and prediction QC plots

---

### 2) Generic simulation, explicit signal mode
Choose the simulated signal structure.

```bash
python -m scripts.run_pipeline --clean --mode sim --sim-signal-mode linear
python -m scripts.run_pipeline --clean --mode sim --sim-signal-mode interaction
python -m scripts.run_pipeline --clean --mode sim --sim-signal-mode mixed
```

Use this when you want to study whether a simpler linear model is sufficient or whether nonlinear interactions justify a deep model.

---

### 3) Generic simulation, choose backend

```bash
python -m scripts.run_pipeline --clean --mode sim --model-backend pytorch
python -m scripts.run_pipeline --clean --mode sim --model-backend tensorflow
```

The deep-learning backend is plug-and-play. The pipeline always reports a logistic-regression baseline alongside the chosen deep-learning backend.

---

### 4) TCGA PRAD mode, API download

```bash
python -m scripts.run_pipeline --clean --mode tcga --download-mode api
```

Use this when you want the simplest real-data run and do not want to install `gdc-client`.

---

### 5) TCGA PRAD mode, GDC client download

```bash
python -m scripts.run_pipeline --clean --mode tcga --download-mode client
```

Use this when you already installed the official GDC Data Transfer Tool.

---

### 6) TCGA PRAD mode with explicit deep-learning backend

```bash
python -m scripts.run_pipeline --clean --mode tcga --download-mode api --model-backend pytorch
python -m scripts.run_pipeline --clean --mode tcga --download-mode api --model-backend tensorflow
```

---

### 7) Interaction-sensitivity ablation experiment

```bash
python scripts/run_interaction_ablation.py
```

This experiment runs the simulator in three modes:
- `linear`
- `interaction`
- `mixed`

and compares:
- the selected deep-learning backend
- centralized logistic regression

Outputs are written to:

```text
outputs/ablation/
```

Typical files:
- `metrics_linear.json`
- `metrics_interaction.json`
- `metrics_mixed.json`
- `interaction_ablation_summary.json`
- `interaction_ablation_summary.md`

This is the recommended experiment to justify *why* a deep model is worth using over logistic regression.

---

## What the simulator is doing

### Generic simulation mode
The default simulator is **generic** and **not PRAD-specific**. It creates:

- generic genes (`GENE_0001`, `GENE_0002`, ...)
- latent coexpression modules
- sample-level heterogeneity
- center-specific batch effects
- overlapping phenotype distributions

### Signal-generation modes

#### Linear mode
Phenotype depends on additive module effects.

Expected result:
- logistic regression should be competitive
- deep learning may offer little advantage

#### Interaction mode
Phenotype depends on nonlinear combinations of latent modules.

Expected result:
- deep learning should show a clearer advantage over logistic regression

#### Mixed mode
Phenotype depends on both additive and interaction-driven signal.

Expected result:
- closer to a biologically plausible middle ground
- deep learning may show a modest but meaningful edge

This experiment is the mathematical and biological justification for choosing a deep model when the phenotype is driven by multi-gene interactions rather than only additive effects.

---

## Optional PRAD-informed simulation

If you set:

```yaml
sim_generator_mode: tcga_matched
```

and these files exist:

```text
data/raw/tcga_prad/expression_matrix.csv
data/raw/tcga_prad/clinical.tsv
```

the simulator can learn:
- gene expression scale
- variance structure
- covariance / coexpression structure

and generate **PRAD-informed synthetic data**.

If those files are not present, and fallback is enabled, it falls back to generic simulation.

---

## TCGA PRAD mode in detail

### Required clinical file
Place your extracted clinical TSV here:

```text
data/raw/tcga_prad/clinical.tsv
```

### What TCGA mode does
TCGA mode:
1. downloads or reads TCGA PRAD RNA-seq files
2. maps files to TCGA cases
3. builds an expression matrix
4. extracts labels from clinical metadata
5. creates center datasets from real samples
6. runs training and evaluation with small-cohort safeguards

### Gleason-related columns
The pipeline prefers columns such as:
- `diagnoses.gleason_score`
- `diagnoses.gleason_grade_group`

### Small real-cohort safeguards
If the real labeled PRAD cohort is very small, the pipeline may:
- reduce the number of centers adaptively
- skip augmentation
- switch to repeated CV on a single real cohort
- perform feature selection inside each CV split using training data only

This is intentional and makes the evaluation more scientifically credible.

---

## Deep-learning backend selection

FedOmics now supports two deep-learning implementations with matched architectures:

### PyTorch
Defined in:
```text
src/models/model_pytorch.py
```

### TensorFlow
Defined in:
```text
src/models/model_tensorflow.py
```

The selected backend is controlled either through:

### Config file
```yaml
model_backend: pytorch
```

or runtime override:

```bash
python -m scripts.run_pipeline --clean --mode sim --model-backend tensorflow
```

The logistic-regression baseline is always included in results for comparison.

---

## Why this backend choice matters biologically and mathematically

Logistic regression assumes the phenotype is largely a **linear additive function** of gene expression. That is often a good baseline for transcriptomic tasks.

The deep model becomes more justifiable when the phenotype is driven by:
- gene-gene interactions
- coactivated pathways
- nonlinear combinations of latent biological programs

That is exactly why the repository includes the interaction-sensitivity ablation.

If logistic regression performs just as well in the **linear** setting, that is a useful result. If the deep model pulls ahead in the **interaction** setting, that is your evidence that a nonlinear model is capturing signal that a linear model cannot.

---

## Train-only augmentation for small PRAD cohorts

This option still exists and is controlled in `configs/config.yaml`:

```yaml
tcga_train_aug_enabled: false
tcga_train_aug_ratio: 0.5
tcga_train_aug_balance_mode: conservative
tcga_disable_augmentation_below_n: 30
```

Important design rule:
- synthetic rows may augment **training only**
- validation should remain **real-only**

This is a core design principle for trustworthy evaluation.

---

## Output files

After a standard run, inspect:

```text
data/processed/
```

Typical files:
- `metrics.json`
- `predictions.csv`
- `roc_points.csv`
- `threshold.json`
- `selected_genes.csv`
- `chi2_scores.csv`
- `selected_genes_cv.csv` (for single-cohort repeated CV mode)

QC outputs are written to:

```text
data/processed/qc/
```

Typical QC plots:
- `pca_by_label.png`
- `pca_by_center.png`
- `gene_correlation_heatmap.png`
- `prediction_probability_histogram.png`
- backend-specific prediction histograms

---

## Key configuration parameters

Main config file:

```text
configs/config.yaml
```

Important entries include:

```yaml
model_backend
learning_rate
epochs
batch_size
chi2_top_k
hidden_dim_1
hidden_dim_2

tcga_cv_top_k
tcga_single_cohort_cv_splits
tcga_single_cohort_cv_test_size

tcga_train_aug_enabled
tcga_train_aug_ratio
tcga_train_aug_balance_mode

action # (informal grouping below)
sim_generator_mode
sim_signal_mode
sim_total_genes
sim_num_modules
sim_num_signal_modules
sim_module_size
sim_signal_strength
sim_interaction_strength
sim_noise_scale
sim_center_shift_scale
sim_label_noise
sim_global_shift_scale
```

---

## Recommended user paths

### New user, generic demo
```bash
python -m scripts.run_pipeline --clean --mode sim
```

### New user who wants to compare linear vs nonlinear signal
```bash
python scripts/run_interaction_ablation.py
```

### PRAD-specific real-data example
```bash
python -m scripts.run_pipeline --clean --mode tcga --download-mode api
```

### TensorFlow user
```bash
python -m scripts.run_pipeline --clean --mode sim --model-backend tensorflow
```

### PyTorch user
```bash
python -m scripts.run_pipeline --clean --mode sim --model-backend pytorch
```

---

## MIT License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## Summary

FedOmics now supports:
- **generic simulation mode** for reusable federated omics experiments
- **interaction-sensitivity ablation** to test when deep learning is mathematically justified
- **plug-and-play PyTorch and TensorFlow backends**
- **real TCGA PRAD mode** for a clinically grounded example
- **train-only augmentation** of small real cohorts when needed
- **QC plotting** for both simulated structure and prediction behavior

For most new users, the best starting point is:

```bash
python -m scripts.run_pipeline --clean --mode sim
```

If your goal is to justify deep learning over logistic regression, the best next step is:

```bash
python scripts/run_interaction_ablation.py
```
