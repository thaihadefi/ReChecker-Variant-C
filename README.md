# ReChecker Variant C

ReChecker Variant C is a deep learning pipeline for detecting reentrancy vulnerabilities in Solidity smart contract gadgets. It enhances the ReChecker baseline by incorporating a structure-aware input representation: FastText subword embeddings, an explicit `[SEP]` boundary marker, learned segment embeddings (W vs C regions), length-aware sequence budgeting, and leakage-safe cross-validation.

## Core Concepts

A smart contract reentrancy gadget is split into two semantic regions:
- **W-function ($W$)**: The target function containing external calls or suspected reentrancy points.
- **C-function ($C$)**: Surrounding contract context (state variables, modifiers, auxiliary functions).

### Representation & Architecture

1. **FastText Embeddings**: Captures Solidity identifier subwords and out-of-vocabulary tokens.
2. **`[SEP]` Boundary Token**: Explicitly marks the boundary: $W \oplus [\text{SEP}] \oplus C$.
3. **Segment Embeddings**: Learned vectors indicating token region ($W$, $[\text{SEP}]$, $C$, $[\text{PAD}]$).
4. **Length-Aware Budgeting**: Allocates dedicated token capacity to both $W$ and $C$ (e.g. 60% $W$, 40% $C$) instead of naive prefix truncation.
5. **Masked BiLSTM + Additive Attention**: Masking ensures padding tokens never receive attention weights or distort recurrent states.
6. **Leakage-Safe Evaluation**: `StratifiedGroupKFold` clustering by contract duplicates prevents data leakage across folds; embeddings are fitted strictly per-fold on training data.

---

## Model & Experiment Configurations

### Default Hyperparameters

| Parameter | Default Value | Description |
|---|---:|---|
| `max_len` | `500` | Maximum input token sequence length |
| `w_ratio` | `0.6` | Ratio of token budget allocated to W-function (~300 tokens) |
| `vector_dim` | `300` | Token and segment embedding dimension |
| `hidden_units` | `300` | BiLSTM hidden units |
| `dense_units` | `300` | Classification head units |
| `batch_size` | `32` | Training batch size |
| `epochs` | `40` | Maximum epochs |
| `patience` | `5` | Early-stopping patience (epochs without improvement on `early_stop_monitor`) |
| `token_dropout` | `0.0` | Training-time probability of zeroing a whole token vector (`--token-dropout`) |
| `learning_rate` | `0.002` | Adam optimizer learning rate |
| `folds` | `5` | Outer cross-validation folds (`StratifiedGroupKFold`) |
| `validation_folds` | `5` | Inner validation folds for threshold tuning & early stopping |
| `min_recall` | `0.95` | Target recall constraint when `threshold_strategy=recall` |
| `threshold_strategy` | `recall` | Decision threshold selection: `recall` (hits `min_recall`) or `eer` (equal-error-rate operating point) |
| `early_stop_monitor` | `val_pr_auc` | Early-stopping metric: `val_loss`, `val_accuracy`, or `val_pr_auc` (recommended for imbalanced data) |

### Ablation Variants

| Variant | Alias | Embedding | `[SEP]` | Segment Embedding | Sequence Policy |
|---|---|---|:---:|:---:|---|
| `b0` | `baseline` | Word2Vec | No | No | Flat prefix truncation |
| `b1` | - | FastText | No | No | Flat prefix truncation |
| `b2` | - | FastText | Yes | No | Budgeted W/C prefix |
| `b3` | `variant_c` | FastText | Yes | Yes | Budgeted W/C prefix |
| `b4` | - | FastText | Yes | Yes | Budgeted W/C head-tail |

---

## Alternate Dataset: DAppSCAN

`Dataset/reentrancy_1671.txt` is extracted from 185 contracts total. To test generalization on
an independent, group-disjoint corpus, `data/build_dappscan_dataset.py` converts audit findings
from [InPlusLab/DAppSCAN](https://github.com/InPlusLab/DAppSCAN) (real audit-report-derived SWC-107
labels) into the same gadget format:

```bash
git clone --depth 1 https://github.com/InPlusLab/DAppSCAN.git /path/to/DAppSCAN
python3 -m data.build_dappscan_dataset \
  --dappscan-root /path/to/DAppSCAN \
  --output Dataset/dappscan_reentrancy.txt \
  --neg-ratio 2.0
```

Then point any run at it with `--data-file Dataset/dappscan_reentrancy.txt`.

---

## Installation

### 1. Setup Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify Installation

Run the test suite to verify configuration, data loading, tokenization, and models:

```bash
python3 -m unittest discover -s tests -v
```

---

## Usage

### Run Full Benchmark (All Variants & Folds)

Launches process-isolated runs for variants `b0` through `b4` across all 5 folds:

```bash
./experiment.sh EXPERIMENT/b0-b4-benchmark
```

### Run a Single Variant (e.g. Variant C / `b3`)

```bash
./experiment.sh EXPERIMENT/b3-run --variant b3
```

### Quick Smoke Test

Run a fast sanity check with reduced length, folds, and epochs:

```bash
./experiment.sh EXPERIMENT/smoke \
  --variant b3 \
  --folds 2 \
  --validation-folds 2 \
  --max-len 100 \
  --epochs 1
```

### Run a Specific Fold

Run fold 0 directly via Python CLI:

```bash
python3 main.py --run-dir EXPERIMENT/b3-fold-0 --fold 0 --variant b3
```

### Threshold Strategy & Early Stopping

```bash
# Equal-error-rate threshold instead of the recall-constrained default
python3 main.py --run-dir EXPERIMENT/b3-eer --variant b3 --threshold-strategy eer

# Early-stop on validation loss instead of PR-AUC
python3 main.py --run-dir EXPERIMENT/b3-valloss --variant b3 --early-stop-monitor val_loss
```

### Multi-Seed Runs

Run and aggregate the same benchmark across several seeds to check result stability instead of relying on a single seed:

```bash
# Train all listed seeds, then aggregate mean/std per variant into multi_seed_summary.json
python3 main.py --run-all --seeds 1,2,3,4 --run-dir EXPERIMENT/multi-seed --variant all

# Re-aggregate an existing multi-seed run (e.g. after adding more seed subdirectories) without retraining
python3 main.py --summarize --seeds 1,2,3,4 --run-dir EXPERIMENT/multi-seed --variant all
```

### Summarize Results

Generate aggregate metrics and paired differences against baseline (`b0`):

```bash
# Full summary (requires all folds complete)
python3 main.py --run-dir EXPERIMENT/b0-b4-benchmark --summarize --variant all

# Partial summary during an ongoing run
python3 main.py --run-dir EXPERIMENT/b3-run --summarize --variant b3 --allow-partial-summary
```

### Cross-Dataset Validation (1671 → DAppSCAN)

To evaluate true out-of-distribution generalization, train on `Dataset/reentrancy_1671.txt` and evaluate zero-shot on audit findings from DAppSCAN (`Dataset/dappscan_reentrancy.txt` and `Dataset/dappscan_reentrancy_ratio1.txt`):

```bash
# Run 5-seed cross-dataset evaluation for Variant C (b3) and Baseline (b0)
PYTHONPATH=. python3 -u experiments/run_cross_dataset.py --variants b3,b0 --seeds 42,1,2,3,4
```

Results are saved to `EXPERIMENT/cross-dataset/cross_dataset_results.json`:

| Test Set | Variant | PR-AUC | F1 | Recall | Precision | EER |
|---|---|---|---|---|---|---|
| **DAppSCAN 2:1** (342 samples) | **`b3` (Variant C)** | **37.52% ± 2.57%** | **49.26% ± 0.82%** | **94.21% ± 5.48%** | 33.38% ± 0.38% | **47.24% ± 1.60%** |
| | `b0` (Baseline) | 33.35% ± 0.86% | 45.35% ± 5.71% | 78.60% ± 24.40% | 33.94% ± 2.30% | 53.42% ± 1.17% |
| **DAppSCAN 1:1** (228 samples) | **`b3` (Variant C)** | **53.55% ± 2.72%** | **65.20% ± 1.55%** | **94.21% ± 5.48%** | 49.91% ± 0.66% | **48.16% ± 2.28%** |
| | `b0` (Baseline) | 50.31% ± 1.75% | 58.97% ± 9.82% | 78.60% ± 24.40% | 50.57% ± 2.52% | 53.33% ± 1.51% |

### Generalization Follow-up (pruning, expansion, label audit, distribution shift)

`experiments/run_cross_dataset.py` also accepts training-side interventions, and stores
test labels/probabilities so distribution analyses need no retraining:

| Flag | Effect |
|---|---|
| `--dedup` | Keep one sample per exact-duplicate group (1671 has only 185 unique gadgets) |
| `--abstract-ids` | `VAR17`/`FUN3` → `VAR`/`FUN`, numeric literals → `NUM` |
| `--min-token-groups K` | Tokens seen in fewer than K training groups → `[UNK]` (vocabulary fitted on the fit split only) |
| `--hidden-units/--dense-units/--dropout/--token-dropout` | Smaller / more regularized model; token dropout zeroes whole token vectors during training |
| `--extra-train FILE` | Append another gadget file to the training source |
| `--train-fraction F` | Label-stratified random fraction of training groups (learning curve) |
| `--test-files name=path,...` | Override the DAppSCAN test sets |

```bash
# Extra datasets (DAppSCAN full pool: all 114 positives + all 834 negatives; SmartBugs-curated expansion)
python3 -m data.build_dappscan_dataset --dappscan-root /path/to/DAppSCAN --output Dataset/dappscan_reentrancy_pool.txt --neg-ratio 100
python3 -m data.build_smartbugs_dataset --smartbugs-root /path/to/smartbugs-curated --output Dataset/smartbugs_reentrancy.txt

# Whole grid (settings: experiments/run_generalization_grid.sh --list), then aggregate
experiments/run_generalization_grid.sh baseline dedup dedup_all dedup_sb
PYTHONPATH=. python3 experiments/summarize_generalization.py --reference baseline

# Prevalence sweep + label-free output adjustment on saved probabilities
PYTHONPATH=. python3 experiments/distribution_shift.py --results EXPERIMENT/generalization/dedup_all/cross_dataset_results.json

# Label-noise audit and audited-location visibility on DAppSCAN
python3 -m data.audit_dappscan --dappscan-root /path/to/DAppSCAN
```

Results (5 seeds each, ROC-AUC on the full DAppSCAN pool of 948 gadgets; aggregated in
`EXPERIMENT/generalization/summary.md`, which also covers the 2:1 and 1:1 test sets):

| Setting | Fit samples | `b3` | `b0` |
|---|---:|---|---|
| `baseline` (original protocol) | 1338 | 53.49% ± 2.18% | 48.64% ± 1.39% |
| `dedup` | 148 | 47.59% ± 2.34% | 48.02% ± 1.43% |
| `dedup_all` (dedup + abstract-ids + vocab prune + regularization) | 147 | 52.51% ± 1.34% | **54.80% ± 1.45%** |
| `dedup_all_sb` (`dedup_all` + SmartBugs) | 206 | **58.10% ± 1.10%** | 54.75% ± 0.56% |
| `dedup_all_frac50` (half of the groups) | 73 | 53.36% ± 1.10% | 53.39% ± 2.18% |

- `dedup_all_sb` beats `baseline` for both variants on all three test sets in 5/5 seeds
  (b3 +4.6 to +5.5 points); with 5 paired seeds the smallest attainable Wilcoxon p is 0.0625.
- The label audit (`EXPERIMENT/label-audit/`) found 5/20 sampled positives and 0/20 negatives
  clearly mislabeled (about 8% noise weighted to 2:1), and only 42/114 positives whose audited
  location falls inside the b3 input window.
- Under the prevalence sweep (`EXPERIMENT/generalization/distribution_shift_summary.md`) ROC-AUC
  is stable across 10% to 67% positives. The recall threshold flags about 96% of samples, so its
  F1 equals the all-positive F1. Platt and EM prior correction collapse because the shift is in
  the features, not only in the label prior.

### Output Directory Structure

Each run outputs to `EXPERIMENT/<run-name>/`:
- `config.json`: Run configuration parameters
- `dataset.json`: Dataset statistics and extraction diagnostics
- `embeddings/`: Saved fold-specific Word2Vec/FastText models
- `models/`: Trained model weights (`.weights.h5`)
- `manifests/`: Reload manifests with decision thresholds
- `results/`: Per-fold evaluation metrics and predictions
- `summary.json`: Multi-fold summary and paired statistical comparisons

---

## Project Structure

```text
.
├── Dataset/
│   ├── reentrancy_1671.txt              # Dataset of 1,671 labeled Solidity gadgets
│   ├── dappscan_reentrancy.txt          # DAppSCAN reentrancy gadgets (2:1 ratio, 342 samples)
│   ├── dappscan_reentrancy_ratio1.txt   # DAppSCAN reentrancy gadgets (1:1 ratio, 228 samples)
│   ├── dappscan_reentrancy_pool.txt     # Full DAppSCAN pool (114 positives + 834 negatives)
│   └── smartbugs_reentrancy.txt         # SmartBugs-curated training expansion (31 + 60)
├── config/
│   └── config.py               # Validated ExperimentConfig dataclass & CLI helpers
├── data/                       # Parsing, normalization, extraction, and loader
│   ├── extraction.py
│   ├── grouping.py
│   ├── loader.py
│   ├── normalization.py
│   ├── parser.py
│   ├── tokenizer.py
│   ├── types.py
│   ├── build_dappscan_dataset.py  # Converts DAppSCAN audit data into the gadget format
│   ├── build_smartbugs_dataset.py # Converts SmartBugs-curated into the gadget format
│   └── audit_dappscan.py          # Label-noise audit sampling and audited-location visibility
├── models/                     # Model architectures & attention
│   ├── attention.py
│   ├── inspection.py
│   └── models.py
├── representation/             # Word2Vec/FastText embeddings & sequence policies
│   ├── embeddings.py
│   ├── pruning.py              # Dedup, identifier abstraction, vocabulary pruning, group subsampling
│   ├── sequences.py
│   ├── variants.py
│   └── vectorization.py
├── experiments/                # Training runner, splits, metrics, artifacts, reporting
│   ├── artifacts.py
│   ├── distribution_shift.py   # Prevalence sweep and label-free output adjustment
│   ├── metrics.py
│   ├── reporting.py
│   ├── reproducibility.py
│   ├── run_cross_dataset.py    # Zero-shot cross-dataset evaluation (1671 -> DAppSCAN)
│   ├── run_generalization_grid.sh  # Named settings for the generalization grid
│   ├── runner.py
│   ├── splits.py
│   └── summarize_generalization.py  # Aggregates the grid with paired Wilcoxon tests
├── tests/                      # Unit and regression test suite
├── experiment.sh               # Memory-bounded shell runner
├── main.py                     # CLI entry point
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```
