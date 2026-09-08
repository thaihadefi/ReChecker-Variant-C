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
| `epochs` | `40` | Maximum epochs (early stopping patience: 5) |
| `learning_rate` | `0.002` | Adam optimizer learning rate |
| `folds` | `5` | Outer cross-validation folds (`StratifiedGroupKFold`) |
| `validation_folds` | `5` | Inner validation folds for threshold tuning & early stopping |
| `min_recall` | `0.95` | Target recall constraint for optimal decision threshold |

### Ablation Variants

| Variant | Alias | Embedding | `[SEP]` | Segment Embedding | Sequence Policy |
|---|---|---|:---:|:---:|---|
| `b0` | `baseline` | Word2Vec | No | No | Flat prefix truncation |
| `b1` | - | FastText | No | No | Flat prefix truncation |
| `b2` | - | FastText | Yes | No | Budgeted W/C prefix |
| `b3` | `variant_c` | FastText | Yes | Yes | Budgeted W/C prefix |
| `b4` | - | FastText | Yes | Yes | Budgeted W/C head-tail |

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

### Summarize Results

Generate aggregate metrics and paired differences against baseline (`b0`):

```bash
# Full summary (requires all folds complete)
python3 main.py --run-dir EXPERIMENT/b0-b4-benchmark --summarize --variant all

# Partial summary during an ongoing run
python3 main.py --run-dir EXPERIMENT/b3-run --summarize --variant b3 --allow-partial-summary
```

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
│   └── reentrancy_1671.txt     # Dataset of 1,671 labeled Solidity gadgets
├── rechecker/                  # Core package
│   ├── data/                   # Parsing, normalization, extraction, grouping
│   ├── representation/         # Word2Vec/FastText embeddings, sequence policies
│   ├── modeling/               # Masked BiLSTM and additive attention
│   ├── experiments/            # Runner, splits, metrics, artifacts, reporting
│   ├── config.py               # Experiment configuration dataclass
│   └── cli.py                  # CLI orchestration
├── tests/                      # Unit and regression test suite
├── experiment.sh               # Memory-bounded shell runner
├── main.py                     # CLI entry point
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## License

Released under the MIT License for academic and research use.
