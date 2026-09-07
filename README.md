# Comparative Experiments: Baseline ReChecker vs. Variant C

Code used to run the comparison experiments between the original ReChecker (Word2Vec + BiLSTM/Attention) and Variant C (FastText embeddings + a [SEP] boundary token + segment embeddings) for smart contract reentrancy detection. Results and analysis are in the report, not repeated here — this README only covers running the code.

Implemented with TensorFlow 2 / tf.keras and Gensim 4+.

---

## 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Dataset

`reentrancy_1671.txt` (1,671 gadgets: 576 vulnerable, 1,095 non-vulnerable), taken from the original ReChecker repo, already included here.

---

## 3. Running Experiments

### Experiment 1: Single 80/20 Stratified Split (Report Section 3)

```bash
python build_dataset.py        # dataset.pkl (Baseline + Variant C)
python build_dataset_raw.py    # dataset_raw.pkl (Baseline Raw)
python run_experiment.py       # evaluates Baseline + Variant C
python run_experiment_raw.py   # evaluates Baseline Raw
```

Reproducibility note: `build_dataset.py`/`build_dataset_raw.py` fit Word2Vec/FastText with `workers=1` for reproducible vectors. The original numbers in Section 3 of the report were produced before this fix (`workers=3`, not exactly reproducible run-to-run) — re-running the commands above uses `workers=1` vectors, so results will be close to, but not bit-identical to, the report's numbers.

### Experiment 2: 5-Fold Stratified Cross-Validation (Report Section 4)

```bash
python run_kfold.py
```

If this OOMs, run one fold per subprocess instead (same result, resumable):

```bash
bash run_kfold_driver.sh
```

### Experiment 3: StratifiedGroupKFold by 185 duplicate-structure contract clusters (Report Section 6)

```bash
bash run_kfold_group_driver.sh
```

### Experiment 4: max_len = 500 (mentioned as a limitation in Report Section 7, not a standalone results section)

```bash
python run_maxlen500.py
python run_maxlen500_pc_only.py
```

Just a quick check on a single random split (not StratifiedGroupKFold) to see whether longer sequences cause training difficulty — not used to compare Baseline vs. Variant C, so the report doesn't quote specific numbers from this run.

---

## 4. File Structure

| File | Description |
| --- | --- |
| `reentrancy_1671.txt` | Original dataset, 1,671 gadgets |
| `data_pipeline.py` | Gadget parsing, splits W-function/C-function |
| `clean_fragment.py` | Normalizes identifiers to `VAR#`/`FUN#` |
| `tokenize_common.py` | Solidity tokenizer, shared by both pipelines |
| `models_common.py` | Model architectures (Attention, Baseline, Variant C) |
| `build_dataset.py` / `build_dataset_raw.py` | Preprocessing for normalized / raw variants |
| `run_experiment.py` / `run_experiment_raw.py` | Runs the single-split evaluation |
| `run_kfold.py` | 5-Fold CV, single process |
| `run_one_fold.py` + `run_kfold_driver.sh` | 5-Fold CV, one fold per process (avoids OOM) |
| `run_one_fold_group.py` + `run_kfold_group_driver.sh` | StratifiedGroupKFold, one fold per process |
| `summarize_kfold.py` | Aggregates `results_kfold*_partial.json` into mean ± std |
| `run_maxlen500.py`, `run_maxlen500_pc_only.py` | max_len=500 experiment |
| `requirements.txt` | Python dependencies |