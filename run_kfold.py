"""
5-Fold stratified cross-validation for all three variants (ReChecker goc,
ReChecker goc that, Phuong an C), addressing two things flagged as needed
for a stable conclusion:
  1. One split -> mean+-std over 5 folds, so the comparison isn't resting on
     a single lucky/unlucky train/test draw.
  2. Embedding leakage -> Word2Vec/FastText are now fit PER FOLD using only
     that fold's train tokens (workers=1, deterministic), then used to
     vectorize both that fold's train and test gadgets. No fold's embedding
     ever sees that fold's test gadgets.
Same architecture/hyperparameters as the single-split run (models_common.py),
same early stopping policy. This does not yet do contract-deduplication
(separate, larger follow-up already flagged in the report).
"""
import re
import os
import gc
import json
import time
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec, FastText
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, confusion_matrix,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from data_pipeline import load_gadgets
from clean_fragment import clean_fragment
from tokenize_common import tokenize_line
from build_dataset import fixed_vectorize, fixed_segments, VEC_DIM
from models_common import build_baseline_model, build_phuong_an_c_model

SEED = 42
N_FOLDS = 5
function_regex = re.compile(r'function(\d)+')

np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_records():
    gadgets = load_gadgets()
    records = []
    for g in gadgets:
        raw_lines = g["raw_lines"]
        cleaned = clean_fragment(raw_lines)
        assert len(cleaned) == len(raw_lines)
        n_w = len(g["w_lines"])
        w_cleaned, c_cleaned = cleaned[:n_w], cleaned[n_w:]

        w_toks = [t for l in w_cleaned for t in tokenize_line(l)]
        c_toks = [t for l in c_cleaned for t in tokenize_line(l)]
        base_tokens = w_toks + c_toks
        if c_toks:
            pc_tokens = w_toks + ["[SEP]"] + c_toks
            pc_segments = [0] * len(w_toks) + [2] + [1] * len(c_toks)
        else:
            pc_tokens = w_toks
            pc_segments = [0] * len(w_toks)

        raw_toks = [t for l in raw_lines for t in tokenize_line(l)]

        def bslice(lines):
            bs = False
            for l in lines:
                toks = tokenize_line(l)
                bs = any(function_regex.match(t) for t in toks)
            return bs

        records.append({
            "label": g["label"],
            "base_tokens": base_tokens,
            "pc_tokens": pc_tokens,
            "pc_segments": pc_segments,
            "raw_tokens": raw_toks,
            "backwards_slice_clean": bslice(w_cleaned + c_cleaned),
            "backwards_slice_raw": bslice(raw_lines),
        })
    return records


def eer_from_roc(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2.0)


def evaluate(y_true, y_prob_pos):
    y_pred = (y_prob_pos >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_vulnerable": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_vulnerable": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_vulnerable": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob_pos),
        "pr_auc": average_precision_score(y_true, y_prob_pos),
        "eer": eer_from_roc(y_true, y_prob_pos),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else float("nan"),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else float("nan"),
    }


def fit_embed(corpus, use_fasttext=False):
    cls = FastText if use_fasttext else Word2Vec
    return cls(corpus, min_count=1, vector_size=VEC_DIM, sg=0, seed=SEED, workers=1)


def train_eval(build_model_fn, X_train, y_train_idx, X_test, y_test_idx, seg_train=None, seg_test=None, epochs=40):
    y_train_cat = to_categorical(y_train_idx, num_classes=2)
    y_test_cat = to_categorical(y_test_idx, num_classes=2)
    cw_arr = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train_idx)
    class_weight = {0: cw_arr[0], 1: cw_arr[1]}

    tf.keras.backend.clear_session()
    gc.collect()
    tf.random.set_seed(SEED)
    model = build_model_fn()
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    if seg_train is None:
        model.fit(X_train, y_train_cat, batch_size=64, epochs=epochs, class_weight=class_weight,
                  validation_data=(X_test, y_test_cat), callbacks=[es], verbose=0)
        prob = model.predict(X_test, batch_size=64, verbose=0)[:, 1]
    else:
        model.fit([X_train, seg_train], y_train_cat, batch_size=64, epochs=epochs, class_weight=class_weight,
                   validation_data=([X_test, seg_test], y_test_cat), callbacks=[es], verbose=0)
        prob = model.predict([X_test, seg_test], batch_size=64, verbose=0)[:, 1]
    result = evaluate(y_test_idx, prob)
    # explicit cleanup: repeated model creation inside this loop was
    # observed to leak memory (TF/Keras graph state + tf.function retracing)
    # until the process got OOM-killed by the container's memcg -- clear the
    # session and force a GC pass after every model fit, not just per fold.
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def main():
    t0 = time.time()
    records = build_records()
    y_all = np.array([r["label"] for r in records])
    n = len(records)
    print("total gadgets:", n)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n), y_all))  # deterministic given fixed random_state -> safe to resume against

    fold_results = {"baseline": [], "raw": [], "phuong_an_c": []}
    resume_path = "results_kfold_partial.json"
    start_fold = 0
    if os.path.exists(resume_path):
        with open(resume_path) as f:
            saved = json.load(f)
        # only trust folds where ALL THREE variants already have an entry
        n_complete = min(len(saved.get(k, [])) for k in ["baseline", "raw", "phuong_an_c"])
        if n_complete > 0:
            fold_results = {k: saved[k][:n_complete] for k in ["baseline", "raw", "phuong_an_c"]}
            start_fold = n_complete
            print(f"resuming from fold {start_fold+1}/{N_FOLDS} (found {n_complete} complete fold(s) in {resume_path})")

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        if fold_i < start_fold:
            continue
        t_fold = time.time()
        print(f"\n===== FOLD {fold_i+1}/{N_FOLDS} ===== train={len(train_idx)} test={len(test_idx)}")

        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]

        # --- baseline (clean, Word2Vec), fit embedding on TRAIN fold only ---
        w2v = fit_embed([r["base_tokens"] for r in train_records], use_fasttext=False)
        Xb_train = np.stack([fixed_vectorize(r["base_tokens"], w2v.wv, backwards_slice=r["backwards_slice_clean"]) for r in train_records])
        Xb_test = np.stack([fixed_vectorize(r["base_tokens"], w2v.wv, backwards_slice=r["backwards_slice_clean"]) for r in test_records])
        m_base = evaluate_variant = train_eval(build_baseline_model, Xb_train, y_all[train_idx], Xb_test, y_all[test_idx])
        fold_results["baseline"].append(m_base)
        print("baseline:", {k: round(v, 4) for k, v in m_base.items()})

        # --- raw (no clean_fragment, Word2Vec), fit embedding on TRAIN fold only ---
        w2v_raw = fit_embed([r["raw_tokens"] for r in train_records], use_fasttext=False)
        Xr_train = np.stack([fixed_vectorize(r["raw_tokens"], w2v_raw.wv, backwards_slice=r["backwards_slice_raw"]) for r in train_records])
        Xr_test = np.stack([fixed_vectorize(r["raw_tokens"], w2v_raw.wv, backwards_slice=r["backwards_slice_raw"]) for r in test_records])
        m_raw = train_eval(build_baseline_model, Xr_train, y_all[train_idx], Xr_test, y_all[test_idx])
        fold_results["raw"].append(m_raw)
        print("raw:", {k: round(v, 4) for k, v in m_raw.items()})

        # --- Phuong an C (clean + [SEP]/segment, FastText), fit embedding on TRAIN fold only ---
        ft = fit_embed([r["pc_tokens"] for r in train_records], use_fasttext=True)
        Xc_train = np.stack([fixed_vectorize(r["pc_tokens"], ft.wv, backwards_slice=r["backwards_slice_clean"]) for r in train_records])
        Xc_test = np.stack([fixed_vectorize(r["pc_tokens"], ft.wv, backwards_slice=r["backwards_slice_clean"]) for r in test_records])
        Sc_train = np.stack([fixed_segments(r["pc_segments"], backwards_slice=r["backwards_slice_clean"]) for r in train_records])
        Sc_test = np.stack([fixed_segments(r["pc_segments"], backwards_slice=r["backwards_slice_clean"]) for r in test_records])
        m_pc = train_eval(build_phuong_an_c_model, Xc_train, y_all[train_idx], Xc_test, y_all[test_idx], seg_train=Sc_train, seg_test=Sc_test)
        fold_results["phuong_an_c"].append(m_pc)
        print("phuong_an_c:", {k: round(v, 4) for k, v in m_pc.items()})

        print(f"fold {fold_i+1} done in {time.time()-t_fold:.0f}s, total elapsed {time.time()-t0:.0f}s")
        with open("results_kfold_partial.json", "w") as f:
            json.dump(fold_results, f, indent=2, default=float)

    summary = {}
    for variant, folds in fold_results.items():
        keys = folds[0].keys()
        summary[variant] = {
            k: {"mean": float(np.mean([f[k] for f in folds])), "std": float(np.std([f[k] for f in folds]))}
            for k in keys
        }
    with open("results_kfold.json", "w") as f:
        json.dump({"folds": fold_results, "summary": summary}, f, indent=2, default=float)
    print("\n\nSUMMARY:")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\ntotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
