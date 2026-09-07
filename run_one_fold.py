"""Run exactly ONE fold of the 5-fold CV and append its result to
results_kfold_partial.json, then exit. Run as a fresh subprocess per fold
(see run_kfold_driver.sh) so the OS fully reclaims memory between folds --
clear_session()+gc.collect() inside a long-lived process were not enough to
stop the previous run from being OOM-killed by the container's 6GB cgroup
limit around fold 3.
"""
import sys
import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

from run_kfold import build_records, fit_embed, train_eval
from build_dataset import fixed_vectorize, fixed_segments
from models_common import build_baseline_model, build_phuong_an_c_model

SEED = 42
N_FOLDS = 5
RESUME_PATH = "results_kfold_partial.json"


def main():
    fold_i = int(sys.argv[1])
    records = build_records()
    y_all = np.array([r["label"] for r in records])
    n = len(records)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(n), y_all))
    train_idx, test_idx = splits[fold_i]
    print(f"FOLD {fold_i+1}/{N_FOLDS} train={len(train_idx)} test={len(test_idx)}")

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]

    w2v = fit_embed([r["base_tokens"] for r in train_records], use_fasttext=False)
    Xb_train = np.stack([fixed_vectorize(r["base_tokens"], w2v.wv, backwards_slice=r["backwards_slice_clean"]) for r in train_records])
    Xb_test = np.stack([fixed_vectorize(r["base_tokens"], w2v.wv, backwards_slice=r["backwards_slice_clean"]) for r in test_records])
    m_base = train_eval(build_baseline_model, Xb_train, y_all[train_idx], Xb_test, y_all[test_idx])
    print("baseline:", m_base)

    w2v_raw = fit_embed([r["raw_tokens"] for r in train_records], use_fasttext=False)
    Xr_train = np.stack([fixed_vectorize(r["raw_tokens"], w2v_raw.wv, backwards_slice=r["backwards_slice_raw"]) for r in train_records])
    Xr_test = np.stack([fixed_vectorize(r["raw_tokens"], w2v_raw.wv, backwards_slice=r["backwards_slice_raw"]) for r in test_records])
    m_raw = train_eval(build_baseline_model, Xr_train, y_all[train_idx], Xr_test, y_all[test_idx])
    print("raw:", m_raw)

    ft = fit_embed([r["pc_tokens"] for r in train_records], use_fasttext=True)
    Xc_train = np.stack([fixed_vectorize(r["pc_tokens"], ft.wv, backwards_slice=r["backwards_slice_clean"]) for r in train_records])
    Xc_test = np.stack([fixed_vectorize(r["pc_tokens"], ft.wv, backwards_slice=r["backwards_slice_clean"]) for r in test_records])
    Sc_train = np.stack([fixed_segments(r["pc_segments"], backwards_slice=r["backwards_slice_clean"]) for r in train_records])
    Sc_test = np.stack([fixed_segments(r["pc_segments"], backwards_slice=r["backwards_slice_clean"]) for r in test_records])
    m_pc = train_eval(build_phuong_an_c_model, Xc_train, y_all[train_idx], Xc_test, y_all[test_idx], seg_train=Sc_train, seg_test=Sc_test)
    print("phuong_an_c:", m_pc)

    if os.path.exists(RESUME_PATH):
        with open(RESUME_PATH) as f:
            saved = json.load(f)
    else:
        saved = {"baseline": [], "raw": [], "phuong_an_c": []}

    # this fold's index should be exactly len(saved["baseline"]) if run in order
    saved["baseline"].append(m_base)
    saved["raw"].append(m_raw)
    saved["phuong_an_c"].append(m_pc)
    with open(RESUME_PATH, "w") as f:
        json.dump(saved, f, indent=2, default=float)
    print(f"fold {fold_i+1} appended. total folds saved: {len(saved['baseline'])}")


if __name__ == "__main__":
    main()
