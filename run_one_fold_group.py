"""
One fold of GroupKFold CV where the 'group' is the exact normalized-token
duplicate cluster (base_tokens after clean_fragment) a gadget belongs to.
Purpose: the earlier 5-fold run (run_one_fold.py, StratifiedKFold) fixed the
embedding-leakage issue but NOT this one -- duplicate/near-duplicate gadgets
(only 185 distinct normalized-token clusters out of 1671 gadgets; largest
cluster repeats 189 times, 0 mixed-label clusters) can and do get split
across train/test by plain StratifiedKFold, so a model can still "recognize"
a test gadget's near-identical twin from training. StratifiedGroupKFold guarantees
every gadget in the same cluster lands on the same side of the split (while
still balancing the label ratio per fold, unlike plain GroupKFold which was
tried first and produced wildly skewed per-fold test-set positive rates),
so this measures how much of the earlier results depended on that leakage.
Run as its own subprocess per fold (see run_kfold_group_driver.sh), same
OOM-avoidance reasoning as run_one_fold.py.
"""
import sys
import os
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from run_kfold import build_records, fit_embed, train_eval
from build_dataset import fixed_vectorize, fixed_segments
from models_common import build_baseline_model, build_phuong_an_c_model

N_FOLDS = 5
RESUME_PATH = "results_kfold_group_partial.json"


def main():
    fold_i = int(sys.argv[1])
    records = build_records()
    y_all = np.array([r["label"] for r in records])
    n = len(records)

    # group id = the gadget's normalized (base_tokens) sequence -- exact
    # match here means same structural pattern post clean_fragment, our
    # proxy for "duplicate/cloned contract" established during the dup audit.
    key_to_gid = {}
    groups = np.empty(n, dtype=np.int64)
    for i, r in enumerate(records):
        key = tuple(r["base_tokens"])
        gid = key_to_gid.setdefault(key, len(key_to_gid))
        groups[i] = gid
    print(f"total gadgets={n}, distinct duplicate-groups={len(key_to_gid)}")

    # plain GroupKFold produced wildly skewed test-set class balance across
    # folds (17%-55% positive rate observed) because duplicate clusters are
    # label-homogeneous -- StratifiedGroupKFold keeps the group constraint
    # (no cluster split across train/test) while balancing label ratio
    # per fold, so folds stay comparable to each other and to sec.4's
    # StratifiedKFold results.
    gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(gkf.split(np.zeros(n), y_all, groups=groups))
    train_idx, test_idx = splits[fold_i]
    print(f"FOLD {fold_i+1}/{N_FOLDS} train={len(train_idx)} test={len(test_idx)} "
          f"test_pos_rate={y_all[test_idx].mean():.3f}")

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

    saved["baseline"].append(m_base)
    saved["raw"].append(m_raw)
    saved["phuong_an_c"].append(m_pc)
    with open(RESUME_PATH, "w") as f:
        json.dump(saved, f, indent=2, default=float)
    print(f"fold {fold_i+1} appended. total folds saved: {len(saved['baseline'])}")


if __name__ == "__main__":
    main()
