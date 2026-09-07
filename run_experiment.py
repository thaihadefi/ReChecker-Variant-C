import pickle
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, confusion_matrix,
    classification_report,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from models_common import build_baseline_model, build_phuong_an_c_model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

with open("dataset.pkl", "rb") as f:
    D = pickle.load(f)

X_base, X_pc, Seg_pc, y = D["X_base"], D["X_pc"], D["Seg_pc"], D["y"]
idx = np.arange(len(y))

idx_train, idx_test = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
print("train:", len(idx_train), "test:", len(idx_test))
print("train label balance:", np.bincount(y[idx_train]))
print("test label balance:", np.bincount(y[idx_test]))

class_weight_arr = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y[idx_train])
class_weight = {0: class_weight_arr[0], 1: class_weight_arr[1]}
print("class_weight:", class_weight)


def eer_from_roc(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2.0
    return float(eer)


def evaluate(name, y_true, y_prob_pos):
    y_pred = (y_prob_pos >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_vulnerable": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_vulnerable": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_vulnerable": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob_pos),
        "pr_auc": average_precision_score(y_true, y_prob_pos),
        "eer": eer_from_roc(y_true, y_prob_pos),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print("  confusion_matrix:", metrics["confusion_matrix"])
    print(classification_report(y_true, y_pred, target_names=["not-vulnerable", "vulnerable"], zero_division=0))
    return metrics


results = {}

# ---------------- baseline: ReChecker goc ----------------
print("\n\n########## Training baseline (ReChecker goc) ##########")
Xb_train, Xb_test = X_base[idx_train], X_base[idx_test]
y_train_cat = to_categorical(y[idx_train], num_classes=2)
y_test_cat = to_categorical(y[idx_test], num_classes=2)

tf.random.set_seed(SEED)
model_base = build_baseline_model()
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
hist_base = model_base.fit(
    Xb_train, y_train_cat, batch_size=64, epochs=40, class_weight=class_weight,
    validation_data=(Xb_test, y_test_cat), callbacks=[es], verbose=2,
)
prob_base = model_base.predict(Xb_test, batch_size=64)[:, 1]
results["baseline"] = evaluate("ReChecker goc", y[idx_test], prob_base)
results["baseline"]["epochs_trained"] = len(hist_base.history["loss"])

# ---------------- Phuong an C ----------------
print("\n\n########## Training Phuong an C ##########")
Xc_train, Xc_test = X_pc[idx_train], X_pc[idx_test]
Sc_train, Sc_test = Seg_pc[idx_train], Seg_pc[idx_test]

tf.random.set_seed(SEED)
model_pc = build_phuong_an_c_model()
es2 = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
hist_pc = model_pc.fit(
    [Xc_train, Sc_train], y_train_cat, batch_size=64, epochs=40, class_weight=class_weight,
    validation_data=([Xc_test, Sc_test], y_test_cat), callbacks=[es2], verbose=2,
)
prob_pc = model_pc.predict([Xc_test, Sc_test], batch_size=64)[:, 1]
results["phuong_an_c"] = evaluate("Phuong an C", y[idx_test], prob_pc)
results["phuong_an_c"]["epochs_trained"] = len(hist_pc.history["loss"])

with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nsaved results.json")
