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

from models_common import build_baseline_model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

with open("dataset_raw.pkl", "rb") as f:
    D = pickle.load(f)
X_raw, y = D["X_raw"], D["y"]
idx = np.arange(len(y))

# identical split as run_experiment.py (same seed, same y, same order)
idx_train, idx_test = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
print("train:", len(idx_train), "test:", len(idx_test))

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


print("\n\n########## Training ReChecker goc THAT (raw, no clean_fragment) ##########")
Xr_train, Xr_test = X_raw[idx_train], X_raw[idx_test]
y_train_cat = to_categorical(y[idx_train], num_classes=2)
y_test_cat = to_categorical(y[idx_test], num_classes=2)

tf.random.set_seed(SEED)
model_raw = build_baseline_model()
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
hist = model_raw.fit(
    Xr_train, y_train_cat, batch_size=64, epochs=40, class_weight=class_weight,
    validation_data=(Xr_test, y_test_cat), callbacks=[es], verbose=2,
)
prob_raw = model_raw.predict(Xr_test, batch_size=64)[:, 1]
result = evaluate("ReChecker goc that (khong clean_fragment)", y[idx_test], prob_raw)
result["epochs_trained"] = len(hist.history["loss"])

with open("results_raw.json", "w") as f:
    json.dump(result, f, indent=2, default=float)
print("\nsaved results_raw.json")
