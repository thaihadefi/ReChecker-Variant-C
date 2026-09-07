"""
Resume run_maxlen500.py after it OOM-killed while fitting FastText for the
Phuong-an-C half (baseline half already completed and saved in
results_maxlen500_partial.json). Runs ONLY the Phuong-an-C part, as its own
fresh process for full memory headroom, then merges into results_maxlen500.json.
"""
import gc
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from run_kfold import build_records, fit_embed, evaluate
from build_dataset import fixed_vectorize, fixed_segments
from models_common import build_phuong_an_c_model
from sklearn.utils import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
MAX_LEN = 500

np.random.seed(SEED)
tf.random.set_seed(SEED)

records = build_records()
y_all = np.array([r["label"] for r in records])
n = len(records)
idx = np.arange(n)
idx_train, idx_test = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=SEED)
print(f"train={len(idx_train)} test={len(idx_test)} test_pos={y_all[idx_test].sum()}")

train_records = [records[i] for i in idx_train]
test_records = [records[i] for i in idx_test]

with open("results_maxlen500_partial.json") as f:
    results = json.load(f)

y_train_cat = to_categorical(y_all[idx_train], num_classes=2)
y_test_cat = to_categorical(y_all[idx_test], num_classes=2)
cw_arr = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_all[idx_train])
class_weight = {0: cw_arr[0], 1: cw_arr[1]}

print("fitting Phuong an C FastText...")
ft = fit_embed([r["pc_tokens"] for r in train_records], use_fasttext=True)
print("FastText fit done, vectorizing...")
Xc_train = np.stack([fixed_vectorize(r["pc_tokens"], ft.wv, max_len=MAX_LEN, backwards_slice=r["backwards_slice_clean"]) for r in train_records])
Xc_test = np.stack([fixed_vectorize(r["pc_tokens"], ft.wv, max_len=MAX_LEN, backwards_slice=r["backwards_slice_clean"]) for r in test_records])
Sc_train = np.stack([fixed_segments(r["pc_segments"], max_len=MAX_LEN, backwards_slice=r["backwards_slice_clean"]) for r in train_records])
Sc_test = np.stack([fixed_segments(r["pc_segments"], max_len=MAX_LEN, backwards_slice=r["backwards_slice_clean"]) for r in test_records])
print("Xc_train", Xc_train.shape)
del ft
gc.collect()

tf.keras.backend.clear_session(); gc.collect()
model = build_phuong_an_c_model(max_len=MAX_LEN)
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit([Xc_train, Sc_train], y_train_cat, batch_size=32, epochs=40, class_weight=class_weight,
          validation_data=([Xc_test, Sc_test], y_test_cat), callbacks=[es], verbose=2)
prob = model.predict([Xc_test, Sc_test], batch_size=32, verbose=0)[:, 1]
results["phuong_an_c"] = evaluate(y_all[idx_test], prob)
results["phuong_an_c"]["epochs_trained"] = len(model.history.history["loss"])
print("phuong_an_c:", results["phuong_an_c"])

with open("results_maxlen500.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("DONE")
