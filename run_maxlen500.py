"""
Single 80/20-split diagnostic: does raising the fixed sequence length from
100 to 500 tokens (still far short of the median 2665-token gadget, but
covering ~20% of gadgets in full vs ~2.7% at len=100, and capturing much
more of the C-function for the rest) change the ReChecker-goc vs
Phuong-an-C picture? Only these two variants are run (the 'raw' variant's
weakness is already established by the k-fold results) and only a single
stratified split (same seed=42 split as the original mục 3 experiment) --
full 5-fold at this length was judged too expensive for this environment
(LSTM cost scales with sequence length, so ~5x the per-epoch cost of the
len=100 runs), and is flagged as a disclosed limitation in the report.
"""
import gc
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from run_kfold import build_records, fit_embed, evaluate
from build_dataset import fixed_vectorize, fixed_segments
from models_common import build_baseline_model, build_phuong_an_c_model
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

results = {}

# --- baseline (Word2Vec), embedding fit on TRAIN only (same leakage fix as sec.4) ---
print("fitting baseline Word2Vec...")
w2v = fit_embed([r["base_tokens"] for r in train_records], use_fasttext=False)
Xb_train = np.stack([fixed_vectorize(r["base_tokens"], w2v.wv, max_len=MAX_LEN, backwards_slice=r["backwards_slice_clean"]) for r in train_records])
Xb_test = np.stack([fixed_vectorize(r["base_tokens"], w2v.wv, max_len=MAX_LEN, backwards_slice=r["backwards_slice_clean"]) for r in test_records])
print("Xb_train", Xb_train.shape)

y_train_cat = to_categorical(y_all[idx_train], num_classes=2)
y_test_cat = to_categorical(y_all[idx_test], num_classes=2)
cw_arr = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_all[idx_train])
class_weight = {0: cw_arr[0], 1: cw_arr[1]}

tf.keras.backend.clear_session(); gc.collect()
model = build_baseline_model(max_len=MAX_LEN)
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model.fit(Xb_train, y_train_cat, batch_size=32, epochs=40, class_weight=class_weight,
          validation_data=(Xb_test, y_test_cat), callbacks=[es], verbose=2)
prob = model.predict(Xb_test, batch_size=32, verbose=0)[:, 1]
results["baseline"] = evaluate(y_all[idx_test], prob)
results["baseline"]["epochs_trained"] = len(model.history.history["loss"])
print("baseline:", results["baseline"])
del model, Xb_train, Xb_test
tf.keras.backend.clear_session(); gc.collect()

with open("results_maxlen500_partial.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

# --- Phuong an C (FastText + [SEP]/segment), embedding fit on TRAIN only ---
print("fitting Phuong an C FastText...")
ft = fit_embed([r["pc_tokens"] for r in train_records], use_fasttext=True)
Xc_train = np.stack([fixed_vectorize(r["pc_tokens"], ft.wv, max_len=MAX_LEN, backwards_slice=r["backwards_slice_clean"]) for r in train_records])
Xc_test = np.stack([fixed_vectorize(r["pc_tokens"], ft.wv, max_len=MAX_LEN, backwards_slice=r["backwards_slice_clean"]) for r in test_records])
Sc_train = np.stack([fixed_segments(r["pc_segments"], max_len=MAX_LEN, backwards_slice=r["backwards_slice_clean"]) for r in train_records])
Sc_test = np.stack([fixed_segments(r["pc_segments"], max_len=MAX_LEN, backwards_slice=r["backwards_slice_clean"]) for r in test_records])
print("Xc_train", Xc_train.shape)

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
