"""
Builds a THIRD variant: 'ReChecker goc that' (real) -- exactly what
SmConVulDetector.py actually runs, which (per direct inspection of the repo)
imports clean_fragment but never calls it. So this variant skips VAR#/FUN#
normalization entirely and trains Word2Vec directly on the raw identifiers
(contract names, real variable names, addresses, literals, ...), matching
the literal executable behavior of the original repo rather than the
documented design in README.md.
Same 1671 gadgets, same tokenizer, same 100-step fixed vectorization, same
train/test split (by construction: gadget order is identical to build_dataset.py).
"""
import re
import pickle
import numpy as np
from gensim.models import Word2Vec

from data_pipeline import load_gadgets
from tokenize_common import tokenize_line
from build_dataset import fixed_vectorize, VEC_DIM, SEED

function_regex = re.compile(r'function(\d)+')


def build():
    gadgets = load_gadgets()
    records = []
    for g in gadgets:
        # NO clean_fragment() call here -- this is the point: reproduces the
        # real (uncleaned) input SmConVulDetector.py actually feeds in.
        raw_lines = g["raw_lines"]
        toks = []
        for line in raw_lines:
            toks.extend(tokenize_line(line))

        backwards_slice = False
        for line in raw_lines:
            lt = tokenize_line(line)
            if any(function_regex.match(t) for t in lt):
                backwards_slice = True
            else:
                backwards_slice = False

        records.append({
            "label": g["label"],
            "tokens": toks,
            "backwards_slice": backwards_slice,
        })
    return records


if __name__ == "__main__":
    records = build()
    print("records:", len(records))
    corpus = [r["tokens"] for r in records]

    print("training Word2Vec on RAW (unnormalized) identifiers...")
    w2v = Word2Vec(corpus, min_count=1, vector_size=VEC_DIM, sg=0, seed=SEED, workers=1)

    X_raw = np.stack([
        fixed_vectorize(r["tokens"], w2v.wv, backwards_slice=r["backwards_slice"])
        for r in records
    ])
    y = np.array([r["label"] for r in records], dtype=np.int64)
    print("X_raw", X_raw.shape, "y", y.shape)

    with open("dataset_raw.pkl", "wb") as f:
        pickle.dump({"X_raw": X_raw, "y": y}, f)
    print("saved dataset_raw.pkl")
