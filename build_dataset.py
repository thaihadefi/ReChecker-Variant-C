"""
Builds the two parallel training sets:
  - Baseline (ReChecker goc): Word2Vec(CBOW,300d) over W+C tokens concatenated
    (exactly as AutoExtractCode.py/vectorize_fragment.py do), fixed 100-step
    zero-padded sequence, forward/backward slice exactly like the original.
  - Phuong an C: same W/C token content, but with a literal [SEP] token
    inserted between the W-function block and the C-function block(s), a
    per-token segment id (0=W, 1=C, 2=SEP), and FastText(CBOW,300d) instead
    of Word2Vec for the embedding.
Both use the SAME 1671 gadgets, SAME clean_fragment() normalization, and
SAME train/test split (fixed seed) for a fair, isolated comparison of just
the embedding+attention-segment change, per the professor's instruction to
keep the dataset fixed and do one focused comparison at a time.
"""
import re
import pickle
import numpy as np
from gensim.models import Word2Vec, FastText

from data_pipeline import load_gadgets
from clean_fragment import clean_fragment
from tokenize_common import tokenize_line

MAX_LEN = 100
VEC_DIM = 300
SEED = 42

function_regex = re.compile(r'function(\d)+')


def build():
    gadgets = load_gadgets()
    records = []
    for g in gadgets:
        cleaned = clean_fragment(g["raw_lines"])
        # clean_fragment() silently DROPS any line matching the multi-line-
        # comment-close regex, so cleaned can be shorter than raw_lines; the
        # fixed offset slice below only lands on the true W/C boundary if no
        # line was dropped. Verified 0/1671 gadgets trigger this on the
        # actual dataset, but assert it explicitly so a future re-run (e.g.
        # a different data file) fails loudly instead of silently
        # misaligning the [SEP]/segment boundary.
        assert len(cleaned) == len(g["raw_lines"]), (
            "clean_fragment() dropped a line -- W/C block boundary no longer valid for this gadget"
        )
        n_w = len(g["w_lines"])
        w_cleaned = cleaned[:n_w]
        c_cleaned = cleaned[n_w:]

        w_toks = []
        for line in w_cleaned:
            w_toks.extend(tokenize_line(line))
        c_toks = []
        for line in c_cleaned:
            c_toks.extend(tokenize_line(line))

        # baseline token stream: W then C, no separator (matches original)
        base_tokens = w_toks + c_toks
        # backward-slice flag exactly as FragmentVectorizer.tokenize_fragment
        # computes it (checked line by line, final value wins) -- reproduced
        # here at line level across the whole fragment for fidelity.
        backwards_slice = False
        for line in (w_cleaned + c_cleaned):
            toks = tokenize_line(line)
            if any(function_regex.match(t) for t in toks):
                backwards_slice = True
            else:
                backwards_slice = False

        # Phuong an C token stream: W + [SEP] + C (only if a C block exists)
        if c_toks:
            pc_tokens = w_toks + ["[SEP]"] + c_toks
            pc_segments = [0] * len(w_toks) + [2] + [1] * len(c_toks)
        else:
            pc_tokens = w_toks
            pc_segments = [0] * len(w_toks)

        records.append({
            "label": g["label"],
            "base_tokens": base_tokens,
            "pc_tokens": pc_tokens,
            "pc_segments": pc_segments,
            "backwards_slice": backwards_slice,
        })
    return records


def fixed_vectorize(tokens, embeddings, max_len=MAX_LEN, vec_dim=VEC_DIM, backwards_slice=False):
    vectors = np.zeros((max_len, vec_dim), dtype=np.float32)
    n = min(len(tokens), max_len)
    if backwards_slice:
        for i in range(n):
            tok = tokens[len(tokens) - 1 - i]
            if tok in embeddings:
                vectors[max_len - 1 - i] = embeddings[tok]
    else:
        for i in range(n):
            tok = tokens[i]
            if tok in embeddings:
                vectors[i] = embeddings[tok]
    return vectors


def fixed_segments(segments, max_len=MAX_LEN, backwards_slice=False):
    seg_arr = np.full((max_len,), 3, dtype=np.int32)  # 3 = PAD
    n = min(len(segments), max_len)
    if backwards_slice:
        for i in range(n):
            seg_arr[max_len - 1 - i] = segments[len(segments) - 1 - i]
    else:
        for i in range(n):
            seg_arr[i] = segments[i]
    return seg_arr


if __name__ == "__main__":
    records = build()
    print("records:", len(records))

    base_corpus = [r["base_tokens"] for r in records]
    pc_corpus = [r["pc_tokens"] for r in records]

    print("training Word2Vec (baseline)...")
    w2v = Word2Vec(base_corpus, min_count=1, vector_size=VEC_DIM, sg=0, seed=SEED, workers=1)
    print("training FastText (Phuong an C)...")
    ft = FastText(pc_corpus, min_count=1, vector_size=VEC_DIM, sg=0, seed=SEED, workers=1)

    X_base = np.stack([
        fixed_vectorize(r["base_tokens"], w2v.wv, backwards_slice=r["backwards_slice"])
        for r in records
    ])
    X_pc = np.stack([
        fixed_vectorize(r["pc_tokens"], ft.wv, backwards_slice=r["backwards_slice"])
        for r in records
    ])
    Seg_pc = np.stack([
        fixed_segments(r["pc_segments"], backwards_slice=r["backwards_slice"])
        for r in records
    ])
    y = np.array([r["label"] for r in records], dtype=np.int64)

    print("X_base", X_base.shape, "X_pc", X_pc.shape, "Seg_pc", Seg_pc.shape, "y", y.shape)

    with open("dataset.pkl", "wb") as f:
        pickle.dump({"X_base": X_base, "X_pc": X_pc, "Seg_pc": Seg_pc, "y": y}, f)
    print("saved dataset.pkl")
