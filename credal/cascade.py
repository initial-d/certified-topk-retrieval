"""Second application: certified top-k of an expensive re-ranker in a retrieval cascade.

Resolution = one cross-encoder (CE) call. Approximate score = bi-encoder (BGE) cosine. Pool P(q) = bi-encoder top-M.
Target: the CE top-k within P(q) (the standard cascade decision).
Step 1 (this module): CE scores for every (query, pool document), cached; CPU-friendly and resumable.
Usage: python -m credal.cascade score [datasets...]"""
import os
import sys

import numpy as np

from . import encode
from .data import CACHE_DIR, DATASETS, load, self_mask

M = 100
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def pool(name, M=M):
    ds = load(name)
    X, Q = encode.get(name, "bge", "docs"), encode.get(name, "bge", "queries")
    S = Q @ X.T
    sm = self_mask(ds)
    rows = np.flatnonzero(sm >= 0)
    S[rows, sm[rows]] = -np.inf
    idx = np.argsort(-S, 1)[:, :M]
    return ds, idx, np.take_along_axis(S, idx, 1)


def score_bgebase(name, M_big=1000):
    """Aligned cascade: bge-small top-M_big re-scored by bge-base (cosine)."""
    path = os.path.join(CACHE_DIR, f"{name}.cascade_bgebase.npz")
    if os.path.exists(path):
        return
    ds, idx, sh = pool(name, M_big)
    Xb, Qb = encode.get(name, "bgebase", "docs"), encode.get(name, "bgebase", "queries")
    s = np.einsum("qd,qmd->qm", Qb, Xb[idx]).astype(np.float32)
    np.savez(path, idx=idx, sh=sh.astype(np.float32), s=s)
    print(f"[cascade-bgebase] {name}: pool {M_big}", flush=True)


def score(name, device="cpu"):
    path = os.path.join(CACHE_DIR, f"{name}.cascade.npz")
    if os.path.exists(path):
        return
    from sentence_transformers import CrossEncoder
    ds, idx, sh = pool(name)
    ce = CrossEncoder(CE_MODEL, device=device, max_length=256)
    pairs = [(ds.query_texts[i], ds.doc_texts[d]) for i in range(len(idx)) for d in idx[i]]
    s = np.asarray(ce.predict(pairs, batch_size=128, show_progress_bar=False), np.float32).reshape(len(idx), M)
    np.savez(path, idx=idx, sh=sh.astype(np.float32), s=s)
    print(f"[cascade] {name}: {len(pairs):,} CE pairs", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "score":
        for n in sys.argv[2:] or DATASETS:
            score(n)
    elif sys.argv[1] == "bgebase":
        for n in sys.argv[2:] or DATASETS:
            score_bgebase(n)
