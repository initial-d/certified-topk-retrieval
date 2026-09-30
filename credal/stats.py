"""Compact per-query statistics for the residual-box family, from which every calibration analysis is computed fast.

For each (dataset, quantizer, k) and query q (exact scores s, approximate sh, residual norms r):
  alpha_dec  = max_{t in T} (s_(k+1) - sh_t) / (||q|| r_t) v 0            (Thm. characterization)
  alpha_cov  = max_d |s_d - sh_d| / (||q|| r_d)                            (coverage of the symmetric box)
  V          = sorted (s_(k) - sh_j) / (||q|| r_j) for j not in T, truncated at VMAX
               -> oracle cost |A(q)|(c) = k + #{V <= c}
  WT         = (s_(k+m) - sh_t)/(||q|| r_t) inputs for the worst-case miss count m*(c)
  R          = smallest depth of the approximate ranking containing T      (rank boxes)
  feat       = index-side query features (no exact scores) for Mondrian calibration
  wc_*       = winner's-curse samples: standardized errors (s - sh)/(||q|| r) for winners / random documents
Usage: python -m credal.stats [datasets...]"""
import os
import pickle
import sys
import warnings

import numpy as np

from . import encode
from .data import CACHE_DIR, DATASETS, load, self_mask
from .credal import required_depth
from .index import quantize

warnings.filterwarnings("ignore", category=RuntimeWarning)
QUANTS = ["pq16", "pq32", "pq96", "opq32", "sq4", "bin"]
KS = [1, 5, 10, 20, 50]
VMAX = 1.5


def path(name, q):
    return os.path.join(CACHE_DIR, f"{name}.{q}.stats.pkl")


def compute(name, qname, model="bge", seed=0):
    ds = load(name)
    X, Q = encode.get(name, model, "docs"), encode.get(name, model, "queries")
    Xh, r, _ = quantize(qname, X)
    S, Sh = Q @ X.T, Q @ Xh.T
    sm = self_mask(ds)
    rows = np.flatnonzero(sm >= 0)
    S[rows, sm[rows]] = -1e9
    Sh[rows, sm[rows]] = -1e9
    qn = np.linalg.norm(Q, axis=1)
    E = (S - Sh) / (qn[:, None] * r[None, :] + 1e-8)
    E[rows, sm[rows]] = 0
    nq = len(Q)
    out = dict(dataset=name, quant=qname, N=X.shape[0], cal=ds.calib_idx, te=ds.test_idx,
               alpha_cov=np.abs(E).max(1), per_k={})
    order = np.argsort(-S, 1)[:, : max(KS) * 2 + 1]
    for k in KS:
        T = order[:, :k]
        s_sorted = np.take_along_axis(S, order, 1)
        sk, sk1 = s_sorted[:, k - 1], s_sorted[:, k]
        shT, rT = np.take_along_axis(Sh, T, 1), r[T]
        alpha_dec = np.maximum(((sk1[:, None] - shT) / (qn[:, None] * rT + 1e-8)).max(1), 0)
        V = []
        for i in range(nq):
            v = (sk[i] - Sh[i]) / (qn[i] * r + 1e-8)
            v[T[i]] = np.inf
            if sm[i] >= 0:
                v[sm[i]] = np.inf
            V.append(np.sort(v[v <= VMAX]).astype(np.float32))
        # m*(c) inputs: w[i, t, m] = (s_(k+m) - sh_t)/(||q|| r_t), m = 1..k  (u_t(c) < s_(k+m) <=> c < w)
        W = (s_sorted[:, k:2 * k][:, None, :] - shT[:, :, None]) / (qn[:, None, None] * rT[:, :, None] + 1e-8)
        kth = sk[:, None]
        R = np.array([required_depth(S[i], Sh[i], k) for i in range(nq)])
        # index-side features (approximate information only)
        sh_sorted = -np.sort(-Sh, 1)[:, : max(k + 1, 100)]
        rbar = np.array([r[np.argpartition(-Sh[i], k)[:k]].mean() for i in range(nq)])
        feat = dict(gap=(sh_sorted[:, k - 1] - sh_sorted[:, k]) / (qn * rbar + 1e-8),
                    spread=(sh_sorted[:, k - 1] - sh_sorted[:, 99]) / (qn * rbar + 1e-8),
                    rbar=rbar)
        out["per_k"][k] = dict(alpha_dec=alpha_dec, V=V, W=W.astype(np.float32), R=R, feat=feat)
    rng = np.random.RandomState(seed)
    T10 = order[:, :10]
    out["wc_winners"] = np.take_along_axis(E, T10, 1).ravel().astype(np.float32)
    out["wc_random"] = E[np.arange(nq)[:, None], rng.randint(X.shape[0], size=(nq, 20))].ravel().astype(np.float32)
    # approximate top-10 (what a naive index returns): its standardized errors, for the selection-bias contrast
    A10 = np.argsort(-Sh, 1)[:, :10]
    out["wc_approx_top"] = np.take_along_axis(E, A10, 1).ravel().astype(np.float32)
    return out


def get(name, q):
    p = path(name, q)
    if not os.path.exists(p):
        pickle.dump(compute(name, q), open(p, "wb"))
    return pickle.load(open(p, "rb"))


# ------------------------------------------------------------------ fast derived quantities
def oracle_cost(stk, idx, c, k):
    return np.array([k + np.searchsorted(stk["V"][i], c, side="right") for i in idx])


def worst_miss(stk, idx, c):
    """m*(c) = max{m : #{t : u_t(c) < s_(k+m)} >= m} (Thm. worst-case loss); u_t(c) < s_(k+m) <=> c < W[t, m]."""
    W = stk["W"][idx]                        # (n, k, k)
    cnt = (c < W).sum(1)                     # (n, m): #winners below the (k+m)-th utility
    m = np.arange(1, W.shape[2] + 1)[None, :]
    ok = cnt >= m
    return np.where(ok.any(1), (ok * m).max(1), 0)


if __name__ == "__main__":
    for n in sys.argv[1:] or DATASETS:
        for q in QUANTS:
            get(n, q)
            print("[stats]", n, q, flush=True)
