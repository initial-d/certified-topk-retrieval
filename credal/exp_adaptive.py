"""Query-adaptive decision calibration (normalized conformal) and hybrid stopping, versus conformal fixed depth.

Cost of a verify-before-certify run with per-query constant c(q) is evaluated by the oracle cost
|A(q)| = k + #{j not in T : sh_j + c(q) ||q|| r_j >= s_(k)}  (Prop. 6; Alg. 1 is within 1.4% of it, Table 4).
Normalized nonconformity: alpha_g(q) = alpha(q) / g(q) with g computed from the approximate index only, so the
split-conformal guarantee of Thm. 5 is unchanged (c(q) = c_hat * g(q)).
Hybrid: run Alg. 1 with delta/2 and stop at the conformal depth R_hat(delta/2) if reached first; error <= delta
(union bound). The variant is chosen by mean cost on an inner split of the CALIBRATION queries only.
Usage: python -m credal.exp_adaptive [datasets...]"""
import argparse
import json
import os
import warnings

import numpy as np

from . import encode
from .credal import conformal_quantile, decision_nonconformity
from .data import DATASETS, RESULTS_DIR, load, self_mask
from .exp_confdepth import required_depth
from .index import quantize

warnings.filterwarnings("ignore", category=RuntimeWarning)
K = 10
DELTA = 0.1


def scales(sh_row, r):
    """Candidate query scales from approximate information only (no exact scores)."""
    top = np.sort(np.partition(sh_row, -100)[-100:])[::-1]
    o = np.argpartition(-sh_row, K)[:K]
    return {
        "none": 1.0,
        "gap10_100": max(top[K - 1] - top[99], 1e-6),      # how far the k-th score sits above the bulk
        "std100": max(top.std(), 1e-6),
        "inv_gap1_10": 1.0 / max(top[0] - top[K - 1], 1e-6),
        "r_top10": float(r[o].mean()),
    }


def cert_time_rerank_order(s_row, sh_row, r, qn, c, max_m):
    """Re-rank in decreasing approximate score; after m re-scorings (the approximate top-m) check the
    verify-before-certify condition with l = sh (unrefined), u = sh + c||q||r. Returns the first certifying m
    (or max_m + 1 if none within max_m). Certified at m iff the k-th largest exact score among the refined prefix,
    L, exceeds every unrefined lower bound (sh_(m+1)) and every unrefined upper bound."""
    order = np.argsort(-sh_row, kind="stable")[: max_m + 1]
    u = sh_row + c * qn * r
    # max upper bound over the suffix (all documents not in the prefix); only the first max_m+1 prefixes matter
    mask = np.ones(len(sh_row), bool)
    import heapq
    heap = []
    u_sorted_idx = np.argsort(-u, kind="stable")
    ptr = 0
    for m in range(1, max_m + 1):
        d = order[m - 1]
        mask[d] = False
        v = s_row[d]
        if len(heap) < K:
            heapq.heappush(heap, v)
        elif v > heap[0]:
            heapq.heapreplace(heap, v)
        if m < K:
            continue
        L = heap[0]
        nxt = sh_row[order[m]] if m < len(order) else -np.inf
        while ptr < len(u_sorted_idx) and not mask[u_sorted_idx[ptr]]:
            ptr += 1
        umax = u[u_sorted_idx[ptr]] if ptr < len(u_sorted_idx) else -np.inf
        if L > nxt and umax < L:
            return m
    return max_m + 1


def oracle_cost(s_row, sh_row, r, qn, c):
    kth = np.partition(s_row, -K)[-K]
    notT = s_row < kth - 1e-6
    return int(K + (notT & (sh_row + c * qn * r >= kth)).sum())


def run(name, qname, model="bge", seed=0):
    ds = load(name)
    X, Q = encode.get(name, model, "docs"), encode.get(name, model, "queries")
    Xh, r, _ = quantize(qname, X)
    S, Sh = Q @ X.T, Q @ Xh.T
    sm = self_mask(ds)
    rows = np.flatnonzero(sm >= 0)
    S[rows, sm[rows]] = -1e9
    Sh[rows, sm[rows]] = -1e9
    qn = np.linalg.norm(Q, axis=1)
    n = len(Q)
    alpha = np.array([decision_nonconformity(S[i], Sh[i], r, qn[i], K) for i in range(n)])
    G = [scales(Sh[i], r) for i in range(n)]
    names = list(G[0])
    g = {v: np.array([G[i][v] for i in range(n)]) for v in names}
    Rq = np.array([required_depth(S[i], Sh[i]) for i in range(n)])
    cal, te = ds.calib_idx, ds.test_idx

    def evaluate(variant, idx_fit, idx_eval, delta):
        c_hat = conformal_quantile(alpha[idx_fit] / g[variant][idx_fit], delta)
        cost = np.array([oracle_cost(S[i], Sh[i], r, qn[i], c_hat * g[variant][i]) for i in idx_eval])
        ok = alpha[idx_eval] <= c_hat * g[variant][idx_eval]
        return cost, ok

    # variant selection on an inner split of the calibration queries
    rng = np.random.RandomState(seed)
    p = rng.permutation(cal)
    fit, val = p[: len(p) // 2], p[len(p) // 2:]
    inner = {v: float(evaluate(v, fit, val, DELTA)[0].mean()) for v in names}
    chosen = min(inner, key=inner.get)

    out = dict(dataset=name, quant=qname, inner_mean_cost=inner, chosen=chosen, test={})
    R_hat = min(conformal_quantile(Rq[cal], DELTA), X.shape[0])
    out["test"]["confdepth"] = dict(mean=float(R_hat), p95=float(R_hat), recovered=float((Rq[te] <= R_hat).mean()))
    for v in names:
        cost, ok = evaluate(v, cal, te, DELTA)
        out["test"][v] = dict(mean=float(cost.mean()), p95=float(np.percentile(cost, 95)), valid=float(ok.mean()))
    # certified early stopping of standard re-ranking (approximate-score order), alone and capped at the
    # conformal depth (hybrid). The returned set is the exact top-k of the re-scored prefix, which is correct
    # iff that prefix contains the exact top-k, i.e. iff R(q) <= stopping depth.
    def early_stop(delta, cap):
        c_hat = conformal_quantile(alpha[cal], delta)
        m = np.array([cert_time_rerank_order(S[i], Sh[i], r, qn[i], c_hat, cap) for i in te])
        stop = np.minimum(m, cap)
        return stop, (Rq[te] <= stop)
    N = X.shape[0]
    stop, ok = early_stop(DELTA, N)
    out["test"]["rerank_cert"] = dict(mean=float(stop.mean()), p95=float(np.percentile(stop, 95)), valid=float(ok.mean()))
    R2 = int(min(conformal_quantile(Rq[cal], DELTA / 2), N))
    stop, ok = early_stop(DELTA / 2, R2)
    out["test"]["hybrid"] = dict(mean=float(stop.mean()), p95=float(np.percentile(stop, 95)), valid=float(ok.mean()),
                                 R_hat_half=R2)
    out["test"]["oracle_depth"] = dict(mean=float(Rq[te].mean()), p95=float(np.percentile(Rq[te], 95)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    ap.add_argument("--quant", default="pq16,pq32,pq96,opq32,sq4,bin")
    args = ap.parse_args()
    path = os.path.join(RESULTS_DIR, "adaptive.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name in args.datasets:
        for q in args.quant.split(","):
            key = f"{name}.{q}"
            if key in out:
                continue
            out[key] = r = run(name, q)
            json.dump(out, open(path, "w"), indent=1)
            t = r["test"]
            print(f"{key:15s} confdepth={t['confdepth']['mean']:6.0f} v={t['confdepth']['recovered']:.2f} "
                  f"alg1={t['none']['mean']:7.1f}/{t['none']['p95']:6.0f} "
                  f"rerank_cert={t['rerank_cert']['mean']:7.1f}/{t['rerank_cert']['p95']:6.0f} v={t['rerank_cert']['valid']:.2f} "
                  f"hybrid={t['hybrid']['mean']:7.1f}/{t['hybrid']['p95']:6.0f} v={t['hybrid']['valid']:.2f} "
                  f"oracle={t['oracle_depth']['mean']:6.1f}", flush=True)


if __name__ == "__main__":
    main()
