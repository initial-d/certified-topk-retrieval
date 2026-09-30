"""E1: index-layer credal sets — validity, certification, and adaptive shrinkage cost.

Usage: python -m credal.exp_certify [datasets...] [--models bge,e5] [--quant pq16,pq32]
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

from . import encode
from .credal import (conformal_quantile, decision_nonconformity, intervals, nonconformity, oracle_lower_bound,
                     refine, undominated)
from .data import DATASETS, RESULTS_DIR, load, self_mask
from .index import CODE_BYTES, quantize

K = 10
DELTAS = [0.01, 0.05, 0.1, 0.2]
MAIN_DELTA = 0.1
LAM = 0.0  # asymmetric intervals [sh, sh + c r]; symmetric (lam = 1) reported as ablation
STRATEGIES = ["lucb", "score", "widest", "random"]   # "lucb" is Algorithm 1; "score" = upper-bound order
RERANK_DEPTHS = [10, 20, 50, 100, 200, 500]
NEG = -1e9
# numpy 2.0 + Accelerate emits spurious FP warnings from matmul; values are checked finite below
warnings.filterwarnings("ignore", category=RuntimeWarning)


def hits(s_row, got, kth):
    """Tie-aware overlap with the exact top-k: returned documents whose exact score reaches s_(k).
    (Corpora such as ArguAna contain duplicate documents, so the exact top-k set is defined up to ties.)"""
    from .credal import topk_hits
    return topk_hits(s_row, got, K)


def run(ds_name, model, qname, seed=0):
    ds = load(ds_name)
    X = encode.get(ds_name, model, "docs")
    Q = encode.get(ds_name, model, "queries")
    t0 = time.time()
    Xh, r, _ = quantize(qname, X)
    t_q = time.time() - t0
    S = Q @ X.T
    Sh = Q @ Xh.T
    selfm = self_mask(ds)
    mask = np.zeros(S.shape, bool)
    rows = np.flatnonzero(selfm >= 0)
    mask[rows, selfm[rows]] = True
    S[mask] = NEG
    Sh[mask] = NEG
    qn = np.linalg.norm(Q, axis=1)
    assert np.isfinite(S).all() and np.isfinite(Sh).all()
    alpha = nonconformity(S, Sh, r, qn, mask)
    alpha_dec = np.array([decision_nonconformity(S[i], Sh[i], r, qn[i], K) for i in range(len(Q))])
    cal, te = ds.calib_idx, ds.test_idx

    out = dict(dataset=ds_name, model=model, quant=qname, bytes=CODE_BYTES[qname], N=len(X),
               n_cal=len(cal), n_test=len(te), train_sec=t_q, mean_r=float(r.mean()))
    out["alpha_dec"], out["alpha_sim"] = alpha_dec.tolist(), alpha.tolist()
    # --- validity: simultaneous interval coverage vs decision condition (*) coverage
    out["coverage"], out["coverage_dec"] = {}, {}
    for delta in DELTAS:
        c = conformal_quantile(alpha[cal], delta)
        out["coverage"][str(delta)] = dict(c=c, test_cov=float((alpha[te] <= c).mean()))
        c = conformal_quantile(alpha_dec[cal], delta)
        out["coverage_dec"][str(delta)] = dict(c=c, test_cov=float((alpha_dec[te] <= c).mean()))
    out["alpha_max_test"] = float(alpha[te].max())

    # exact top-k and naive (un-reranked) quantized top-k
    true_top = np.argsort(-S[te], 1)[:, :K]
    naive_top = np.argsort(-Sh[te], 1)[:, :K]
    kth = np.take_along_axis(S[te], true_top[:, -1:], 1)[:, 0]
    out["naive_exact_recall"] = float(np.mean([hits(S[qi], b, kth[i]) / K for i, (qi, b) in enumerate(zip(te, naive_top))]))

    # --- fixed-depth re-ranking baselines
    out["rerank"] = {}
    order_h = np.argsort(-Sh[te], 1)
    for R in RERANK_DEPTHS:
        cand = order_h[:, :R]
        sc = np.take_along_axis(S[te], cand, 1)
        got = np.take_along_axis(cand, np.argsort(-sc, 1)[:, :K], 1)
        rec = np.array([hits(S[qi], b, kth[i]) / K for i, (qi, b) in enumerate(zip(te, got))])
        out["rerank"][R] = dict(exact_recall=float(rec.mean()), perfect=float((rec == 1).mean()))

    # --- credal certification + shrinkage. Refinement is run for the decision-calibrated sets; for the
    # (much wider) simultaneous / Cauchy-Schwarz boxes we report the oracle cost, which lucb matches closely.
    rng = np.random.RandomState(seed)
    out["credal"] = {}
    c_dec = out["coverage_dec"][str(MAIN_DELTA)]["c"]
    settings = (("decision", c_dec, alpha_dec, LAM, STRATEGIES),
                ("decision_sym", c_dec, alpha_dec, 1.0, ["lucb"]),
                ("simultaneous", out["coverage"][str(MAIN_DELTA)]["c"], alpha, 1.0, []),
                ("cauchy_schwarz", 1.0, alpha, 1.0, []))
    for label, c, alph, lam, strategies in settings:
        rec = dict(U0=[], cert0=[], lower_bound=[], valid=[], sound=[])
        per = {s: dict(n_ref=[], certified=[], exact_recall=[], sec=[]) for s in strategies}
        for i, qi in enumerate(te):
            l, u = intervals(Sh[qi], r, qn[qi], c, lam)
            l[mask[qi]] = NEG
            u[mask[qi]] = NEG
            U0 = undominated(l, u, K)
            rec["U0"].append(len(U0))
            rec["cert0"].append(len(U0) == K)
            rec["valid"].append(bool(alph[qi] <= c))
            rec["sound"].append(hits(S[qi], U0, kth[i]) >= K)
            rec["lower_bound"].append(oracle_lower_bound(u, S[qi], K))
            for strat in strategies:
                t1 = time.time()
                res = refine(l, u, S[qi], K, strat, rng)
                per[strat]["sec"].append(time.time() - t1)
                per[strat]["n_ref"].append(res["n_ref"])
                per[strat]["certified"].append(res["certified"])
                per[strat]["exact_recall"].append(hits(S[qi], res["topk"], kth[i]) / K)
        summ = dict(
            c=c,
            U0_mean=float(np.mean(rec["U0"])), U0_median=float(np.median(rec["U0"])),
            U0_p90=float(np.percentile(rec["U0"], 90)),
            cert_without_refine=float(np.mean(rec["cert0"])),
            valid_rate=float(np.mean(rec["valid"])),
            sound_rate=float(np.mean(rec["sound"])),
            lower_bound_mean=float(np.mean(rec["lower_bound"])),
            U0_all=rec["U0"], lower_bound_all=rec["lower_bound"],
        )
        for strat in strategies:
            p = per[strat]
            summ[strat] = dict(n_ref_mean=float(np.mean(p["n_ref"])), n_ref_median=float(np.median(p["n_ref"])),
                               n_ref_p90=float(np.percentile(p["n_ref"], 90)),
                               certified=float(np.mean(p["certified"])),
                               exact_recall=float(np.mean(p["exact_recall"])),
                               perfect=float(np.mean(np.array(p["exact_recall"]) == 1)),
                               ms=1000 * float(np.mean(p["sec"])))
            summ[strat]["n_ref_all"] = p["n_ref"]
        out["credal"][label] = summ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    ap.add_argument("--models", default="bge")
    ap.add_argument("--quant", default="pq16,pq32,pq96,opq32,sq4,bin")
    args = ap.parse_args()
    os.makedirs(os.path.join(RESULTS_DIR, "certify"), exist_ok=True)
    for ds in args.datasets:
        for m in args.models.split(","):
            for qn in args.quant.split(","):
                path = os.path.join(RESULTS_DIR, "certify", f"{ds}.{m}.{qn}.json")
                if os.path.exists(path):
                    continue
                t = time.time()
                res = run(ds, m, qn)
                json.dump(res, open(path, "w"))
                cr = res["credal"]["decision"]
                print(f"[E1] {ds:9s} {m:6s} {qn:6s} cov@.1={res['coverage_dec']['0.1']['test_cov']:.3f} "
                      f"simLB={res['credal']['simultaneous']['lower_bound_mean']:.0f} "
                      f"symU0={res['credal']['decision_sym']['U0_mean']:.0f} "
                      f"c={cr['c']:.3f} U0={cr['U0_mean']:.1f} cert0={cr['cert_without_refine']:.2f} "
                      f"lucb={cr['lucb']['n_ref_mean']:.1f} upper={cr['score']['n_ref_mean']:.1f} "
                      f"LB={cr['lower_bound_mean']:.1f} rec={cr['lucb']['exact_recall']:.3f} "
                      f"naive={res['naive_exact_recall']:.3f} ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
