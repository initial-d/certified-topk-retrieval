"""Conformal fixed-depth re-ranking: the strongest non-adaptive baseline with the same marginal guarantee.

R(q) = smallest depth R such that the exact top-k is contained (tie-aware) in the approximate top-R.
R_hat = conformal quantile of R(q) on calibration queries; at test time re-rank the approximate top-R_hat.
P(top-k recovered) >= 1 - delta by the same split-conformal argument as Thm. 5.
Compared with credal shrinkage (Alg. 1) at the same delta on the same split: mean / P95 exact re-scorings, recovery.
Usage: python -m credal.exp_confdepth [datasets...]"""
import argparse
import json
import os
import warnings

import numpy as np

from . import encode
from .credal import conformal_quantile
from .data import DATASETS, RESULTS_DIR, load, self_mask
from .index import quantize

warnings.filterwarnings("ignore", category=RuntimeWarning)
K = 10
DELTA = 0.1
SPLITS = 500


def required_depth(s_row, sh_row, k=K):
    """Smallest R with k documents of exact score >= s_(k) among the approximate top-R."""
    from .credal import required_depth as _rd
    return _rd(s_row, sh_row, k)


def run(name, qname, model="bge"):
    ds = load(name)
    X, Q = encode.get(name, model, "docs"), encode.get(name, model, "queries")
    Xh, _, _ = quantize(qname, X)
    S, Sh = Q @ X.T, Q @ Xh.T
    sm = self_mask(ds)
    rows = np.flatnonzero(sm >= 0)
    S[rows, sm[rows]] = -1e9
    Sh[rows, sm[rows]] = -1e9
    Rq = np.array([required_depth(S[i], Sh[i]) for i in range(len(Q))])
    cal, te = ds.calib_idx, ds.test_idx
    R_hat = conformal_quantile(Rq[cal], DELTA)
    R_hat = int(min(R_hat, X.shape[0]))
    cert = json.load(open(os.path.join(RESULTS_DIR, "certify", f"{name}.{model}.{qname}.json")))
    dec = cert["credal"]["decision"]["lucb"]
    n_ref = np.array(dec["n_ref_all"])
    # validity of the conformal depth over random re-splits (the credal counterpart is in coverage.json)
    rng = np.random.RandomState(0)
    covs, depths = [], []
    for _ in range(SPLITS):
        p = rng.permutation(len(Q))
        c = conformal_quantile(Rq[p[: len(Q) // 2]], DELTA)
        covs.append((Rq[p[len(Q) // 2:]] <= c).mean())
        depths.append(min(c, X.shape[0]))
    return dict(
        dataset=name, quant=qname, R_hat=R_hat,
        confdepth=dict(cost_mean=float(R_hat), cost_p95=float(R_hat), perfect=float((Rq[te] <= R_hat).mean()),
                       cov_resplit=(float(np.mean(covs)), float(np.std(covs))),
                       depth_resplit=(float(np.mean(depths)), float(np.std(depths)))),
        credal=dict(cost_mean=float(n_ref.mean()), cost_p95=float(np.percentile(n_ref, 95)),
                    cost_median=float(np.median(n_ref)), perfect=dec["perfect"]),
        oracle_depth_mean=float(Rq[te].mean()),   # per-query oracle depth (unattainable without the exact scores)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    ap.add_argument("--quant", default="pq16,pq32,pq96,opq32,sq4,bin")
    args = ap.parse_args()
    out = {}
    path = os.path.join(RESULTS_DIR, "confdepth.json")
    if os.path.exists(path):
        out = json.load(open(path))
    for name in args.datasets:
        for q in args.quant.split(","):
            key = f"{name}.{q}"
            if key in out:
                continue
            out[key] = r = run(name, q)
            json.dump(out, open(path, "w"), indent=1)
            print(f"{key:16s} conformal depth R_hat={r['R_hat']:5d} perfect={r['confdepth']['perfect']:.3f} | "
                  f"credal mean={r['credal']['cost_mean']:7.1f} p95={r['credal']['cost_p95']:7.0f} "
                  f"perfect={r['credal']['perfect']:.3f} | oracle depth mean={r['oracle_depth_mean']:.1f}", flush=True)


if __name__ == "__main__":
    main()
