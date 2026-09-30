"""Actual runs of Algorithm 1 (not oracle costs) at risk-controlled levels (Cor. 4) and at the exact level.

Risk-controlled levels are calibrated on the calibration queries with conformal risk control on the worst-case
miss fraction m*(c)/k (Thm. 3). On test queries we run Algorithm 1 with the calibrated upper bounds (l = sh) and report
the actual number of resolutions, the actual miss fraction of the returned set, and the exact-recovery rate.
Usage: python -m credal.exp_risk_actual [datasets...]"""
import json
import os
import sys
import warnings

import numpy as np

from . import encode
from . import stats as ST
from .credal import conformal_quantile, intervals, refine, topk_hits
from .data import DATASETS, RESULTS_DIR, load, self_mask
from .exp_calib import C_GRID, EPS, K, crc_level
from .index import quantize

warnings.filterwarnings("ignore", category=RuntimeWarning)
QUANTS = ["pq32", "pq96"]


def run(name, q):
    ds = load(name)
    X, Q = encode.get(name, "bge", "docs"), encode.get(name, "bge", "queries")
    Xh, r, _ = quantize(q, X)
    sm = self_mask(ds)
    st = ST.get(name, q)
    s10 = st["per_k"][K]
    cal, te = st["cal"], st["te"]
    Lcal = np.stack([ST.worst_miss(s10, cal, c) / K for c in C_GRID], 1)
    levels = {"exact_0.1": conformal_quantile(s10["alpha_dec"][cal], 0.1)}
    for e in EPS:
        c = crc_level(Lcal, e)
        if np.isfinite(c):
            levels[f"eps_{e}"] = float(c)
    qn = np.linalg.norm(Q, axis=1)
    out = {}
    for lab, c in levels.items():
        cost, miss = [], []
        for i in te:
            s, sh = Q[i] @ X.T, Q[i] @ Xh.T
            if sm[i] >= 0:
                s[sm[i]] = sh[sm[i]] = -1e9
            l, u = intervals(sh, r, qn[i], c, 0.0)
            res = refine(l, u, s, K, "lucb")
            kth = np.partition(s, -K)[-K]
            cost.append(res["n_ref"])
            miss.append(1 - topk_hits(s, res["topk"], K) / K)
        cost, miss = np.array(cost), np.array(miss)
        out[lab] = dict(c=float(c), cost_mean=float(cost.mean()), cost_p95=float(np.percentile(cost, 95)),
                        miss=float(miss.mean()), exact=float((miss == 0).mean()),
                        worst_miss_bound=float((ST.worst_miss(s10, te, c) / K).mean()))
        o = out[lab]
        print(f"{name:9s} {q:5s} {lab:10s} c={c:.3f} cost {o['cost_mean']:7.1f} p95 {o['cost_p95']:6.0f} "
              f"miss {o['miss']:.4f} (worst-case {o['worst_miss_bound']:.4f}) exact {o['exact']:.3f}", flush=True)
    return out


def main():
    path = os.path.join(RESULTS_DIR, "risk_actual.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in sys.argv[1:] or DATASETS:
        for q in QUANTS:
            if f"{name}.{q}" not in res:
                res[f"{name}.{q}"] = run(name, q)
                json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
