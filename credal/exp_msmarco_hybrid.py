"""Truncated residual boxes at MS MARCO scale, with ACTUAL runs of Algorithm 1 (from passD: approximate top-M per query
with exact scores). Same protocol as exp_hybrid: Rbar chosen on a tune half of the calibration queries, theta calibrated on
the other half at delta = 0.1, test queries untouched. Compared with rank boxes (conformal depth on the full calibration half)
and full residual boxes (oracle cost from passB/report).
Usage: python -m credal.exp_msmarco_hybrid [quants...]"""
import json
import os
import sys
import warnings

import numpy as np

from .credal import conformal_quantile, refine
from .data import RESULTS_DIR

warnings.filterwarnings("ignore", category=RuntimeWarning)
K = 10
DELTA = 0.1
DELTA_R = [0.005, 0.01, 0.02, 0.025]
OUT = os.path.join(RESULTS_DIR, "msmarco")


def load(q):
    A = np.load(os.path.join(OUT, "passA.npz"))
    D = np.load(os.path.join(OUT, f"passD.{q}.npz"))
    top_s, top_i = A["top_s"], A["top_i"]
    idx, sh, s, r, qn = D["idx"], D["sh"], D["s"], D["r"], D["qn"]
    nq, M = idx.shape
    kth, k1 = top_s[:, K - 1], top_s[:, K]
    win = np.zeros((nq, M), bool)
    for i in range(nq):
        win[i] = np.isin(idx[i], top_i[i, :K]) | (s[i] >= kth[i] - 1e-6)
    # smallest depth containing all winners (> M if some winner is outside the kept prefix)
    cnt = win.cumsum(1)
    # tie-aware sufficient depth: the prefix must contain every option strictly above s_(k) and enough ties
    R = np.full(nq, M + 1)
    for i in range(nq):
        n_above = int((top_s[i] > kth[i] + 1e-6).sum())
        ok = (np.cumsum(s[i] > kth[i] + 1e-6) == n_above) & (np.cumsum(np.abs(s[i] - kth[i]) <= 1e-6) >= K - n_above)
        if ok.any():
            R[i] = int(np.argmax(ok) + 1)
    w = qn[:, None] * r
    alpha = np.maximum(np.where(win, (k1[:, None] - sh) / (w + 1e-8), -np.inf).max(1), 0)
    alpha = np.where(cnt[:, -1] >= K, alpha, np.inf)      # a winner beyond M: treat as not coverable by truncation
    V = np.where(win, np.inf, (kth[:, None] - sh) / (w + 1e-8))
    return dict(idx=idx, sh=sh, s=s, w=w, win=win, R=R, alpha=alpha, V=V, kth=kth, cal=D["cal"], te=D["te"], M=M,
                top_s=top_s, top_i=top_i)


def oracle(d, i, Rbar, th):
    return K + int((d["V"][i, :Rbar] <= th).sum())


def run(q, seed=0):
    d = load(q)
    rng = np.random.RandomState(seed)
    cal = rng.permutation(d["cal"])
    tune, cal2, te = cal[: len(cal) // 2], cal[len(cal) // 2:], d["te"]
    best = None
    for dr in DELTA_R:
        Rb = conformal_quantile(d["R"][tune], dr)
        if not np.isfinite(Rb) or Rb > d["M"]:
            continue
        Rb = int(Rb)
        th = conformal_quantile(np.where(d["R"][tune] <= Rb, d["alpha"][tune], np.inf), DELTA)
        if not np.isfinite(th):
            continue
        c = np.mean([oracle(d, i, Rb, th) for i in tune])
        if best is None or c < best[0]:
            best = (c, dr, Rb)
    _, dr, Rbar = best
    th = conformal_quantile(np.where(d["R"][cal2] <= Rbar, d["alpha"][cal2], np.inf), DELTA)
    costs, exact = [], []
    for i in te:
        l = d["sh"][i, :Rbar].astype(np.float64)
        u = l + th * d["w"][i, :Rbar]
        res = refine(l, u, d["s"][i, :Rbar].astype(np.float64), K, "lucb")
        costs.append(res["n_ref"])
        # tie-aware, against the whole corpus: min returned score >= largest exact score of any option not returned
        ret = set(d["idx"][i, res["topk"]].tolist())
        outside = max(v for v, j in zip(d["top_s"][i], d["top_i"][i]) if j not in ret)
        exact.append(bool(d["s"][i, res["topk"]].min() >= outside - 1e-6))
    costs = np.array(costs)
    Rhat = conformal_quantile(d["R"][d["cal"]], DELTA)
    rep = json.load(open(os.path.join(OUT, "report.json")))[q]["0.1"]
    out = dict(quant=q, delta_R=dr, Rbar=Rbar, theta=float(th),
               hybrid=dict(mean=float(costs.mean()), p95=float(np.percentile(costs, 95)), max=int(costs.max()),
                           exact=float(np.mean(exact)),
                           valid=float(np.mean(np.where(d["R"][te] <= Rbar, d["alpha"][te], np.inf) <= th)),
                           oracle_mean=float(np.mean([oracle(d, i, Rbar, th) for i in te]))),
               rank=dict(Rhat=float(Rhat), exact=float(np.mean(d["R"][te] <= Rhat))),
               residual_oracle=dict(mean=rep["cost_dec_mean"], p95=rep["cost_dec_p95"]))
    h = out["hybrid"]
    print(f"{q:5s} Rbar={Rbar} (dR={dr}) th={th:.3f} | hybrid actual mean {h['mean']:.1f} p95 {h['p95']:.0f} "
          f"exact {h['exact']:.3f} valid {h['valid']:.3f} | rank {Rhat:.0f} exact {out['rank']['exact']:.3f} | "
          f"residual oracle {rep['cost_dec_mean']:.0f} | hybrid/rank {h['mean'] / Rhat:.2f}", flush=True)
    return out


def main():
    path = os.path.join(OUT, "hybrid.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for q in sys.argv[1:] or ["pq32", "bin", "pq96", "sq4"]:
        if os.path.exists(os.path.join(OUT, f"passD.{q}.npz")):
            res[q] = run(q)
            json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
