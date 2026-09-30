"""Calibration analyses on the residual family (fast, from credal.stats caches).

  delta curve      : oracle cost of decision vs coverage calibration, delta in DELTAS
  risk control     : conformal risk control of the worst-case miss fraction m*(c)/k at level eps
  mondrian         : per-bin validity of marginal vs Mondrian (bins of the index-side feature `gap`)
  n sensitivity    : validity mean/std and cost vs calibration-set size n (random subsamples)
  k sensitivity    : decision vs coverage cost ratio and rank-box ratio vs k
  winner's curse   : summary of standardized errors of winners / random documents / approximate top-10
Usage: python -m credal.exp_calib"""
import json
import os
import warnings

import numpy as np

from . import stats as ST
from .credal import conformal_quantile
from .data import DATASETS, RESULTS_DIR

warnings.filterwarnings("ignore", category=RuntimeWarning)
DELTAS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
EPS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
NS = [25, 50, 100, 200, 400]
REPS = 300
K = 10
C_GRID = np.linspace(0, 1.5, 1501)


def crc_level(loss_cal, eps, B=1.0):
    """Conformal risk control (Angelopoulos et al. 2024): smallest c with (n R_n(c) + B) / (n + 1) <= eps.
    loss_cal: (n, len(C_GRID)) non-increasing in c."""
    n = loss_cal.shape[0]
    risk = (loss_cal.sum(0) + B) / (n + 1)
    ok = np.flatnonzero(risk <= eps)
    return C_GRID[ok[0]] if len(ok) else np.inf


def analyse(name, q):
    st = ST.get(name, q)
    cal, te = st["cal"], st["te"]
    out = {}
    s = st["per_k"][K]
    # ---- delta curve
    dc = []
    for d in DELTAS:
        cd = conformal_quantile(s["alpha_dec"][cal], d)
        cc = conformal_quantile(st["alpha_cov"][cal], d)
        Rh = min(conformal_quantile(s["R"][cal], d), st["N"])
        dc.append(dict(delta=d, c_dec=cd, c_cov=cc,
                       cost_dec=float(ST.oracle_cost(s, te, cd, K).mean()),
                       cost_dec_p95=float(np.percentile(ST.oracle_cost(s, te, cd, K), 95)),
                       cost_cov=float(ST.oracle_cost(s, te, cc, K).mean()) if cc <= ST.VMAX else None,
                       depth=float(Rh),
                       valid_dec=float((s["alpha_dec"][te] <= cd).mean()),
                       valid_cov=float((st["alpha_cov"][te] <= cc).mean())))
    out["delta_curve"] = dc
    # ---- risk control on the worst-case miss fraction
    Lcal = np.stack([ST.worst_miss(s, cal, c) / K for c in C_GRID], 1)
    rc = []
    for e in EPS:
        c = crc_level(Lcal, e)
        if not np.isfinite(c):
            continue
        rc.append(dict(eps=e, c=float(c), cost=float(ST.oracle_cost(s, te, c, K).mean()),
                       miss=float((ST.worst_miss(s, te, c) / K).mean()),
                       exact=float((ST.worst_miss(s, te, c) == 0).mean())))
    out["risk_control"] = rc
    # the CRC guarantee is an expectation over calibration and test draws: average over random re-splits
    pool_all = np.concatenate([cal, te])
    Lall = np.stack([ST.worst_miss(s, pool_all, c) / K for c in C_GRID], 1)       # (n, grid)
    rngr = np.random.RandomState(2)
    rs = {e: [] for e in EPS}
    for _ in range(REPS):
        p = rngr.permutation(len(pool_all))
        cc, tt = p[: len(p) // 2], p[len(p) // 2:]
        risk = (Lall[cc].sum(0) + 1) / (len(cc) + 1)
        for e in EPS:
            ok = np.flatnonzero(risk <= e)
            if len(ok):
                rs[e].append(Lall[tt, ok[0]].mean())
    for x in rc:
        x["miss_resplit"] = float(np.mean(rs[x["eps"]])) if rs[x["eps"]] else None
    # ---- Mondrian on tertiles of the index-side gap feature, over repeated random splits
    # (bins from the label-free feature over all queries; per-bin validity is the quantity Mondrian guarantees)
    g = s["feat"]["gap"]
    pool = np.concatenate([cal, te])
    b = np.digitize(g, np.quantile(g[pool], [1 / 3, 2 / 3]))
    rngm = np.random.RandomState(1)
    acc = {j: dict(vm=[], vb=[], cm=[], cb=[]) for j in range(3)}
    for _ in range(REPS):
        p = rngm.permutation(pool)
        cc, tt = p[: len(p) // 2], p[len(p) // 2:]
        cm = conformal_quantile(s["alpha_dec"][cc], 0.1)
        for j in range(3):
            cj, tj = cc[b[cc] == j], tt[b[tt] == j]
            cb = conformal_quantile(s["alpha_dec"][cj], 0.1)
            acc[j]["vm"].append((s["alpha_dec"][tj] <= cm).mean())
            acc[j]["vb"].append((s["alpha_dec"][tj] <= cb).mean())
            acc[j]["cm"].append(ST.oracle_cost(s, tj, cm, K).mean())
            acc[j]["cb"].append(ST.oracle_cost(s, tj, cb, K).mean() if np.isfinite(cb) else np.nan)
    out["mondrian"] = [dict(bin=j, valid_marginal=float(np.mean(v["vm"])), valid_mondrian=float(np.mean(v["vb"])),
                            cost_marginal=float(np.mean(v["cm"])), cost_mondrian=float(np.nanmean(v["cb"])))
                       for j, v in acc.items()]
    # ---- calibration-set size
    rng = np.random.RandomState(0)
    ns = []
    for n in NS:
        if n > len(pool) // 2:
            continue
        cov, cost = [], []
        for _ in range(REPS):
            p = rng.permutation(pool)
            c = conformal_quantile(s["alpha_dec"][p[:n]], 0.1)
            tt = p[len(pool) // 2:]
            cov.append((s["alpha_dec"][tt] <= c).mean())
            cost.append(ST.oracle_cost(s, tt, c, K).mean() if np.isfinite(c) else np.nan)
        ns.append(dict(n=n, cov_mean=float(np.mean(cov)), cov_std=float(np.std(cov)),
                       cost_mean=float(np.nanmean(cost))))
    out["n_sensitivity"] = ns
    # ---- k
    ks = []
    for k, sk in st["per_k"].items():
        cd = conformal_quantile(sk["alpha_dec"][cal], 0.1)
        cc = conformal_quantile(st["alpha_cov"][cal], 0.1)
        Rh = min(conformal_quantile(sk["R"][cal], 0.1), st["N"])
        ks.append(dict(k=int(k), cost_dec=float(ST.oracle_cost(sk, te, cd, k).mean()),
                       cost_cov=float(ST.oracle_cost(sk, te, cc, k).mean()), depth=float(Rh),
                       oracle_depth=float(sk["R"][te].mean())))
    out["k_sensitivity"] = ks
    # ---- winner's curse
    q_ = lambda a: [float(x) for x in np.percentile(a, [5, 25, 50, 75, 95])]
    out["winners_curse"] = dict(winners=q_(st["wc_winners"]), random=q_(st["wc_random"]),
                                approx_top=q_(st["wc_approx_top"]),
                                mean_winners=float(st["wc_winners"].mean()), mean_random=float(st["wc_random"].mean()),
                                mean_approx_top=float(st["wc_approx_top"].mean()))
    return out


def main():
    res = {}
    for name in DATASETS:
        for q in ST.QUANTS:
            if not os.path.exists(ST.path(name, q)):
                continue
            r = res[f"{name}.{q}"] = analyse(name, q)
            d = next(x for x in r["delta_curve"] if x["delta"] == 0.1)
            wc = r["winners_curse"]
            print(f"{name:9s} {q:6s} dec {d['cost_dec']:7.1f} cov {d['cost_cov']} | risk "
                  + " ".join(f"eps{x['eps']}:{x['cost']:.0f}/{x['miss']:.3f}" for x in r["risk_control"][:4])
                  + f" | WC mean win {wc['mean_winners']:+.3f} rand {wc['mean_random']:+.3f}", flush=True)
    json.dump(res, open(os.path.join(RESULTS_DIR, "calib.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
