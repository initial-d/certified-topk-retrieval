"""Certified top-k of a cross-encoder in a bi-encoder -> cross-encoder cascade (second application).

Imprecision model (anchored interval credal set). Resolve the bi-encoder top-k first ("anchors", which a certificate
must verify anyway). For the other pool documents,
    centre f_q(d) = a_q + b * sh_d,   u_d(c) = f_q(d) + c * sigma_q,   l_d = f_q(d),
with a global slope b fitted on a FIT split, and a_q, sigma_q from the query's own anchors (intercept, residual RMS).
The whole construction is replayed on calibration queries, so the conformal guarantee of Thm. 5 applies with the
decision nonconformity alpha(q) = max_{t in T, t not anchor} (s_(k+1) - f_q(t)) / sigma_q  v 0.
Competitors with the same guarantee: simultaneous calibration of the same intervals and conformal fixed depth.
Cost = cross-encoder calls (anchors included).
Usage: python -m credal.exp_cascade [datasets...]"""
import json
import os
import sys
import warnings

import numpy as np

from .credal import conformal_quantile, refine, required_depth, topk_correct
from .data import CACHE_DIR, DATASETS, RESULTS_DIR, load

warnings.filterwarnings("ignore", category=RuntimeWarning)
K = 10
DELTA = 0.1
SPLITS = 500


def anchored(sh, s, b, floor_frac=0.1):
    """Per-query intercept and scale from the anchors (bi-encoder top-k = first K pool positions)."""
    A = slice(0, K)
    a = float(np.mean(s[A] - b * sh[A]))
    res = s[A] - (a + b * sh[A])
    sig = max(float(np.sqrt(np.mean(res ** 2))), floor_frac * float(np.std(s[A])), 1e-3)
    return a + b * sh, sig


def per_query_slope(sh, s):
    A = slice(0, K)
    bb = np.polyfit(sh[A], s[A], 1)[0] if np.std(sh[A]) > 0 else 0.0
    return max(bb, 0.0)


def query_stats(sh, s, b):
    f, sig = anchored(sh, s, b)
    order = np.argsort(-s)
    T, s_k, s_k1 = order[:K], s[order[K - 1]], s[order[K]]
    free = np.arange(len(s)) >= K                     # not an anchor
    Tf = T[free[T]]
    alpha_dec = max(float(((s_k1 - f[Tf]) / sig).max()) if len(Tf) else 0.0, 0.0)
    alpha_sim = float((np.abs(s[free] - f[free]) / sig).max())
    R = required_depth(s, -np.arange(len(s), dtype=float), K)            # pool is sorted by sh
    return dict(f=f, sig=sig, T=T, s_k=s_k, alpha_dec=alpha_dec, alpha_sim=alpha_sim, R=R)


def oracle_cost(st, c, s):
    u = st["f"] + c * st["sig"]
    need = np.zeros(len(s), bool)
    need[:K] = True                      # anchors
    need[st["T"]] = True
    need |= u >= st["s_k"]
    return int(need.sum())


def run_alg1(st, c, s):
    """Alg. 1 with the anchors already resolved; returns (CE calls, correct)."""
    l = st["f"].copy()
    u = st["f"] + c * st["sig"]
    res = refine(l, u, s, K, "lucb", resolved=np.arange(K))
    # refine() counts refinements of unresolved documents only; anchors cost K calls up front
    return K + res["n_ref"], topk_correct(s, res["topk"], K)


def run(name, seed=0, slope="global", stage="ce"):
    z = np.load(os.path.join(CACHE_DIR, f"{name}.cascade{'' if stage == 'ce' else '_' + stage}.npz"))
    SH, SS = z["sh"], z["s"]
    ds = load(name)
    rng = np.random.RandomState(seed)
    cal_all = rng.permutation(ds.calib_idx)
    fit, cal, te = np.sort(cal_all[: len(cal_all) // 2]), np.sort(cal_all[len(cal_all) // 2:]), ds.test_idx
    # global slope: pooled within-query regression (query-centred) on the fit split
    xs = np.concatenate([SH[i] - SH[i].mean() for i in fit])
    ys = np.concatenate([SS[i] - SS[i].mean() for i in fit])
    b = max(float(xs @ ys / (xs @ xs)), 0.0)
    st = {}
    for i in np.concatenate([cal, te]):
        bi = per_query_slope(SH[i], SS[i]) if slope == "per_query" else b
        st[i] = query_stats(SH[i], SS[i], bi)
    a_dec = np.array([st[i]["alpha_dec"] for i in cal])
    a_sim = np.array([st[i]["alpha_sim"] for i in cal])
    R_cal = np.array([st[i]["R"] for i in cal])
    c_dec, c_sim = conformal_quantile(a_dec, DELTA), conformal_quantile(a_sim, DELTA)
    R_hat = int(min(conformal_quantile(R_cal, DELTA), SH.shape[1]))
    out = dict(dataset=name, stage=stage, pool=int(SH.shape[1]), slope=slope, b=b, n_fit=len(fit), n_cal=len(cal), n_test=len(te),
               c_dec=c_dec, c_sim=c_sim, R_hat=R_hat)
    cost_dec = np.array([oracle_cost(st[i], c_dec, SS[i]) for i in te])
    cost_sim = np.array([oracle_cost(st[i], c_sim, SS[i]) if np.isfinite(c_sim) else SH.shape[1] for i in te])
    alg = [run_alg1(st[i], c_dec, SS[i]) for i in te]
    R_te = np.array([st[i]["R"] for i in te])
    out["decision"] = dict(oracle_mean=float(cost_dec.mean()), oracle_p95=float(np.percentile(cost_dec, 95)),
                           alg1_mean=float(np.mean([a[0] for a in alg])),
                           alg1_p95=float(np.percentile([a[0] for a in alg], 95)),
                           correct=float(np.mean([a[1] for a in alg])),
                           valid=float(np.mean([st[i]["alpha_dec"] <= c_dec for i in te])))
    out["simultaneous"] = dict(oracle_mean=float(cost_sim.mean()), oracle_p95=float(np.percentile(cost_sim, 95)),
                               valid=float(np.mean([st[i]["alpha_sim"] <= c_sim for i in te])))
    out["confdepth"] = dict(cost=R_hat, correct=float((R_te <= R_hat).mean()))
    out["oracle_depth_mean"] = float(R_te.mean())
    out["full_rerank"] = SH.shape[1]
    # fixed depths without calibration (reference curve)
    out["fixed_curve"] = {int(R): float((R_te <= R).mean())
                          for R in (10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 1000) if R <= SH.shape[1]}
    out["cost_dec_all"] = [int(a[0]) for a in alg]
    # delta curve (oracle costs) for decision vs coverage calibration and rank boxes
    out["delta_curve"] = []
    for dl in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
        cd_, cc_ = conformal_quantile(a_dec, dl), conformal_quantile(a_sim, dl)
        Rh_ = int(min(conformal_quantile(R_cal, dl), SH.shape[1]))
        out["delta_curve"].append(dict(
            delta=dl, cost_dec=float(np.mean([oracle_cost(st[i], cd_, SS[i]) for i in te])),
            cost_cov=float(np.mean([oracle_cost(st[i], cc_, SS[i]) if np.isfinite(cc_) else SH.shape[1] for i in te])),
            depth=Rh_))
    # validity over random re-splits of cal+test (fit split and slope fixed)
    pool_q = np.concatenate([cal, te])
    ad = np.array([st[i]["alpha_dec"] for i in pool_q])
    Rq = np.array([st[i]["R"] for i in pool_q])
    cov_d, cov_r = [], []
    for _ in range(SPLITS):
        p = rng.permutation(len(pool_q))
        h = len(p) // 2
        cov_d.append((ad[p[h:]] <= conformal_quantile(ad[p[:h]], DELTA)).mean())
        cov_r.append((Rq[p[h:]] <= conformal_quantile(Rq[p[:h]], DELTA)).mean())
    out["resplit"] = dict(decision=(float(np.mean(cov_d)), float(np.std(cov_d))),
                          confdepth=(float(np.mean(cov_r)), float(np.std(cov_r))))
    return out


def main():
    names = sys.argv[1:] or DATASETS
    path = os.path.join(RESULTS_DIR, "cascade.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for n in names:
        for stage in ("ce", "bgebase"):
          if not os.path.exists(os.path.join(CACHE_DIR, f"{n}.cascade{'' if stage == 'ce' else '_' + stage}.npz")):
            continue
          for slope in ("global",):
            r = res[f"{n}.{stage}.{slope}"] = run(n, slope=slope, stage=stage)
            d = r["decision"]
            print(f"{n:9s} {stage:8s} decision: calls {d['alg1_mean']:5.1f} (oracle {d['oracle_mean']:5.1f}, "
                  f"p95 {d['alg1_p95']:5.0f}) correct {d['correct']:.3f} | simult: {r['simultaneous']['oracle_mean']:5.1f} "
                  f"| conf depth {r['confdepth']['cost']:3d} correct {r['confdepth']['correct']:.3f} | "
                  f"oracle depth {r['oracle_depth_mean']:5.1f} | resplit dec {r['resplit']['decision'][0]:.3f} "
                  f"depth {r['resplit']['confdepth'][0]:.3f}", flush=True)
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
