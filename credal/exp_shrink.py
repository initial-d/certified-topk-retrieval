"""E3: evidence-driven credal shrinkage.

Credal set over calibration parameters theta = (w, tau_1..tau_M) of the mixture retrieval model
    p_theta(d | q) = sum_m w_m softmax(s^m(q, .) / tau_m)
given n labelled calibration queries:  Theta_n = { theta : l_n(theta) >= max l_n - chi2_{dof, 1-a} / 2 }
(likelihood-based / profile-likelihood credal set).  K_n(q) = conv{ p_theta(.|q) : theta in Theta_n }.
We track how K_n(q) contracts on held-out queries as n grows.
Usage: python -m credal.exp_shrink [datasets...]   (needs results/uq/<name>.npz from exp_uq)
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

from . import encode
from .credal import upper_entropy_hull
from .data import DATASETS, RESULTS_DIR

MODELS = list(encode.ENSEMBLE)
GRID = np.logspace(-3, 0, 31)
J = 4000
NS = [5, 10, 20, 50, 100, 200, 400]
REPS = 5
MAX_EXT = 120     # extreme points kept per credal set (random subsample of the retained thetas)
N_TEST = 150
LEVEL = 0.95


def softmax_grid(S):
    """S: (n,) -> (G, n) softmax at every grid temperature."""
    z = S[None, :] / GRID[:, None]
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def run(name, seed=0):
    z = np.load(os.path.join(RESULTS_DIR, "uq", f"{name}.npz"), allow_pickle=True)
    rel_pool, cal, te = z["rel_pool"], z["cal"], z["te"]
    S = {m: z[f"S_{m}"] for m in MODELS}
    M = len(MODELS)
    has_rel = np.array([rp.sum() > 0 for rp in rel_pool])
    cal = cal[has_rel[cal]]
    rng = np.random.RandomState(seed)
    te = rng.permutation(te[has_rel[te]])[:N_TEST]

    # rho[m, g, i] = P_{m, tau_g}(relevant | q_i)
    def rho_of(idx):
        out = np.zeros((M, len(GRID), len(idx)))
        for j, i in enumerate(idx):
            mask = rel_pool[i] > 0
            for a, m in enumerate(MODELS):
                out[a, :, j] = softmax_grid(S[m][i])[:, mask].sum(1)
        return out
    rho_cal, rho_te = rho_of(cal), rho_of(te)
    SM_te = [np.stack([softmax_grid(S[m][i]) for m in MODELS]) for i in te]   # (M, G, n_pool) per query

    W = rng.dirichlet(np.ones(M), J)
    G = rng.randint(len(GRID), size=(J, M))
    ar = np.arange(M)

    def loglik(rho):  # (J, n)
        mix = np.einsum("jm,jmn->jn", W, rho[ar[None, :], G])
        return np.log(np.clip(mix, 1e-300, None))
    ll_cal = loglik(rho_cal)
    ll_te = loglik(rho_te).sum(1)
    theta_star = ll_te.argmax()            # best theta for held-out queries (proxy for the truth)
    thr = stats.chi2.ppf(LEVEL, df=2 * M - 1) / 2

    out = dict(dataset=name, n_cal_avail=len(cal), n_test=len(te), J=J, per_n={})
    for n in NS + [len(cal)]:
        if n > len(cal):
            continue
        rec = dict(retained=[], GH=[], width_rel=[], tv_radius=[], star_in=[], test_ll_lower=[])
        for rep in range(REPS if n < len(cal) else 1):
            sub = rng.choice(len(cal), n, replace=False)
            l = ll_cal[:, sub].sum(1)
            keep = np.flatnonzero(l >= l.max() - thr)
            rec["retained"].append(len(keep))
            rec["star_in"].append(bool(theta_star in keep))
            ext = keep if len(keep) <= MAX_EXT else rng.choice(keep, MAX_EXT, replace=False)
            # P(relevant) interval on held-out queries
            mix = np.einsum("jm,jmn->jn", W[keep], rho_te[ar[None, :], G[keep]])
            rec["width_rel"].append(float((mix.max(0) - mix.min(0)).mean()))
            rec["test_ll_lower"].append(float(np.log(np.clip(mix.min(0), 1e-300, None)).mean()))
            gh, tv = [], []
            for SMq in SM_te[:60]:
                P = np.einsum("jm,jmn->jn", W[ext], SMq[ar[None, :], G[ext]])
                Hs = -(np.clip(P, 1e-12, 1) * np.log(np.clip(P, 1e-12, 1))).sum(1)
                gh.append(upper_entropy_hull(P, iters=150) - Hs.min())
                tv.append(0.5 * np.abs(P - P.mean(0)).sum(1).max())
            rec["GH"].append(float(np.mean(gh)))
            rec["tv_radius"].append(float(np.mean(tv)))
        out["per_n"][n] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in rec.items()}
        pn = out["per_n"][n]
        print(f"[E3] {name} n={n:4d} retained={pn['retained'][0]:7.1f} GH={pn['GH'][0]:.4f} "
              f"width(P rel)={pn['width_rel'][0]:.4f} TV={pn['tv_radius'][0]:.4f} theta*-in={pn['star_in'][0]:.2f}",
              flush=True)
    add_slopes(out)
    return out


def add_slopes(out, min_retained=10):
    """log-log slope of the contraction over n >= 10 while the likelihood region is still resolved by the
    finite theta sample (mean retained >= min_retained); beyond that the width is a discretization artefact."""
    ns = np.array([int(n) for n in out["per_n"] if int(n) >= 10 and out["per_n"][n]["retained"][0] >= min_retained])
    out["slope_n_range"] = [int(ns.min()), int(ns.max())]
    for key in ("width_rel", "tv_radius", "GH"):
        y = np.array([out["per_n"][n if n in out["per_n"] else str(n)][key][0] for n in ns])
        out[f"slope_{key}"] = float(np.polyfit(np.log(ns), np.log(y), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()
    os.makedirs(os.path.join(RESULTS_DIR, "shrink"), exist_ok=True)
    for name in args.datasets:
        res = run(name)
        json.dump(res, open(os.path.join(RESULTS_DIR, "shrink", f"{name}.json"), "w"))
        print(f"[E3] {name} slopes: width {res['slope_width_rel']:.2f}  TV {res['slope_tv_radius']:.2f}  "
              f"GH {res['slope_GH']:.2f}", flush=True)


if __name__ == "__main__":
    main()
