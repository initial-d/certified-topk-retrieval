"""Actionability check: does each layer's epistemic imprecision predict the value of ITS OWN remedy?

Remedies and their per-query value (test queries):
  refine  (index layer A)          : 1 - exact recall@10 of the unrefined quantized top-10 (PQ32)
  ensemble (representation layer B): nDCG@10(fused credal-centre ranking) - nDCG@10(primary)
Difficulty (irreducible)           : 1 - nDCG@10(primary)
Predictors are read from results/uq/<ds>.npz (computed without any knowledge of these targets).
Usage: python -m credal.exp_action"""
import json
import os
import warnings

import numpy as np
from scipy import stats

from . import encode
from .data import DATASETS, RESULTS_DIR, load, self_mask
from .index import quantize
from .metrics import ndcg_at_k

warnings.filterwarnings("ignore", category=RuntimeWarning)
K = 10
MODELS = list(encode.ENSEMBLE)
PRED = ["Credal-A-GH", "Credal-A-|U|", "Credal-B-GH", "Credal-B-|U|", "Credal-B-Hlo", "Entropy", "MaxScore",
        "Bayes-EU", "EnsJac"]


def targets(name):
    ds = load(name)
    z = np.load(os.path.join(RESULTS_DIR, "uq", f"{name}.npz"), allow_pickle=True)
    res = json.load(open(os.path.join(RESULTS_DIR, "uq", f"{name}.json")))
    te = z["te"]
    # value of refinement: naive quantized top-10 vs exact top-10 of the primary model
    X, Q = encode.get(name, "bge", "docs"), encode.get(name, "bge", "queries")
    Xh, _, _ = quantize("pq32", X)
    S, Sh = Q @ X.T, Q @ Xh.T
    sm = self_mask(ds)
    rows = np.flatnonzero(sm >= 0)
    S[rows, sm[rows]] = -1e9
    Sh[rows, sm[rows]] = -1e9
    kth = np.partition(S, -K, axis=1)[:, -K]
    naive = np.argpartition(-Sh, K, axis=1)[:, :K]
    v_refine = 1 - (np.take_along_axis(S, naive, 1) >= kth[:, None] - 1e-6).sum(1) / K
    # value of ensembling: fused (equal-weight calibrated mixture) minus primary nDCG@10
    pools, rel_pool = z["pools"], z["rel_pool"]
    taus = res["taus"]
    did_pos = {d: i for i, d in enumerate(ds.doc_ids)}
    grades = [np.array(list(ds.qrels[q].values()), float) for q in ds.query_ids]
    fused_rel = np.zeros((len(pools), K))
    for i in range(len(pools)):
        P = []
        for m in MODELS:
            zz = z[f"S_{m}"][i] / taus[m]
            e = np.exp(zz - zz.max())
            P.append(e / e.sum())
        o = np.argsort(-np.mean(P, 0))[:K]
        fused_rel[i, : len(o)] = rel_pool[i][o]
    v_ens = ndcg_at_k(fused_rel, grades, K) - z["ndcg"]
    return te, dict(refine=v_refine[te], ensemble=v_ens[te], difficulty=1 - z["ndcg"][te]), \
        {p: -z[f"pred_{p}"][te] for p in PRED}   # predictors re-oriented: higher = more imprecision/uncertainty


def main():
    out = {}
    for name in DATASETS:
        te, T, P = targets(name)
        out[name] = {t: {p: float(stats.kendalltau(P[p], T[t])[0]) for p in PRED} for t in T}
        print(f"\n{name}  (Kendall tau; predictors oriented as 'more imprecision')")
        print(" " * 14 + "".join(f"{t:>12s}" for t in T))
        for p in PRED:
            print(f"{p:14s}" + "".join(f"{out[name][t][p]:+12.3f}" for t in T))
    json.dump(out, open(os.path.join(RESULTS_DIR, "action.json"), "w"), indent=1)
    print("\nMEAN over datasets")
    for p in PRED:
        print(f"{p:14s}" + "".join(f"{np.mean([out[d][t][p] for d in DATASETS]):+12.3f}"
                                   for t in ("refine", "ensemble", "difficulty")))


if __name__ == "__main__":
    main()
