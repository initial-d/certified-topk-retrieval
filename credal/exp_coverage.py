"""Validity of decision-calibrated credal sets over many random calibration/test splits and several k.
Usage: python -m credal.exp_coverage [datasets...]"""
import argparse
import json
import os
import warnings

import numpy as np

from . import encode
from .credal import conformal_quantile, decision_nonconformity
from .data import DATASETS, RESULTS_DIR, load, self_mask
from .index import quantize

warnings.filterwarnings("ignore", category=RuntimeWarning)
KS = [1, 5, 10, 20, 50]
DELTAS = [0.05, 0.1, 0.2]
SPLITS = 500


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
    rng = np.random.RandomState(seed)
    n = len(Q)
    out = {}
    for k in KS:
        al = np.array([decision_nonconformity(S[i], Sh[i], r, qn[i], k) for i in range(n)])
        out[k] = {}
        for delta in DELTAS:
            cov = []
            for _ in range(SPLITS):
                p = rng.permutation(n)
                c = conformal_quantile(al[p[: n // 2]], delta)
                cov.append((al[p[n // 2:]] <= c).mean())
            out[k][delta] = (float(np.mean(cov)), float(np.std(cov)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    ap.add_argument("--quant", default="pq32,pq96,sq4")
    args = ap.parse_args()
    res = {}
    for name in args.datasets:
        for qn in args.quant.split(","):
            res[f"{name}.{qn}"] = run(name, qn)
            print(name, qn, {k: {d: f"{m:.3f}±{s:.3f}" for d, (m, s) in v.items()}
                             for k, v in res[f"{name}.{qn}"].items()}, flush=True)
    json.dump(res, open(os.path.join(RESULTS_DIR, "coverage.json"), "w"))


if __name__ == "__main__":
    main()
