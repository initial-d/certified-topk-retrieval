"""Review-8 checks: (1) ties at the k-th/(k+1)-th exact score and termination of Algorithm 1;
(2) Algorithm 1 cost relative to the oracle cost |A(q)| on correctly returned queries only (configurations of Table 7).
Usage: python -m credal.exp_ties_oracle"""
import json
import os
import warnings

import numpy as np

from . import encode
from . import stats as ST
from .credal import conformal_quantile, intervals, refine, topk_correct
from .data import DATASETS, RESULTS_DIR, load, self_mask
from .index import quantize

warnings.filterwarnings("ignore", category=RuntimeWarning)
K = 10
TIE = 1e-6


def run(name, q):
    ds = load(name)
    X, Q = encode.get(name, "bge", "docs"), encode.get(name, "bge", "queries")
    Xh, r, _ = quantize(q, X)
    sm = self_mask(ds)
    st = ST.get(name, q)["per_k"][K]
    cal, te = ds.calib_idx, ds.test_idx
    c = conformal_quantile(st["alpha_dec"][cal], 0.1)
    qn = np.linalg.norm(Q, axis=1)
    A = ST.oracle_cost(st, te, c, K)
    rows = []
    for a, i in enumerate(te):
        s, sh = Q[i] @ X.T, Q[i] @ Xh.T
        if sm[i] >= 0:
            s[sm[i]] = sh[sm[i]] = -1e9
        srt = np.sort(s)[::-1]
        tie = bool(srt[K - 1] - srt[K] <= TIE)
        l, u = intervals(sh, r, qn[i], c, 0.0)
        res = refine(l, u, s, K, "lucb")
        correct = topk_correct(s, res["topk"], K)
        rows.append(dict(tie=tie, certified=res["certified"], correct=correct, cost=res["n_ref"], oracle=int(A[a])))
    return rows


def main():
    out = {}
    for name in DATASETS:
        for q in ("pq32", "pq96", "sq4"):
            rows = run(name, q)
            ok = [x for x in rows if x["correct"]]
            ratio = np.array([x["cost"] / x["oracle"] for x in ok])
            out[f"{name}.{q}"] = dict(
                n_test=len(rows), n_correct=len(ok), n_ties=sum(x["tie"] for x in rows),
                n_uncertified=sum(not x["certified"] for x in rows),
                n_uncertified_without_tie=sum((not x["certified"]) and not x["tie"] for x in rows),
                all_mean_cost=float(np.mean([x["cost"] for x in rows])),
                all_mean_oracle=float(np.mean([x["oracle"] for x in rows])),
                correct_ratio_mean=float(ratio.mean()), correct_ratio_median=float(np.median(ratio)),
                correct_ratio_min=float(ratio.min()), correct_ratio_max=float(ratio.max()),
                correct_below_oracle=int((ratio < 1 - 1e-12).sum()),
                correct_mean_cost=float(np.mean([x["cost"] for x in ok])),
                correct_mean_oracle=float(np.mean([x["oracle"] for x in ok])))
            o = out[f"{name}.{q}"]
            print(f"{name:9s} {q:5s} n={o['n_test']} correct={o['n_correct']} ties={o['n_ties']} "
                  f"uncertified={o['n_uncertified']} (w/o tie {o['n_uncertified_without_tie']}) | correct: ratio mean "
                  f"{o['correct_ratio_mean']:.4f} max {o['correct_ratio_max']:.3f} below-oracle {o['correct_below_oracle']}",
                  flush=True)
    json.dump(out, open(os.path.join(RESULTS_DIR, "ties_oracle.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
