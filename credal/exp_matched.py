"""Fixed-depth re-ranking at a budget matched to credal shrinkage (post-hoc companion of exp_certify).
For each E1 result, re-rank the top-R approximate candidates exactly, with R = the mean number of refinements
used by Algorithm 1, and record exact recall and the fraction of queries whose top-k is fully recovered.
Usage: python -m credal.exp_matched"""
import glob
import json
import os
import warnings

import numpy as np

from . import encode
from .data import RESULTS_DIR, load, self_mask
from .index import quantize

warnings.filterwarnings("ignore", category=RuntimeWarning)
K = 10
GRID = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]


def main():
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "certify", "*.json"))):
        res = json.load(open(path))
        if "matched" in res:
            continue
        ds = load(res["dataset"])
        X, Q = encode.get(ds.name, res["model"], "docs"), encode.get(ds.name, res["model"], "queries")
        Xh, _, _ = quantize(res["quant"], X)
        te = ds.test_idx
        S, Sh = Q[te] @ X.T, Q[te] @ Xh.T
        sm = self_mask(ds)[te]
        rows = np.flatnonzero(sm >= 0)
        S[rows, sm[rows]] = -1e9
        Sh[rows, sm[rows]] = -1e9
        true_top = np.argsort(-S, 1)[:, :K]
        order_h = np.argsort(-Sh, 1)

        def at(R):
            cand = order_h[:, :R]
            got = np.take_along_axis(cand, np.argsort(-np.take_along_axis(S, cand, 1), 1)[:, :K], 1)
            from .credal import topk_hits
            rec = np.array([topk_hits(S[i], got[i], K) for i in range(len(got))]) / K
            return dict(R=int(R), exact_recall=float(rec.mean()), perfect=float((rec == 1).mean()))
        res["rerank_curve"] = [at(R) for R in GRID if R <= X.shape[0]]
        cr = res["credal"]["decision"]["lucb"]
        R = int(np.ceil(cr["n_ref_mean"]))
        res["matched"] = dict(budget=R, fixed=at(R),
                              credal=dict(exact_recall=cr["exact_recall"], perfect=cr["perfect"]))
        # smallest fixed depth whose perfect-recovery rate reaches that of the certified runs
        need = next((c["R"] for c in res["rerank_curve"] if c["perfect"] >= cr["perfect"]), None)
        res["matched"]["fixed_depth_needed"] = need
        json.dump(res, open(path, "w"))
        m = res["matched"]
        print(f"{os.path.basename(path):28s} budget={R:5d} fixed perfect={m['fixed']['perfect']:.3f} "
              f"credal perfect={m['credal']['perfect']:.3f} depth needed={need}", flush=True)


if __name__ == "__main__":
    main()
