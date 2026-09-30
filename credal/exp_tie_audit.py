"""Audit of the tie-aware correctness criterion.

Correct top-k (possibly with ties at tau = s_(k)): |R| = k and min_{R} s >= max_{not R} s  (all options strictly above tau
are returned; the remaining slots are filled with options tied at tau). The earlier criterion "every returned option has
s >= tau" is too weak with ties. This script recomputes, with both criteria, every correctness label and depth that the paper
uses, restricted to queries with a tie at the k-th score (on all other queries the two criteria coincide):
  * Alg. 1 correctness on all 30 BEIR configurations (exp_certify protocol),
  * the sufficient re-ranking depth R(q) (rank boxes, truncated boxes),
  * the miss fraction used for risk control (exp_risk_actual, 10 configurations),
  * the correctness flags of the joint-update procedures at every grid level (exp_joint v3, 12 configurations).
Usage: python -m credal.exp_tie_audit"""
import json
import os
import warnings

import numpy as np

from . import encode
from . import stats as ST
from .credal import conformal_quantile, intervals, refine
from .data import DATASETS, RESULTS_DIR, load, self_mask
from .index import quantize

warnings.filterwarnings("ignore", category=RuntimeWarning)
K, TOL = 10, 1e-6


def correct_new(s, R):
    R = np.asarray(R)
    mask = np.ones(len(s), bool)
    mask[R] = False
    return len(R) == K and s[R].min() >= s[mask].max() - TOL


def correct_old(s, R):
    return bool((s[np.asarray(R)] >= np.partition(s, -K)[-K] - TOL).all())


def miss_new(s, R):
    tau = np.partition(s, -K)[-K]
    above = s > tau + TOL
    inR = np.zeros(len(s), bool)
    inR[np.asarray(R)] = True
    got = (above & inR).sum() + min((inR & ~above & (s >= tau - TOL)).sum(), K - above.sum())
    return 1 - got / K


def miss_old(s, R):
    return 1 - (s[np.asarray(R)] >= np.partition(s, -K)[-K] - TOL).sum() / K


def depth_new(s, sh):
    """Smallest R such that the approximate top-R contains a correct top-k."""
    order = np.argsort(-sh, kind="stable")
    tau = np.partition(s, -K)[-K]
    above = s[order] > tau + TOL
    tie = np.abs(s[order] - tau) <= TOL
    n_above = int((s > tau + TOL).sum())
    ca, ct = np.cumsum(above), np.cumsum(tie)
    ok = (ca == n_above) & (ct >= K - n_above)
    return int(np.argmax(ok) + 1)


def depth_old(s, sh):
    kth = np.partition(s, -K)[-K]
    good = s[np.argsort(-sh, kind="stable")] >= kth - TOL
    return int(np.searchsorted(np.cumsum(good), K) + 1)


def main():
    out = {"certify": {}, "depth": {}, "risk": {}, "joint": {}}
    for name in DATASETS:
        ds = load(name)
        X, Q = encode.get(name, "bge", "docs"), encode.get(name, "bge", "queries")
        sm = self_mask(ds)
        qn = np.linalg.norm(Q, axis=1)
        S = Q @ X.T
        S[np.flatnonzero(sm >= 0), sm[sm >= 0]] = -1e9
        srt = -np.sort(-S, 1)[:, :K + 1]
        tied = np.flatnonzero(srt[:, K - 1] - srt[:, K] <= TOL)
        for q in ST.QUANTS:
            Xh, r, _ = quantize(q, X)
            st = ST.get(name, q)["per_k"][K]
            c = conformal_quantile(st["alpha_dec"][ds.calib_idx], 0.1)
            d_cert = d_depth = 0
            risk_levels = []
            rj = os.path.join(RESULTS_DIR, "risk_actual.json")
            if q in ("pq32", "pq96") and os.path.exists(rj):
                risk_levels = [v["c"] for v in json.load(open(rj)).get(f"{name}.{q}", {}).values()]
            d_risk = 0
            for i in tied:
                sh = Q[i] @ Xh.T
                if sm[i] >= 0:
                    sh[sm[i]] = -1e9
                s = S[i]
                d_depth += depth_new(s, sh) != depth_old(s, sh)
                if i in set(ds.test_idx):
                    l, u = intervals(sh, r, qn[i], c, 0.0)
                    R = refine(l, u, s, K, "lucb")["topk"]
                    d_cert += correct_new(s, R) != correct_old(s, R)
                    for cr in risk_levels:
                        l, u = intervals(sh, r, qn[i], cr, 0.0)
                        R = refine(l, u, s, K, "lucb")["topk"]
                        d_risk += abs(miss_new(s, R) - miss_old(s, R)) > 1e-12
            out["certify"][f"{name}.{q}"] = int(d_cert)
            out["depth"][f"{name}.{q}"] = int(d_depth)
            if risk_levels:
                out["risk"][f"{name}.{q}"] = int(d_risk)
        out.setdefault("n_tied", {})[name] = int(len(tied))
        print(name, "tied queries", len(tied), "| certify diffs", sum(out["certify"][f"{name}.{q}"] for q in ST.QUANTS),
              "| depth diffs", sum(out["depth"][f"{name}.{q}"] for q in ST.QUANTS),
              "| risk diffs", sum(v for k_, v in out["risk"].items() if k_.startswith(name)), flush=True)
    # joint updates: correctness flags at every grid level, tied queries only
    from .exp_joint import GRID_N, estimate_rho, instance, prepare, run_procedure_v3, PROCS_V3
    jpath = os.path.join(RESULTS_DIR, "joint_v3.json")
    for key in json.load(open(jpath)):
        name, q = key.split(".")
        ds, X, Xn, S, Sh, r, qn, alpha = prepare(name, q)
        n = S.shape[0]
        p = np.random.RandomState(0).permutation(n)
        rho = estimate_rho(p[: n // 4], S, Sh, r, qn, Xn)
        grid = np.linspace(0, 1.6 * conformal_quantile(alpha[p[: n // 4]], 0.1), GRID_N + 1)[1:]
        srt = -np.sort(-S, 1)[:, :K + 1]
        tied = [i for i in p[n // 4:] if srt[i, K - 1] - srt[i, K] <= TOL]
        diffs = 0
        for i in tied:
            sh, w, s, nbr, kth = instance(i, S, Sh, r, qn, Xn, grid[-1])
            U = np.argsort(-(Sh[i] + grid[-1] * qn[i] * r))[:3000]
            for th in grid:
                for lab, sft, shr in PROCS_V3:
                    # re-run to obtain the returned set on the universe, then judge against the whole corpus
                    import credal.exp_joint as EJ
                    Rl = _returned(EJ, sh, w, s, nbr, th, rho if (sft or shr) else 0.0, sft, shr)
                    R = U[Rl]
                    diffs += correct_new(S[i], R) != correct_old(S[i], R)
        out["joint"][key] = dict(tied=len(tied), diffs=int(diffs))
        print("joint", key, "tied", len(tied), "diffs", diffs, flush=True)
    json.dump(out, open(os.path.join(RESULTS_DIR, "tie_audit.json"), "w"), indent=1)
    print("TOTAL certify", sum(out["certify"].values()), "depth", sum(out["depth"].values()),
          "risk", sum(out["risk"].values()), "joint", sum(v["diffs"] for v in out["joint"].values()))


def _returned(EJ, sh, w, s, nbr, theta, rho, use_shift, use_shrink):
    """Copy of exp_joint.run_procedure_v3 that returns the selected indices (within the universe)."""
    n = len(sh)
    kappa = np.sqrt(1 - rho ** 2) if use_shrink else 1.0
    shift, cnt = np.zeros(n), np.zeros(n)
    width = theta * w
    res = np.zeros(n, bool)
    l, u = sh.copy(), sh + width
    while True:
        R = np.argpartition(-l, K - 1)[:K]
        inR = np.zeros(n, bool)
        inR[R] = True
        Lk = l[R].min()
        chall = ~inR & (u >= Lk)
        verify = inR & ~res
        if not chall.any() and not verify.any():
            break
        opts = []
        if verify.any():
            opts.append(np.where(verify, l, np.inf).argmin())
        cu = chall & ~res
        if cu.any():
            opts.append(np.where(cu, u, -np.inf).argmax())
        if not opts:
            break
        j = max(opts, key=lambda t: u[t] - l[t])
        res[j] = True
        l[j] = u[j] = s[j]
        if rho != 0 and (use_shift or use_shrink):
            e = (s[j] - sh[j]) / w[j]
            nb = nbr[j][~res[nbr[j]]]
            shift[nb] += e
            cnt[nb] += 1
            cc = sh[nb] + (rho * shift[nb] / cnt[nb] * w[nb] if use_shift else 0)
            l[nb] = cc
            u[nb] = cc + width[nb] * kappa
    return np.argpartition(-l, K - 1)[:K]


if __name__ == "__main__":
    main()
