"""Truncated residual boxes (hybrid of residual and rank boxes).

For a fixed depth Rbar: options ranked (by cheap score) within the top Rbar get residual boxes [sh, sh + theta w];
options beyond Rbar get u = l = -inf ("certainly not selected"). For fixed Rbar the family is nested in theta, so
Thm. characterization / validity apply with the decision nonconformity
    alpha_hyb(q) = alpha_dec(q) if R(q) <= Rbar else +inf.
Rbar is chosen on a TUNE half of the calibration queries (as the depth quantile at a level delta_R, delta_R itself chosen
by mean cost on the tune half); theta is then calibrated on the other half at level delta. Test queries untouched.
Cost of Algorithm 1 is measured by actual runs (BEIR) and by oracle cost k + #{rank <= Rbar, j not in T, u_j >= s_(k)}.
Usage: python -m credal.exp_hybrid [datasets...]"""
import hashlib
import json
import os
import sys
import warnings

import numpy as np

from . import encode
from .credal import conformal_quantile, decision_nonconformity, refine, required_depth, topk_correct
from .data import DATASETS, RESULTS_DIR, load, self_mask
from .index import quantize

warnings.filterwarnings("ignore", category=RuntimeWarning)
K = 10
DELTA = 0.1
# cut-off levels: at most delta/4, so that the calibrated theta is finite with high probability (Prop. truncated (c));
# if no level is available on the tune half, no cut-off is used (plain residual boxes)
DELTA_R = [0.005, 0.01, 0.02, 0.025]


def oracle_cost_rank(vr, Rbar, theta):
    """vr: (R_max,) standardized thresholds v_j = (s_(k) - sh_j)/w_j in cheap-score rank order, +inf for winners."""
    return K + int((vr[:Rbar] <= theta).sum())


def choose(alpha, R, VR, idx_tune, delta):
    """Pick delta_R on the tune half by mean oracle cost (theta also fitted on the tune half)."""
    best = None
    for dr in DELTA_R:
        Rbar = conformal_quantile(R[idx_tune], dr)
        if not np.isfinite(Rbar):
            continue
        Rbar = int(Rbar)
        a = np.where(R[idx_tune] <= Rbar, alpha[idx_tune], np.inf)
        th = conformal_quantile(a, delta)
        if not np.isfinite(th):
            continue
        cost = np.mean([oracle_cost_rank(VR[i], Rbar, th) for i in idx_tune])
        if best is None or cost < best[0]:
            best = (cost, dr, Rbar)
    return best


def prepare(name, q):
    ds = load(name)
    X, Q = encode.get(name, "bge", "docs"), encode.get(name, "bge", "queries")
    Xh, r, _ = quantize(q, X)
    S, Sh = Q @ X.T, Q @ Xh.T
    sm = self_mask(ds)
    rows = np.flatnonzero(sm >= 0)
    S[rows, sm[rows]] = -1e9
    Sh[rows, sm[rows]] = -1e9
    qn = np.linalg.norm(Q, axis=1)
    order = np.argsort(-Sh, 1)
    alpha = np.array([decision_nonconformity(S[i], Sh[i], r, qn[i], K) for i in range(len(Q))])
    R = np.zeros(len(Q), int)
    VR = np.zeros(S.shape, np.float32)
    for i in range(len(Q)):
        kth = np.partition(S[i], -K)[-K]
        good = S[i][order[i]] >= kth - 1e-6
        R[i] = required_depth(S[i], Sh[i], K)
        v = (kth - Sh[i][order[i]]) / (qn[i] * r[order[i]] + 1e-8)
        v[good] = np.inf
        VR[i] = v
    return ds, alpha, R, VR


def resplit(name, q, reps=200, seed=1, log=None):
    """Paired comparison on identical random splits (tune / calibration / test drawn from all queries).
    Every split is used: if no cut-off can be chosen on the tune half, or the calibrated theta is infinite, the truncated
    procedure falls back to full verification (all N options resolved), which is always correct (Prop. truncated).
    Costs are oracle costs for residual and truncated boxes (Alg. 1 is within a few percent of them) and exact for rank boxes."""
    ds, alpha, R, VR = prepare(name, q)
    n, N = VR.shape
    rng = np.random.RandomState(seed)
    rec = {k: [] for k in ("val_t", "val_r", "val_res", "cost_t", "cost_r", "cost_res", "p95_t", "p95_res", "fallback",
                           "delta_R", "Rbar", "n_beyond")}
    pooled = {"t": [], "res": [], "r": []}
    for _ in range(reps):
        p = rng.permutation(n)
        cal, te = p[: n // 2], p[n // 2:]
        tune, cal2 = cal[: len(cal) // 2], cal[len(cal) // 2:]
        best = choose(alpha, R, VR, tune, DELTA)
        Rbar = best[2] if best is not None else N          # no admissible cut-off: plain residual boxes
        th = conformal_quantile(np.where(R[cal2] <= Rbar, alpha[cal2], np.inf), DELTA)
        fb = not np.isfinite(th)
        n_beyond = int((R[cal2] > Rbar).sum())
        n2 = len(cal2)
        kq = int(np.ceil((n2 + 1) * (1 - DELTA)))            # conformal order-statistic index in cal2
        if fb:
            ct = np.full(len(te), N)
            ok = np.ones(len(te), bool)
        else:
            ct = np.array([oracle_cost_rank(VR[i], Rbar, th) for i in te])
            ok = np.where(R[te] <= Rbar, alpha[te], np.inf) <= th
        c_dec = conformal_quantile(alpha[cal], DELTA)
        cres = np.array([K + int((VR[i] <= c_dec).sum()) for i in te])
        Rh = min(conformal_quantile(R[cal], DELTA), N)
        if log is not None:
            log.append(dict(
                split=_, perm_sha1=hashlib.sha1(p.tobytes()).hexdigest()[:12],
                n_tune=len(tune), n_cal=n2, n_test=len(te), delta_R=None if best is None else best[1],
                Rbar=int(Rbar), max_winner_depth_tune=int(R[tune].max()), n_beyond_cal=n_beyond,
                quantile_index=kq, n_infinite_scores=int((~np.isfinite(np.where(R[cal2] <= Rbar, alpha[cal2], np.inf))).sum()),
                theta=None if fb else float(th), fallback=bool(fb),
                fallback_reason=None if not fb else (
                    "no admissible cut-off" if best is None else
                    f"{n_beyond} calibration winners beyond the cut-off > {n2 - kq} allowed"),
                mean_cost_truncated=float(ct.mean()), mean_cost_residual=float(cres.mean()), rank_depth=float(Rh)))
        rec["fallback"].append(fb)
        rec["delta_R"].append(best[1] if best is not None else np.nan)
        rec["Rbar"].append(Rbar)
        rec["n_beyond"].append(int((R[cal2] > Rbar).sum()))
        pooled["t"].extend(ct.tolist())
        pooled["res"].extend(cres.tolist())
        pooled["r"].extend([Rh] * len(te))
        rec["val_t"].append(ok.mean())
        rec["val_r"].append(np.mean(R[te] <= Rh))
        rec["val_res"].append(np.mean(alpha[te] <= c_dec))
        rec["cost_t"].append(ct.mean())
        rec["cost_r"].append(Rh)
        rec["cost_res"].append(cres.mean())
        rec["p95_t"].append(np.percentile(ct, 95))
        rec["p95_res"].append(np.percentile(cres, 95))
    m = {k: np.array(v, float) for k, v in rec.items()}
    return dict(reps=reps, fallback_rate=float(m["fallback"].mean()),
                valid_hybrid=(float(m["val_t"].mean()), float(m["val_t"].std())),
                valid_rank=(float(m["val_r"].mean()), float(m["val_r"].std())),
                valid_residual=(float(m["val_res"].mean()), float(m["val_res"].std())),
                cost_rank=float(m["cost_r"].mean()), cost_hybrid=float(m["cost_t"].mean()),
                cost_residual=float(m["cost_res"].mean()),
                cost_ratio=float(m["cost_t"].mean() / m["cost_r"].mean()),
                cost_ratio_residual=float(m["cost_res"].mean() / m["cost_r"].mean()),
                p95_hybrid=float(m["p95_t"].mean()), p95_residual=float(m["p95_res"].mean()),
                wins_hybrid=float(np.mean(m["cost_t"] < m["cost_r"])),
                # tail statistics: mean over splits of the per-split P95 (reported as such) and P95 of the pooled
                # per-query costs over all splits (includes fallback splits)
                pooled_p95_hybrid=float(np.percentile(pooled["t"], 95)),
                pooled_p95_residual=float(np.percentile(pooled["res"], 95)),
                pooled_p95_rank=float(np.percentile(pooled["r"], 95)),
                n_fallback=int(m["fallback"].sum()), n_test=len(te), n_cal=int(n // 2), n_tune=int(n // 4),
                delta_R_counts={str(k): int(v) for k, v in zip(*np.unique(m["delta_R"].astype(str), return_counts=True))},
                max_n_beyond=int(m["n_beyond"].max()))


def run(name, q, seed=0):
    ds = load(name)
    X, Q = encode.get(name, "bge", "docs"), encode.get(name, "bge", "queries")
    Xh, r, _ = quantize(q, X)
    S, Sh = Q @ X.T, Q @ Xh.T
    sm = self_mask(ds)
    rows = np.flatnonzero(sm >= 0)
    S[rows, sm[rows]] = -1e9
    Sh[rows, sm[rows]] = -1e9
    qn = np.linalg.norm(Q, axis=1)
    nq, N = S.shape
    order = np.argsort(-Sh, 1)
    alpha = np.array([decision_nonconformity(S[i], Sh[i], r, qn[i], K) for i in range(nq)])
    R = np.zeros(nq, int)
    VR = np.zeros((nq, N), np.float32)
    for i in range(nq):
        kth = np.partition(S[i], -K)[-K]
        good = S[i][order[i]] >= kth - 1e-6
        R[i] = required_depth(S[i], Sh[i], K)
        v = (kth - Sh[i][order[i]]) / (qn[i] * r[order[i]] + 1e-8)
        v[good] = np.inf
        VR[i] = v
    rng = np.random.RandomState(seed)
    cal = rng.permutation(ds.calib_idx)
    tune, cal2, te = cal[: len(cal) // 2], cal[len(cal) // 2:], ds.test_idx
    best = choose(alpha, R, VR, tune, DELTA)
    dr, Rbar = (best[1], best[2]) if best is not None else (None, N)
    a2 = np.where(R[cal2] <= Rbar, alpha[cal2], np.inf)
    th = conformal_quantile(a2, DELTA)
    fallback = not np.isfinite(th)
    # actual Algorithm 1 on the truncated boxes
    costs, correct = [], []
    for i in (te if not fallback else []):
        l = np.full(N, -1e9)
        u = np.full(N, -1e9)
        top = order[i][:Rbar]
        l[top] = Sh[i][top]
        u[top] = Sh[i][top] + th * qn[i] * r[top]
        res = refine(l, u, S[i], K, "lucb")
        kth = np.partition(S[i], -K)[-K]
        costs.append(res["n_ref"])
        correct.append(topk_correct(S[i], res["topk"], K))
    costs = np.array(costs) if not fallback else np.full(len(te), N)
    correct = correct if not fallback else [True] * len(te)
    # references on the SAME test queries: rank boxes and full residual boxes, both calibrated on the full cal half
    Rhat = int(conformal_quantile(R[ds.calib_idx], DELTA))
    c_dec = conformal_quantile(alpha[ds.calib_idx], DELTA)
    res_cost = np.array([K + int((VR[i] <= c_dec).sum()) for i in te])
    out = dict(dataset=name, quant=q, delta_R=dr, Rbar=int(Rbar), theta=float(th), fallback=bool(fallback),
               hybrid=dict(mean=float(costs.mean()), p95=float(np.percentile(costs, 95)), max=int(costs.max()),
                           exact=float(np.mean(correct)),
                           valid=float(np.mean(np.where(R[te] <= Rbar, alpha[te], np.inf) <= th)) if not fallback else 1.0,
                           oracle_mean=float(np.mean([oracle_cost_rank(VR[i], Rbar, th) for i in te])) if not fallback else float(N)),
               rank=dict(mean=float(Rhat), exact=float((R[te] <= Rhat).mean())),
               residual=dict(oracle_mean=float(res_cost.mean()), oracle_p95=float(np.percentile(res_cost, 95)),
                             valid=float((alpha[te] <= c_dec).mean())))
    h = out["hybrid"]
    print(f"{name:9s} {q:5s} Rbar={Rbar:5d} (dR={dr}) th={th:.3f} | hybrid mean {h['mean']:7.1f} p95 {h['p95']:6.0f} "
          f"exact {h['exact']:.3f} | rank {Rhat:5d} exact {out['rank']['exact']:.3f} | residual oracle "
          f"{res_cost.mean():7.1f} p95 {np.percentile(res_cost, 95):6.0f} | hybrid/rank {h['mean'] / Rhat:.2f}", flush=True)
    return out


def main_resplit(path=None, log_path=None):
    """Re-split comparison. With log_path, one JSON line per split is written (reproducibility record)."""
    path = path or os.path.join(RESULTS_DIR, "hybrid_resplit.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in DATASETS:
        for q in ("pq16", "pq32", "pq96", "opq32", "sq4", "bin"):
            if f"{name}.{q}" in res:
                continue
            log = [] if log_path else None
            r = res[f"{name}.{q}"] = resplit(name, q, log=log)
            if log_path:
                with open(log_path, "a") as f:
                    for row in log:
                        f.write(json.dumps(dict(config=f"{name}.{q}", **row)) + "\n")
            json.dump(res, open(path, "w"), indent=1)
            print(f"{name:9s} {q:5s} valid trunc {r['valid_hybrid'][0]:.3f} rank {r['valid_rank'][0]:.3f} "
                  f"res {r['valid_residual'][0]:.3f} | cost trunc/rank {r['cost_ratio']:.2f} res/rank "
                  f"{r['cost_ratio_residual']:.2f} | p95 trunc {r['p95_hybrid']:.0f} rank {r['cost_rank']:.0f} | "
                  f"fallback {r['n_fallback']}/{r['reps']} (max beyond {r['max_n_beyond']}) | pooled p95 trunc "
                  f"{r['pooled_p95_hybrid']:.0f} res {r['pooled_p95_residual']:.0f} rank {r['pooled_p95_rank']:.0f}", flush=True)


def main():
    if sys.argv[1:2] == ["resplit"]:
        return main_resplit()
    if sys.argv[1:2] == ["resplit-verify"]:
        # reproduce the re-split experiment from scratch into separate files, with per-split records
        out = os.path.join(RESULTS_DIR, "repro")
        os.makedirs(out, exist_ok=True)
        for f in ("hybrid_resplit.json", "hybrid_resplit_splits.jsonl"):
            if os.path.exists(os.path.join(out, f)):
                os.remove(os.path.join(out, f))
        return main_resplit(os.path.join(out, "hybrid_resplit.json"), os.path.join(out, "hybrid_resplit_splits.jsonl"))
    path = os.path.join(RESULTS_DIR, "hybrid.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in sys.argv[1:] or DATASETS:
        for q in ("pq16", "pq32", "pq96", "opq32", "sq4", "bin"):
            if f"{name}.{q}" not in res:
                res[f"{name}.{q}"] = run(name, q)
                json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
