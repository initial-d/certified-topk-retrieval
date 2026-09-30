"""Decision-safe domain certificate (stage A/B check).

Safe domain S_0 = {v distinct, interior: u0_i >= v_(k+1) for every i in T_k(v)} (frozen decision-calibrated upper bounds u0).
After exact observations, the returned set R (k best observed) is certified if R is feasible and no other top-k answer is
feasible: for all h in 0..k-1,  b_h < y_{h+1}  or  c_h < k - h, with b_0 = +inf, b_h = min_{r<=h} u0(i_r),
c_h = #{unobserved j : u0_j >= y_{h+1}}. Feasibility of R: if m > k, min_{R} u0 >= y_{k+1}; if m = k, always (interior).
The schedule is the LUCB rule of Algorithm 1; both stopping rules are evaluated on the same run.
Usage: python -m credal.exp_domain [selftest]"""
import bisect
import json
import os
import sys
import warnings

import numpy as np

from . import encode
from . import stats as ST
from .credal import conformal_quantile, intervals, topk_correct
from .data import DATASETS, RESULTS_DIR, load, self_mask
from .index import quantize

warnings.filterwarnings("ignore", category=RuntimeWarning)
TOL = 1e-9


def domain_certificate(y_obs, u0_obs, u0_unobs_sorted, n_unobs_counter, k):
    """y_obs, u0_obs: observed exact scores and their frozen upper bounds (any order).
    u0_unobs_sorted: callable t -> #{unobserved j: u0_j >= t}.  Returns 'certified' | 'other' | 'empty' | 'tie'."""
    m = len(y_obs)
    order = np.argsort(-y_obs)
    y = y_obs[order]
    u = u0_obs[order]
    top = y[: min(m, k + 1)]
    if len(top) > 1 and np.min(np.diff(-top)) <= TOL:
        return "tie"                                       # conservative: exact ties among the relevant observed scores
    b = np.minimum.accumulate(u[:k])                        # b_h for h = 1..k
    R_feasible = (b[k - 1] >= y[k] - TOL) if m > k else True
    other = False
    for h in range(k):
        bh = np.inf if h == 0 else b[h - 1]
        if bh >= y[h] and n_unobs_counter(y[h]) >= k - h:
            other = True
            break
    if R_feasible and not other:
        return "certified"
    if not R_feasible and not other:
        return "empty"
    return "other"


def run_query(sh, s, w, c, k):
    """LUCB schedule (Alg. 1, lambda = 0); returns (old_cost, new_cost, new_status, old_correct, new_correct)."""
    n = len(sh)
    u0 = sh + c * w
    l, u = sh.astype(np.float64).copy(), u0.astype(np.float64).copy()
    res = np.zeros(n, bool)
    u0_sorted = np.sort(u0)
    obs_u0_sorted = []                                     # sorted u0 of observed options
    obs_idx = []

    def count_unobs(t):
        return (n - np.searchsorted(u0_sorted, t, side="left")) - (len(obs_u0_sorted) - bisect.bisect_left(obs_u0_sorted, t))

    new_cost, new_status, new_R = None, None, None
    steps = 0
    order_u = np.argsort(-u0)
    while True:
        # working set as in credal.refine: options with u >= L_k matter
        R = np.argpartition(-l, k - 1)[:k]
        inR = np.zeros(n, bool)
        inR[R] = True
        Lk = l[R].min()
        chall = ~inR & (u >= Lk)
        verify = inR & ~res
        if not chall.any() and not verify.any():
            break
        if new_cost is None and len(obs_idx) >= k:
            st = domain_certificate(s[obs_idx], u0[obs_idx], None, count_unobs, k)
            if st in ("certified", "empty"):
                new_cost, new_status = steps, st
                oi = np.array(obs_idx)
                new_R = oi[np.argsort(-s[oi])[:k]] if st == "certified" else None
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
        obs_idx.append(j)
        bisect.insort(obs_u0_sorted, u0[j])
        steps += 1
    R = np.argpartition(-l, k - 1)[:k]
    old_correct = topk_correct(s, R, k)
    if new_cost is None:                                   # new rule never fired before the old one: same stop
        new_cost, new_status, new_correct = steps, "old", old_correct
    elif new_status == "empty":
        new_cost, new_correct = n, True                    # full verification fallback
    else:
        new_correct = topk_correct(s, new_R, k)
    return steps, new_cost, new_status, old_correct, new_correct


def mechanism(sh, s, w, c, k):
    """Necessary condition for any gain: some true top-(k-1) winner has u0 < s_(k)."""
    o = np.argsort(-s)
    u0 = sh + c * w
    return bool(k >= 2 and (u0[o[: k - 1]] < s[o[k - 1]]).any())


def selftest(trials=400, seed=0):
    """Compare the closed-form 'other answer exists' test with brute force over answer sets via an LP."""
    from itertools import combinations
    from scipy.optimize import linprog
    rng = np.random.RandomState(seed)
    a_, b_, eps = 0.0, 12.0, 1e-4
    agree = 0
    for _ in range(trials):
        N = rng.randint(3, 8)
        k = rng.randint(1, min(3, N - 1) + 1)
        m = rng.randint(k, N)
        u0 = rng.uniform(1, 11, N)
        obs = rng.choice(N, m, replace=False)
        y = np.sort(rng.uniform(1, 11, m))[::-1]
        yv = dict(zip(obs, y))
        unobs = [j for j in range(N) if j not in yv]
        R = set(obs[np.argsort(-np.array([yv[i] for i in obs]))[:k]])
        # brute force: exists answer A != R, |A| = k, and v consistent
        def feasible(A):
            A = set(A)
            idx = {j: t for t, j in enumerate(unobs)}
            nv = len(unobs)
            Aub, bub = [], []
            def lin(j):
                v = np.zeros(nv)
                if j in yv:
                    return v, yv[j]
                v[idx[j]] = 1
                return v, 0.0
            nonA = [j for j in range(N) if j not in A]
            for i in A:
                vi, ci = lin(i)
                for j in nonA:
                    vj, cj = lin(j)
                    Aub.append(vj - vi); bub.append(ci - cj - eps)      # v_j + eps <= v_i
                    Aub.append(vj); bub.append(u0[i] - cj)              # v_j <= u0_i  (safety: u0_i >= v_(k+1))
            if nv == 0:
                return all(bb >= -1e-12 for bb in bub)
            r = linprog(np.zeros(nv), A_ub=np.array(Aub), b_ub=np.array(bub),
                        bounds=[(a_ + eps, b_ - eps)] * nv, method="highs")
            return r.status == 0
        brute = any(set(A) != R and feasible(A) for A in combinations(range(N), k))
        ys = np.array([yv[i] for i in obs]); us = u0[obs]
        cnt = lambda t: sum(u0[j] >= t for j in unobs)
        # closed-form 'other exists' is independent of R feasibility: recompute
        order = np.argsort(-ys); yy, uu = ys[order], us[order]; bb = np.minimum.accumulate(uu[:k])
        other = any(((np.inf if h == 0 else bb[h - 1]) >= yy[h]) and cnt(yy[h]) >= k - h for h in range(k))
        agree += other == brute
    print(f"selftest: closed form agrees with LP brute force on {agree}/{trials} random histories")
    return agree == trials


def main(names=None):
    out = {}
    for name in names or DATASETS:
        ds = load(name)
        X, Q = encode.get(name, "bge", "docs"), encode.get(name, "bge", "queries")
        sm = self_mask(ds)
        qn = np.linalg.norm(Q, axis=1)
        for q in ("pq32", "pq96", "sq4", "bin"):
            Xh, r, _ = quantize(q, X)
            st = ST.get(name, q)
            for k in (1, 5, 10, 20):
                c = conformal_quantile(st["per_k"][k]["alpha_dec"][ds.calib_idx], 0.1)
                rows = []
                for i in ds.test_idx:
                    s, sh = Q[i] @ X.T, Q[i] @ Xh.T
                    if sm[i] >= 0:
                        s[sm[i]] = sh[sm[i]] = -1e9
                    w = qn[i] * r
                    old, new, status, oc, nc = run_query(sh, s, w, c, k)
                    rows.append((old, new, status, oc, nc, mechanism(sh, s, w, c, k)))
                old = np.array([x[0] for x in rows], float)
                new = np.array([x[1] for x in rows], float)
                key = f"{name}.{q}.k{k}"
                out[key] = dict(
                    n=len(rows), old_mean=float(old.mean()), new_mean=float(new.mean()),
                    old_p95=float(np.percentile(old, 95)), new_p95=float(np.percentile(new, 95)),
                    ratio=float(new.mean() / old.mean()), frac_earlier=float(np.mean(new < old)),
                    frac_later=float(np.mean(new > old)), empty=int(sum(x[2] == "empty" for x in rows)),
                    old_correct=float(np.mean([x[3] for x in rows])), new_correct=float(np.mean([x[4] for x in rows])),
                    mechanism=float(np.mean([x[5] for x in rows])))
                o = out[key]
                print(f"{key:20s} mech {o['mechanism']:.2f} | old {o['old_mean']:7.1f} new {o['new_mean']:7.1f} "
                      f"ratio {o['ratio']:.3f} p95 {o['old_p95']:.0f}->{o['new_p95']:.0f} earlier {o['frac_earlier']:.2f} "
                      f"later {o['frac_later']:.2f} empty {o['empty']} | correct {o['old_correct']:.3f}/{o['new_correct']:.3f}",
                      flush=True)
                json.dump(out, open(os.path.join(RESULTS_DIR, "domain.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        selftest()
    else:
        main(sys.argv[1:] or None)
