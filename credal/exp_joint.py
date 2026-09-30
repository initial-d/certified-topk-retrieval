"""Joint updates: does resolving one document help bound similar ones? (procedure-specific decision calibration)

Procedure P_theta (LUCB rule of Alg. 1 on a fixed candidate universe U = top-M options by the initial upper bound at
theta_max). When option i is resolved with standardized error eps_i = (s_i - sh_i)/w_i, every unresolved option j among
the L nearest neighbours of i (cosine of reconstructed vectors, within U) receives
    centre_j = sh_j + rho * mean(eps of its resolved neighbours) * w_j,   width_j = theta * kappa * w_j,
    l_j = centre_j, u_j = centre_j + width_j,     kappa = sqrt(1 - rho^2) once j has a resolved neighbour, else 1.
rho = 0 gives the baseline (Alg. 1 with residual boxes, lambda = 0) on the same universe and grid.

Calibration without monotonicity: on a grid Theta, alpha(q) = smallest theta in Theta such that P_theta' returns the exact
top-k for every theta' >= theta in Theta (+inf if none). This envelope is monotone by construction, so the conformal
quantile gives P(P_thetahat errs) <= delta for this specific procedure. rho is estimated on a tune split disjoint from the
calibration split. Costs are actual resolution counts on test queries.
Usage: python -m credal.exp_joint [dataset.quant ...]"""
import json
import os
import sys
import warnings

import numpy as np

from . import encode
from .credal import conformal_quantile, decision_nonconformity, topk_correct
from .data import RESULTS_DIR, load, self_mask
from .index import quantize

warnings.filterwarnings("ignore", category=RuntimeWarning)
K, DELTA, M, L = 10, 0.1, 3000, 10
GRID_N = 24


def _correct_global(s_u, R):
    """Correctness against the WHOLE corpus from the universe: s_u holds the universe scores followed by the global
    threshold appended by instance() (see _GLOBAL)."""
    return _GLOBAL["check"](R)


_GLOBAL = {}


def run_procedure(sh, w, s, nbr, theta, rho, kth, centered=False):
    """LUCB verify-before-certify with joint updates. Returns (resolutions, correct).
    centered=True uses errors relative to the running mean error of this query's resolved options (local component)."""
    n = len(sh)
    kappa = np.sqrt(1 - rho ** 2)
    shift = np.zeros(n)
    cnt = np.zeros(n)
    width = theta * w
    res = np.zeros(n, bool)
    l = sh.copy()
    u = sh + width
    steps = 0
    esum, ecnt = 0.0, 0
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
        steps += 1
        if rho != 0:
            e = (s[j] - sh[j]) / w[j]
            esum += e
            ecnt += 1
            if centered:
                e = e - esum / ecnt
            nb = nbr[j][~res[nbr[j]]]
            shift[nb] += e
            cnt[nb] += 1
            c = sh[nb] + rho * shift[nb] / cnt[nb] * w[nb]
            l[nb] = c
            u[nb] = c + width[nb] * kappa
    R = np.argpartition(-l, K - 1)[:K]
    return steps, _correct_global(s, R)


def prepare(name, q):
    ds = load(name)
    X, Q = encode.get(name, "bge", "docs"), encode.get(name, "bge", "queries")
    Xh, r, _ = quantize(q, X)
    Xn = Xh / np.linalg.norm(Xh, axis=1, keepdims=True)
    S, Sh = Q @ X.T, Q @ Xh.T
    sm = self_mask(ds)
    rows = np.flatnonzero(sm >= 0)
    S[rows, sm[rows]] = -1e9
    Sh[rows, sm[rows]] = -1e9
    qn = np.linalg.norm(Q, axis=1)
    alpha = np.array([decision_nonconformity(S[i], Sh[i], r, qn[i], K) for i in range(len(Q))])
    return ds, X, Xn, S, Sh, r, qn, alpha


def instance(i, S, Sh, r, qn, Xn, theta_max):
    w_all = qn[i] * r + 1e-8
    U = np.argsort(-(Sh[i] + theta_max * w_all))[:M]
    G = Xn[U] @ Xn[U].T
    np.fill_diagonal(G, -2)
    nbr = np.argsort(-G, 1)[:, :L]
    kth = np.partition(S[i], -K)[-K]
    _GLOBAL["check"] = lambda R, U=U, s_full=S[i]: topk_correct(s_full, U[np.asarray(R)], K)
    return Sh[i][U].astype(np.float64), w_all[U], S[i][U].astype(np.float64), nbr, kth


def estimate_rho(idx, S, Sh, r, qn, Xn, centered=False):
    a, b = [], []
    for i in idx:
        top = np.argsort(-Sh[i])[:500]
        e = (S[i][top] - Sh[i][top]) / (qn[i] * r[top] + 1e-8)
        if centered:
            e = e - e.mean()
        G = Xn[top] @ Xn[top].T
        np.fill_diagonal(G, -2)
        a.append(e)
        b.append(e[G.argmax(1)])
    return float(np.corrcoef(np.concatenate(a), np.concatenate(b))[0, 1])


def run(name, q, seed=0):
    ds, X, Xn, S, Sh, r, qn, alpha = prepare(name, q)
    rng = np.random.RandomState(seed)
    cal_all = rng.permutation(ds.calib_idx)
    tune, cal, te = cal_all[: len(cal_all) // 2], cal_all[len(cal_all) // 2:], ds.test_idx
    rho = estimate_rho(tune, S, Sh, r, qn, Xn)
    c_ref = conformal_quantile(alpha[cal], DELTA)
    grid = np.linspace(0, 1.6 * c_ref, GRID_N + 1)[1:]
    out = dict(dataset=name, quant=q, rho=rho, grid_max=float(grid[-1]))
    for lab, rr in (("baseline", 0.0), ("joint", rho)):
        # envelope nonconformity on calibration queries
        a = []
        for i in cal:
            sh, w, s, nbr, kth = instance(i, S, Sh, r, qn, Xn, grid[-1])
            ok = np.array([run_procedure(sh, w, s, nbr, th, rr, kth)[1] for th in grid])
            bad = np.flatnonzero(~ok)
            if len(bad) == 0:
                a.append(grid[0])
            elif bad[-1] == len(grid) - 1:
                a.append(np.inf)                  # wrong at the largest level: not certifiable on this grid
            else:
                a.append(grid[bad[-1] + 1])
        th = conformal_quantile(np.array(a), DELTA)
        if not np.isfinite(th):
            out[lab] = dict(theta=None)
            continue
        costs, corr = [], []
        for i in te:
            sh, w, s, nbr, kth = instance(i, S, Sh, r, qn, Xn, grid[-1])
            c, ok = run_procedure(sh, w, s, nbr, th, rr, kth)
            costs.append(c)
            corr.append(ok)
        costs = np.array(costs)
        out[lab] = dict(theta=float(th), mean=float(costs.mean()), p95=float(np.percentile(costs, 95)),
                        exact=float(np.mean(corr)), calib_inf=int(np.sum(~np.isfinite(a))))
    b, j = out["baseline"], out["joint"]
    if b.get("theta") and j.get("theta"):
        print(f"{name:9s} {q:5s} rho={rho:.3f} | baseline mean {b['mean']:7.1f} p95 {b['p95']:6.0f} exact {b['exact']:.3f} "
              f"| joint mean {j['mean']:7.1f} p95 {j['p95']:6.0f} exact {j['exact']:.3f} | ratio {j['mean'] / b['mean']:.3f}",
              flush=True)
    return out


def main():
    path = os.path.join(RESULTS_DIR, "joint.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for key in sys.argv[1:]:
        if key in res:
            continue
        name, q = key.split(".")
        res[key] = run(name, q)
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__" and sys.argv[1:2] not in (["paired"], ["v3"]):
    main()


# ------------------------------------------------------------------ fair paired evaluation over re-splits
def precompute(name, q, seed=0):
    """rho from a quarter of the queries (never used again); for the remaining queries, cost and correctness of both
    procedures at every grid level."""
    ds, X, Xn, S, Sh, r, qn, alpha = prepare(name, q)
    n = len(alpha)
    rng = np.random.RandomState(seed)
    p = rng.permutation(n)
    rho_idx, pool = p[: n // 4], p[n // 4:]
    rho = estimate_rho(rho_idx, S, Sh, r, qn, Xn)
    rho_c = estimate_rho(rho_idx, S, Sh, r, qn, Xn, centered=True)
    c_ref = conformal_quantile(alpha[pool], DELTA)
    grid = np.linspace(0, 1.6 * c_ref, GRID_N + 1)[1:]
    procs = (("baseline", 0.0, False), ("joint", rho, False), ("joint_c", rho_c, True))
    cost = {lab: np.zeros((len(pool), len(grid))) for lab, _, _ in procs}
    ok = {lab: np.zeros((len(pool), len(grid)), bool) for lab, _, _ in procs}
    for a, i in enumerate(pool):
        sh, w, s, nbr, kth = instance(i, S, Sh, r, qn, Xn, grid[-1])
        for b, th in enumerate(grid):
            for lab, rr, cen in procs:
                cost[lab][a, b], ok[lab][a, b] = run_procedure(sh, w, s, nbr, th, rr, kth, cen)
    return dict(rho=rho, rho_c=rho_c, grid=grid, cost=cost, ok=ok, n_pool=len(pool))


def envelope(okrow, grid):
    bad = np.flatnonzero(~okrow)
    if len(bad) == 0:
        return 0                                   # index into grid
    return np.inf if bad[-1] == len(grid) - 1 else bad[-1] + 1


def paired(pc, reps=200, seed=1):
    grid, n = pc["grid"], pc["n_pool"]
    rng = np.random.RandomState(seed)
    out = {lab: dict(cost=[], valid=[], pooled=[]) for lab in pc["cost"]}
    env = {lab: np.array([envelope(pc["ok"][lab][a], grid) for a in range(n)], float) for lab in out}
    for _ in range(reps):
        p = rng.permutation(n)
        cal, te = p[: n // 2], p[n // 2:]
        for lab in out:
            kq = conformal_quantile(env[lab][cal], DELTA)       # a grid index (or inf)
            if not np.isfinite(kq):
                c = np.full(len(te), np.nan)
                v = np.ones(len(te), bool)
            else:
                kq = int(kq)
                c = pc["cost"][lab][te, kq]
                v = pc["ok"][lab][te, kq]
            out[lab]["cost"].append(np.nanmean(c))
            out[lab]["valid"].append(v.mean())
            out[lab]["pooled"].extend(c.tolist())
    return {lab: dict(mean=float(np.nanmean(v["cost"])), valid=float(np.mean(v["valid"])),
                      p95_pool=float(np.nanpercentile(v["pooled"], 95)),
                      n_inf=int(np.sum(~np.isfinite(env[lab]))))
            for lab, v in out.items()} | dict(rho=pc["rho"], rho_c=pc.get("rho_c"),
                                              wins=float(np.mean(np.array(out["joint"]["cost"]) < np.array(out["baseline"]["cost"]))),
                                              wins_c=float(np.mean(np.array(out["joint_c"]["cost"]) < np.array(out["baseline"]["cost"]))))


def main_paired(keys):
    path = os.path.join(RESULTS_DIR, "joint_paired_v2.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for key in keys:
        if key in res:
            continue
        name, q = key.split(".")
        r = res[key] = paired(precompute(name, q))
        json.dump(res, open(path, "w"), indent=1)
        b, j, c = r["baseline"], r["joint"], r["joint_c"]
        print(f"{key:14s} rho={r['rho']:.3f} rho_c={r['rho_c']:.3f} | base {b['mean']:7.1f}/{b['p95_pool']:5.0f} v{b['valid']:.3f} "
              f"| joint {j['mean'] / b['mean']:.3f} | centered {c['mean']:7.1f}/{c['p95_pool']:5.0f} v{c['valid']:.3f} "
              f"ratio {c['mean'] / b['mean']:.3f} p95 ratio {c['p95_pool'] / b['p95_pool']:.3f} "
              f"(cheaper in {100 * r['wins_c']:.0f}% of splits)", flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["paired"]:
    main_paired(sys.argv[2:])


# ------------------------------------------------------------------ v3: boundary fixes, original baseline, ablation, timing
import time as _time

from .credal import refine as _refine


def run_procedure_v3(sh, w, s, nbr, theta, rho, kth, use_shift, use_shrink):
    """As run_procedure, with the two components of the joint update switchable (ablation)."""
    n = len(sh)
    kappa = np.sqrt(1 - rho ** 2) if use_shrink else 1.0
    shift, cnt = np.zeros(n), np.zeros(n)
    width = theta * w
    res = np.zeros(n, bool)
    l, u = sh.copy(), sh + width
    steps = 0
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
        steps += 1
        if rho != 0 and (use_shift or use_shrink):
            e = (s[j] - sh[j]) / w[j]
            nb = nbr[j][~res[nbr[j]]]
            shift[nb] += e
            cnt[nb] += 1
            c = sh[nb] + (rho * shift[nb] / cnt[nb] * w[nb] if use_shift else 0)
            l[nb] = c
            u[nb] = c + width[nb] * kappa
    R = np.argpartition(-l, K - 1)[:K]
    # correctness against the exact top-k of the WHOLE corpus: a winner outside the universe makes the run wrong
    return steps, _correct_global(s, R)


PROCS_V3 = (("grid_baseline", False, False), ("joint", True, True), ("center_only", True, False), ("width_only", False, True))


def precompute_v3(name, q, seed=0):
    ds, X, Xn, S, Sh, r, qn, alpha = prepare(name, q)
    n, N = S.shape
    rng = np.random.RandomState(seed)
    p = rng.permutation(n)
    rho_idx, pool = p[: n // 4], p[n // 4:]
    rho = estimate_rho(rho_idx, S, Sh, r, qn, Xn)
    c_ref = conformal_quantile(alpha[rho_idx], DELTA)              # grid range from the disjoint quarter only
    grid = np.linspace(0, 1.6 * c_ref, GRID_N + 1)[1:]
    cost = {lab: np.zeros((len(pool), len(grid))) for lab, _, _ in PROCS_V3}
    ok = {lab: np.zeros((len(pool), len(grid)), bool) for lab, _, _ in PROCS_V3}
    tsec = {lab: 0.0 for lab, _, _ in PROCS_V3}
    t_nbr = 0.0
    for a, i in enumerate(pool):
        t0 = _time.time()
        sh, w, s, nbr, kth = instance(i, S, Sh, r, qn, Xn, grid[-1])
        t_nbr += _time.time() - t0
        for b, th in enumerate(grid):
            for lab, sft, shr in PROCS_V3:
                t0 = _time.time()
                cost[lab][a, b], ok[lab][a, b] = run_procedure_v3(sh, w, s, nbr, th, rho if (sft or shr) else 0.0, kth,
                                                                  sft, shr)
                tsec[lab] += _time.time() - t0
    runs = len(pool) * len(grid)
    return dict(rho=rho, n_rho=len(rho_idx), grid=grid, cost=cost, ok=ok, pool=pool, N=N,
                alpha_pool=alpha[pool], S=S, Sh=Sh, r=r, qn=qn,
                ms_per_run={lab: 1000 * t / runs for lab, t in tsec.items()}, ms_neighbours=1000 * t_nbr / len(pool))


def paired_v3(pc, reps=200, seed=1):
    grid, pool, N = pc["grid"], pc["pool"], pc["N"]
    n = len(pool)
    rng = np.random.RandomState(seed)
    labs = [lab for lab, _, _ in PROCS_V3]
    env = {lab: np.array([envelope(pc["ok"][lab][a], grid) for a in range(n)], float) for lab in labs}
    splits = [rng.permutation(n) for _ in range(reps)]
    # original method: decision-calibrated residual boxes on the full corpus, continuous level, Algorithm 1 (LUCB)
    levels = {}
    for p in splits:
        c = conformal_quantile(pc["alpha_pool"][p[: n // 2]], DELTA)
        levels.setdefault(c, None)
    orig_cost, orig_ok = {}, {}
    S, Sh, r, qn = pc["S"], pc["Sh"], pc["r"], pc["qn"]
    for c in levels:
        cc, oo = np.zeros(n), np.zeros(n, bool)
        for a, i in enumerate(pool):
            kth = np.partition(S[i], -K)[-K]
            l, u = Sh[i].copy(), Sh[i] + c * qn[i] * r
            res = _refine(l, u, S[i], K, "lucb")
            cc[a] = res["n_ref"]
            oo[a] = topk_correct(S[i], res["topk"], K)
        orig_cost[c], orig_ok[c] = cc, oo
    out = {lab: dict(cost=[], valid=[], pooled=[], fallback=0) for lab in labs + ["original"]}
    for p in splits:
        cal, te = p[: n // 2], p[n // 2:]
        for lab in labs:
            kq = conformal_quantile(env[lab][cal], DELTA)
            if not np.isfinite(kq):          # fallback: full verification of the whole corpus, always correct
                c, v = np.full(len(te), float(N)), np.ones(len(te), bool)
                out[lab]["fallback"] += 1
            else:
                c, v = pc["cost"][lab][te, int(kq)], pc["ok"][lab][te, int(kq)]
            out[lab]["cost"].append(c.mean())
            out[lab]["valid"].append(v.mean())
            out[lab]["pooled"].extend(c.tolist())
        c0 = conformal_quantile(pc["alpha_pool"][cal], DELTA)
        out["original"]["cost"].append(orig_cost[c0][te].mean())
        out["original"]["valid"].append(orig_ok[c0][te].mean())
        out["original"]["pooled"].extend(orig_cost[c0][te].tolist())
    res = {lab: dict(mean=float(np.mean(v["cost"])), valid=float(np.mean(v["valid"])),
                     p95_pool=float(np.percentile(v["pooled"], 95)), fallback=v["fallback"])
           for lab, v in out.items()}
    base = np.array(out["original"]["cost"])
    for lab in labs:
        res[lab]["wins_vs_original"] = float(np.mean(np.array(out[lab]["cost"]) < base))
    res["wins_joint_vs_grid"] = float(np.mean(np.array(out["joint"]["cost"]) < np.array(out["grid_baseline"]["cost"])))
    res.update(rho=pc["rho"], n_rho=pc["n_rho"], n_pool=n, grid_max=float(grid[-1]), grid_step=float(grid[1] - grid[0]),
               ms_per_run=pc["ms_per_run"], ms_neighbours=pc["ms_neighbours"], n_orig_levels=len(levels))
    return res


def main_v3(keys):
    path = os.path.join(RESULTS_DIR, "joint_v3.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for key in keys:
        if key in res:
            continue
        name, q = key.split(".")
        r = res[key] = paired_v3(precompute_v3(name, q))
        json.dump(res, open(path, "w"), indent=1)
        o, g, j, c, w_ = (r[x] for x in ("original", "grid_baseline", "joint", "center_only", "width_only"))
        print(f"{key:14s} rho={r['rho']:.3f} | original {o['mean']:7.1f}/{o['p95_pool']:5.0f} v{o['valid']:.3f} | grid "
              f"{g['mean'] / o['mean']:.3f} | joint {j['mean'] / o['mean']:.3f} (p95 {j['p95_pool'] / o['p95_pool']:.3f}, "
              f"v{j['valid']:.3f}, fb {j['fallback']}) | center {c['mean'] / o['mean']:.3f} | width {w_['mean'] / o['mean']:.3f} "
              f"| ms/run grid {r['ms_per_run']['grid_baseline']:.1f} joint {r['ms_per_run']['joint']:.1f} nbr {r['ms_neighbours']:.0f}",
              flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["v3"]:
    main_v3(sys.argv[2:])
