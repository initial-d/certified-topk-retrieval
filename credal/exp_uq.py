"""E2 + E4: credal uncertainty decomposition for query performance prediction, selective retrieval
(abstention) and credal adaptive-k retrieval sets.

Credal layers per query q (pool P(q) = union of the members' exact top-100):
  B  (representation) : K_B = conv{ softmax(s^m / tau_m) : m = 1..M }, M embedding models
  A  (index)          : PQ score box of the primary model, decision-calibrated (see exp_certify)
  AB (combined)       : probability-interval envelope of every member's index box
Usage: python -m credal.exp_uq [datasets...]
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

from . import encode
from .credal import (conformal_quantile, decision_nonconformity, entropy, hull_measures, hull_undominated,
                     intervals, lower_entropy_pri, softmax_intervals, undominated, upper_entropy_pri)
from .data import DATASETS, RESULTS_DIR, load, self_mask
from .index import quantize, quantize_queries
from .metrics import (evaluate_predictor, ndcg_at_k, qpp_original_paper, qpp_perturbation, qpp_score_based,
                      recall_at_k, topk_search)

warnings.filterwarnings("ignore", category=RuntimeWarning)
MODELS = list(encode.ENSEMBLE)
PRIMARY = encode.PRIMARY
POOL_K = 100
K = 10
QUANT = "pq32"
DELTA = 0.1
LAM = 0.0
TAU_GRID = np.logspace(-3, 0, 61)


def fit_tau(Sp, rel_pool, idx):
    """Listwise MLE of the softmax temperature: maximize sum_q log P(relevant | q) on calibration queries."""
    best, arg = -np.inf, None
    for tau in TAU_GRID:
        ll = 0.0
        for i in idx:
            if rel_pool[i].sum() == 0:
                continue
            z = Sp[i] / tau
            z = z - z.max()
            p = np.exp(z) / np.exp(z).sum()
            ll += np.log(max(p[rel_pool[i] > 0].sum(), 1e-300))
        if ll > best:
            best, arg = ll, tau
    return arg


def run(name):
    ds = load(name)
    nq = len(ds.query_ids)
    selfm = self_mask(ds)
    did_pos = {d: i for i, d in enumerate(ds.doc_ids)}
    rel_sets = [{did_pos[d]: g for d, g in ds.qrels[q].items() if d in did_pos} for q in ds.query_ids]
    all_grades = [np.array(list(r.values()), dtype=float) for r in rel_sets]
    n_rel = np.array([len(r) for r in rel_sets])

    Xs, Qs, tops, full = {}, {}, {}, {}
    for m in MODELS:
        Xs[m] = encode.get(name, m, "docs")
        Qs[m] = encode.get(name, m, "queries")
        idx, sc, S = topk_search(Qs[m], Xs[m], POOL_K, selfm)
        tops[m] = (idx, sc)
        full[m] = S

    # ---- pools and pooled scores
    pools = [np.unique(np.concatenate([tops[m][0][i] for m in MODELS])) for i in range(nq)]
    Sp = {m: [full[m][i, pools[i]] for i in range(nq)] for m in MODELS}
    rel_pool = [np.array([rel_sets[i].get(int(d), 0) for d in pools[i]], dtype=float) for i in range(nq)]

    # ---- primary retrieval quality (targets)
    def rel_of(idx):
        return np.array([[rel_sets[i].get(int(d), 0) for d in idx[i]] for i in range(nq)], dtype=float)
    rel_primary = rel_of(tops[PRIMARY][0])
    ndcg = ndcg_at_k(rel_primary, all_grades, K)
    fail = (recall_at_k(rel_primary, n_rel, K) == 0).astype(int)

    cal, te = ds.calib_idx, ds.test_idx
    taus = {m: fit_tau(Sp[m], rel_pool, cal) for m in MODELS}

    # fused (credal-centre) ranking: equal-weight mixture of calibrated members
    def member_probs(i):
        P = []
        for m in MODELS:
            z = Sp[m][i] / taus[m]
            z = np.exp(z - z.max())
            P.append(z / z.sum())
        return np.array(P)
    fused_idx = np.zeros((nq, K), dtype=int)
    for i in range(nq):
        pbar = member_probs(i).mean(0)
        fused_idx[i] = pools[i][np.argsort(-pbar)[:K]]
    ndcg_fused = ndcg_at_k(rel_of(fused_idx), all_grades, K)

    # ---- baselines on the primary model
    Xp, Qp = Xs[PRIMARY], Qs[PRIMARY]
    corpus_score = Qp @ Xp.mean(0)
    pred = qpp_score_based(tops[PRIMARY][1], corpus_score)
    pred["Perturb"] = qpp_perturbation(Qp, Xp, tops[PRIMARY][0][:, :K], mask_idx=selfm)
    Xh, r, qobj = quantize(QUANT, Xp)
    Qh = quantize_queries(qobj, QUANT, Qp)
    pred.update(qpp_original_paper(Qp, Qh, Xp, r, tops[PRIMARY][0], K=10))

    # ---- index layer A (primary): decision-calibrated PQ box
    Sfull, Shfull = full[PRIMARY], Qp @ Xh.T
    Sfull = np.where(np.isfinite(Sfull), Sfull, -1e9)
    Shfull[Sfull <= -1e9] = -1e9
    qn = np.linalg.norm(Qp, axis=1)
    alpha = np.array([decision_nonconformity(Sfull[i], Shfull[i], r, qn[i], K) for i in range(nq)])
    c = conformal_quantile(alpha[cal], DELTA)
    UA = np.zeros(nq)
    HA_up, HA_lo = np.zeros(nq), np.zeros(nq)
    for i in range(nq):
        l, u = intervals(Shfull[i], r, qn[i], c, LAM)
        UA[i] = len(undominated(l, u, K))
        top = np.argsort(-Shfull[i])[:POOL_K]
        lo, hi = softmax_intervals(l[top], u[top], taus[PRIMARY])
        HA_up[i], HA_lo[i] = upper_entropy_pri(lo, hi), lower_entropy_pri(lo, hi)

    # member index boxes for the combined AB layer
    boxes = {}
    for m in MODELS:
        if m == PRIMARY:
            boxes[m] = (Shfull, r, c)
            continue
        Xh_m, r_m, _ = quantize(QUANT, Xs[m])
        Sh_m = Qs[m] @ Xh_m.T
        S_m = np.where(np.isfinite(full[m]), full[m], -1e9)
        Sh_m[S_m <= -1e9] = -1e9
        qn_m = np.linalg.norm(Qs[m], axis=1)
        al = np.array([decision_nonconformity(S_m[i], Sh_m[i], r_m, qn_m[i], K) for i in cal])
        boxes[m] = (Sh_m, r_m, conformal_quantile(al, DELTA))

    # ---- representation layer B, combined AB, and adaptive-k sets
    keys = ["H_up", "H_lo", "GH", "TU", "AU", "EU", "UB", "EnsJac", "HAB_up", "HAB_lo", "H_primary"]
    B = {k: np.zeros(nq) for k in keys}
    ub_recall, ub_size, cons_prec, cons_size = np.zeros(nq), np.zeros(nq), np.full(nq, np.nan), np.zeros(nq)
    fused_rank_rel = []  # relevance of the fused ranking over the whole pool, for matched-budget comparison
    for i in range(nq):
        P = member_probs(i)
        hm = hull_measures(P)
        B["H_primary"][i] = entropy(P[MODELS.index(PRIMARY)])
        for k_ in ("H_up", "H_lo", "GH", "TU", "AU", "EU"):
            B[k_][i] = hm[k_]
        S_i = np.array([Sp[m][i] for m in MODELS])
        UB = hull_undominated(S_i, K)
        B["UB"][i] = len(UB)
        rel_ub = rel_pool[i][UB]
        ub_recall[i] = (rel_ub > 0).sum() / max(n_rel[i], 1)
        ub_size[i] = len(UB)
        tops10 = [set(np.argsort(-S_i[m])[:K].tolist()) for m in range(len(MODELS))]
        jac = [len(a & b) / len(a | b) for j, a in enumerate(tops10) for b in tops10[j + 1:]]
        B["EnsJac"][i] = np.mean(jac)
        cons = set.intersection(*tops10)
        cons_size[i] = len(cons)
        if cons:
            cons_prec[i] = (rel_pool[i][list(cons)] > 0).mean()
        fused_rank_rel.append(rel_pool[i][np.argsort(-P.mean(0))])
        # AB: envelope of member probability intervals over the pool
        los, his = [], []
        for m in MODELS:
            Sh_m, r_m, c_m = boxes[m]
            l, u = intervals(Sh_m[i, pools[i]], r_m[pools[i]], 1.0, c_m, LAM)
            lo, hi = softmax_intervals(l, u, taus[m])
            los.append(lo)
            his.append(hi)
        lo, hi = np.min(los, 0), np.max(his, 0)
        B["HAB_up"][i], B["HAB_lo"][i] = upper_entropy_pri(lo, hi), lower_entropy_pri(lo, hi)

    # oriented so that higher = more confident
    pred.update({
        "Entropy": -B["H_primary"], "EnsJac": B["EnsJac"], "Bayes-TU": -B["TU"], "Bayes-AU": -B["AU"], "Bayes-EU": -B["EU"],
        "Credal-B-Hup": -B["H_up"], "Credal-B-Hlo": -B["H_lo"], "Credal-B-GH": -B["GH"], "Credal-B-|U|": -B["UB"],
        "Credal-A-|U|": -UA, "Credal-A-Hup": -HA_up, "Credal-A-GH": -(HA_up - HA_lo),
        "Credal-AB-Hup": -B["HAB_up"], "Credal-AB-GH": -(B["HAB_up"] - B["HAB_lo"]),
    })

    res = dict(dataset=name, n_cal=len(cal), n_test=len(te), taus=taus, c_index=c,
               ndcg_primary=float(ndcg[te].mean()), ndcg_fused=float(ndcg_fused[te].mean()),
               fail_rate=float(fail[te].mean()), eval={}, eval_fused={})
    fail_f = np.array([1 if (fr[:K] > 0).sum() == 0 else 0 for fr in fused_rank_rel])
    for k_, v in pred.items():
        res["eval"][k_] = evaluate_predictor(v[te], ndcg[te], fail[te])
        res["eval_fused"][k_] = evaluate_predictor(v[te], ndcg_fused[te], fail_f[te])

    # ---- E4: credal adaptive-k versus fixed k at a matched average budget
    mk = ub_size[te].mean()
    fixed = {}
    for kk in sorted({int(np.floor(mk)), int(np.ceil(mk))}):
        fixed[kk] = float(np.mean([(fr[:kk] > 0).sum() / max(n_rel[i], 1)
                                   for i, fr in zip(te, [fused_rank_rel[j] for j in te])]))
    res["adaptive_k"] = dict(ub_size_mean=float(mk), ub_size_median=float(np.median(ub_size[te])),
                             ub_recall=float(ub_recall[te].mean()), fixed_k_recall=fixed,
                             fixed10_recall=float(np.mean([(fused_rank_rel[j][:K] > 0).sum() / max(n_rel[j], 1) for j in te])),
                             consensus_size=float(cons_size[te].mean()),
                             consensus_precision=float(np.nanmean(cons_prec[te])),
                             fused10_precision=float(np.mean([(fused_rank_rel[j][:K] > 0).mean() for j in te])))
    np.savez_compressed(os.path.join(RESULTS_DIR, "uq", f"{name}.npz"),
                        pools=np.array(pools, dtype=object), rel_pool=np.array(rel_pool, dtype=object),
                        **{f"S_{m}": np.array(Sp[m], dtype=object) for m in MODELS},
                        ndcg=ndcg, fail=fail, cal=cal, te=te, n_rel=n_rel,
                        **{f"pred_{k_}": v for k_, v in pred.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()
    os.makedirs(os.path.join(RESULTS_DIR, "uq"), exist_ok=True)
    for name in args.datasets:
        path = os.path.join(RESULTS_DIR, "uq", f"{name}.json")
        if os.path.exists(path):
            continue
        t = time.time()
        res = run(name)
        json.dump(res, open(path, "w"), default=float)
        print(f"[E2] {name} nDCG={res['ndcg_primary']:.3f} fused={res['ndcg_fused']:.3f} ({time.time()-t:.0f}s)")
        for k_, v in sorted(res["eval"].items(), key=lambda kv: -kv[1]["kendall"]):
            print(f"   {k_:15s} tau={v['kendall']:+.3f} r={v['pearson']:+.3f} AUROC={v['auroc_fail']:.3f} "
                  f"eAURC={v['e_aurc']:+.3f}")
        print("   adaptive-k:", res["adaptive_k"], flush=True)


if __name__ == "__main__":
    main()
