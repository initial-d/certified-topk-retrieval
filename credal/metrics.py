"""Retrieval metrics, QPP baselines, and predictor evaluation."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------------- retrieval metrics
def ndcg_at_k(rel_topk, all_rels, k=10):
    """rel_topk: (nq, >=k) graded relevance of the ranking; all_rels: list of per-query grade arrays."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = ((2 ** rel_topk[:, :k] - 1) * disc).sum(1)
    idcg = np.array([((2 ** np.sort(g)[::-1][:k] - 1) * disc[:min(k, len(g))]).sum() for g in all_rels])
    return dcg / np.maximum(idcg, 1e-12)


def recall_at_k(rel_topk, n_rel, k):
    return (rel_topk[:, :k] > 0).sum(1) / np.maximum(n_rel, 1)


def topk_search(Q, X, k, mask_idx=None):
    """Exact inner-product top-k (Q: (nq,D), X: (N,D)); returns (idx, scores, full score matrix)."""
    S = Q @ X.T
    if mask_idx is not None:
        rows = np.flatnonzero(mask_idx >= 0)
        S[rows, mask_idx[rows]] = -np.inf
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    sc = np.take_along_axis(S, idx, 1)
    o = np.argsort(-sc, 1)
    return np.take_along_axis(idx, o, 1), np.take_along_axis(sc, o, 1), S


# ----------------------------------------------------------------------------- QPP baselines
# All predictors are oriented a priori so that HIGHER = MORE CONFIDENT (no sign fitting on test data).
def qpp_score_based(top_scores, corpus_score, k_nqc=100, k_wig=5, k_smv=100):
    s = top_scores
    out = {}
    out["MaxScore"] = s[:, 0]
    out["NQC"] = s[:, :k_nqc].std(1) / np.abs(corpus_score)
    out["WIG"] = s[:, :k_wig].mean(1) - corpus_score
    sm = s[:, :k_smv]
    mu = sm.mean(1, keepdims=True)
    out["SMV"] = (sm * np.abs(np.log(np.clip(sm, 1e-6, None) / np.clip(mu, 1e-6, None)))).mean(1) / np.abs(corpus_score)
    return out


def qpp_perturbation(Q, X, top10, sigma_norm=0.1, T=10, seed=0, mask_idx=None):
    """Dense perturbation robustness: mean Jaccard of top-10 under isotropic query noise."""
    rng = np.random.RandomState(seed)
    D = Q.shape[1]
    jac = np.zeros(len(Q))
    base = [set(r) for r in top10]
    for _ in range(T):
        Qn = Q + rng.randn(*Q.shape).astype(np.float32) * sigma_norm / np.sqrt(D)
        Qn /= np.linalg.norm(Qn, axis=1, keepdims=True)
        idx, _, _ = topk_search(Qn, X, 10, mask_idx)
        jac += np.array([len(b & set(r)) / len(b | set(r)) for b, r in zip(base, idx)])
    return jac / T


def qpp_original_paper(Q, Qhat, X, r_corpus, nn_idx, K=10, eps=1e-6):
    """Geometric baseline: quantization stability S_q, neighbourhood density N_q, and their harmonic mean C_q."""
    sigma2 = (r_corpus ** 2).mean()
    S_q = np.exp(-np.linalg.norm(Q - Qhat, axis=1) ** 2 / (2 * sigma2))
    d2 = ((Q[:, None, :] - X[nn_idx[:, :K]]) ** 2).sum(-1)
    N_q = K / (d2.sum(1) + eps)
    C_q = 2 * S_q * N_q / (S_q + N_q)
    return dict(OrigS=S_q, OrigN=N_q, OrigC=C_q)


# ----------------------------------------------------------------------------- predictor evaluation
def aurc(conf, util):
    """Area under the risk-coverage curve (risk = 1 - util), abstaining on least confident first.
    Lower is better. Also returns the oracle AURC for normalization."""
    def _curve(order):
        u = util[order]
        return (1 - np.cumsum(u) / np.arange(1, len(u) + 1)).mean()
    return _curve(np.argsort(-conf, kind="stable")), _curve(np.argsort(-util, kind="stable"))


def evaluate_predictor(conf, ndcg, fail):
    ok = np.isfinite(conf)
    conf, ndcg, fail = conf[ok], ndcg[ok], fail[ok]
    a, a_star = aurc(conf, ndcg)
    rand = 1 - ndcg.mean()
    res = dict(
        pearson=stats.pearsonr(conf, ndcg)[0],
        kendall=stats.kendalltau(conf, ndcg)[0],
        spearman=stats.spearmanr(conf, ndcg)[0],
        auroc_fail=roc_auc_score(fail, -conf) if 0 < fail.sum() < len(fail) else np.nan,
        aurc=a,
        # fraction of the gap between random abstention and the oracle that is closed
        e_aurc=(rand - a) / max(rand - a_star, 1e-12),
    )
    return res
