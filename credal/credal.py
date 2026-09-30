"""Credal retrieval core.

Notation (per query q, corpus index d):
  s_d   exact score <q, x_d>            (unknown to the index)
  sh_d  approximate score <q, x_hat_d>  (what the ANN index computes)
  r_d   residual norm ||x_d - x_hat_d|| (stored with the code, 1 float)
  [l_d, u_d] = [sh_d - lam c ||q|| r_d, sh_d + c ||q|| r_d]   score interval; c = 1, lam = 1 is the
               Cauchy-Schwarz (deterministic) box, c = c_delta the decision-calibrated conformal constant.
"""
import math

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------------- score intervals
def conformal_quantile(scores, delta):
    """Split-conformal quantile: the ceil((n+1)(1-delta))-th smallest score (inf if it does not exist)."""
    scores = np.sort(np.asarray(scores))
    n = len(scores)
    k = math.ceil((n + 1) * (1 - delta))
    return np.inf if k > n else float(scores[k - 1])


def nonconformity(S, Sh, r, qnorm, mask=None, eta=1e-8):
    """Per-query simultaneous nonconformity  max_d |s_d - sh_d| / (||q|| r_d + eta).
    S, Sh: (nq, N); mask: (nq, N) bool of entries to ignore (e.g. query==doc in ArguAna)."""
    A = np.abs(S - Sh) / (qnorm[:, None] * r[None, :] + eta)
    if mask is not None:
        A = np.where(mask, 0.0, A)
    return A.max(1)


def decision_nonconformity(s, sh, r, qnorm, k, eta=1e-8):
    """Smallest c such that condition (*) of Thm. 2 holds for this query:
    every exact top-k document t has  u_t = sh_t + c ||q|| r_t >= s_(k+1).
    Under (*), every certificate issued by `refine` (verify-before-certify) is exactly the top-k."""
    order = np.argsort(-s)
    T = order[:k]
    return max(float(((s[order[k]] - sh[T]) / (qnorm * r[T] + eta)).max()), 0.0)


def intervals(sh, r, qnorm, c, lam=0.0):
    """[sh - lam c ||q|| r, sh + c ||q|| r]. Correctness only needs the upper side (Thm. 2);
    the lower side only prunes, so lam trades pruning power against nothing but efficiency."""
    w = c * qnorm * r
    return sh - lam * w, sh + w


# ----------------------------------------------------------------------------- tie-aware correctness
TIE_TOL = 1e-6


def topk_correct(s, R, k):
    """R is a correct top-k iff |R| = k and min_R s >= max_{not R} s (all options strictly above s_(k) are returned; the
    remaining slots hold options tied at s_(k))."""
    R = np.asarray(R)
    out = np.ones(len(s), bool)
    out[R] = False
    return bool(len(R) == k and s[R].min() >= s[out].max() - TIE_TOL)


def topk_hits(s, R, k):
    """Number of the k slots filled correctly: all returned options strictly above tau = s_(k) count, and returned options
    tied at tau count up to the number of slots left for ties."""
    tau = np.partition(s, -k)[-k]
    inR = np.zeros(len(s), bool)
    inR[np.asarray(R)] = True
    above = s > tau + TIE_TOL
    tied = np.abs(s - tau) <= TIE_TOL
    return int((above & inR).sum() + min((tied & inR).sum(), k - above.sum()))


def required_depth(s, sh, k):
    """Smallest R such that the top-R by cheap score contains a correct top-k (tie-aware)."""
    order = np.argsort(-sh, kind="stable")
    tau = np.partition(s, -k)[-k]
    so = s[order]
    n_above = int((s > tau + TIE_TOL).sum())
    ok = (np.cumsum(so > tau + TIE_TOL) == n_above) & (np.cumsum(np.abs(so - tau) <= TIE_TOL) >= k - n_above)
    return int(np.argmax(ok) + 1)


# ----------------------------------------------------------------------------- credal top-k
def kth_largest(a, k):
    return np.partition(a, -k)[-k]


def undominated(l, u, k):
    """Indices of documents not k-dominated under Walley maximality:
    d is k-dominated iff at least k documents j satisfy l_j > u_d  (<=> u_d < L_k)."""
    return np.flatnonzero(u >= kth_largest(l, k))


def oracle_lower_bound(u, s, k):
    """Refinements that ANY verify-before-certify run must perform, and that an oracle achieves:
    the k true top documents plus every other document whose upper bound reaches s_(k)."""
    order = np.argsort(-s)
    notT = np.ones(len(s), bool)
    notT[order[:k]] = False
    return int(k + (notT & (u >= s[order[k - 1]])).sum())


def refine(l, u, s, k, strategy="lucb", rng=None, budget=None, resolved=None):
    """Credal shrinkage by exact re-scoring (Alg. 1): collapses [l_d, u_d] to {s_d} one document at a time.

    Verify-before-certify: stops when the k documents with the largest lower bounds are all exact and
    every other document's upper bound is below their minimum L_k (checked over the whole corpus).
    Only a prefix of the documents sorted by initial upper bound is ever touched (the working set);
    it is extended whenever L_k drops below its threshold, so the global check stays exact."""
    N = len(l)
    l, u = l.astype(np.float64).copy(), u.astype(np.float64).copy()
    exact = np.zeros(N, bool)
    if resolved is not None:            # options already resolved before the run (e.g. cascade anchors)
        exact[resolved] = True
        l[resolved] = u[resolved] = s[resolved]
    order_u = np.argsort(-u)
    neg_u_sorted = -u[order_u]          # ascending; upper bounds after pre-resolution (never change outside W)
    n_ref, certified = 0, False
    w = int(np.searchsorted(neg_u_sorted, -kth_largest(l, k), side="right"))
    U0 = w
    while True:
        W = order_u[:w]
        lW = l[W]
        top_local = np.argpartition(-lW, k - 1)[:k]
        Lk = lW[top_local].min()
        need = int(np.searchsorted(neg_u_sorted, -Lk, side="right"))
        if need > w:                    # L_k decreased (an over-estimated lower bound was refined)
            w = need
            continue
        inT = np.zeros(w, bool)
        inT[top_local] = True
        uW, eW = u[W], exact[W]
        chall = ~inT & (uW >= Lk)
        verify = inT & ~eW
        if not chall.any() and not verify.any():
            certified = True
            break
        if budget is not None and n_ref >= budget:
            break
        cand = verify | (chall & ~eW)
        if not cand.any():
            break                       # only exact ties remain
        if strategy == "lucb":
            # weakest believed-top vs strongest challenger; refine the wider unrefined one
            opts = []
            if verify.any():
                opts.append(np.where(verify, lW, np.inf).argmin())
            if (chall & ~eW).any():
                opts.append(np.where(chall & ~eW, uW, -np.inf).argmax())
            j = max(opts, key=lambda t: uW[t] - lW[t])
        elif strategy in ("upper", "score"):   # upper-bound order (compared with LUCB in the paper) (with lam = 0: sh + c r, not the plain sh order)
            j = np.where(cand, uW, -np.inf).argmax()   # for lam = 0, same order as sh)
        elif strategy == "widest":
            j = np.where(cand, uW - lW, -np.inf).argmax()
        elif strategy == "random":
            j = rng.choice(np.flatnonzero(cand))
        else:
            raise ValueError(strategy)
        d = W[j]
        l[d] = u[d] = s[d]
        exact[d] = True
        n_ref += 1
    W = order_u[:w]
    topk = W[np.argpartition(-l[W], k - 1)[:k]]
    return dict(n_ref=n_ref, certified=certified, topk=topk, U0=U0)


# ----------------------------------------------------------------------------- probability level
def softmax_intervals(l, u, tau):
    """Tight lower/upper probabilities of p = softmax(s / tau) over s in the box [l, u] (Prop. 2)."""
    a, b = l / tau, u / tau
    m = b.max()
    ea, eb = np.exp(a - m), np.exp(b - m)
    Sa, Sb = ea.sum(), eb.sum()
    # far-away documents underflow to 0/0; their probability bounds are 0
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.nan_to_num(ea / (ea + np.maximum(Sb - eb, 0)), nan=0.0)
        hi = np.nan_to_num(eb / (eb + np.maximum(Sa - ea, 0)), nan=0.0)
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def entropy(p, axis=-1):
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log(p)).sum(axis)


def upper_entropy_pri(lo, hi, iters=60):
    """Max entropy over {p : lo <= p <= hi, sum p = 1}: p_i = clip(t, lo_i, hi_i) (water filling)."""
    a, b = 0.0, 1.0
    for _ in range(iters):
        t = (a + b) / 2
        if np.clip(t, lo, hi).sum() > 1:
            b = t
        else:
            a = t
    return entropy(np.clip((a + b) / 2, lo, hi))


def lower_entropy_pri(lo, hi):
    """Greedy vertex of the probability-interval set (mass to largest uppers first).
    It is a feasible point, hence an upper bound on the true minimum entropy."""
    p = lo.copy()
    rest = 1.0 - p.sum()
    for i in np.argsort(-hi):
        add = min(hi[i] - lo[i], rest)
        p[i] += add
        rest -= add
        if rest <= 0:
            break
    return entropy(p)


# ----------------------------------------------------------------------------- finitely generated credal sets
def upper_entropy_hull(P, iters=300, lr=0.5):
    """max_{w in simplex} H(w @ P) for extreme points P (M, n); concave -> exponentiated gradient."""
    M = P.shape[0]
    w = np.full(M, 1.0 / M)
    best = entropy(P).max()
    for _ in range(iters):
        p = np.clip(w @ P, EPS, 1)
        g = -(P * (np.log(p) + 1)).sum(1)
        w = w * np.exp(lr * (g - g.max()))
        w /= w.sum()
    return max(best, entropy(w @ P))


def hull_measures(P):
    """Credal (upper/lower entropy) and Bayesian (mutual information) decompositions."""
    Hm = entropy(P)
    H_up = upper_entropy_hull(P)
    H_lo = Hm.min()
    TU = entropy(P.mean(0))
    AU = Hm.mean()
    return dict(H_up=H_up, H_lo=H_lo, GH=H_up - H_lo, TU=TU, AU=AU, EU=TU - AU)


def hull_undominated(S, k):
    """k-undominated set of a finite credal set generated by score vectors S (M, n).
    Softmax is monotone per member, so p^m_j > p^m_d for all m  <=>  s^m_j > s^m_d for all m,
    independently of the members' temperatures."""
    dom = np.ones((S.shape[1], S.shape[1]), bool)  # dom[j, d]: j dominates d
    for m in range(S.shape[0]):
        dom &= S[m][:, None] > S[m][None, :]
    return np.flatnonzero(dom.sum(0) < k)
