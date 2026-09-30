"""E1 at MS MARCO scale (8.8M passages, primary model), streaming on MPS/CPU with torch.

Pass A (once): exact top-100 per query (fp32 scores of the fp16-stored vectors, which define "exact").
Pass B (per quantizer): streaming over corpus chunks, per query
  - alpha_sim = max_d |s_d - sh_d| / (||q|| r_d)                       (simultaneous calibration)
  - R(q) = 1 + #{d : sh_d > min_{t in T} sh_t}                         (conformal fixed depth)
  - exact oracle cost |A(q)| = k + #{d not in T : sh_d + c ||q|| r_d >= s_(k)} at the decision-calibrated constants
Pass C: the same count at the simultaneously calibrated constants (known only after pass B).
Alg. 1 itself is not run at this scale; its cost is the oracle cost to within 1.4% on BEIR (Table 4).
Usage: python -m credal.exp_msmarco passA | quant <q> | passB <q> | report   (quant before passB)
"""
import json
import os
import sys
import time

import numpy as np

from . import msmarco
from .credal import conformal_quantile
from .data import CACHE_DIR, RESULTS_DIR

K = 10
TOP = 100
DOC_CHUNK = 200_000
Q_BATCH = 512
# smoke test: MS_LIMIT=<docs>,<queries> restricts the corpus prefix and the number of queries
LIMIT = tuple(int(x) for x in os.environ["MS_LIMIT"].split(",")) if os.environ.get("MS_LIMIT") else None
DELTAS = [0.05, 0.1, 0.2]
OUT = os.path.join(RESULTS_DIR, "msmarco_smoke" if os.environ.get("MS_LIMIT") else "msmarco")


def _torch():
    """torch and faiss each ship an OpenMP runtime; never load both in one process (quantization runs separately)."""
    import torch
    return torch, torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def queries():
    Q = np.load(os.path.join(CACHE_DIR, "msmarco.bge.queries.dev.npy")).astype(np.float32)
    if LIMIT:
        Q = Q[: LIMIT[1]]
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(Q))
    return Q, np.sort(perm[: len(Q) // 2]), np.sort(perm[len(Q) // 2:])


def corpus():
    X = msmarco.corpus()
    return X[: LIMIT[0]] if LIMIT else X


def chunks(n):
    for a in range(0, n, DOC_CHUNK):
        yield a, min(n, a + DOC_CHUNK)


def passA():
    torch, DEV = _torch()
    X = corpus()
    Q, _, _ = queries()
    nq, N = len(Q), X.shape[0]
    best_s = np.full((nq, TOP), -np.inf, np.float32)
    best_i = np.zeros((nq, TOP), np.int64)
    xsum = np.zeros(X.shape[1], np.float64)
    t = time.time()
    for a, b in chunks(N):
        Xc = torch.from_numpy(np.asarray(X[a:b], dtype=np.float32)).to(DEV)
        xsum += Xc.sum(0).cpu().numpy().astype(np.float64)
        for qa in range(0, nq, Q_BATCH):
            Qb = torch.from_numpy(Q[qa:qa + Q_BATCH]).to(DEV)
            S = Qb @ Xc.T
            v, i = torch.topk(S, TOP, dim=1)
            cs = np.concatenate([best_s[qa:qa + Q_BATCH], v.cpu().numpy()], 1)
            ci = np.concatenate([best_i[qa:qa + Q_BATCH], i.cpu().numpy() + a], 1)
            o = np.argsort(-cs, 1)[:, :TOP]
            best_s[qa:qa + Q_BATCH] = np.take_along_axis(cs, o, 1)
            best_i[qa:qa + Q_BATCH] = np.take_along_axis(ci, o, 1)
        print(f"[passA] {b:,}/{N:,} ({time.time() - t:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "passA.npz"), top_s=best_s, top_i=best_i, xmean=(xsum / N).astype(np.float32))


def quantize_corpus(qname):
    """Train on a 500k sample, encode everything chunk-wise; reconstructions to an fp16 memmap, residual norms."""
    import faiss
    from .index import QUANTIZERS, _binary
    X = corpus()
    N, D = X.shape
    tag = "smoke." if LIMIT else ""
    path = os.path.join(CACHE_DIR, f"msmarco.bge.{tag}{qname}.xhat.f16")
    rpath = os.path.join(CACHE_DIR, f"msmarco.bge.{tag}{qname}.r.npy")
    if os.path.exists(rpath):
        return np.memmap(path, dtype=np.float16, mode="r", shape=(N, D)), np.load(rpath)
    rng = np.random.RandomState(0)
    sample = np.asarray(X[np.sort(rng.choice(N, min(N, 500_000), replace=False))], dtype=np.float32)
    _, obj = (None, None) if qname == "bin" else QUANTIZERS[qname](sample)
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(N, D))
    r = np.zeros(N, np.float32)
    for a, b in chunks(N):
        Xc = np.ascontiguousarray(X[a:b], dtype=np.float32)
        if qname == "bin":
            Xh = _binary(Xc)[0]
        elif isinstance(obj, faiss.Index):
            Xh = obj.sa_decode(obj.sa_encode(Xc))
        else:
            Xh = obj.decode(obj.compute_codes(Xc))
        Xh = Xh.astype(np.float16)
        mm[a:b] = Xh
        r[a:b] = np.linalg.norm(Xc - Xh.astype(np.float32), axis=1)
    mm.flush()
    np.save(rpath, r)
    return np.memmap(path, dtype=np.float16, mode="r", shape=(N, D)), r


def passB(qname):
    torch, DEV = _torch()
    A = np.load(os.path.join(OUT, "passA.npz"))
    top_s, top_i = A["top_s"], A["top_i"]
    X = corpus()
    path = os.path.join(CACHE_DIR, f"msmarco.bge.{'smoke.' if LIMIT else ''}{qname}")
    Xh = np.memmap(path + ".xhat.f16", dtype=np.float16, mode="r", shape=X.shape)
    r = np.load(path + ".r.npy")
    Q, cal, te = queries()
    nq, N = len(Q), X.shape[0]
    qn = np.linalg.norm(Q, axis=1)
    T = top_i[:, :K]
    s_k, s_k1 = top_s[:, K - 1], top_s[:, K]
    # winners' approximate scores and residuals: decision nonconformity and the conformal-depth threshold
    sh_T = np.einsum("qd,qkd->qk", Q, np.asarray(Xh[T.ravel()], dtype=np.float32).reshape(nq, K, -1))
    r_T = r[T]
    alpha_dec = np.maximum(((s_k1[:, None] - sh_T) / (qn[:, None] * r_T + 1e-8)).max(1), 0)
    worst = sh_T.min(1)
    # decision-calibrated constants are known before the pass (they depend on the winners only): exact cost counts
    c_dec = {d: conformal_quantile(alpha_dec[cal], d) for d in DELTAS}
    cvals = np.array([c_dec[d] for d in DELTAS], np.float32)
    cost_exact = np.zeros((nq, len(DELTAS)), np.int64)
    alpha_sim = np.zeros(nq, np.float32)
    above = np.zeros(nq, np.int64)
    t = time.time()
    for a, b in chunks(N):
        Xc = torch.from_numpy(np.asarray(X[a:b], dtype=np.float32)).to(DEV)
        Xhc = torch.from_numpy(np.asarray(Xh[a:b], dtype=np.float32)).to(DEV)
        rc = torch.from_numpy(r[a:b]).to(DEV)
        for qa in range(0, nq, Q_BATCH):
            sl = slice(qa, qa + Q_BATCH)
            Qb = torch.from_numpy(Q[sl]).to(DEV)
            qnb = torch.from_numpy(qn[sl]).to(DEV)[:, None]
            S, Sh = Qb @ Xc.T, Qb @ Xhc.T
            den = qnb * rc[None, :] + 1e-8
            alpha_sim[sl] = np.maximum(alpha_sim[sl], ((S - Sh).abs() / den).amax(1).cpu().numpy())
            # includes the worst winner itself and near-ties (MPS vs CPU rounding): conservative depth
            above[sl] += (Sh >= torch.from_numpy(worst[sl] - 2e-6).to(DEV)[:, None]).sum(1).cpu().numpy()
            skb = torch.from_numpy(s_k[sl]).to(DEV)[:, None]
            for j, c in enumerate(cvals):
                cost_exact[sl, j] += (Sh + float(c) * den >= skb).sum(1).cpu().numpy()
        print(f"[passB {qname}] {b:,}/{N:,} ({time.time() - t:.0f}s)", flush=True)
    # cost_exact counted the winners that pass the test; |A| = k + (#passing non-winners)
    uT = sh_T[:, :, None] + cvals[None, None, :] * (qn[:, None, None] * r_T[:, :, None] + 1e-8)
    cost_exact = K + cost_exact - (uT >= s_k[:, None, None]).sum(1)
    np.savez(os.path.join(OUT, f"passB.{qname}.npz"), alpha_dec=alpha_dec, alpha_sim=alpha_sim,
             R=above, cal=cal, te=te, cost_exact=cost_exact, c_dec=cvals)


def passC(qname):
    """Exact oracle cost at the simultaneously calibrated constants (known only after pass B)."""
    torch, DEV = _torch()
    A = np.load(os.path.join(OUT, "passA.npz"))
    z = np.load(os.path.join(OUT, f"passB.{qname}.npz"))
    X = corpus()
    path = os.path.join(CACHE_DIR, f"msmarco.bge.{'smoke.' if LIMIT else ''}{qname}")
    Xh = np.memmap(path + ".xhat.f16", dtype=np.float16, mode="r", shape=X.shape)
    r = np.load(path + ".r.npy")
    Q, cal, te = queries()
    nq, N = len(Q), X.shape[0]
    qn = np.linalg.norm(Q, axis=1)
    T, s_k = A["top_i"][:, :K], A["top_s"][:, K - 1]
    cvals = np.array([conformal_quantile(z["alpha_sim"][cal], d) for d in DELTAS], np.float32)
    cnt = np.zeros((nq, len(DELTAS)), np.int64)
    t = time.time()
    for a, b in chunks(N):
        Xhc = torch.from_numpy(np.asarray(Xh[a:b], dtype=np.float32)).to(DEV)
        rc = torch.from_numpy(r[a:b]).to(DEV)
        for qa in range(0, nq, Q_BATCH):
            sl = slice(qa, qa + Q_BATCH)
            Sh = torch.from_numpy(Q[sl]).to(DEV) @ Xhc.T
            den = torch.from_numpy(qn[sl]).to(DEV)[:, None] * rc[None, :] + 1e-8
            skb = torch.from_numpy(s_k[sl]).to(DEV)[:, None]
            for j, c in enumerate(cvals):
                cnt[sl, j] += (Sh + float(c) * den >= skb).sum(1).cpu().numpy()
        print(f"[passC {qname}] {b:,}/{N:,} ({time.time() - t:.0f}s)", flush=True)
    sh_T = np.einsum("qd,qkd->qk", Q, np.asarray(Xh[T.ravel()], dtype=np.float32).reshape(nq, K, -1))
    uT = sh_T[:, :, None] + cvals[None, None, :] * (qn[:, None, None] * r[T][:, :, None] + 1e-8)
    np.savez(os.path.join(OUT, f"passC.{qname}.npz"), cost_sim=K + cnt - (uT >= s_k[:, None, None]).sum(1), c_sim=cvals)


def passD(qname, M=10000):
    """Approximate top-M per query (by sh) with exact and approximate scores and residual norms, for truncated boxes."""
    torch, DEV = _torch()
    X = corpus()
    path = os.path.join(CACHE_DIR, f"msmarco.bge.{'smoke.' if LIMIT else ''}{qname}")
    Xh = np.memmap(path + ".xhat.f16", dtype=np.float16, mode="r", shape=X.shape)
    r = np.load(path + ".r.npy")
    Q, cal, te = queries()
    nq, N = len(Q), X.shape[0]
    best_sh = torch.full((nq, M), -1e9)
    best_s = torch.zeros((nq, M))
    best_i = torch.zeros((nq, M), dtype=torch.int64)
    t = time.time()
    for a, b in chunks(N):
        Xc = torch.from_numpy(np.asarray(X[a:b], dtype=np.float32)).to(DEV)
        Xhc = torch.from_numpy(np.asarray(Xh[a:b], dtype=np.float32)).to(DEV)
        for qa in range(0, nq, Q_BATCH):
            sl = slice(qa, qa + Q_BATCH)
            Qb = torch.from_numpy(Q[sl]).to(DEV)
            Sh = Qb @ Xhc.T
            v, i = torch.topk(Sh, min(M, Sh.shape[1]), dim=1)
            S = torch.gather(Qb @ Xc.T, 1, i)
            csh = torch.cat([best_sh[sl], v.cpu()], 1)
            cs = torch.cat([best_s[sl], S.cpu()], 1)
            ci = torch.cat([best_i[sl], i.cpu() + a], 1)
            o = torch.topk(csh, M, dim=1).indices
            best_sh[sl], best_s[sl], best_i[sl] = (torch.gather(csh, 1, o), torch.gather(cs, 1, o),
                                                   torch.gather(ci, 1, o))
        print(f"[passD {qname}] {b:,}/{N:,} ({time.time() - t:.0f}s)", flush=True)
    idx = best_i.numpy().astype(np.int32)
    np.savez(os.path.join(OUT, f"passD.{qname}.npz"), idx=idx, sh=best_sh.numpy(), s=best_s.numpy(),
             r=r[idx].astype(np.float32), qn=np.linalg.norm(Q, axis=1), cal=cal, te=te)


def report():
    res = {}
    for f in sorted(os.listdir(OUT)):
        if not f.startswith("passB."):
            continue
        qname = f.split(".")[1]
        z = np.load(os.path.join(OUT, f))
        cal, te = z["cal"], z["te"]
        out = {}
        for delta in DELTAS:
            cd = conformal_quantile(z["alpha_dec"][cal], delta)
            cs = conformal_quantile(z["alpha_sim"][cal], delta)
            Rh = conformal_quantile(z["R"][cal], delta)
            cost_dec = z["cost_exact"][te, DELTAS.index(delta)]
            pc = os.path.join(OUT, f"passC.{qname}.npz")
            cost_sim = np.load(pc)["cost_sim"][te, DELTAS.index(delta)] if os.path.exists(pc) else None
            out[delta] = dict(
                c_dec=cd, c_sim=cs, valid_dec=float((z["alpha_dec"][te] <= cd).mean()),
                valid_sim=float((z["alpha_sim"][te] <= cs).mean()),
                cost_dec_mean=float(cost_dec.mean()), cost_dec_p95=float(np.percentile(cost_dec, 95)),
                cost_sim_mean=None if cost_sim is None else float(cost_sim.mean()),
                confdepth=float(Rh), confdepth_recovered=float((z["R"][te] <= Rh).mean()),
                oracle_depth_mean=float(z["R"][te].mean()))
        res[qname] = out
        o = out[0.1]
        print(f"{qname:6s} dec: c={o['c_dec']:.3f} valid={o['valid_dec']:.3f} cost={o['cost_dec_mean']:.0f}/p95 "
              f"{o['cost_dec_p95']:.0f} | sim: c={o['c_sim']:.3f} cost={o['cost_sim_mean']} | confdepth={o['confdepth']:.0f} "
              f"rec={o['confdepth_recovered']:.3f} | oracle depth={o['oracle_depth_mean']:.0f}")
    json.dump(res, open(os.path.join(OUT, "report.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "quant":
        quantize_corpus(sys.argv[2])
    elif cmd == "passA":
        passA()
    elif cmd == "passB":
        passB(sys.argv[2])
    elif cmd == "passC":
        passC(sys.argv[2])
    elif cmd == "passD":
        passD(sys.argv[2])
    elif cmd == "report":
        report()
