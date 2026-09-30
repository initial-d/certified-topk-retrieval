"""Vector quantizers exposing reconstructions x_hat and residual norms r = ||x - x_hat||.

Each quantizer is trained on the corpus only (queries never touch training)."""
import faiss
import numpy as np


def _pq(X, m, nbits=8, seed=0):
    pq = faiss.ProductQuantizer(X.shape[1], m, nbits)
    pq.cp.seed = seed
    pq.train(X)
    return pq.decode(pq.compute_codes(X)), pq


def _opq(X, m, nbits=8, seed=0):
    opq = faiss.OPQMatrix(X.shape[1], m)
    idx = faiss.IndexPreTransform(opq, faiss.IndexPQ(X.shape[1], m, nbits))
    idx.train(X)
    idx.add(X)
    return idx.reconstruct_n(0, X.shape[0]), idx


def _sq(X, qtype):
    sq = faiss.ScalarQuantizer(X.shape[1], qtype)
    sq.train(X)
    return sq.decode(sq.compute_codes(X)), sq


def _binary(X):
    # 1 bit / dim with the L2-optimal per-vector scale  alpha = ||x||_1 / D
    alpha = np.abs(X).mean(1, keepdims=True)
    return (np.sign(X) * alpha).astype(np.float32), None


QUANTIZERS = {
    "pq16": lambda X: _pq(X, 16),
    "pq32": lambda X: _pq(X, 32),
    "pq96": lambda X: _pq(X, 96),
    "opq32": lambda X: _opq(X, 32),
    "sq4": lambda X: _sq(X, faiss.ScalarQuantizer.QT_4bit),
    "bin": _binary,
}
# bytes per vector, for reporting (D = 384)
CODE_BYTES = {"pq16": 16, "pq32": 32, "pq96": 96, "opq32": 32, "sq4": 192, "bin": 48 + 4}
# every code additionally stores r_d as one float16 (2 bytes) for the credal intervals


def quantize(qname, X):
    """Returns (X_hat, r, trained quantizer). Training uses the corpus only."""
    Xh, obj = QUANTIZERS[qname](np.ascontiguousarray(X))
    Xh = Xh.astype(np.float32)
    r = np.linalg.norm(X - Xh, axis=1).astype(np.float32)
    return Xh, r, obj


def quantize_queries(obj, qname, Q):
    """Quantize queries with a trained corpus quantizer (original-paper stability score)."""
    if qname == "bin":
        return _binary(Q)[0]
    if isinstance(obj, faiss.Index):
        # OPQ pipeline: encode/decode through the index
        codes = obj.sa_encode(np.ascontiguousarray(Q))
        return obj.sa_decode(codes)
    return obj.decode(obj.compute_codes(np.ascontiguousarray(Q)))
