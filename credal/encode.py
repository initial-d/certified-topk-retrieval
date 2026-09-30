"""Encode corpora and queries with several small embedding models, cached as .npy."""
import os
import sys

import numpy as np

from .data import CACHE_DIR, DATASETS, load

# name -> (hf id, query prefix, doc prefix)
MODELS = {
    "bge": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5": ("intfloat/e5-small-v2", "query: ", "passage: "),
    "gte": ("thenlper/gte-small", "", ""),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    # expensive stage of the aligned cascade (same family as the primary model, 768 dimensions)
    "bgebase": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
}
PRIMARY = "bge"
ENSEMBLE = ["bge", "e5", "gte", "minilm"]   # representation-layer members (bgebase is a cascade stage only)


def cache_path(ds, model, kind):
    return os.path.join(CACHE_DIR, f"{ds}.{model}.{kind}.npy")


def get(ds_name, model, kind):
    return np.load(cache_path(ds_name, model, kind))


def encode_all(datasets=DATASETS, models=("bge", "e5", "gte", "minilm")):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    for m in models:
        hf, qp, dp = MODELS[m]
        st = None
        for name in datasets:
            if all(os.path.exists(cache_path(name, m, k)) for k in ("docs", "queries")):
                continue
            if st is None:
                st = SentenceTransformer(hf, device=device)
                st.max_seq_length = 256
            ds = load(name)
            for kind, texts, pre in (("queries", ds.query_texts, qp), ("docs", ds.doc_texts, dp)):
                path = cache_path(name, m, kind)
                if os.path.exists(path):
                    continue
                # sort by length for efficient batching, then restore order
                order = np.argsort([len(t) for t in texts])
                emb = st.encode([pre + texts[i] for i in order], batch_size=128,
                                normalize_embeddings=True, show_progress_bar=False,
                                convert_to_numpy=True)
                out = np.empty_like(emb)
                out[order] = emb
                np.save(path, out.astype(np.float32))
                print(f"[encode] {name} {m} {kind} {out.shape}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--models":
        encode_all(args[2:] or DATASETS, args[1].split(","))
    else:
        encode_all(args or DATASETS)
