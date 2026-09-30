"""MS MARCO passage (8.8M) with the primary model: chunked, resumable encoding to an fp16 memmap.
Usage: python -m credal.msmarco encode"""
import csv
import json
import os
import sys

import numpy as np

from .data import CACHE_DIR, DATA_DIR
from .encode import MODELS, PRIMARY

MS = os.path.join(DATA_DIR, "msmarco")
CHUNK = 100_000
DIM = 384


def doc_ids():
    path = os.path.join(CACHE_DIR, "msmarco.docids.npy")
    if not os.path.exists(path):
        ids = [json.loads(l)["_id"] for l in open(os.path.join(MS, "corpus.jsonl"))]
        np.save(path, np.array(ids))
    return np.load(path)


def load_queries():
    """dev (MS MARCO dev small, 6980 queries, sparse labels) and test (TREC DL19, graded)."""
    qtext = {str(json.loads(l)["_id"]): json.loads(l)["text"] for l in open(os.path.join(MS, "queries.jsonl"))}
    out = {}
    for split in ("dev", "test"):
        qrels = {}
        with open(os.path.join(MS, "qrels", f"{split}.tsv")) as f:
            rd = csv.reader(f, delimiter="\t")
            next(rd)
            for qid, did, s in rd:
                if int(s) > 0:
                    qrels.setdefault(qid, {})[did] = int(s)
        qids = sorted(q for q in qrels if q in qtext)
        out[split] = (qids, [qtext[q] for q in qids], {q: qrels[q] for q in qids})
    return out


def corpus_path(model=PRIMARY):
    return os.path.join(CACHE_DIR, f"msmarco.{model}.docs.f16")


def corpus(model=PRIMARY):
    n = len(doc_ids())
    return np.memmap(corpus_path(model), dtype=np.float16, mode="r", shape=(n, DIM))


def encode(model=PRIMARY):
    from sentence_transformers import SentenceTransformer
    import torch
    hf, qp, dp = MODELS[model]
    st = SentenceTransformer(hf, device="mps" if torch.backends.mps.is_available() else "cpu")
    st.max_seq_length = 256
    qs = load_queries()
    for split, (qids, texts, _) in qs.items():
        p = os.path.join(CACHE_DIR, f"msmarco.{model}.queries.{split}.npy")
        if not os.path.exists(p):
            np.save(p, st.encode([qp + t for t in texts], batch_size=256, normalize_embeddings=True).astype(np.float32))
    n = len(doc_ids())
    mm = np.memmap(corpus_path(model), dtype=np.float16, mode="r+" if os.path.exists(corpus_path(model)) else "w+",
                   shape=(n, DIM))
    done_path = corpus_path(model) + ".done"
    done = set(int(x) for x in open(done_path).read().split()) if os.path.exists(done_path) else set()
    buf, start = [], 0
    with open(os.path.join(MS, "corpus.jsonl")) as f:
        for i, line in enumerate(f):
            c = i // CHUNK
            if c in done:
                continue
            r = json.loads(line)
            buf.append(((r.get("title") or "") + " " + r["text"]).strip())
            if len(buf) == 1:
                start = i
            if len(buf) == CHUNK or i == n - 1:
                order = np.argsort([len(t) for t in buf])
                emb = st.encode([dp + buf[j] for j in order], batch_size=256, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
                out = np.empty_like(emb)
                out[order] = emb
                mm[start:start + len(buf)] = out.astype(np.float16)
                mm.flush()
                with open(done_path, "a") as g:
                    g.write(f"{c}\n")
                print(f"[msmarco] chunk {c} ({start + len(buf):,}/{n:,})", flush=True)
                buf = []


if __name__ == "__main__":
    if sys.argv[1:] == ["encode"]:
        doc_ids()
        encode()
