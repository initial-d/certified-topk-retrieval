"""BEIR dataset loading and deterministic calibration/test query splits."""
import csv
import json
import os
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "results")

DATASETS = ["scifact", "nfcorpus", "arguana", "fiqa", "scidocs"]


@dataclass
class Dataset:
    name: str
    doc_ids: list          # corpus order
    doc_texts: list
    query_ids: list        # queries that have test qrels, sorted
    query_texts: list
    qrels: dict            # qid -> {did: grade}
    calib_idx: np.ndarray  # indices into query_ids
    test_idx: np.ndarray


def _read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def load(name, seed=0):
    d = os.path.join(DATA_DIR, name)
    doc_ids, doc_texts = [], []
    for r in _read_jsonl(os.path.join(d, "corpus.jsonl")):
        doc_ids.append(str(r["_id"]))
        title = r.get("title") or ""
        doc_texts.append((title + " " + r["text"]).strip())
    qtext = {str(r["_id"]): r["text"] for r in _read_jsonl(os.path.join(d, "queries.jsonl"))}
    qrels = {}
    with open(os.path.join(d, "qrels", "test.tsv")) as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for qid, did, score in rd:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    query_ids = sorted(q for q in qrels if q in qtext)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(query_ids))
    half = len(perm) // 2
    return Dataset(name, doc_ids, doc_texts, query_ids, [qtext[q] for q in query_ids],
                   {q: qrels[q] for q in query_ids}, np.sort(perm[:half]), np.sort(perm[half:]))


def relevance_matrix(ds, top_ids):
    """top_ids: (nq, k) corpus indices -> (nq, k) graded relevance."""
    did_index = {i: d for i, d in enumerate(ds.doc_ids)}
    out = np.zeros(top_ids.shape, dtype=np.float32)
    for qi, qid in enumerate(ds.query_ids):
        rel = ds.qrels[qid]
        for j, di in enumerate(top_ids[qi]):
            out[qi, j] = rel.get(did_index[int(di)], 0)
    return out


def self_mask(ds):
    """Corpus index of the doc identical to each query (ArguAna convention), or -1."""
    pos = {d: i for i, d in enumerate(ds.doc_ids)}
    return np.array([pos.get(q, -1) for q in ds.query_ids])
