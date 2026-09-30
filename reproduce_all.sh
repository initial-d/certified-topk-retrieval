#!/bin/sh
# Reproduce every number, table and figure of the paper from the BEIR / MS MARCO data.
# Resumable scripts skip results that already exist; pass --fresh to delete all results first.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
if [ "$1" = "--fresh" ]; then
  find ../results -maxdepth 2 -name "*.json" ! -path "*/superseded/*" -delete
  rm -f ../results/repro/*.jsonl
fi

# 1. embeddings (BEIR: 5 models; MS MARCO: primary model)
$PY -m credal.encode
$PY -m credal.encode --models bgebase
$PY -m credal.msmarco encode

# 2. quantized indexes: certification, validity, same-guarantee comparison, calibration analyses
$PY -m credal.exp_certify
$PY -m credal.exp_matched
$PY -m credal.exp_coverage
$PY -m credal.exp_adaptive
$PY -m credal.stats
$PY -m credal.exp_calib
$PY -m credal.exp_risk_actual

# 3. cascades
$PY -m credal.cascade score
$PY -m credal.cascade bgebase
$PY -m credal.exp_cascade

# 4. MS MARCO (streamed): exact top-100, per-quantizer statistics, approximate top-10k for truncated boxes
./run_msmarco.sh
for q in pq32 bin pq96 sq4; do $PY -m credal.exp_msmarco passD $q; done
$PY -m credal.exp_msmarco_hybrid

# 5. truncated residual boxes (appendix): one split, paired re-splits, and an independent re-run with per-split records
$PY -m credal.exp_hybrid
$PY -m credal.exp_hybrid resplit
$PY -m credal.exp_hybrid resplit-verify
$PY - <<'EOF'
import json
a = json.load(open("../results/hybrid_resplit.json")); b = json.load(open("../results/repro/hybrid_resplit.json"))
bad = [(k, f) for k in a for f in a[k] if a[k][f] != b[k][f] and not (isinstance(a[k][f], float) and abs(a[k][f] - b[k][f]) < 1e-9)]
assert not bad, bad
print("re-split results reproduced exactly")
EOF

# 5b. joint updates (order-dependent procedure, envelope calibration), paired re-splits
$PY -m credal.exp_joint v3 scifact.pq32 nfcorpus.pq32 arguana.pq32 scifact.bin nfcorpus.bin arguana.bin scifact.pq96 \
    scidocs.pq32 scifact.pq16 nfcorpus.opq32 arguana.opq32 nfcorpus.pq96

# 5c. ties / termination check and oracle comparison on correctly returned queries (Table 7 configurations)
$PY -m credal.exp_ties_oracle

# 6. LaTeX tables/macros (written to ../paper/tables) and figures (../paper/figures)
$PY -m credal.make_tables
$PY -m credal.make_figures
$PY -m credal.figures_b1
echo DONE
