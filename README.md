# Decision-calibrated verification of top-k retrieval

Code for calibrating interval bounds on approximate retrieval scores for the **top-k decision** rather than for coverage of
every score, and for verifying the exact top-k with as few exact re-scorings as possible.

Given cheap scores ŝ (quantized vectors, or a first-stage model) and a nested family of intervals `[l(θ), u(θ)]` around them,
a *verify-before-certify* run re-scores options one at a time and stops once the k best re-scored options are ahead of every
remaining upper bound. The code

* computes the decision nonconformity `α(q) = max_{t ∈ T_k} (s_(k+1) − ŝ_t) / (‖q‖ r_t) ∨ 0` and its split-conformal
  quantile (`credal/credal.py`),
* runs the verification loop (LUCB-style rule, `credal.credal.refine`),
* compares residual-based boxes, rank-based boxes (conformally calibrated re-ranking depth), truncated boxes and joint
  updates on BEIR and MS MARCO, and
* regenerates all tables and figures.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt     # Python 3.9
mkdir -p data && cd data
for d in scifact nfcorpus arguana fiqa scidocs msmarco; do
  curl -LO https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$d.zip && unzip -q $d.zip && rm $d.zip
done
```

Embeddings are not distributed; `reproduce_all.sh` computes them with the public checkpoints
`BAAI/bge-small-en-v1.5`, `intfloat/e5-small-v2`, `thenlper/gte-small`, `sentence-transformers/all-MiniLM-L6-v2`,
`BAAI/bge-base-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` (MS MARCO encoding takes several hours).

## Reproduce

```bash
./reproduce_all.sh           # resumable: skips results that already exist
./reproduce_all.sh --fresh   # delete all results first
```

Outputs: `results/*.json` (numbers), `paper/tables/*.tex` (LaTeX tables and macros), `paper/figures/*.pdf`.
Every reported number is produced by `credal/make_tables.py`; none is typed by hand.

## Modules

| Module | Content |
|---|---|
| `credal/credal.py` | conformal quantile, decision nonconformity, tie-aware correctness and depth, verification loop, oracle cost |
| `credal/index.py` | PQ / OPQ / SQ4 / binary quantizers with reconstructions and residual norms |
| `credal/stats.py`, `exp_calib.py` | per-query statistics; δ curves, risk control, group-conditional calibration, winner's curse |
| `credal/exp_certify.py`, `exp_matched.py`, `exp_coverage.py` | certification cost and validity (quantized indexes) |
| `credal/exp_confdepth.py`, `exp_adaptive.py` | conformally calibrated re-ranking depth (rank boxes) under the same guarantee |
| `credal/exp_risk_actual.py` | risk-controlled verification, actual runs |
| `credal/cascade.py`, `exp_cascade.py` | bi-encoder → cross-encoder and small → base cascades |
| `credal/msmarco.py`, `exp_msmarco.py`, `run_msmarco.sh` | MS MARCO (8.8M passages), streamed exact statistics |
| `credal/exp_hybrid.py`, `exp_msmarco_hybrid.py` | truncated residual boxes, paired re-splits with per-split records |
| `credal/exp_joint.py` | order-dependent joint updates with envelope calibration |
| `credal/exp_ties_oracle.py`, `exp_tie_audit.py` | tie handling, termination, cost vs. oracle cost on correct queries |
| `credal/exp_uq.py`, `exp_shrink.py`, `exp_action.py` | exploratory multi-model uncertainty analyses |
| `credal/make_tables.py`, `figures_b1.py`, `make_figures.py` | tables, macros, figures |

## Protocol

| Quantity | Value |
|---|---|
| query split | 50/50 calibration/test, `numpy.random.RandomState(0).permutation` (`credal.data.load`) |
| level | δ = 0.1, k = 10 unless stated |
| conformal quantile | order statistic ⌈(n+1)(1−δ)⌉; +∞ if the index exceeds n or the statistic is infinite |
| infinite calibrated level | fall back to full verification (cost N, always correct) |
| validity re-splits | 500 random re-splits, seed 0 |
| truncated boxes | 200 re-splits, seed 1; tune = first quarter, calibration = second quarter, test = second half; cut-off levels δ_R ∈ {0.005, 0.01, 0.02, 0.025} |
| joint updates | ρ̂ and grid range from a random quarter (seed 0) never reused; 24-level grid; 200 re-splits (seed 1) |
| tail statistic | 95th percentile of per-query costs pooled over all test queries and splits |

## Ties

With τ = s_(k), a returned set R is a correct top-k iff `|R| = k` and `min_{R} s ≥ max_{not R} s` (tolerance 1e-6): all
options strictly above τ are returned, the remaining slots are tied options (`credal.credal.topk_correct`). The miss
fraction is `1 − (|R ∩ H| + min(|R ∩ E|, k − |H|)) / k` with H = {s > τ}, E = {s = τ} (`topk_hits`); the sufficient
re-ranking depth is the shortest cheap-score prefix containing a correct top-k (`required_depth`). The verification loop
breaks ties among lower bounds by index; if only resolved options remain as candidates or challengers (exact ties at the
threshold) it stops and returns the k largest lower bounds, so every run terminates after at most N re-scorings.
`exp_tie_audit.py` compares this criterion with the naive one ("every returned score ≥ τ") on all tied queries.

## Resumable scripts

Experiment scripts skip keys that already exist in their result file. **Delete the result file whenever the protocol
changes**, otherwise old and new entries are silently mixed.
