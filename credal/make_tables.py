"""Generate LaTeX tables (paper/tables/*.tex) from results/*.json. Every number in the paper comes from here."""
import glob
import json
import os

import numpy as np

from .data import DATASETS, RESULTS_DIR, load

PAPER = os.path.join(os.path.dirname(RESULTS_DIR), "paper", "tables")
NICE = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna", "scidocs": "SciDocs", "fiqa": "FiQA"}
QNICE = {"pq16": "PQ16", "pq32": "PQ32", "pq96": "PQ96", "opq32": "OPQ32", "sq4": "SQ4", "bin": "Binary"}
DOMAIN = {"scifact": "scientific claims", "nfcorpus": "bio-medical", "arguana": "counter-arguments",
          "scidocs": "citation prediction", "fiqa": "financial QA"}


def w(name, s):
    os.makedirs(PAPER, exist_ok=True)
    open(os.path.join(PAPER, name), "w").write(s)
    print("wrote", name)


def fmt_int(x):
    return f"{x:,.0f}".replace(",", "{,}")


def table_data():
    rows = []
    for d in DATASETS:
        ds = load(d)
        rows.append(f"{NICE[d]} & {DOMAIN[d]} & {fmt_int(len(ds.doc_ids))} & {len(ds.query_ids)} "
                    f"({len(ds.calib_idx)}/{len(ds.test_idx)}) \\\\")
    w("data.tex", "\\begin{tabular}{llrr}\n\\toprule\nDataset & Domain & $N$ & Queries (cal/test)\\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def load_certify():
    out = {}
    for p in glob.glob(os.path.join(RESULTS_DIR, "certify", "*.json")):
        r = json.load(open(p))
        out[(r["dataset"], r["quant"])] = r
    return out


def table_coverage():
    cov = json.load(open(os.path.join(RESULTS_DIR, "coverage.json")))
    rows = []
    for d in DATASETS:
        for i, q in enumerate(["pq32", "pq96", "sq4"]):
            key = f"{d}.{q}"
            if key not in cov:
                continue
            v = cov[key]["10"]
            cells = " & ".join(f"{v[str(dl)][0]:.3f}\\,{{\\scriptsize$\\pm${v[str(dl)][1]:.3f}}}" for dl in (0.05, 0.1, 0.2))
            k_range = [cov[key][k]["0.1"][0] for k in cov[key]]
            name = f"\\multirow{{3}}{{*}}{{{NICE[d]}}}" if i == 0 else ""
            rows.append(f"{name} & {QNICE[q]} & {cells} & {min(k_range):.3f}--{max(k_range):.3f} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    w("coverage.tex", "\\begin{tabular}{llcccc}\n\\toprule\n"
      "& & \\multicolumn{3}{c}{$k=10$: coverage at $1-\\delta$} & $\\delta=0.1$, \\\\\n"
      "Dataset & Index & $0.95$ & $0.90$ & $0.80$ & $k\\in\\{1,\\dots,50\\}$ \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def certify_row(r, first, nrows, d):
    cr = r["credal"]
    dec = cr["decision"]
    m = r.get("matched", {})
    need = m.get("fixed_depth_needed")
    need = "$>$5000" if need is None else str(need)
    name = f"\\multirow{{{nrows}}}{{*}}{{{NICE[d]}}}" if first else ""
    return (f"{name} & {QNICE[r['quant']]} & {r['naive_exact_recall']:.3f} & {dec['c']:.3f} & "
            f"{fmt_int(cr['cauchy_schwarz']['lower_bound_mean'])} & {fmt_int(cr['simultaneous']['lower_bound_mean'])} & "
            f"{dec['lower_bound_mean']:.1f} & {dec['lucb']['n_ref_mean']:.1f} & "
            f"{dec['lucb']['exact_recall']:.3f} & {dec['lucb']['perfect']:.3f} & "
            f"{m.get('fixed', {}).get('perfect', float('nan')):.3f} & {need} \\\\")


CERT_HEAD = ("\\begin{tabular}{llcc rrr r ccc r}\n\\toprule\n"
             "& & Naive & & \\multicolumn{3}{c}{Oracle cost $|A(q)|$} & Alg.~1 & \\multicolumn{2}{c}{Certified} & "
             "Fixed@ & Depth \\\\\n\\cmidrule(lr){5-7}\\cmidrule(lr){9-10}\n"
             "Dataset & Index & R@10 & $\\hat c$ & C--S & Cov. & Dec. & cost & R@10 & Perfect & "
             "budget & needed \\\\\n\\midrule\n")


def table_certify(quants, fname):
    res = load_certify()
    rows = []
    for d in DATASETS:
        qs = [q for q in quants if (d, q) in res]
        for i, q in enumerate(qs):
            rows.append(certify_row(res[(d, q)], i == 0, len(qs), d))
        if qs:
            rows.append("\\midrule")
    w(fname, CERT_HEAD + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def table_strategies():
    res = load_certify()
    rows = []
    for q in ["pq32", "pq96", "sq4"]:
        vals = {s: [] for s in ("lucb", "score", "widest", "random")}
        lb, u_sym, u_asym = [], [], []
        for d in DATASETS:
            if (d, q) not in res:
                continue
            dec = res[(d, q)]["credal"]["decision"]
            for s in vals:
                vals[s].append(dec[s]["n_ref_mean"] / dec["lower_bound_mean"])
            u_asym.append(dec["U0_mean"])
            u_sym.append(res[(d, q)]["credal"]["decision_sym"]["U0_mean"])
        if not vals["lucb"]:
            continue
        rows.append(f"{QNICE[q]} & " + " & ".join(f"{np.mean(vals[s]):.3f}" for s in vals)
                    + f" & {np.mean(u_sym):.0f} & {np.mean(u_asym):.0f} \\\\")
    w("strategies.tex", "\\begin{tabular}{l cccc rr}\n\\toprule\n"
      "& \\multicolumn{4}{c}{Refinements / oracle cost $|A(q)|$} & \\multicolumn{2}{c}{Initial $|U_{10}|$}\\\\\n"
      "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\n"
      "Index & LUCB (Alg.~1) & Upper-bound & Widest & Random & $\\lambda=1$ & $\\lambda=0$ \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def load_uq():
    return {d: json.load(open(p)) for d in DATASETS
            for p in [os.path.join(RESULTS_DIR, "uq", f"{d}.json")] if os.path.exists(p)}


GROUPS = [
    ("Score-based (primary model)", ["MaxScore", "NQC", "WIG", "SMV", "Entropy"]),
    ("Embedding / geometric", ["Perturb", "OrigS", "OrigN", "OrigC"]),
    ("Ensemble (Bayesian)", ["EnsJac", "Bayes-TU", "Bayes-AU", "Bayes-EU"]),
    ("Credal (ours)", ["Credal-A-|U|", "Credal-A-Hup", "Credal-A-GH", "Credal-B-Hup", "Credal-B-Hlo", "Credal-B-GH",
                       "Credal-B-|U|", "Credal-AB-Hup", "Credal-AB-GH"]),
]
PNICE = {"MaxScore": "MaxScore", "NQC": "NQC", "WIG": "WIG", "SMV": "SMV", "Perturb": "Perturbation", "Entropy": "Softmax entropy",
         "OrigS": "$S_q$ (quant.\\ stability)", "OrigN": "$N_q$ (density)", "OrigC": "$C_q$ (harmonic)",
         "EnsJac": "Top-10 Jaccard", "Bayes-TU": "Total entropy", "Bayes-AU": "Expected entropy",
         "Bayes-EU": "Mutual information",
         "Credal-A-|U|": "A: $|U_{10}^A|$", "Credal-A-Hup": "A: $\\overline H$", "Credal-A-GH": "A: $\\overline H-\\underline H$",
         "Credal-B-Hup": "B: $\\overline H$", "Credal-B-Hlo": "B: $\\underline H$",
         "Credal-B-GH": "B: $\\overline H-\\underline H$", "Credal-B-|U|": "B: $|U_{10}^B|$",
         "Credal-AB-Hup": "AB: $\\overline H$", "Credal-AB-GH": "AB: $\\overline H-\\underline H$"}


def table_qpp(metric, fname, which="eval"):
    """Rows: predictors grouped by family; columns: datasets + mean. Best value per column in bold."""
    uq = load_uq()
    ds = [d for d in DATASETS if d in uq]
    keys = [k for _, ks in GROUPS for k in ks]
    V = np.array([[uq[d][which][k][metric] for d in ds] for k in keys])
    V = np.hstack([V, np.nanmean(V, 1, keepdims=True)])
    best = np.nanmax(V, 0)   # all metrics are higher-is-better (AURC is reported as e-AURC)
    base = (lambda v: f"{v:.3f}") if metric == "auroc_fail" else (lambda v: f"{v:+.3f}")
    fmt = lambda v: "sat." if np.isnan(v) else base(v)   # constant predictor (saturated upper entropy)
    rows, i = [], 0
    for gname, ks in GROUPS:
        rows.append(f"\\multicolumn{{{len(ds) + 2}}}{{l}}{{\\emph{{{gname}}}}} \\\\")
        for k in ks:
            cells = [("\\textbf{" + fmt(v) + "}") if (not np.isnan(v) and np.isclose(v, best[j])) else fmt(v)
                     for j, v in enumerate(V[i])]
            rows.append(f"\\quad {PNICE[k]} & " + " & ".join(cells) + " \\\\")
            i += 1
    head = ("\\begin{tabular}{l" + "c" * (len(ds) + 1) + "}\n\\toprule\nPredictor & "
            + " & ".join(NICE[d] for d in ds) + " & Mean \\\\\n\\midrule\n")
    w(fname, head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_adaptive():
    uq = load_uq()
    rows = []
    for d in DATASETS:
        if d not in uq:
            continue
        a = uq[d]["adaptive_k"]
        fk = a["fixed_k_recall"]
        kk = sorted(int(x) for x in fk)
        # linear interpolation of fixed-k recall at the mean credal set size
        if len(kk) == 2:
            t = a["ub_size_mean"] - kk[0]
            fixed = fk[str(kk[0])] * (1 - t) + fk[str(kk[1])] * t
        else:
            fixed = fk[str(kk[0])]
        rows.append(f"{NICE[d]} & {a['ub_size_mean']:.1f} & {a['ub_size_median']:.0f} & {a['ub_recall']:.3f} & "
                    f"{fixed:.3f} & {a['fixed10_recall']:.3f} & {a['consensus_size']:.1f} & "
                    f"{a['consensus_precision']:.3f} & {a['fused10_precision']:.3f} \\\\")
    w("adaptive.tex", "\\begin{tabular}{l rr ccc r cc}\n\\toprule\n"
      "& \\multicolumn{2}{c}{$|U^B_{10}|$} & \\multicolumn{3}{c}{Recall} & \\multicolumn{3}{c}{Consensus core}\\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
      "Dataset & mean & med. & $U^B_{10}$ & fixed $k{=}$mean & fixed $k{=}10$ & size & P & P@10 (fused) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_significance(ours="Credal-B-Hlo", rivals=("Entropy", "Bayes-AU", "MaxScore", "Perturb", "OrigC"), B=2000):
    """Paired bootstrap over test queries: 95% CI of Kendall-tau(ours) - Kendall-tau(rival)."""
    from scipy import stats
    rng = np.random.RandomState(0)
    rows = []
    for d in DATASETS:
        p = os.path.join(RESULTS_DIR, "uq", f"{d}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        te, y = z["te"], z["ndcg"]
        cells = []
        for rv in rivals:
            a, b = z[f"pred_{ours}"][te], z[f"pred_{rv}"][te]
            yt = y[te]
            diffs = []
            for _ in range(B):
                i = rng.randint(len(te), size=len(te))
                diffs.append(stats.kendalltau(a[i], yt[i])[0] - stats.kendalltau(b[i], yt[i])[0])
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            mark = "$^\\uparrow$" if lo > 0 else ("$^\\downarrow$" if hi < 0 else "")
            cells.append(f"[{lo:+.3f}, {hi:+.3f}]{mark}")
        rows.append(f"{NICE[d]} & " + " & ".join(cells) + " \\\\")
    w("significance.tex", "\\begin{tabular}{l" + "c" * len(rivals) + "}\n\\toprule\n"
      "Dataset & " + " & ".join(PNICE[r] for r in rivals) + " \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_confdepth():
    """Same guarantee, actual procedures: rank boxes (conformal depth, constant cost, actual recovery) versus
    Algorithm 1 on decision-calibrated residual boxes (actual resolutions and actual recovery, from exp_certify)
    versus the same certificate checked in cheap-score order (actual stopping depth)."""
    a = json.load(open(os.path.join(RESULTS_DIR, "adaptive.json")))
    cert = load_certify()
    rows, rat_mean, rat_p95, rat_rerank, rat_oracle = [], [], [], [], []
    for d in DATASETS:
        qs = [q for q in ["pq32", "pq96", "sq4", "bin"] if f"{d}.{q}" in a]
        for i, q in enumerate(qs):
            t = a[f"{d}.{q}"]["test"]
            al = cert[(d, q)]["credal"]["decision"]["lucb"]
            p95 = float(np.percentile(al["n_ref_all"], 95))
            name = f"\\multirow{{{len(qs)}}}{{*}}{{{NICE[d]}}}" if i == 0 else ""
            rows.append(f"{name} & {QNICE[q]} & {t['confdepth']['mean']:.0f} & {t['confdepth']['recovered']:.3f} & "
                        f"{al['n_ref_mean']:.1f} & {p95:.0f} & {al['perfect']:.3f} & "
                        f"{t['rerank_cert']['mean']:.1f} & {t['rerank_cert']['p95']:.0f} & "
                        f"{t['oracle_depth']['mean']:.1f} \\\\")
        rows.append("\\midrule")
    for k_, v in a.items():
        d, q = k_.split(".")
        al = cert[(d, q)]["credal"]["decision"]["lucb"]
        R = v["test"]["confdepth"]["mean"]
        rat_mean.append(al["n_ref_mean"] / R)
        rat_p95.append(np.percentile(al["n_ref_all"], 95) / R)
        rat_rerank.append(v["test"]["rerank_cert"]["mean"] / R)
        rat_oracle.append(v["test"]["oracle_depth"]["mean"] / R)
    w("confdepth.tex", "\\begin{tabular}{ll rc rrc rr r}\n\\toprule\n"
      "& & \\multicolumn{2}{c}{Rank boxes} & \\multicolumn{3}{c}{Residual boxes, Alg.~1} & "
      "\\multicolumn{2}{c}{$\\hat s$ order} & Oracle \\\\\n"
      "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n"
      "Dataset & Index & $\\hat R$ & exact & mean & P95 & exact & mean & P95 & depth \\\\\n\\midrule\n"
      + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    f = lambda x: (float(np.median(x)), float(np.min(x)), float(np.max(x)), int(np.sum(np.array(x) < 1)), len(x))
    rk_or = [1 / x for x in rat_oracle]                          # rank cost / mean oracle depth
    rs_or = [m_ / o_ for m_, o_ in zip(rat_mean, rat_oracle)]     # residual (Alg. 1) cost / mean oracle depth
    return {"alg1_mean": f(rat_mean), "alg1_p95": f(rat_p95), "rerank_mean": f(rat_rerank), "oracle_mean": f(rat_oracle),
            "rank_over_oracle": f(rk_or), "res_over_oracle": f(rs_or)}


def table_msmarco():
    """MS MARCO (8.8M passages, primary model, delta = 0.1): streamed exact statistics (exp_msmarco)."""
    path = os.path.join(RESULTS_DIR, "msmarco", "report.json")
    if not os.path.exists(path):
        w("msmarco.tex", "\\todo{MS MARCO results pending (code/run\\_msmarco.sh)}\n")
        return None
    rep = json.load(open(path))
    rows = []
    for q in ["pq32", "pq96", "sq4", "bin"]:
        if q not in rep:
            continue
        o = rep[q]["0.1"]
        sim = "--" if o["cost_sim_mean"] is None else fmt_int(o["cost_sim_mean"])
        rows.append(f"{QNICE[q]} & {o['valid_dec']:.3f} & {sim} & {fmt_int(o['cost_dec_mean'])} & "
                    f"{fmt_int(o['cost_dec_p95'])} & {fmt_int(o['confdepth'])} & {o['confdepth_recovered']:.3f} & "
                    f"{fmt_int(o['oracle_depth_mean'])} \\\\")
    w("msmarco.tex", "\\begin{tabular}{l c r rr rc r}\n\\toprule\n"
      "& & Cov. & \\multicolumn{2}{c}{Residual, decision} & \\multicolumn{2}{c}{Rank boxes} & Oracle \\\\\n"
      "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
      "Index & valid & mean & mean & P95 & $\\hat R$ & valid & depth \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return rep


def table_determinacy():
    """Same residual family (symmetric boxes, lambda = 1): decision vs coverage calibration -> level, credal top-10
    set size (determinacy) and oracle resolution cost."""
    res = load_certify()
    rows = []
    for d in DATASETS:
        qs = [q for q in ["pq32", "pq96", "sq4", "bin"] if (d, q) in res]
        for i, q in enumerate(qs):
            cr = res[(d, q)]["credal"]
            name = f"\\multirow{{{len(qs)}}}{{*}}{{{NICE[d]}}}" if i == 0 else ""
            rows.append(f"{name} & {QNICE[q]} & {cr['decision_sym']['c']:.3f} & {cr['simultaneous']['c']:.3f} & "
                        f"{cr['decision_sym']['U0_mean']:.0f} & {cr['simultaneous']['U0_mean']:.0f} & "
                        f"{cr['decision']['lower_bound_mean']:.0f} & {cr['simultaneous']['lower_bound_mean']:.0f} \\\\")
        rows.append("\\midrule")
    w("determinacy.tex", "\\begin{tabular}{ll cc rr rr}\n\\toprule\n"
      "& & \\multicolumn{2}{c}{Level $\\hat\\theta$} & \\multicolumn{2}{c}{$|U_{10}|$ (credal top-10)} & "
      "\\multicolumn{2}{c}{Oracle cost $|A(q)|$}\\\\\n"
      "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
      "Dataset & Index & dec. & cov. & dec. & cov. & dec. & cov. \\\\\n\\midrule\n"
      + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def table_cascade():
    path = os.path.join(RESULTS_DIR, "cascade.json")
    if not os.path.exists(path):
        w("cascade.tex", "\\todo{cascade results pending}\n")
        return None
    c = json.load(open(path))
    rows = []
    for stage, title in (("ce", "bi-encoder $\\to$ cross-encoder (pool 100)"),
                         ("bgebase", "BGE-small $\\to$ BGE-base (pool 1000)")):
        rows.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for d in DATASETS:
            r = c.get(f"{d}.{stage}.global")
            if not r:
                continue
            de, si, cd = r["decision"], r["simultaneous"], r["confdepth"]
            rows.append(f"\\quad {NICE[d]} & {de['alg1_mean']:.1f} & {de['alg1_p95']:.0f} & "
                        f"{r['resplit']['decision'][0]:.3f} & {si['oracle_mean']:.1f} & {cd['cost']} & "
                        f"{r['resplit']['confdepth'][0]:.3f} & {r['oracle_depth_mean']:.1f} & "
                        f"{si['oracle_mean'] / de['oracle_mean']:.1f}$\\times$ \\\\")
    w("cascade.tex", "\\begin{tabular}{l rrc r rc r r}\n\\toprule\n"
      "& \\multicolumn{3}{c}{Anchored, decision-cal.} & Coverage & \\multicolumn{2}{c}{Rank boxes} & Oracle & "
      "Cov./dec. \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}\n"
      "Dataset & mean & P95 & valid$_{500}$ & mean & $\\hat R$ & valid$_{500}$ & depth & ratio \\\\\n"
      "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    ratios = {st_: [c[k_]["simultaneous"]["oracle_mean"] / c[k_]["decision"]["oracle_mean"]
                    for k_ in c if k_.split(".")[1] == st_] for st_ in ("ce", "bgebase")}
    return {k_: (float(min(v)), float(max(v))) for k_, v in ratios.items() if v}


def table_risk():
    """Actual runs of Algorithm 1 at risk-controlled levels (exp_risk_actual)."""
    ra = json.load(open(os.path.join(RESULTS_DIR, "risk_actual.json")))
    rows = []
    for d in DATASETS:
        for q in ("pq32", "pq96"):
            r = ra.get(f"{d}.{q}")
            if not r:
                continue
            ex = r["exact_0.1"]
            cells = []
            for e in (0.01, 0.02, 0.05):
                x = r.get(f"eps_{e}")
                cells.append("-- & -- & --" if x is None else
                             f"{x['cost_mean']:.0f}\\,({x['cost_p95']:.0f}) & {x['miss']:.3f} & {x['exact']:.2f}")
            rows.append(f"{NICE[d]} & {QNICE[q]} & {ex['cost_mean']:.0f}\\,({ex['cost_p95']:.0f}) & " + " & ".join(cells) + " \\\\")
    w("risk.tex", "\\begin{tabular}{ll r rcc rcc rcc}\n\\toprule\n"
      "& & Exact & \\multicolumn{3}{c}{$\\varepsilon=0.01$} & \\multicolumn{3}{c}{$\\varepsilon=0.02$} & "
      "\\multicolumn{3}{c}{$\\varepsilon=0.05$} \\\\\n"
      "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}\n"
      "Dataset & Index & $\\delta{=}0.1$ & cost & miss & exact & cost & miss & exact & cost & miss & exact \\\\\n"
      "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    ratio = lambda e: [r[f"eps_{e}"]["cost_mean"] / r["exact_0.1"]["cost_mean"] for r in ra.values() if f"eps_{e}" in r]
    below = all(x["miss"] <= x["worst_miss_bound"] + 1e-12 for r in ra.values() for x in r.values())
    return float(np.median(ratio(0.02))), float(np.median(ratio(0.05))), below


def summary_b1():
    """Aggregates for the new analyses, exposed as LaTeX macros."""
    cal = json.load(open(os.path.join(RESULTS_DIR, "calib.json")))
    wc_pq = [r["winners_curse"]["mean_winners"] for k_, r in cal.items() if "pq" in k_ or "opq" in k_]
    wc_rand = [r["winners_curse"]["mean_random"] for k_, r in cal.items() if "pq" in k_ or "opq" in k_]
    wc_sq = [r["winners_curse"]["mean_winners"] for k_, r in cal.items() if k_.endswith("sq4")]
    risk_dev = [x["miss_resplit"] / x["eps"] for r in cal.values() for x in r["risk_control"] if x.get("miss_resplit")]
    mon_m = [x["valid_marginal"] for r in cal.values() for x in r["mondrian"]]
    mon_b = [x["valid_mondrian"] for r in cal.values() for x in r["mondrian"]]
    r02, r05, below = table_risk()
    assert below, "actual miss exceeded the worst-case bound of Thm. 3"
    d01 = [next(x for x in r["delta_curve"] if x["delta"] == 0.01) for r in cal.values()]
    ratio01 = [x["cost_cov"] / x["cost_dec"] for x in d01 if x["cost_cov"]]
    cr = table_cascade()
    m = {
        "SUMWCPQ": f"{min(wc_pq):+.2f} to {max(wc_pq):+.2f}",
        "SUMWCRAND": f"{max(abs(x) for x in wc_rand):.3f}",
        "SUMWCSQ": f"{max(abs(x) for x in wc_sq):.3f}",
        "SUMRISKMAX": f"{max(risk_dev):.2f}",
        "SUMRISKTWO": f"{r02:.2f}",
        "SUMRISKFIVE": f"{r05:.2f}",
        "SUMMONM": f"{min(mon_m):.3f}--{max(mon_m):.3f}",
        "SUMMONB": f"{min(mon_b):.3f}--{max(mon_b):.3f}",
        "SUMRATIOSMALLDELTA": f"{np.median(ratio01):.1f}",
        "SUMCASCE": f"{cr['ce'][0]:.2f}--{cr['ce'][1]:.2f}" if cr else "--",
        "SUMCASBGE": f"{cr['bgebase'][0]:.1f}--{cr['bgebase'][1]:.1f}" if cr else "--",
    }
    return m


def table_hybrid():
    """Paired comparison of rank, residual and truncated residual boxes on identical re-splits (BEIR), and one split on
    MS MARCO (actual runs for truncated boxes). Main table: 4 quantizers; appendix: all 30 BEIR configurations."""
    hr = json.load(open(os.path.join(RESULTS_DIR, "hybrid_resplit.json")))
    h = json.load(open(os.path.join(RESULTS_DIR, "hybrid.json")))
    head = ("\\begin{tabular}{ll rr rr rr cc}\n\\toprule\n"
            "& & \\multicolumn{2}{c}{Rank} & \\multicolumn{2}{c}{Residual} & \\multicolumn{2}{c}{Truncated residual} & & \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
            "Dataset & Index & mean & P95$_{\\rm pool}$ & mean & P95$_{\\rm pool}$ & mean & P95$_{\\rm pool}$ & valid & fallbacks \\\\\n"
            "\\midrule\n")

    def rows_for(quants):
        rows = []
        for d in DATASETS:
            qs = [q for q in quants if f"{d}.{q}" in hr]
            for i, q in enumerate(qs):
                v = hr[f"{d}.{q}"]
                name = f"\\multirow{{{len(qs)}}}{{*}}{{{NICE[d]}}}" if i == 0 else ""
                rows.append(f"{name} & {QNICE[q]} & {v['cost_rank']:.0f} & {v['pooled_p95_rank']:.0f} & "
                            f"{v['cost_residual']:.1f} & {v['pooled_p95_residual']:.0f} & "
                            f"{v['cost_hybrid']:.1f} & {v['pooled_p95_hybrid']:.0f} & {v['valid_hybrid'][0]:.3f} & "
                            f"{v['n_fallback']}/{v['reps']} \\\\")
            rows.append("\\midrule")
        return rows

    rows = rows_for(["pq32", "pq96", "sq4", "bin"])
    msp = os.path.join(RESULTS_DIR, "msmarco", "hybrid.json")
    m = json.load(open(msp)) if os.path.exists(msp) else {}
    for i, q in enumerate([q for q in ["pq32", "pq96", "sq4", "bin"] if q in m]):
        x = m[q]
        name = "\\multirow{4}{*}{MS~MARCO$^\\ast$}" if i == 0 else ""
        rows.append(f"{name} & {QNICE[q]} & {x['rank']['Rhat']:.0f} & {x['rank']['Rhat']:.0f} & "
                    f"{fmt_int(x['residual_oracle']['mean'])} & {fmt_int(x['residual_oracle']['p95'])} & "
                    f"{x['hybrid']['mean']:.1f} & {x['hybrid']['p95']:.0f} & {x['hybrid']['valid']:.3f} & 0/1 \\\\")
    w("hybrid.tex", head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    w("hybrid_full.tex", head + "\n".join(rows_for(["pq16", "pq32", "pq96", "opq32", "sq4", "bin"])[:-1])
      + "\n\\bottomrule\n\\end{tabular}\n")
    tr = [v["cost_ratio"] for v in hr.values()]
    rr = [v["cost_ratio_residual"] for v in hr.values()]
    fb = [v["n_fallback"] for v in hr.values()]
    val = [v["valid_hybrid"][0] for v in hr.values()]
    gap = [v["hybrid"]["mean"] / v["hybrid"]["oracle_mean"] - 1 for v in h.values() if not v.get("fallback")]
    out = {"TRWINS": f"{sum(x < 1 for x in tr)} of {len(tr)}", "TRMED": f"{np.median(tr):.2f}",
           "RESWINS": f"{sum(x < 1 for x in rr)} of {len(rr)}", "RESMED": f"{np.median(rr):.2f}",
           "TRFB": f"{sum(fb)}", "TRFBTOT": f"{sum(v['reps'] for v in hr.values())}", "TRFBMAX": f"{max(fb)}",
           "TRFBN": f"{sum(x > 0 for x in fb)}",
           "TRTAILRES": f"{sum(v['pooled_p95_hybrid'] < v['pooled_p95_residual'] for v in hr.values())} of {len(hr)}",
           "TRMAX": f"{max(tr):.1f}",
           "TRVALID": f"{min(val):.3f}--{max(val):.3f}", "TRGAP": f"{100 * max(gap):.1f}\\%",
           "TRBEATRES": f"{sum(v['cost_hybrid'] < v['cost_residual'] for v in hr.values())}",
           "TRPNINEFIVE": f"{sum(v['pooled_p95_hybrid'] > v['pooled_p95_rank'] for v in hr.values())} of {len(hr)}"}
    if m:
        r_ = {q: m[q]["hybrid"]["mean"] / m[q]["rank"]["Rhat"] for q in m}
        out.update({"MSHYBPQ": f"{m['pq32']['hybrid']['mean']:,.0f}".replace(",", "{,}"),
                    "MSHYBPQRATIO": f"{r_['pq32']:.2f}", "MSHYBBIN": f"{m['bin']['hybrid']['mean']:.0f}",
                    "MSHYBBINRATIO": f"{r_['bin']:.2f}",
                    "MSHYBFINE": f"{min(r_['pq96'], r_['sq4']):.2f}--{max(r_['pq96'], r_['sq4']):.2f}",
                    "MSHYBVALID": f"{min(m[q]['hybrid']['valid'] for q in m):.3f}--{max(m[q]['hybrid']['valid'] for q in m):.3f}"})
    return out


def table_joint():
    """Joint updates (appendix): paired re-splits against the original decision-calibrated LUCB (actual costs)."""
    path = os.path.join(RESULTS_DIR, "joint_v3.json")
    if not os.path.exists(path):
        return {}
    r = json.load(open(path))
    order = ["pq16", "pq32", "opq32", "pq96", "bin"]
    keys = sorted(r, key=lambda k: (order.index(k.split(".")[1]), DATASETS.index(k.split(".")[0])))
    rows = []
    for k in keys:
        d, q = k.split(".")
        o = r[k]["original"]
        ratio = lambda lab, f="mean": r[k][lab][f] / o[f]
        rows.append(f"{NICE[d]} & {QNICE[q]} & {r[k]['rho']:.2f} & {o['mean']:.1f} & {o['p95_pool']:.0f} & {o['valid']:.3f} & "
                    f"{ratio('grid_baseline'):.2f} & {ratio('joint'):.2f} & {ratio('joint', 'p95_pool'):.2f} & "
                    f"{r[k]['joint']['valid']:.3f} & {ratio('center_only'):.2f} & {ratio('width_only'):.2f} \\\\")
    w("joint.tex", "\\begin{tabular}{ll c rrc c ccc cc}\n\\toprule\n"
      "& & & \\multicolumn{3}{c}{Original (Alg.~1)} & Grid & \\multicolumn{3}{c}{Joint updates} & "
      "\\multicolumn{2}{c}{Ablation} \\\\\n"
      "\\cmidrule(lr){4-6}\\cmidrule(lr){8-10}\\cmidrule(lr){11-12}\n"
      "Dataset & Index & $\\hat\\rho$ & mean & P95$_{\\rm pool}$ & valid & mean & mean & P95$_{\\rm pool}$ & valid & "
      "centre & width \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    nb = [k for k in r if not k.endswith(".bin")]
    bi = [k for k in r if k.endswith(".bin")]
    rat = lambda ks, lab, f="mean": [r[k][lab][f] / r[k]["original"][f] for k in ks]
    m = rat(nb, "joint")
    return {"JTN": f"{sum(x < 1 for x in m)} of {len(m)}", "JTMED": f"{np.median(m):.2f}",
            "JTRANGE": f"{min(m):.2f}--{max(m):.2f}", "JTPNF": f"{np.median(rat(nb, 'joint', 'p95_pool')):.2f}",
            "JTSAVE": f"{100 * (1 - np.median(m)):.0f}",
            "JTGRID": f"{min(rat(nb, 'grid_baseline')):.2f}--{max(rat(nb, 'grid_baseline')):.2f}",
            "JTWIDTH": f"{min(rat(nb, 'width_only')):.2f}--{max(rat(nb, 'width_only')):.2f}",
            "JTVALID": f"{min(r[k]['joint']['valid'] for k in r):.3f}--{max(r[k]['joint']['valid'] for k in r):.3f}",
            "JTBIN": f"{min(rat(bi, 'joint')):.1f}--{max(rat(bi, 'joint')):.1f}",
            "JTFB": f"{sum(r[k][lab]['fallback'] for k in r for lab in ('grid_baseline', 'joint', 'center_only', 'width_only'))}",
            "JTNRHO": f"{min(r[k]['n_rho'] for k in r)}--{max(r[k]['n_rho'] for k in r)}",
            "JTMSNBR": f"{np.median([r[k]['ms_neighbours'] for k in r]):.0f}",
            "JTMSRATIO": f"{np.median([r[k]['ms_per_run']['joint'] / r[k]['ms_per_run']['grid_baseline'] for k in nb]):.2f}"}


def table_ties_oracle():
    """Alg. 1 cost relative to |A(q)| on correctly returned queries only (Table 7 configurations), and tie statistics."""
    path = os.path.join(RESULTS_DIR, "ties_oracle.json")
    if not os.path.exists(path):
        return {}
    t = json.load(open(path))
    rows = []
    for q in ("pq32", "pq96", "sq4"):
        ks = [f"{d}.{q}" for d in DATASETS if f"{d}.{q}" in t]
        n = sum(t[k]["n_test"] for k in ks)
        nc = sum(t[k]["n_correct"] for k in ks)
        mean_c = np.average([t[k]["correct_ratio_mean"] for k in ks], weights=[t[k]["n_correct"] for k in ks])
        rows.append(f"{QNICE[q]} & {n} & {nc} & {mean_c:.3f} & {max(t[k]['correct_ratio_max'] for k in ks):.2f} & "
                    f"{sum(t[k]['correct_below_oracle'] for k in ks)} & "
                    f"{np.mean([t[k]['all_mean_cost'] / t[k]['all_mean_oracle'] for k in ks]):.3f} \\\\")
    w("ties_oracle.tex", "\\begin{tabular}{l rr ccc c}\n\\toprule\n"
      "& \\multicolumn{5}{c}{Correctly returned test queries} & All queries \\\\\n\\cmidrule(lr){2-6}\n"
      "Index & queries & correct & mean ratio & max ratio & below $|A(q)|$ & ratio of means \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    return {"TIEQ": f"{sum(t[k]['n_ties'] for k in t)}", "TIEN": f"{sum(t[k]['n_test'] for k in t)}",
            "TIEUNCERT": f"{sum(t[k]['n_uncertified'] for k in t)}",
            "TIEUNCERTNOTIE": f"{sum(t[k]['n_uncertified_without_tie'] for k in t)}",
            "TIEBELOW": f"{sum(t[k]['correct_below_oracle'] for k in t)}"}


def summary_numbers():
    """Aggregates quoted in the abstract / text; also written to results/summary.json."""
    res = load_certify()
    out = {}
    ratios = {q: [res[(d, q)]["credal"]["simultaneous"]["lower_bound_mean"] / res[(d, q)]["credal"]["decision"]["lower_bound_mean"]
                  for d in DATASETS if (d, q) in res] for q in QNICE}
    out["simult_over_decision"] = {q: (float(np.min(v)), float(np.max(v))) for q, v in ratios.items() if v}
    allr = [x for v in ratios.values() for x in v]
    out["simult_over_decision_all"] = (float(np.min(allr)), float(np.median(allr)), float(np.max(allr)))
    gap = [res[k]["credal"]["decision"]["lucb"]["n_ref_mean"] / res[k]["credal"]["decision"]["lower_bound_mean"] - 1
           for k in res]
    out["alg1_over_oracle_max_pct"] = 100 * float(np.max(gap))
    wins = [res[k]["matched"]["credal"]["perfect"] > res[k]["matched"]["fixed"]["perfect"] for k in res if "matched" in res[k]]
    out["matched_wins"] = (int(np.sum(wins)), len(wins))
    need = [res[k]["matched"]["fixed_depth_needed"] / res[k]["matched"]["budget"] for k in res
            if "matched" in res[k] and res[k]["matched"]["fixed_depth_needed"]]
    out["depth_needed_over_budget"] = (float(np.min(need)), float(np.median(need)), float(np.max(need)))
    cov = json.load(open(os.path.join(RESULTS_DIR, "coverage.json")))
    c10 = [v[k]["0.1"][0] for v in cov.values() for k in v]
    out["coverage_delta0.1"] = (float(np.min(c10)), float(np.max(c10)))
    uq = load_uq()
    if uq:
        mk = {k: float(np.mean([uq[d]["eval"][k]["kendall"] for d in uq])) for k in uq[next(iter(uq))]["eval"]}
        out["kendall_mean"] = dict(sorted(mk.items(), key=lambda kv: -kv[1]))
    c10x = [v[k]["0.1"][0] for key, v in cov.items() for k in v if not (key.endswith(".sq4") and k == "1")]
    out["coverage_delta0.1_excl_sq4k1"] = (float(np.min(c10x)), float(np.max(c10x)))
    slopes = [json.load(open(p))["slope_width_rel"] for p in glob.glob(os.path.join(RESULTS_DIR, "shrink", "*.json"))]
    out["shrink_slopes"] = (float(np.min(slopes)), float(np.max(slopes))) if slopes else None
    if os.path.exists(os.path.join(RESULTS_DIR, "adaptive.json")):
        out["vs_confdepth"] = table_confdepth()
    json.dump(out, open(os.path.join(RESULTS_DIR, "summary.json"), "w"), indent=1)
    lo_, med_, hi_ = out["simult_over_decision_all"]
    macros = {
        "SUMCOV": f"({out['coverage_delta0.1_excl_sq4k1'][0]:.3f}--{out['coverage_delta0.1_excl_sq4k1'][1]:.3f} at "
                  f"$1-\\delta=0.90$ for every dataset, quantizer and $k$ except SQ4 with $k=1$)",
        "SUMRATIO": f"{lo_:.1f}--{hi_:.1f}$\\times$ (median {med_:.1f}$\\times$)",
        "SUMALGGAP": f"{out['alg1_over_oracle_max_pct']:.1f}\\%",
        "SUMWINS": f"{out['matched_wins'][0]} of {out['matched_wins'][1]}",
        "SUMDEPTH": f"{out['depth_needed_over_budget'][1]:.2f}$\\times$",
        **({} if "vs_confdepth" not in out else {
            "SUMCDALG": "{:.2f}".format(out["vs_confdepth"]["alg1_mean"][0]),
            "SUMCDALGRANGE": "{:.2f}--{:.2f}".format(out["vs_confdepth"]["alg1_mean"][1], out["vs_confdepth"]["alg1_mean"][2]),
            "SUMCDALGWINS": "{} of {}".format(out["vs_confdepth"]["alg1_mean"][3], out["vs_confdepth"]["alg1_mean"][4]),
            "SUMCDPNF": "{:.2f}".format(out["vs_confdepth"]["alg1_p95"][0]),
            "SUMCDRERANK": "{:.2f}".format(out["vs_confdepth"]["rerank_mean"][0]),
            "SUMCDORACLE": "{:.2f}".format(out["vs_confdepth"]["oracle_mean"][0]),
            "SUMRKOR": "{:.2f} (range {:.2f}--{:.2f})".format(*out["vs_confdepth"]["rank_over_oracle"][:3]),
            "SUMRSOR": "{:.2f} (range {:.2f}--{:.2f})".format(*out["vs_confdepth"]["res_over_oracle"][:3])}),
        "SUMSLOPE": (f"${out['shrink_slopes'][0]:.2f}$ and ${out['shrink_slopes'][1]:.2f}$" if slopes else "--"),
    }
    if os.path.exists(os.path.join(RESULTS_DIR, "calib.json")):
        macros.update(summary_b1())
    if os.path.exists(os.path.join(RESULTS_DIR, "hybrid_resplit.json")):
        macros.update(table_hybrid())
    macros.update(table_joint())
    macros.update(table_ties_oracle())
    ms = table_msmarco()
    if ms:
        o = {q: ms[q]["0.1"] for q in ("pq32", "pq96", "sq4", "bin")}
        cr = {q: o[q]["cost_sim_mean"] / o[q]["cost_dec_mean"] for q in o}
        rr = {q: o[q]["cost_dec_mean"] / o[q]["confdepth"] for q in o}
        macros.update({
            "MSVALID": f"{min(x['valid_dec'] for x in o.values()):.3f}--{max(x['valid_dec'] for x in o.values()):.3f}",
            "MSDEPTHVALID": f"{min(x['confdepth_recovered'] for x in o.values()):.3f}--{max(x['confdepth_recovered'] for x in o.values()):.3f}",
            "MSCOVRATIO": f"{min(cr.values()):.1f}--{max(cr.values()):.1f}",
            "MSRESPQ": f"{o['pq32']['cost_dec_mean']:,.0f}".replace(",", "{,}"),
            "MSRANKPQ": f"{o['pq32']['confdepth']:,.0f}".replace(",", "{,}"),
            "MSRATIOPQ": f"{rr['pq32']:.0f}",
            "MSRATIOBIN": f"{rr['bin']:.0f}",
            "MSRATIOFINE": f"{min(rr['pq96'], rr['sq4']):.2f}--{max(rr['pq96'], rr['sq4']):.2f}",
            "MSCPQ": f"{o['pq32']['c_dec']:.2f}", "MSCSQ": f"{o['sq4']['c_dec']:.3f}",
            "MSCOVPQ": f"{o['pq32']['c_sim']:.2f}", "MSCOVSQ": f"{o['sq4']['c_sim']:.2f}",
        })
    w("summary_macros.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    lo, med, hi = out["simult_over_decision_all"]
    v = out.get("vs_confdepth")
    w("abstract_numbers.tex",
      f"On five BEIR corpora with six quantizers, decision-calibrated boxes have a "
      f"{lo:.1f}--{hi:.0f}$\\times$ (median {med:.1f}$\\times$) lower oracle resolution cost than coverage-calibrated ones at "
      f"the same guarantee on the decision, with empirical validity {out['coverage_delta0.1_excl_sq4k1'][0]:.3f}--"
      f"{out['coverage_delta0.1_excl_sq4k1'][1]:.3f} at nominal 0.90 over 500 re-splits; the actual resolution procedure on residual "
      f"boxes uses a median {v['alg1_mean'][0]:.2f}$\\times$ the resolutions of the conformal re-ranking depth on average but "
      f"{v['alg1_p95'][0]:.1f}$\\times$ at the 95th percentile. "
      f"\\input{{tables/abstract_msmarco}}\n")
    msp = os.path.join(RESULTS_DIR, "msmarco", "report.json")
    if os.path.exists(msp):
        rep = json.load(open(msp))
        o = {q: rep[q]["0.1"] for q in ("pq32", "pq96", "sq4", "bin")}
        cr = [x["cost_sim_mean"] / x["cost_dec_mean"] for x in o.values()]
        txt = (f"On MS~MARCO (8.8M passages) decision calibration remains valid and lowers the oracle cost "
               f"{min(cr):.1f}--{max(cr):.1f}$\\times$ relative to coverage calibration, while for coarse codes residual boxes "
               f"become an order of magnitude more expensive than rank boxes.")
    else:
        txt = ""
    w("abstract_msmarco.tex", txt + "\n")
    return out


def main():
    table_data()
    if os.path.exists(os.path.join(RESULTS_DIR, "coverage.json")):
        table_coverage()
    table_certify(["pq32", "pq96", "sq4"], "certify.tex")
    table_certify(["pq16", "pq32", "pq96", "opq32", "sq4", "bin"], "certify_full.tex")
    table_strategies()
    if load_uq():
        table_qpp("kendall", "qpp_kendall.tex")
        table_qpp("auroc_fail", "qpp_auroc.tex")
        table_qpp("e_aurc", "qpp_eaurc.tex")
        table_qpp("pearson", "qpp_pearson.tex")
        table_qpp("kendall", "qpp_kendall_fused.tex", which="eval_fused")
        table_adaptive()
        table_significance()
    table_msmarco()
    table_determinacy()
    table_cascade()
    print(json.dumps(summary_numbers(), indent=1))


if __name__ == "__main__":
    main()
