"""Publication figures for the B1 paper (paper/figures/*.pdf). One visual system:
fixed categorical order  blue = decision-calibrated (ours), orange = coverage-calibrated, aqua = rank boxes,
gray = uncalibrated / reference. Single axis per panel, direct labels where few series, thin marks."""
import glob
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


def clean_log_y(ax):
    """Plain-number labels on log y axes, at most ~4 major ticks, no minor labels."""
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())

from .data import DATASETS, RESULTS_DIR
from .make_tables import NICE, QNICE

FIG = os.path.join(os.path.dirname(RESULTS_DIR), "paper", "figures")
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8f8d86", "#e4e3df"
COL = dict(dec=BLUE, cov=ORANGE, depth=AQUA, ref=MUTED)
LAB = dict(dec="Decision-calibrated (ours)", cov="Coverage-calibrated", depth="Rank boxes (conformal depth)")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.8,
    "pdf.fonttype": 42, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def legend_top(fig, handles, ncol, y=1.02):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), fontsize=7,
               handlelength=1.6, columnspacing=1.4)


# ------------------------------------------------------------------ Fig. 1: concept on a real query
def fig_concept(name="scifact", q="pq96", k=5, show=26):
    from . import encode
    from .credal import conformal_quantile
    from .data import load
    from .index import quantize
    st = pickle.load(open(os.path.join(os.path.dirname(RESULTS_DIR), "cache", f"{name}.{q}.stats.pkl"), "rb"))
    ds = load(name)
    X, Q = encode.get(name, "bge", "docs"), encode.get(name, "bge", "queries")
    Xh, r, _ = quantize(q, X)
    cal = st["cal"]
    c_dec = conformal_quantile(st["per_k"][k]["alpha_dec"][cal], 0.1)
    c_cov = conformal_quantile(st["alpha_cov"][cal], 0.1)
    # pick a test query whose decision-calibrated set is valid, with a non-winner outside its interval
    best = None
    for i in st["te"]:
        s, sh = Q[i] @ X.T, Q[i] @ Xh.T
        w = np.linalg.norm(Q[i]) * r
        o = np.argsort(-(sh + c_dec * w))[:show]
        T = set(np.argsort(-s)[:k])
        if st["per_k"][k]["alpha_dec"][i] > c_dec or not T <= set(o):
            continue
        miss = sum((s[d] > sh[d] + c_dec * w[d]) or (s[d] < sh[d] - c_dec * w[d]) for d in o if d not in T)
        if miss >= 2 and (best is None or miss > best[0]):
            best = (miss, i)
    i = best[1]
    s, sh = Q[i] @ X.T, Q[i] @ Xh.T
    w = np.linalg.norm(Q[i]) * r
    o = np.argsort(-sh)[:show]
    T = set(np.argsort(-s)[:k])
    sk1 = np.sort(s)[::-1][k]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True)
    for ax, c, lam, title in ((axes[0], c_cov, 1.0, f"Coverage-calibrated ($\\hat\\theta$={c_cov:.2f})"),
                              (axes[1], c_dec, 0.0, f"Decision-calibrated ($\\hat\\theta$={c_dec:.2f}, $l=\\hat s$)")):
        x = np.arange(len(o))
        lo, hi = sh[o] - lam * c * w[o], sh[o] + c * w[o]
        win = np.array([d in T for d in o])
        for j in range(len(o)):
            ax.plot([x[j], x[j]], [lo[j], hi[j]], color=BLUE if win[j] else "#b9c7d9", lw=3.2,
                    solid_capstyle="round", zorder=2)
        ax.scatter(x[win], s[o][win], s=16, color=INK, zorder=4, label="true score, winner")
        ax.scatter(x[~win], s[o][~win], s=16, facecolor="white", edgecolor=INK, lw=0.9, zorder=4,
                   label="true score, other")
        ax.axhline(sk1, color=ORANGE, lw=1.1, ls=(0, (4, 2)), zorder=3)
        ax.text(len(o) - 0.5, sk1, " $s_{(k+1)}$", color=ORANGE, va="bottom", ha="right", fontsize=7)
        ax.set_title(title, color=INK)
        ax.set_xlabel("documents, ordered by approximate score $\\hat s$")
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("score")
    handles = [Line2D([], [], color=BLUE, lw=3.2, label="interval of a true top-$k$ document"),
               Line2D([], [], color="#b9c7d9", lw=3.2, label="interval of another document"),
               Line2D([], [], marker="o", ls="", color=INK, ms=4, label="true score"),
               Line2D([], [], color=ORANGE, ls=(0, (4, 2)), lw=1.1, label="$(k{+}1)$-th true score")]
    legend_top(fig, handles, 4)
    fig.tight_layout(w_pad=1.0)
    save(fig, "concept.pdf")


# ------------------------------------------------------------------ Fig. 2: winner's curse
def fig_winners_curse(quants=("pq32", "pq96", "sq4", "bin")):
    cache = os.path.join(os.path.dirname(RESULTS_DIR), "cache")
    fig, axes = plt.subplots(1, len(quants), figsize=(7.0, 1.9), sharey=False)
    for ax, q in zip(axes, quants):
        data = {"random": [], "approx_top": [], "winners": []}
        for d in DATASETS:
            p = os.path.join(cache, f"{d}.{q}.stats.pkl")
            if os.path.exists(p):
                st = pickle.load(open(p, "rb"))
                for key in data:
                    data[key].append(st[f"wc_{key}"])
        data = {k_: np.concatenate(v) for k_, v in data.items()}
        lo, hi = np.percentile(np.concatenate(list(data.values())), [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        for key, col, lab in (("random", MUTED, "random documents"), ("approx_top", YELLOW, "approximate top-10"),
                              ("winners", BLUE, "true top-10")):
            h, e = np.histogram(data[key], bins=bins, density=True)
            ax.fill_between((e[:-1] + e[1:]) / 2, h, color=col, alpha=0.18, lw=0)
            ax.plot((e[:-1] + e[1:]) / 2, h, color=col, lw=1.4, label=lab)
            ax.axvline(np.median(data[key]), color=col, lw=0.9, ls=(0, (2, 2)))
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title(QNICE[q], color=INK)
        ax.set_yticks([])
        ax.set_xlabel("$(s-\\hat s)/(\\|q\\|\\,r)$")
    h, l_ = axes[0].get_legend_handles_labels()
    legend_top(fig, h, 3)
    fig.tight_layout(w_pad=0.8)
    save(fig, "winners_curse.pdf")


# ------------------------------------------------------------------ Fig. 3: validity (calibration plot)
def fig_validity():
    cov = json.load(open(os.path.join(RESULTS_DIR, "calib.json")))
    rs = json.load(open(os.path.join(RESULTS_DIR, "coverage.json")))
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3))
    ax = axes[0]
    rng = np.random.RandomState(0)
    for key, v in rs.items():
        for kk, dd in v.items():
            if key.endswith(".sq4") and kk == "1":
                continue
            for dl, (m, sd) in dd.items():
                ax.scatter(1 - float(dl) + rng.uniform(-0.012, 0.012), m, s=5, color=BLUE, alpha=0.45, lw=0)
    ax.text(0.765, 0.985, "each dot: one dataset $\\times$ quantizer $\\times$ $k$,\nmean over 500 re-splits",
            fontsize=6.3, color=INK2, va="top")
    ax.plot([0.75, 1], [0.75, 1], color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel("nominal $1-\\delta$")
    ax.set_ylabel("empirical validity")
    ax.set_title("500 re-splits, all datasets, quantizers, $k$", color=INK)
    ax.set_xlim(0.76, 0.97)
    ax.set_ylim(0.76, 1.0)
    ax = axes[1]
    ns = {}
    for key, r in cov.items():
        for x in r["n_sensitivity"]:
            ns.setdefault(x["n"], []).append((x["cov_mean"], x["cov_std"]))
    n_ = sorted(ns)
    m = [np.mean([a for a, _ in ns[n]]) for n in n_]
    sd = [np.mean([b for _, b in ns[n]]) for n in n_]
    ax.fill_between(n_, np.array(m) - sd, np.array(m) + sd, color=BLUE, alpha=0.15, lw=0)
    ax.plot(n_, m, color=BLUE, marker="o", ms=3)
    ax.axhline(0.9, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xticks(n_)
    ax.set_xticklabels([str(n) for n in n_])
    ax.set_xlabel("calibration queries $n$")
    ax.set_ylabel("validity (mean $\\pm$ sd over splits)")
    ax.set_title("$\\delta=0.1$, $k=10$", color=INK)
    fig.tight_layout(w_pad=1.5)
    save(fig, "validity.pdf")


# ------------------------------------------------------------------ Fig. 4: cost vs delta
def fig_delta(configs=(("scifact", "pq32"), ("fiqa", "pq32"), ("scidocs", "pq96"), ("nfcorpus", "bin")),
              cascades=(("arguana", "bgebase"), ("nfcorpus", "bgebase"))):
    cov = json.load(open(os.path.join(RESULTS_DIR, "calib.json")))
    cas = json.load(open(os.path.join(RESULTS_DIR, "cascade.json")))
    panels = [(f"{NICE[d]} / {QNICE[q]}", cov[f"{d}.{q}"]["delta_curve"]) for d, q in configs if f"{d}.{q}" in cov]
    panels += [(f"{NICE[d]} / cascade", cas[f"{d}.{s}.global"]["delta_curve"]) for d, s in cascades
               if f"{d}.{s}.global" in cas]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 1.95))
    for ax, (title, dc) in zip(axes, panels):
        x = [1 - p["delta"] for p in dc]
        for key in ("cov", "depth", "dec"):
            y = [p[f"cost_{key}"] if key != "depth" else p["depth"] for p in dc]
            y = [np.nan if v is None else v for v in y]
            ax.plot(x, y, color=COL[key], marker="o", ms=2.8, label=LAB[key])
        ax.set_yscale("log")
        clean_log_y(ax)
        ax.set_title(title, color=INK)
        ax.set_xlabel("$1-\\delta$")
    axes[0].set_ylabel("resolutions / query")
    h, l_ = axes[0].get_legend_handles_labels()
    legend_top(fig, h[::-1], 3)
    fig.tight_layout(w_pad=0.5)
    save(fig, "delta.pdf")


# ------------------------------------------------------------------ Fig. 5: cost distribution (tails)
def fig_tails(configs=(("scifact", "pq32"), ("fiqa", "pq32"), ("nfcorpus", "bin"))):
    fig, axes = plt.subplots(1, len(configs), figsize=(6.4, 1.95))
    for ax, (d, q) in zip(axes, configs):
        r = json.load(open(os.path.join(RESULTS_DIR, "certify", f"{d}.bge.{q}.json")))
        ad = json.load(open(os.path.join(RESULTS_DIR, "adaptive.json")))[f"{d}.{q}"]["test"]
        c = np.sort(np.array(r["credal"]["decision"]["lucb"]["n_ref_all"]))
        ax.plot(c, np.arange(1, len(c) + 1) / len(c), color=BLUE, drawstyle="steps-post", label=LAB["dec"])
        Rh = ad["confdepth"]["mean"]
        ax.plot([Rh, Rh], [0, 1], color=AQUA, label=LAB["depth"])
        ax.axvline(c.mean(), color=BLUE, lw=0.8, ls=(0, (2, 2)))
        ax.text(c.mean(), 0.04, " mean", color=BLUE, fontsize=6.5)
        ax.set_xscale("log")
        ax.set_title(f"{NICE[d]} / {QNICE[q]}", color=INK)
        ax.set_xlabel("resolutions / query")
    axes[0].set_ylabel("fraction of queries")
    h, _ = axes[0].get_legend_handles_labels()
    legend_top(fig, h, 2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "tails.pdf")


# ------------------------------------------------------------------ Fig. 6: risk control
def fig_risk(configs=(("scifact", "pq32"), ("nfcorpus", "pq32"), ("fiqa", "pq32"), ("scidocs", "pq32"),
                      ("arguana", "pq32"))):
    cov = json.load(open(os.path.join(RESULTS_DIR, "calib.json")))
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.2))
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    for col, (d, q) in zip(cols, configs):
        rc = cov.get(f"{d}.{q}", {}).get("risk_control")
        if not rc:
            continue
        ra = json.load(open(os.path.join(RESULTS_DIR, "risk_actual.json")))[f"{d}.{q}"]
        exact = ra["exact_0.1"]["cost_mean"]
        eps_ = [x["eps"] for x in rc if f"eps_{x['eps']}" in ra]
        axes[0].plot(eps_, [ra[f"eps_{e}"]["cost_mean"] / exact for e in eps_], color=col, marker="o", ms=2.8,
                     label=NICE[d])
        pos = [(x["eps"], x["miss_resplit"]) for x in rc if x.get("miss_resplit")]      # zero realized loss cannot be drawn on log axes
        axes[1].plot(*zip(*pos), color=col, marker="o", ms=2.8)
    axes[0].axhline(1, color=INK2, lw=0.8, ls=(0, (3, 2)))
    axes[0].set_ylabel("Alg. 1 resolutions relative\nto exact certification ($\\delta=0.1$)")
    axes[1].plot([0.002, 0.1], [0.002, 0.1], color=INK2, lw=0.8, ls=(0, (3, 2)))
    axes[1].set_ylabel("worst-case miss fraction\n(mean over 300 re-splits)")
    axes[1].text(0.0022, 0.075, "dashed: guarantee\n$\\mathbb{E}[\\ell]\\leq\\varepsilon$", fontsize=6.3, color=INK2, va="top")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("risk level $\\varepsilon$")
    axes[1].set_yscale("log")
    h, _ = axes[0].get_legend_handles_labels()
    legend_top(fig, h, 5)
    fig.tight_layout(w_pad=1.5)
    save(fig, "risk.pdf")


# ------------------------------------------------------------------ Fig. 7: k sensitivity
def fig_k(quant="pq32"):
    cov = json.load(open(os.path.join(RESULTS_DIR, "calib.json")))
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.1))
    for d, col in zip(DATASETS, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]):
        r = cov.get(f"{d}.{quant}")
        if not r:
            continue
        ks = [x["k"] for x in r["k_sensitivity"]]
        axes[0].plot(ks, [x["cost_cov"] / x["cost_dec"] for x in r["k_sensitivity"]], color=col, marker="o", ms=2.8,
                     label=NICE[d])
        axes[1].plot(ks, [x["depth"] / x["cost_dec"] for x in r["k_sensitivity"]], color=col, marker="o", ms=2.8)
    axes[0].set_ylabel("coverage / decision cost")
    axes[1].set_ylabel("rank-box / residual-box cost")
    axes[1].axhline(1, color=INK2, lw=0.8, ls=(0, (3, 2)))
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20, 50])
        ax.set_xticklabels(["1", "5", "10", "20", "50"])
        ax.set_xlabel("$k$")
    h, _ = axes[0].get_legend_handles_labels()
    legend_top(fig, h, 5)
    fig.tight_layout(w_pad=1.5)
    save(fig, "k_sensitivity.pdf")


# ------------------------------------------------------------------ Fig. 8: Mondrian
def fig_mondrian(quant="pq32"):
    cov = json.load(open(os.path.join(RESULTS_DIR, "calib.json")))
    ds = [d for d in DATASETS if f"{d}.{quant}" in cov]
    fig, ax = plt.subplots(figsize=(4.6, 2.0))
    wbar = 0.13
    for j, d in enumerate(ds):
        for b, x in enumerate(cov[f"{d}.{quant}"]["mondrian"]):
            xpos = j + (b - 1) * 2.4 * wbar
            ax.bar(xpos - wbar / 2, x["valid_marginal"], wbar, color=MUTED, lw=0)
            ax.bar(xpos + wbar / 2, x["valid_mondrian"], wbar, color=BLUE, lw=0)
    ax.axhline(0.9, color=ORANGE, lw=0.9, ls=(0, (3, 2)))
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([NICE[d] for d in ds])
    ax.set_ylim(0.84, 0.96)
    ax.set_ylabel("validity per bin")
    ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MUTED, label="marginal calibration"),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Mondrian (per bin)"),
               Line2D([], [], color=ORANGE, ls=(0, (3, 2)), label="$1-\\delta$")]
    legend_top(fig, handles, 3)
    fig.tight_layout()
    save(fig, "mondrian.pdf")


# ------------------------------------------------------------------ Fig: calibrated levels versus corpus size
def fig_scaling():
    rep = json.load(open(os.path.join(RESULTS_DIR, "msmarco", "report.json")))
    quants = ["pq32", "pq96", "sq4", "bin"]
    fig, axes = plt.subplots(1, len(quants), figsize=(7.0, 1.95))
    for ax, q in zip(axes, quants):
        pts = []
        for d in DATASETS:
            r = json.load(open(os.path.join(RESULTS_DIR, "certify", f"{d}.bge.{q}.json")))
            pts.append((r["N"], r["coverage_dec"]["0.1"]["c"], r["coverage"]["0.1"]["c"]))
        o = rep[q]["0.1"]
        pts.append((8841823, o["c_dec"], o["c_sim"]))
        pts.sort()
        N = [p[0] for p in pts]
        ax.plot(N, [p[2] for p in pts], color=COL["cov"], marker="o", ms=3, label="Coverage-calibrated level")
        ax.plot(N, [p[1] for p in pts], color=COL["dec"], marker="o", ms=3, label="Decision-calibrated level")
        ax.scatter([N[-1]], [pts[-1][2]], s=40, facecolor="none", edgecolor=COL["cov"], lw=1.2, zorder=5)
        ax.scatter([N[-1]], [pts[-1][1]], s=40, facecolor="none", edgecolor=COL["dec"], lw=1.2, zorder=5)
        ax.set_xscale("log")
        ax.set_title(QNICE[q], color=INK)
        ax.set_xlabel("corpus size $N$")
        ax.set_ylim(0, None)
    axes[0].set_ylabel("calibrated level $\\hat\\theta$ ($\\delta=0.1$)")
    axes[-1].annotate("MS MARCO", (8841823, rep["bin"]["0.1"]["c_sim"]), xytext=(-40, -28), textcoords="offset points",
                      fontsize=6.3, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    h, _ = axes[0].get_legend_handles_labels()
    legend_top(fig, h, 2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "scaling.pdf")


# ------------------------------------------------------------------ Fig: imprecision families relative to rank boxes
def fig_families():
    h = json.load(open(os.path.join(RESULTS_DIR, "hybrid.json")))
    hr = json.load(open(os.path.join(RESULTS_DIR, "hybrid_resplit.json")))
    ad = json.load(open(os.path.join(RESULTS_DIR, "adaptive.json")))
    ms = json.load(open(os.path.join(RESULTS_DIR, "msmarco", "hybrid.json")))
    quants = ["pq16", "pq32", "pq96", "opq32", "sq4", "bin"]
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    rng = np.random.RandomState(0)
    for j, q in enumerate(quants):
        for d in DATASETS:
            key = f"{d}.{q}"
            res = hr[key]["cost_ratio_residual"]                # paired: same re-splits for all three families
            tr = hr[key]["cost_ratio"]
            ax.scatter(j - 0.15 + rng.uniform(-0.06, 0.06), res, s=14, color=BLUE, alpha=0.8, lw=0)
            ax.scatter(j + 0.15 + rng.uniform(-0.06, 0.06), tr, s=14, color=MAGENTA, alpha=0.9, lw=0)
        if q in ms:
            m = ms[q]
            ax.scatter(j - 0.15, m["residual_oracle"]["mean"] / m["rank"]["Rhat"], s=46, marker="D", facecolor="white",
                       edgecolor=BLUE, lw=1.4, zorder=5)
            ax.scatter(j + 0.15, m["hybrid"]["mean"] / m["rank"]["Rhat"], s=46, marker="D", facecolor="white",
                       edgecolor=MAGENTA, lw=1.4, zorder=5)
    ax.axhline(1, color=AQUA, lw=1.4)
    ax.text(-0.45, 1.03, "rank boxes", color=AQUA, fontsize=6.8, va="bottom", ha="left")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_ylim(0.45, 30)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}$\\times$"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(quants)))
    ax.set_xticklabels([QNICE[q] for q in quants])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("resolutions relative to\nrank boxes (same guarantee)")
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="Residual boxes"),
               Line2D([], [], marker="o", ls="", color=MAGENTA, ms=4, label="Truncated residual boxes"),
               Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=INK2, ms=5,
                      label="MS MARCO (others: BEIR)")]
    legend_top(fig, handles, 3)
    fig.tight_layout()
    save(fig, "families.pdf")


# ------------------------------------------------------------------ Appendix: all configurations
def fig_delta_all():
    cov = json.load(open(os.path.join(RESULTS_DIR, "calib.json")))
    quants = ["pq16", "pq32", "pq96", "opq32", "sq4", "bin"]
    fig, axes = plt.subplots(len(quants), len(DATASETS), figsize=(7.2, 8.6))
    for i, q in enumerate(quants):
        for j, d in enumerate(DATASETS):
            ax = axes[i, j]
            dc = cov[f"{d}.{q}"]["delta_curve"]
            x = [1 - p["delta"] for p in dc]
            for key in ("cov", "depth", "dec"):
                y = [p["depth"] if key == "depth" else p[f"cost_{key}"] for p in dc]
                ax.plot(x, [np.nan if v is None else v for v in y], color=COL[key], marker="o", ms=1.8, lw=1.3,
                        label=LAB[key])
            ax.set_yscale("log")
            clean_log_y(ax)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(NICE[d], color=INK)
            if j == 0:
                ax.set_ylabel(QNICE[q], color=INK)
            if i == len(quants) - 1:
                ax.set_xlabel("$1-\\delta$", fontsize=7)
            else:
                ax.set_xticklabels([])
    h, _ = axes[0, 0].get_legend_handles_labels()
    legend_top(fig, h[::-1], 3, y=1.0)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    save(fig, "delta_all.pdf")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["concept", "winners_curse", "validity", "delta", "tails", "risk", "k", "mondrian"]
    fns = dict(concept=fig_concept, winners_curse=fig_winners_curse, validity=fig_validity, delta=fig_delta,
               tails=fig_tails, risk=fig_risk, k=fig_k, mondrian=fig_mondrian, delta_all=fig_delta_all,
               scaling=fig_scaling, families=fig_families)
    for w_ in which:
        fns[w_]()
