"""Paper figures (paper/figures/*.pdf) from results/*.json."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import DATASETS, RESULTS_DIR
from .make_tables import NICE, QNICE

FIG = os.path.join(os.path.dirname(RESULTS_DIR), "paper", "figures")
# reference categorical palette, fixed order (slot 1 blue, 2 orange, 3 aqua, 4 yellow, 5 magenta)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2, "pdf.fonttype": 42,
})


def fig_cost(quant="pq32"):
    files = {json.load(open(p))["dataset"]: json.load(open(p))
             for p in glob.glob(os.path.join(RESULTS_DIR, "certify", f"*.bge.{quant}.json"))}
    ds = [d for d in DATASETS if d in files and "rerank_curve" in files[d]]
    fig, axes = plt.subplots(1, len(ds), figsize=(1.45 * len(ds) + 0.4, 2.2), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), ds):
        r = files[d]
        cur = r["rerank_curve"]
        ax.plot([c["R"] for c in cur], [c["perfect"] for c in cur], color=SERIES[0], marker="o", ms=3,
                label="Fixed depth (uncalibrated)")
        dec = r["credal"]["decision"]["lucb"]
        ax.scatter([dec["n_ref_mean"]], [dec["perfect"]], s=46, color=SERIES[1], zorder=5,
                   edgecolor="white", linewidth=1.5, label="Residual boxes (Alg. 1)")
        ad = json.load(open(os.path.join(RESULTS_DIR, "adaptive.json"))).get(f"{d}.{quant}")
        if ad:
            cd = ad["test"]["confdepth"]
            ax.scatter([cd["mean"]], [cd["recovered"]], s=70, marker="s", facecolor="none", zorder=6,
                       edgecolor=SERIES[2], linewidth=2, label="Rank boxes (conformal depth)")
        ax.set_xscale("log")
        ax.set_title(NICE[d], color=INK, fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("exact top-10 recovered")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("exact re-scorings per query", fontsize=8, color=INK)
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.88))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"cost_{quant}.pdf"))
    print("wrote", f"cost_{quant}.pdf")


def fig_shrink():
    ds = [d for d in DATASETS if os.path.exists(os.path.join(RESULTS_DIR, "shrink", f"{d}.json"))]
    if not ds:
        return
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.1))
    for key, ax, lab in (("width_rel", axes[0], "P(relevant) interval width"),
                         ("GH", axes[1], "epistemic gap (nats)")):
        for i, d in enumerate(ds):
            r = json.load(open(os.path.join(RESULTS_DIR, "shrink", f"{d}.json")))
            # same truncation as the slope fit: region resolved by >= 10 sampled parameters
            ns = sorted(int(n) for n in r["per_n"] if r["per_n"][n]["retained"][0] >= 10)
            y = [r["per_n"][str(n)][key][0] for n in ns]
            ax.plot(ns, y, color=SERIES[i], marker="o", ms=3,
                    label=f"{NICE[d]} (slope {r['slope_' + key]:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("labelled calibration queries $n$")
        ax.set_ylabel(lab)
    axes[1].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "shrink.pdf"))
    print("wrote shrink.pdf")


if __name__ == "__main__":
    for q in ("pq32", "pq96"):
        fig_cost(q)
    fig_shrink()
