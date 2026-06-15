#!/usr/bin/env python3
"""
Analyze RefRover benchmark results.

Answers four research questions from benchmark_results.tsv:
  1. Does selection rule matter?         (MAG yield by rule × k × matrix)
  2. Does k matter?                      (yield curve over k per rule)
  3. Do proxy scores predict MAG yield?  (frac_variance / effective_rank /
                                          tiered_axis_count vs weighted_mags)
  4. Jaccard vs containment matrix?      (cross-matrix ranking agreement)

Usage:
    python scripts/analyze_benchmark.py \\
        --results work/benchmark/benchmark_results.tsv \\
        --outdir  analysis/

Requires: numpy, pandas, scipy, matplotlib (all available in refrover-benchmark env).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed on HPC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ── consistent colour / style per rule ───────────────────────────────────────

RULE_COLORS = {
    "random": "#999999",
    "maxmin": "#4477AA",
    "css":    "#EE6677",
    "kmedoids":   "#228833",
    "archetype":  "#CCBB44",
    "greedy_var": "#AA3377",
}
DEFAULT_COLOR = "#333333"


# ── data loading ─────────────────────────────────────────────────────────────

def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    n_total = len(df)
    if "error" in df.columns:
        errors = df["error"].notna()
        df_ok = df[~errors].copy()
        n_err = int(errors.sum())
    else:
        df_ok = df.copy()
        n_err = 0
    print(f"Loaded {n_total} cells: {len(df_ok)} succeeded, {n_err} errors")
    if n_err:
        print(f"  {n_err} error cells excluded (check error.json files for details)")
    return df_ok


# ── stat helpers ──────────────────────────────────────────────────────────────

def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p)


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


# ── plot 1: heatmap of mean weighted MAGs ────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    matrices = sorted(df["matrix"].unique())
    fig, axes = plt.subplots(
        1, len(matrices), figsize=(5 * len(matrices), 4), squeeze=False
    )
    for ax, mat in zip(axes[0], matrices):
        sub = df[df["matrix"] == mat]
        p = (
            sub.groupby(["rule", "k"])["weighted_mags"]
            .mean()
            .unstack("k")
        )
        im = ax.imshow(p.values, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(p.columns)))
        ax.set_xticklabels(p.columns)
        ax.set_yticks(range(len(p.index)))
        ax.set_yticklabels(p.index)
        ax.set_xlabel("k")
        ax.set_title(f"matrix: {mat}")
        for i in range(p.values.shape[0]):
            for j in range(p.values.shape[1]):
                ax.text(j, i, f"{p.values[i, j]:.1f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if p.values[i, j] > p.values.max() * 0.6 else "black")
        plt.colorbar(im, ax=ax, label="mean weighted MAGs")
    fig.suptitle("Mean weighted MAG yield per (matrix, rule, k)", y=1.02)
    fig.tight_layout()
    out = outdir / "01_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── plot 2: yield vs k curves ─────────────────────────────────────────────────

def plot_k_curves(df: pd.DataFrame, outdir: Path) -> None:
    matrices = sorted(df["matrix"].unique())
    fig, axes = plt.subplots(
        1, len(matrices), figsize=(5 * len(matrices), 4),
        squeeze=False, sharey=True,
    )
    for ax, mat in zip(axes[0], matrices):
        sub = df[df["matrix"] == mat]
        for rule, grp in sub.groupby("rule"):
            curve = (
                grp.groupby("k")["weighted_mags"]
                .agg(["mean", "sem"])
                .reset_index()
            )
            c = RULE_COLORS.get(rule, DEFAULT_COLOR)
            ax.plot(curve["k"], curve["mean"], color=c, label=rule, marker="o", lw=1.8)
            ax.fill_between(
                curve["k"],
                curve["mean"] - curve["sem"],
                curve["mean"] + curve["sem"],
                alpha=0.15, color=c,
            )
        ax.set_xlabel("k  (references per focal assembly)")
        ax.set_ylabel("mean weighted MAGs")
        ax.set_title(f"matrix: {mat}")
        ax.legend(title="rule", fontsize=8)
    fig.suptitle("Weighted MAG yield vs k  (mean ± SEM over focal assemblies)")
    fig.tight_layout()
    out = outdir / "02_k_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── plot 3: proxy score correlations ─────────────────────────────────────────

def plot_proxy_correlations(df: pd.DataFrame, outdir: Path) -> None:
    proxies = [c for c in ["frac_variance", "effective_rank", "tiered_axis_count"]
               if c in df.columns]
    if not proxies:
        print("  No proxy score columns found — skipping correlation plot")
        return

    matrices = sorted(df["matrix"].unique())
    n_row, n_col = len(proxies), len(matrices)
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(4.5 * n_col, 3.8 * n_row), squeeze=False
    )
    for ri, proxy in enumerate(proxies):
        for ci, mat in enumerate(matrices):
            ax = axes[ri][ci]
            sub = df[df["matrix"] == mat].dropna(subset=[proxy, "weighted_mags"])
            for rule, grp in sub.groupby("rule"):
                c = RULE_COLORS.get(rule, DEFAULT_COLOR)
                ax.scatter(grp[proxy], grp["weighted_mags"],
                           alpha=0.35, s=14, color=c, label=rule)
            x = sub[proxy].to_numpy(float)
            y = sub["weighted_mags"].to_numpy(float)
            r, pval = spearman(x, y)
            # regression line
            if np.isfinite(r):
                m, b, *_ = stats.linregress(x[np.isfinite(x) & np.isfinite(y)],
                                             y[np.isfinite(x) & np.isfinite(y)])
                xl = np.array([x[np.isfinite(x)].min(), x[np.isfinite(x)].max()])
                ax.plot(xl, m * xl + b, "k--", lw=0.9, alpha=0.6)
            ax.set_xlabel(proxy.replace("_", " "), fontsize=9)
            ax.set_ylabel("weighted MAGs", fontsize=9)
            ax.set_title(f"{mat}  |  r={r:.2f}  {fmt_p(pval)}", fontsize=9)
            if ri == 0 and ci == 0:
                ax.legend(title="rule", fontsize=7, markerscale=1.5)

    fig.suptitle("Proxy scores vs real MAG yield", y=1.01)
    fig.tight_layout()
    out = outdir / "03_proxy_correlations.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── plot 4: Jaccard vs containment matrix agreement ──────────────────────────

def plot_matrix_comparison(df: pd.DataFrame, outdir: Path) -> None:
    if df["matrix"].nunique() < 2:
        print("  Only one matrix present — skipping Jaccard vs Containment comparison")
        return

    merged = (
        df.pivot_table(index=["rule", "k", "focal"],
                       columns="matrix", values="weighted_mags")
        .dropna()
        .reset_index()
    )
    if "jaccard" not in merged.columns or "containment" not in merged.columns:
        print("  Need both 'jaccard' and 'containment' matrices — skipping comparison")
        return

    rules = sorted(merged["rule"].unique())
    fig, axes = plt.subplots(
        1, len(rules), figsize=(4 * len(rules), 4),
        squeeze=False, sharey=True, sharex=True,
    )
    for ax, rule in zip(axes[0], rules):
        sub = merged[merged["rule"] == rule]
        c = RULE_COLORS.get(rule, DEFAULT_COLOR)
        ax.scatter(sub["jaccard"], sub["containment"],
                   alpha=0.45, s=16, color=c)
        lim = max(sub[["jaccard", "containment"]].max().max() * 1.08, 1)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        r, pval = spearman(sub["jaccard"].to_numpy(), sub["containment"].to_numpy())
        ax.set_title(f"{rule}\nr={r:.2f}  {fmt_p(pval)}", fontsize=9)
        ax.set_xlabel("Jaccard matrix")
        if ax is axes[0][0]:
            ax.set_ylabel("Containment matrix")
    fig.suptitle("Weighted MAGs: Jaccard vs Containment feature matrix\n(each dot = one focal assembly × k cell)")
    fig.tight_layout()
    out = outdir / "04_matrix_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── plot 5: per-focal variability  ────────────────────────────────────────────

def plot_focal_variability(df: pd.DataFrame, outdir: Path) -> None:
    """Box plots of weighted_mags per focal assembly (best k per rule, jaccard)."""
    sub = df[df["matrix"] == df["matrix"].iloc[0]]
    focal_means = (
        sub.groupby("focal")["weighted_mags"].mean()
        .sort_values(ascending=False)
    )
    focals_ordered = focal_means.index.tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(focals_ordered) * 0.35), 4))
    data = [sub[sub["focal"] == f]["weighted_mags"].values for f in focals_ordered]
    ax.boxplot(data, patch_artist=True,
               boxprops=dict(facecolor="#4477AA", alpha=0.6),
               medianprops=dict(color="black", lw=1.5),
               flierprops=dict(marker=".", markersize=3))
    ax.set_xticks(range(1, len(focals_ordered) + 1))
    ax.set_xticklabels(focals_ordered, rotation=90, fontsize=7)
    ax.set_ylabel("weighted MAGs (across rules × k)")
    ax.set_title(f"Per-focal assembly variability  (matrix: {df['matrix'].iloc[0]})")
    fig.tight_layout()
    out = outdir / "05_focal_variability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── text summary ──────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n── Mean weighted MAGs by (matrix, rule, k) ──────────────────────────")
    summary = (
        df.groupby(["matrix", "rule", "k"])["weighted_mags"]
        .agg(mean="mean", std="std", n="count")
        .round(2)
    )
    print(summary.to_string())

    proxies = [c for c in ["frac_variance", "effective_rank", "tiered_axis_count"]
               if c in df.columns]
    if proxies:
        print("\n── Spearman correlation: proxy score vs weighted_mags ───────────────")
        rows = []
        for mat in sorted(df["matrix"].unique()):
            sub = df[df["matrix"] == mat]
            for proxy in proxies:
                r, pval = spearman(sub[proxy].to_numpy(float),
                                   sub["weighted_mags"].to_numpy(float))
                rows.append({"matrix": mat, "proxy": proxy,
                              "spearman_r": round(r, 3), "p_value": round(pval, 4)})
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n── Best (matrix, rule, k) by mean weighted MAGs ────────────────────")
    best = (
        df.groupby(["matrix", "rule", "k"])["weighted_mags"]
        .mean()
        .reset_index()
        .sort_values("weighted_mags", ascending=False)
        .head(10)
        .round(2)
    )
    print(best.to_string(index=False))


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results", required=True,
        help="benchmark_results.tsv from 'refrover aggregate-results'",
    )
    ap.add_argument(
        "--outdir", default="analysis",
        help="Directory for output plots (default: analysis/)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load(Path(args.results))
    if df.empty:
        print("No successful cells to analyze.")
        sys.exit(1)

    print(f"\nGenerating plots → {outdir}/")
    plot_heatmap(df, outdir)
    plot_k_curves(df, outdir)
    plot_proxy_correlations(df, outdir)
    plot_matrix_comparison(df, outdir)
    plot_focal_variability(df, outdir)
    print_summary(df)
    print(f"\nDone. Open {outdir}/ to view plots.")


if __name__ == "__main__":
    main()
