#!/usr/bin/env python3
"""
Analyze RefRover MAG benchmark results — the Tier-1 proxy vs Tier-2 yield question.

Reads ``benchmark_results.tsv`` (from ``refrover aggregate-results``) and answers
the open research question in CLAUDE.md / PLAN.md §8:

    Does the cheap Tier-1 proxy (frac_variance) rank selection configurations the
    same way real MAG yield (Tier-2) does — so we can pick a selector *without*
    paying for alignment + MetaBAT2 + CheckM2?

Why this script exists alongside ``analyze_benchmark.R``:
  * The R script makes the plots but (a) does ``k <- as.integer(k)``, which turns
    every ``k="adaptive"`` row into NA, and (b) reports a single *pooled* Spearman
    that is the wrong statistic (see below). This script fixes both and prints the
    correct decomposition. No R/tidyverse install needed — pandas/numpy/scipy only.

The statistical point
---------------------
``frac_variance`` is the fraction of *that matrix's own* total cross-sample
variance a selection explains. The containment matrix and the Jaccard matrix have
different total variances, so frac_variance is **not comparable across matrices** —
a pooled correlation over both matrices is driven by the between-matrix mean offset
(a Simpson's-paradox confound), not by whether the proxy tracks yield. Likewise the
focal assembly is a massive blocking factor (some focals yield ~50 MAGs, others
~30), so any correlation must condition on focal too.

So we report, for each proxy × yield metric:
  1. POOLED Spearman            — the confounded number, shown only as a warning.
  2. WITHIN-MATRIX Spearman     — removes the cross-matrix scale confound.
  3. WITHIN-(matrix, focal)     — the fair test: does the proxy pick the right
                                  rule/k when matrix & focal are held fixed?
  4. TOP-1 HIT RATE             — per (matrix, focal), does argmax(proxy) select
                                  the same config as argmax(yield)? The most
                                  decision-relevant summary: "if I trust the proxy
                                  to pick the config, how often do I get the real
                                  winner?"

Also emitted: the mean-yield leaderboard, a Jaccard-vs-containment cross-matrix
agreement summary, and an adaptive-k vs fixed-k comparison.

Usage
-----
    python scripts/analyze_benchmark.py \
        --results work/benchmark3/benchmark_results.tsv \
        --outdir  work/benchmark3/analysis/

Outputs (TSV, plus a printed summary):
    leaderboard.tsv            mean/median yield per (matrix, rule, k_label)
    proxy_correlations.tsv     pooled / within-matrix / within-focal Spearman
    top1_hit_rate.tsv          proxy-picks-real-winner rate per matrix
    cross_matrix.tsv           jaccard vs containment yield agreement per rule
    adaptive_vs_fixed.tsv      adaptive-k vs fixed-k yield per matrix, rule
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROXY_COLS = ["frac_variance", "effective_rank", "tiered_axis_count"]
MIN_N_FOR_R = 4  # need at least this many points for a meaningful Spearman


# ── load & normalize ──────────────────────────────────────────────────────────

CELL_KEY = ["matrix", "rule", "k", "focal"]


def load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split aggregated results into (successful cells, error cells).

    Defensively drops stale error rows: a cell that failed once then succeeded on
    a --resume retry can appear as both (older aggregate_results globbed error.json
    independently of result.json). A success for a cell is authoritative — an error
    row for the same (matrix, rule, k, focal) is discarded so the failure report
    isn't inflated and the cell isn't double-counted.
    """
    raw = pd.read_csv(path, sep="\t")
    if "error" not in raw.columns:
        return raw.copy(), raw.iloc[0:0].copy()

    ok = raw[raw["error"].isna()].copy()
    err = raw[raw["error"].notna()].copy()
    if set(CELL_KEY).issubset(raw.columns) and not err.empty:
        succeeded = set(map(tuple, ok[CELL_KEY].astype(str).itertuples(index=False)))
        keep = ~err[CELL_KEY].astype(str).apply(tuple, axis=1).isin(succeeded)
        err = err[keep].copy()
    return ok, err


def add_k_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve ``k`` so adaptive rows survive.

    ``k_label`` is the original string ("10" or "adaptive") — the axis you group
    by. ``k_eff`` is the integer actually used: ``k_actual`` for adaptive rows,
    else ``int(k)``. The R script's ``as.integer(k)`` silently NA's "adaptive";
    this keeps both views.
    """
    df = df.copy()
    df["k_label"] = df["k"].astype(str)
    k_actual = pd.to_numeric(df.get("k_actual"), errors="coerce")
    k_int = pd.to_numeric(df["k"], errors="coerce")
    df["k_eff"] = k_actual.where(df["k_label"] == "adaptive", k_int)
    return df


def pick_yield_metrics(df: pd.DataFrame) -> list[str]:
    """Yield metrics to analyze, primary first. sum_qs preferred over weighted_mags."""
    return [c for c in ("sum_qs", "weighted_mags") if c in df.columns]


# ── analyses ──────────────────────────────────────────────────────────────────

def leaderboard(df: pd.DataFrame, yvar: str) -> pd.DataFrame:
    out = (
        df.groupby(["matrix", "rule", "k_label"], dropna=False)[yvar]
        .agg(mean="mean", median="median", std="std", n="count")
        .reset_index()
        .sort_values("mean", ascending=False, ignore_index=True)
    )
    return out


def _grouped_spearman(df: pd.DataFrame, group_cols: list[str],
                      proxy: str, yvar: str) -> pd.DataFrame:
    """Spearman(proxy, yvar) within each group; one row per group meeting MIN_N."""
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        x = g[proxy].to_numpy(dtype=float)
        y = g[yvar].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < MIN_N_FOR_R or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
            continue
        r, p = spearmanr(x[ok], y[ok])
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        rec.update(rho=r, p=p, n=int(ok.sum()))
        rows.append(rec)
    return pd.DataFrame(rows)


def proxy_correlations(df: pd.DataFrame, yvar: str) -> pd.DataFrame:
    """Pooled / within-matrix / within-(matrix,focal) Spearman for each proxy.

    The within-focal row is an n-weighted mean of per-(matrix,focal) rho — the
    fair test. Pooled is included but flagged: it is confounded by the cross-matrix
    frac_variance scale offset and by between-focal yield differences.
    """
    proxies = [c for c in PROXY_COLS if c in df.columns]
    rows = []
    for proxy in proxies:
        x = df[proxy].to_numpy(dtype=float)
        y = df[yvar].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= MIN_N_FOR_R:
            r, p = spearmanr(x[ok], y[ok])
            rows.append(dict(proxy=proxy, scope="pooled (CONFOUNDED)",
                             rho=r, p=p, n=int(ok.sum()), n_groups=1))

        per_matrix = _grouped_spearman(df, ["matrix"], proxy, yvar)
        for _, r in per_matrix.iterrows():
            rows.append(dict(proxy=proxy, scope=f"within-matrix[{r['matrix']}]",
                             rho=r["rho"], p=r["p"], n=int(r["n"]), n_groups=1))

        per_mf = _grouped_spearman(df, ["matrix", "focal"], proxy, yvar)
        if not per_mf.empty:
            w = per_mf["n"]
            rows.append(dict(
                proxy=proxy, scope="within-(matrix,focal) [FAIR TEST, n-wtd mean]",
                rho=float(np.average(per_mf["rho"], weights=w)),
                p=np.nan, n=int(w.sum()), n_groups=int(len(per_mf)),
            ))
    return pd.DataFrame(rows)


def top1_hit_rate(df: pd.DataFrame, yvar: str) -> pd.DataFrame:
    """Per (matrix, focal): does the config with max proxy == config with max yield?

    A config here is a (rule, k_label) combination. Hit rate is the fraction of
    (matrix, focal) blocks where trusting the proxy to pick the config would have
    landed on the real yield winner. Reported per matrix and per proxy.
    """
    proxies = [c for c in PROXY_COLS if c in df.columns]
    rows = []
    for proxy in proxies:
        for matrix, gm in df.groupby("matrix"):
            hits = total = 0
            for _, g in gm.groupby("focal"):
                g = g.dropna(subset=[proxy, yvar])
                if len(g) < 2:
                    continue
                total += 1
                proxy_pick = g.loc[g[proxy].idxmax(), ["rule", "k_label"]]
                yield_win = g.loc[g[yvar].idxmax(), ["rule", "k_label"]]
                if proxy_pick.equals(yield_win):
                    hits += 1
            if total:
                rows.append(dict(proxy=proxy, matrix=matrix,
                                 hit_rate=hits / total, n_focals=total))
    return pd.DataFrame(rows)


def cross_matrix(df: pd.DataFrame, yvar: str) -> pd.DataFrame:
    """Jaccard-vs-containment yield agreement per rule (paired on rule,k_label,focal)."""
    wide = (
        df.pivot_table(index=["rule", "k_label", "focal"], columns="matrix",
                       values=yvar, aggfunc="mean")
        .reset_index()
    )
    if not {"jaccard", "containment"}.issubset(wide.columns):
        return pd.DataFrame()
    rows = []
    for rule, g in wide.groupby("rule"):
        pair = g[["jaccard", "containment"]].dropna()
        if len(pair) < MIN_N_FOR_R:
            continue
        r, p = spearmanr(pair["jaccard"], pair["containment"])
        rows.append(dict(
            rule=rule, rho=r, p=p, n=len(pair),
            mean_jaccard=pair["jaccard"].mean(),
            mean_containment=pair["containment"].mean(),
            containment_minus_jaccard=pair["containment"].mean() - pair["jaccard"].mean(),
        ))
    return pd.DataFrame(rows)


def adaptive_vs_fixed(df: pd.DataFrame, yvar: str) -> pd.DataFrame:
    """Adaptive-k vs each fixed-k yield per (matrix, rule), plus mean k chosen."""
    out = (
        df.groupby(["matrix", "rule", "k_label"], dropna=False)
        .agg(mean_yield=(yvar, "mean"), mean_k_eff=("k_eff", "mean"), n=(yvar, "count"))
        .reset_index()
        .sort_values(["matrix", "rule", "k_label"], ignore_index=True)
    )
    return out


# ── driver ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path,
                    help="benchmark_results.tsv from refrover aggregate-results")
    ap.add_argument("--outdir", type=Path, default=Path("benchmark_analysis"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ok, err = load(args.results)
    ok = add_k_columns(ok)
    metrics = pick_yield_metrics(ok)
    if not metrics:
        raise SystemExit("No yield column (sum_qs / weighted_mags) found in results.")
    yvar = metrics[0]

    print(f"Loaded {len(ok) + len(err)} cells: {len(ok)} ok, {len(err)} errors")
    if len(err):
        cols = [c for c in ["matrix", "rule", "k", "focal", "error"] if c in err.columns]
        print("  Failed cells:")
        print(err[cols].to_string(index=False))
    print(f"Primary yield metric: {yvar}"
          f"{'  (weighted_mags fallback — sum_qs absent)' if yvar != 'sum_qs' else ''}\n")

    lb = leaderboard(ok, yvar)
    corr = proxy_correlations(ok, yvar)
    hits = top1_hit_rate(ok, yvar)
    xmat = cross_matrix(ok, yvar)
    avf = adaptive_vs_fixed(ok, yvar)

    lb.to_csv(args.outdir / "leaderboard.tsv", sep="\t", index=False)
    corr.to_csv(args.outdir / "proxy_correlations.tsv", sep="\t", index=False)
    hits.to_csv(args.outdir / "top1_hit_rate.tsv", sep="\t", index=False)
    xmat.to_csv(args.outdir / "cross_matrix.tsv", sep="\t", index=False)
    avf.to_csv(args.outdir / "adaptive_vs_fixed.tsv", sep="\t", index=False)

    def show(title, frame):
        print(f"── {title} " + "─" * max(0, 60 - len(title)))
        print("(none)" if frame.empty else frame.to_string(index=False))
        print()

    show(f"Leaderboard (top 12 by mean {yvar})", lb.head(12))
    show(f"Proxy → {yvar}: Spearman decomposition", corr)
    show(f"Top-1 hit rate (proxy picks the real {yvar} winner)", hits)
    show(f"Cross-matrix agreement (jaccard vs containment {yvar})", xmat)
    show(f"Adaptive-k vs fixed-k (mean {yvar})", avf)

    print(f"All tables → {args.outdir}/")
    print("\nRead the correlations top-down: trust WITHIN-(matrix,focal) and TOP-1 "
          "HIT RATE.\nA large POOLED rho with weak within-focal rho = the proxy is "
          "only separating\nthe matrices, not ranking configs — do NOT use it to "
          "choose the matrix.")


if __name__ == "__main__":
    main()
