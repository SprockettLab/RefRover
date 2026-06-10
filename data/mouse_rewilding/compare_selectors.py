"""
Compare prototype selector strategies on the cross-sample containment matrix.

Selectors evaluated
-------------------
  random          Baseline: k assemblies drawn uniformly at random from candidates
  top_containment Sample-centric: k assemblies with highest containment(query, assembly)
  top_var         Variance-centric: k assemblies with highest cross-sample containment variance
  maxmin          Diversity-centric: greedy MaxMin on containment-profile Pearson distances

Metric: fraction of total cross-sample containment variance "explained" by selected
prototypes, measured as sum(var(selected columns)) / sum(var(all columns)).
Note: this sums individual variances without orthogonalising, so correlated selections
overcount shared variance. It is a conservative upper bound; correlated selectors will
look better than they are in practice.

Adaptive k
----------
For each sample, runs the MaxMin greedy selector while tracking the marginal
*residual* variance gain at each step (variance of the new column after regressing
out variance already explained by selected columns). The elbow of this gain curve
is the per-sample optimal k — the point of diminishing returns.

Outputs
-------
  selector_comparison.tsv       long-format (sample, selector, k, variance_explained)
  selector_agreement_k5.tsv     pairwise mean Jaccard overlap of selections at k=5
  adaptive_k_ground_truth.tsv   per-sample optimal k + metadata
  saturation_curves.tsv         per-sample per-step cumulative and marginal variance
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
MIN_CONTAINMENT = 0.02
K_VALUES = [3, 5, 8, 10]
MAX_K = 20

# ── Helpers ───────────────────────────────────────────────────────────────────

def residual_var(y: np.ndarray, X_cols: list[np.ndarray]) -> float:
    """Variance of y unexplained by the columns in X_cols (OLS residual)."""
    if not X_cols:
        return float(np.var(y, ddof=1))
    X = np.column_stack(X_cols)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.var(y - X @ coef, ddof=1))


def maxmin_greedy(dist_sub: np.ndarray, seed: int, k: int) -> list[int]:
    """
    Greedy MaxMin on a precomputed distance matrix (local indices).
    Returns list of local indices of length min(k, n).
    """
    n = dist_sub.shape[0]
    k = min(k, n)
    selected = [seed]
    remaining = [i for i in range(n) if i != seed]
    if not remaining:
        return selected
    min_dist = dist_sub[seed, remaining].copy()
    for _ in range(k - 1):
        if not remaining:
            break
        best_local = int(np.argmax(min_dist))
        best = remaining[best_local]
        selected.append(best)
        remaining.pop(best_local)
        min_dist = np.delete(min_dist, best_local)
        if remaining:
            min_dist = np.minimum(min_dist, dist_sub[best, remaining])
    return selected


# ── Selectors ─────────────────────────────────────────────────────────────────

def select_random(k, nc, rng):
    return list(rng.choice(nc, size=min(k, nc), replace=False))

def select_top_containment(query_sub, k):
    return list(np.argsort(query_sub)[::-1][:k])

def select_top_var(col_vars_sub, k):
    return list(np.argsort(col_vars_sub)[::-1][:k])

def select_maxmin(dist_sub, query_sub, k):
    seed = int(np.argmax(query_sub))  # highest containment = most likely to map
    return maxmin_greedy(dist_sub, seed, k)


# ── Saturation curve ──────────────────────────────────────────────────────────

def saturation_curve(cont_sub: np.ndarray, col_vars_sub: np.ndarray,
                     total_var: float, max_k: int = MAX_K):
    """
    Greedy forward selection by residual variance (NOT MaxMin order).

    At each step, add the assembly that contributes the most unique cross-sample
    variance not already explained by the selected set. This guarantees
    monotonically decreasing marginal gains — a prerequisite for elbow detection.

    Uses incremental Gram-Schmidt so cost is O(n_samp × n_cand × max_k).

    Returns
    -------
    gains : ndarray, length <= max_k
        gains[i] = marginal residual variance of the (i+1)-th selected assembly,
        normalised by total_var.
    cumvars : ndarray, same length
        Cumulative sum of gains.
    selected_order : list of local column indices in selection order.
    """
    n_samp, n_cand = cont_sub.shape

    # Mean-centre each column (needed for correct Pearson-style projection)
    cols = cont_sub - cont_sub.mean(axis=0)

    # Seed: highest individual variance
    first = int(np.argmax(col_vars_sub))
    selected = [first]
    remaining = list(range(n_cand))
    remaining.remove(first)

    # Initialise orthonormal basis Q for the column space of selected assemblies
    v0 = cols[:, first]
    norm0 = np.linalg.norm(v0)
    Q = (v0 / norm0).reshape(-1, 1) if norm0 > 1e-12 else np.zeros((n_samp, 1))

    gains = [col_vars_sub[first] / total_var]
    cumvars = [gains[0]]

    limit = min(max_k - 1, len(remaining))
    for _ in range(limit):
        if not remaining:
            break
        # Compute residuals for all remaining columns at once (vectorised)
        Y = cols[:, remaining]          # n_samp × n_remaining
        proj = Q @ (Q.T @ Y)            # n_samp × n_remaining
        resids = Y - proj               # n_samp × n_remaining
        resid_vars = np.var(resids, axis=0, ddof=1)

        best_local = int(np.argmax(resid_vars))
        best = remaining[best_local]
        gain = resid_vars[best_local] / total_var
        gains.append(gain)
        cumvars.append(cumvars[-1] + gain)

        # Extend Q with the new column (Gram-Schmidt)
        v = cols[:, best] - Q @ (Q.T @ cols[:, best])
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            Q = np.column_stack([Q, v / norm])

        selected.append(best)
        remaining.pop(best_local)

    return np.array(gains), np.array(cumvars), selected


def elbow_from_gains(gains: np.ndarray, saturation_fraction: float = 0.1,
                     k_min: int = 1) -> int:
    """
    Threshold-based elbow: k = first step where marginal gain < fraction × first_gain.

    Works correctly on a monotonically decreasing gains curve.  The largest-drop
    heuristic fails when MaxMin selection order mixes high- and low-variance steps.
    """
    if len(gains) < 2 or gains[0] == 0:
        return max(k_min, 1)
    threshold = saturation_fraction * gains[0]
    idxs = np.where(gains < threshold)[0]
    if len(idxs) == 0:
        return max(k_min, len(gains))
    return max(k_min, int(idxs[0]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...", flush=True)
    cont = pd.read_csv(DATA_DIR / "containment_wide.tsv", sep="\t", index_col=0)
    manifest = pd.read_csv(DATA_DIR / "sample_manifest.tsv", sep="\t")
    samples = cont.index.tolist()
    assemblies = cont.columns.tolist()
    print(f"  {len(samples)} samples × {len(assemblies)} assemblies")

    col_vars = cont.var(axis=0)
    total_var = float(col_vars.sum())
    asm_idx = {a: i for i, a in enumerate(assemblies)}

    print("Pre-computing containment-profile distance matrix...", flush=True)
    corr_mat = cont.corr().values.astype(float)  # assemblies × assemblies
    np.fill_diagonal(corr_mat, 1.0)
    dist_mat = np.clip(1.0 - corr_mat, 0.0, 2.0)
    np.fill_diagonal(dist_mat, 0.0)

    # ── Saturation curves + adaptive k ────────────────────────────────────────
    print("Computing saturation curves per sample...", flush=True)
    ak_rows = []
    curve_rows = []

    for i, sid in enumerate(samples, 1):
        row = cont.loc[sid]
        cands = row[row >= MIN_CONTAINMENT].index.tolist()
        if len(cands) < 2:
            continue

        cidx = np.array([asm_idx[c] for c in cands])
        cont_sub = cont.values[:, cidx]
        col_vars_sub = col_vars.values[cidx]

        gains, cumvars, sel_order = saturation_curve(cont_sub, col_vars_sub, total_var)
        opt_k = elbow_from_gains(gains)

        ak_rows.append({
            "sample_id": sid,
            "optimal_k": opt_k,
            "n_candidates": len(cands),
        })
        for step, (g, cv) in enumerate(zip(gains, cumvars), start=1):
            curve_rows.append({
                "sample_id": sid,
                "k": step,
                "marginal_var": g,
                "cumulative_var": cv,
            })

        if i % 30 == 0:
            print(f"  {i}/{len(samples)}", flush=True)

    ak_df = pd.DataFrame(ak_rows).merge(
        manifest[["Sample_ID", "Mouse_Strain", "Time_Point", "Trial_ID"]],
        left_on="sample_id", right_on="Sample_ID", how="left"
    )
    ak_df.to_csv(DATA_DIR / "adaptive_k_ground_truth.tsv", sep="\t", index=False)

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(DATA_DIR / "saturation_curves.tsv", sep="\t", index=False)

    print(f"  Adaptive k: median={ak_df['optimal_k'].median():.0f}  "
          f"mean={ak_df['optimal_k'].mean():.1f}  "
          f"range=[{ak_df['optimal_k'].min()}, {ak_df['optimal_k'].max()}]")

    # ── Selector comparison ────────────────────────────────────────────────────
    print("\nComputing variance explained per selector × k × sample...", flush=True)
    SELECTORS = ["random", "top_containment", "top_var", "maxmin"]
    comp_rows = []
    agree_sets: dict[tuple, dict] = {(s, k): {} for s in SELECTORS for k in K_VALUES}

    for i, sid in enumerate(samples, 1):
        row = cont.loc[sid]
        cands = row[row >= MIN_CONTAINMENT].index.tolist()
        if not cands:
            continue
        nc = len(cands)
        cidx = np.array([asm_idx[c] for c in cands])
        query_sub = row.values[cidx]
        cv_sub = col_vars.values[cidx]
        dist_sub = dist_mat[np.ix_(cidx, cidx)]

        for k in K_VALUES:
            for sel in SELECTORS:
                if sel == "random":
                    rng_i = np.random.default_rng(42 + abs(hash(sid)) % 10**6)
                    local_idx = select_random(k, nc, rng_i)
                elif sel == "top_containment":
                    local_idx = select_top_containment(query_sub, k)
                elif sel == "top_var":
                    local_idx = select_top_var(cv_sub, k)
                else:  # maxmin
                    local_idx = select_maxmin(dist_sub, query_sub, k)

                sel_names = [cands[j] for j in local_idx]

                # Simple metric: sum of individual column variances (overcounts correlated cols)
                ve = col_vars[sel_names].sum() / total_var if total_var > 0 else 0.0

                # Unique metric: fraction of ALL 182 column variances explained via
                # orthogonal projection onto the selected columns (no double-counting).
                # Computation: Q from QR of selected columns; project full matrix onto Q;
                # captured = (total_var - sum residual variances) / total_var
                sel_global = [asm_idx[n] for n in sel_names]
                X = cont.values - cont.values.mean(axis=0)  # centre all cols
                Xp = X[:, sel_global]                       # n_samp × k
                if Xp.shape[1] > 0:
                    Q, _ = np.linalg.qr(Xp, mode="reduced")
                    proj = Q @ (Q.T @ X)           # n_samp × n_assemblies
                    resid_all = X - proj
                    resid_vars = np.var(resid_all, axis=0, ddof=1).sum()
                    uve = (total_var - resid_vars) / total_var if total_var > 0 else 0.0
                else:
                    uve = 0.0

                comp_rows.append({
                    "sample_id": sid, "selector": sel, "k": k,
                    "variance_explained": ve,
                    "unique_var_explained": uve,
                    "n_selected": len(sel_names),
                    "n_candidates": nc,
                })
                agree_sets[(sel, k)][sid] = set(sel_names)

        if i % 30 == 0:
            print(f"  {i}/{len(samples)}", flush=True)

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(DATA_DIR / "selector_comparison.tsv", sep="\t", index=False)

    summary = (comp_df.groupby(["selector", "k"])[["variance_explained", "unique_var_explained"]]
               .mean().round(4))
    print("\nMean variance explained by selector and k:")
    print(summary.to_string())

    # ── Pairwise agreement at k=5 ──────────────────────────────────────────────
    print("\nComputing pairwise selector agreement (Jaccard) at k=5...", flush=True)
    agree_df = pd.DataFrame(index=SELECTORS, columns=SELECTORS, dtype=float)
    for s1 in SELECTORS:
        for s2 in SELECTORS:
            jacs = []
            for sid in samples:
                a = agree_sets.get((s1, 5), {}).get(sid)
                b = agree_sets.get((s2, 5), {}).get(sid)
                if a is None or b is None or not (a | b):
                    continue
                jacs.append(len(a & b) / len(a | b))
            agree_df.loc[s1, s2] = round(np.mean(jacs), 3) if jacs else 0.0

    agree_df.to_csv(DATA_DIR / "selector_agreement_k5.tsv", sep="\t")
    print("Selector agreement (mean Jaccard at k=5):")
    print(agree_df.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
