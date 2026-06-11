"""
Compute per-assembly species-level weights for ContainmentSelector.

After running explore_species.R (which produces gtdb_species_stats.tsv),
this script estimates how much differential coverage signal each assembly
would provide across samples.

Weight derivation
-----------------
We don't have GTDB gather results for assemblies (only for reads). Instead
we use the cross-sample containment matrix as a proxy:

  weight_j = CV of containment(reads_i, assembly_j) across all samples i,
             restricted to assemblies above a mean-containment floor.

This captures: assemblies whose reads-mapping rate varies strongly across
samples are the most informative for differential coverage binning.

The three-regime filter (from the R analysis) can optionally restrict to
"sweet-spot" species — mid-abundance, high-CV — and use the fraction of each
assembly's content coming from those species as an additional weight factor.

Outputs
-------
  assembly_weights.tsv : sample_id, weight, cv_containment, mean_containment
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
CONTAINMENT_WIDE = DATA_DIR / "containment_wide.tsv"
SPECIES_STATS    = DATA_DIR / "gtdb_species_stats.tsv"
SPECIES_MATRIX   = DATA_DIR / "gtdb_species_matrix.tsv"
OUT_WEIGHTS      = DATA_DIR / "assembly_weights.tsv"


def containment_based_weights(
    containment_wide: pd.DataFrame,
    min_mean_containment: float = 0.005,
) -> pd.Series:
    """
    Per-assembly weight = CV of containment across all samples.

    Assemblies that are consistently high (stable core) or consistently
    low (rare / absent everywhere) get low weight. Assemblies with high
    variance in how well reads map to them are the most informative.
    """
    mean_cont = containment_wide.mean(axis=0)
    std_cont  = containment_wide.std(axis=0)
    cv_cont   = std_cont / (mean_cont + 1e-9)

    # Exclude assemblies too rarely detected to carry differential signal
    eligible = mean_cont >= min_mean_containment
    weights = cv_cont.where(eligible, other=0.0)
    return weights


def species_regime_weights(
    species_stats: pd.DataFrame,
    species_matrix: pd.DataFrame,
    containment_wide: pd.DataFrame,
    low_pct: float = 33.0,
    high_pct: float = 67.0,
) -> pd.Series:
    """
    Weight assemblies by the fraction of their content from 'sweet-spot' species:
    mid-abundance, high-CV taxa that vary most across samples.

    Because we don't have assembly-level GTDB gather, we approximate the
    species composition of each assembly by: for each sample (assembly),
    use that sample's own GTDB gather result to identify its dominant species.

    weight_j = sum over species s detected in sample j of:
               CV(s) × f_unique_to_query(s, j)
               ...restricted to mid-abundance species
    """
    if species_stats is None or species_matrix is None:
        print("  Species stats not available; using containment-only weights", file=sys.stderr)
        return pd.Series(dtype=float)

    # Identify mid-abundance species (the "sweet spot")
    log_mean = np.log10(species_stats["mean_abund"] + 1e-9)
    lo = np.percentile(log_mean, low_pct)
    hi = np.percentile(log_mean, high_pct)
    sweetspot = species_stats[
        (log_mean >= lo) & (log_mean <= hi)
    ].set_index("name")

    if sweetspot.empty:
        return pd.Series(dtype=float)

    # For each assembly (column in species_matrix), compute weighted CV score
    assembly_ids = containment_wide.columns.tolist()
    sample_ids   = species_matrix.columns.tolist()

    weights = {}
    for asm_id in assembly_ids:
        if asm_id not in sample_ids:
            weights[asm_id] = 0.0
            continue

        # Species abundances in this sample
        abund = species_matrix.loc[
            species_matrix.index.intersection(sweetspot.index), asm_id
        ]
        if abund.empty or abund.sum() == 0:
            weights[asm_id] = 0.0
            continue

        # Weight = sum of (abundance × CV) for sweet-spot species
        cv_vals = sweetspot.loc[abund.index, "cv"]
        weights[asm_id] = float((abund * cv_vals).sum())

    return pd.Series(weights, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-species", action="store_true",
                        help="Incorporate species-level sweet-spot weighting (requires gtdb outputs)")
    parser.add_argument("--min-mean-containment", type=float, default=0.005)
    parser.add_argument("--low-pct", type=float, default=33.0,
                        help="Lower percentile for mid-abundance species window")
    parser.add_argument("--high-pct", type=float, default=67.0,
                        help="Upper percentile for mid-abundance species window")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if not CONTAINMENT_WIDE.exists():
        sys.exit(f"ERROR: {CONTAINMENT_WIDE} not found. Run compute_containment.py first.")

    print(f"Loading containment matrix...", flush=True)
    cont = pd.read_csv(CONTAINMENT_WIDE, sep="\t", index_col=0)
    print(f"  {cont.shape[0]} samples x {cont.shape[1]} assemblies")

    # ── Compute weights ───────────────────────────────────────────────────────
    print("Computing containment-based weights...", flush=True)
    w_cont = containment_based_weights(cont, args.min_mean_containment)

    if args.use_species and SPECIES_STATS.exists() and SPECIES_MATRIX.exists():
        print("Computing species-regime weights...", flush=True)
        sp_stats  = pd.read_csv(SPECIES_STATS, sep="\t")
        sp_matrix = pd.read_csv(SPECIES_MATRIX, sep="\t", index_col=0)
        w_species = species_regime_weights(
            sp_stats, sp_matrix, cont,
            low_pct=args.low_pct, high_pct=args.high_pct,
        )
        # Normalise each weight vector to [0, 1] then take geometric mean
        def _norm(s):
            r = s - s.min()
            return r / (r.max() + 1e-9)
        w_combined = (_norm(w_cont) * _norm(w_species.reindex(w_cont.index, fill_value=0))) ** 0.5
        label = "combined (containment × species)"
    else:
        w_combined = w_cont
        label = "containment CV only"

    # ── Summary and output ────────────────────────────────────────────────────
    mean_cont = cont.mean(axis=0)
    std_cont  = cont.std(axis=0)
    cv_cont   = std_cont / (mean_cont + 1e-9)

    result = pd.DataFrame({
        "sample_id":        w_combined.index,
        "weight":           w_combined.values,
        "cv_containment":   cv_cont.reindex(w_combined.index).values,
        "mean_containment": mean_cont.reindex(w_combined.index).values,
    }).sort_values("weight", ascending=False)

    result.to_csv(OUT_WEIGHTS, sep="\t", index=False)
    print(f"\nWeights ({label}) -> {OUT_WEIGHTS}")

    print(f"\nTop 10 highest-weight assemblies:")
    print(result.head(10).to_string(index=False))

    print(f"\nBottom 10 (lowest signal):")
    print(result.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
