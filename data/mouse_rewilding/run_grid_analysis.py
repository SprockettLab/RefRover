"""
Run the full Tier-1 grid + analysis on the rewilded-mouse dataset (PLAN.md §9
step 6). Mouse-specific glue around the dataset-agnostic package modules.

Pipeline:
  feature_spaces  →  grid (feature × rule × k → scores + overlap)  →  per-sample
  features  →  analysis (winners, win-rates, advantage, feature→method rule,
  divergence)  →  TSVs + report.md

Feature spaces built here (all the uniform samples × assemblies shape):
  - jaccard      : assembly-vs-assembly Jaccard (from assembly sketches)
  - containment  : reads-vs-assembly containment (containment_wide.tsv)
  - gtdb         : GTDB taxa-abundance bridge. Uses the REAL assembly-side gather
                   (gtdb_assembly_matrix.tsv) once it covers all assemblies;
                   until then falls back to the 1:1 reads proxy and says so.

Everything is restricted to the assemblies that actually have a sketch (so every
matrix is square on the same sample set). Thresholds default to 0.0 (every
reference eligible) — the per-feature-space candidate threshold is open question
§10 Q3; pass --threshold to explore it.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from refrover.analysis import (
    divergence,
    fit_winner_rule,
    method_advantage,
    win_rates,
    winner_feature_contrast,
    winning_method,
)
from refrover.feature_spaces import (
    gtdb_abundance_matrix,
    jaccard_matrix,
    load_gtdb_species_matrix,
)
from refrover.features import feature_table
from refrover.grid import FeatureSpaceSpec, headline, overlap_matrix, run_grid
from refrover.similarity import load_containment_matrix

HERE = Path(__file__).resolve().parent
ASM_SKETCHES = HERE.parent / "sourmash_sketch_assemblies"
CONTAINMENT_WIDE = HERE / "containment_wide.tsv"
GTDB_SPECIES = HERE / "gtdb_species_matrix.tsv"          # reads gather (samples × taxa)
GTDB_ASSEMBLY = HERE / "gtdb_assembly_matrix.tsv"        # assembly gather (assemblies × taxa)

DEFAULT_RULES = ["random", "maxmin", "kmedoids", "greedy_var", "css", "all_vs_all"]


def _dominant_clade(taxa_row: pd.Series) -> str:
    """Genus of a sample/assembly's most-abundant GTDB taxon (a coarse clade)."""
    if taxa_row.max() <= 0:
        return "unknown"
    name = str(taxa_row.idxmax())
    # names look like "GCF_001688725.2 Bacteroides caecimuris" -> genus "Bacteroides"
    m = re.match(r"\S+\s+(\S+)", name)
    return m.group(1) if m else name


def build_feature_spaces(sketch_ids: list[str], threshold: float):
    """
    Build the three feature-space specs and return them with the common sample set.

    The common set is the intersection of samples present in *every* data source
    (assembly sketch, containment matrix both axes, GTDB reads gather) so no
    matrix has an all-NaN row. Assembly sketches include set-aside samples not in
    the manifest/containment/GTDB; they are dropped here. Returns
    ``(specs, sample_taxa, gtdb_note, common)``.
    """
    sample_taxa = load_gtdb_species_matrix(GTDB_SPECIES)          # samples × taxa
    cont_full = load_containment_matrix(CONTAINMENT_WIDE)         # reads × assemblies

    # Candidate samples present across sketch + containment (both axes) + reads gather.
    base = (
        set(sketch_ids)
        & set(cont_full.index) & set(cont_full.columns)
        & set(sample_taxa.index)
    )

    # GTDB real (assembly gather) vs proxy — real only if it covers ~all candidates.
    assembly_taxa = None
    gtdb_note = "proxy (reads gather as 1:1 assembly stand-in)"
    if GTDB_ASSEMBLY.exists():
        at = load_gtdb_species_matrix(GTDB_ASSEMBLY)
        covered = len(base & set(at.index))
        if covered >= int(0.99 * len(base)):
            assembly_taxa = at
            gtdb_note = "real (assembly-side GTDB gather)"
            base &= set(at.index)
        else:
            gtdb_note = f"proxy ({covered}/{len(base)} assemblies gathered so far)"

    common = sorted(base)
    specs = []

    # 1. Jaccard (assembly sketches)
    sigs = {sid: ASM_SKETCHES / f"{sid}.sig" for sid in common}
    jac = jaccard_matrix(sigs).reindex(index=common, columns=common)
    specs.append(FeatureSpaceSpec("jaccard", jac, threshold=threshold))

    # 2. Containment (reads vs assemblies)
    cont = cont_full.reindex(index=common, columns=common)
    specs.append(FeatureSpaceSpec("containment", cont, threshold=threshold))

    # 3. GTDB taxa-abundance bridge
    gtdb = gtdb_abundance_matrix(sample_taxa, assembly_taxa).reindex(
        index=common, columns=common
    )
    taxa_for_clades = assembly_taxa if assembly_taxa is not None else sample_taxa
    clades = pd.Series(
        {sid: _dominant_clade(taxa_for_clades.loc[sid]) for sid in common}
    )
    specs.append(FeatureSpaceSpec("gtdb", gtdb, threshold=threshold, clades=clades))

    return specs, sample_taxa, gtdb_note, common


def _table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a fenced plain-text table (no tabulate dependency)."""
    return "```\n" + df.to_string(index=False) + "\n```"


def write_report(outdir: Path, *, specs, k_values, gtdb_note, scores, overlap,
                 winners, rates, div, contrast, features):
    """Human-readable markdown summary."""
    lines = ["# RefRover Tier-1 grid analysis — rewilded mouse\n"]
    lines.append(f"- Samples: **{scores['sample_id'].nunique()}**")
    lines.append(f"- Feature spaces: {', '.join(s.name for s in specs)}  "
                 f"(gtdb = {gtdb_note})")
    lines.append(f"- Rules: {', '.join(sorted(scores['rule'].unique()))}")
    lines.append(f"- k sweep: {k_values}\n")

    lines.append("## Headline leaderboard (mean tiered_axis_count)\n")
    lines.append(_table(headline(scores).round(4)))

    lines.append("\n## Win rates (best deployable rule per sample)\n")
    lines.append(_table(rates.round(3)))

    lines.append("\n## Feature → winning-method rules (decision tree per fs, k)\n")
    lines.append("CV accuracy is the honest number; a rule only beats chance when "
                 "**cv_accuracy > baseline** (lift > 0). Train-vs-CV gap shows overfitting.\n")
    for fs in sorted(scores["feature_space"].unique()):
        for k in k_values:
            try:
                rule = fit_winner_rule(winners, features, feature_space=fs, k=k, max_depth=3)
            except Exception as exc:  # noqa: BLE001
                lines.append(f"\n### {fs}, k={k}\n_(skipped: {exc})_")
                continue
            cv = (f"{rule['cv_accuracy']:.2f}±{rule['cv_std']:.2f} "
                  f"(lift {rule['lift']:+.2f}, {rule['cv_folds']}-fold)"
                  if rule["cv_accuracy"] is not None else "n/a (class too small)")
            lines.append(
                f"\n### {fs}, k={k}\n"
                f"- CV acc: **{cv}**  | baseline {rule['baseline_accuracy']:.2f} "
                f"| train {rule['train_accuracy']:.2f} | n={rule['n_samples']} "
                f"| winners={rule['classes']}\n"
            )
            lines.append("```\n" + rule["rules"].strip() + "\n```")

    lines.append("\n## Most divergent samples (lowest mean rule agreement)\n")
    lines.append(_table(div.head(10).round(3)))

    (outdir / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=HERE / "grid_analysis")
    ap.add_argument("--k", default="3,5,8,10", help="comma-separated k values")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--rules", default=",".join(DEFAULT_RULES),
                    help="comma-separated rule ids (taxonomy_stratified runs on gtdb only)")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="restrict to the first N samples (quick runs)")
    args = ap.parse_args()

    k_values = [int(x) for x in args.k.split(",")]
    rules = args.rules.split(",")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Candidate assemblies = those with a sketch; build_feature_spaces narrows this
    # to the intersection present across all data sources.
    sketch_ids = sorted(p.stem for p in ASM_SKETCHES.glob("*.sig"))
    if args.n_samples:
        sketch_ids = sketch_ids[: args.n_samples]
    print(f"{len(sketch_ids)} assembly sketches", flush=True)

    print("Building feature spaces ...", flush=True)
    specs, sample_taxa, gtdb_note, common = build_feature_spaces(sketch_ids, args.threshold)
    print(f"  common sample set: {len(common)} samples  | gtdb: {gtdb_note}", flush=True)

    print("Running grid (feature × rule × k) ...", flush=True)
    result = run_grid(specs, rules, k_values, query_ids=common)
    result.scores.to_csv(args.outdir / "grid_scores.tsv", sep="\t", index=False)
    result.overlap.to_csv(args.outdir / "grid_overlap.tsv", sep="\t", index=False)

    print("Computing per-sample features ...", flush=True)
    abundance = sample_taxa.reindex(index=common)
    # Features from the containment space (the no-DB middle tier) as the matrix view.
    cont_spec = next(s for s in specs if s.name == "containment")
    features = feature_table(cont_spec.matrix, abundance=abundance,
                             threshold=args.threshold, k_max=max(k_values),
                             query_ids=common)
    features.to_csv(args.outdir / "per_sample_features.tsv", sep="\t")

    print("Analysis ...", flush=True)
    winners = winning_method(result.scores)
    rates = win_rates(winners)
    adv = method_advantage(result.scores)
    div = divergence(result.overlap)
    contrast = winner_feature_contrast(winners, features)

    winners.to_csv(args.outdir / "winners.tsv", sep="\t", index=False)
    rates.to_csv(args.outdir / "win_rates.tsv", sep="\t", index=False)
    adv.to_csv(args.outdir / "method_advantage.tsv", sep="\t", index=False)
    div.to_csv(args.outdir / "divergence.tsv", sep="\t", index=False)
    contrast.to_csv(args.outdir / "winner_feature_contrast.tsv", sep="\t", index=False)
    for fs in sorted(result.scores["feature_space"].unique()):
        for k in k_values:
            overlap_matrix(result.overlap, fs, k).to_csv(
                args.outdir / f"overlap_matrix_{fs}_k{k}.tsv", sep="\t"
            )

    write_report(args.outdir, specs=specs, k_values=k_values, gtdb_note=gtdb_note,
                 scores=result.scores, overlap=result.overlap, winners=winners,
                 rates=rates, div=div, contrast=contrast, features=features)
    print(f"Done. Wrote TSVs + report.md to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
