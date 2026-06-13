"""
Run sourmash gather for all *assemblies* against GTDB — the assembly side of the
PLAN.md §3.1 taxa-routing bridge.

Mirror image of run_gtdb_gather.py (which gathers *reads*). That script answers
"how abundant is each GTDB taxon in sample i's reads?" (samples × taxa). This one
answers "what GTDB taxa is assembly j made of?" (assemblies × taxa) — the
`assembly_j -> its GTDB taxa` arrow in §3.1. Feed both matrices to
`refrover.feature_spaces.gtdb_abundance_matrix(sample_taxa, assembly_taxa)`.

Speed: by default this gathers against the community sub-database
(gtdb_subdb/community_species.zip, ~2k sigs) that build_community_subdb.py
already distilled from the reads. Every taxon in an assembly was present in that
sample's reads, so the sub-db is a complete catalogue for assemblies too — and it
runs in ~seconds/sample (vs ~26 min against the full 143k-sig GTDB). Pass
--full-db to gather against the full database instead.

Output: one CSV per assembly in assembly_gather/, plus an aggregated
taxa × assembly matrix (gtdb_assembly_matrix.tsv) in the *same* taxa-major layout
as gtdb_species_matrix.tsv — so feature_spaces.load_gtdb_species_matrix() loads
either one (it transposes to the row-major shape the bridge expects).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT       = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover")
FULL_DB    = ROOT / "gtdb-reps-rs226-k31.dna.zip"
SUBDB      = ROOT / "data/mouse_rewilding/gtdb_subdb/community_species.zip"
ASM_DIR    = ROOT / "data/sourmash_sketch_assemblies"
MANIFEST   = ROOT / "data/mouse_rewilding/sample_manifest.tsv"
OUT_DIR    = ROOT / "data/mouse_rewilding/assembly_gather"
OUT_MATRIX = ROOT / "data/mouse_rewilding/gtdb_assembly_matrix.tsv"
THRESHOLD_BP = 5_000


def run_gather_cli(sig_path: Path, out_csv: Path, db_path: Path) -> int:
    """Run sourmash gather via CLI. Returns number of matches (header excluded)."""
    cmd = [
        "sourmash", "gather",
        str(sig_path),
        str(db_path),
        "-k", "31",
        "--threshold-bp", str(THRESHOLD_BP),
        "-o", str(out_csv),
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:200]}", file=sys.stderr)
        return 0
    if not out_csv.exists():
        # No matches: write a header-only CSV so the sample counts as "done".
        out_csv.write_text(
            "intersect_bp,f_orig_query,f_match,f_unique_to_query,f_unique_weighted,"
            "average_abund,median_abund,std_abund,filename,name,md5,f_match_orig,"
            "unique_intersect_bp,gather_result_rank,remaining_bp,query_filename,"
            "query_name,query_md5,query_bp,ksize,moltype,scaled,query_n_hashes,"
            "query_abundance,query_containment_ani,match_containment_ani,"
            "average_containment_ani,max_containment_ani,potential_false_negative,"
            "n_unique_weighted_found,sum_weighted_found,total_weighted_hashes\n"
        )
        return 0
    return sum(1 for _ in open(out_csv)) - 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Process only the first N assemblies (testing; default: all)")
    parser.add_argument("--full-db", action="store_true",
                        help="Gather against the full GTDB (slow) instead of the community sub-db")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip gather, just re-aggregate existing CSVs (over the full sample list)")
    parser.add_argument("--shard", metavar="i/N", default=None,
                        help="HPC array mode: gather only shard i of N (1-based; samples "
                             "assigned round-robin). Each shard gathers a disjoint subset and "
                             "writes only its own CSVs — aggregation is SKIPPED. Run once with "
                             "--aggregate-only (no --shard) after all shards finish.")
    args = parser.parse_args()

    db_path = FULL_DB if args.full_db else SUBDB
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")

    shard_i = shard_n = None
    if args.shard:
        try:
            shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            sys.exit(f"--shard must look like i/N (e.g. 3/20), got '{args.shard}'")
        if not (1 <= shard_i <= shard_n):
            sys.exit(f"--shard i/N requires 1 <= i <= N, got {shard_i}/{shard_n}")
        if args.aggregate_only:
            sys.exit("--shard and --aggregate-only are mutually exclusive "
                     "(aggregation must see all shards' CSVs; run it without --shard).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, sep="\t")
    all_sample_ids = manifest["Sample_ID"].tolist()

    if args.n_samples:
        all_sample_ids = all_sample_ids[: args.n_samples]
        print(f"Processing {args.n_samples} assemblies (subset mode)", flush=True)

    # The gather pass works on this shard's slice; aggregation always uses the full list.
    if shard_i is not None:
        sample_ids = all_sample_ids[shard_i - 1 :: shard_n]
        print(f"Shard {shard_i}/{shard_n}: {len(sample_ids)} of {len(all_sample_ids)} "
              "assemblies assigned to this task", flush=True)
    else:
        sample_ids = all_sample_ids

    # Only assemblies that actually have a sketch (some samples fail to assemble).
    todo = [
        sid for sid in sample_ids
        if not (OUT_DIR / f"{sid}_gather.csv").exists()
        and (ASM_DIR / f"{sid}.sig").exists()
    ]
    missing_sketch = [sid for sid in sample_ids if not (ASM_DIR / f"{sid}.sig").exists()]
    already_done = len([sid for sid in sample_ids if (OUT_DIR / f"{sid}_gather.csv").exists()])
    print(f"DB: {db_path.name}", flush=True)
    print(f"{already_done} already done, {len(todo)} to process, "
          f"{len(missing_sketch)} without an assembly sketch (skipped)", flush=True)
    if missing_sketch:
        print(f"  no-sketch: {', '.join(missing_sketch[:5])}"
              f"{' ...' if len(missing_sketch) > 5 else ''}", flush=True)

    if todo and not args.aggregate_only:
        t_total = time.time()
        for i, sid in enumerate(todo, 1):
            t0 = time.time()
            print(f"[{i}/{len(todo)}] {sid}...", end=" ", flush=True)
            n_matches = run_gather_cli(
                ASM_DIR / f"{sid}.sig", OUT_DIR / f"{sid}_gather.csv", db_path
            )
            elapsed = time.time() - t0
            eta_min = (time.time() - t_total) / i * (len(todo) - i) / 60
            print(f"{n_matches} matches, {elapsed:.1f}s  (ETA {eta_min:.0f} min)", flush=True)

    # ── Aggregate into a taxa × assembly matrix (taxa-major, like the reads one) ──
    # A shard must never write a partial matrix: it only produces CSVs. Aggregation
    # runs once over the FULL sample list (shard is None / --aggregate-only).
    if shard_i is not None:
        print(f"\nShard {shard_i}/{shard_n} done. Aggregation skipped — run "
              "`python run_assembly_gather.py --aggregate-only` once all shards finish.",
              flush=True)
        return

    print("\nAggregating gather results...", flush=True)
    records = []
    for sid in all_sample_ids:
        csv_path = OUT_DIR / f"{sid}_gather.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, usecols=["name", "f_unique_to_query"])
        if df.empty:
            continue
        df["assembly_id"] = sid
        records.append(df[["assembly_id", "name", "f_unique_to_query"]])

    if not records:
        print("No gather results found.")
        return

    long_df = pd.concat(records, ignore_index=True)
    asm_ids = [s for s in all_sample_ids if (OUT_DIR / f"{s}_gather.csv").exists()]
    wide_df = long_df.pivot_table(
        index="name", columns="assembly_id", values="f_unique_to_query", fill_value=0
    ).reindex(columns=asm_ids, fill_value=0)

    wide_df.to_csv(OUT_MATRIX, sep="\t")
    print(f"Assembly matrix: {wide_df.shape[0]:,} taxa x {wide_df.shape[1]} assemblies "
          f"-> {OUT_MATRIX}", flush=True)
    print("Load it with feature_spaces.load_gtdb_species_matrix() as `assembly_taxa`.",
          flush=True)


if __name__ == "__main__":
    main()
