"""
Run sourmash gather for all samples against the GTDB RS226 database.

Uses the sourmash CLI (subprocess) per sample — the CLI's Rust backend
is ~3-5x faster than the Python API for the linear prefetch scan.

At ~26 min/sample on an M-chip laptop, 183 samples = ~80 hours.
Practical options:
  - Run on a subset with --n-samples N (e.g. 20 = ~9 hours)
  - Run on a compute server where the linear scan is faster
  - Resume-capable: already-done samples are skipped automatically

Output: one CSV per sample in gtdb_gather/, plus an aggregated
species x sample matrix (gtdb_species_matrix.tsv).
"""

import csv
import subprocess
import sys
import time
import argparse
from pathlib import Path
import pandas as pd

DB_PATH    = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/gtdb-reps-rs226-k31.dna.zip")
READS_DIR  = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/sourmash_sketch_reads")
MANIFEST   = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/data/mouse_rewilding/sample_manifest.tsv")
OUT_DIR    = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/data/mouse_rewilding/gtdb_gather")
THRESHOLD_BP = 5_000


def run_gather_cli(sig_path: Path, out_csv: Path) -> int:
    """Run sourmash gather via CLI. Returns number of matches."""
    cmd = [
        "sourmash", "gather",
        str(sig_path),
        str(DB_PATH),
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
        out_csv.write_text("intersect_bp,f_orig_query,f_match,f_unique_to_query,f_unique_weighted,average_abund,median_abund,std_abund,filename,name,md5,f_match_orig,unique_intersect_bp,gather_result_rank,remaining_bp,query_filename,query_name,query_md5,query_bp,ksize,moltype,scaled,query_n_hashes,query_abundance,query_containment_ani,match_containment_ani,average_containment_ani,max_containment_ani,potential_false_negative,n_unique_weighted_found,sum_weighted_found,total_weighted_hashes\n")
        return 0
    return sum(1 for _ in open(out_csv)) - 1  # lines minus header


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Process only the first N samples (for testing; default: all)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip gather, just re-aggregate existing CSVs")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, sep="\t")
    sample_ids = manifest["Sample_ID"].tolist()

    if args.n_samples:
        sample_ids = sample_ids[:args.n_samples]
        print(f"Processing {args.n_samples} samples (subset mode)", flush=True)

    todo = [
        sid for sid in sample_ids
        if not (OUT_DIR / f"{sid}_gather.csv").exists()
        and (READS_DIR / f"{sid}.sig").exists()
    ]
    already_done = len([sid for sid in sample_ids if (OUT_DIR / f"{sid}_gather.csv").exists()])
    print(f"{already_done} already done, {len(todo)} to process", flush=True)

    if todo and not args.aggregate_only:
        # Estimate total time
        if already_done == 0:
            print(f"  Note: first sample took ~26 min on M-chip laptop. "
                  f"ETA for {len(todo)} samples: ~{len(todo)*26/60:.0f} hours", flush=True)
        t_total = time.time()
        for i, sid in enumerate(todo, 1):
            t0 = time.time()
            print(f"[{i}/{len(todo)}] {sid}...", end=" ", flush=True)

            sig_path = READS_DIR / f"{sid}.sig"
            out_csv = OUT_DIR / f"{sid}_gather.csv"
            n_matches = run_gather_cli(sig_path, out_csv)

            elapsed = time.time() - t0
            done_so_far = already_done + i
            eta_min = (time.time() - t_total) / i * (len(todo) - i) / 60
            print(f"{n_matches} matches, {elapsed/60:.1f} min  (ETA {eta_min:.0f} min)", flush=True)

    # ── Aggregate into species × sample matrix ───────────────────────────────
    print("\nAggregating gather results...", flush=True)
    records = []
    for sid in sample_ids:
        csv_path = OUT_DIR / f"{sid}_gather.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, usecols=["name", "f_unique_to_query"])
        if df.empty:
            continue
        df["sample_id"] = sid
        records.append(df[["sample_id", "name", "f_unique_to_query"]])

    if not records:
        print("No gather results found.")
        return

    long_df = pd.concat(records, ignore_index=True)

    # Pivot to wide: rows = species, columns = samples
    wide_df = long_df.pivot_table(
        index="name", columns="sample_id", values="f_unique_to_query", fill_value=0
    )
    wide_df = wide_df.reindex(columns=sample_ids, fill_value=0)

    out_wide = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/data/mouse_rewilding/gtdb_species_matrix.tsv")
    wide_df.to_csv(out_wide, sep="\t")
    print(f"Species matrix: {wide_df.shape[0]:,} species x {wide_df.shape[1]} samples -> {out_wide}", flush=True)

    out_long = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/data/mouse_rewilding/gtdb_gather_long.tsv")
    long_df.to_csv(out_long, sep="\t", index=False)
    print(f"Long format: {len(long_df):,} rows -> {out_long}", flush=True)

    # Quick summary
    n_species = wide_df.shape[0]
    prevalence = (wide_df > 0).sum(axis=1)
    print(f"\nTop 10 most prevalent species:")
    for name, prev in prevalence.nlargest(10).items():
        print(f"  {prev:3d}/{len(sample_ids)} samples  {name[:70]}")


if __name__ == "__main__":
    main()
