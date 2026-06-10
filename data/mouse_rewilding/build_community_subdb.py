"""
Build a community-specific sub-database for fast gather.

Step 1: Run full gather on a handful of representative samples against GTDB RS226
        to identify all species present in this dataset.
Step 2: Extract those signatures from the full database into a small sub-database.
Step 3: All 183 samples can then be gathered against the sub-database in minutes.

At ~26 min per sample for the full 143k-sig GTDB scan, 3 representative samples
takes ~80 min. The resulting sub-database (~2000-3000 sigs) reduces per-sample
gather time to ~seconds. Total time for 183 samples: ~1 hour vs. ~80 hours.
"""

import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

DB_PATH    = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/gtdb-reps-rs226-k31.dna.zip")
READS_DIR  = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/sourmash_sketch_reads")
MANIFEST   = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/data/mouse_rewilding/sample_manifest.tsv")
OUT_DIR    = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/data/mouse_rewilding/gtdb_gather")
SUBDB_DIR  = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/data/mouse_rewilding/gtdb_subdb")
SUBDB_PATH = SUBDB_DIR / "community_species.zip"
THRESHOLD_BP = 5_000


def select_representative_samples(manifest: pd.DataFrame, n: int = 5) -> list[str]:
    """
    Pick representative samples that together span the dataset's diversity.
    Strategy: one per trial, favouring POST timepoints (more rewilded signal).
    """
    chosen = []
    for trial in manifest["Trial_ID"].unique():
        sub = manifest[manifest["Trial_ID"] == trial]
        # prefer POST; fall back to any
        post = sub[sub["Time_Point"] == "POST"]
        pool = post if len(post) > 0 else sub
        sid = pool.iloc[0]["Sample_ID"]
        chosen.append(sid)
        if len(chosen) >= n:
            break
    return chosen


def run_gather_full_db(sig_path: Path, out_csv: Path) -> int:
    """Run gather against the full GTDB database. Returns match count."""
    if out_csv.exists():
        return sum(1 for _ in open(out_csv)) - 1
    cmd = [
        "sourmash", "gather",
        str(sig_path), str(DB_PATH),
        "-k", "31",
        "--threshold-bp", str(THRESHOLD_BP),
        "-o", str(out_csv),
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_csv.exists():
        print(f"  FAILED: {result.stderr[:200]}", file=sys.stderr)
        return 0
    return sum(1 for _ in open(out_csv)) - 1


def build_subdb(gather_csvs: list[Path]) -> Path:
    """
    Extract all matched species from gather results and build a sub-database.
    Returns path to the sub-database zip.
    """
    SUBDB_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all unique md5 hashes across gather results
    all_md5s = set()
    for csv_path in gather_csvs:
        df = pd.read_csv(csv_path, usecols=["md5"])
        all_md5s.update(df["md5"].str[:8].tolist())  # sourmash uses 8-char prefix

    print(f"  {len(all_md5s)} unique species signatures across all representative samples")

    # Write a picklist file for sourmash extract
    picklist_path = SUBDB_DIR / "community_picklist.csv"
    with open(picklist_path, "w") as f:
        f.write("md5short\n")
        for md5 in sorted(all_md5s):
            f.write(md5 + "\n")

    # Extract matching signatures from the full database
    cmd = [
        "sourmash", "sig", "extract",
        str(DB_PATH),
        "--picklist", f"{picklist_path}:md5short:md5short",
        "-o", str(SUBDB_PATH),
        "--quiet",
    ]
    print(f"  Extracting {len(all_md5s)} signatures from GTDB...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sourmash sig extract failed:\n{result.stderr}")

    size_mb = SUBDB_PATH.stat().st_size / 1e6
    print(f"  Sub-database: {SUBDB_PATH} ({len(all_md5s)} signatures, {size_mb:.0f} MB)")
    return SUBDB_PATH


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, sep="\t")

    # ── Step 1: Full gather on representative samples ────────────────────────
    if SUBDB_PATH.exists():
        print(f"Sub-database already exists: {SUBDB_PATH}", flush=True)
    else:
        reps = select_representative_samples(manifest, n=5)
        print(f"Representative samples for sub-database construction: {reps}", flush=True)

        gather_csvs = []
        t_total = time.time()
        for i, sid in enumerate(reps, 1):
            t0 = time.time()
            sig_path = READS_DIR / f"{sid}.sig"
            out_csv = OUT_DIR / f"{sid}_gather.csv"
            print(f"[{i}/{len(reps)}] Gathering {sid} against full GTDB...", flush=True)
            if not sig_path.exists():
                print(f"  WARNING: sig not found for {sid}, skipping")
                continue
            n = run_gather_full_db(sig_path, out_csv)
            elapsed = time.time() - t0
            print(f"  {n} species in {elapsed/60:.1f} min", flush=True)
            gather_csvs.append(out_csv)

        print(f"\nBuilding community sub-database...", flush=True)
        build_subdb([p for p in gather_csvs if p.exists()])

    # ── Step 2: Gather all samples against sub-database ─────────────────────
    print(f"\nGathering all {len(manifest)} samples against sub-database...", flush=True)
    sample_ids = manifest["Sample_ID"].tolist()
    todo = [
        sid for sid in sample_ids
        if not (OUT_DIR / f"{sid}_gather.csv").exists()
        and (READS_DIR / f"{sid}.sig").exists()
    ]
    already_done = len(sample_ids) - len(todo)
    print(f"{already_done} done, {len(todo)} to go", flush=True)

    t_total = time.time()
    for i, sid in enumerate(todo, 1):
        t0 = time.time()
        sig_path = READS_DIR / f"{sid}.sig"
        out_csv = OUT_DIR / f"{sid}_gather.csv"
        cmd = [
            "sourmash", "gather",
            str(sig_path), str(SUBDB_PATH),
            "-k", "31",
            "--threshold-bp", str(THRESHOLD_BP),
            "-o", str(out_csv),
            "--quiet",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        n = sum(1 for _ in open(out_csv)) - 1 if out_csv.exists() else 0
        elapsed = time.time() - t0
        eta_min = (time.time() - t_total) / i * (len(todo) - i) / 60
        print(f"[{i}/{len(todo)}] {sid}: {n} matches, {elapsed:.0f}s  (ETA {eta_min:.0f} min)", flush=True)

    # ── Step 3: Aggregate ────────────────────────────────────────────────────
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

    prevalence = (wide_df > 0).sum(axis=1)
    print(f"\nTop 10 most prevalent species:")
    for name, prev in prevalence.nlargest(10).items():
        print(f"  {prev:3d}/{len(sample_ids)} samples  {name[:70]}")


if __name__ == "__main__":
    main()
