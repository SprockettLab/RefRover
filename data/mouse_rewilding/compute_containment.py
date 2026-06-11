"""
Compute cross-sample containment matrix from sourmash .sig files.

For each query (read sketch), computes the fraction of its k-mer hashes
present in each reference (assembly sketch): containment(reads_i, asm_j).

Outputs a long-format TSV: query_id, reference_id, containment
"""

import json
import os
import glob
import time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

READ_DIR  = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/sourmash_sketch_reads")
ASM_DIR   = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/sourmash_sketch_assemblies")
MANIFEST  = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/data/mouse_rewilding/sample_manifest.tsv")
OUT_LONG  = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/data/mouse_rewilding/containment_long.tsv")
OUT_WIDE  = Path("/Users/danielsprockett/Documents/Projects/WF22_RefRover/RefRover/data/mouse_rewilding/containment_wide.tsv")
N_WORKERS = max(1, cpu_count() - 2)


def load_sig(path: Path) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    mins = data[0]["signatures"][0]["mins"]
    arr = np.array(mins, dtype=np.uint64)
    arr.sort()
    return arr


def load_all_assemblies(sample_ids: list[str]) -> dict[str, np.ndarray]:
    asms = {}
    for sid in sample_ids:
        p = ASM_DIR / f"{sid}.sig"
        if p.exists():
            asms[sid] = load_sig(p)
    print(f"Loaded {len(asms)} assembly sketches", flush=True)
    return asms


# Module-level global so forked workers share without pickling
_ASM_HASHES: dict[str, np.ndarray] = {}
_ASM_IDS: list[str] = []


def _init_worker(asm_hashes, asm_ids):
    global _ASM_HASHES, _ASM_IDS
    _ASM_HASHES = asm_hashes
    _ASM_IDS = asm_ids


def _compute_row(query_id: str) -> list[tuple[str, str, float]] | None:
    read_sig = READ_DIR / f"{query_id}.sig"
    if not read_sig.exists():
        print(f"  no read sig: {query_id}", flush=True)
        return None

    read_h = load_sig(read_sig)
    n_read = len(read_h)
    if n_read == 0:
        return None

    rows = []
    for ref_id in _ASM_IDS:
        asm_h = _ASM_HASHES[ref_id]
        n_intersect = len(np.intersect1d(read_h, asm_h, assume_unique=True))
        rows.append((query_id, ref_id, n_intersect / n_read))
    return rows


def main():
    manifest = pd.read_csv(MANIFEST, sep="\t")
    sample_ids = manifest["Sample_ID"].tolist()

    print(f"Loading {len(sample_ids)} assembly sketches...", flush=True)
    t0 = time.time()
    asm_hashes = load_all_assemblies(sample_ids)
    asm_ids = sorted(asm_hashes.keys())
    print(f"  done in {time.time()-t0:.1f}s — {len(asm_ids)} assemblies", flush=True)

    print(f"Computing containment matrix ({len(sample_ids)} queries × {len(asm_ids)} refs) "
          f"using {N_WORKERS} workers...", flush=True)
    t0 = time.time()

    with Pool(
        processes=N_WORKERS,
        initializer=_init_worker,
        initargs=(asm_hashes, asm_ids),
    ) as pool:
        results = pool.map(_compute_row, sample_ids)

    elapsed = time.time() - t0
    print(f"  done in {elapsed/60:.1f} min", flush=True)

    # Flatten to long format
    records = [rec for rows in results if rows for rec in rows]
    long_df = pd.DataFrame(records, columns=["query_id", "reference_id", "containment"])
    long_df.to_csv(OUT_LONG, sep="\t", index=False)
    print(f"Long format: {len(long_df):,} rows → {OUT_LONG}", flush=True)

    # Pivot to wide matrix (rows=queries, cols=refs)
    wide_df = long_df.pivot(index="query_id", columns="reference_id", values="containment")
    wide_df = wide_df.reindex(index=sample_ids, columns=asm_ids)
    wide_df.to_csv(OUT_WIDE, sep="\t")
    print(f"Wide matrix: {wide_df.shape} → {OUT_WIDE}", flush=True)


if __name__ == "__main__":
    main()
