#!/usr/bin/env python3
"""
Build samples.tsv for the AMY1 dataset on cbsupoole.

Usage (from the RefRover working directory):
    python scripts/make_manifest.py

Writes samples.tsv with columns: sample_id, assembly, r1, r2
"""
import os
import pandas as pd

ASM_DIR = "/workdir/Sprockett/Projects/CU15_AMY1_Copy_Number/sn-mg-pipeline/output/assemble/megahit"
READS_DIR = "/workdir/Sprockett/Projects/CU15_AMY1_Copy_Number/sn-mg-pipeline/output/qc/host_filter/nonhost"

rows = []
for f in sorted(os.listdir(ASM_DIR)):
    if not f.endswith(".contigs.fasta"):
        continue
    sid = f.replace(".contigs.fasta", "")
    r1 = os.path.join(READS_DIR, f"{sid}.R1.fastq.gz")
    r2 = os.path.join(READS_DIR, f"{sid}.R2.fastq.gz")
    asm = os.path.join(ASM_DIR, f)
    if not os.path.exists(r1):
        print(f"WARNING: no R1 for {sid}, skipping")
        continue
    rows.append({
        "sample_id": sid,
        "assembly": asm,
        "r1": r1,
        "r2": r2 if os.path.exists(r2) else "",
    })

df = pd.DataFrame(rows)
df.to_csv("samples.tsv", sep="\t", index=False)
print(f"Wrote {len(df)} samples to samples.tsv")
print(df[["sample_id", "r2"]].to_string())
