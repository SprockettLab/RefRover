"""
MetaBAT2 formatter: jgi_summarize_bam_contig_depths format.

Columns: contigName, contigLen, totalAvgDepth, sample1, sample1-var, sample2, sample2-var, ...
Variance columns are set to 0 (RefRover does not compute variance from CoverM mean output).
"""

from pathlib import Path
import pandas as pd


def write_metabat2(coverage_df: pd.DataFrame, outdir: Path, force: bool = False) -> Path:
    """Write depth.txt in MetaBAT2 jgi_summarize_bam_contig_depths format."""
    out = outdir / "depth.txt"
    if out.exists() and not force:
        return out

    depth_cols = [c for c in coverage_df.columns if c != "length"]
    rows = []
    for contig, row in coverage_df.iterrows():
        rec = {
            "contigName": contig,
            "contigLen": int(row.get("length", 0)),
            "totalAvgDepth": row[depth_cols].mean(),
        }
        for col in depth_cols:
            rec[col] = row[col]
            rec[f"{col}-var"] = 0.0
        rows.append(rec)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, sep="\t", index=False)
    return out
