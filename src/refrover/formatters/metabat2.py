"""
MetaBAT2 formatter: jgi_summarize_bam_contig_depths format.

Columns: contigName, contigLen, totalAvgDepth, sample1, sample1-var, sample2, sample2-var, ...

The per-sample variance is taken from the coverage table's '{sample}_var' column
when present (CoverM run with `--methods variance`). If a sample has no paired
variance column, its variance falls back to 0.0.
"""

from pathlib import Path
import pandas as pd

from ._columns import mean_columns, sample_of, var_column


def write_metabat2(coverage_df: pd.DataFrame, outdir: Path, force: bool = False) -> Path:
    """Write depth.txt in MetaBAT2 jgi_summarize_bam_contig_depths format."""
    out = outdir / "depth.txt"
    if out.exists() and not force:
        return out

    depth_cols = mean_columns(coverage_df)
    rows = []
    for contig, row in coverage_df.iterrows():
        rec = {
            "contigName": contig,
            "contigLen": int(row.get("length", 0)),
            "totalAvgDepth": row[depth_cols].mean(),
        }
        for col in depth_cols:
            sample = sample_of(col)
            vcol = var_column(coverage_df, col)
            rec[sample] = row[col]
            rec[f"{sample}-var"] = row[vcol] if vcol is not None else 0.0
        rows.append(rec)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, sep="\t", index=False)
    return out
