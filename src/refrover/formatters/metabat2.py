"""
MetaBAT2 formatter: jgi_summarize_bam_contig_depths format.

Columns: contigName, contigLen, totalAvgDepth, sample1, sample1-var, sample2, sample2-var, ...

The per-sample variance is taken from the coverage table's '{sample}_var' column
when present (CoverM run with `--methods variance`). If a sample has no paired
variance column, its variance falls back to 0.0.
"""

import warnings
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

    # MetaBAT2's parser throws `bad_lexical_cast` on an empty field — which is
    # exactly how pandas serializes a NaN depth (a contig with no mapped reads in
    # some co-map sample, or an all-NaN totalAvgDepth). Coerce those to 0.0, the
    # correct depth for "no coverage", so one NaN can't sink the whole cell. Warn
    # so a systematic upstream coverage problem is still visible.
    num_cols = [c for c in df_out.columns if c != "contigName"]
    n_nan = int(df_out[num_cols].isna().to_numpy().sum())
    if n_nan:
        warnings.warn(
            f"write_metabat2: filled {n_nan} NaN depth/variance value(s) with 0.0 "
            f"in {out} (contigs with no coverage in a co-map sample)."
        )
        df_out[num_cols] = df_out[num_cols].fillna(0.0)

    df_out.to_csv(out, sep="\t", index=False)
    return out
