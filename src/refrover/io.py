from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"sample_id", "assembly"}
OPTIONAL_COLUMNS = {"r1", "r2", "long_reads"}


def read_manifest(path: Path | str) -> pd.DataFrame:
    """
    Read and validate a RefRover sample manifest (TSV).

    Required columns: sample_id, assembly
    At least one of r1 or long_reads must be present per row.
    r2 requires r1; absent r2 = single-end short reads.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = df.columns.str.strip().str.lower()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    if df["sample_id"].duplicated().any():
        dupes = df.loc[df["sample_id"].duplicated(keep=False), "sample_id"].unique()
        raise ValueError(f"Duplicate sample_id values: {sorted(dupes)}")

    has_r1 = "r1" in df.columns
    has_long = "long_reads" in df.columns
    if not has_r1 and not has_long:
        raise ValueError("Manifest must have at least one of: r1, long_reads")

    for _, row in df.iterrows():
        sid = row["sample_id"]
        r1 = row.get("r1") if has_r1 else None
        long = row.get("long_reads") if has_long else None
        r2 = row.get("r2") if "r2" in df.columns else None

        if pd.isna(r1) and pd.isna(long):
            raise ValueError(
                f"Sample '{sid}': at least one of r1 or long_reads must be provided"
            )
        if not pd.isna(r2) and pd.isna(r1):
            raise ValueError(f"Sample '{sid}': r2 provided but r1 is missing")

    return df.reset_index(drop=True)


def write_assignments(assignments: pd.DataFrame, path: Path | str) -> None:
    """Write prototype assignment table produced by a selector."""
    assignments.to_csv(path, sep="\t", index=False)
