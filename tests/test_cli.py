"""CLI integration tests for the containment selector wiring."""

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from refrover.cli import main


@pytest.fixture
def containment_files(tmp_path):
    """
    Write a 4-sample containment matrix TSV and a matching manifest.
    Two groups: (s0, s1) and (s2, s3); within-group containment high,
    cross-group low.
    """
    ids = ["s0", "s1", "s2", "s3"]
    mat = np.array([
        [1.00, 0.40, 0.05, 0.04],
        [0.42, 1.00, 0.06, 0.05],
        [0.05, 0.04, 1.00, 0.45],
        [0.06, 0.05, 0.43, 1.00],
    ])
    cont = pd.DataFrame(mat, index=ids, columns=ids)
    cont.index.name = "query_id"
    cont_path = tmp_path / "containment.tsv"
    cont.to_csv(cont_path, sep="\t")

    manifest = pd.DataFrame({
        "sample_id": ids,
        "assembly": [f"assemblies/{s}.fasta" for s in ids],
        "r1": [f"reads/{s}_R1.fastq.gz" for s in ids],
    })
    man_path = tmp_path / "manifest.tsv"
    manifest.to_csv(man_path, sep="\t", index=False)

    return cont_path, man_path, tmp_path


def test_select_containment_writes_assignments(containment_files):
    cont_path, man_path, tmp_path = containment_files
    outdir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(main, [
        "select",
        "--selector", "containment",
        "--containment-matrix", str(cont_path),
        "--manifest", str(man_path),
        "--k", "2",
        "--min-jaccard", "0.05",
        "--outdir", str(outdir),
    ])
    assert result.exit_code == 0, result.output

    assignments = pd.read_csv(outdir / "assignments.tsv", sep="\t")
    assert set(assignments["sample_id"]) == {"s0", "s1", "s2", "s3"}
    assert (assignments["n_prototypes"] <= 2).all()
    # Each sample's own assembly should be the first prototype.
    for _, row in assignments.iterrows():
        first = str(row["prototype_ids"]).split(",")[0]
        assert first == row["sample_id"]


def test_select_containment_requires_matrix(containment_files):
    _, man_path, tmp_path = containment_files
    runner = CliRunner()
    result = runner.invoke(main, [
        "select",
        "--selector", "containment",
        "--manifest", str(man_path),
        "--k", "2",
        "--outdir", str(tmp_path / "out2"),
    ])
    assert result.exit_code != 0
    assert "requires --containment-matrix" in result.output


def test_select_jaccard_requires_sketches(containment_files):
    _, man_path, tmp_path = containment_files
    runner = CliRunner()
    result = runner.invoke(main, [
        "select",
        "--selector", "maxmin",
        "--manifest", str(man_path),
        "--k", "2",
        "--outdir", str(tmp_path / "out3"),
    ])
    assert result.exit_code != 0
    assert "--jaccard-matrix" in result.output or "--sketches" in result.output
