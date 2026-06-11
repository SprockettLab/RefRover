"""Test that the pipeline records run provenance."""

import json

import pandas as pd

from refrover.pipeline import RefRoverPipeline
from refrover.selectors import ContainmentSelector, MaxMinSelector


def test_provenance_records_containment_run(tmp_path):
    manifest = pd.DataFrame({
        "sample_id": ["s0", "s1"],
        "assembly": ["a0.fasta", "a1.fasta"],
        "r1": ["s0_R1.fq", "s1_R1.fq"],
    })
    cont = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["s0", "s1"], columns=["s0", "s1"])

    pipe = RefRoverPipeline(
        manifest=manifest,
        selector=ContainmentSelector(k=2, min_containment=0.05),
        binners=["metabat2", "semibin2"],
        threads=4,
        outdir=tmp_path,
        containment_matrix=cont,
    )
    pipe._write_provenance(tmp_path / "params.json")

    data = json.loads((tmp_path / "params.json").read_text())
    assert data["selector"] == "ContainmentSelector"
    assert data["k"] == 2
    assert data["min_similarity"] == 0.05
    assert data["binners"] == ["metabat2", "semibin2"]
    assert data["n_samples"] == 2
    assert data["uses_containment_matrix"] is True
    assert "refrover_version" in data
    assert "timestamp_utc" in data


def test_provenance_records_jaccard_selector(tmp_path):
    manifest = pd.DataFrame({"sample_id": ["s0"], "assembly": ["a0.fasta"], "r1": ["x.fq"]})
    pipe = RefRoverPipeline(
        manifest=manifest,
        selector=MaxMinSelector(k=5, min_jaccard=0.2),
        binners=["generic"],
        outdir=tmp_path,
    )
    pipe._write_provenance(tmp_path / "params.json")

    data = json.loads((tmp_path / "params.json").read_text())
    assert data["selector"] == "MaxMinSelector"
    assert data["k"] == 5
    assert data["min_similarity"] == 0.2
    assert data["uses_containment_matrix"] is False
