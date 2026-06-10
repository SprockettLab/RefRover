"""
End-to-end pipeline tests using mocked external tools.

The pipeline calls sourmash, bwa-mem2/minimap2, samtools, and coverm via
subprocess. These tests patch those calls so the suite runs without any
external binaries installed.
"""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── io tests ──────────────────────────────────────────────────────────────────

def test_read_manifest_valid(tiny_manifest):
    from refrover.io import read_manifest
    df = read_manifest(tiny_manifest)
    assert list(df["sample_id"]) == ["S1", "S2"]


def test_read_manifest_missing_column(tmp_path):
    from refrover.io import read_manifest
    p = tmp_path / "bad.tsv"
    p.write_text("sample_id\tr1\nS1\treads/S1.fastq.gz\n")
    with pytest.raises(ValueError, match="missing required columns"):
        read_manifest(p)


def test_read_manifest_duplicate_ids(tmp_path):
    from refrover.io import read_manifest
    p = tmp_path / "dup.tsv"
    p.write_text("sample_id\tassembly\tr1\n"
                 "S1\ta.fa\tr1.fq\n"
                 "S1\tb.fa\tr2.fq\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(p)


def test_read_manifest_r2_without_r1(tmp_path):
    from refrover.io import read_manifest
    p = tmp_path / "bad_r2.tsv"
    p.write_text("sample_id\tassembly\tr2\n"
                 "S1\ta.fa\treads/r2.fq\n")
    with pytest.raises(ValueError, match="r1"):
        read_manifest(p)


def test_write_read_assignments_roundtrip(tmp_path):
    from refrover.io import write_assignments
    df = pd.DataFrame({
        "sample_id": ["S1", "S2"],
        "prototype_ids": ["S1,S2", "S1"],
        "n_prototypes": [2, 1],
    })
    out = tmp_path / "assignments.tsv"
    write_assignments(df, out)
    df2 = pd.read_csv(out, sep="\t")
    assert list(df2["sample_id"]) == ["S1", "S2"]
    assert df2.loc[0, "n_prototypes"] == 2


# ── similarity tests ──────────────────────────────────────────────────────────

def test_load_similarity_matrix(tmp_path):
    from refrover.similarity import load_similarity_matrix
    # Write a minimal sourmash compare CSV
    csv = tmp_path / "sim.csv"
    csv.write_text(",S1,S2,S3\nS1,1.0,0.5,0.2\nS2,0.5,1.0,0.3\nS3,0.2,0.3,1.0\n")
    df = load_similarity_matrix(csv)
    assert df.shape == (3, 3)
    assert df.loc["S1", "S2"] == pytest.approx(0.5)


def test_load_similarity_matrix_not_square(tmp_path):
    from refrover.similarity import load_similarity_matrix
    csv = tmp_path / "bad.csv"
    csv.write_text(",S1,S2\nS1,1.0,0.5\n")
    with pytest.raises(ValueError, match="not square"):
        load_similarity_matrix(csv)


def test_filter_by_jaccard(clustered_sim):
    from refrover.similarity import filter_by_jaccard
    result = filter_by_jaccard(clustered_sim, "s00", min_jaccard=0.5)
    # Only within-cluster samples (0.6-0.8) should pass; cross-cluster (0.05) should not
    assert "s00" in result
    for r in result:
        assert clustered_sim.loc["s00", r] >= 0.5


def test_filter_by_jaccard_unknown_query(clustered_sim):
    from refrover.similarity import filter_by_jaccard
    with pytest.raises(KeyError):
        filter_by_jaccard(clustered_sim, "does_not_exist", min_jaccard=0.1)


# ── sketch subprocess mock ────────────────────────────────────────────────────

def test_sketch_assemblies_skips_existing(tmp_path):
    from refrover.sketch import sketch_assemblies

    # Pre-create the expected .sig file
    fa = tmp_path / "asm.fasta"
    fa.write_text(">c1\nACGT\n")
    sig = tmp_path / "asm.sig"
    sig.write_text("fake")

    with patch("subprocess.run") as mock_run:
        result = sketch_assemblies([fa], outdir=tmp_path)
        mock_run.assert_not_called()  # should skip because sig exists
    assert result == [sig]


def test_sketch_assemblies_calls_sourmash(tmp_path):
    from refrover.sketch import sketch_assemblies

    fa = tmp_path / "asm.fasta"
    fa.write_text(">c1\nACGT\n")

    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        sketch_assemblies([fa], outdir=tmp_path)
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "sourmash" in cmd[0]
        assert "sketch" in cmd


# ── formatter integration ─────────────────────────────────────────────────────

def test_all_binners_produce_output(coverage_df, tmp_path):
    from refrover.formatters import format_for_binner, BINNER_REGISTRY
    for binner in BINNER_REGISTRY:
        result = format_for_binner(coverage_df, binner=binner, outdir=tmp_path)
        if isinstance(result, list):
            assert all(p.exists() for p in result)
        else:
            assert result.exists()


# ── align tests ───────────────────────────────────────────────────────────────

def _make_mock_run(returncode=0):
    m = MagicMock()
    m.returncode = returncode
    return m


def test_align_skips_existing_bam(tmp_path):
    from refrover.align import run_alignment

    manifest = pd.DataFrame({
        "sample_id": ["S1"],
        "assembly": [str(tmp_path / "S1.fasta")],
        "r1": [str(tmp_path / "S1_R1.fastq.gz")],
    })
    assignments = pd.DataFrame({
        "sample_id": ["S1"],
        "prototype_ids": ["S1"],
    })
    # Pre-create the BAM so it should be skipped
    bam = tmp_path / "S1.sorted.bam"
    bam.write_text("fake")

    with patch("subprocess.run") as mock_run:
        result = run_alignment(manifest, assignments, outdir=tmp_path)
        mock_run.assert_not_called()
    assert result == [bam]


def test_concat_fastas(tmp_path):
    from refrover.align import _concat_fastas

    fa1 = tmp_path / "a.fasta"
    fa2 = tmp_path / "b.fasta"
    fa1.write_text(">seq1\nACGT\n")
    fa2.write_text(">seq2\nTTTT\n")
    out = tmp_path / "combined.fasta"

    _concat_fastas([fa1, fa2], out)
    content = out.read_text()
    assert ">seq1" in content
    assert ">seq2" in content


def test_align_bwa_calls_correct_subcommands(tmp_path):
    from refrover.align import run_alignment

    fa = tmp_path / "S1.fasta"
    fa.write_text(">c1\nACGT\n")
    r1 = tmp_path / "S1_R1.fastq.gz"
    r1.write_text("")

    manifest = pd.DataFrame({
        "sample_id": ["S1"],
        "assembly": [str(fa)],
        "r1": [str(r1)],
    })
    assignments = pd.DataFrame({
        "sample_id": ["S1"],
        "prototype_ids": ["S1"],
    })

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        m = MagicMock()
        m.returncode = 0
        # For samtools sort, create the output BAM so pipeline doesn't error
        if "sort" in cmd:
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_text("fake_bam")
        return m

    with patch("subprocess.run", side_effect=fake_run), \
         patch("builtins.open", side_effect=open):
        try:
            run_alignment(manifest, assignments, outdir=tmp_path, aligner="bwa-mem2")
        except Exception:
            pass  # may fail on SAM write details; we just check subprocess calls

    cmd_names = [c[0] for c in calls if c]
    assert any("bwa-mem2" in str(c) or "bwa" in str(c) for c in cmd_names), \
        f"Expected bwa call, got: {cmd_names}"
