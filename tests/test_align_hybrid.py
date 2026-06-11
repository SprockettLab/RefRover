"""
Tests for hybrid (short + long read) alignment merging.

Hybrid samples align short reads with bwa/minimap2 and long reads with minimap2,
then merge both into one SAM per sample. The long-read alignment must actually be
written to disk and appended (a prior bug discarded it via the capturing _run
helper, leaving the merge to fail on a missing file).
"""

from pathlib import Path

import pandas as pd

import refrover.align as align_mod
from refrover.align import _merge_long_reads_bwa


def test_merge_long_reads_appends_long_alignments(tmp_path, monkeypatch):
    short_sam = tmp_path / "aln.sam"
    short_sam.write_text(
        "@HD\tVN:1.6\n"
        "@SQ\tSN:c1\tLN:100\n"
        "shortread1\t0\tc1\t1\t60\t4M\t*\t0\t0\tACGT\tIIII\n"
    )

    def fake_preset(minimap2_bin, ref_fa, reads, out_sam, preset, threads):
        # Stand in for minimap2: write a long-read SAM with its own header.
        assert preset == "map-ont"
        Path(out_sam).write_text(
            "@HD\tVN:1.6\n"
            "@SQ\tSN:c1\tLN:100\n"
            "longread1\t0\tc1\t1\t60\t100M\t*\t0\t0\t*\t*\n"
        )

    monkeypatch.setattr(align_mod, "_align_minimap2_preset", fake_preset)

    row = pd.Series({"sample_id": "S1", "long_reads": str(tmp_path / "S1_ont.fq")})
    _merge_long_reads_bwa(row, tmp_path / "ref.fa", short_sam, tmp_path, threads=4)

    content = short_sam.read_text()
    assert "shortread1" in content          # original short alignment kept
    assert "longread1" in content           # long alignment appended (the fix)
    assert content.count("@HD") == 1        # long header not duplicated into body


def test_merge_long_reads_does_not_raise_on_valid_inputs(tmp_path, monkeypatch):
    """Regression: the old _run-based path left long.sam unwritten and raised."""
    short_sam = tmp_path / "aln.sam"
    short_sam.write_text("@HD\tVN:1.6\nr1\t0\tc1\t1\t60\t4M\t*\t0\t0\tACGT\tIIII\n")

    monkeypatch.setattr(
        align_mod, "_align_minimap2_preset",
        lambda *a, **k: Path(a[3]).write_text("@HD\tVN:1.6\nlr\t0\tc1\t1\t60\t9M\t*\t0\t0\t*\t*\n"),
    )

    row = pd.Series({"sample_id": "S1", "long_reads": str(tmp_path / "x.fq")})
    _merge_long_reads_bwa(row, tmp_path / "ref.fa", short_sam, tmp_path, threads=1)
    assert "lr" in short_sam.read_text()
