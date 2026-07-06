"""Tests for BAM completeness / truncation detection in mag_benchmark.

A truncated BAM (killed alignment, out-of-disk) is missing the BGZF EOF marker
but can still have a stale sibling .bai — feeding it to CoverM dies with
BamTruncatedRecord. `_bam_complete` must reject it so alignment reruns.
"""
from refrover.mag_benchmark import _BGZF_EOF, _bam_complete, _bgzf_eof_ok


def _write_indexed_bam(bam, body: bytes, eof: bool):
    bam.write_bytes(body + (_BGZF_EOF if eof else b""))
    (bam.parent / (bam.name + ".bai")).write_bytes(b"index")


def test_eof_ok_true_for_marker_terminated_file(tmp_path):
    bam = tmp_path / "s.sorted.bam"
    bam.write_bytes(b"BAM\x01payload" + _BGZF_EOF)
    assert _bgzf_eof_ok(bam) is True


def test_eof_ok_false_for_truncated_file(tmp_path):
    bam = tmp_path / "s.sorted.bam"
    bam.write_bytes(b"BAM\x01payload-but-cut-off")  # no EOF marker
    assert _bgzf_eof_ok(bam) is False


def test_eof_ok_false_for_tiny_file(tmp_path):
    bam = tmp_path / "s.sorted.bam"
    bam.write_bytes(b"\x1f\x8b")  # smaller than the 28-byte marker
    assert _bgzf_eof_ok(bam) is False


def test_bam_complete_true_when_indexed_and_eof_present(tmp_path):
    bam = tmp_path / "s.sorted.bam"
    _write_indexed_bam(bam, b"BAM\x01payload", eof=True)
    assert _bam_complete(bam) is True


def test_bam_complete_false_when_truncated_despite_stale_index(tmp_path):
    # The 369TP2 failure mode: EOF marker gone but a .bai lingers.
    bam = tmp_path / "s.sorted.bam"
    _write_indexed_bam(bam, b"BAM\x01payload", eof=False)
    assert _bam_complete(bam) is False


def test_bam_complete_false_when_no_index(tmp_path):
    bam = tmp_path / "s.sorted.bam"
    bam.write_bytes(b"BAM\x01payload" + _BGZF_EOF)  # complete but unindexed
    assert _bam_complete(bam) is False


def test_bam_complete_false_when_missing(tmp_path):
    assert _bam_complete(tmp_path / "absent.sorted.bam") is False
