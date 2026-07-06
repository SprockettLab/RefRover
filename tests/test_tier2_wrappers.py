"""Tests for the Tier-2 leaf wrappers: mag_quality, binning, checkm2."""

from unittest.mock import patch

import pandas as pd
import pytest

from refrover.binning import list_bins, run_metabat2
from refrover.checkm2 import load_quality_report, run_checkm2
from refrover.mag_quality import count_mags


# ── mag_quality.count_mags ──────────────────────────────────────────────────
@pytest.fixture
def quality():
    # 5 bins spanning the tiers.
    return pd.DataFrame(
        {
            "Name":          ["b1",  "b2",  "b3",  "b4",  "b5"],
            "Completeness":  [95.0,  92.0,  70.0,  60.0,  30.0],
            "Contamination": [2.0,   8.0,   3.0,   12.0,  1.0],
        }
    )
    # b1: high (95/2). b2: comp>=90 but cont 8>5 -> medium. b3: medium (70/3).
    # b4: comp 60>=50 but cont 12>10 -> low. b5: comp 30<50 -> low.


def test_count_mags_tiers(quality):
    out = count_mags(quality)
    assert out["n_bins"] == 5
    assert out["n_high"] == 1          # b1
    assert out["n_medium"] == 2        # b2, b3
    assert out["n_low"] == 2           # b4, b5
    assert out["weighted_mags"] == pytest.approx(2 * 1 + 1 * 2)  # 4.0


def test_count_mags_empty():
    out = count_mags(pd.DataFrame({"Completeness": [], "Contamination": []}))
    assert out == {"n_bins": 0, "n_high": 0, "n_medium": 0, "n_low": 0,
                   "weighted_mags": 0.0, "sum_qs": 0.0}


def test_count_mags_sum_qs(quality):
    out = count_mags(quality)
    # b1: 95 - 5*2 = 85; b2: 92 - 5*8 = 52; b3: 70 - 5*3 = 55;
    # b4: 60 - 5*12 = -0 (negative, excluded); b5: 30 - 5*1 = 25
    expected = 85.0 + 52.0 + 55.0 + 25.0
    assert out["sum_qs"] == pytest.approx(expected)


def test_count_mags_custom_thresholds(quality):
    # Loosen high to 90/10 -> b1 and b2 both become high.
    out = count_mags(quality, high=(90.0, 10.0))
    assert out["n_high"] == 2


# ── binning.run_metabat2 ────────────────────────────────────────────────────
def test_run_metabat2_invokes_cli_and_returns_dir(tmp_path):
    contigs = tmp_path / "focal.fasta"
    contigs.write_text(">c1\nACGT\n")
    depth = tmp_path / "depth.txt"
    depth.write_text("contigName\tcontigLen\n")
    outdir = tmp_path / "bins"

    def fake_run(cmd, **kw):
        # MetaBAT2 would write bins; emulate one bin file.
        (outdir).mkdir(parents=True, exist_ok=True)
        (outdir / "bin.1.fa").write_text(">c1\nACGT\n")
        return type("R", (), {"returncode": 0, "stderr": ""})()

    with patch("subprocess.run", side_effect=fake_run) as m:
        result = run_metabat2(contigs, depth, outdir, threads=4, min_contig=2000)
    assert result == outdir
    cmd = m.call_args[0][0]
    assert cmd[0] == "metabat2"
    assert "-m" in cmd and "2000" in cmd and "-t" in cmd and "4" in cmd
    assert list_bins(outdir) == [outdir / "bin.1.fa"]


def test_run_metabat2_idempotent(tmp_path):
    outdir = tmp_path / "bins"
    outdir.mkdir()
    (outdir / "bin.1.fa").write_text(">c\nA\n")
    with patch("subprocess.run") as m:
        run_metabat2(tmp_path / "f.fasta", tmp_path / "d.txt", outdir)
    m.assert_not_called()  # existing bins -> skip


def test_run_metabat2_raises_on_failure(tmp_path):
    fail = type("R", (), {"returncode": 1, "stderr": "boom"})()
    with patch("subprocess.run", return_value=fail):
        with pytest.raises(RuntimeError, match="MetaBAT2 failed"):
            run_metabat2(tmp_path / "f.fasta", tmp_path / "d.txt", tmp_path / "out")


# ── checkm2.run_checkm2 ─────────────────────────────────────────────────────
def _write_report(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"Name": ["bin.1"], "Completeness": [93.0], "Contamination": [2.0]}
    ).to_csv(outdir / "quality_report.tsv", sep="\t", index=False)


def test_run_checkm2_invokes_cli_and_parses(tmp_path):
    bins = tmp_path / "bins"
    bins.mkdir()
    out = tmp_path / "checkm2_out"

    def fake_run(cmd, **kw):
        _write_report(out)
        return type("R", (), {"returncode": 0, "stderr": ""})()

    with patch("shutil.which", return_value="/usr/bin/checkm2"), \
         patch("subprocess.run", side_effect=fake_run) as m:
        df = run_checkm2(bins, out, db_path="/db/uniref.dmnd", threads=4)
    cmd = m.call_args[0][0]
    assert cmd[:2] == ["checkm2", "predict"]
    assert "--database_path" in cmd and "/db/uniref.dmnd" in cmd
    assert df.loc[0, "Completeness"] == 93.0


def test_run_checkm2_idempotent(tmp_path):
    out = tmp_path / "checkm2_out"
    _write_report(out)
    with patch("subprocess.run") as m:
        df = run_checkm2(tmp_path / "bins", out)
    m.assert_not_called()  # existing report -> load, don't re-run
    assert df.loc[0, "Name"] == "bin.1"


def test_run_checkm2_raises_when_no_report(tmp_path):
    ok_no_file = type("R", (), {"returncode": 0, "stderr": ""})()
    with patch("shutil.which", return_value="/usr/bin/checkm2"), \
         patch("subprocess.run", return_value=ok_no_file):
        with pytest.raises(RuntimeError, match="no quality_report"):
            run_checkm2(tmp_path / "bins", tmp_path / "out")


def test_run_checkm2_auto_force_on_partial_dir(tmp_path):
    # Simulate a partial run: outdir exists but quality_report.tsv is missing.
    out = tmp_path / "checkm2_out"
    out.mkdir()
    (out / "diamond.tsv").write_text("partial\n")  # leftover from a crashed run

    def fake_run(cmd, **kw):
        _write_report(out)
        return type("R", (), {"returncode": 0, "stderr": ""})()

    with patch("shutil.which", return_value="/usr/bin/checkm2"), \
         patch("subprocess.run", side_effect=fake_run) as m:
        run_checkm2(tmp_path / "bins", out)
    cmd = m.call_args[0][0]
    assert "--force" in cmd


def test_load_quality_report_roundtrip(tmp_path):
    _write_report(tmp_path / "o")
    df = load_quality_report(tmp_path / "o" / "quality_report.tsv")
    assert list(df.columns) == ["Name", "Completeness", "Contamination"]


def test_run_checkm2_fails_fast_when_binary_missing(tmp_path):
    # An unresolvable binary must raise an actionable RuntimeError before any
    # subprocess call — not a raw FileNotFoundError that silently sinks cells.
    with patch("shutil.which", return_value=None), \
         patch("subprocess.run") as m:
        with pytest.raises(RuntimeError, match="CheckM2 binary not found"):
            run_checkm2(tmp_path / "bins", tmp_path / "out", checkm2_path="nope")
    m.assert_not_called()
