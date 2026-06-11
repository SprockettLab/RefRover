"""Tests for the package containment-matrix computation."""


import pytest

from refrover.containment import (
    compute_containment_matrix,
    sigs_by_sample,
    write_containment_matrix,
)
from refrover.similarity import load_containment_matrix


def _write_sig(hashes, name, path, ksize=31):
    """Build a sourmash signature from raw hashes and save it to path."""
    from sourmash import MinHash, SourmashSignature
    from sourmash.sourmash_args import SaveSignaturesToLocation

    mh = MinHash(n=0, ksize=ksize, scaled=1)
    for h in hashes:
        mh.add_hash(h)
    sig = SourmashSignature(mh, name=name)
    with SaveSignaturesToLocation(str(path)) as save:
        save.add(sig)


@pytest.fixture
def sketches(tmp_path):
    """
    Two samples rA, rB with read and assembly sketches.
    reads rA = {1,2,3,4}, reads rB = {5,6,7,8}
    asm   rA = {1,2,3,4}, asm   rB = {3,4,5,6}
    Expected containment (reads row vs assembly col):
        rA->rA 1.0   rA->rB 0.5
        rB->rA 0.0   rB->rB 0.5
    """
    reads_dir = tmp_path / "reads"
    asm_dir = tmp_path / "asm"
    reads_dir.mkdir()
    asm_dir.mkdir()

    _write_sig([1, 2, 3, 4], "rA", reads_dir / "rA.sig")
    _write_sig([5, 6, 7, 8], "rB", reads_dir / "rB.sig")
    _write_sig([1, 2, 3, 4], "rA", asm_dir / "rA.sig")
    _write_sig([3, 4, 5, 6], "rB", asm_dir / "rB.sig")

    return reads_dir, asm_dir


def test_sigs_by_sample_maps_stems(sketches):
    reads_dir, asm_dir = sketches
    rmap = sigs_by_sample(reads_dir)
    assert set(rmap) == {"rA", "rB"}
    assert rmap["rA"].name == "rA.sig"


def test_containment_values(sketches):
    reads_dir, asm_dir = sketches
    matrix = compute_containment_matrix(
        sigs_by_sample(reads_dir), sigs_by_sample(asm_dir)
    )
    assert matrix.loc["rA", "rA"] == pytest.approx(1.0)
    assert matrix.loc["rA", "rB"] == pytest.approx(0.5)
    assert matrix.loc["rB", "rA"] == pytest.approx(0.0)
    assert matrix.loc["rB", "rB"] == pytest.approx(0.5)


def test_matrix_shape_and_labels(sketches):
    reads_dir, asm_dir = sketches
    matrix = compute_containment_matrix(
        sigs_by_sample(reads_dir), sigs_by_sample(asm_dir)
    )
    assert matrix.shape == (2, 2)
    assert list(matrix.index) == ["rA", "rB"]
    assert list(matrix.columns) == ["rA", "rB"]
    assert matrix.index.name == "query_id"


def test_write_and_reload_roundtrip(sketches, tmp_path):
    reads_dir, asm_dir = sketches
    matrix = compute_containment_matrix(
        sigs_by_sample(reads_dir), sigs_by_sample(asm_dir)
    )
    out = write_containment_matrix(matrix, tmp_path / "containment_matrix.tsv")
    reloaded = load_containment_matrix(out)
    assert reloaded.loc["rA", "rB"] == pytest.approx(0.5)
    assert list(reloaded.columns) == ["rA", "rB"]


def test_feeds_containment_selector(sketches):
    """The computed matrix is directly consumable by ContainmentSelector."""
    from refrover.selectors import ContainmentSelector

    reads_dir, asm_dir = sketches
    matrix = compute_containment_matrix(
        sigs_by_sample(reads_dir), sigs_by_sample(asm_dir)
    )
    sel = ContainmentSelector(k=2, min_containment=0.01)
    result = sel.select(matrix, "rA")
    assert result[0] == "rA"  # own assembly first (containment 1.0)
    assert set(result) <= {"rA", "rB"}
