"""Tests for the three feature-space builders (PLAN.md §3, build step 1)."""

import numpy as np
import pandas as pd
import pytest

from refrover.feature_spaces import (
    containment_matrix,
    gtdb_abundance_matrix,
    jaccard_matrix,
    load_gtdb_species_matrix,
)


def _write_sig(hashes, name, path, ksize=31):
    """Build a scaled=1 sourmash signature from raw hashes and save it."""
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
    Two samples sA, sB with read and assembly sketches.
        reads sA = {1,2,3,4}   reads sB = {5,6,7,8}
        asm   sA = {1,2,3,4}   asm   sB = {3,4,5,6}
    Jaccard(asmA, asmB) = |{3,4}| / |{1..6}| = 2/6.
    Containment(reads_row, asm_col):
        sA->sA 1.0   sA->sB 0.5
        sB->sA 0.0   sB->sB 0.5
    """
    reads_dir = tmp_path / "reads"
    asm_dir = tmp_path / "asm"
    reads_dir.mkdir()
    asm_dir.mkdir()

    _write_sig([1, 2, 3, 4], "sA", reads_dir / "sA.sig")
    _write_sig([5, 6, 7, 8], "sB", reads_dir / "sB.sig")
    _write_sig([1, 2, 3, 4], "sA", asm_dir / "sA.sig")
    _write_sig([3, 4, 5, 6], "sB", asm_dir / "sB.sig")

    read_sigs = {"sA": reads_dir / "sA.sig", "sB": reads_dir / "sB.sig"}
    asm_sigs = {"sA": asm_dir / "sA.sig", "sB": asm_dir / "sB.sig"}
    return read_sigs, asm_sigs


# ── shared uniform-shape contract ───────────────────────────────────────────
def _assert_uniform_shape(matrix, ids):
    assert list(matrix.index) == ids
    assert list(matrix.columns) == ids
    assert matrix.index.name == "sample_id"
    assert matrix.columns.name == "assembly_id"


# ── 1. jaccard_matrix ───────────────────────────────────────────────────────
def test_jaccard_values_and_symmetry(sketches):
    _, asm_sigs = sketches
    m = jaccard_matrix(asm_sigs)
    _assert_uniform_shape(m, ["sA", "sB"])
    assert m.loc["sA", "sA"] == pytest.approx(1.0)
    assert m.loc["sB", "sB"] == pytest.approx(1.0)
    assert m.loc["sA", "sB"] == pytest.approx(2 / 6)
    assert m.loc["sA", "sB"] == pytest.approx(m.loc["sB", "sA"])  # symmetric


# ── 2. containment_matrix ───────────────────────────────────────────────────
def test_containment_values_and_axes(sketches):
    read_sigs, asm_sigs = sketches
    m = containment_matrix(read_sigs, asm_sigs)
    _assert_uniform_shape(m, ["sA", "sB"])
    assert m.loc["sA", "sA"] == pytest.approx(1.0)
    assert m.loc["sA", "sB"] == pytest.approx(0.5)
    assert m.loc["sB", "sA"] == pytest.approx(0.0)
    assert m.loc["sB", "sB"] == pytest.approx(0.5)


def test_containment_not_necessarily_symmetric(sketches):
    read_sigs, asm_sigs = sketches
    m = containment_matrix(read_sigs, asm_sigs)
    assert m.loc["sA", "sB"] != m.loc["sB", "sA"]  # directional


# ── 3. gtdb_abundance_matrix (the §3.1 bridge) ──────────────────────────────
@pytest.fixture
def taxa_tables():
    """
    Three samples, three taxa. sample_taxa = abundance of taxon in sample's reads.
        taxa:     tX    tY    tZ
        s1        0.5   0.5   0.0
        s2        0.0   0.0   1.0
        s3        0.2   0.2   0.6
    """
    sample_taxa = pd.DataFrame(
        [[0.5, 0.5, 0.0],
         [0.0, 0.0, 1.0],
         [0.2, 0.2, 0.6]],
        index=["s1", "s2", "s3"],
        columns=["tX", "tY", "tZ"],
    )
    return sample_taxa


def test_gtdb_bridge_explicit_assembly_composition(taxa_tables):
    sample_taxa = taxa_tables
    # Assemblies with hand-chosen taxonomic make-up.
    #   a1 is "all tX", a2 is "all tZ", a3 is "half tY half tZ"
    assembly_taxa = pd.DataFrame(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, 0.5, 0.5]],
        index=["a1", "a2", "a3"],
        columns=["tX", "tY", "tZ"],
    )
    m = gtdb_abundance_matrix(sample_taxa, assembly_taxa)
    assert list(m.index) == ["s1", "s2", "s3"]
    assert list(m.columns) == ["a1", "a2", "a3"]
    assert m.index.name == "sample_id"
    assert m.columns.name == "assembly_id"
    # M[i,j] = Σ_t sample_taxa[i,t] · assembly_taxa[j,t]
    # s1 (.5,.5,0) · a1 (1,0,0) = 0.5
    assert m.loc["s1", "a1"] == pytest.approx(0.5)
    # s1 · a3 (0,.5,.5) = .5*.5 = 0.25
    assert m.loc["s1", "a3"] == pytest.approx(0.25)
    # s2 (0,0,1) · a2 (0,0,1) = 1.0
    assert m.loc["s2", "a2"] == pytest.approx(1.0)
    # s2 · a1 = 0
    assert m.loc["s2", "a1"] == pytest.approx(0.0)


def test_gtdb_bridge_matches_numpy_product(taxa_tables):
    sample_taxa = taxa_tables
    m = gtdb_abundance_matrix(sample_taxa)  # proxy: assembly_taxa = sample_taxa
    expected = sample_taxa.to_numpy() @ sample_taxa.to_numpy().T
    np.testing.assert_allclose(m.to_numpy(), expected)


def test_gtdb_bridge_proxy_default_is_gram_matrix(taxa_tables):
    """With no assembly_taxa, M = B·Bᵀ — square, symmetric, sample-labelled."""
    sample_taxa = taxa_tables
    m = gtdb_abundance_matrix(sample_taxa)
    assert list(m.index) == ["s1", "s2", "s3"]
    assert list(m.columns) == ["s1", "s2", "s3"]
    # aliased index/columns must not clobber each other's axis name
    assert m.index.name == "sample_id"
    assert m.columns.name == "assembly_id"
    np.testing.assert_allclose(m.to_numpy(), m.to_numpy().T)  # symmetric


def test_gtdb_bridge_aligns_disjoint_and_shared_taxa():
    """Taxa present in only one input contribute 0; only shared taxa count."""
    sample_taxa = pd.DataFrame(
        [[1.0, 2.0, 99.0]], index=["s1"], columns=["tX", "tY", "only_sample"]
    )
    assembly_taxa = pd.DataFrame(
        [[3.0, 4.0, 99.0]], index=["a1"], columns=["tX", "tY", "only_asm"]
    )
    m = gtdb_abundance_matrix(sample_taxa, assembly_taxa)
    # only tX, tY are shared: 1*3 + 2*4 = 11; the "only_*" columns drop out.
    assert m.loc["s1", "a1"] == pytest.approx(11.0)


def test_gtdb_bridge_no_shared_taxa_raises():
    sample_taxa = pd.DataFrame([[1.0]], index=["s1"], columns=["tX"])
    assembly_taxa = pd.DataFrame([[1.0]], index=["a1"], columns=["tZ"])
    with pytest.raises(ValueError, match="share no taxa"):
        gtdb_abundance_matrix(sample_taxa, assembly_taxa)


def test_gtdb_feeds_variance_proxy(taxa_tables):
    """The bridge output is directly consumable by the Tier-1 score function."""
    from refrover.benchmark import unique_variance_explained

    m = gtdb_abundance_matrix(taxa_tables)
    frac = unique_variance_explained(m, list(m.columns))
    assert frac == pytest.approx(1.0)  # all columns span the column space


# ── load_gtdb_species_matrix (taxa-major TSV → samples × taxa) ──────────────
def test_load_gtdb_species_matrix_transposes(tmp_path):
    # Mimic run_gtdb_gather.py output: rows = taxa, columns = samples.
    wide = pd.DataFrame(
        [[0.5, 0.0, 0.2],
         [0.5, 0.0, 0.2],
         [0.0, 1.0, 0.6]],
        index=["tX", "tY", "tZ"],
        columns=["s1", "s2", "s3"],
    )
    wide.index.name = "name"
    path = tmp_path / "gtdb_species_matrix.tsv"
    wide.to_csv(path, sep="\t")

    sample_taxa = load_gtdb_species_matrix(path)
    assert list(sample_taxa.index) == ["s1", "s2", "s3"]
    assert list(sample_taxa.columns) == ["tX", "tY", "tZ"]
    assert sample_taxa.index.name == "sample_id"
    assert sample_taxa.loc["s2", "tZ"] == pytest.approx(1.0)
