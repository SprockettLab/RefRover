"""
Feature-space builders (PLAN.md §3, build step 1).

A *feature space* is the matrix we select references on. PLAN.md decomposes the
problem into ``FEATURE SPACE × SELECTION RULE × k`` and insists all three feature
spaces be normalized to **one shape** so a single scoring function compares them
apples-to-apples:

    a ``samples × per-sample-assemblies`` DataFrame, where

        M[i, j] = "signal of reference assembly j in sample i"

    rows  (index)   = sample_ids whose reads we map   (axis name "sample_id")
    cols  (columns) = candidate reference assemblies   (axis name "assembly_id")

Because sample ↔ assembly is 1:1 (PLAN §2) every matrix here is square and
labelled by the same sample_ids on both axes. Each *column* is a reference's
vector over samples — the unified representation of PLAN §4, and exactly what
``benchmark.unique_variance_explained`` (and the upcoming score family) consume.

Three builders, one per feature space in PLAN §3:

    jaccard_matrix          — assembly-vs-assembly Jaccard (the cheap floor)
    containment_matrix      — reads-vs-assembly containment (no-DB middle tier)
    gtdb_abundance_matrix   — GTDB taxa-routed abundance (the §3.1 bridge, NEW)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from refrover.containment import _load_minhash, compute_containment_matrix

_SAMPLE_AXIS = "sample_id"
_ASSEMBLY_AXIS = "assembly_id"


def _normalize_axes(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the uniform axis names every feature space shares.

    Uses ``rename_axis`` (which returns fresh axis objects) rather than mutating
    ``.name`` in place: in the GTDB proxy case the index and columns are the same
    ``Index`` instance, so in-place naming would clobber one with the other.
    """
    return matrix.rename_axis(index=_SAMPLE_AXIS, columns=_ASSEMBLY_AXIS)


# ── 1. Jaccard (assembly sketches) ──────────────────────────────────────────
def jaccard_matrix(
    assembly_sigs: dict[str, Path | str],
    *,
    ksize: int = 31,
) -> pd.DataFrame:
    """
    Build the Jaccard feature space: M[i, j] = Jaccard(assembly_i, assembly_j).

    The cheap "floor" of PLAN §3 — it never sees read abundance, only how
    genetically similar two assemblies look. Computed directly from the assembly
    MinHash sketches via the sourmash Python API (no subprocess, no CLI compare).

    Parameters
    ----------
    assembly_sigs : dict
        sample_id -> assembly sketch (.sig) path. Keys label both axes.
    ksize : int
        k-mer size; all sketches must share it.

    Returns
    -------
    pd.DataFrame
        Square samples × assemblies matrix (symmetric, unit diagonal), labelled
        by sample_id. Index name "sample_id", columns name "assembly_id".
    """
    ids = sorted(assembly_sigs)
    minhashes = {a: _load_minhash(assembly_sigs[a], ksize) for a in ids}

    n = len(ids)
    mat = np.eye(n, dtype=np.float64)
    for a in range(n):
        mh_a = minhashes[ids[a]]
        for b in range(a + 1, n):
            j = float(mh_a.jaccard(minhashes[ids[b]]))
            mat[a, b] = j
            mat[b, a] = j

    df = pd.DataFrame(mat, index=ids, columns=ids)
    return _normalize_axes(df)


# ── 2. Reads × assembly containment (no database) ───────────────────────────
def containment_matrix(
    read_sigs: dict[str, Path | str],
    assembly_sigs: dict[str, Path | str],
    *,
    ksize: int = 31,
) -> pd.DataFrame:
    """
    Build the containment feature space: M[i, j] = fraction of sample i's read
    k-mers found in assembly j.

    The no-DB middle tier of PLAN §3 — directly about whether reads will map,
    needing only read + assembly sketches. This is a thin uniform-axes wrapper
    over ``containment.compute_containment_matrix`` (the canonical
    implementation); it only renames the axes to the shared
    ``sample_id`` / ``assembly_id`` convention so the three feature spaces line
    up exactly.

    Parameters
    ----------
    read_sigs, assembly_sigs : dict
        sample_id -> read / assembly sketch (.sig) path.
    ksize : int
        k-mer size; read and assembly sketches must share it.

    Returns
    -------
    pd.DataFrame
        samples × assemblies containment matrix in [0, 1]. Not symmetric
        (containment is directional). Index name "sample_id", columns name
        "assembly_id".
    """
    df = compute_containment_matrix(read_sigs, assembly_sigs, ksize=ksize)
    return _normalize_axes(df)


# ── 3. GTDB taxa-abundance (the §3.1 taxa-routing bridge) ────────────────────
def gtdb_abundance_matrix(
    sample_taxa: pd.DataFrame,
    assembly_taxa: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the GTDB-abundance feature space — the §3.1 taxa-routing bridge.

    GTDB gather natively yields ``samples × taxa`` (abundance of each GTDB taxon
    in each sample's reads). But we map reads against *assemblies*, not GTDB
    genomes, so we route the signal back onto assemblies:

        assembly_j → its GTDB taxa → abundance of those taxa in sample i = M[i, j]

    Concretely this is a taxa-aligned bilinear product::

        M[i, j] = Σ_t  sample_taxa[i, t] · assembly_taxa[j, t]

    "Assembly j is mostly Bacteroides + Akkermansia; how much Bacteroides +
    Akkermansia is in sample i?" — that number is the cell. It is a
    taxonomy-mediated, low-rank version of the containment matrix.

    Parameters
    ----------
    sample_taxa : pd.DataFrame
        ``samples × taxa`` abundance — rows = sample_ids (whose reads we map),
        columns = GTDB taxon names, values = abundance of that taxon in the
        sample's reads (e.g. gather ``f_unique_to_query``). Load the mouse-set
        matrix with :func:`load_gtdb_species_matrix`.
    assembly_taxa : pd.DataFrame, optional
        ``assemblies × taxa`` composition — rows = assembly sample_ids, columns =
        GTDB taxon names, values = weight of that taxon in the assembly (from
        gathering each *assembly* against GTDB). If omitted, ``sample_taxa`` is
        reused as the composition: because sample ↔ assembly is 1:1 (PLAN §2),
        assembly j's read-gather profile is a faithful stand-in for assembly j's
        taxonomic make-up. That degenerate case makes M = B·Bᵀ, a Gram matrix of
        taxa-abundance profiles, runnable on the existing mouse data with no
        assembly-side gather.

    Returns
    -------
    pd.DataFrame
        samples × assemblies abundance-signal matrix. Index = ``sample_taxa``
        rows (sample_id), columns = ``assembly_taxa`` rows (assembly_id).

    Notes
    -----
    Only taxa shared by both inputs contribute; a taxon present in one and absent
    from the other has implied abundance/weight 0 and so contributes 0 to every
    product, exactly as dropping it does.
    """
    if assembly_taxa is None:
        assembly_taxa = sample_taxa

    shared = sample_taxa.columns.intersection(assembly_taxa.columns)
    if len(shared) == 0:
        raise ValueError(
            "sample_taxa and assembly_taxa share no taxa columns; "
            "cannot route abundance through taxonomy."
        )

    B = sample_taxa[shared].to_numpy(dtype=float)       # samples × taxa
    A = assembly_taxa[shared].to_numpy(dtype=float)     # assemblies × taxa
    M = B @ A.T                                         # samples × assemblies

    df = pd.DataFrame(M, index=sample_taxa.index, columns=assembly_taxa.index)
    return _normalize_axes(df)


def load_gtdb_species_matrix(path: Path | str) -> pd.DataFrame:
    """
    Load ``gtdb_species_matrix.tsv`` as a ``samples × taxa`` DataFrame.

    ``run_gtdb_gather.py`` writes the matrix taxa-major (rows = GTDB taxon name,
    columns = sample_id). This loads and transposes it to the sample-major shape
    :func:`gtdb_abundance_matrix` expects, missing cells filled with 0.

    Returns
    -------
    pd.DataFrame
        Rows = sample_id, columns = GTDB taxon name, values = abundance.
    """
    wide = pd.read_csv(path, sep="\t", index_col=0)  # taxa × samples
    sample_taxa = wide.T.fillna(0.0)                 # samples × taxa
    sample_taxa.index.name = _SAMPLE_AXIS
    sample_taxa.columns.name = "taxon"
    return sample_taxa
