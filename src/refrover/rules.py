"""
Selection rules (PLAN.md §5, build step 2) — the unified column-vector interface.

PLAN §4: *represent each candidate reference as its column vector over samples.*
Then every geometric rule operates on vectors in the **same sample-space**, and
the feature space (Jaccard / containment / GTDB-abundance, all the uniform
`samples × assemblies` shape from `feature_spaces.py`) just decides what's in the
vectors. This dissolves the old "is this selector valid on this matrix?" question:
one rule, any feature matrix.

A **Rule** does three things in `select(M, query_id)`:

1. **Candidate filter** (per-feature-space, but uniform in code): sample *i* only
   considers references its reads can plausibly map to — the columns where the
   query's own row clears a threshold (min Jaccard / min containment / min shared
   abundance, depending on the feature space). "Per-sample" lives entirely in
   *which columns are eligible*.
2. **Anchor**: the query's own assembly is always pinned first — the reference its
   reads are guaranteed to map to — so every rule is handed the same anchor and
   differs only in how it picks the remaining k−1.
3. **Rank**: pick the rest from the candidate column vectors by the rule's
   objective (span the differential signal).

This replaces the legacy `selectors/` package (kept intact for the existing
pipeline/CLI). `ContainmentSelector` collapses into the grid here: it was just
"maxmin on the containment feature space," i.e. the `(containment, maxmin)` cell.
"""

from __future__ import annotations

import random
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Shared geometry over reference column vectors
# ──────────────────────────────────────────────────────────────────────────────
def _centered(values: np.ndarray) -> np.ndarray:
    """Mean-centre each column (reference vector) over samples (axis 0)."""
    return values - values.mean(axis=0, keepdims=True)


def _correlation_distance(columns: np.ndarray) -> np.ndarray:
    """
    Pairwise distance between reference column vectors: 1 − Pearson correlation.

    `columns` is samples × references. Correlation measures how similarly two
    references vary *across samples* — the quantity differential coverage cares
    about — and is scale-invariant, so a feature space's absolute magnitudes don't
    bias the geometry. Constant columns (a reference with no cross-sample
    variation) have undefined correlation; we treat them as uncorrelated
    (distance 1), matching the legacy ContainmentSelector.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(columns, rowvar=False)
    corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    return dist


# ──────────────────────────────────────────────────────────────────────────────
# Base rule
# ──────────────────────────────────────────────────────────────────────────────
class Rule(ABC):
    """
    A selection rule over a uniform `samples × assemblies` feature matrix.

    Parameters
    ----------
    k : int
        Number of references to return (the budget).
    threshold : float
        Candidate filter: a reference *j* is eligible for query *i* only if
        ``M[i, j] >= threshold``. The threshold is feature-space-appropriate
        (min Jaccard / min containment / min abundance); the *code* is uniform
        because every feature space is the same shape with ``M[i, j]`` = "signal
        of reference j in sample i". Default 0.0 = no filtering (every reference
        with non-negative signal is eligible).
    """

    #: stable id used in the registry and the grid's `rule` column
    name: str = ""

    def __init__(self, k: int, threshold: float = 0.0):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        self.k = k
        self.threshold = threshold

    # -- candidate filter (uniform across feature spaces) --------------------
    def _candidates(self, M: pd.DataFrame, query_id: str) -> list[str]:
        """Eligible reference ids for query: query's row >= threshold, query first."""
        if query_id not in M.index:
            raise KeyError(f"query_id '{query_id}' not found in feature matrix rows")
        row = M.loc[query_id]
        candidates = row[row >= self.threshold].index.tolist()
        if not candidates:
            raise ValueError(
                f"No candidates for '{query_id}' at threshold={self.threshold}. "
                "Lower the threshold or check the feature matrix."
            )
        # Pin the query's own assembly first when it is eligible; otherwise order
        # by descending signal to the query so the strongest anchor leads.
        if query_id in candidates:
            return [query_id] + [c for c in candidates if c != query_id]
        return sorted(candidates, key=lambda c: -float(row[c]))

    def select(self, M: pd.DataFrame, query_id: str) -> list[str]:
        """Return up to k reference ids for query_id, query-anchored first."""
        candidates = self._candidates(M, query_id)
        if len(candidates) <= self.k:
            if len(candidates) < self.k:
                warnings.warn(
                    f"Only {len(candidates)} candidates available (k={self.k}). "
                    "Returning all candidates.",
                    UserWarning,
                    stacklevel=2,
                )
            return candidates
        return self._rank(M, query_id, candidates)

    @abstractmethod
    def _rank(self, M: pd.DataFrame, query_id: str, candidates: list[str]) -> list[str]:
        """Pick k references from `candidates` (already query-anchored first)."""


# ──────────────────────────────────────────────────────────────────────────────
# Rules
# ──────────────────────────────────────────────────────────────────────────────
class RandomRule(Rule):
    """Baseline: the query anchor plus k−1 uniformly random eligible references."""

    name = "random"

    def __init__(self, k: int, threshold: float = 0.0, seed: int | None = 0):
        super().__init__(k, threshold)
        self.seed = seed

    def _rank(self, M, query_id, candidates):
        rng = random.Random(self.seed)
        anchor = candidates[0]
        others = candidates[1:]
        return [anchor] + rng.sample(others, self.k - 1)


class MaxMinRule(Rule):
    """
    Greedy MaxMin on reference column vectors: each step adds the reference whose
    minimum correlation-distance to the already-selected set is largest. Maximises
    spread in sample-space. (The legacy ContainmentSelector is this rule on the
    containment feature space.)
    """

    name = "maxmin"

    def _rank(self, M, query_id, candidates):
        dist = _correlation_distance(M[candidates].to_numpy(dtype=float))
        selected = [0]  # query anchor
        remaining = list(range(1, len(candidates)))
        min_dist = dist[0, remaining].copy()
        while len(selected) < self.k and remaining:
            best_pos = int(np.argmax(min_dist))
            best = remaining[best_pos]
            selected.append(best)
            remaining.pop(best_pos)
            min_dist = np.delete(min_dist, best_pos)
            if remaining:
                min_dist = np.minimum(min_dist, dist[best, remaining])
        return [candidates[i] for i in selected]


class GreedyVarRule(Rule):
    """
    Greedy variance maximization: score each candidate by its cross-sample column
    variance (how much its signal differs across samples — its differential
    potential) times its minimum correlation-distance to the already-selected set
    (a redundancy penalty). Query-anchored first.
    """

    name = "greedy_var"

    def _rank(self, M, query_id, candidates):
        X = M[candidates].to_numpy(dtype=float)
        dist = _correlation_distance(X)
        var_scores = X.var(axis=0, ddof=1)
        selected = [0]
        remaining = list(range(1, len(candidates)))
        while len(selected) < self.k and remaining:
            min_dist = dist[np.ix_(remaining, selected)].min(axis=1)
            scores = var_scores[remaining] * min_dist
            best_pos = int(np.argmax(scores))
            selected.append(remaining.pop(best_pos))
        return [candidates[i] for i in selected]


class KMedoidsRule(Rule):
    """
    k-medoids (PAM) clustering of reference column vectors in correlation-distance
    space; the k medoids are the prototypes. Medoids are real references (valid
    alignment targets). The query anchor is forced into the result.
    """

    name = "kmedoids"

    def __init__(self, k: int, threshold: float = 0.0, random_state: int | None = 42):
        super().__init__(k, threshold)
        self.random_state = random_state

    def _rank(self, M, query_id, candidates):
        dist = _correlation_distance(M[candidates].to_numpy(dtype=float))
        rng = np.random.default_rng(self.random_state)
        medoids = _kmedoids(dist, self.k, rng)
        # Force the query anchor (index 0) into the medoid set.
        if 0 not in medoids:
            closest = min(medoids, key=lambda i: dist[0, i])
            medoids[medoids.index(closest)] = 0
        return [candidates[i] for i in medoids]


class CSSRule(Rule):
    """
    Column Subset Selection via greedy pivoted-QR (PLAN §5, NEW headline rule).

    The §6 score measures variance captured by orthogonal projection onto the
    selected columns; CSS *greedily maximises exactly that quantity*. Each step:
    pick the eligible reference whose **current residual** (the part not already
    explained by the chosen set) has the largest norm, then orthogonalize the
    whole matrix against it. "Pick the most informative reference, subtract what
    it explains, repeat on what's left over" — never wastes a pick on a redundant
    reference. Likely the upper bound among cheap rules; the rule matched to the
    scoreboard.

    The residual is taken over the **full** feature matrix (all references as the
    reconstruction target), but only eligible candidates may be *selected*.
    """

    name = "css"

    def _rank(self, M, query_id, candidates):
        cols = list(M.columns)
        pos = {c: i for i, c in enumerate(cols)}
        X = _centered(M.to_numpy(dtype=float))          # samples × all references
        R = X.copy()
        cand_idx = [pos[c] for c in candidates]

        def _orthogonalize(col_idx: int) -> None:
            v = R[:, col_idx]
            nv = float(np.linalg.norm(v))
            if nv <= 1e-12:
                return
            q = v / nv
            R[:] = R - np.outer(q, q @ R)

        selected = [pos[candidates[0]]]                 # query anchor
        _orthogonalize(selected[0])
        chosen = {selected[0]}
        while len(selected) < self.k:
            best, best_norm = None, -1.0
            for j in cand_idx:
                if j in chosen:
                    continue
                norm = float(np.linalg.norm(R[:, j]))
                if norm > best_norm:
                    best_norm, best = norm, j
            if best is None or best_norm <= 1e-12:
                break  # remaining candidates add no new orthogonal signal
            selected.append(best)
            chosen.add(best)
            _orthogonalize(best)
        return [cols[i] for i in selected]


class AllVsAllRule(Rule):
    """
    Ceiling (testing-only): select **every** reference — the O(n²) "map everything
    against everything" line that defines 100% signal at maximum compute. Ignores
    k and the candidate threshold; returns all columns of M, query first.
    """

    name = "all_vs_all"

    def select(self, M, query_id):  # bypass k/threshold entirely
        if query_id not in M.index:
            raise KeyError(f"query_id '{query_id}' not found in feature matrix rows")
        cols = list(M.columns)
        if query_id in cols:
            return [query_id] + [c for c in cols if c != query_id]
        return cols

    def _rank(self, M, query_id, candidates):  # pragma: no cover - select() overridden
        raise NotImplementedError


class ArchetypeRule(Rule):
    """
    Archetype analysis on reference column vectors: find k extreme points spanning
    the convex hull of references in sample-space, then map each to its nearest
    real reference. Spans the diversity rather than summarising it (preferred over
    medoids for differential coverage). Requires the `archetypes` package.
    """

    name = "archetype"

    def __init__(self, k: int, threshold: float = 0.0, random_state: int | None = 42):
        super().__init__(k, threshold)
        self.random_state = random_state

    def _rank(self, M, query_id, candidates):
        try:
            from archetypes import AA
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "ArchetypeRule requires the 'archetypes' package: pip install archetypes"
            ) from exc

        # Each reference is a point in sample-space: rows of Xᵀ (references × samples).
        P = M[candidates].to_numpy(dtype=float).T
        k = min(self.k, len(candidates))
        aa = AA(n_archetypes=k, random_state=self.random_state)
        aa.fit(P)

        selected, used = [], set()
        for arch in aa.archetypes_:
            for idx in np.argsort(np.linalg.norm(P - arch, axis=1)):
                if candidates[idx] not in used:
                    selected.append(candidates[idx])
                    used.add(candidates[idx])
                    break
        # Force the query anchor in.
        if query_id not in selected:
            selected[0] = query_id
        return selected


class TaxonomyStratifiedRule(Rule):
    """
    One representative per GTDB clade, favouring abundant + variable clades
    (PLAN §5; GTDB feature space only — needs taxonomy labels, ✗ for Jaccard /
    containment). Within each clade the representative is the reference with the
    highest cross-sample variance; clades are ranked by their representative's
    (mean signal × variance) and the top-k taken. Query-anchored first.

    Parameters
    ----------
    clades : pd.Series
        reference_id (assembly_id) -> clade label. Required.
    """

    name = "taxonomy_stratified"

    def __init__(self, k: int, threshold: float = 0.0, clades: pd.Series | None = None):
        super().__init__(k, threshold)
        if clades is None:
            raise ValueError(
                "TaxonomyStratifiedRule requires `clades` (reference_id -> clade "
                "label); it is defined on the GTDB feature space only."
            )
        self.clades = clades

    def _rank(self, M, query_id, candidates):
        X = M[candidates].to_numpy(dtype=float)
        var = dict(zip(candidates, X.var(axis=0, ddof=1)))
        mean = dict(zip(candidates, X.mean(axis=0)))

        # Per clade, the highest-variance representative; clade score = its mean×var.
        by_clade: dict[str, str] = {}
        clade_score: dict[str, float] = {}
        for c in candidates:
            clade = self.clades.get(c, c)  # unlabelled refs are their own clade
            if clade not in by_clade or var[c] > var[by_clade[clade]]:
                by_clade[clade] = c
                clade_score[clade] = float(mean[c] * var[c])

        ordered = sorted(by_clade, key=lambda cl: -clade_score[cl])
        reps = [by_clade[cl] for cl in ordered]
        # Query anchor first, then top clades up to k.
        result = [query_id] + [r for r in reps if r != query_id]
        return result[: self.k]


# ──────────────────────────────────────────────────────────────────────────────
# k-medoids (PAM) on a precomputed distance matrix — shared helper
# ──────────────────────────────────────────────────────────────────────────────
def _kmedoids(dist: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """Greedy furthest-first init + PAM swap improvement. Returns medoid indices."""
    n = len(dist)
    k = min(k, n)
    medoids = [int(rng.integers(n))]
    while len(medoids) < k:
        medoids.append(int(np.argmax(dist[:, medoids].min(axis=1))))

    def total_cost(meds):
        return dist[:, meds].min(axis=1).sum()

    improved = True
    while improved:
        improved = False
        for i in range(len(medoids)):
            current = total_cost(medoids)
            for cand in range(n):
                if cand in medoids:
                    continue
                trial = medoids.copy()
                trial[i] = cand
                if total_cost(trial) < current - 1e-10:
                    medoids[i] = cand
                    improved = True
                    break
            if improved:
                break
    return medoids


RULE_REGISTRY: dict[str, type[Rule]] = {
    cls.name: cls
    for cls in (
        RandomRule,
        MaxMinRule,
        KMedoidsRule,
        ArchetypeRule,
        GreedyVarRule,
        CSSRule,
        AllVsAllRule,
        TaxonomyStratifiedRule,
    )
}

__all__ = [
    "Rule",
    "RandomRule",
    "MaxMinRule",
    "KMedoidsRule",
    "ArchetypeRule",
    "GreedyVarRule",
    "CSSRule",
    "AllVsAllRule",
    "TaxonomyStratifiedRule",
    "RULE_REGISTRY",
]
