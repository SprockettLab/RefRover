"""
Quality-weighted MAG counting from a CheckM2 report (Tier-2, PLAN.md §8).

Turns a CheckM2 ``quality_report.tsv`` (one row per bin, with Completeness and
Contamination) into the MAG-yield numbers the Tier-2 benchmark compares methods
on. Tiers follow MIMAG-style thresholds and reuse the §6 high/medium weighting
(strong MAGs worth more), so the downstream score is directly comparable in
spirit to the upstream ``tiered_axis_count`` proxy we are trying to validate.
"""

from __future__ import annotations

import pandas as pd

# (min completeness, max contamination) for each tier.
HIGH = (90.0, 5.0)
MEDIUM = (50.0, 10.0)
HIGH_WEIGHT = 2.0
MEDIUM_WEIGHT = 1.0


def count_mags(
    quality: pd.DataFrame,
    *,
    high: tuple[float, float] = HIGH,
    medium: tuple[float, float] = MEDIUM,
    high_weight: float = HIGH_WEIGHT,
    medium_weight: float = MEDIUM_WEIGHT,
    completeness_col: str = "Completeness",
    contamination_col: str = "Contamination",
) -> dict:
    """
    Count quality-weighted MAGs from a CheckM2 quality table.

    A bin is **high** when completeness ≥ high[0] and contamination ≤ high[1];
    **medium** when completeness ≥ medium[0] and contamination ≤ medium[1] (and
    not already high); **low** otherwise. The headline is
    ``weighted_mags = high_weight·n_high + medium_weight·n_medium`` (low counts 0),
    mirroring the §6 tiered weighting so it lines up with the upstream proxy.

    Returns ``{"n_bins", "n_high", "n_medium", "n_low", "weighted_mags"}``.
    An empty table yields all zeros.
    """
    n_bins = int(len(quality))
    if n_bins == 0:
        return {
            "n_bins": 0, "n_high": 0, "n_medium": 0, "n_low": 0,
            "weighted_mags": 0.0, "sum_qs": 0.0,
        }

    comp = quality[completeness_col].to_numpy(dtype=float)
    cont = quality[contamination_col].to_numpy(dtype=float)

    is_high = (comp >= high[0]) & (cont <= high[1])
    is_medium = (~is_high) & (comp >= medium[0]) & (cont <= medium[1])
    n_high = int(is_high.sum())
    n_medium = int(is_medium.sum())
    n_low = n_bins - n_high - n_medium

    # Quality Score: completeness − 5 × contamination (dRep / Olm et al. 2017).
    # Penalises contamination 5× more than incompleteness; captures continuous
    # variation within and between MIMAG tiers. Only bins above the medium floor
    # (QS > 0) contribute — a negative QS bin adds nothing useful downstream.
    qs = comp - 5.0 * cont
    sum_qs = float(qs[qs > 0].sum())

    return {
        "n_bins": n_bins,
        "n_high": n_high,
        "n_medium": n_medium,
        "n_low": n_low,
        "weighted_mags": high_weight * n_high + medium_weight * n_medium,
        "sum_qs": sum_qs,
    }


__all__ = ["count_mags", "HIGH", "MEDIUM", "HIGH_WEIGHT", "MEDIUM_WEIGHT"]
