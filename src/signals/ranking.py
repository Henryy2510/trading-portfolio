"""Cross-sectional ranking helpers."""

from __future__ import annotations

import pandas as pd


def cross_sectional_rank(feature: pd.DataFrame) -> pd.DataFrame:
    """Rank features row-wise to [0, 1] by cross-sectional strength."""
    return feature.rank(axis=1, pct=True)
