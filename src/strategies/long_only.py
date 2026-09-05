"""Long-only strategy rules."""

from __future__ import annotations

import pandas as pd


def top_quantile(signal: pd.DataFrame, q: float = 0.2) -> pd.DataFrame:
    """Select rows where signal is in top quantile."""
    if not (0 < q <= 1):
        raise ValueError("q must be between 0 and 1.")
    if q == 1:
        return signal.notna()

    threshold = signal.quantile(1 - q, axis=1)
    selected = signal.ge(threshold, axis=0)
    return selected.fillna(False) & signal.notna()
