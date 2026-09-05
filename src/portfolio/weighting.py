"""Portfolio weighting utilities."""

from __future__ import annotations

import pandas as pd


def equal_weight(selection: pd.DataFrame) -> pd.DataFrame:
    """Equal weight selected long-only positions."""
    selected = selection.fillna(False).astype(bool)
    selected_count = selected.sum(axis=1).replace(0, pd.NA)
    weights = selected.div(selected_count, axis=0)
    return weights.fillna(0.0)


def rebalance_weights(weights: pd.DataFrame, frequency: str = "daily") -> pd.DataFrame:
    """Simple rebalance schedule wrapper.

    - daily: returns input unchanged.
    - weekly: rebalance on weekly index boundary and hold weights between rebalance days.
    """
    if not weights.empty and frequency == "weekly":
        rebalance_dates = weights.groupby(weights.index.to_period("W")).tail(1).index
        mask = weights.index.isin(rebalance_dates)
        return weights.where(mask).ffill().fillna(0.0)

    return weights.fillna(0.0)
