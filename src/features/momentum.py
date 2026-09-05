"""Momentum feature."""

from __future__ import annotations

import pandas as pd


def momentum(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Return percentage change over a rolling window."""
    return close.pct_change(periods=window)
