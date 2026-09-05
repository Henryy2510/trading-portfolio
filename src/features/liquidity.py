"""Liquidity-related feature."""

from __future__ import annotations

import pandas as pd


def volume_ratio(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Return 20-day volume surprise ratio."""
    return volume / volume.rolling(window=window).mean()
