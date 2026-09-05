"""Volatility feature."""

from __future__ import annotations

import pandas as pd


def volatility(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Return rolling window volatility of daily returns."""
    returns = close.pct_change()
    return returns.rolling(window=window).std()
