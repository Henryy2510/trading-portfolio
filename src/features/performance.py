"""Portfolio-level feature calculations."""

from __future__ import annotations

import pandas as pd


def _safe_expected_returns(close: pd.DataFrame, frequency: int = 252) -> pd.Series:
    """Return expected returns with optional pypfopt fallback."""
    returns = close.pct_change().dropna()

    if returns.empty:
        return pd.Series(dtype=float)

    try:
        from pypfopt import expected_returns as pypfopt_expected_returns
    except ModuleNotFoundError:
        return returns.mean() * frequency

    return pypfopt_expected_returns.mean_historical_return(close, frequency=frequency)


def _safe_sample_cov(close: pd.DataFrame, frequency: int = 252) -> pd.DataFrame:
    """Return sample covariance with optional pypfopt fallback."""
    returns = close.pct_change().dropna()

    if returns.empty:
        return pd.DataFrame()

    try:
        from pypfopt import risk_models
    except ModuleNotFoundError:
        return returns.cov() * frequency

    return risk_models.sample_cov(close, frequency=frequency)


def expected_returns_and_covariance(
    close: pd.DataFrame,
    *,
    frequency: int = 252,
    dropna: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate mean historical returns and sample covariance from close matrix."""
    if close.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    close_clean = close.copy()
    if dropna:
        close_clean = close_clean.dropna(how="all")

    mu = _safe_expected_returns(close_clean, frequency=frequency)
    cov = _safe_sample_cov(close_clean, frequency=frequency)
    return mu, cov


def rolling_correlation(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling return correlation matrix statistics for exploration."""
    returns = close.pct_change().dropna()
    if returns.empty:
        return pd.DataFrame()

    return returns.rolling(window=window).corr(pairwise=True).sort_index()

