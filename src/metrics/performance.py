"""Standardized portfolio performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _returns_from_weights(close: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    aligned_weights = weights.reindex(index=close.index, columns=close.columns).fillna(0.0)
    asset_returns = close.pct_change().fillna(0.0)
    return (asset_returns * aligned_weights.shift(1).fillna(0.0)).sum(axis=1)


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def compute_basic_report(
    portfolio_obj: object | None,
    close: pd.DataFrame,
    weights: pd.DataFrame,
    initial_cash: float,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    """Compute a standardized metric dict for V0."""
    returns = _returns_from_weights(close, weights)
    returns = returns[returns.notna()]
    if returns.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "number_of_trades": 0.0,
            "turnover": 0.0,
            "exposure": 0.0,
            "ending_portfolio_value": float(initial_cash),
        }

    equity = (1 + returns).cumprod() * initial_cash
    days = max(len(returns), 1)
    total_return = float(equity.iloc[-1] / initial_cash - 1.0)
    cagr = float((1.0 + total_return) ** (trading_days_per_year / days) - 1.0)
    annualized_vol = float(returns.std(ddof=0) * np.sqrt(trading_days_per_year))
    sharpe = float(returns.mean() / (returns.std(ddof=0) + 1e-12) * np.sqrt(trading_days_per_year))
    win_rate = float((returns > 0).mean())
    turns = weights.diff().abs().sum(axis=1).fillna(0.0)
    exposure = float((weights.sum(axis=1) / 1.0).clip(0, 1).mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "win_rate": win_rate,
        "number_of_trades": float((weights != 0).sum().sum()),
        "turnover": float(turns.mean()),
        "exposure": exposure,
        "ending_portfolio_value": float(equity.iloc[-1]),
    }


def compute_comprehensive_metrics(returns: pd.Series, rf_rate: float = 0.02) -> pd.Series:
    """Calculate broad strategy-level metrics used in the notebook comparison."""
    if returns.empty or returns.std() == 0:
        return pd.Series(
            {
                "Annual Return": 0,
                "Annual Volatility": 0,
                "Sharpe Ratio": 0,
                "Max Drawdown": 0,
                "Calmar Ratio": 0,
                "Sortino Ratio": 0,
                "Skewness": 0,
                "Kurtosis": 0,
                "VaR (95%)": 0,
                "CVaR (95%)": 0,
            }
        )

    returns = returns.dropna()
    annual_return = (1 + returns.mean()) ** 252 - 1
    annual_volatility = returns.std() * (252 ** 0.5)
    sharpe_ratio = (annual_return - rf_rate) / annual_volatility if annual_volatility != 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * (252 ** 0.5) if len(negative_returns) > 0 else 0
    sortino_ratio = (annual_return - rf_rate) / downside_deviation if downside_deviation != 0 else 0

    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()

    return pd.Series(
        {
            "Annual Return": annual_return,
            "Annual Volatility": annual_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar_ratio,
            "Sortino Ratio": sortino_ratio,
            "Skewness": returns.skew(),
            "Kurtosis": returns.kurtosis(),
            "VaR (95%)": var_95,
            "CVaR (95%)": cvar_95,
        }
    )
