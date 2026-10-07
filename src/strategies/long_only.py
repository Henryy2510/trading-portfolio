"""Long-only strategy rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def top_quantile(signal: pd.DataFrame, q: float = 0.2) -> pd.DataFrame:
    """Select rows where signal is in top quantile."""
    if not (0 < q <= 1):
        raise ValueError("q must be between 0 and 1.")
    if q == 1:
        return signal.notna()

    threshold = signal.quantile(1 - q, axis=1)
    selected = signal.ge(threshold, axis=0)
    return selected.fillna(False) & signal.notna()


def _to_vector(x: pd.Series | np.ndarray | list[float], n: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if n is not None and arr.shape[0] != n:
        raise ValueError("Size mismatch for optimization input.")
    return arr


def optimize_max_sharpe_constrained(
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    rf_rate: float = 0.02,
    max_weight: float = 0.4,
    min_weight: float = 0.05,
) -> np.ndarray:
    """Optimize portfolio for max Sharpe ratio with box constraints."""
    mu_vec = _to_vector(mu, n=None)
    cov_mat = np.asarray(cov, dtype=float)
    n = len(mu_vec)

    if cov_mat.shape != (n, n):
        raise ValueError("cov must be a square matrix matching mu size")

    if cov_mat.size == 0:
        return np.array([1 / n] * n)

    def objective(weights: np.ndarray) -> float:
        portfolio_return = float(np.dot(weights, mu_vec))
        variance = float(np.dot(weights.T, np.dot(cov_mat, weights)))
        if variance <= 0:
            return 1e9
        portfolio_vol = float(np.sqrt(variance))
        return -(portfolio_return - rf_rate) / portfolio_vol

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    bounds = [(min_weight, max_weight) for _ in range(n)]
    x0 = np.array([1 / n] * n)

    try:
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if result.success and np.isfinite(result.fun):
            weights = _to_vector(result.x, n=n)
            if np.all(np.isfinite(weights)) and abs(weights.sum() - 1.0) <= 1e-4:
                return weights / weights.sum()
    except Exception:
        pass

    return np.array([1 / n] * n)


def dynamic_portfolio_optimization(
    data: pd.DataFrame,
    lookback_window: int = 63,
    rebalance_freq: int = 21,
    rf_rate: float = 0.02,
    max_weight: float = 0.4,
    min_weight: float = 0.05,
    trading_days_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run rolling max-Sharpe optimization with weight bounds."""
    returns = data.pct_change().dropna()
    n_assets = len(data.columns)
    weights_history = pd.DataFrame(index=returns.index, columns=data.columns, dtype=float)
    portfolio_returns = pd.Series(index=returns.index, dtype=float)

    if n_assets == 0 or returns.empty:
        return weights_history.dropna(how="all"), portfolio_returns.dropna()

    if lookback_window <= 0 or rebalance_freq <= 0:
        raise ValueError("lookback_window and rebalance_freq must be positive.")

    current_weights = np.array([1 / n_assets] * n_assets)
    for i in range(lookback_window, len(returns)):
        current_date = returns.index[i]
        if i % rebalance_freq == 0 or i == lookback_window:
            hist_returns = returns.iloc[i - lookback_window : i]
            mu = hist_returns.mean() * trading_days_per_year
            cov = hist_returns.cov() * trading_days_per_year

            try:
                current_weights = optimize_max_sharpe_constrained(
                    mu=mu,
                    cov=cov,
                    rf_rate=rf_rate,
                    max_weight=max_weight,
                    min_weight=min_weight,
                )
            except Exception:
                current_weights = np.array([1 / n_assets] * n_assets)

        weights_history.loc[current_date] = current_weights
        portfolio_returns.loc[current_date] = float(np.dot(current_weights, returns.iloc[i]))

    return weights_history.dropna(), portfolio_returns.dropna()


def kelly_criterion_optimization(mu: pd.Series | np.ndarray, cov: pd.DataFrame | np.ndarray, confidence_level: float = 0.6) -> np.ndarray:
    """Estimate Kelly-like weights using return/risk ratio proxy."""
    mu_vec = _to_vector(mu, n=None)
    volatilities = np.sqrt(np.diag(np.asarray(cov, dtype=float)))
    n_assets = len(mu_vec)
    kelly_weights = np.zeros(n_assets)

    for i in range(n_assets):
        if volatilities[i] > 0 and mu_vec[i] > 0:
            sharpe = mu_vec[i] / volatilities[i]
            win_prob = min(0.5 + sharpe * 0.1, 0.85)
            odds_ratio = mu_vec[i] / volatilities[i]
            kelly_fraction = (odds_ratio * win_prob - (1 - win_prob)) / odds_ratio
            kelly_weights[i] = max(0.0, kelly_fraction * confidence_level)

    if kelly_weights.sum() > 0:
        return kelly_weights / kelly_weights.sum()
    return np.array([1 / n_assets] * n_assets)


def dynamic_kelly_optimization(
    data: pd.DataFrame,
    lookback_window: int = 63,
    rebalance_freq: int = 21,
    confidence_level: float = 0.6,
    trading_days_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run rolling Kelly-based optimization."""
    returns = data.pct_change().dropna()
    n_assets = len(data.columns)
    weights_history = pd.DataFrame(index=returns.index, columns=data.columns, dtype=float)
    portfolio_returns = pd.Series(index=returns.index, dtype=float)

    if n_assets == 0 or returns.empty:
        return weights_history.dropna(how="all"), portfolio_returns.dropna()

    if lookback_window <= 0 or rebalance_freq <= 0:
        raise ValueError("lookback_window and rebalance_freq must be positive.")

    current_weights = np.array([1 / n_assets] * n_assets)
    for i in range(lookback_window, len(returns)):
        current_date = returns.index[i]
        if i % rebalance_freq == 0 or i == lookback_window:
            hist_returns = returns.iloc[i - lookback_window : i]
            mu = hist_returns.mean() * trading_days_per_year
            cov = hist_returns.cov() * trading_days_per_year
            try:
                current_weights = kelly_criterion_optimization(mu, cov, confidence_level=confidence_level)
            except Exception:
                current_weights = np.array([1 / n_assets] * n_assets)

        weights_history.loc[current_date] = current_weights
        portfolio_returns.loc[current_date] = float(np.dot(current_weights, returns.iloc[i]))

    return weights_history.dropna(), portfolio_returns.dropna()


def risk_parity_optimization(
    cov: pd.DataFrame | np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
    learning_rate: float = 0.05,
) -> np.ndarray:
    """Iterative risk-parity solver with normalization and floor constraints."""
    cov_mat = np.asarray(cov, dtype=float)
    n_assets = cov_mat.shape[0]
    if cov_mat.shape != (n_assets, n_assets):
        raise ValueError("cov must be square")

    weights = np.array([1 / n_assets] * n_assets)
    for _ in range(max_iterations):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
        if portfolio_vol <= 0:
            break

        marginal_risk = np.dot(cov_mat, weights) / portfolio_vol
        risk_contributions = weights * marginal_risk
        target_risk = portfolio_vol ** 2 / n_assets
        risk_diff = risk_contributions - target_risk

        if np.max(np.abs(risk_diff)) < tolerance:
            break

        weight_updates = -learning_rate * risk_diff / np.where(marginal_risk == 0, 1e-12, marginal_risk)
        weights = np.maximum(weights + weight_updates, 1e-8)
        weights = weights / weights.sum()

    return weights


def dynamic_risk_parity_optimization(
    data: pd.DataFrame,
    lookback_window: int = 63,
    rebalance_freq: int = 21,
    trading_days_per_year: int = 252,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run rolling risk parity optimization."""
    returns = data.pct_change().dropna()
    n_assets = len(data.columns)
    weights_history = pd.DataFrame(index=returns.index, columns=data.columns, dtype=float)
    portfolio_returns = pd.Series(index=returns.index, dtype=float)

    if n_assets == 0 or returns.empty:
        return weights_history.dropna(how="all"), portfolio_returns.dropna()

    if lookback_window <= 0 or rebalance_freq <= 0:
        raise ValueError("lookback_window and rebalance_freq must be positive.")

    current_weights = np.array([1 / n_assets] * n_assets)
    for i in range(lookback_window, len(returns)):
        current_date = returns.index[i]
        if i % rebalance_freq == 0 or i == lookback_window:
            hist_returns = returns.iloc[i - lookback_window : i]
            cov = hist_returns.cov() * trading_days_per_year

            try:
                eigenvals = np.linalg.eigvals(cov.fillna(0).to_numpy() if isinstance(cov, pd.DataFrame) else np.asarray(cov))
                if np.all(np.real(eigenvals) > 0):
                    current_weights = risk_parity_optimization(cov)
                else:
                    corr = hist_returns.corr()
                    vol_avg = hist_returns.std().mean() * np.sqrt(trading_days_per_year)
                    cov_regularized = corr * (vol_avg ** 2)
                    current_weights = risk_parity_optimization(cov_regularized)
            except Exception:
                current_weights = np.array([1 / n_assets] * n_assets)

        weights_history.loc[current_date] = current_weights
        portfolio_returns.loc[current_date] = float(np.dot(current_weights, returns.iloc[i]))

    return weights_history.dropna(), portfolio_returns.dropna()


def comprehensive_dynamic_optimization(
    data: pd.DataFrame,
    lookback_window: int = 63,
    rebalance_freq: int = 21,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Run all implemented dynamic strategies for easy comparison."""
    methods = {
        "Max_Sharpe_Dynamic": dynamic_portfolio_optimization(data, lookback_window, rebalance_freq),
        "Kelly_Dynamic": dynamic_kelly_optimization(data, lookback_window, rebalance_freq),
        "Risk_Parity_Dynamic": dynamic_risk_parity_optimization(data, lookback_window, rebalance_freq),
    }
    return methods
