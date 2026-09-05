"""VectorBT backtest wrapper."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from src.metrics.performance import compute_basic_report


def _normalize_inputs(close: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = close.sort_index().sort_index(axis=1).astype(float).sort_index()
    weights_aligned = weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    return prices, weights_aligned


def _try_vectorbt_orders(portfolio_cls, close: pd.DataFrame, weights: pd.DataFrame, fees: float, slippage: float, initial_cash: float):
    close = close.fillna(method="ffill").fillna(method="bfill")

    order_calls = [
        dict(size=weights, size_type="targetpercent", direction=1, init_cash=initial_cash, fees=fees, slippage=slippage),
        dict(size=weights, size_type="target_percent", direction=1, init_cash=initial_cash, fees=fees, slippage=slippage),
        dict(size=weights, size_type="percent", direction=1, init_cash=initial_cash, fees=fees, slippage=slippage),
        dict(size=weights, size_type="value", direction=1, init_cash=initial_cash, fees=fees, slippage=slippage),
    ]

    for kwargs in order_calls:
        try:
            return portfolio_cls.from_orders(close=close, **kwargs)
        except TypeError:
            continue

    raise RuntimeError("No supported from_orders signature found for this vectorbt version.")


def _try_vectorbt_signals(portfolio_cls, close: pd.DataFrame, weights: pd.DataFrame, fees: float, slippage: float, initial_cash: float):
    entries = (weights > 0).astype(int)
    exits = entries.ne(entries.shift(1).fillna(0)).astype(int)
    signal_calls = [
        dict(entries=entries, exits=exits, size=weights, direction=1, init_cash=initial_cash, fees=fees, slippage=slippage),
        dict(entries=entries, exits=exits, size=weights.fillna(0.0), direction=1, init_cash=initial_cash, fees=fees),
        dict(entries=entries, exits=exits, size=weights.fillna(0.0), direction=1),
    ]

    for kwargs in signal_calls:
        try:
            return portfolio_cls.from_signals(close=close, **kwargs)
        except TypeError:
            continue

    raise RuntimeError("No supported from_signals signature found for this vectorbt version.")


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    fees: float = 0.0,
    slippage: float = 0.0,
    initial_cash: float = 100_000.0,
    frequency: Literal["daily", "weekly"] = "daily",
) -> tuple[object, dict[str, float]]:
    """
    Run vectorbt backtest from target weights.

    Returns:
        portfolio (vectorbt object), report (dict)
    """
    prices, weights = _normalize_inputs(prices, weights)

    try:
        import vectorbt as vbt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "vectorbt is not installed. Install vectorbt to run backtests."
        ) from exc

    try:
        portfolio = vbt.Portfolio
    except AttributeError as exc:
        raise RuntimeError("vectorbt.Portfolio is unavailable in installed vectorbt version.") from exc

    errors: list[Exception] = []
    for runner in (_try_vectorbt_orders, _try_vectorbt_signals):
        try:
            portfolio_obj = runner(portfolio, prices, weights, fees, slippage, initial_cash)
            break
        except Exception as exc:  # pragma: no cover - dependency behavior dependent
            errors.append(exc)
            portfolio_obj = None
            continue
    else:
        raise RuntimeError("Could not create vectorbt portfolio with known API variants.") from errors[-1]

    if frequency == "weekly":
        # vectorbt accepts freq only through index; use weekly close index as proxy.
        pass

    report = compute_basic_report(portfolio_obj, prices, weights, initial_cash)
    return portfolio_obj, report
