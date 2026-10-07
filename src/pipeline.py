"""End-to-end workflow primitives for the VNQuant project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from config import DEFAULT_TICKERS, INITIAL_CASH, MODEL_SELECTION_WEIGHTS, RISK_FREE_RATE
from src.data.provider import split_train_test
from src.metrics.performance import compute_comprehensive_metrics
from src.portfolio.weighting import best_portfolio_model, select_portfolio_model
from src.strategies.long_only import comprehensive_dynamic_optimization


@dataclass
class DynamicWorkflowResult:
    """Result container for dynamic model workflow."""

    train_prices: pd.DataFrame
    test_prices: pd.DataFrame
    optimization_results: dict[str, tuple[pd.DataFrame, pd.Series]]
    model_metrics: pd.DataFrame
    ranked_models: pd.DataFrame
    best_model: str
    best_callable: Callable[..., tuple[pd.DataFrame, pd.Series]] | None
    backtest_report: dict[str, float] | None = None


def run_dynamic_workflow(
    close_prices: pd.DataFrame,
    split_date: str,
    *,
    lookback_window: int = 63,
    rebalance_freq: int = 21,
    run_backtest: bool = False,
    fees: float = 0.0,
    slippage: float = 0.0,
    initial_cash: float = INITIAL_CASH,
    rf_rate: float = RISK_FREE_RATE,
    model_weights: dict[str, float] | None = None,
) -> DynamicWorkflowResult:
    """Run the full dynamic strategy comparison workflow."""
    if model_weights is None:
        model_weights = MODEL_SELECTION_WEIGHTS

    train_prices, test_prices = split_train_test(close_prices, split_date)

    optimization_results = comprehensive_dynamic_optimization(
        train_prices,
        lookback_window=lookback_window,
        rebalance_freq=rebalance_freq,
    )

    metrics = {
        name: compute_comprehensive_metrics(returns, rf_rate=rf_rate)
        for name, (_, returns) in optimization_results.items()
    }
    model_metrics = pd.DataFrame(metrics).T
    ranked_models = select_portfolio_model(model_metrics, weights=model_weights)

    best_model_name, best_callable = best_portfolio_model(ranked_models)
    if best_callable is None:
        raise RuntimeError(f"No implementation found for best model: {best_model_name}")

    best_weights, _ = optimization_results[best_model_name]
    backtest_report = None

    if run_backtest:
        _, backtest_report = _run_backtest(test_prices, best_weights, fees, slippage, initial_cash)
        # Keep a tiny sanity hook for notebook parity:
        if isinstance(backtest_report, dict):
            backtest_report["best_model"] = best_model_name

    return DynamicWorkflowResult(
        train_prices=train_prices,
        test_prices=test_prices,
        optimization_results=optimization_results,
        model_metrics=model_metrics,
        ranked_models=ranked_models,
        best_model=best_model_name,
        best_callable=best_callable,
        backtest_report=backtest_report,
    )


def _run_backtest(prices: pd.DataFrame, weights: pd.DataFrame, fees: float, slippage: float, initial_cash: float) -> tuple[object, dict[str, float]]:
    from src.backtest.engine import run_backtest

    return run_backtest(
        prices=prices,
        weights=weights,
        fees=fees,
        slippage=slippage,
        initial_cash=initial_cash,
    )


def get_default_universe(tickers: list[str] | None = None) -> list[str]:
    """Return default asset list used by notebooks and local experimentation."""
    return [ticker for ticker in (tickers or DEFAULT_TICKERS)]
