from __future__ import annotations

import pandas as pd

from src.portfolio import select_portfolio_model
from src.strategies import comprehensive_dynamic_optimization


def test_select_portfolio_model() -> None:
    metrics = pd.DataFrame(
        {
            "Sharpe Ratio": [1.2, 1.1, 0.8],
            "Max Drawdown": [-0.05, -0.10, -0.08],
            "Calmar Ratio": [1.0, 0.8, 0.9],
            "Annual Volatility": [0.15, 0.12, 0.10],
            "Annual Return": [0.30, 0.25, 0.20],
        },
        index=["Model_A", "Model_B", "Model_C"],
    )

    ranking = select_portfolio_model(metrics)

    assert "model" in ranking.columns
    assert ranking.iloc[0]["model"] == "Model_A"
    assert ranking.iloc[0]["Final_Rank"] == 1


def test_comprehensive_dynamic_optimization_shapes() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122],
            "BBB": [50, 49, 50, 52, 51, 53, 55, 58, 60, 61, 62, 63, 65],
        },
        index=pd.date_range("2025-01-01", periods=13, freq="D"),
    )

    results = comprehensive_dynamic_optimization(prices, lookback_window=5, rebalance_freq=2)

    assert isinstance(results, dict)
    for name, (weights, returns) in results.items():
        assert not weights.empty
        assert len(weights.columns) == 2
        assert len(returns) == len(weights)
