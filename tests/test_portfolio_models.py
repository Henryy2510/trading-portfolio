from __future__ import annotations

import pandas as pd

import src.pipeline as pipeline
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


def test_dynamic_workflow_recomputes_test_weights_with_train_warmup(monkeypatch) -> None:
    dates = pd.bdate_range("2025-01-01", periods=120)
    prices = pd.DataFrame(
        {
            "AAA": [100 + i + (i % 4) for i in range(len(dates))],
            "BBB": [80 + i * 0.7 + (i % 3) for i in range(len(dates))],
        },
        index=dates,
    )
    captured: dict[str, pd.DataFrame] = {}

    def optimizer(data: pd.DataFrame, lookback_window: int, rebalance_freq: int):
        index = data.pct_change().dropna().index
        weights = pd.DataFrame(0.5, index=index, columns=data.columns)
        returns = pd.Series([0.01 if i % 2 else -0.005 for i in range(len(index))], index=index)
        return weights, returns

    def fake_backtest(test_prices, weights, fees, slippage, initial_cash):
        captured["prices"] = test_prices
        captured["weights"] = weights
        return None, {"total_return": 0.01}

    monkeypatch.setattr(
        pipeline,
        "comprehensive_dynamic_optimization",
        lambda data, **kwargs: {"Mock_Dynamic": optimizer(data, **kwargs)},
    )
    monkeypatch.setattr(pipeline, "best_portfolio_model", lambda _ranked: ("Mock_Dynamic", optimizer))
    monkeypatch.setattr(pipeline, "_run_backtest", fake_backtest)

    result = pipeline.run_dynamic_workflow(
        prices,
        split_date=dates[90].strftime("%Y-%m-%d"),
        lookback_window=10,
        rebalance_freq=5,
        run_backtest=True,
    )

    assert result.backtest_report["best_model"] == "Mock_Dynamic"
    assert captured["weights"].index.equals(result.test_prices.index)
    assert captured["weights"].notna().all().all()
    assert (captured["weights"].sum(axis=1) > 0).any()
