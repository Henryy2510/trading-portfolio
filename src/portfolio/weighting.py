"""Portfolio weighting utilities."""

from __future__ import annotations

import pandas as pd


def equal_weight(selection: pd.DataFrame) -> pd.DataFrame:
    """Equal weight selected long-only positions."""
    selected = selection.fillna(False).astype(bool)
    selected_count = selected.sum(axis=1).replace(0, pd.NA)
    weights = selected.div(selected_count, axis=0)
    return weights.fillna(0.0)


def rebalance_weights(weights: pd.DataFrame, frequency: str = "daily") -> pd.DataFrame:
    """Simple rebalance schedule wrapper.

    - daily: returns input unchanged.
    - weekly: rebalance on weekly index boundary and hold weights between rebalance days.
    """
    if not weights.empty and frequency == "weekly":
        rebalance_dates = weights.groupby(weights.index.to_period("W")).tail(1).index
        mask = weights.index.isin(rebalance_dates)
        return weights.where(mask).ffill().fillna(0.0)

    return weights.fillna(0.0)


def select_portfolio_model(metrics_df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Score/ rank multiple portfolio strategy performance rows and return ranked frame."""
    if weights is None:
        weights = {
            "sharpe_ratio": 0.20,
            "max_drawdown": 0.30,
            "calmar_ratio": 0.30,
            "volatility": 0.20,
        }

    required = {"Sharpe Ratio", "Max Drawdown", "Calmar Ratio", "Annual Volatility"}
    if not required.issubset(metrics_df.columns):
        missing = sorted(required - set(metrics_df.columns))
        raise ValueError(f"Missing metrics columns: {missing}")

    scoring_df = pd.DataFrame(index=metrics_df.index)

    sharpe_ranks = metrics_df["Sharpe Ratio"].rank(ascending=False, method="dense")
    drawdown_ranks = metrics_df["Max Drawdown"].rank(ascending=True, method="dense")
    calmar_ranks = metrics_df["Calmar Ratio"].rank(ascending=False, method="dense")
    volatility_ranks = metrics_df["Annual Volatility"].rank(ascending=True, method="dense")

    def rank_to_score(rank: pd.Series) -> pd.Series:
        return rank.apply(lambda r: 3.0 if r <= 1 else (2.0 if r <= 2 else 1.0))

    scoring_df["Sharpe_Score"] = rank_to_score(sharpe_ranks)
    scoring_df["Drawdown_Score"] = rank_to_score(drawdown_ranks)
    scoring_df["Calmar_Score"] = rank_to_score(calmar_ranks)
    scoring_df["Volatility_Score"] = rank_to_score(volatility_ranks)

    scoring_df["Total_Score"] = (
        scoring_df["Sharpe_Score"] * weights["sharpe_ratio"]
        + scoring_df["Drawdown_Score"] * weights["max_drawdown"]
        + scoring_df["Calmar_Score"] * weights["calmar_ratio"]
        + scoring_df["Volatility_Score"] * weights["volatility"]
    )

    result = pd.DataFrame(index=metrics_df.index)
    result["Sharpe_Ratio"] = metrics_df["Sharpe Ratio"]
    result["Max_Drawdown"] = metrics_df["Max Drawdown"]
    result["Calmar_Ratio"] = metrics_df["Calmar Ratio"]
    result["Volatility"] = metrics_df["Annual Volatility"]
    result["Annual_Return"] = metrics_df["Annual Return"]
    result["Final_Rank"] = scoring_df["Total_Score"].rank(ascending=False, method="dense").astype(int)

    model_index_name = metrics_df.index.name or "model"
    return result.sort_values("Final_Rank").reset_index(names=[model_index_name]).rename(columns={model_index_name: "model"})


def calculate_best_portfolio_model(metrics_df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Backward-compatible alias for notebook naming.

    The original notebook exposes ``calculate_best_portfolio_model``; keep both
    entry points to minimize migration friction.
    """
    return select_portfolio_model(metrics_df, weights=weights)


def resolve_model_callable(model_name: str):
    """Map a ranked model name to a strategy function.

    Import is deferred to this function to avoid circular imports from strategy modules.
    """
    from src.strategies.long_only import (  # local import to avoid circular dependency
        dynamic_portfolio_optimization,
        dynamic_kelly_optimization,
        dynamic_risk_parity_optimization,
    )

    model_map = {
        "Risk_Parity_Dynamic": dynamic_risk_parity_optimization,
        "Kelly_Dynamic": dynamic_kelly_optimization,
        "Max_Sharpe_Dynamic": dynamic_portfolio_optimization,
    }
    return model_map.get(model_name)


def best_portfolio_model(ranked_models: pd.DataFrame):
    """Return the highest-ranked model name and callable."""
    if ranked_models.empty:
        raise ValueError("ranked_models is empty")

    if "model" not in ranked_models.columns:
        raise ValueError("ranked_models must include a 'model' column.")

    best_model_name = ranked_models.iloc[0]["model"]
    return best_model_name, resolve_model_callable(best_model_name)
