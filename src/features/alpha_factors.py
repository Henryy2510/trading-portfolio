"""Module 2: Hybrid Alpha Factory (Fundamental BCTC + Technical/Momentum).

Adheres to WorldQuant Quantitative Research standards:
- Point-in-time calculation with publication lag (Anti-lookahead bias).
- Robust normalization: Winsorization -> Cross-Sectional Z-Score -> Sector Neutralization.
- Factor Information Coefficient (Rank IC) validation with statistical testing (t-stat, p-value).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    FACTOR_WEIGHTS,
    SECTOR_MAPPING,
    WINSORIZATION_LIMITS,
)
from src.data.data_loader import FundamentalData, MarketData


@dataclass
class AlphaEvaluationReport:
    """Summary of factor predictive power metrics."""
    factor_name: str
    mean_ic: float
    std_ic: float
    ic_ir: float
    t_stat: float
    p_value: float


class AlphaFactory:
    """Computes, processes, and neutralizes cross-sectional alpha factors."""

    def __init__(
        self,
        sector_mapping: Optional[Dict[str, str]] = None,
        factor_weights: Optional[Dict[str, float]] = None,
        winsorize_limits: Tuple[float, float] = WINSORIZATION_LIMITS,
    ):
        self.sector_mapping = sector_mapping or SECTOR_MAPPING
        self.factor_weights = factor_weights or FACTOR_WEIGHTS
        self.winsorize_limits = winsorize_limits

    def compute_fundamental_factors(
        self,
        fundamental_data: FundamentalData,
        as_of_dates: pd.DatetimeIndex,
    ) -> Dict[str, pd.DataFrame]:
        """Compute fundamental factors point-in-time without look-ahead bias.

        For each date t, only statements with announced_date <= t are visible.
        Factors generated:
        - revenue_growth_yoy: (Rev_t - Rev_{t-4}) / Rev_{t-4}
        - gross_margin: Gross Profit / Revenue
        - gpa: Gross Profit / Total Assets
        - cost_efficiency: - (SG&A / Revenue)
        """
        df_statements = fundamental_data.statements.copy()
        tickers = sorted(df_statements["ticker"].unique())

        # Sort chronological
        df_statements["announced_date"] = pd.to_datetime(df_statements["announced_date"])
        df_statements["period_end_date"] = pd.to_datetime(df_statements["period_end_date"])
        df_statements.sort_values(by=["ticker", "period_end_date"], inplace=True)

        # Precompute quarterly metrics
        df_statements["rev_yoy"] = df_statements.groupby("ticker")["revenue"].pct_change(periods=4)
        df_statements["gross_margin"] = df_statements["gross_profit"] / df_statements["revenue"]
        df_statements["gpa"] = df_statements["gross_profit"] / df_statements["total_assets"]
        df_statements["cost_efficiency"] = - (df_statements["sga_expense"] / df_statements["revenue"])

        # Construct point-in-time matrices (Dates x Tickers)
        factor_names = ["revenue_growth_yoy", "gross_margin", "gpa", "cost_efficiency"]
        raw_matrices = {f: pd.DataFrame(index=as_of_dates, columns=tickers, dtype=float) for f in factor_names}

        # Vectorized / merge-asof point-in-time lookup
        for tkr in tickers:
            sub = df_statements[df_statements["ticker"] == tkr].dropna(subset=["announced_date"]).sort_values("announced_date")
            if sub.empty:
                continue

            # Create daily series reindexed to as_of_dates
            # Merge_asof logic: for each date in as_of_dates, take latest statement published on or before that date
            ref_df = pd.DataFrame({"date": as_of_dates})
            merged = pd.merge_asof(
                ref_df,
                sub,
                left_on="date",
                right_on="announced_date",
                direction="backward"
            )
            merged.set_index("date", inplace=True)

            raw_matrices["revenue_growth_yoy"][tkr] = merged["rev_yoy"]
            raw_matrices["gross_margin"][tkr] = merged["gross_margin"]
            raw_matrices["gpa"][tkr] = merged["gpa"]
            raw_matrices["cost_efficiency"][tkr] = merged["cost_efficiency"]

        return raw_matrices

    def compute_momentum_factor(
        self,
        prices: pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """Compute rolling 60-day price momentum factor: (P_t - P_{t-w}) / P_{t-w}."""
        return prices.pct_change(window)

    def process_cross_sectional_factor(
        self,
        raw_factor: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply WorldQuant Factor Standardization:

        1. Winsorization (clip outliers at percentiles).
        2. Cross-sectional Z-score standardization (mean 0, unit variance).
        3. Sector Neutralization (subtract sector-level cross-sectional mean).
        """
        processed = raw_factor.copy()
        lower_pct, upper_pct = self.winsorize_limits

        for dt in processed.index:
            row = processed.loc[dt].dropna()
            if len(row) < 3:
                continue

            # 1. Winsorization
            q_low = np.percentile(row.values, lower_pct * 100)
            q_high = np.percentile(row.values, upper_pct * 100)
            winsorized = np.clip(row.values, q_low, q_high)

            # 2. Cross-sectional Z-Score
            mu = np.mean(winsorized)
            sigma = np.std(winsorized)
            if sigma > 1e-8:
                z_scores = (winsorized - mu) / sigma
            else:
                z_scores = winsorized - mu

            z_series = pd.Series(z_scores, index=row.index)

            # 3. Sector Neutralization
            sectors = pd.Series([self.sector_mapping.get(tkr, "Other") for tkr in row.index], index=row.index)
            sector_means = z_series.groupby(sectors).transform("mean")
            neutralized = z_series - sector_means

            processed.loc[dt, row.index] = neutralized

        return processed

    def construct_composite_alpha(
        self,
        market_data: MarketData,
        fundamental_data: FundamentalData,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Construct weighted multi-factor composite alpha matrix (T x N)."""
        dates = market_data.prices.index

        # 1. Compute individual factors
        factor_dict = self.compute_fundamental_factors(fundamental_data, dates)
        factor_dict["price_momentum_60d"] = self.compute_momentum_factor(market_data.prices, window=60)

        # 2. Process and standardize each factor
        processed_factors = {}
        composite_alpha = pd.DataFrame(0.0, index=dates, columns=market_data.prices.columns)

        for fname, raw_df in factor_dict.items():
            processed = self.process_cross_sectional_factor(raw_df)
            processed_factors[fname] = processed
            w = self.factor_weights.get(fname, 0.0)
            composite_alpha = composite_alpha.add(processed.fillna(0.0) * w)

        return composite_alpha, processed_factors

    @staticmethod
    def evaluate_factor_ic(
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_name: str = "Factor",
    ) -> AlphaEvaluationReport:
        """Evaluate factor Information Coefficient (Rank IC) against forward returns."""
        common_dates = factor.index.intersection(forward_returns.index)
        ic_series = []

        for dt in common_dates:
            f_vals = factor.loc[dt].dropna()
            r_vals = forward_returns.loc[dt].dropna()
            overlap = f_vals.index.intersection(r_vals.index)

            if len(overlap) >= 4:
                # Spearman rank correlation
                ic, _ = stats.spearmanr(f_vals.loc[overlap], r_vals.loc[overlap])
                if not np.isnan(ic):
                    ic_series.append(ic)

        ic_arr = np.array(ic_series)
        if len(ic_arr) == 0:
            return AlphaEvaluationReport(factor_name, 0.0, 0.0, 0.0, 0.0, 1.0)

        mean_ic = float(np.mean(ic_arr))
        std_ic = float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 1e-4
        ic_ir = mean_ic / (std_ic + 1e-8)
        t_stat, p_val = stats.ttest_1samp(ic_arr, popmean=0.0)

        return AlphaEvaluationReport(
            factor_name=factor_name,
            mean_ic=mean_ic,
            std_ic=std_ic,
            ic_ir=ic_ir,
            t_stat=float(t_stat),
            p_value=float(p_val),
        )
