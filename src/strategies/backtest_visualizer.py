"""Module 5: Institutional Backtest Engine & 4-Panel Investor Visualization Suite.

Simulates Vietnam equity microstructure (T+2.5, round-trip fees/taxes, cash yield).
Produces institutional-grade metrics and the required 4-Panel Investor Dashboard.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    ANNUAL_PROFIT_TARGET_VND,
    INITIAL_CAPITAL_VND,
    REPORTS_DIR,
    RISK_FREE_RATE,
    ROUND_TRIP_FRICTION,
    SECTOR_MAPPING,
    TRADING_DAYS_PER_YEAR,
)
from econometrics_garch import MarketRegime


@dataclass
class BacktestMetrics:
    """Summary of institutional portfolio performance metrics."""
    initial_capital_vnd: float
    ending_capital_vnd: float
    total_profit_vnd: float
    cumulative_return_pct: float
    cagr_pct: float
    annual_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    annual_turnover_pct: float
    benchmark_cumulative_return_pct: float
    goal_target_vnd: float
    goal_achieved: bool
    goal_achievement_pct: float


class InstitutionalBacktester:
    """Simulates point-in-time execution with friction and cash interest."""

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL_VND,
        risk_free_rate: float = RISK_FREE_RATE,
        friction_rate: float = ROUND_TRIP_FRICTION,
        sector_mapping: Optional[Dict[str, str]] = None,
    ):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.friction_rate = friction_rate
        self.sector_mapping = sector_mapping or SECTOR_MAPPING

    def run(
        self,
        dates: pd.DatetimeIndex,
        stock_returns: pd.DataFrame,
        weights_history: pd.DataFrame,
        cash_weights: pd.Series,
        floor_history: pd.Series,
        var_history: pd.Series,
        regime_history: pd.Series,
        benchmark_prices: pd.Series,
        target_profit_vnd: float = ANNUAL_PROFIT_TARGET_VND,
    ) -> Tuple[pd.DataFrame, BacktestMetrics]:
        """Execute daily simulation accounting for daily stock returns and cash yields."""
        T = len(dates)
        tickers = stock_returns.columns.tolist()

        daily_rf = (1.0 + self.risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

        portfolio_values = np.zeros(T)
        portfolio_values[0] = self.initial_capital
        daily_portfolio_returns = np.zeros(T)

        curr_stock_w = {tkr: weights_history.iloc[0].get(tkr, 0.0) for tkr in tickers}
        curr_cash_w = cash_weights.iloc[0]

        # Simulation loop
        for t in range(1, T):
            dt = dates[t]

            # 1. Intra-period asset returns
            stock_rets_t = stock_returns.loc[dt]
            weighted_equity_return = sum(curr_stock_w.get(tkr, 0.0) * stock_rets_t.get(tkr, 0.0) for tkr in tickers)
            cash_return = curr_cash_w * daily_rf

            gross_return = weighted_equity_return + cash_return

            # Check if rebalancing occurs at day t
            is_rebalance_day = False
            prev_target_w = weights_history.iloc[t - 1]
            curr_target_w = weights_history.iloc[t]

            # If target weights changed, compute turnover and apply transaction friction
            turnover = 0.0
            if not prev_target_w.equals(curr_target_w):
                is_rebalance_day = True
                turnover = sum(abs(curr_target_w.get(tkr, 0.0) - curr_stock_w.get(tkr, 0.0)) for tkr in tickers)
                turnover += abs(cash_weights.iloc[t] - curr_cash_w)

            friction_loss = (turnover / 2.0) * self.friction_rate if is_rebalance_day else 0.0
            net_return = gross_return - friction_loss

            daily_portfolio_returns[t] = net_return
            portfolio_values[t] = portfolio_values[t - 1] * (1.0 + net_return)

            # Update weights for next step
            curr_stock_w = {tkr: curr_target_w.get(tkr, 0.0) for tkr in tickers}
            curr_cash_w = cash_weights.iloc[t]

        # Construct results DataFrame
        df_results = pd.DataFrame(
            {
                "portfolio_value": portfolio_values,
                "daily_return": daily_portfolio_returns,
                "floor_value": floor_history.values,
                "cash_weight": cash_weights.values,
                "var_95": var_history.values,
                "regime": regime_history.values,
            },
            index=dates,
        )

        # Merge stock weights
        for tkr in tickers:
            df_results[f"w_{tkr}"] = weights_history[tkr].values

        # High-water mark & Drawdown
        df_results["hwm"] = df_results["portfolio_value"].cummax()
        df_results["drawdown"] = (df_results["portfolio_value"] - df_results["hwm"]) / df_results["hwm"]

        # Benchmark returns
        rebased_bmk = (benchmark_prices / benchmark_prices.iloc[0]) * self.initial_capital
        df_results["benchmark_value"] = rebased_bmk.loc[dates]
        df_results["benchmark_drawdown"] = (
            df_results["benchmark_value"] - df_results["benchmark_value"].cummax()
        ) / df_results["benchmark_value"].cummax()

        # Compute Institutional Performance Metrics
        ending_val = portfolio_values[-1]
        total_profit = ending_val - self.initial_capital
        cum_ret = (ending_val / self.initial_capital) - 1.0

        n_years = T / TRADING_DAYS_PER_YEAR
        cagr = (ending_val / self.initial_capital) ** (1.0 / max(0.1, n_years)) - 1.0

        daily_r_clean = daily_portfolio_returns[1:]
        ann_vol = float(np.std(daily_r_clean) * np.sqrt(TRADING_DAYS_PER_YEAR))

        excess_returns = daily_r_clean - daily_rf
        sharpe = (float(np.mean(excess_returns)) / (float(np.std(excess_returns)) + 1e-8)) * np.sqrt(TRADING_DAYS_PER_YEAR)

        downside_returns = excess_returns[excess_returns < 0]
        downside_std = float(np.std(downside_returns)) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 1e-4
        sortino = (cagr - self.risk_free_rate) / downside_std

        mdd = float(df_results["drawdown"].min())
        calmar = cagr / abs(mdd) if abs(mdd) > 1e-4 else 0.0

        win_rate = float(np.sum(daily_r_clean > 0) / len(daily_r_clean))

        # Annual turnover rate
        weight_diffs = weights_history.diff().abs().sum(axis=1) + cash_weights.diff().abs()
        annual_turnover = float(weight_diffs.sum() / 2.0 / max(0.1, n_years))

        bmk_cum_ret = (rebased_bmk.iloc[-1] / rebased_bmk.iloc[0]) - 1.0

        target_total_vnd = self.initial_capital + target_profit_vnd
        goal_achieved = ending_val >= target_total_vnd
        goal_pct = (total_profit / target_profit_vnd) * 100.0

        metrics = BacktestMetrics(
            initial_capital_vnd=self.initial_capital,
            ending_capital_vnd=ending_val,
            total_profit_vnd=total_profit,
            cumulative_return_pct=cum_ret * 100.0,
            cagr_pct=cagr * 100.0,
            annual_volatility_pct=ann_vol * 100.0,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=mdd * 100.0,
            win_rate_pct=win_rate * 100.0,
            annual_turnover_pct=annual_turnover * 100.0,
            benchmark_cumulative_return_pct=bmk_cum_ret * 100.0,
            goal_target_vnd=target_total_vnd,
            goal_achieved=goal_achieved,
            goal_achievement_pct=goal_pct,
        )

        return df_results, metrics


class InvestorDashboardVisualizer:
    """Generates the Institutional 4-Panel Investor Dashboard."""

    def __init__(self, sector_mapping: Optional[Dict[str, str]] = None):
        self.sector_mapping = sector_mapping or SECTOR_MAPPING
        sns.set_theme(style="darkgrid")
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    def plot_4panel_dashboard(
        self,
        df_results: pd.DataFrame,
        metrics: BacktestMetrics,
        factor_ic_reports: List[Tuple[str, float, float]],  # (name, ic, t_stat)
        cond_vol_series: pd.Series,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Render and save the required 4-Panel Investor Dashboard."""
        if output_path is None:
            output_path = REPORTS_DIR / "investor_dashboard.png"

        fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=150)
        fig.suptitle(
            f"INSTITUTIONAL QUANTITATIVE PORTFOLIO DASHBOARD (VIETNAM EQUITIES)\n"
            f"Capital: {metrics.initial_capital_vnd:,.0f} VNĐ | Ending: {metrics.ending_capital_vnd:,.0f} VNĐ "
            f"(Profit: {metrics.total_profit_vnd:+,.0f} VNĐ, Goal: {metrics.goal_achievement_pct:.1f}%) | "
            f"Sharpe: {metrics.sharpe_ratio:.2f} | Max DD: {metrics.max_drawdown_pct:.2f}%",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        dates = df_results.index

        # ----------------------------------------------------------------------
        # PANEL 1: Wealth Curve vs Target Goal, Benchmark & VaR Bands
        # ----------------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(dates, df_results["portfolio_value"], label="Quant Portfolio (VND)", color="#00E5FF", linewidth=2.0)
        ax1.plot(dates, df_results["benchmark_value"], label="VN-Index Benchmark (VND)", color="#FFA000", linewidth=1.5, linestyle="--")
        ax1.plot(dates, df_results["floor_value"], label="CPPI Protected Floor (VND)", color="#FF5252", linewidth=1.5, linestyle=":")

        # Target goal line
        target_line = np.linspace(metrics.initial_capital_vnd, metrics.goal_target_vnd, len(dates))
        ax1.plot(dates, target_line, label=f"Tuition Target Goal ({metrics.goal_target_vnd:,.0f} VND)", color="#00E676", linewidth=1.8, linestyle="-.")

        # VaR band
        var_lower = df_results["portfolio_value"] * (1.0 - df_results["var_95"])
        ax1.fill_between(dates, var_lower, df_results["portfolio_value"], color="#00E5FF", alpha=0.15, label="1-Day 95% VaR Band")

        ax1.set_title("Panel 1: Wealth Trajectory vs Tuition Target & Floor", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Portfolio Value (VNĐ)")
        ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x*1e-6:.2f}M"))
        ax1.legend(loc="upper left", fontsize=8)

        # ----------------------------------------------------------------------
        # PANEL 2: Dynamic Asset & Sector Rotation Area Chart
        # ----------------------------------------------------------------------
        ax2 = axes[0, 1]
        sector_alloc = pd.DataFrame(index=dates)
        sector_alloc["Cash/FixedIncome"] = df_results["cash_weight"]

        # Aggregate stock weights by sector
        unique_sectors = sorted(set(self.sector_mapping.values()))
        for sec in unique_sectors:
            sec_cols = [f"w_{tkr}" for tkr in self.sector_mapping if self.sector_mapping[tkr] == sec and f"w_{tkr}" in df_results.columns]
            if sec_cols:
                sector_alloc[sec] = df_results[sec_cols].sum(axis=1)
            else:
                sector_alloc[sec] = 0.0

        # Stacked area plot
        ax2.stackplot(dates, sector_alloc.T, labels=sector_alloc.columns, alpha=0.85)
        ax2.set_title("Panel 2: Dynamic Asset & Sector Rotation", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Portfolio Allocation Weight")
        ax2.set_ylim(0, 1.0)
        ax2.legend(loc="lower left", fontsize=7, ncol=3)

        # ----------------------------------------------------------------------
        # PANEL 3: Underwater Drawdown vs GJR-GARCH Volatility Regime
        # ----------------------------------------------------------------------
        ax3 = axes[1, 0]
        ax3.plot(dates, df_results["drawdown"] * 100.0, label="Quant Strategy Drawdown (%)", color="#FF1744", linewidth=1.5)
        ax3.plot(dates, df_results["benchmark_drawdown"] * 100.0, label="VN-Index Drawdown (%)", color="#78909C", linewidth=1.0, linestyle="--")
        ax3.fill_between(dates, df_results["drawdown"] * 100.0, 0, color="#FF1744", alpha=0.2)
        ax3.set_ylabel("Drawdown (%)", color="#FF1744")
        ax3.tick_params(axis="y", labelcolor="#FF1744")

        # Secondary y-axis for GJR-GARCH Volatility
        ax3_twin = ax3.twinx()
        aligned_vol = cond_vol_series.reindex(dates).bfill() * 100.0
        ax3_twin.plot(dates, aligned_vol, label="GJR-GARCH Annualized Vol (%)", color="#7C4DFF", linewidth=1.2, linestyle=":")
        ax3_twin.set_ylabel("GARCH Vol Forecast (%)", color="#7C4DFF")
        ax3_twin.tick_params(axis="y", labelcolor="#7C4DFF")
        ax3.set_title("Panel 3: Underwater Drawdown & GJR-GARCH Risk Regime", fontsize=11, fontweight="bold")
        ax3.legend(loc="lower left", fontsize=8)
        ax3_twin.legend(loc="upper right", fontsize=8)

        # ----------------------------------------------------------------------
        # PANEL 4: Fundamental Factor Attribution & Rank IC
        # ----------------------------------------------------------------------
        ax4 = axes[1, 1]
        f_names = [item[0] for item in factor_ic_reports]
        f_ics = [item[1] for item in factor_ic_reports]
        f_tstats = [item[2] for item in factor_ic_reports]

        x_pos = np.arange(len(f_names))
        colors = ["#00E676" if ic > 0 else "#FF5252" for ic in f_ics]

        bars = ax4.barh(x_pos, f_ics, color=colors, alpha=0.85, height=0.55)
        ax4.set_yticks(x_pos)
        ax4.set_yticklabels(f_names, fontsize=8)
        ax4.set_xlabel("Mean Information Coefficient (Rank IC)")
        ax4.axvline(0, color="gray", linestyle="--", linewidth=0.8)

        # Annotate t-stats on bars
        for i, (b, t_s) in enumerate(zip(bars, f_tstats)):
            width = b.get_width()
            align = "left" if width >= 0 else "right"
            offset = 0.003 if width >= 0 else -0.003
            ax4.text(width + offset, b.get_y() + b.get_height() / 2, f"t={t_s:.2f}", va="center", ha=align, fontsize=8, fontweight="bold")

        ax4.set_title("Panel 4: Alpha Factor Quality & Predictive Power (Rank IC & t-stat)", fontsize=11, fontweight="bold")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path
