"""Main entrypoint for Institutional Quantitative Research & Portfolio Management (VN Equities).

Executes the End-to-End 5-Pillar Pipeline:
- Module 1: Top-Down Macro Regime & Econometric Hypothesis Testing.
- Module 2: Hybrid Fundamental + Momentum Alpha Factory (Anti-Lookahead, Sector Neutral).
- Module 3: Dynamic GJR-GARCH(1,1,1) Risk Modeling & Ledoit-Wolf Shrinkage.
- Module 4: Goal-Based CPPI Capital Allocation & Volatility Targeting (8M VND -> Tuition Goal).
- Module 5: Microstructure Backtesting & 4-Panel Investor Dashboard.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from alpha_factors import AlphaFactory
from backtest_visualizer import (
    InstitutionalBacktester,
    InvestorDashboardVisualizer,
)
from config import (
    ANNUAL_PROFIT_TARGET_VND,
    DEFAULT_TICKERS,
    END_DATE,
    INITIAL_CAPITAL_VND,
    QUARTERLY_PROFIT_TARGET_VND,
    REBALANCE_FREQUENCY_DAYS,
    REPORTS_DIR,
    RISK_FREE_RATE,
    ROUND_TRIP_FRICTION,
    SECTOR_MAPPING,
    START_DATE,
)
from data_loader import load_dataset
from econometrics_garch import (
    CovarianceEstimator,
    EconometricTester,
    GJRGARCHRiskEngine,
    MacroRegimeModel,
)
from portfolio_optimizer import GoalBasedPortfolioOptimizer


def run_pipeline(
    horizon: str = "12m",
    initial_capital: float = INITIAL_CAPITAL_VND,
    use_synthetic: bool = True,
) -> None:
    """Run the complete quantitative pipeline and log institutional audit findings."""
    print("=" * 80)
    print("WORLDQUANT INSTITUTIONAL QUANT WORKFLOW | VIETNAM EQUITIES PIPELINE")
    print("=" * 80)

    target_profit = ANNUAL_PROFIT_TARGET_VND if horizon == "12m" else QUARTERLY_PROFIT_TARGET_VND
    print(f"[*] Capital: {initial_capital:,.0f} VNĐ | Horizon: {horizon.upper()} | Target Profit: {target_profit:+,.0f} VNĐ")

    # --------------------------------------------------------------------------
    # DATA INGESTION
    # --------------------------------------------------------------------------
    print("\n[STEP 1/5] Ingesting Market & Fundamental Data (Anti-Lookahead Publication Delays)...")
    market_data, fund_data = load_dataset(use_synthetic=use_synthetic)
    dates = market_data.prices.index
    T, N = market_data.prices.shape
    print(f"  -> Processed {N} tickers across {T} daily bars ({dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')})")
    print(f"  -> Fundamental Statements loaded: {len(fund_data.statements)} quarterly filings.")

    # --------------------------------------------------------------------------
    # MODULE 1: ECONOMETRIC TESTING & MACRO REGIME
    # --------------------------------------------------------------------------
    print("\n[STEP 2/5] Module 1: Econometric Hypothesis Testing & Macro Regime Detection...")
    vnindex_ret = market_data.macro["VNINDEX_RET"]
    audit_report = EconometricTester.audit_time_series(vnindex_ret, variable_name="VN-Index Returns")

    print(f"  -> [ADF Test]: Stat = {audit_report.adf_stat:.4f}, p-value = {audit_report.adf_pvalue:.2e} "
          f"(Stationary: {audit_report.is_stationary_adf})")
    print(f"  -> [KPSS Test]: Stat = {audit_report.kpss_stat:.4f}, p-value = {audit_report.kpss_pvalue:.4f} "
          f"(Stationary: {audit_report.is_stationary_kpss})")
    print(f"  -> [Ljung-Box]: p-value = {audit_report.ljung_box_pvalue:.4f} "
          f"(Autocorrelation: {audit_report.has_autocorrelation})")
    print(f"  -> [Engle's ARCH Test]: LM-Stat = {audit_report.arch_lm_stat:.4f}, p-value = {audit_report.arch_lm_pvalue:.2e} "
          f"(ARCH Effect: {audit_report.has_arch_effect})")

    macro_model = MacroRegimeModel(lookback_window=63)

    # --------------------------------------------------------------------------
    # MODULE 2: HYBRID ALPHA FACTORY
    # --------------------------------------------------------------------------
    print("\n[STEP 3/5] Module 2: Hybrid Alpha Factory (Point-in-Time Fundamental + Momentum)...")
    alpha_factory = AlphaFactory()
    composite_alpha, processed_factors = alpha_factory.construct_composite_alpha(market_data, fund_data)

    # Forward 21-day returns for Information Coefficient (Rank IC) validation
    forward_rets_21d = market_data.prices.pct_change(21).shift(-21)

    factor_ic_reports = []
    print("  -> Factor Predictive Power (Information Coefficient Analysis):")
    for fname, f_df in processed_factors.items():
        eval_rep = alpha_factory.evaluate_factor_ic(f_df, forward_rets_21d, factor_name=fname)
        factor_ic_reports.append((fname, eval_rep.mean_ic, eval_rep.t_stat))
        print(f"     * {fname:<22}: Mean IC = {eval_rep.mean_ic:+.4f} | IC-IR = {eval_rep.ic_ir:+.2f} | "
              f"t-stat = {eval_rep.t_stat:+.2f} (p={eval_rep.p_value:.3f})")

    # --------------------------------------------------------------------------
    # MODULE 3 & 4: DYNAMIC RISK & GOAL-BASED PORTFOLIO OPTIMIZATION
    # --------------------------------------------------------------------------
    print("\n[STEP 4/5] Modules 3 & 4: GJR-GARCH Risk Modeling & Goal-Based CPPI Dynamic Allocation...")
    risk_engine = GJRGARCHRiskEngine()
    optimizer = GoalBasedPortfolioOptimizer(initial_capital=initial_capital)

    # Pre-allocate tracking dataframes
    weights_history = pd.DataFrame(index=dates, columns=market_data.prices.columns, dtype=float)
    cash_weights = pd.Series(index=dates, dtype=float)
    floor_history = pd.Series(index=dates, dtype=float)
    var_history = pd.Series(index=dates, dtype=float)
    regime_history = pd.Series(index=dates, dtype=object)
    cond_vol_series = pd.Series(index=dates, dtype=float)

    hwm = initial_capital
    curr_alloc = None
    prev_stock_w = None

    # Sequential rebalancing simulation
    rebalance_step_indices = set(range(0, T, REBALANCE_FREQUENCY_DAYS))

    for t_idx in range(T):
        dt = dates[t_idx]

        if t_idx in rebalance_step_indices or curr_alloc is None:
            # 1. Macro Regime
            regime, regime_mult = macro_model.detect_regime(market_data.macro, dt)

            # 2. GJR-GARCH Risk Estimation on recent returns
            sub_window = market_data.returns.loc[:dt].iloc[-252:]
            port_proxy_ret = sub_window.mean(axis=1)
            risk_forecast = risk_engine.fit_and_forecast(port_proxy_ret)

            # 3. Ledoit-Wolf Shrinkage Covariance
            cov_matrix, shrinkage = CovarianceEstimator.estimate_ledoit_wolf(sub_window)

            # 4. Alpha signals at date dt
            alpha_t = composite_alpha.loc[dt].fillna(0.0)

            # 5. Goal-Based CPPI Allocation
            curr_val = initial_capital if t_idx == 0 else hwm  # Approximation at rebalance step
            curr_alloc = optimizer.allocate(
                current_portfolio_value=curr_val,
                high_water_mark=hwm,
                current_step=t_idx,
                total_steps=T,
                alphas=alpha_t,
                cov_matrix=cov_matrix,
                risk_forecast=risk_forecast,
                regime=regime,
                regime_multiplier=regime_mult,
                prev_stock_weights=prev_stock_w,
            )
            prev_stock_w = curr_alloc.stock_weights

        # Record daily parameters
        for tkr in market_data.prices.columns:
            weights_history.loc[dt, tkr] = curr_alloc.stock_weights.get(tkr, 0.0)
        cash_weights.loc[dt] = curr_alloc.cash_weight
        floor_history.loc[dt] = curr_alloc.floor_value
        var_history.loc[dt] = risk_forecast.var_95
        regime_history.loc[dt] = curr_alloc.regime.value
        cond_vol_series.loc[dt] = risk_forecast.conditional_vol_annual

    # --------------------------------------------------------------------------
    # MODULE 5: INSTITUTIONAL BACKTESTING & VISUALIZATION
    # --------------------------------------------------------------------------
    print("\n[STEP 5/5] Module 5: Institutional Backtest Engine & 4-Panel Investor Dashboard...")
    backtester = InstitutionalBacktester(
        initial_capital=initial_capital,
        risk_free_rate=RISK_FREE_RATE,
        friction_rate=ROUND_TRIP_FRICTION,
        sector_mapping=SECTOR_MAPPING,
    )

    df_results, metrics = backtester.run(
        dates=dates,
        stock_returns=market_data.returns,
        weights_history=weights_history,
        cash_weights=cash_weights,
        floor_history=floor_history,
        var_history=var_history,
        regime_history=regime_history,
        benchmark_prices=market_data.macro["VNINDEX"],
        target_profit_vnd=target_profit,
    )

    # Render 4-Panel Investor Dashboard
    dashboard_visualizer = InvestorDashboardVisualizer()
    plot_path = dashboard_visualizer.plot_4panel_dashboard(
        df_results=df_results,
        metrics=metrics,
        factor_ic_reports=factor_ic_reports,
        cond_vol_series=cond_vol_series,
    )

    # Print Institutional Performance Audit
    print("=" * 80)
    print("INSTITUTIONAL QUANTITATIVE AUDIT REPORT")
    print("=" * 80)
    print(f"Initial Capital          : {metrics.initial_capital_vnd:,.0f} VNĐ")
    print(f"Ending Portfolio Capital : {metrics.ending_capital_vnd:,.0f} VNĐ")
    print(f"Net Realized Profit      : {metrics.total_profit_vnd:+,.0f} VNĐ ({metrics.cumulative_return_pct:+.2f}%)")
    print(f"Benchmark Return (VN-Ind): {metrics.benchmark_cumulative_return_pct:+.2f}%")
    print(f"CAGR                     : {metrics.cagr_pct:.2f}%")
    print(f"Annualized Volatility    : {metrics.annual_volatility_pct:.2f}%")
    print(f"Sharpe Ratio (Rf=4.5%)   : {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio            : {metrics.sortino_ratio:.2f}")
    print(f"Calmar Ratio             : {metrics.calmar_ratio:.2f}")
    print(f"Maximum Drawdown (MDD)   : {metrics.max_drawdown_pct:.2f}% (Constraint: <= 6.0%)")
    print(f"Win Rate (Daily)         : {metrics.win_rate_pct:.2f}%")
    print(f"Annualized Turnover Rate : {metrics.annual_turnover_pct:.2f}%")
    print(f"Tuition Target Goal      : {metrics.goal_target_vnd:,.0f} VNĐ")
    print(f"Goal Completion Status   : {'[PASSED] ACHIEVED' if metrics.goal_achieved else '[PROGRESSING]'} ({metrics.goal_achievement_pct:.1f}%)")
    print(f"Dashboard Visual Saved At: {plot_path.resolve()}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="WorldQuant VN Equities Quantitative Management Pipeline")
    parser.add_argument("--horizon", type=str, default="12m", choices=["12m", "3m"], help="Tuition investment horizon")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL_VND, help="Initial capital in VND")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use realistic synthetic market generator")
    args = parser.parse_args()

    run_pipeline(
        horizon=args.horizon,
        initial_capital=args.capital,
        use_synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
