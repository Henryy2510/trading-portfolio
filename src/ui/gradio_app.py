"""Gradio Web Application for WorldQuant Institutional Portfolio Management (VN Equities).

Features:
- Parameter configuration (Capital, Horizon, Floor protection, Volatility targeting, Rf).
- Real-time institutional econometric testing & risk calculations.
- 4-Panel Interactive Bokeh Dashboard via gr.Plot.
- Full KPI Metrics Summary & Statistical Hypothesis Testing Report.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Silence matplotlib cache warning
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from typing import Dict, List, Tuple
import gradio as gr
import numpy as np
import pandas as pd

from src.config import (
    ANNUAL_PROFIT_TARGET_VND,
    DEFAULT_TICKERS,
    INITIAL_CAPITAL_VND,
    QUARTERLY_PROFIT_TARGET_VND,
    REBALANCE_FREQUENCY_DAYS,
    RISK_FREE_RATE,
    ROUND_TRIP_FRICTION,
    SECTOR_MAPPING,
    TARGET_PORTFOLIO_VOLATILITY,
)
from src.data import load_dataset
from src.features import AlphaFactory
from src.risk import (
    CovarianceEstimator,
    EconometricTester,
    GJRGARCHRiskEngine,
    MacroRegimeModel,
)
from src.portfolio import GoalBasedPortfolioOptimizer
from src.backtest import BacktestMetrics, InstitutionalBacktester
from src.visualization import (
    create_drawdown_garch_chart,
    create_factor_ic_chart,
    create_sector_allocation_chart,
    create_wealth_chart,
)


def run_quant_simulation(
    initial_capital: float,
    horizon_choice: str,
    target_profit: float,
    floor_percent: float,
    target_vol: float,
    risk_free_rate: float,
    max_stock_weight: float,
):
    """Execute full quantitative pipeline with user parameters and produce Bokeh charts."""
    horizon_code = "12m" if "12" in horizon_choice else "3m"

    # 1. Ingest Market & Fundamental Data
    market_data, fund_data = load_dataset(use_synthetic=True)
    dates = market_data.prices.index
    T, N = market_data.prices.shape

    # 2. Econometric Testing
    vnindex_ret = market_data.macro["VNINDEX_RET"]
    audit_report = EconometricTester.audit_time_series(vnindex_ret, variable_name="VN-Index Returns")
    macro_model = MacroRegimeModel(lookback_window=63)

    # 3. Hybrid Alpha Factory
    alpha_factory = AlphaFactory()
    composite_alpha, processed_factors = alpha_factory.construct_composite_alpha(market_data, fund_data)
    forward_rets_21d = market_data.prices.pct_change(21).shift(-21)

    factor_ic_reports = []
    for fname, f_df in processed_factors.items():
        eval_rep = alpha_factory.evaluate_factor_ic(f_df, forward_rets_21d, factor_name=fname)
        factor_ic_reports.append((fname, eval_rep.mean_ic, eval_rep.t_stat))

    # 4. GJR-GARCH Risk & Goal-Based CPPI Optimization
    risk_engine = GJRGARCHRiskEngine()
    optimizer = GoalBasedPortfolioOptimizer(
        initial_capital=initial_capital,
        floor_percent=floor_percent,
        risk_free_rate=risk_free_rate,
        target_vol=target_vol,
        max_stock_weight=max_stock_weight,
    )

    weights_history = pd.DataFrame(index=dates, columns=market_data.prices.columns, dtype=float)
    cash_weights = pd.Series(index=dates, dtype=float)
    floor_history = pd.Series(index=dates, dtype=float)
    var_history = pd.Series(index=dates, dtype=float)
    regime_history = pd.Series(index=dates, dtype=object)
    cond_vol_series = pd.Series(index=dates, dtype=float)

    hwm = initial_capital
    curr_alloc = None
    prev_stock_w = None
    rebalance_step_indices = set(range(0, T, REBALANCE_FREQUENCY_DAYS))

    for t_idx in range(T):
        dt = dates[t_idx]
        if t_idx in rebalance_step_indices or curr_alloc is None:
            regime, regime_mult = macro_model.detect_regime(market_data.macro, dt)
            sub_window = market_data.returns.loc[:dt].iloc[-252:]
            port_proxy_ret = sub_window.mean(axis=1)
            risk_forecast = risk_engine.fit_and_forecast(port_proxy_ret)
            cov_matrix, shrinkage = CovarianceEstimator.estimate_ledoit_wolf(sub_window)
            alpha_t = composite_alpha.loc[dt].fillna(0.0)

            curr_val = initial_capital if t_idx == 0 else hwm
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

        for tkr in market_data.prices.columns:
            weights_history.loc[dt, tkr] = curr_alloc.stock_weights.get(tkr, 0.0)
        cash_weights.loc[dt] = curr_alloc.cash_weight
        floor_history.loc[dt] = curr_alloc.floor_value
        var_history.loc[dt] = risk_forecast.var_95
        regime_history.loc[dt] = curr_alloc.regime.value
        cond_vol_series.loc[dt] = risk_forecast.conditional_vol_annual

    # 5. Backtest Simulation
    backtester = InstitutionalBacktester(
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
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

    # 6. Generate Interactive Bokeh Charts
    fig_wealth = create_wealth_chart(df_results, metrics, height=400)
    fig_sector = create_sector_allocation_chart(df_results, SECTOR_MAPPING, height=400)
    fig_drawdown = create_drawdown_garch_chart(df_results, cond_vol_series, height=370)
    fig_ic = create_factor_ic_chart(factor_ic_reports, height=370)

    # 7. Construct KPI Summary HTML Card
    status_badge = (
        "<span style='color: #00E676; font-weight: bold;'>✔ HOÀN THÀNH MỤC TIÊU</span>"
        if metrics.goal_achieved else
        "<span style='color: #FFA000; font-weight: bold;'>⏳ ĐANG TIẾN TRIỂN</span>"
    )

    kpi_html = f"""
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 10px;'>
        <div style='background: #1E222D; border: 1px solid #2A2E39; border-radius: 8px; padding: 14px; text-align: center;'>
            <div style='font-size: 11px; color: #787B86; text-transform: uppercase;'>Vốn Kết Thúc</div>
            <div style='font-size: 22px; font-weight: bold; color: #00E5FF;'>{metrics.ending_capital_vnd:,.0f} VNĐ</div>
            <div style='font-size: 12px; color: #00E676;'>+{metrics.total_profit_vnd:,.0f} VNĐ ({metrics.cumulative_return_pct:+.2f}%)</div>
        </div>
        <div style='background: #1E222D; border: 1px solid #2A2E39; border-radius: 8px; padding: 14px; text-align: center;'>
            <div style='font-size: 11px; color: #787B86; text-transform: uppercase;'>Mục Tiêu Học Phí</div>
            <div style='font-size: 22px; font-weight: bold; color: #FFF;'>{metrics.goal_target_vnd:,.0f} VNĐ</div>
            <div style='font-size: 12px;'>{status_badge} ({metrics.goal_achievement_pct:.1f}%)</div>
        </div>
        <div style='background: #1E222D; border: 1px solid #2A2E39; border-radius: 8px; padding: 14px; text-align: center;'>
            <div style='font-size: 11px; color: #787B86; text-transform: uppercase;'>Sharpe / Sortino</div>
            <div style='font-size: 22px; font-weight: bold; color: #B388FF;'>{metrics.sharpe_ratio:.2f} <span style='font-size: 14px; color: #787B86;'>/ {metrics.sortino_ratio:.2f}</span></div>
            <div style='font-size: 12px; color: #787B86;'>Rf = {risk_free_rate*100:.1f}%</div>
        </div>
        <div style='background: #1E222D; border: 1px solid #2A2E39; border-radius: 8px; padding: 14px; text-align: center;'>
            <div style='font-size: 11px; color: #787B86; text-transform: uppercase;'>Max Drawdown (MDD)</div>
            <div style='font-size: 22px; font-weight: bold; color: {'#00E676' if abs(metrics.max_drawdown_pct) <= 6.0 else '#FF5252'};'>{metrics.max_drawdown_pct:.2f}%</div>
            <div style='font-size: 12px; color: #787B86;'>Ngưỡng chặn an toàn: &le; 6.0%</div>
        </div>
        <div style='background: #1E222D; border: 1px solid #2A2E39; border-radius: 8px; padding: 14px; text-align: center;'>
            <div style='font-size: 11px; color: #787B86; text-transform: uppercase;'>Biến Động / Winrate</div>
            <div style='font-size: 22px; font-weight: bold; color: #FFA000;'>{metrics.annual_volatility_pct:.2f}% <span style='font-size: 14px; color: #787B86;'>/ {metrics.win_rate_pct:.1f}%</span></div>
            <div style='font-size: 12px; color: #787B86;'>Vòng quay: {metrics.annual_turnover_pct:.1f}%/năm</div>
        </div>
    </div>
    """

    # 8. Econometric Diagnostics Report Table (Markdown)
    econometrics_md = f"""
### Báo Cáo Kiểm Định Kinh Tế Lượng Chuẩn Quỹ (Econometric Hypothesis Testing)
| Phép Kiểm Định Thống Kê | Mục Đích | Chỉ Số Thực Nghiệm | p-value | Kết Luận Thống Kê |
| :--- | :--- | :--- | :--- | :--- |
| **Augmented Dickey-Fuller (ADF)** | Kiểm định tính dừng $I(0)$ | Stat = `{audit_report.adf_stat:.4f}` | `{audit_report.adf_pvalue:.2e}` | **{'Bác bỏ H0 - Chuỗi Dừng' if audit_report.is_stationary_adf else 'Không dừng'}** |
| **KPSS Test** | Kiểm định nghiệm đơn vị | Stat = `{audit_report.kpss_stat:.4f}` | `{audit_report.kpss_pvalue:.4f}` | **{'Chấp nhận H0 - Không có Unit Root' if audit_report.is_stationary_kpss else 'Có Unit Root'}** |
| **Ljung-Box (Lag 10)** | Tự tương quan tuyến tính | - | `{audit_report.ljung_box_pvalue:.4f}` | **{'Không tự tương quan' if not audit_report.has_autocorrelation else 'Có tự tương quan'}** |
| **Engle's ARCH LM Test** | Biến động cụm (Conditional Heteroskedasticity) | LM-Stat = `{audit_report.arch_lm_stat:.4f}` | `{audit_report.arch_lm_pvalue:.2e}` | **{'Bác bỏ H0 - Tồn tại hiệu ứng ARCH (Bắt buộc dùng GARCH)' if audit_report.has_arch_effect else 'Không có ARCH'}** |
    """

    return kpi_html, fig_wealth, fig_sector, fig_drawdown, fig_ic, econometrics_md


def build_app():
    """Build Gradio UI interface."""
    with gr.Blocks(title="WorldQuant Institutional VN Portfolio") as demo:
        gr.Markdown(
            """
            # 🏛️ WORLDQUANT INSTITUTIONAL PORTFOLIO MANAGEMENT SYSTEM
            ### Hệ thống Quản trị Danh mục Định lượng Đầu-Cuối cho Thị trường Chứng khoán Việt Nam (HOSE/VN30)
            *Bảo toàn vốn và tối ưu hóa mục tiêu học phí với thuật toán **CPPI Floor Protection**, **GJR-GARCH Volatility Targeting** và **Interactive Bokeh Canvas**.*
            """
        )

        with gr.Row():
            # Controls Sidebar (Width = 1)
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Tham Số Đầu Tư & Rủi Ro")
                inp_capital = gr.Number(
                    value=INITIAL_CAPITAL_VND,
                    label="Vốn Khởi Điểm ($V_0$ VNĐ)",
                    step=500_000,
                )
                inp_horizon = gr.Radio(
                    choices=["12 Tháng (Học Phí Năm)", "3 Tháng (Học Phí Quý)"],
                    value="12 Tháng (Học Phí Năm)",
                    label="Khung Thời Gian Đầu Tư",
                )
                inp_target = gr.Number(
                    value=ANNUAL_PROFIT_TARGET_VND,
                    label="Mục Tiêu Lãi Đóng Học Phí (VNĐ)",
                    step=50_000,
                )
                inp_floor = gr.Slider(
                    minimum=0.80,
                    maximum=0.98,
                    value=0.95,
                    step=0.01,
                    label="Tỷ lệ Bảo vệ Sàn CPPI (Floor %)",
                    info="Đảm bảo vốn không giảm dưới ngưỡng này để đóng học phí",
                )
                inp_vol = gr.Slider(
                    minimum=0.08,
                    maximum=0.25,
                    value=TARGET_PORTFOLIO_VOLATILITY,
                    step=0.01,
                    label="Biến Động Danh Mục Mục Tiêu (Target Vol)",
                    info="GJR-GARCH sẽ tự động phanh tỷ trọng khi vol thị trường vượt mức này",
                )
                inp_rf = gr.Slider(
                    minimum=0.02,
                    maximum=0.08,
                    value=RISK_FREE_RATE,
                    step=0.005,
                    label="Lãi Suất Tiền Gửi / Không Rủi Ro ($R_f$)",
                )
                inp_max_w = gr.Slider(
                    minimum=0.15,
                    maximum=0.45,
                    value=0.30,
                    step=0.05,
                    label="Tỷ Trọng Tối Đa Mỗi Cổ Phiếu ($w_{max}$)",
                )

                btn_run = gr.Button("▶ THỰC THI ĐỊNH LƯỢNG & TỐI ƯU HÓA", variant="primary")

                # Dynamic update of target profit when horizon changes
                def on_horizon_change(choice):
                    return ANNUAL_PROFIT_TARGET_VND if "12" in choice else QUARTERLY_PROFIT_TARGET_VND

                inp_horizon.change(fn=on_horizon_change, inputs=[inp_horizon], outputs=[inp_target])

            # Visualization & Metrics Main Panel (Width = 3)
            with gr.Column(scale=3):
                out_kpi = gr.HTML()

                with gr.Row():
                    out_plot_wealth = gr.Plot(label="Panel 1: Wealth Trajectory vs. Target & Floor")
                    out_plot_sector = gr.Plot(label="Panel 2: Dynamic Asset & Sector Rotation")

                with gr.Row():
                    out_plot_dd = gr.Plot(label="Panel 3: Underwater Drawdown vs. GJR-GARCH Volatility")
                    out_plot_ic = gr.Plot(label="Panel 4: Alpha Factor Quality (Rank IC & t-stat)")

                with gr.Accordion("📊 Xem Báo Cáo Kiểm Định Thống Kê & Giả Thuyết Kinh Tế Lượng", open=False):
                    out_econometrics = gr.Markdown()

        # Connect button click
        btn_run.click(
            fn=run_quant_simulation,
            inputs=[inp_capital, inp_horizon, inp_target, inp_floor, inp_vol, inp_rf, inp_max_w],
            outputs=[out_kpi, out_plot_wealth, out_plot_sector, out_plot_dd, out_plot_ic, out_econometrics],
        )

        # Trigger run once at startup
        demo.load(
            fn=run_quant_simulation,
            inputs=[inp_capital, inp_horizon, inp_target, inp_floor, inp_vol, inp_rf, inp_max_w],
            outputs=[out_kpi, out_plot_wealth, out_plot_sector, out_plot_dd, out_plot_ic, out_econometrics],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    custom_css = """
    .gradio-container { background-color: #0E1117 !important; color: #D1D4DC !important; }
    .bk-root { background-color: transparent !important; }
    """
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Base(), css=custom_css)
