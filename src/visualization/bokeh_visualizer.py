"""Interactive Bokeh Visualization Engine for Institutional Quantitative Research.

Produces interactive Bokeh charts tailored for the Gradio Web UI:
- Panel 1: Wealth Curve vs Target Goal, Benchmark & CPPI Floor with Dynamic VaR Band.
- Panel 2: Dynamic Asset & Sector Rotation Stacked Area Chart.
- Panel 3: Underwater Drawdown & GJR-GARCH Volatility Regime (Dual-Axis).
- Panel 4: Alpha Factor Quality & Rank IC Bar Chart with t-statistic tooltips.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from bokeh.models import (
    ColumnDataSource,
    DatetimeTickFormatter,
    HoverTool,
    Legend,
    LegendItem,
    LinearAxis,
    NumeralTickFormatter,
    Range1d,
)
from bokeh.palettes import Category10, Spectral8
from bokeh.plotting import figure

from src.backtest.institutional import BacktestMetrics
from src.config import SECTOR_MAPPING


# Theme styling constants
THEME_BG = "#131722"         # TradingView dark background
THEME_PLOT_BG = "#1E222D"    # Chart area background
THEME_GRID = "#2A2E39"       # Grid lines
THEME_TEXT = "#D1D4DC"       # Main text color
THEME_MUTED = "#787B86"      # Subtext / labels
COLOR_CYAN = "#00E5FF"       # Portfolio curve
COLOR_AMBER = "#FFA000"      # Benchmark
COLOR_RED = "#FF5252"        # Floor / Drawdown
COLOR_GREEN = "#00E676"      # Target Goal / Positive IC
COLOR_PURPLE = "#B388FF"     # GJR-GARCH Volatility


def _apply_theme(p: figure) -> figure:
    """Apply modern institutional dark styling to a Bokeh figure."""
    p.background_fill_color = THEME_PLOT_BG
    p.border_fill_color = THEME_BG
    p.outline_line_color = THEME_GRID
    p.title.text_color = THEME_TEXT
    p.title.text_font_size = "13pt"
    p.title.text_font_style = "bold"

    p.xaxis.axis_line_color = THEME_GRID
    p.yaxis.axis_line_color = THEME_GRID
    p.xaxis.major_tick_line_color = THEME_GRID
    p.yaxis.major_tick_line_color = THEME_GRID
    p.xaxis.minor_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.xaxis.major_label_text_color = THEME_MUTED
    p.yaxis.major_label_text_color = THEME_MUTED
    p.xaxis.axis_label_text_color = THEME_TEXT
    p.yaxis.axis_label_text_color = THEME_TEXT

    p.xgrid.grid_line_color = THEME_GRID
    p.ygrid.grid_line_color = THEME_GRID
    p.xgrid.grid_line_dash = [2, 4]
    p.ygrid.grid_line_dash = [2, 4]
    p.toolbar.logo = None
    return p


def create_wealth_chart(
    df_results: pd.DataFrame,
    metrics: BacktestMetrics,
    height: int = 420,
) -> figure:
    """Panel 1: Interactive Wealth Trajectory vs. Target Goal, Benchmark & CPPI Floor."""
    dates = df_results.index
    target_line = np.linspace(metrics.initial_capital_vnd, metrics.goal_target_vnd, len(dates))
    var_lower = df_results["portfolio_value"] * (1.0 - df_results["var_95"])

    source_data = dict(
        date=dates,
        date_str=[d.strftime("%Y-%m-%d") for d in dates],
        portfolio_value=df_results["portfolio_value"].values,
        benchmark_value=df_results["benchmark_value"].values,
        floor_value=df_results["floor_value"].values,
        target_value=target_line,
        var_lower=var_lower.values,
        drawdown_pct=df_results["drawdown"].values * 100.0,
    )
    source = ColumnDataSource(data=source_data)

    p = figure(
        x_axis_type="datetime",
        title="Panel 1: Wealth Trajectory vs. Tuition Target & Floor (VND)",
        height=height,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    _apply_theme(p)

    # 1-Day 95% VaR Confidence Band
    p.varea(
        x="date",
        y1="var_lower",
        y2="portfolio_value",
        source=source,
        fill_color=COLOR_CYAN,
        fill_alpha=0.18,
        legend_label="1-Day 95% VaR Band",
    )

    # Curves
    r_port = p.line("date", "portfolio_value", source=source, color=COLOR_CYAN, line_width=2.5, legend_label="Quant Portfolio")
    r_bmk = p.line("date", "benchmark_value", source=source, color=COLOR_AMBER, line_width=1.8, line_dash="dashed", legend_label="VN-Index Benchmark")
    r_floor = p.line("date", "floor_value", source=source, color=COLOR_RED, line_width=1.8, line_dash="dotted", legend_label="CPPI Protected Floor")
    r_target = p.line("date", "target_value", source=source, color=COLOR_GREEN, line_width=2.0, line_dash="dashdot", legend_label=f"Goal ({metrics.goal_target_vnd:,.0f} VND)")

    # Formatters
    p.xaxis.formatter = DatetimeTickFormatter(months="%b %Y", years="%Y")
    p.yaxis.formatter = NumeralTickFormatter(format="0,0 a")
    p.yaxis.axis_label = "Portfolio Value (VNĐ)"

    # Hover Tool
    hover = HoverTool(
        renderers=[r_port],
        tooltips=[
            ("Date", "@date_str"),
            ("Portfolio Value", "@portfolio_value{0,0} VNĐ"),
            ("Benchmark Value", "@benchmark_value{0,0} VNĐ"),
            ("CPPI Floor", "@floor_value{0,0} VNĐ"),
            ("Target Line", "@target_value{0,0} VNĐ"),
            ("Drawdown", "@drawdown_pct{+0.2f}%"),
        ],
        mode="vline",
    )
    p.add_tools(hover)

    p.legend.location = "top_left"
    p.legend.background_fill_color = THEME_PLOT_BG
    p.legend.background_fill_alpha = 0.8
    p.legend.label_text_color = THEME_TEXT
    p.legend.border_line_color = THEME_GRID
    p.legend.click_policy = "hide"

    return p


def create_sector_allocation_chart(
    df_results: pd.DataFrame,
    sector_mapping: Optional[Dict[str, str]] = None,
    height: int = 420,
) -> figure:
    """Panel 2: Dynamic Asset & Sector Rotation Stacked Area Chart."""
    sector_mapping = sector_mapping or SECTOR_MAPPING
    dates = df_results.index

    # Aggregate weights by sector
    unique_sectors = sorted(set(sector_mapping.values()))
    alloc_data = {"date": dates, "date_str": [d.strftime("%Y-%m-%d") for d in dates]}
    alloc_data["Cash"] = df_results["cash_weight"].values

    for sec in unique_sectors:
        sec_cols = [f"w_{tkr}" for tkr in sector_mapping if sector_mapping[tkr] == sec and f"w_{tkr}" in df_results.columns]
        if sec_cols:
            alloc_data[sec] = df_results[sec_cols].sum(axis=1).values
        else:
            alloc_data[sec] = np.zeros(len(dates))

    layers = ["Cash"] + unique_sectors
    palette = [COLOR_CYAN] + list(Category10[max(3, len(unique_sectors))])[:len(unique_sectors)]

    # Compute stacked cumulative areas
    cum_bottom = np.zeros(len(dates))
    area_series = {}
    for layer in layers:
        layer_top = cum_bottom + alloc_data[layer]
        area_series[f"{layer}_bottom"] = cum_bottom.copy()
        area_series[f"{layer}_top"] = layer_top.copy()
        alloc_data[f"{layer}_pct"] = alloc_data[layer] * 100.0
        cum_bottom = layer_top

    combined_data = {**alloc_data, **area_series}
    source = ColumnDataSource(data=combined_data)

    p = figure(
        x_axis_type="datetime",
        title="Panel 2: Dynamic Asset & Sector Rotation",
        height=height,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        y_range=(0.0, 1.05),
    )
    _apply_theme(p)

    for i, layer in enumerate(layers):
        p.varea(
            x="date",
            y1=f"{layer}_bottom",
            y2=f"{layer}_top",
            source=source,
            fill_color=palette[i % len(palette)],
            fill_alpha=0.75,
            legend_label=layer,
        )

    p.xaxis.formatter = DatetimeTickFormatter(months="%b %Y", years="%Y")
    p.yaxis.formatter = NumeralTickFormatter(format="0%")
    p.yaxis.axis_label = "Allocation Weight"

    # Hover
    hover_tooltips = [("Date", "@date_str")] + [(f"{layer} Allocation", f"@{layer}_pct{{0.1f}}%") for layer in layers]
    hover = HoverTool(tooltips=hover_tooltips, mode="vline")
    p.add_tools(hover)

    p.legend.location = "bottom_left"
    p.legend.orientation = "horizontal"
    p.legend.background_fill_color = THEME_PLOT_BG
    p.legend.background_fill_alpha = 0.8
    p.legend.label_text_color = THEME_TEXT
    p.legend.border_line_color = THEME_GRID
    p.legend.label_text_font_size = "8pt"
    p.legend.click_policy = "hide"

    return p


def create_drawdown_garch_chart(
    df_results: pd.DataFrame,
    cond_vol_series: pd.Series,
    height: int = 380,
) -> figure:
    """Panel 3: Interactive Underwater Drawdown & GJR-GARCH Volatility Regime (Dual-Axis)."""
    dates = df_results.index
    aligned_vol = cond_vol_series.reindex(dates).bfill().values * 100.0
    dd_strat = df_results["drawdown"].values * 100.0
    dd_bmk = df_results["benchmark_drawdown"].values * 100.0

    source = ColumnDataSource(data=dict(
        date=dates,
        date_str=[d.strftime("%Y-%m-%d") for d in dates],
        drawdown_strat=dd_strat,
        drawdown_bmk=dd_bmk,
        garch_vol=aligned_vol,
        zeros=np.zeros(len(dates)),
    ))

    min_dd = min(float(np.min(dd_strat)), float(np.min(dd_bmk)), -10.0) * 1.15
    max_vol = max(float(np.max(aligned_vol)), 30.0) * 1.15

    p = figure(
        x_axis_type="datetime",
        title="Panel 3: Underwater Drawdown & GJR-GARCH Volatility Regime",
        height=height,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        y_range=(min_dd, 2.0),
    )
    _apply_theme(p)

    # Strategy Drawdown Area
    p.varea(
        x="date",
        y1="drawdown_strat",
        y2="zeros",
        source=source,
        fill_color=COLOR_RED,
        fill_alpha=0.30,
    )
    r_dd = p.line("date", "drawdown_strat", source=source, color=COLOR_RED, line_width=1.8, legend_label="Quant Drawdown (%)")
    p.line("date", "drawdown_bmk", source=source, color=THEME_MUTED, line_width=1.2, line_dash="dashed", legend_label="VN-Index Drawdown (%)")

    # Extra y-axis for GJR-GARCH Volatility
    p.extra_y_ranges = {"vol_axis": Range1d(start=0.0, end=max_vol)}
    p.add_layout(
        LinearAxis(
            y_range_name="vol_axis",
            axis_label="GJR-GARCH Annual Vol (%)",
            axis_label_text_color=COLOR_PURPLE,
            major_label_text_color=COLOR_PURPLE,
            axis_line_color=COLOR_PURPLE,
            major_tick_line_color=COLOR_PURPLE,
        ),
        "right",
    )
    p.line("date", "garch_vol", source=source, color=COLOR_PURPLE, line_width=1.8, line_dash="dotted", y_range_name="vol_axis", legend_label="GARCH Vol Forecast (%)")

    p.xaxis.formatter = DatetimeTickFormatter(months="%b %Y", years="%Y")
    p.yaxis.formatter = NumeralTickFormatter(format="0.0")
    p.yaxis.axis_label = "Drawdown (%)"

    hover = HoverTool(
        renderers=[r_dd],
        tooltips=[
            ("Date", "@date_str"),
            ("Quant Drawdown", "@drawdown_strat{+0.2f}%"),
            ("Benchmark Drawdown", "@drawdown_bmk{+0.2f}%"),
            ("GARCH Forecast Vol", "@garch_vol{0.2f}%"),
        ],
        mode="vline",
    )
    p.add_tools(hover)

    p.legend.location = "bottom_left"
    p.legend.background_fill_color = THEME_PLOT_BG
    p.legend.background_fill_alpha = 0.8
    p.legend.label_text_color = THEME_TEXT
    p.legend.border_line_color = THEME_GRID
    p.legend.click_policy = "hide"

    return p


def create_factor_ic_chart(
    factor_ic_reports: List[Tuple[str, float, float]],  # (name, ic, t_stat)
    height: int = 380,
) -> figure:
    """Panel 4: Interactive Factor Information Coefficient (Rank IC) & t-statistic Bar Chart."""
    f_names = [item[0] for item in factor_ic_reports]
    f_ics = [item[1] for item in factor_ic_reports]
    f_tstats = [item[2] for item in factor_ic_reports]

    colors = [COLOR_GREEN if ic >= 0 else COLOR_RED for ic in f_ics]

    source = ColumnDataSource(data=dict(
        factor=f_names,
        ic=f_ics,
        ic_pct=[ic * 100.0 for ic in f_ics],
        t_stat=f_tstats,
        color=colors,
    ))

    p = figure(
        y_range=f_names,
        title="Panel 4: Alpha Factor Quality & Predictive Power (Rank IC & t-stat)",
        height=height,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,reset,save",
    )
    _apply_theme(p)

    r_bars = p.hbar(y="factor", right="ic", height=0.55, source=source, color="color", alpha=0.85)

    p.xaxis.axis_label = "Mean Information Coefficient (Rank IC)"
    p.yaxis.axis_label = "Alpha Factor"

    # Hover
    hover = HoverTool(
        renderers=[r_bars],
        tooltips=[
            ("Factor", "@factor"),
            ("Rank IC", "@ic{+0.0004}"),
            ("t-statistic", "@t_stat{+0.2f}"),
        ],
    )
    p.add_tools(hover)

    return p
