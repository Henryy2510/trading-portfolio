"""Module 4: Goal-Based Portfolio Management & Regime-Aware Dynamic Optimization.

Solves the tuition goal allocation problem:
- Initial Capital V_0 = 8,000,000 VND.
- Target Profit: +440,000 VND / 12M (+5.5%) or +100,000 VND / 3M (+1.25%).
- Constant Proportion Portfolio Insurance (CPPI) floor protection to guarantee tuition preservation.
- Long-only constraints (w_i >= 0), turnover penalty, sector bounds.
- Dynamic GJR-GARCH Volatility Targeting scaling.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import (
    CPPI_FLOOR_PERCENT,
    CPPI_MULTIPLIER_BASE,
    INITIAL_CAPITAL_VND,
    RISK_FREE_RATE,
    ROUND_TRIP_FRICTION,
    SECTOR_MAPPING,
    TARGET_PORTFOLIO_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
)
from src.risk.econometrics_garch import MarketRegime, RiskForecast


@dataclass
class AllocationResult:
    """Detailed allocation breakdown for the portfolio at time t."""
    stock_weights: Dict[str, float]  # Ticker -> weight in total portfolio (sum <= 1.0)
    cash_weight: float               # Weight in risk-free asset / cash
    risky_budget: float              # Fraction allocated to risky assets
    floor_value: float               # Current CPPI floor value in VND
    cushion_vnd: float               # Current cushion (Portfolio Value - Floor)
    target_vol_scale: float          # GARCH volatility targeting multiplier
    regime: MarketRegime


class GoalBasedPortfolioOptimizer:
    """Manages capital preservation and growth towards a specific financial target."""

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL_VND,
        floor_percent: float = CPPI_FLOOR_PERCENT,
        risk_free_rate: float = RISK_FREE_RATE,
        base_multiplier: float = CPPI_MULTIPLIER_BASE,
        target_vol: float = TARGET_PORTFOLIO_VOLATILITY,
        sector_mapping: Optional[Dict[str, str]] = None,
        max_stock_weight: float = 0.30,
        max_sector_weight: float = 0.45,
        turnover_penalty_lambda: float = 0.005,
    ):
        self.initial_capital = initial_capital
        self.floor_percent = floor_percent
        self.risk_free_rate = risk_free_rate
        self.base_multiplier = base_multiplier
        self.target_vol = target_vol
        self.sector_mapping = sector_mapping or SECTOR_MAPPING
        self.max_stock_weight = max_stock_weight
        self.max_sector_weight = max_sector_weight
        self.turnover_penalty_lambda = turnover_penalty_lambda

    def calculate_cppi_floor(
        self,
        current_step: int,
        total_steps: int,
        high_water_mark: float,
    ) -> float:
        """Calculate dynamic present value of protected floor.

        F_t = V_floor * (1 + R_f)^(-(T - t)/252).
        Includes a ratchet mechanism based on high-water mark gains.
        """
        remaining_years = max(0.0, (total_steps - current_step) / TRADING_DAYS_PER_YEAR)
        discount_factor = (1.0 + self.risk_free_rate) ** (-remaining_years)

        # Baseline protected floor
        base_floor = self.initial_capital * self.floor_percent

        # Ratchet mechanism: if portfolio achieved large profits, protect 90% of HWM
        ratchet_floor = high_water_mark * 0.90
        effective_terminal_floor = max(base_floor, ratchet_floor)

        return effective_terminal_floor * discount_factor

    def optimize_risky_subportfolio(
        self,
        alphas: pd.Series,
        cov_matrix: np.ndarray,
        prev_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve Mean-Variance with turnover regularization under Long-only constraints.

        max_x: alpha^T x - 0.5 * gamma * x^T Cov x - lambda * ||x - x_prev||_1
        s.t.: sum(x) = 1, 0 <= x_i <= max_stock_weight, sector limits.
        """
        tickers = alphas.index.tolist()
        N = len(tickers)
        if N == 0:
            return np.array([])

        if prev_weights is None or len(prev_weights) != N:
            prev_weights = np.ones(N) / N

        alpha_vec = alphas.values
        gamma_risk = 5.0  # Risk aversion parameter

        def objective(x: np.ndarray) -> float:
            port_ret = float(np.dot(alpha_vec, x))
            port_var = float(x.T @ cov_matrix @ x)
            turnover = float(np.sum(np.abs(x - prev_weights)))
            # Minimize negative utility
            return -(port_ret - 0.5 * gamma_risk * port_var - self.turnover_penalty_lambda * turnover)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}  # Budget constraint
        ]

        # Sector exposure constraints
        unique_sectors = sorted(set(self.sector_mapping.values()))
        for sec in unique_sectors:
            sec_indices = [i for i, tkr in enumerate(tickers) if self.sector_mapping.get(tkr) == sec]
            if sec_indices:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda x, idx=sec_indices: self.max_sector_weight - np.sum(x[idx])
                })

        # Bounds: Long-only w_i in [0, max_stock_weight]
        bounds = [(0.0, self.max_stock_weight) for _ in range(N)]

        # Initial guess (equal weight or previous weight)
        x0 = np.clip(prev_weights, 0.0, self.max_stock_weight)
        x0 = x0 / np.sum(x0)

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6}
        )

        if res.success:
            weights = np.maximum(0.0, res.x)
            return weights / np.sum(weights)
        else:
            # Fallback to normalized positive alphas or equal weight
            pos_alpha = np.maximum(0.0, alpha_vec)
            if np.sum(pos_alpha) > 1e-6:
                w = pos_alpha / np.sum(pos_alpha)
            else:
                w = np.ones(N) / N
            return w

    def allocate(
        self,
        current_portfolio_value: float,
        high_water_mark: float,
        current_step: int,
        total_steps: int,
        alphas: pd.Series,
        cov_matrix: np.ndarray,
        risk_forecast: RiskForecast,
        regime: MarketRegime,
        regime_multiplier: float,
        prev_stock_weights: Optional[Dict[str, float]] = None,
    ) -> AllocationResult:
        """Execute CPPI + Volatility Targeting + Sub-portfolio optimization."""
        tickers = alphas.index.tolist()
        N = len(tickers)

        # 1. CPPI Floor & Cushion Calculation
        floor_value = self.calculate_cppi_floor(current_step, total_steps, high_water_mark)
        cushion = max(0.0, current_portfolio_value - floor_value)

        # Multiplier adjusted by Macro Regime
        effective_m = self.base_multiplier * regime_multiplier

        # CPPI risky exposure in VND
        if cushion <= 0:
            # Floor breached or touched: 100% Cash / Safe Asset
            risky_budget = 0.0
            vol_scale = 1.0
        else:
            unscaled_risky_vnd = min(current_portfolio_value, effective_m * cushion)
            base_risky_fraction = unscaled_risky_vnd / current_portfolio_value

            # 2. Dynamic GJR-GARCH Volatility Scaling
            # When conditional vol is higher than target, de-lever
            forecast_vol = max(1e-4, risk_forecast.conditional_vol_annual)
            vol_scale = min(1.0, self.target_vol / forecast_vol)

            risky_budget = min(1.0, base_risky_fraction * vol_scale)

        cash_weight = 1.0 - risky_budget

        # 3. Optimize stock weights inside risky sleeve
        if risky_budget > 0.01:
            prev_w_arr = np.array([prev_stock_weights.get(t, 0.0) if prev_stock_weights else 1.0/N for t in tickers])
            if np.sum(prev_w_arr) > 1e-4:
                prev_w_arr = prev_w_arr / np.sum(prev_w_arr)
            else:
                prev_w_arr = np.ones(N) / N

            sub_w = self.optimize_risky_subportfolio(alphas, cov_matrix, prev_w_arr)
            stock_weights = {tkr: float(sub_w[i] * risky_budget) for i, tkr in enumerate(tickers)}
        else:
            stock_weights = {tkr: 0.0 for tkr in tickers}

        return AllocationResult(
            stock_weights=stock_weights,
            cash_weight=cash_weight,
            risky_budget=risky_budget,
            floor_value=floor_value,
            cushion_vnd=cushion,
            target_vol_scale=vol_scale,
            regime=regime,
        )
