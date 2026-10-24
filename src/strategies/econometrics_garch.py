"""Module 1 & 3: Econometric Testing, Macro Regime Classification & GJR-GARCH Dynamic Risk Modeling.

Implements institutional quantitative standards:
- Statistical Stationarity (ADF, KPSS).
- Autocorrelation & ARCH effect tests (Ljung-Box, Engle's ARCH LM test).
- Top-Down Macro Regime Identification (Expansion, Recovery, Contraction, Crisis).
- Dynamic GJR-GARCH(1,1,1) Volatility Modeling with Skewed Student-t distribution.
- Dynamic Value-at-Risk (VaR 95%), Expected Shortfall (CVaR 95%), and Ledoit-Wolf Shrinkage Covariance.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.covariance import ledoit_wolf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

from config import (
    CVAR_CONFIDENCE,
    DEFAULT_SEED,
    GARCH_DIST,
    GARCH_O,
    GARCH_P,
    GARCH_Q,
    TRADING_DAYS_PER_YEAR,
    VAR_SIGNIFICANCE_LEVEL,
)


class MarketRegime(Enum):
    EXPANSION = "Expansion"       # Bullish, low-to-moderate vol, accommodative macro
    RECOVERY = "Recovery"         # Transition from bottom, stabilizing macro
    CONTRACTION = "Contraction"   # Rising interest rates, decelerating momentum
    CRISIS = "Crisis"             # High volatility spike, severe liquidity squeeze


@dataclass
class EconometricAuditReport:
    """Report holding diagnostic test statistics."""
    variable_name: str
    adf_stat: float
    adf_pvalue: float
    is_stationary_adf: bool
    kpss_stat: float
    kpss_pvalue: float
    is_stationary_kpss: bool
    ljung_box_pvalue: float
    has_autocorrelation: bool
    arch_lm_stat: float
    arch_lm_pvalue: float
    has_arch_effect: bool


@dataclass
class RiskForecast:
    """Forecast conditional risk metrics for next trading period."""
    conditional_vol_daily: float
    conditional_vol_annual: float
    var_95: float
    cvar_95: float
    gjr_gamma_leverage: float
    gjr_alpha: float
    gjr_beta: float


class EconometricTester:
    """Performs rigorous statistical hypothesis testing on financial time series."""

    @staticmethod
    def audit_time_series(series: pd.Series, variable_name: str = "Series") -> EconometricAuditReport:
        clean_s = series.dropna()
        if len(clean_s) < 30:
            raise ValueError(f"Series {variable_name} has insufficient observations ({len(clean_s)}).")

        # 1. Augmented Dickey-Fuller (ADF)
        adf_res = adfuller(clean_s, autolag="AIC")
        adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])
        is_stat_adf = adf_p < 0.05

        # 2. KPSS Test
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_res = kpss(clean_s, regression="c", nlags="auto")
            kpss_stat, kpss_p = float(kpss_res[0]), float(kpss_res[1])
            is_stat_kpss = kpss_p > 0.05  # H0: stationary
        except Exception:
            kpss_stat, kpss_p, is_stat_kpss = 0.0, 1.0, True

        # 3. Ljung-Box Test for serial autocorrelation
        lb_res = acorr_ljungbox(clean_s, lags=[10], return_df=True)
        lb_p = float(lb_res["lb_pvalue"].iloc[0])
        has_autocorr = lb_p < 0.05

        # 4. Engle's ARCH Test
        try:
            arch_res = het_arch(clean_s, nlags=5)
            arch_stat, arch_p = float(arch_res[0]), float(arch_res[1])
            has_arch = arch_p < 0.05
        except Exception:
            arch_stat, arch_p, has_arch = 0.0, 1.0, False

        return EconometricAuditReport(
            variable_name=variable_name,
            adf_stat=adf_stat,
            adf_pvalue=adf_p,
            is_stationary_adf=is_stat_adf,
            kpss_stat=kpss_stat,
            kpss_pvalue=kpss_p,
            is_stationary_kpss=is_stat_kpss,
            ljung_box_pvalue=lb_p,
            has_autocorrelation=has_autocorr,
            arch_lm_stat=arch_stat,
            arch_lm_pvalue=arch_p,
            has_arch_effect=has_arch,
        )


class MacroRegimeModel:
    """Classifies market into macro regimes and outputs CPPI risk-scaling multipliers."""

    def __init__(self, lookback_window: int = 63):
        self.lookback_window = lookback_window

    def detect_regime(
        self,
        macro_df: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> Tuple[MarketRegime, float]:
        """Classify current regime using rolling macro indicators and return CPPI multiplier scale.

        Multiplier scale:
        - EXPANSION: 1.20 (favorable macro, higher equity risk tolerance)
        - RECOVERY: 1.00 (neutral/recovering)
        - CONTRACTION: 0.70 (defensive)
        - CRISIS: 0.30 (capital preservation)
        """
        history = macro_df.loc[:current_date]
        if len(history) < self.lookback_window:
            return MarketRegime.RECOVERY, 1.0

        recent = history.iloc[-self.lookback_window:]
        vnindex_prices = recent["VNINDEX"]
        interbank = recent["INTERBANK_RATE"]

        # Market trend: price vs 50-day SMA
        sma_50 = vnindex_prices.rolling(50).mean().iloc[-1]
        curr_price = vnindex_prices.iloc[-1]
        market_trend_bull = curr_price > sma_50

        # Market drawdown
        peak = vnindex_prices.max()
        mdd = (curr_price - peak) / peak

        # Rate velocity (interbank 3-month delta)
        rate_delta = interbank.iloc[-1] - interbank.iloc[0]

        # Recent volatility
        daily_ret = vnindex_prices.pct_change().dropna()
        vol = daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        if mdd < -0.15 or vol > 0.28:
            return MarketRegime.CRISIS, 0.30
        elif rate_delta > 0.008 or (not market_trend_bull and vol > 0.20):
            return MarketRegime.CONTRACTION, 0.70
        elif market_trend_bull and rate_delta <= 0.002:
            return MarketRegime.EXPANSION, 1.20
        else:
            return MarketRegime.RECOVERY, 1.00


class GJRGARCHRiskEngine:
    """Estimates GJR-GARCH(1,1,1) with Skewed Student-t distribution and dynamic VaR/CVaR."""

    def __init__(
        self,
        p: int = GARCH_P,
        o: int = GARCH_O,
        q: int = GARCH_Q,
        dist: str = GARCH_DIST,
    ):
        self.p = p
        self.o = o
        self.q = q
        self.dist = dist

    def fit_and_forecast(
        self,
        returns: pd.Series,
        alpha_level: float = VAR_SIGNIFICANCE_LEVEL,
    ) -> RiskForecast:
        """Fit GJR-GARCH(1,1,1) on return series (scaled by 100 for numerical stability)."""
        clean_ret = returns.dropna()
        if len(clean_ret) < 100:
            # Fallback to rolling sample stats if sample too small
            sample_std = float(clean_ret.std())
            annual_vol = sample_std * np.sqrt(TRADING_DAYS_PER_YEAR)
            var_95 = 1.645 * sample_std
            cvar_95 = 2.06 * sample_std
            return RiskForecast(sample_std, annual_vol, var_95, cvar_95, 0.0, 0.05, 0.90)

        # Rescale returns to percentage for robust GARCH optimizer convergence
        scaled_r = clean_ret * 100.0

        try:
            model = arch_model(
                scaled_r,
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                mean="Constant",
                rescale=False,
            )
            res = model.fit(disp="off", show_warning=False)

            # Extract parameters
            params = res.params
            alpha_val = float(params.get("alpha[1]", 0.05))
            gamma_val = float(params.get("gamma[1]", 0.08))  # Asymmetry / leverage term
            beta_val = float(params.get("beta[1]", 0.85))

            # 1-step ahead conditional variance forecast
            f_cast = res.forecast(horizon=1)
            forecast_var = float(f_cast.variance.iloc[-1, 0])
            daily_cond_vol = np.sqrt(forecast_var) / 100.0
            annual_cond_vol = daily_cond_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

            # Parametric VaR & CVaR under Skewed Student-t residuals
            residuals = res.std_resid.dropna()
            q_alpha = float(np.percentile(residuals, alpha_level * 100))
            mu = float(params.get("mu", 0.0)) / 100.0

            # Dynamic VaR (positive value representing loss threshold)
            var_95 = max(0.0, -(mu + q_alpha * daily_cond_vol))

            # Dynamic CVaR (Expected Shortfall)
            tail_residuals = residuals[residuals <= q_alpha]
            expected_tail = float(tail_residuals.mean()) if len(tail_residuals) > 0 else q_alpha * 1.25
            cvar_95 = max(0.0, -(mu + expected_tail * daily_cond_vol))

            return RiskForecast(
                conditional_vol_daily=daily_cond_vol,
                conditional_vol_annual=annual_cond_vol,
                var_95=var_95,
                cvar_95=cvar_95,
                gjr_gamma_leverage=gamma_val,
                gjr_alpha=alpha_val,
                gjr_beta=beta_val,
            )
        except Exception:
            # Fallback in case of solver non-convergence
            sample_std = float(clean_ret.std())
            annual_vol = sample_std * np.sqrt(TRADING_DAYS_PER_YEAR)
            return RiskForecast(
                conditional_vol_daily=sample_std,
                conditional_vol_annual=annual_vol,
                var_95=1.645 * sample_std,
                cvar_95=2.06 * sample_std,
                gjr_gamma_leverage=0.0,
                gjr_alpha=0.05,
                gjr_beta=0.90,
            )


class CovarianceEstimator:
    """Calculates robust Ledoit-Wolf Shrinkage Covariance Matrix."""

    @staticmethod
    def estimate_ledoit_wolf(returns_matrix: pd.DataFrame) -> Tuple[np.ndarray, float]:
        clean_df = returns_matrix.dropna()
        n_samples, n_features = clean_df.shape
        if n_samples < 5:
            # Fallback for insufficient sample size
            return np.eye(n_features) * 0.0004, 0.0

        X = clean_df.values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shrunk_cov, shrinkage = ledoit_wolf(X)
        return shrunk_cov, float(shrinkage)
