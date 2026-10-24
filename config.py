"""Configuration module for Institutional Quantitative Research & Portfolio Management.

Tailored for Vietnam Equities (HOSE/VN30/VN100) with Goal-Based Capital Allocation.
Standards conform to WorldQuant Quantitative Research protocols.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ==============================================================================
# 1. TIME HORIZON & CALENDAR CONFIGURATION
# ==============================================================================
START_DATE: str = "2023-01-01"
END_DATE: str = "2025-12-31"
SPLIT_DATE: str = "2024-12-31"  # In-Sample (2023-2024) vs Out-of-Sample (2025)
TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_SEED: int = 42

# Vietnam Market Micro-Structure & Settlement
T_SETTLEMENT_DAYS: float = 2.5
HOSE_PRICE_LIMIT: float = 0.07  # +/- 7% daily price band
ROUND_TRIP_FRICTION: float = 0.0025  # 0.15% brokerage + 0.10% transaction tax on sales
FEES: float = 0.0015
SLIPPAGE: float = 0.0010

# ==============================================================================
# 2. GOAL-BASED INVESTOR CAPITAL ALLOCATION
# ==============================================================================
INITIAL_CAPITAL_VND: float = 8_000_000.0  # V_0: 8,000,000 VND
ANNUAL_PROFIT_TARGET_VND: float = 440_000.0  # +5.5% annual tuition target
QUARTERLY_PROFIT_TARGET_VND: float = 100_000.0  # +1.25% quarterly tuition target
MAX_DRAWDOWN_TOLERANCE: float = 0.06  # MDD constraint <= 6%
CPPI_FLOOR_PERCENT: float = 0.95  # Protect 95% minimum capital to guarantee tuition
RISK_FREE_RATE: float = 0.045  # 4.5% annual Vietnamese 12-month deposit / interbank rate
CPPI_MULTIPLIER_BASE: float = 3.5  # Multiplier m for cushion

# Backward compatibility aliases
INITIAL_CASH: float = INITIAL_CAPITAL_VND
INITIAL_CASH_NOTIONAL: float = INITIAL_CAPITAL_VND

# ==============================================================================
# 3. UNIVERSE & SECTOR TAXONOMY (VN30 Core)
# ==============================================================================
SECTOR_MAPPING: Dict[str, str] = {
    "VCB": "Banking",
    "MBB": "Banking",
    "TCB": "Banking",
    "CTG": "Banking",
    "FPT": "Technology",
    "MWG": "Retail",
    "VNM": "Consumer",
    "HPG": "Materials",
    "VHM": "RealEstate",
    "VIC": "RealEstate",
    "GAS": "Energy",
}

DEFAULT_TICKERS: List[str] = list(SECTOR_MAPPING.keys())

# ==============================================================================
# 4. ALPHA FACTOR SPECIFICATIONS (FUNDAMENTAL & MOMENTUM)
# ==============================================================================
FUNDAMENTAL_REPORTING_LAG_DAYS: int = 30  # Anti-Lookahead bias lag
WINSORIZATION_LIMITS: tuple[float, float] = (0.01, 0.99)
FACTOR_WEIGHTS: Dict[str, float] = {
    "revenue_growth_yoy": 0.25,
    "gross_margin": 0.25,
    "gpa": 0.25,
    "cost_efficiency": 0.15,
    "price_momentum_60d": 0.10,
}

# ==============================================================================
# 5. ECONOMETRIC & DYNAMIC RISK (GJR-GARCH & VAR)
# ==============================================================================
GARCH_P: int = 1
GARCH_O: int = 1  # Leverage term for GJR-GARCH
GARCH_Q: int = 1
GARCH_DIST: str = "skewt"  # Skewed Student-t distribution for fat tails
VAR_SIGNIFICANCE_LEVEL: float = 0.05
CVAR_CONFIDENCE: float = 0.95
TARGET_PORTFOLIO_VOLATILITY: float = 0.12  # 12% annualized target volatility
REBALANCE_FREQUENCY_DAYS: int = 21  # Monthly rebalancing cycle (21 trading days)

# ==============================================================================
# 6. STORAGE & REPOSITORY PATHS
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
REPORTS_DIR: Path = BASE_DIR / "reports_output"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
