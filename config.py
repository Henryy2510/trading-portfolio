"""Project configuration for VNQuant V0."""

from __future__ import annotations

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

INITIAL_CASH = 500_000_000
FEES = 0.001
SLIPPAGE = 0.0005

MOMENTUM_WINDOW = 20
TOP_QUANTILE = 0.2
REBALANCE_FREQUENCY = "weekly"

DEFAULT_TICKERS = [
    "FPT",
    "HPG",
    "VCB",
    "VNM",
    "MWG",
    "SSI",
    "MBB",
    "TCB",
    "VIC",
    "VHM",
]

TRADING_DAYS_PER_YEAR = 252
