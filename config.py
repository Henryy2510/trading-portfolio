"""Project configuration for VNQuant V0."""

from __future__ import annotations

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
SPLIT_DATE = "2025-09-07"

DEFAULT_SEED = 11230998
RISK_FREE_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252

INITIAL_CASH = 500_000_000
FEES = 0.001
SLIPPAGE = 0.0005
INITIAL_CASH_NOTIONAL = 10_000_000

MOMENTUM_WINDOW = 20
TOP_QUANTILE = 0.2
REBALANCE_FREQUENCY = "weekly"
REBALANCE_FREQUENCY_DAYS = 21

LOOKBACK_WINDOW = 63
VOLATILITY_WINDOW = 20
EMA_FAST = 12
EMA_SLOW = 26
EMA_SIGNAL = 9

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

# Data source fallback for local experiments
LOCAL_STOCK_CLOSE_CSV = "src/data/stock_data.csv"

# Selection / strategy defaults
MODEL_SELECTION_WEIGHTS = {
    "sharpe_ratio": 0.20,
    "max_drawdown": 0.30,
    "calmar_ratio": 0.30,
    "volatility": 0.20,
}
