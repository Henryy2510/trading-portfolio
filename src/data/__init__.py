"""Data ingestion and processing layer for VNQuant."""

from src.data.provider import load_price_matrix, split_train_test
from src.data.data_loader import (
    FundamentalData,
    MarketData,
    RealisticSyntheticMarketGenerator,
    load_dataset,
)

__all__ = [
    "load_price_matrix",
    "split_train_test",
    "FundamentalData",
    "MarketData",
    "RealisticSyntheticMarketGenerator",
    "load_dataset",
]
