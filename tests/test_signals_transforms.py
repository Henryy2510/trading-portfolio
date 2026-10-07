from __future__ import annotations

import pandas as pd

from src.signals import generate_technical_signals
from src.signals.transforms import calculate_bollinger_bands, calculate_macd, calculate_rsi


def test_technical_signal_bounds() -> None:
    data = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 101, 103, 104, 103, 105, 104, 108, 110, 112],
            "BBB": [50, 50, 51, 52, 51, 53, 54, 55, 54, 56, 58, 59],
        },
        index=pd.date_range("2025-01-01", periods=12, freq="D"),
    )

    signals = generate_technical_signals(data)
    assert signals.shape == data.shape
    assert signals.min().min() >= -1
    assert signals.max().max() <= 1


def test_supporting_signal_helpers_have_same_shape() -> None:
    series = pd.Series([1, 2, 3, 2, 4, 5, 6], dtype=float)
    rsi = calculate_rsi(series, window=3)
    macd, macd_signal = calculate_macd(series, fast=3, slow=5, signal=2)
    upper, mid, lower = calculate_bollinger_bands(series, window=3)

    assert list(rsi.shape) == list(series.shape)
    assert list(macd.shape) == list(series.shape)
    assert list(macd_signal.shape) == list(series.shape)
    assert list(upper.shape) == list(series.shape)
    assert list(mid.shape) == list(series.shape)
    assert list(lower.shape) == list(series.shape)
