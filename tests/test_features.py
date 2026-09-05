from __future__ import annotations

import pandas as pd

from src.features.momentum import momentum
from src.features.volatility import volatility
from src.features.liquidity import volume_ratio


def test_momentum_calculation():
    close = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [4.0, 4.0, 2.0, 2.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    result = momentum(close, window=2)
    expected = close.pct_change(2)
    assert result.equals(expected)


def test_volatility_calculation():
    close = pd.DataFrame({"A": [1, 2, 3, 4, 5]}, dtype=float)
    result = volatility(close, window=2)
    expected = close.pct_change().rolling(2).std()
    assert result.equals(expected)


def test_volume_ratio_calculation():
    volume = pd.DataFrame({"A": [10, 10, 20, 40]}, dtype=float)
    result = volume_ratio(volume, window=2)
    expected = volume / volume.rolling(2).mean()
    assert result.equals(expected)
