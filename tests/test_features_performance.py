from __future__ import annotations

import pandas as pd

from src.features.performance import expected_returns_and_covariance


def test_expected_returns_and_covariance_fallback() -> None:
    close = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 103.0],
            "BBB": [50.0, 51.0, 52.0, 53.0],
        },
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
    )

    mu, cov = expected_returns_and_covariance(close, frequency=252)

    assert mu.index.tolist() == ["AAA", "BBB"]
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)

    returns = close.pct_change().dropna()
    assert mu.round(10).iloc[0] == pd.Series(returns.mean() * 252, index=close.columns).round(10).iloc[0]
    assert not cov.isna().any().any()
