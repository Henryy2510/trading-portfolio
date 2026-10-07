from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import load_price_matrix, split_train_test


def test_load_price_matrix_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "prices.csv"
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "AAA": [100.0, 101.0, 102.0],
            "BBB": [50.0, 49.5, 50.5],
        }
    )
    df.to_csv(csv_path, index=False)

    prices = load_price_matrix(source=csv_path, tickers=["AAA", "BBB"])

    assert not prices.empty
    assert list(prices.columns) == ["AAA", "BBB"]
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_split_train_test_overlap_date() -> None:
    close = pd.DataFrame(
        {
            "AAA": [1.0, 2.0, 3.0, 4.0],
            "BBB": [2.0, 3.0, 4.0, 5.0],
        },
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
    )

    train, test = split_train_test(close, "2025-01-03")

    assert train.index[-1] == pd.Timestamp("2025-01-03")
    assert test.index[0] == pd.Timestamp("2025-01-03")
