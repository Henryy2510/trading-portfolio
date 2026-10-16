from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from src.data import load_price_matrix, split_train_test
from src.data.provider import vndataprovider


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


def test_vnstock_provider_normalizes_and_persists_ohlcv(tmp_path: Path) -> None:
    def fetcher(ticker: str, start: str, end: str) -> pd.DataFrame:
        assert (ticker, start, end) == ("AAA", "2025-01-01", "2025-01-03")
        return pd.DataFrame(
            {
                "time": ["2025-01-02", "2025-01-01"],
                "open": [11, 10],
                "high": [12, 11],
                "low": [10, 9],
                "close": [11.5, 10.5],
                "volume": [1100, 1000],
            }
        )

    provider = vndataprovider(
        fetcher=fetcher,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )
    matrices = provider.get_history(["AAA"], "2025-01-01", "2025-01-03")

    assert list(matrices["close"].columns) == ["AAA"]
    assert matrices["close"].index.is_monotonic_increasing
    assert matrices["close"].iloc[0, 0] == 10.5
    assert list((tmp_path / "raw").glob("AAA_ohlcv_*.csv"))
    assert list((tmp_path / "processed").glob("close_matrix_*.csv"))


def test_vnstock_provider_calls_unified_market_api(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    class BoundEquity:
        def ohlcv(self, *, start: str, end: str) -> pd.DataFrame:
            calls.append(("AAA", start, end))
            return pd.DataFrame(
                {
                    "time": ["2025-01-01"],
                    "open": [10],
                    "high": [11],
                    "low": [9],
                    "close": [10.5],
                    "volume": [1000],
                }
            )

    class EquityDomain:
        def __call__(self, ticker: str) -> BoundEquity:
            assert ticker == "AAA"
            return BoundEquity()

    class Market:
        def __init__(self) -> None:
            self.equity = EquityDomain()

    monkeypatch.setitem(sys.modules, "vnstock", SimpleNamespace(Market=Market))

    frame = vndataprovider()._fetch_single("AAA", "2025-01-01", "2025-01-02")

    assert calls == [("AAA", "2025-01-01", "2025-01-02")]
    assert frame.loc[0, "close"] == 10.5
