from __future__ import annotations

import pandas as pd

from src.data.cleaner import normalize_datetime_index, clean_ohlcv_frame, build_ohlcv_matrices


def test_datetime_sorting_and_duplicates():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-04"],
            "open": [1, 2, 3, 4],
            "high": [1, 2, 3, 4],
            "low": [1, 2, 3, 4],
            "close": [1, 2, 3, 4],
            "volume": [10, 20, 30, 40],
        }
    )
    cleaned = normalize_datetime_index(raw)
    assert cleaned.index.is_monotonic_increasing
    assert cleaned.index.nunique() == len(cleaned.index)


def test_clean_ohlcv_invalid_values():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "open": [1, -1, 2],
            "high": [1, 2, 2],
            "low": [1, 1, 1],
            "close": [1, 2, -3],
            "volume": [10, 0, 20],
        }
    )
    cleaned = clean_ohlcv_frame(raw)
    assert cleaned.loc[pd.Timestamp("2024-01-02"), "open"] != cleaned.loc[pd.Timestamp("2024-01-02"), "open"]
    assert pd.isna(cleaned.loc[pd.Timestamp("2024-01-02"), "open"])
    assert pd.isna(cleaned.loc[pd.Timestamp("2024-01-03"), "close"])
    assert pd.isna(cleaned.loc[pd.Timestamp("2024-01-02"), "volume"])


def test_build_ohlcv_matrix_column_alignment():
    frame_a = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [10, 11],
        }
    )
    frame_b = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "open": [3, 4],
            "high": [3, 4],
            "low": [3, 4],
            "close": [3, 4],
            "volume": [12, 13],
        }
    )

    matrices = build_ohlcv_matrices({"AAA": frame_a, "BBB": frame_b})
    assert set(matrices["close"].columns) == {"AAA", "BBB"}
    assert set(matrices["open"].index) == {
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    }
