"""Data cleaning helpers for OHLCV data matrices."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


def _pick_date_col(columns: Iterable[str], preferred: str = "date") -> str:
    normalized = {str(c).strip().lower(): c for c in columns}
    if preferred in normalized:
        return normalized[preferred]

    for candidate in ("date", "datetime", "trading_date", "tradingdate"):
        if candidate in normalized:
            return normalized[candidate]

    # fallback: if caller already passed date as index
    return ""


def normalize_datetime_index(
    data: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Normalize date/time and enforce sorted, unique index."""
    frame = data.copy()

    if frame.index.dtype.kind in "M":
        return frame[~frame.index.duplicated(keep="last")].sort_index()

    date_column = _pick_date_col(frame.columns, preferred=date_col.lower())
    if date_column:
        frame = frame.copy()
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.dropna(subset=[date_column])
        frame = frame.loc[~frame[date_column].duplicated(keep="last")]
        frame = frame.set_index(date_column).sort_index()
    else:
        raise ValueError(
            "Could not locate date column. Provide a DataFrame with datetime index "
            "or a date-like column."
        )

    return frame


def clean_ohlcv_frame(
    data: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Normalize one ticker OHLCV frame."""
    frame = normalize_datetime_index(data, date_col=date_col).copy()
    # unify column names
    frame.columns = [str(c).strip().lower() for c in frame.columns]

    for field in OHLCV_FIELDS:
        if field not in frame.columns:
            raise ValueError(f"Missing required field '{field}' in raw OHLCV frame.")

    selected = frame[list(OHLCV_FIELDS)].apply(pd.to_numeric, errors="coerce")
    selected = selected.replace([np.inf, -np.inf], np.nan)
    selected["open"] = selected["open"].mask(selected["open"] <= 0, np.nan)
    selected["high"] = selected["high"].mask(selected["high"] <= 0, np.nan)
    selected["low"] = selected["low"].mask(selected["low"] <= 0, np.nan)
    selected["close"] = selected["close"].mask(selected["close"] <= 0, np.nan)
    selected["volume"] = selected["volume"].mask(selected["volume"] <= 0, np.nan)
    return selected


def build_ohlcv_matrices(
    stock_frames: Mapping[str, pd.DataFrame],
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """Return OHLCV matrices keyed by field."""
    matrices: dict[str, list[pd.DataFrame]] = {f: [] for f in OHLCV_FIELDS}
    aligned: dict[str, dict[str, pd.Series]] = {f: {} for f in OHLCV_FIELDS}

    for ticker, frame in stock_frames.items():
        cleaned = clean_ohlcv_frame(frame, date_col=date_col)
        for field in OHLCV_FIELDS:
            aligned[field][ticker] = cleaned[field]

    for field in OHLCV_FIELDS:
        if not aligned[field]:
            raise ValueError(f"No data collected for field '{field}'.")
        matrices[field] = pd.DataFrame(aligned[field]).sort_index()

    return matrices
