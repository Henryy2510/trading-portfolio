"""VN stock data provider abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .cleaner import build_ohlcv_matrices

FetchFunction = Callable[[str, str | date, str | date], pd.DataFrame]


def _coerce_slice_value(value: str | date | None) -> str | date | None:
    if value is None:
        return None

    if isinstance(value, date):
        return value

    return str(value)


# Check ticker legitimacy
def _validate_tickers(requested: Iterable[str], available: Iterable[str]) -> list[str]:
    requested_list = [str(ticker).strip().upper() for ticker in requested]
    available_set = {str(ticker).strip().upper() for ticker in available}
    missing = sorted(set(requested_list) - available_set)
    if missing:
        raise ValueError(f"Missing tickers in source data: {missing}")
    return requested_list


# reformat time of dataframe
def _coerce_frame_source(source: object) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, str | Path):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Data source not found: {path}")

        frame = pd.read_csv(path, parse_dates=True)
        if frame.empty:
            raise ValueError(f"Data source is empty: {path}")
        return frame

    raise TypeError(
        "Unsupported source type. Pass DataFrame, CSV path, or None with provider."
    )


# convert price to numeric
def _normalize_price_matrix(
    frame: pd.DataFrame, tickers: Iterable[str] | None = None
) -> pd.DataFrame:
    frame = frame.copy()

    if frame.index.dtype.kind == "M":
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    elif "date" in [str(col).lower() for col in frame.columns]:
        date_col = next(col for col in frame.columns if str(col).lower() == "date")
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col])
        frame = frame.loc[~frame[date_col].duplicated(keep="last")]
        frame = frame.set_index(date_col).sort_index()
    else:
        raise ValueError(
            "Cannot infer date index from source. CSV should have date index or a 'date' column."
        )

    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_index()

    if tickers is not None:
        requested = _validate_tickers(tickers, frame.columns)
        frame = frame[requested]

    return frame


# Make a dataset suitable for trading and backtest
def split_train_test(
    prices: pd.DataFrame,
    split_date: str | date,
    *,
    include_split_in_train: bool = True,
    include_split_in_test: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a close-matrix into train/test by date boundary."""
    split = pd.Timestamp(split_date)

    if include_split_in_train and include_split_in_test:
        return prices.loc[:split].copy(), prices.loc[split:].copy()

    if include_split_in_train:
        return prices.loc[:split].copy(), prices.loc[:split].iloc[:-1].copy()

    if include_split_in_test:
        return prices.loc[: split - pd.Timedelta(days=1)].copy(), prices.loc[
            split:
        ].copy()

    return prices.loc[: split - pd.Timedelta(days=1)].copy(), prices.loc[
        split + pd.Timedelta(days=1) :
    ].copy()


# Get dataframe with a format of tickers, start end date,...
def load_price_matrix(
    source: pd.DataFrame | str | Path | None = None,
    tickers: Iterable[str] | None = None,
    *,
    start: str | date | None = None,
    end: str | date | None = None,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    provider: VNDataProvider | None = None,
) -> pd.DataFrame:
    """Load close price matrix from local CSV or VNDataProvider."""
    if source is None:
        if provider is None:
            raise ValueError("Provide `source` or `provider` to load market data.")
        if tickers is None:
            raise ValueError("tickers must be provided when loading from provider.")

        raw = provider.get_history(
            tickers=[str(t).upper() for t in tickers],
            start=_coerce_slice_value(start or start_date),
            end=_coerce_slice_value(end or end_date),
        )
        frame = raw["close"]
    else:
        frame = _coerce_frame_source(source)
        frame = _normalize_price_matrix(frame, tickers=tickers)

    start_value = _coerce_slice_value(start or start_date)
    end_value = _coerce_slice_value(end or end_date)
    if start_value is not None:
        frame = frame.loc[pd.Timestamp(start_value) :]
    if end_value is not None:
        frame = frame.loc[: pd.Timestamp(end_value)]

    if frame.empty:
        raise ValueError("No data after date filtering.")

    return frame


def _coerce_frame(raw: object) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        return raw.copy()

    if isinstance(raw, list):
        return pd.DataFrame(raw)

    if isinstance(raw, dict):
        if not raw:
            return pd.DataFrame()
        if all(isinstance(v, dict) for v in raw.values()):
            return pd.DataFrame.from_dict(raw, orient="index")
        return pd.DataFrame(raw)

    return pd.DataFrame(raw)


def _normalize_vnstock_output(result: object) -> pd.DataFrame:
    frame = _coerce_frame(result)
    if frame.empty:
        raise ValueError("VNStock adapter returned empty data.")
    return frame


@dataclass
class vndataprovider:
    """Minimal VNDataProvider wrapper.

    A custom fetcher can be injected for offline or custom vendor use:
    fetcher(ticker, start, end) -> DataFrame
    """

    exchange: str = "HOSE"
    date_col: str = "date"
    fetcher: FetchFunction | None = None
    source: str = "vnstock"
    source_kwargs: dict = field(default_factory=dict)

    def get_universe(self, exchange: str | None = None) -> list[str]:
        raise NotImplementedError(
            "VNDataProvider.get_universe is a placeholder for V0. "
            "Implement when vnstock API discovery is finalized."
        )

    def get_history(
        self,
        tickers: list[str],
        start,
        end,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data and return matrices by field."""
        if not tickers:
            raise ValueError("tickers must not be empty")

        raw_frames: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            raw = self._fetch_single(str(ticker).upper(), start=start, end=end)
            raw_frames[str(ticker).upper()] = raw

        return build_ohlcv_matrices(raw_frames, date_col=self.date_col)

    def _fetch_single(self, ticker: str, start, end) -> pd.DataFrame:
        if self.fetcher is not None:
            frame = self.fetcher(ticker, start, end)
            return _normalize_vnstock_output(frame)

        try:
            import vnstock as vn
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "vnstock is not installed. Install vnstock or pass a custom fetcher."
            ) from exc

        # Try a few adapter shapes from existing vnstock releases.
        adapters: list[tuple[Callable[..., object], tuple[str, ...]]] = []
        if hasattr(vn, "stock_historical_data"):
            adapters.append((vn.stock_historical_data, ("symbol",)))
            adapters.append((vn.stock_historical_data, ("ticker",)))

        if hasattr(vn, "stock"):
            adapters.append((vn.stock, ("symbol",)))

        if hasattr(vn, "history"):
            adapters.append((vn.history, ("symbol",)))

        for adapter, names in adapters:
            for symbol_key in names:
                for kwargs in (
                    {symbol_key: ticker, "from_date": start, "to_date": end},
                    {symbol_key: ticker, "start": start, "end": end},
                ):
                    try:
                        raw = (
                            adapter(**kwargs, **self.source_kwargs)
                            if hasattr(adapter, "__code__")
                            else adapter(**kwargs)
                        )
                        frame = _normalize_vnstock_output(raw)
                        if not frame.empty:
                            return _normalize_vnstock_output(raw)
                    except TypeError:
                        continue
                    except Exception:
                        continue

        raise RuntimeError(
            "Could not infer a supported vnstock API signature for this environment. "
            "Pass VNDataProvider(fetcher=...) with your VN-stock adapter."
        )
