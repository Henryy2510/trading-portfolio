"""VN stock data provider abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Mapping

import pandas as pd

from .cleaner import build_ohlcv_matrices


FetchFunction = Callable[[str, str | date, str | date], pd.DataFrame]


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
class VNDataProvider:
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
                        raw = adapter(**kwargs, **self.source_kwargs) if hasattr(
                            adapter, "__code__"
                        ) else adapter(**kwargs)
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
