"""Signal transforms."""

from __future__ import annotations

import pandas as pd


def clip_signal(signal: pd.DataFrame, lower: float = -1.0, upper: float = 1.0) -> pd.DataFrame:
    """Clip signal to a bounded interval."""
    return signal.clip(lower=lower, upper=upper)
