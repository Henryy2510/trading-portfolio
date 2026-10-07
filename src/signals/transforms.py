"""Signal transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clip_signal(signal: pd.DataFrame, lower: float = -1.0, upper: float = 1.0) -> pd.DataFrame:
    """Clip signal to a bounded interval."""
    return signal.clip(lower=lower, upper=upper)


def calculate_rsi(prices: pd.Series | pd.DataFrame, window: int = 14) -> pd.Series | pd.DataFrame:
    """Calculate RSI score from price series or matrix."""
    if isinstance(prices, pd.DataFrame):
        return prices.apply(lambda s: calculate_rsi(s, window=window))

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    prices: pd.Series | pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series] | tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate MACD and signal line."""
    if isinstance(prices, pd.DataFrame):
        macd_line = prices.apply(lambda s: calculate_macd(s, fast=fast, slow=slow, signal=signal)[0])
        signal_line = prices.apply(lambda s: calculate_macd(s, fast=fast, slow=slow, signal=signal)[1])
        return macd_line, signal_line

    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line


def calculate_bollinger_bands(
    prices: pd.Series | pd.DataFrame,
    window: int = 20,
    num_std: float = 2,
) -> tuple[pd.Series, pd.Series, pd.Series] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate Bollinger Bands with rolling mean and volatility."""
    if isinstance(prices, pd.DataFrame):
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        return sma + std * num_std, sma, sma - std * num_std

    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band


def generate_technical_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Generate bounded technical scores in [-1, 1] for each ticker."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a DataFrame of close prices.")

    signals_df = pd.DataFrame(index=data.index, columns=data.columns)

    for ticker in data.columns:
        prices = data[ticker]
        signal_score = pd.Series(0, index=data.index, dtype=float)

        # RSI signals
        rsi = calculate_rsi(prices)
        volatility = prices.rolling(20).std()
        signal_score += np.where(rsi < 30 - (volatility * 10), 1, 0)
        signal_score += np.where(rsi > 70 + (volatility * 10), -1, 0)

        # MACD signals
        macd_line, signal_line = calculate_macd(prices)
        signal_score += np.where((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)), 1, 0)
        signal_score += np.where((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1)), -1, 0)

        macd_histogram = macd_line - signal_line
        macd_momentum = macd_histogram - macd_histogram.shift(1)
        signal_score += np.where(
            (macd_line > signal_line) & (macd_histogram > 0) & (macd_momentum > 0),
            2,
            0,
        )

        # Bollinger Bands signals
        upper_band, _, lower_band = calculate_bollinger_bands(prices)
        signal_score += np.where(prices < lower_band, 1, 0)
        signal_score += np.where(prices > upper_band, -1, 0)

        # MA crossover
        sma_20 = prices.rolling(window=20).mean()
        sma_50 = prices.rolling(window=50).mean()
        signal_score += np.where((sma_20 > sma_50) & (sma_20.shift(1) <= sma_50.shift(1)), 1, 0)
        signal_score += np.where((sma_20 < sma_50) & (sma_20.shift(1) >= sma_50.shift(1)), -1, 0)

        # Price momentum
        returns_20d = prices.pct_change(10)
        signal_score += np.where(returns_20d > 0.05, 1, 0)
        signal_score += np.where(returns_20d < -0.05, -1, 0)

        # Drawdown + stop-loss style risk signals
        rolling_high_8d = prices.rolling(window=8, min_periods=1).max()
        drawdown_8d = (prices - rolling_high_8d) / rolling_high_8d
        signal_score += np.where(drawdown_8d <= -0.05, -2, 0)

        rolling_high_10d = prices.rolling(window=10, min_periods=1).max()
        stop_loss_triggered = prices < (rolling_high_10d * 0.9)
        signal_score += np.where(stop_loss_triggered, -3, 0)

        signals_df[ticker] = np.clip(signal_score / 7, -1, 1)

    return signals_df
