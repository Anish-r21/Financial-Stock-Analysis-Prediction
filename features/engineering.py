"""Feature engineering utilities for financial time series.

Provides `engineer_features` to add technical indicators used by models.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np

# Constants
MA_SHORT = 20
MA_LONG = 50
EMA_20 = 20
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_STD = 2
VOL_WINDOW = 20


def _compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute the Relative Strength Index (RSI) for a price series.

    Parameters
    - series: Series of prices (typically Close)
    - period: lookback period for RSI (default 14)

    Returns
    - pd.Series: RSI values (0-100)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing via exponential moving average
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0)
    return rsi


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and a next-day target to the DataFrame.

    Parameters
    - df: DataFrame containing at least a `Close` column and a DatetimeIndex

    Returns
    - pd.DataFrame: copy of input with additional columns:
      `MA_20`, `MA_50`, `EMA_20`, `RSI_14`, `MACD`, `MACD_Signal`,
      `Bollinger_Upper`, `Bollinger_Lower`, `Daily_Return`, `Volatility_20`, `Target`

    Notes
    - Rows with NaN resulting from rolling calculations are dropped before returning.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column")

    data = df.copy()

    # Moving averages
    data[f"MA_{MA_SHORT}"] = data["Close"].rolling(window=MA_SHORT, min_periods=MA_SHORT).mean()
    data[f"MA_{MA_LONG}"] = data["Close"].rolling(window=MA_LONG, min_periods=MA_LONG).mean()

    # Exponential moving average
    data["EMA_20"] = data["Close"].ewm(span=EMA_20, adjust=False).mean()

    # RSI
    data["RSI_14"] = _compute_rsi(data["Close"], period=RSI_PERIOD)

    # MACD
    ema_fast = data["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = data["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    data["MACD"] = ema_fast - ema_slow
    data["MACD_Signal"] = data["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()

    # Bollinger Bands
    rolling_ma = data["Close"].rolling(window=BB_WINDOW, min_periods=BB_WINDOW).mean()
    rolling_std = data["Close"].rolling(window=BB_WINDOW, min_periods=BB_WINDOW).std()
    data["Bollinger_Upper"] = rolling_ma + (BB_STD * rolling_std)
    data["Bollinger_Lower"] = rolling_ma - (BB_STD * rolling_std)

    # Returns and volatility
    data["Daily_Return"] = data["Close"].pct_change()
    data["Volatility_20"] = data["Daily_Return"].rolling(window=VOL_WINDOW, min_periods=VOL_WINDOW).std()

    # Target: next day's closing price
    data["Target"] = data["Close"].shift(-1)

    # Drop rows with NaN created by rolling/shift operations
    data = data.dropna().copy()

    return data
