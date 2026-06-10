"""data.fetcher
Functions to fetch OHLCV data from Yahoo Finance using yfinance.

This module uses Streamlit's caching to avoid re-downloading data on re-renders.
"""
from __future__ import annotations

from typing import Optional
import datetime as dt

import pandas as pd
import yfinance as yf
import streamlit as st

# Constants
CACHE_TTL_SECONDS = 24 * 60 * 60  # 1 day


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_data(ticker: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Download OHLCV data for a given ticker and date range.

    Parameters
    - ticker: Stock symbol understood by Yahoo Finance (e.g. 'AAPL', 'RELIANCE.NS')
    - start: Start date as ISO string (YYYY-MM-DD) or any string parsable by pandas.to_datetime
    - end: End date as ISO string (defaults to today if None)

    Returns
    - pd.DataFrame: DataFrame indexed by DatetimeIndex with columns ['Open','High','Low','Close','Volume']

    Raises
    - ValueError: for invalid inputs or when no data is returned
    - ConnectionError: for network / download related failures
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker symbol must be a non-empty string")

    try:
        start_dt = pd.to_datetime(start)
    except Exception:
        raise ValueError(f"Invalid start date: {start}")

    end_dt = pd.to_datetime(end) if end is not None else pd.Timestamp(dt.date.today())

    if start_dt >= end_dt:
        raise ValueError("Start date must be before end date")

    ticker = ticker.strip()

    try:
        df = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"),
                         progress=False, threads=False)
    except Exception as exc:  # network or yfinance internal errors
        raise ConnectionError(f"Failed to fetch data for {ticker}: {exc}")

    if df is None or df.empty:
        raise ValueError(f"No data found for {ticker} between {start_dt.date()} and {end_dt.date()}")

    # Ensure datetime index and expected column names
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Normalize column names (yfinance may return 'Adj Close')
    df = df.rename(columns=lambda c: c.replace(" ", "_"))

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Fetched data missing required columns; got: {list(df.columns)}")

    return df[required_cols].copy()
