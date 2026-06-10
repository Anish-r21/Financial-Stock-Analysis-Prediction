"""Plotly-based chart utilities for the Streamlit dashboard.

Each function returns a Plotly Figure ready to be shown in Streamlit.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_candlestick(df: pd.DataFrame) -> go.Figure:
    """OHLC candlestick chart with volume bars below.

    Parameters
    - df: DataFrame with columns Open, High, Low, Close, Volume and DatetimeIndex
    """
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    volume_bar = go.Bar(x=df.index, y=df["Volume"], name="Volume", marker=dict(color="lightgrey"), yaxis="y2")
    fig.add_trace(volume_bar)

    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, position=0.15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_moving_averages(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="black")))
    if "MA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA_20"], name="MA_20", line=dict(color="blue")))
    if "MA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA_50"], name="MA_50", line=dict(color="orange")))
    if "EMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA_20", line=dict(color="green")))

    fig.update_layout(title="Price with Moving Averages", margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_bollinger_bands(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="black")))
    if "Bollinger_Upper" in df.columns and "Bollinger_Lower" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_Upper"], name="Upper Band", line=dict(color="rgba(0,0,255,0.2)")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger_Lower"], name="Lower Band", line=dict(color="rgba(0,0,255,0.2)")))
        fig.add_trace(
            go.Scatter(
                x=pd.concat([df.index, df.index[::-1]]),
                y=pd.concat([df["Bollinger_Upper"], df["Bollinger_Lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(173,216,230,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="BB Band",
            )
        )

    fig.update_layout(title="Bollinger Bands", margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI_14", line=dict(color="purple")))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title="RSI (14)", margin=dict(l=40, r=40, t=40, b=40), yaxis=dict(range=[0, 100]))
    return fig


def plot_macd(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="orange")))
        hist = df["MACD"] - df["MACD_Signal"]
        fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram", marker_color="grey"))

    fig.update_layout(title="MACD", margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_predictions(y_true: Sequence[float], y_pred_dict: Dict[str, Sequence[float]], dates: Sequence) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, name="Actual", line=dict(color="black")))
    for name, preds in y_pred_dict.items():
        fig.add_trace(go.Scatter(x=dates, y=preds, name=name))
    fig.update_layout(title="Actual vs Predicted", margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_feature_importance(importances: Dict[str, float]) -> go.Figure:
    items = sorted(importances.items(), key=lambda x: x[1])
    names = [i[0] for i in items]
    vals = [i[1] for i in items]

    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h"))
    fig.update_layout(title="Feature Importance", margin=dict(l=120, r=40, t=40, b=40))
    return fig


def plot_lstm_loss(history) -> go.Figure:
    fig = go.Figure()
    if hasattr(history, "history"):
        h = history.history
        fig.add_trace(go.Scatter(x=list(range(1, len(h["loss"]) + 1)), y=h["loss"], name="loss"))
        if "val_loss" in h:
            fig.add_trace(go.Scatter(x=list(range(1, len(h["val_loss"]) + 1)), y=h["val_loss"], name="val_loss"))
    fig.update_layout(title="LSTM Loss", xaxis_title="Epoch", yaxis_title="MSE", margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_returns_distribution(df: pd.DataFrame, column: str = "Daily_Return") -> go.Figure:
    returns = df[column].dropna()
    fig = px.histogram(returns, nbins=100, title="Daily Returns Distribution", marginal="box")

    # Overlay normal distribution curve
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 200)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    # scale y to histogram height
    y = y * (returns.shape[0] * (x[1] - x[0]))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Normal PDF", line=dict(color="red")))
    return fig
