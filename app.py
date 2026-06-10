"""Streamlit app entry point for Financial Stock Analysis & Prediction.

This file wires together data fetching, feature engineering, models, evaluation,
and visualization modules. It keeps minimal logic and delegates work to other
modules; session state is used to persist data and trained models.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import datetime as dt
import time

import pandas as pd
import streamlit as st

from data.fetcher import fetch_data
from features.engineering import engineer_features
from models.linear_model import LinearRegressionModel
from models.random_forest import RandomForestModel
from models.lstm_model import LSTMModel, LOOKBACK
from evaluation.metrics import evaluate_model, compare_models
from visuals.charts import (
    plot_candlestick,
    plot_moving_averages,
    plot_bollinger_bands,
    plot_rsi,
    plot_macd,
    plot_predictions,
    plot_feature_importance,
    plot_lstm_loss,
    plot_returns_distribution,
)

# App-level constants
DEFAULT_TICKER = "AAPL"
DEFAULT_MODELS = ["Linear Regression", "Random Forest", "LSTM"]
TEST_SIZE = 0.2


def init_session_state() -> None:
    """Initialize required keys in Streamlit session_state."""
    keys = {
        "raw_data": None,
        "engineered": None,
        "models": {},
        "results": {},
        "histories": {},
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main() -> None:
    st.set_page_config(page_title="Financial Stock Predictor", layout="wide")
    init_session_state()

    st.sidebar.title("Controls")
    ticker = st.sidebar.text_input("Ticker", value=DEFAULT_TICKER)
    today = dt.date.today()
    default_start = today - dt.timedelta(days=365 * 3)
    start_date, end_date = st.sidebar.date_input("Date range", value=(default_start, today))

    model_choices = st.sidebar.multiselect("Models to train", options=DEFAULT_MODELS, default=DEFAULT_MODELS)

    if st.sidebar.button("Load Data"):
        with st.spinner("Fetching data..."):
            try:
                df = fetch_data(ticker.strip(), start=start_date.isoformat(), end=end_date.isoformat())
                st.session_state["raw_data"] = df
                st.success(f"Loaded {ticker} data: {df.shape[0]} rows")
            except Exception as exc:
                st.session_state["raw_data"] = None
                st.error(f"Failed to load data: {exc}")

    if st.session_state.get("raw_data") is None:
        st.info("No data loaded. Use the sidebar and click 'Load Data'.")
        return

    # Engineer features once data is loaded
    if st.session_state.get("engineered") is None:
        try:
            engineered = engineer_features(st.session_state["raw_data"])
            st.session_state["engineered"] = engineered
        except Exception as exc:
            st.error(f"Feature engineering failed: {exc}")
            return

    df = st.session_state["engineered"]

    # Sidebar train button
    if st.sidebar.button("Train Models"):
        selected = model_choices
        if not selected:
            st.sidebar.error("Select at least one model to train")
        else:
            with st.spinner("Training models..."):
                progress = st.sidebar.progress(0)
                try:
                    train_and_evaluate(df, ticker, selected, progress)
                    st.sidebar.success("Training complete")
                except Exception as exc:
                    st.sidebar.error(f"Training failed: {exc}")

    # Show small data summary
    st.sidebar.markdown("---")
    st.sidebar.write("Data summary:")
    st.sidebar.write(f"Rows: {df.shape[0]}")
    st.sidebar.write(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    st.sidebar.write(f"Latest Close: {df['Close'].iloc[-1]:.2f}")

    # Tabs
    tabs = st.tabs(["Market Overview", "Technical Indicators", "Predictions", "Raw Data"])

    with tabs[0]:
        st.header("Market Overview")
        st.plotly_chart(plot_candlestick(st.session_state["raw_data"]))

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"{df['Close'].iloc[-1]:.2f}")
        c2.metric("52-week High", f"{df['Close'].rolling(window=252).max().iloc[-1]:.2f}")
        c3.metric("52-week Low", f"{df['Close'].rolling(window=252).min().iloc[-1]:.2f}")

        st.plotly_chart(plot_returns_distribution(df))

    with tabs[1]:
        st.header("Technical Indicators")
        st.plotly_chart(plot_moving_averages(df))
        st.markdown("**Moving averages**: Short- and long-term simple moving averages help show trend direction.")

        st.plotly_chart(plot_bollinger_bands(df))
        st.markdown("**Bollinger Bands**: Measure volatility; price near bands may indicate overbought/oversold.")

        st.plotly_chart(plot_rsi(df))
        st.markdown("**RSI**: Relative Strength Index; values above 70 are often considered overbought, below 30 oversold.")

        st.plotly_chart(plot_macd(df))
        st.markdown("**MACD**: Momentum indicator showing trend direction and strength.")

    with tabs[2]:
        st.header("Predictions")
        if not st.session_state.get("models"):
            st.info("No trained models. Train models from the sidebar.")
        else:
            results = st.session_state.get("results", {})
            models = st.session_state.get("models", {})

            # Show comparison table
            if results:
                df_results = compare_models(results)
                st.dataframe(df_results)

            # Plot predictions
            y_true = None
            y_pred_dict = {}
            dates = None

            # reconstruct test split used during training
            split = int((1 - TEST_SIZE) * len(df))
            X = df.drop(columns=["Target"]) if "Target" in df.columns else df.copy()
            y = df["Target"].values
            dates = df.index[split:]

            for name, mdl in models.items():
                try:
                    if name == "LSTM":
                        # For LSTM, provide last LOOKBACK rows from train + test so sequences align
                        X_train = X.iloc[:split]
                        X_test = X.iloc[split:]
                        if len(X_train) < LOOKBACK:
                            st.warning("Not enough training data for LSTM lookback; skipping LSTM predictions")
                            continue
                        X_for_pred = pd.concat([X_train.tail(LOOKBACK), X_test])
                        preds = mdl.predict(X_for_pred)
                        # preds correspond to X_test length
                    else:
                        X_test = X.iloc[split:]
                        preds = mdl.predict(X_test)

                    y_pred_dict[name] = preds
                    y_true = y[split:]
                except Exception as exc:
                    st.error(f"Prediction failed for {name}: {exc}")

            if y_true is not None and y_pred_dict:
                st.plotly_chart(plot_predictions(y_true, y_pred_dict, dates))

            # Show model-specific artifacts
            if "LSTM" in st.session_state.get("histories", {}):
                st.plotly_chart(plot_lstm_loss(st.session_state["histories"]["LSTM"]))

            if "Random Forest" in st.session_state.get("models", {}):
                rf = st.session_state["models"]["Random Forest"]
                params = rf.get_params()
                if "feature_importances" in params:
                    st.plotly_chart(plot_feature_importance(params["feature_importances"]))

            # Next day price metrics
            if y_pred_dict:
                next_vals = {name: float(preds[-1]) for name, preds in y_pred_dict.items()}
                cols = st.columns(len(next_vals))
                for (n, v), col in zip(next_vals.items(), cols):
                    col.metric(f"Next day ({n})", f"{v:.2f}")

    with tabs[3]:
        st.header("Raw Data")
        st.dataframe(df)
        csv = df.to_csv(index=True)
        st.download_button("Download CSV", csv, file_name=f"{ticker}_engineered.csv")


def train_and_evaluate(df: pd.DataFrame, ticker: str, selected_models: List[str], progress: Optional[st.progress] = None) -> None:
    """Train selected models on engineered data, store models and results in session_state.

    Parameters
    - df: engineered DataFrame with `Target` column
    - ticker: ticker symbol (for messaging)
    - selected_models: list of model names to train
    - progress: optional st.progress instance to update progress
    """
    # Prepare data
    if "Target" not in df.columns:
        raise ValueError("Engineered DataFrame must contain 'Target' column")

    split = int((1 - TEST_SIZE) * len(df))
    X = df.drop(columns=["Target"]) 
    y = df["Target"].values

    models: Dict[str, object] = {}
    results: Dict[str, Dict] = {}
    histories: Dict[str, any] = {}

    total = len(selected_models)
    step = 0

    for name in selected_models:
        step += 1
        if progress:
            progress.progress(int(100 * (step - 1) / total))

        if name == "Linear Regression":
            model = LinearRegressionModel()
            model.train(X.iloc[:split], y[:split])
            preds = model.predict(X.iloc[split:])
            models[name] = model
            results[name] = evaluate_model(y[split:], preds, name)

        elif name == "Random Forest":
            model = RandomForestModel()
            model.train(X.iloc[:split], y[:split])
            preds = model.predict(X.iloc[split:])
            models[name] = model
            results[name] = evaluate_model(y[split:], preds, name)

        elif name == "LSTM":
            st.warning("LSTM training may take 1–3 minutes depending on data size.")
            model = LSTMModel()
            # LSTM expects raw arrays; train on full X_train and y_train
            history = model.train(X.iloc[:split], y[:split])
            # For predictions, pass last LOOKBACK rows from train + X_test so alignment matches
            X_train = X.iloc[:split]
            X_test = X.iloc[split:]
            X_for_pred = pd.concat([X_train.tail(LOOKBACK), X_test])
            preds = model.predict(X_for_pred)
            models[name] = model
            histories[name] = history
            results[name] = evaluate_model(y[split:], preds, name)

        else:
            st.warning(f"Unknown model selection: {name}")

        # small delay so progress UI updates smoothly
        time.sleep(0.2)

    # final progress
    if progress:
        progress.progress(100)

    st.session_state["models"] = models
    st.session_state["results"] = results
    st.session_state["histories"] = histories


if __name__ == "__main__":
    main()
