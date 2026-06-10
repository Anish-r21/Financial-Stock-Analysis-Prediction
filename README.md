# Financial Stock Analysis & Prediction

This project is a Streamlit web application for fetching stock data, performing
exploratory data analysis, engineering technical indicators, training several
machine learning models (Linear Regression, Random Forest, LSTM), evaluating
their performance, and visualizing results in an interactive dashboard.

Features
- Fetch OHLCV data from Yahoo Finance with `yfinance`
- Feature engineering: moving averages, EMA, RSI, MACD, Bollinger Bands, returns, volatility
- Models: Linear Regression, Random Forest, LSTM
- Evaluation: MAE, RMSE, R², MAPE and comparison table
- Interactive Plotly charts and CSV export

Quick start
1. Create and activate a Python 3.10+ virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Notes
- LSTM training can take 1–3 minutes depending on data size and hardware.
- The app is designed to work with US tickers (e.g. `AAPL`), Indian tickers (e.g. `RELIANCE.NS`), and crypto (e.g. `BTC-USD`).

Screenshot
![screenshot](docs/screenshot.png)
#Financial Stock Analysis & Prediction

