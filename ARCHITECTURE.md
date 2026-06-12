# Financial Stock Analysis & Prediction Platform — Complete Architecture

## Project Overview

This is a **Streamlit-based web application** that fetches real stock data, performs technical analysis, engineers features, trains machine learning models (Linear Regression, Random Forest, LSTM), compares their performance, and displays everything in an interactive dashboard.

**Tech Stack:**
- **UI**: Streamlit (Python web framework)
- **Data Fetching**: yfinance (Yahoo Finance API)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly (interactive charts)
- **ML Models**: scikit-learn (Linear Regression, Random Forest), TensorFlow/Keras (LSTM)
- **Evaluation**: Custom metrics (MAE, RMSE, R², MAPE)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT APP (app.py)                   │
│                  (Main UI & Orchestration)                  │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬──────────────┬────────────┐
    │            │            │              │            │
    ▼            ▼            ▼              ▼            ▼
┌─────────┐ ┌──────────┐ ┌─────────────┐ ┌────────┐ ┌─────────┐
│  Data   │ │ Features │ │   Models    │ │ Evaluat│ │ Visuals │
│ Fetcher │ │Engineer. │ │ (3 types)   │ │ Metrics│ │ Charts  │
└─────────┘ └──────────┘ └─────────────┘ └────────┘ └─────────┘
     │            │             │            │          │
   yf.download   EMA, RSI,     LR, RF,    Metrics    Plotly
                  MACD, BB      LSTM        Table      Figs
```

---

## Module Breakdown

### 1. **data/fetcher.py** — Data Fetching
**Purpose**: Download OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.

**Key Function**: `fetch_data(ticker, start, end)`
- Takes: ticker (e.g., "AAPL", "RELIANCE.NS", "BTC-USD"), date range
- Returns: Pandas DataFrame with DatetimeIndex, columns [Open, High, Low, Close, Volume]
- Error handling: Graceful messages for invalid tickers, network failures, no data
- Caching: Uses `@st.cache_data` to avoid re-downloading on re-renders

**Example**:
```python
df = fetch_data("AAPL", "2020-01-01", "2023-12-31")
# Returns DataFrame with 1000+ rows of OHLCV data
```

---

### 2. **features/engineering.py** — Feature Engineering
**Purpose**: Add technical indicators to the raw OHLCV data.

**Key Function**: `engineer_features(df)`
- Adds **10+ technical indicators** to the DataFrame:
  - **Moving Averages**: MA_20 (20-day), MA_50, EMA_20 (exponential)
  - **Momentum**: RSI_14 (Relative Strength Index), MACD, MACD_Signal
  - **Volatility**: Bollinger Bands (upper/lower), Volatility_20 (rolling std of returns)
  - **Returns**: Daily_Return (pct change)
  - **Target**: Next day's closing price (what models predict)
- Removes NaN rows created by rolling calculations
- Returns: Enriched DataFrame (shape: [n_samples, 15+ cols])

**Example**:
```python
engineered = engineer_features(raw_data)
# Adds columns: MA_20, MA_50, EMA_20, RSI_14, MACD, MACD_Signal,
#               Bollinger_Upper, Bollinger_Lower, Daily_Return, Volatility_20, Target
```

---

### 3. **models/** — Three ML Model Implementations

#### **3a. Linear Regression** (`models/linear_model.py`)
- **Class**: `LinearRegressionModel`
- **Architecture**: sklearn's LinearRegression with StandardScaler
- **Methods**:
  - `train(X_train, y_train)`: Fit scaler, then fit linear model
  - `predict(X_test)`: Scale features, return predictions
  - `get_params()`: Return coefficients (feature importance)
- **Speed**: Very fast (~milliseconds)
- **Interpretability**: High (see feature weights)

#### **3b. Random Forest** (`models/random_forest.py`)
- **Class**: `RandomForestModel`
- **Architecture**: sklearn's RandomForestRegressor (200 trees, max_depth=10)
- **Methods**: Same as Linear Regression
- **Feature Importance**: Exposed via `get_params()` (tree-based importance)
- **Speed**: Fast (~seconds)
- **Robustness**: Better at capturing non-linear relationships

#### **3c. LSTM** (`models/lstm_model.py`)
- **Class**: `LSTMModel`
- **Architecture** (Keras Sequential):
  ```
  Input (60 timesteps × features)
  → LSTM(128, return_sequences=True)
  → Dropout(0.2)
  → LSTM(64)
  → Dropout(0.2)
  → Dense(32, activation='relu')
  → Dense(1)  [output: next day price]
  ```
- **Sequence Creation**: `create_sequences(features, target, lookback=60)`
  - Uses last 60 days of features to predict next day price
- **Scaling**: Separate MinMaxScaler for features and target
- **Training**: 30 epochs, batch_size=32, validation_split=0.1
- **Speed**: Slow (1-3 minutes depending on data size)
- **Strength**: Captures temporal dependencies in time series

**Common Usage Pattern**:
```python
model = LinearRegressionModel()  # or RandomForestModel(), LSTMModel()
model.train(X_train, y_train)
preds = model.predict(X_test)
params = model.get_params()
```

---

### 4. **evaluation/metrics.py** — Model Evaluation
**Purpose**: Compute regression metrics and compare models.

**Key Functions**:
- `evaluate_model(y_true, y_pred, model_name)`: Returns dict with:
  - **MAE** (Mean Absolute Error): Average absolute deviation
  - **RMSE** (Root Mean Squared Error): Square root of average squared error
  - **R²** (Coefficient of Determination): Goodness of fit (0-1, higher is better)
  - **MAPE** (Mean Absolute Percentage Error): Percentage error

- `compare_models(results_dict)`: Takes dict of {model_name: metrics}, returns DataFrame sorted by RMSE

**Example**:
```python
metrics = evaluate_model(y_test, preds, "Linear Regression")
# Returns: {"model": "Linear Regression", "MAE": 5.23, "RMSE": 7.15, "R2": 0.89, "MAPE": 1.2}
```

---

### 5. **visuals/charts.py** — Interactive Visualization
**Purpose**: Create Plotly charts for the dashboard.

**Chart Functions** (each returns a `plotly.graph_objects.Figure`):
1. **plot_candlestick(df)**: OHLC candlestick + volume bars
2. **plot_moving_averages(df)**: Close price + MA_20, MA_50, EMA_20 overlaid
3. **plot_bollinger_bands(df)**: Close price + shaded BB region
4. **plot_rsi(df)**: RSI line with overbought (70) / oversold (30) bands
5. **plot_macd(df)**: MACD, Signal line, and histogram
6. **plot_predictions(y_true, y_pred_dict, dates)**: Actual vs all predicted prices
7. **plot_feature_importance(importances)**: Horizontal bar chart (RF only)
8. **plot_lstm_loss(history)**: Training & validation loss curves
9. **plot_returns_distribution(df)**: Histogram of daily returns + normal PDF overlay

**Example**:
```python
fig = plot_candlestick(df)
st.plotly_chart(fig)  # Display in Streamlit
```

---

### 6. **app.py** — Main Streamlit Application
**Purpose**: Orchestrate everything into an interactive web dashboard.

**Architecture**: Session state for persistence + sidebar controls + 4 tabs

#### **Sidebar Controls**:
- **Ticker input**: Enter stock symbol (default: "AAPL")
- **Date range picker**: Select time period (default: 3 years back)
- **Model selection**: Checkboxes for Linear Regression, Random Forest, LSTM
- **Load Data button**: Fetch & cache data from yfinance
- **Train Models button**: Train selected models on test data
- **Data summary**: Show row count, date range, latest close price

#### **Tab 1: Market Overview**
- Candlestick chart (OHLCV)
- Metrics cards: Current Price, 52-week High, 52-week Low, Average Volume
- Daily returns distribution histogram

#### **Tab 2: Technical Indicators**
- Moving averages chart + explanation
- Bollinger Bands chart + explanation
- RSI chart + explanation (overbought/oversold)
- MACD chart + explanation

#### **Tab 3: Predictions**
- Comparison table (MAE, RMSE, R², MAPE for all trained models)
- Actual vs Predicted prices chart (all models on one plot)
- LSTM loss curve (if LSTM trained)
- Random Forest feature importance bar chart (if RF trained)
- "Next Day Price" metric cards for each model

#### **Tab 4: Raw Data**
- Full engineered DataFrame displayed
- CSV download button

**Data Flow in app.py**:
```
User Input (Sidebar)
    ↓
[Load Data] → fetch_data() → engineer_features() → store in session_state
    ↓
User selects models & clicks [Train Models]
    ↓
For each model:
  - Split: 80% train, 20% test
  - model.train(X_train, y_train)
  - preds = model.predict(X_test)
  - results[name] = evaluate_model(y_test, preds, name)
  - store in session_state
    ↓
Tabs render charts & tables from session_state
```

**Session State Keys**:
- `raw_data`: Original DataFrame from yfinance
- `engineered`: DataFrame with features + target
- `models`: Dict of trained model objects
- `results`: Dict of evaluation metrics
- `histories`: Dict of LSTM training histories

---

## End-to-End Usage Example

### Step 1: Start the app
```bash
streamlit run app.py
```

### Step 2: Load data
- Enter ticker: `RELIANCE.NS` (Indian stock)
- Pick date range: Last 2 years
- Click "Load Data"
- App fetches ~500 trading days of data, engineers features

### Step 3: Select & train models
- Check: Linear Regression, Random Forest, LSTM
- Click "Train Models"
- App performs 80/20 train/test split:
  - Trains each model on 80% of data
  - Evaluates on 20% test set
  - Stores metrics & predictions

### Step 4: Explore results (3 tabs)
- **Market Overview**: View candlestick, returns distribution
- **Technical Indicators**: Study RSI, MACD, Bollinger Bands
- **Predictions**: Compare model performance, see next-day predictions
- **Raw Data**: Export engineered features as CSV

---

## Key Design Patterns

### 1. **Session State Persistence**
- Prevents re-fetching data / re-training on Streamlit re-renders
- Keeps data & models in memory across tab switches

### 2. **Error Handling**
- Every user input wrapped in try/except
- User-friendly error messages (st.error) instead of raw tracebacks
- Graceful fallbacks (e.g., skip LSTM if TensorFlow not installed)

### 3. **Modular Architecture**
- No logic in app.py; pure delegation to modules
- Each module has single responsibility (fetch, engineer, train, evaluate, visualize)
- Easy to test, modify, or extend

### 4. **Feature Engineering Reproducibility**
- Same `engineer_features()` applied before training and deployment
- All thresholds (window sizes, moving average periods) as constants
- Easy to adjust globally

### 5. **Three Model Diversity**
- **Linear**: Fast, interpretable (coefficients)
- **Random Forest**: Robust, non-linear (feature importance)
- **LSTM**: Deep learning, temporal (slow but powerful)
- Allows user to pick best trade-off for their use case

---

## Data Types & Shapes

| Stage | Shape | Columns | Example |
|-------|-------|---------|---------|
| **Raw (yfinance)** | (N, 5) | [Open, High, Low, Close, Volume] | (252, 5) for 1 year |
| **Engineered** | (N-60, 15) | +MA_20, MA_50, EMA_20, RSI_14, MACD, MACD_Signal, BB_Upper, BB_Lower, Daily_Return, Volatility_20, Target | (792, 15) |
| **Train split** | (80% of N-60, 15) | Features: 14 cols; Target: 1 col | (633, 14) X, (633,) y |
| **Test split** | (20% of N-60, 14) | Features only (no Target) | (159, 14) |
| **LSTM sequences** | (N-60-lookback, lookback, 14) | 3D tensor | (99, 60, 14) |
| **Predictions** | (Test size,) | Scalar predictions | (159,) |

---

## Error Scenarios & Recovery

| Scenario | Error Message | Recovery |
|----------|---------------|----------|
| Invalid ticker | "No data found for XYZ" | User re-enters valid ticker |
| Network failure | "Failed to fetch data: [error]" | User retries with internet |
| Insufficient data | "Not enough training data for LSTM" | Skip LSTM, use Linear/RF |
| Missing dependency | "ModuleNotFoundError: tensorflow" | Install TensorFlow or skip LSTM |
| Bad date range | "Start date must be before end date" | User fixes date picker |

---

## Performance Characteristics

| Model | Training Time | Prediction Time | Memory | Interpretability |
|-------|---------------|-----------------|--------|------------------|
| **Linear Regression** | ~10ms | ~1ms | ~1MB | Very High |
| **Random Forest** | ~5s | ~50ms | ~10MB | High |
| **LSTM** | 1-3 min | ~100ms | ~100MB | Low (black box) |

---

## Extensibility

Future additions:
- **More indicators**: Stochastic, Williams %R, ATR
- **More models**: XGBoost, Prophet, Attention-based
- **Risk metrics**: Sharpe Ratio, Max Drawdown
- **Portfolio optimization**: Combine multiple stocks
- **Real-time predictions**: Deploy via API
- **Backtesting**: Simulate trading strategy

---

## File Structure

```
financial-predictor/
├── app.py                      # Main Streamlit entry
├── requirements.txt            # Dependencies
├── README.md                   # Quick start guide
├── ARCHITECTURE.md             # This file
│
├── data/
│   └── fetcher.py             # yfinance wrapper
│
├── features/
│   └── engineering.py         # Technical indicators
│
├── models/
│   ├── linear_model.py        # Linear Regression wrapper
│   ├── random_forest.py       # Random Forest wrapper
│   └── lstm_model.py          # LSTM implementation
│
├── evaluation/
│   └── metrics.py             # MAE, RMSE, R², MAPE
│
└── visuals/
    └── charts.py              # Plotly chart functions
```

---

## Summary

This platform is a **complete machine learning pipeline** for stock price prediction:
1. **Fetch** real data from Yahoo Finance
2. **Engineer** 10+ technical features
3. **Train** 3 different models (linear, ensemble, deep learning)
4. **Evaluate** with multiple metrics
5. **Visualize** interactively in a web dashboard

Perfect for **learning data science, time series forecasting, and deploying ML models** — all in one package.
