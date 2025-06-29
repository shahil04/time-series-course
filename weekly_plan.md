Great! Let's begin your **Time Series Analysis Journey** with **Week 1 Study Notes + Hands-on Exercises**.

---

# 📘 Week 1: **Time Series Fundamentals with Python**

---

## 🎯 Learning Objectives

By the end of this week, you will:

* Understand the structure of time series data
* Work with datetime indexes
* Load and visualize basic time series datasets
* Perform resampling and basic operations
* Handle missing or irregular time data

---

## 🧠 1. What is Time Series Data?

Time Series is a sequence of data points collected or recorded at **regular time intervals** (e.g., hourly, daily, monthly).

**Examples**:

* Stock prices
* Temperature logs
* Website traffic per hour
* Sales per month

---

## 🧩 2. Components of Time Series

* **Trend**: Long-term increase or decrease
* **Seasonality**: Repeating short-term cycle
* **Cyclic**: Long-term oscillations
* **Noise**: Random variation

---

## 🔧 3. Python Setup

```bash
pip install pandas numpy matplotlib seaborn
```

---

## 📥 4. Sample Dataset: AirPassengers

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url)

# Convert to datetime
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Basic info
print(df.head())
print(df.index)

# Plot
df.plot(figsize=(10, 5), title='Airline Passenger Data')
plt.ylabel("Passengers")
plt.show()
```

---

## 🔀 5. Resampling

```python
# Downsample: Yearly average
print(df.resample('Y').mean())

# Upsample: Fill missing days with forward fill
upsampled = df.resample('D').ffill()
print(upsampled.head(10))
```

---

## 🔧 6. Time-based Indexing and Operations

```python
# Filter by date range
print(df['1955-01-01':'1955-12-01'])

# Rolling mean
df['Rolling_12'] = df['Passengers'].rolling(12).mean()
df[['Passengers', 'Rolling_12']].plot(title='Rolling Mean')
plt.show()
```

---

## 🧪 Week 1 Exercises

### 🔸 Q1: Load a new dataset

Use the `"Daily Minimum Temperatures"` dataset from:

```plaintext
https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv
```

Perform the following:

* Convert date to `datetime`
* Set date as index
* Plot temperature
* Resample to monthly average

### 🔸 Q2: Create missing data

* Drop every 20th row in the dataset
* Fill missing dates using `.asfreq()` + `interpolate()`

---

## ✅ Week 1 Deliverables:

* Notebook or script with:

  * ✅ Dataset loading
  * ✅ Datetime handling
  * ✅ Visualization
  * ✅ Resampling & rolling stats
  * ✅ Missing data handling

---

Would you like me to:

* Prepare this as a **Jupyter Notebook file**?
* Add **Quiz questions** for revision?
* Move to **Week 2: ETSA (Trend, Seasonality, ACF/PACF)** after you're done?

Let me know how you'd like to proceed!
==================

Awesome! Let’s dive into **Week 2: Exploratory Time Series Analysis (ETSA)** – the most important phase where you visually and statistically understand your time series before modeling.

---

# 📘 **Week 2: Exploratory Time Series Analysis (ETSA)**

---

## 🎯 Learning Objectives

By the end of this week, you’ll be able to:

* Break down a time series into **trend**, **seasonality**, and **residual**
* Test for **stationarity** using ADF and KPSS
* Understand and visualize **autocorrelation (ACF)** and **partial autocorrelation (PACF)** plots
* Perform **seasonal decomposition**

---

## 🔧 Required Libraries

```bash
pip install statsmodels
pip install pmdarima
```

---

## 📊 1. Visual Inspection: Trend & Seasonality

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Plot raw time series
df.plot(figsize=(10, 4), title="Air Passengers Over Time")
plt.show()
```

---

## 🔍 2. Decomposition (Trend + Seasonality + Residuals)

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(df['Passengers'], model='multiplicative')
decomp.plot()
plt.show()
```

---

## 🧪 3. Stationarity Check: ADF Test

> **Stationarity** means mean, variance, and autocorrelation are constant over time — a requirement for many time series models.

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Passengers'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

if result[1] < 0.05:
    print("✅ Series is Stationary")
else:
    print("❌ Series is NOT Stationary")
```

---

## 🔁 4. Differencing to Make Series Stationary

```python
df['diff'] = df['Passengers'].diff()
df.dropna().plot(y='diff', title="First Order Differencing")
plt.show()
```

---

## 🔄 5. ACF & PACF Plots

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df['Passengers'], lags=30)
plt.show()

plot_pacf(df['Passengers'], lags=30)
plt.show()
```

These help determine AR/MA terms for ARIMA modeling later.

---

## 🧪 Week 2 Exercises

### 🔸 Q1: Use a different dataset

Load the **daily-min-temperatures** dataset again and:

* Decompose the series
* Check ADF test results
* Plot ACF and PACF
* Difference the series and check ADF again

### 🔸 Q2: Try additive vs multiplicative decomposition

> Use both models in `seasonal_decompose()` and compare visually

---

## ✅ Week 2 Deliverables:

* Plot raw time series + decomposed components
* Perform stationarity tests (ADF)
* Visualize ACF and PACF
* Comment on whether your series is stationary or not

---

## Coming Up: **Week 3 - Time Series Forecasting with ARIMA/SARIMA**

You'll learn:

* When and how to use ARIMA/SARIMA
* How to automatically determine (p,d,q) parameters
* Train/test split for time series
* Forecast future values and validate results

---

Would you like me to:

* Prepare a **Jupyter Notebook template** for Week 2?
* Send **quiz/revision MCQs**?
* Move to **Week 3: ARIMA modeling and project**?

Let me know your next step!

==========

Perfect! Let's begin **Week 3: Time Series Forecasting with ARIMA/SARIMA** – your first classical forecasting model.

---

# 📘 **Week 3: Time Series Forecasting with ARIMA/SARIMA**

---

## 🎯 Learning Objectives

By the end of this week, you'll:

* Understand AR, MA, ARIMA, SARIMA models
* Determine (p, d, q) parameters using ACF/PACF
* Perform model training, forecasting, and evaluation
* Visualize and interpret forecasts
* Build your first real ARIMA forecasting project

---

## 🔍 1. What is ARIMA?

ARIMA stands for:

* **AR (AutoRegression)** – relationship between current and past values
* **I (Integrated)** – differencing to make the series stationary
* **MA (Moving Average)** – relationship between current and past forecast errors

ARIMA is written as **ARIMA(p, d, q)**:

* `p`: order of AR term
* `d`: order of differencing
* `q`: order of MA term

> For **seasonal** series, use **SARIMA**: `SARIMA(p,d,q)(P,D,Q,s)`
> where `s` = seasonality period (e.g., 12 for monthly data)

---

## 🔧 2. Setup

```bash
pip install statsmodels pmdarima
```

---

## 📥 3. Load & Prepare Dataset

```python
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
```

---

## 📊 4. ACF & PACF to Select (p, q)

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

plot_acf(df['Passengers'])
plot_pacf(df['Passengers'])
plt.show()
```

> You’ll estimate `p` from PACF, `q` from ACF

---

## 🤖 5. Auto ARIMA (Automatic Model Selection)

```python
from pmdarima import auto_arima

stepwise_fit = auto_arima(df['Passengers'], seasonal=True, m=12, trace=True)
print(stepwise_fit.summary())
```

---

## 📈 6. Fit ARIMA/SARIMA Model

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

train = df.iloc[:-12]
test = df.iloc[-12:]

model = SARIMAX(train['Passengers'], order=(2,1,2), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)
```

---

## 🔮 7. Forecasting

```python
pred = model_fit.get_forecast(steps=12)
conf_int = pred.conf_int()

# Plot forecast
ax = df.plot(label='Observed', figsize=(10, 5))
pred.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(conf_int.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.title("Forecast vs Actual")
plt.show()
```

---

## 📐 8. Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_true = test['Passengers']
y_pred = pred.predicted_mean

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
```

---

## 🧪 Week 3 Exercises

### 🔸 Q1: Perform ARIMA Forecast

Use the `"daily-min-temperatures.csv"` dataset and:

* Check ADF stationarity
* Use ACF/PACF or `auto_arima` to find optimal model
* Train/test split → forecast → plot
* Evaluate RMSE and MAE

### 🔸 Q2: Compare ARIMA vs SARIMA

Try both on monthly dataset and compare forecast accuracy

---

## ✅ Week 3 Deliverables:

* Train-test split
* Manual ARIMA and Auto ARIMA model
* Forecast and evaluation
* A short conclusion: which model performed better?

---

## 📦 Coming Up Next: **Week 4 – Prophet, Holt-Winters, and VAR (Modern Forecasting)**

Would you like me to:

* Prepare a **Jupyter notebook for Week 3**?
* Share a **dataset pack** for practice?
* Continue with **Week 4: Modern Forecasting Methods**?

Let me know what’s next!

===============
Excellent! Let's move to **Week 4: Modern Time Series Forecasting Techniques** — focusing on **Facebook Prophet**, **Holt-Winters**, and **VAR** for multivariate forecasting.

---

# 📘 **Week 4: Modern Time Series Forecasting Techniques**

---

## 🎯 Learning Objectives

By the end of this week, you will:

* Use **Facebook Prophet** for forecasting with trends, holidays, and seasonality
* Apply **Holt-Winters Exponential Smoothing**
* Learn **VAR (Vector AutoRegression)** for multivariate time series
* Compare these models with ARIMA/SARIMA

---

## 🛠 1. Install Required Packages

```bash
pip install prophet
pip install statsmodels
```

---

## 📊 2. Facebook Prophet

### ✅ Prophet expects a DataFrame with:

```text
- ds: date column
- y: target value column
```

### 🔧 Data Preparation

```python
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
df.columns = ['ds', 'y']  # Rename columns for Prophet
df['ds'] = pd.to_datetime(df['ds'])

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Plot
model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()
```

### ✅ Add custom holidays/seasonality if needed:

```python
model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
```

---

## 📈 3. Holt-Winters (Triple Exponential Smoothing)

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df_hw = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
df_hw['Month'] = pd.to_datetime(df_hw['Month'])
df_hw.set_index('Month', inplace=True)

model = ExponentialSmoothing(df_hw['Passengers'], seasonal='multiplicative', trend='add', seasonal_periods=12)
fit = model.fit()
forecast = fit.forecast(12)

df_hw['Passengers'].plot(label='Observed', figsize=(10, 5))
forecast.plot(label='Forecast', style='--')
plt.title("Holt-Winters Forecast")
plt.legend()
plt.show()
```

---

## 🔁 4. Vector AutoRegression (VAR) – for Multivariate Time Series

### Example: GDP and Inflation over time

```python
from statsmodels.tsa.api import VAR

# Simulate example data
data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv')
data = data[['date', 'gdp', 'consumption']]
data.columns = ['ds', 'gdp', 'consumption']
data['ds'] = pd.to_datetime(data['ds'])
data.set_index('ds', inplace=True)

model = VAR(data)
results = model.fit(maxlags=12, ic='aic')
forecast = results.forecast(data.values[-results.k_ar:], steps=5)

# Convert to DataFrame
forecast_df = pd.DataFrame(forecast, index=pd.date_range(start=data.index[-1], periods=5, freq='Q'),
                           columns=['gdp', 'consumption'])

print(forecast_df)
```

---

## 📏 5. Evaluation (again!)

```python
from sklearn.metrics import mean_squared_error

# RMSE for Prophet
rmse = mean_squared_error(df['y'][-12:], forecast['yhat'][-12:])**0.5
print(f"Prophet RMSE: {rmse:.2f}")
```

---

## 🧪 Week 4 Exercises

### 🔸 Q1: Prophet Forecast

* Use `"daily-min-temperatures.csv"`
* Forecast 30 days ahead
* Plot forecast and components

### 🔸 Q2: Holt-Winters vs ARIMA

* Apply both on `"airline-passengers"`
* Compare RMSE or MAPE

### 🔸 Q3: VAR Forecast

* Use GDP and Inflation from Raotbl6.csv
* Plot actual vs forecasted values

---

## ✅ Week 4 Deliverables:

* Forecast using Prophet, Holt-Winters, and VAR
* Visualizations and error metrics
* Notes on when each model is preferable

---

## Coming Up: **Week 5 – Deep Learning Models (LSTM, GRU, Transformer)**

You’ll build deep learning models using Keras/PyTorch and use them to forecast stock prices, temperature, etc.

---

Would you like me to:

* Prepare a **Jupyter notebook for Week 4**?
* Share **real datasets for Prophet and VAR**?
* Move to **Week 5: Deep Learning with LSTM and GRU**?

Let me know how you want to proceed 🚀

================================

Great! You're now entering **Week 5: Deep Learning for Time Series Forecasting** — where you'll build powerful models using **LSTM**, **GRU**, and optionally **Transformer** architectures for sequence prediction.

---

# 📘 **Week 5: Deep Learning Models for Time Series (LSTM, GRU, Transformer)**

---

## 🎯 Learning Objectives

By the end of this week, you’ll be able to:

* Prepare time series data for deep learning models
* Build, train, and evaluate **LSTM** and **GRU** models using `Keras`
* Optionally experiment with **Transformer-based models**
* Forecast future values and compare with ARIMA/Prophet

---

## ⚙️ 1. Prerequisites & Installation

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

## 📈 2. Load & Normalize Time Series Dataset

Let’s use `airline-passengers.csv` again.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Normalize values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# Convert to supervised learning format
def create_sequences(data, n_steps=12):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, 12)
print(X.shape, y.shape)
```

---

## 🧠 3. Build LSTM Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)
```

---

## 🔮 4. Forecast & Inverse Transform

```python
y_pred = model.predict(X)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_true_rescaled = scaler.inverse_transform(y)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_true_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='LSTM Forecast')
plt.legend()
plt.title("LSTM Forecast vs Actual")
plt.show()
```

---

## 🔁 5. GRU Model (Alternative)

```python
from tensorflow.keras.layers import GRU

gru_model = Sequential([
    GRU(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])

gru_model.compile(optimizer='adam', loss='mse')
gru_model.fit(X, y, epochs=200, verbose=0)
```

---

## 🚀 6. Optional: Transformer Time Series Forecasting (Advanced)

Let me know if you want to include:

* Temporal Fusion Transformers (TFT)
* Attention-based encoder-decoder transformer

---

## 📏 7. Evaluation

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
```

---

## 🧪 Week 5 Exercises

### 🔸 Q1: LSTM Forecasting

* Use `"daily-min-temperatures.csv"`
* Create rolling windows of 30 days → predict next day
* Evaluate and visualize

### 🔸 Q2: Compare LSTM vs GRU

* On the same dataset
* Plot and compare RMSE/MAE

### 🔸 Q3 (Optional): Multi-step forecasting

* Predict next 7–30 days using recursive method

---

## ✅ Week 5 Deliverables

* LSTM and GRU models
* Train/validation curves
* Plots of actual vs predicted
* Error metrics (RMSE, MAE)
* Summary: Which deep model performed better?

---

## Next: **Week 6 — Model Comparison & Final Real-World Project**

* Compare **ARIMA vs Prophet vs LSTM vs GRU**
* Create a leaderboard (RMSE/MAE)
* Build a **complete forecasting dashboard using Streamlit or Flask**
* Deploy to cloud (GCP, AWS, or Hugging Face Spaces)

---

Would you like me to:

* Prepare a **Jupyter Notebook template for Week 5**?
* Share **bonus datasets** for deep models?
* Continue to **Week 6: Model Comparison + Final Project Setup**?

Let me know your next step!

====================================
Awesome! You’ve mastered the key models — now it's time to **compare them**, identify the **best-performing one**, and build your **final project**.

---

# 📘 **Week 6: Model Comparison + Final Real-World Forecasting Project**

---

## 🎯 Learning Objectives

By the end of this week, you’ll:

* Evaluate and compare **ARIMA**, **Prophet**, **Holt-Winters**, **LSTM**, and **GRU**
* Choose the best model based on real data
* Build a complete **end-to-end forecasting app**
* Learn how to **deploy it using Streamlit or Flask**
* Optionally, dockerize or deploy to cloud (Heroku/GCP/Hugging Face Spaces)

---

## 📊 1. Model Comparison Framework

| Model        | RMSE | MAE | Training Time | Interpretability | Comments |
| ------------ | ---- | --- | ------------- | ---------------- | -------- |
| ARIMA        |      |     | ⏱️ Fast       | ✅ High           |          |
| Prophet      |      |     | ⏱️ Moderate   | ✅ High           |          |
| Holt-Winters |      |     | ⏱️ Fast       | ✅ Medium         |          |
| LSTM         |      |     | 🐢 Slower     | ❌ Low            |          |
| GRU          |      |     | 🐢 Slower     | ❌ Low            |          |

---

## 🧪 2. Evaluate All Models on the Same Dataset

Use a dataset like:

* `"airline-passengers.csv"` (monthly, univariate)
* `"daily-min-temperatures.csv"` (daily, univariate)
* `"Retail Sales.csv"` (multivariate, optional)

Run all models:

* Forecast next 12 points
* Compute **RMSE** and **MAE**
* Plot all forecasts on the same graph

```python
# Combine plots for comparison
plt.plot(y_true, label='Actual')
plt.plot(arima_forecast, label='ARIMA')
plt.plot(prophet_forecast, label='Prophet')
plt.plot(lstm_forecast, label='LSTM')
plt.legend()
plt.title("Model Forecast Comparison")
plt.show()
```

---

## 🛠 3. Build Your Final Project

### 🔧 Tech Stack:

* **Backend**: Python (Streamlit or Flask)
* **Model**: Your best-performing model (e.g., Prophet or LSTM)
* **Frontend**: Matplotlib/Plotly for graphs
* **Extras**:

  * Docker (optional)
  * GitHub Actions CI/CD (optional)
  * Cloud: Hugging Face Spaces, GCP, or Heroku

---

### 🎯 Final Project: Sales or Temperature Forecasting App

#### Features:

* 📁 Upload your own time series CSV
* 📊 Choose forecast period (e.g., 7, 30, 90 days)
* 🔍 Visualize raw and forecast data
* 💾 Download forecast as CSV
* 🚀 Deploy online

---

## 📄 Streamlit App Template (Fastest Path)

```python
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("📈 Time Series Forecasting App")

uploaded_file = st.file_uploader("Upload time series CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    st.write("### Raw Data", df.tail())

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    st.write("### Forecasted Data", forecast[['ds', 'yhat']].tail())

    fig = model.plot(forecast)
    st.pyplot(fig)
```

---

## 🧪 Week 6 Exercises

### 🔸 Q1: Build Comparison Table

* RMSE, MAE, Forecast Plot
* Which model works best for each case?

### 🔸 Q2: Create Web App

* Streamlit version or Flask + HTML dashboard

### 🔸 Q3 (Optional): Docker + GitHub CI/CD

---

## ✅ Week 6 Deliverables

* Evaluation notebook with model comparisons
* Final web app: code + working demo
* README with steps to run and interpret

---

## 🎁 Bonus (Optional Week 7):

* Deep dive into **Time Series Feature Engineering**
* Using **`darts`**, **`sktime`**, **`gluonts`**, and **forecasting ensembling**
* Real-world case: forecasting electricity usage, demand, web traffic

---

Would you like me to:

* Prepare a **complete Streamlit app repo template**?
* Help you **dockerize + deploy** the app?
* Generate a **final README + documentation**?

Let me know how you'd like to wrap up your Time Series journey 🚀📈

======================================================

Awesome! You've completed the core of your time series learning path. Now let’s go **beyond traditional forecasting** and tackle **Week 7: Bonus Advanced Topics** — covering feature engineering, model ensembling, new libraries, and real-world industrial use cases.

---

# 📘 **Week 7 (Bonus): Advanced Topics in Time Series Forecasting**

---

## 🎯 Learning Objectives

By the end of this week, you will:

* Engineer advanced features for boosting model performance
* Use high-level forecasting libraries like **Darts**, **GluonTS**, and **sktime**
* Perform **model ensembling** for better accuracy
* Solve a full **real-world case study** (energy demand / web traffic / sales)

---

## 🧩 1. Feature Engineering for Time Series

### ✅ Key Engineered Features:

| Feature                 | Purpose                     |
| ----------------------- | --------------------------- |
| Lag features (t-1, t-2) | Add memory to models        |
| Rolling mean/std        | Detect local trends         |
| Time components         | Hour, day, month, weekday   |
| Fourier terms           | Capture complex seasonality |
| Holiday flags           | Contextual seasonality      |

```python
df['month'] = df.index.month
df['lag_1'] = df['y'].shift(1)
df['rolling_7'] = df['y'].rolling(window=7).mean()
df.dropna(inplace=True)
```

---

## 🧠 2. Use High-Level Libraries

### 📦 A. `Darts` (Unified interface for all models)

```bash
pip install u8darts
```

```python
from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing, Prophet, RNNModel

series = AirPassengersDataset().load()
model = Prophet()
model.fit(series)
forecast = model.predict(12)
series.plot()
forecast.plot()
```

---

### 📦 B. `GluonTS` (Deep Learning + Probabilistic Forecasting)

```bash
pip install gluonts
```

Ideal for:

* Probabilistic forecasts
* Retail/energy domains
* DeepAR, Transformer-based models

---

### 📦 C. `sktime` (scikit-learn style API)

```bash
pip install sktime
```

Great for:

* Pipelines
* GridSearchCV
* Combining regressors and classifiers with time series

---

## 🔗 3. Model Ensembling

* Combine ARIMA + Prophet + LSTM predictions
* Average their outputs or use weighted sum

```python
final_forecast = (arima_pred + prophet_pred + lstm_pred) / 3
```

> Ensembling improves accuracy and robustness.

---

## 🧪 4. Final Real-World Project (Pick One)

### 🔥 Option A: **Energy Demand Forecasting**

* Dataset: [UK Domestic Energy Consumption](https://www.kaggle.com/datasets/jeanmidev/smart-energy-consumption)
* Objective: Predict daily usage per household

### 🛒 Option B: **Retail Sales Forecasting**

* Dataset: [M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
* Objective: Predict daily sales across multiple stores and SKUs

### 🌐 Option C: **Web Traffic Forecasting**

* Dataset: [Wikipedia Web Traffic Time Series](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting)
* Objective: Predict future visits per page

---

## 🧪 Week 7 Exercises

### 🔸 Q1: Feature Engineering Notebook

* Create lag, rolling, date/time, Fourier features
* Use them in `XGBoost` or `RandomForestRegressor`

### 🔸 Q2: Try 1 model from `darts` and 1 from `gluonts`

### 🔸 Q3: Ensemble 2–3 models and evaluate performance

### 🔸 Q4: Complete one **real-world project** (choose above)

---

## ✅ Week 7 Deliverables

* Notebook: Feature Engineering + Model Training + Evaluation
* Final Project: Full code, metrics, visualizations
* Model Comparison Table
* Optional: Deployment via Streamlit or Docker

---

## 🚀 After This Week:

You’ll be ready for:

* **AI & ML roles** requiring time series
* Building commercial-grade forecasting pipelines
* Participating in competitions like **Kaggle**, **Zindi**, or **TimeGPT/TimeBench**

---

Would you like me to:

* Share a **project template** (Streamlit + Darts or XGBoost)?
* Provide **end-to-end code** for one of the final projects?
* Help you **document and publish** your time series portfolio on GitHub?

Let me know your next move — and congratulations on reaching the advanced level of Time Series Analysis! 🧠📈💼

==========================================================

Perfect! Let’s now go **deep into real-world time series forecasting using Deep Learning**. This is an advanced **Week 7 (Deep Learning Focus)** for real-world use cases like sales, energy, or stock price forecasting using:

* ✅ Feature Engineering
* ✅ LSTM, GRU, Bidirectional LSTM
* ✅ Sequence-to-sequence multi-step forecasting
* ✅ Model ensembling
* ✅ Deployment (Streamlit, Docker)

---

# 📘 **Week 7 (Advanced DL Focus): Real-World Time Series Forecasting Using Deep Learning**

---

## 🎯 Learning Objectives

* Prepare time series data with lag & rolling features
* Build & evaluate LSTM, GRU, Bidirectional LSTM models
* Implement multi-step and multi-feature forecasting
* Handle multivariate time series
* Build a complete app for deep-learning based forecasting

---

## 📦 Real Dataset Options (Pick One)

| Dataset               | Use Case                   | Link                                                                                             |
| --------------------- | -------------------------- | ------------------------------------------------------------------------------------------------ |
| 🏠 Energy Consumption | Household-level energy use | [Link](https://www.kaggle.com/datasets/jeanmidev/smart-energy-consumption)                       |
| 📊 Stock Prices       | Predict stock trends       | [Link](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) |
| 🛒 Retail Sales       | Forecast future demand     | [Link](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)                         |

---

## 📈 Step 1: Data Preparation & Feature Engineering

```python
# Lag features, rolling features, time features
df['lag_1'] = df['y'].shift(1)
df['rolling_7'] = df['y'].rolling(7).mean()
df['month'] = df.index.month
df['dayofweek'] = df.index.dayofweek
df.dropna(inplace=True)
```

### Normalize Data

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['y', 'lag_1', 'rolling_7']])
```

---

## 🧠 Step 2: Create Supervised Sequences for Deep Models

```python
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i, 0])  # predict original 'y'
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, n_steps=30)
```

---

## 🧪 Step 3: Build LSTM/GRU/Bidirectional Models

### ✅ LSTM Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=32)
```

---

### ✅ GRU Model (Swap LSTM with GRU)

```python
from tensorflow.keras.layers import GRU

model = Sequential([
    GRU(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])
```

---

### ✅ Bidirectional LSTM (Optional)

```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Bidirectional(LSTM(64), input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])
```

---

## 📏 Step 4: Evaluate Forecast

```python
import numpy as np
from sklearn.metrics import mean_squared_error

y_pred = model.predict(X)
y_inv = scaler.inverse_transform(np.concatenate((y.reshape(-1, 1), np.zeros((len(y), 2))), axis=1))[:, 0]
y_pred_inv = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((len(y_pred), 2))), axis=1))[:, 0]

rmse = np.sqrt(mean_squared_error(y_inv, y_pred_inv))
print("RMSE:", rmse)
```

---

## 🧪 Step 5: Multi-Step Forecasting (Sequence-to-Sequence)

```python
# Predict next 7 or 30 days recursively
last_seq = X[-1]
future = []

for _ in range(30):
    pred = model.predict(last_seq.reshape(1, *last_seq.shape))
    new_step = np.hstack((pred, last_seq[-1, 1:]))  # Use same lag/rolling features
    last_seq = np.vstack((last_seq[1:], new_step))
    future.append(pred[0][0])
```

---

## 🖥 Step 6: Deploy With Streamlit

```python
import streamlit as st

st.title("Deep Learning Forecasting App")
uploaded_file = st.file_uploader("Upload CSV with 'ds', 'y'", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Preprocess and load model
    forecast = model.predict(X[-30:])  # Simplified
    st.line_chart(forecast)
```

---

## 🧪 Week 7 DL-Focused Exercises

### 🔸 Q1: Build an LSTM model on **Energy or Sales Dataset**

* Lag features, rolling window
* Predict next 7–30 days

### 🔸 Q2: Compare GRU vs LSTM vs Bidirectional

* Same dataset
* Compare RMSE/MAE and training time

### 🔸 Q3: Create a Streamlit App

* Upload CSV → Forecast 7 days → Show chart

---

## ✅ Deliverables

* Final Notebook: Data prep, models, evaluation
* Web app (optional Streamlit or Flask)
* RMSE comparison table (LSTM, GRU, BiLSTM)
* Deployment (local or Hugging Face Spaces / GCP)

---

## Next (Final Week - Optional):

* **Ensemble LSTM + Prophet + XGBoost**
* Add **Alerting / Anomaly Detection**
* **Dockerize** full app
* **CI/CD pipeline** with GitHub Actions

---

Would you like me to:

* Provide a **ready-made notebook template** for LSTM-based forecasting?
* Help **package the final Streamlit app** with Docker + GitHub Actions?
* Suggest a real-world **client scenario** (e.g., for job portfolio)?

Let’s make your deep learning time series project production-ready 🚀📈
==============================
