# 📈 Stock Price Prediction using Keras and Other Machine Learning Models

## 🔍 Overview

This project focuses on predicting future stock prices of publicly traded companies using four popular and effective time series forecasting models:

- **Keras LSTM (Long Short-Term Memory)**
- **Facebook Prophet**
- **Auto ARIMA**
- **Hybrid Prophet** (combination of Prophet, ARIMA)

These models are implemented using Python and are made accessible through a user-friendly web interface built with **Streamlit**. Users can enter stock ticker symbols (e.g., AAPL, TSLA) to visualize historical data, forecast future prices, and compare the performance of each model.

---

## 📁 Files and Folder Structure

- `webApp.py`: Main Streamlit web app integrating all models.
- `App.py`: An alternative Streamlit app, possibly with different styling or features.
- `aarima.py`: Streamlit app focused on Auto ARIMA and Hybrid Prophet models.

---

## 📚 Libraries and Frameworks Used

- **Keras** - Deep learning model (LSTM)
- **Prophet** - Time series forecasting library from Facebook
- **pmdarima (Auto ARIMA)** - Automatic ARIMA modeling
- **Hybrid Prophet** - Combines Prophet with other models (ARIMA, LSTM)
- **Streamlit** - Web interface framework
- **Pandas & NumPy** - Data handling and numerical computation
- **Matplotlib & Plotly** - Data visualization
- **yfinance** - Fetch historical stock data from Yahoo Finance

---

## ⚙️ How It Works

### 1. Data Collection
- Users input a stock symbol (e.g., MSFT).
- Historical stock data is downloaded using the `yfinance` API.

### 2. Data Preprocessing
- Missing values handled
- Time series formatting (e.g., date conversion, sliding windows for LSTM)
- Scaling of data for LSTM models

### 3. Forecasting Models

#### ✅ Keras LSTM
- Architecture:
  - Input shape: (1, 100)
  - LSTM layer with 50 units
  - Dense layer with 1 unit (linear activation)
- Trained with `adam` optimizer and `mean_squared_error` loss

#### ✅ Facebook Prophet
- Growth: linear
- Seasonality: additive
- Fit using `fit()` and `predict()` methods

#### ✅ Auto ARIMA
- Automatically chooses `p`, `d`, `q` parameters
- Trained using `fit()` and forecasted using `predict()`

#### ✅ Hybrid Prophet
- Combines Prophet + ARIMA (+ optional LSTM)
- Balances strengths of individual models

### 4. Evaluation
Models are evaluated using:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**

### 5. Streamlit Web Interface
- Input field for stock ticker
- Forecast plot from all models
- Metrics display (MAE, MSE)
- Optional data download/export

---

## 📈 Example Use Cases

- Forecast stock prices for investment decisions
- Educational tool for time series model comparison
- Research and development in stock prediction using hybrid models

---

## 🚀 Future Work

- Hyperparameter tuning for improved accuracy
- Feature engineering (technical indicators, volume)
- Sentiment analysis from financial news or social media
- Model stacking and ensemble learning
- Real-time prediction updates
- Enhanced dashboards and visual reports

---

## 👨‍💻 Author

**Sabyasachi Ghosh**  
📧 [sabyasachighosh008@gmail.com](mailto:sabyasachighosh008@gmail.com)  
📞 +91 75015 39881

---

> ⭐ _Feel free to fork, star, and contribute to this repository!_
