"""
This web app allows users to enter a stock ID and view the predicted stock price for the next day. This uses Prophet and Matplotlib
"""
import time
import streamlit as st
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.pyplot as plt
from pmdarima import auto_arima


START = "2004-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Show raw data
st.subheader('Raw data')
st.write(data.tail())

# 📌 Insert real-time stock price update code here
st.subheader("📈 Live Stock Price Updates")
live_price_placeholder = st.empty()

try:
    stock_info = yf.Ticker(selected_stock).info
    current_price = stock_info.get('currentPrice', 'N/A')

    live_price_placeholder.write(f"🔹 **Current Price of {selected_stock}:** ${current_price}")

    time.sleep(20)  # Refresh every 20 seconds

except Exception as e:
    st.error(f"Error fetching live stock price: {e}")

# Predict forecast with Prophet
df_train = pd.DataFrame({
    'ds': pd.to_datetime(data['Date']).dt.tz_localize(None),
    'y': data['Close'].values.squeeze()
})


# Plot raw data
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.plot(data['Date'], data['Open'], label='Open Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price')
plt.legend()
st.pyplot(plt)

# Predict forecast with Prophet
df_train = pd.DataFrame({
    'ds': pd.to_datetime(data['Date']).dt.tz_localize(None),
    'y': data['Close'].values.squeeze()
})

m = Prophet()
m.add_seasonality(name='monthly', period=30.5, fourier_order=10)
m.add_seasonality(name='quarterly', period=91.25, fourier_order=10)
m.add_seasonality(name='yearly', period=365.25, fourier_order=10)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)

# Plot change points
st.write("Change points")
fig3 = m.plot(forecast)
st.pyplot(fig3)

# Plot seasonality
st.write("Seasonality")
fig4 = m.plot_components(forecast)
st.pyplot(fig4)

# Plot trend
st.write("Trend")
fig5 = m.plot_components(forecast)
st.pyplot(fig5)

# Plot residuals
st.write("Residuals")
fig6 = plt.figure(figsize=(10, 6))
plt.plot(forecast['ds'], forecast['yhat'] - forecast['yhat_lower'])
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Residuals')
st.pyplot(fig6)
# 📌 AUTO ARIMA MODEL FORECASTING

st.subheader('📊 Auto ARIMA Forecast')

# Convert data to time series format
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Fill missing dates and remove NaN values
data = data.asfreq('D').fillna(method='ffill')  # Fill missing dates

# Auto ARIMA to find the best parameters
st.text("Finding the best ARIMA model...")
try:
    arima_model = auto_arima(data['Close'], 
                             seasonal=False, 
                             trace=True, 
                             stepwise=True, 
                             suppress_warnings=True, 
                             error_action="ignore")

    st.text(f"Best ARIMA Model: {arima_model}")

    # Forecast using the best ARIMA model
    future_dates = pd.date_range(start=data.index[-1], periods=period + 1, freq='D')[1:]
    arima_forecast = arima_model.predict(n_periods=period)
    arima_df = pd.DataFrame({'Date': future_dates, 'ARIMA_Predicted_Close': arima_forecast})
    arima_df['Date'] = pd.to_datetime(arima_df['Date'])  # Ensure datetime format

    # Show ARIMA forecast
    st.write(arima_df.tail())

    # 📌 COMPARING PROPHET VS AUTO ARIMA
    st.subheader('📊 Prophet vs Auto ARIMA Forecast')

    # Plot Prophet vs Auto ARIMA
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label="Actual Close Price", color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label="Prophet Forecast", color='green')
    ax.plot(arima_df['Date'], arima_df['ARIMA_Predicted_Close'], label="Auto ARIMA Forecast", color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title(f'{selected_stock} Price Prediction: Prophet vs Auto ARIMA')
    ax.legend()
    st.pyplot(fig)

    st.write("✅ Prophet captures seasonality, Auto ARIMA finds the best model for short-term trends.")

except Exception as e:
    st.error(f"⚠️ Auto ARIMA failed: {e}")
 
# 📌 Hybrid Model: Combining Prophet & Auto ARIMA
# -----------------------------

st.subheader('🔀 Hybrid Model (Prophet + Auto ARIMA)')

# Merge both forecasts
hybrid_df = forecast[['ds', 'yhat']].merge(arima_df, left_on='ds', right_on='Date', how='inner')

# Weighted Hybrid Prediction
alpha = 0.6  # Weight for Prophet
beta = 0.4   # Weight for Auto ARIMA

hybrid_df['Hybrid_Forecast'] = (alpha * hybrid_df['yhat']) + (beta * hybrid_df['ARIMA_Predicted_Close'])

# Show Hybrid forecast
st.write(hybrid_df[['ds', 'Hybrid_Forecast']].tail())

# Plot Hybrid Forecast
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data.index, data['Close'], label="Actual Close Price", color='blue')
ax.plot(forecast['ds'], forecast['yhat'], label="Prophet Forecast", color='green')
ax.plot(arima_df['Date'], arima_df['ARIMA_Predicted_Close'], label="Auto ARIMA Forecast", color='red')
ax.plot(hybrid_df['ds'], hybrid_df['Hybrid_Forecast'], label="Hybrid Forecast", color='purple', linestyle='dashed')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.set_title(f'{selected_stock} Price Prediction: Hybrid Model')
ax.legend()
st.pyplot(fig)

st.write("✅ The Hybrid Model blends Prophet's seasonality with Auto ARIMA's short-term accuracy.")