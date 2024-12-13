"""
This web app allows users to enter a stock ID and view the predicted stock price for the next day. This uses Prophet and Matplotlib
"""

import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.pyplot as plt

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

st.subheader('Raw data')
st.write(data.tail())

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

