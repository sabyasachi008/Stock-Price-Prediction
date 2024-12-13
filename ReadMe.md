# Stock Price Prediction using Keras LSTM and Prophet

## Overview

This project aims to predict the future stock prices of a given company using two popular time series forecasting techniques: Keras LSTM (Long Short-Term Memory) and Prophet by Facebook. The project uses historical stock data to train the models and evaluate their performance.

## Files and Folders

* `webApp.py`: The main application file that uses Streamlit to create a web interface for users to input stock IDs and view predicted stock prices.
* `App.py`: Another application file that uses Streamlit to create a web interface for users to input stock IDs and view predicted stock prices (similar to `webApp.py`).


## Libraries and Frameworks

* `Keras`: A popular deep learning library used to build the LSTM model.
* `Prophet`: A open-source software for forecasting time series data developed by Facebook.
* `Streamlit`: A Python library used to create web interfaces for data science applications.
* `Pandas`: A library used for data manipulation and analysis.
* `NumPy`: A library used for numerical computations.
* `Matplotlib` and `Plotly`: Libraries used for data visualization.
* `yfinance`: A library used to download historical stock data from Yahoo Finance.

## How it Works

1. **Data Collection**: Historical stock data is collected using `yfinance` from Yahoo Finance.
2. **Data Preprocessing**: The collected data is preprocessed using `Pandas` and `NumPy` to prepare it for modeling.
3. **Modeling**: Two models are built using `Keras` LSTM and `Prophet` to forecast future stock prices.
4. **Model Evaluation**: The performance of both models is evaluated using metrics such as mean absolute error (MAE) and mean squared error (MSE).
5. **Web Interface**: A web interface is created using `Streamlit` to allow users to input stock IDs and view predicted stock prices.

## Keras LSTM Model

* The LSTM model is built using `Keras` with the following architecture:
    + Input layer: 1 neuron with a input shape of (1, 100) (100 time steps)
    + LSTM layer: 50 neurons with a return_sequences parameter set to True
    + Dense layer: 1 neuron with a linear activation function
* The model is trained using the `adam` optimizer and `mean_squared_error` loss function.
* The model is evaluated using the `mean_absolute_error` and `mean_squared_error` metrics.

## Prophet Model

* The Prophet model is built using the `Prophet` library with the following parameters:
    + Growth: linear
    + Seasonality: additive
    + Holidays: None
* The model is trained using the `fit` method and evaluated using the `make_future_dataframe` and `predict` methods.

## Web Interface

* The web interface is created using `Streamlit` with the following features:
    + Stock ID input field
    + Predicted stock price chart
    + Model evaluation metrics (MAE and MSE)

## Example Use Cases

* Predicting future stock prices for a given company
* Comparing the performance of Keras LSTM and Prophet models
* Visualizing historical stock data and predicted stock prices

## Future Work

* Improving the performance of both models using techniques such as hyperparameter tuning and feature engineering
* Adding more features to the web interface such as stock news and sentiment analysis
* Exploring other time series forecasting techniques such as ARIMA and Exponential Smoothing.

## Author
* Sabyasachi Ghosh
* sabyasachighosh008@gmail.com
* 7501539881