import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Function to fetch stock data
def get_stock_data(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close']].reset_index()
    df['Days'] = np.arange(len(df))
    return df

# Train Linear Regression model
def train_model(df):
    X = df[['Days']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Predict future stock prices
def predict_future(model, df, days_ahead=30):
    last_day = df['Days'].max()
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).reshape(-1, 1)
    future_prices = model.predict(future_days)
    future_dates = [datetime.today() + timedelta(days=i) for i in range(1, days_ahead + 1)]
    return future_dates, future_prices

# Streamlit Web App
st.title("📈 Stock Price Prediction App")
st.write("Enter a stock ticker to predict the next 30 days of stock prices.")

# User Input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, GOOGL):", "AAPL")

if st.button("Predict"):
    with st.spinner("Fetching data and predicting..."):
        try:
            df = get_stock_data(ticker)
            df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date column is in datetime format
            model = train_model(df)
            future_dates, future_prices = predict_future(model, df)

            # Plot results
            st.subheader(f"📊 Predicted Stock Prices for {ticker}")
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(future_dates, future_prices, color='green', linestyle='dashed', marker='o', label='Future Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Stock Price')
            ax.legend()
            ax.set_title(f'Next 30 Days Stock Price Prediction for {ticker}')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Show predicted prices
            st.subheader("📅 Next 30 Days Predictions:")
            future_data = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices})
            st.dataframe(future_data.style.format({"Predicted Price": "${:.2f}"}))

        except Exception as e:
            st.error(f"Error fetching data: {e}")
