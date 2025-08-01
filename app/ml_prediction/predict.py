import os
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import logging

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('logs/ml_prediction.log', mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

def fetch_data(ticker):
    """
    Fetches historical price data from yfinance.
    """
    try:
        data = yf.download(ticker, start="2020-01-01", end=pd.to_datetime('today').strftime('%Y-%m-%d'))
        if data.empty:
            logger.warning(f"No data found for ticker: {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for ticker {ticker}: {e}")
        return pd.DataFrame()

def create_features(df):
    """
    Creates features for the ML model.
    """
    df['price_change'] = df['Close'].diff()
    df['5_day_ma'] = df['Close'].rolling(window=5).mean()
    df['10_day_ma'] = df['Close'].rolling(window=10).mean()
    df['target'] = (df['price_change'].shift(-1) > 0).astype(int)
    return df.dropna()

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_model(X, y):
    """
    Trains an LSTM model.
    """
    time_steps = 10
    X, y = create_dataset(X, y, time_steps)

    if X.shape[0] == 0:
        logger.warning("Not enough data to train the model after creating sequences.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Model accuracy: {accuracy}")

    return model

def main(tickers):
    """
    Main function to fetch data, train a model, and make predictions.
    """
    try:
        sentiment_df = pd.read_csv("data/processed/company_sentiment.csv")
    except FileNotFoundError:
        logger.error("company_sentiment.csv not found. Please run the sentiment analysis first.")
        sentiment_df = pd.DataFrame(columns=['company', 'normalized_sentiment'])

    predictions = {}
    for ticker in tickers:
        data = fetch_data(ticker)
        if not data.empty:
            featured_data = create_features(data)

            # Merge sentiment data
            ticker_sentiment = sentiment_df[sentiment_df['company'] == ticker]
            if not ticker_sentiment.empty:
                sentiment_score = ticker_sentiment.iloc[0]['normalized_sentiment']
                featured_data['sentiment'] = sentiment_score
            else:
                featured_data['sentiment'] = 0

            featured_data = featured_data.dropna()

            if not featured_data.empty:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(featured_data[['5_day_ma', '10_day_ma', 'sentiment']])

                X = scaled_data
                y = featured_data['target'].values

                model = train_model(X, y)

                if model:
                    time_steps = 10
                    # Prepare the latest data for prediction
                    latest_data = scaled_data[-time_steps:]
                    latest_data = latest_data.reshape(1, time_steps, 3)

                    prediction = model.predict(latest_data)[0][0]
                    predictions[ticker] = "up" if prediction > 0.5 else "down"
                    logger.info(f"Prediction for {ticker}: {'up' if prediction > 0.5 else 'down'}")

    # Save predictions
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['company', 'ml_prediction'])
    predictions_df.to_csv("data/ml_predictions.csv", index=False)
    logger.info("Successfully saved ML predictions.")

if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "GOOGL", "MSFT"]
    main(tickers)
