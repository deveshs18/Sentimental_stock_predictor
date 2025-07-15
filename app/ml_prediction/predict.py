import os
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

def train_model(X, y):
    """
    Trains a logistic regression model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy}")
    return model

def main(tickers):
    """
    Main function to fetch data, train a model, and make predictions.
    """
    predictions = {}
    for ticker in tickers:
        data = fetch_data(ticker)
        if not data.empty:
            featured_data = create_features(data)
            if not featured_data.empty:
                X = featured_data[['5_day_ma', '10_day_ma']]
                y = featured_data['target']
                model = train_model(X, y)
                latest_data = featured_data[['5_day_ma', '10_day_ma']].iloc[-1].values.reshape(1, -1)
                prediction = model.predict(latest_data)[0]
                predictions[ticker] = "up" if prediction == 1 else "down"
                logger.info(f"Prediction for {ticker}: {'up' if prediction == 1 else 'down'}")

    # Save predictions
    predictions_df = pd.DataFrame(list(predictions.items()), columns=['company', 'ml_prediction'])
    predictions_df.to_csv("data/ml_predictions.csv", index=False)
    logger.info("Successfully saved ML predictions.")

if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "GOOGL", "MSFT"]
    main(tickers)
