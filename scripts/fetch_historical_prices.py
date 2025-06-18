import yfinance as yf
import os

# Define stock tickers and period
TICKERS = ['AAPL', 'MSFT', 'GOOGL']
START_DATE = '2014-01-01'
END_DATE = '2023-12-31'

# Define data directory
DATA_DIR = 'data/historical_prices/'

def fetch_historical_prices():
    """
    Fetches historical stock prices for the defined tickers and saves them to CSV files.
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for ticker in TICKERS:
        try:
            print(f"Downloading data for {ticker}...")
            data = yf.download(ticker, start=START_DATE, end=END_DATE)

            if data.empty:
                print(f"No data found for {ticker} for the specified period.")
                continue

            file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            data.to_csv(file_path)
            print(f"Successfully saved data for {ticker} to {file_path}")

        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")

if __name__ == "__main__":
    fetch_historical_prices()
