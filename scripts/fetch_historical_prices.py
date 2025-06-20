import yfinance as yf
import os

# Define stock tickers and period
TICKERS = ['AAPL', 'MSFT', 'GOOGL']
START_DATE = '2014-01-01'
END_DATE = '2023-12-31'

from datetime import datetime, timedelta # Added import

# Define data directory
DATA_DIR = 'data/historical_prices/'

def fetch_single_stock_data(ticker, years_of_data=5):
    """
    Fetches historical stock prices for a single ticker and saves it to a CSV file.
    """
    # Calculate start and end dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_of_data * 365.25) # Account for leap years roughly

    # Format dates as YYYY-MM-DD strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory: {DATA_DIR}")

    try:
        print(f"Downloading data for {ticker} from {start_date_str} to {end_date_str}...")
        data = yf.download(ticker, start=start_date_str, end=end_date_str)

        if data.empty:
            print(f"No data found for {ticker} for the period {start_date_str} to {end_date_str}.")
            return None

        # Convert ticker to uppercase for filename consistency
        ticker_upper = ticker.upper()
        file_path = os.path.join(DATA_DIR, f"{ticker_upper}.csv")
        data.to_csv(file_path)
        print(f"Successfully saved data for {ticker_upper} to {file_path}")
        return file_path

    except Exception as e:
        print(f"Error downloading or saving data for {ticker}: {e}")
        return None

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
