import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_current_stock_data(ticker_symbol: str) -> dict:
    """
    Fetches current stock data including price, previous close, day's range,
    and calculates 20-day and 50-day Simple Moving Averages (SMAs).

    Args:
        ticker_symbol (str): The stock ticker symbol.

    Returns:
        dict: A dictionary containing current stock data and SMAs.
              Returns an empty dict if data fetching fails.
    """
    data = {}
    try:
        ticker = yf.Ticker(ticker_symbol)

        # Fetch current market price using 'info'
        info = ticker.info
        data['current_price'] = info.get('currentPrice', info.get('regularMarketPrice'))
        if data['current_price'] is None: # Fallback for some tickers or market states
            data['current_price'] = info.get('previousClose')
            logger.warning(f"Using previousClose as currentPrice for {ticker_symbol} as currentPrice/regularMarketPrice is None.")

        data['previous_close'] = info.get('previousClose')
        data['day_high'] = info.get('dayHigh')
        data['day_low'] = info.get('dayLow')
        data['volume'] = info.get('volume')
        data['market_cap'] = info.get('marketCap')
        data['company_name'] = info.get('shortName', info.get('longName')) # For display

        # Fetch historical data for SMA calculation (approx 3 months for 50-day SMA)
        hist = ticker.history(period="3mo")
        if not hist.empty:
            # Calculate SMAs
            data['sma_20'] = hist['Close'].rolling(window=20).mean().iloc[-1] if len(hist) >= 20 else None
            data['sma_50'] = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
        else:
            data['sma_20'] = None
            data['sma_50'] = None
            logger.warning(f"Could not fetch enough historical data for SMA calculation for {ticker_symbol}.")

        # Log the fetched data for debugging
        logger.debug(f"Fetched yfinance data for {ticker_symbol}: {data}")

    except Exception as e:
        logger.error(f"Error fetching data for {ticker_symbol} using yfinance: {e}", exc_info=False) # Set exc_info to False for cleaner logs unless debugging yfinance
        # Return empty dict or specific error structure if preferred
        return {
            'current_price': None, 'previous_close': None, 'day_high': None, 'day_low': None,
            'volume': None, 'market_cap': None, 'company_name': None, 'sma_20': None, 'sma_50': None, 'error': str(e)
        }

    # Filter out None values before returning for cleaner output, but keep error if present
    # cleaned_data = {k: v for k, v in data.items() if v is not None or k == 'error'}
    # return cleaned_data
    return data


if __name__ == '__main__':
    # Test cases
    logging.basicConfig(level=logging.DEBUG)
    test_tickers = ["AAPL", "GOOGL", "MSFT", "NONEXISTENTTICKERXYZ"]
    for t in test_tickers:
        print(f"\n--- Fetching data for {t} ---")
        stock_info = get_current_stock_data(t)
        if stock_info.get('error'):
            print(f"Error for {t}: {stock_info['error']}")
        else:
            for key, value in stock_info.items():
                if isinstance(value, (int, float)) and value is not None:
                    print(f"  {key}: {value:.2f}" if pd.notna(value) and key not in ['volume', 'market_cap'] else f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
        print("--- End ---")

    # Example: How it might be used with normalization utils
    # from normalization_utils import normalize_company_name, get_ticker_for_company, _load_normalization_data
    # _load_normalization_data() # Load NASDAQ data etc.
    # company_query = "Google"
    # official_name = normalize_company_name(company_query)
    # if official_name:
    #     ticker_sym = get_ticker_for_company(official_name)
    #     if ticker_sym:
    #         print(f"\n--- Fetching data for '{company_query}' (Normalized: '{official_name}', Ticker: {ticker_sym}) ---")
    #         goog_data = get_current_stock_data(ticker_sym)
    #         print(goog_data)
    #     else:
    #         print(f"Could not find ticker for {official_name}")
    # else:
    #     print(f"Could not normalize company: {company_query}")
