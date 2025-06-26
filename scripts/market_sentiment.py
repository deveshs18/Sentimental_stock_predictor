# scripts/market_sentiment.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging

logger = logging.getLogger(__name__)

class MarketSentiment:
    def __init__(self):
        self.vix_symbol = '^VIX'  # Volatility Index
        self.sp500_symbol = '^GSPC'  # S&P 500
        self.fred_api_key = 'YOUR_FRED_API_KEY'  # Register at FRED for free API key
        
    def get_market_indicators(self, days_back=30):
        """Fetch market indicators for sentiment analysis."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.debug(f"MarketSentiment: Fetching market data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            vix_data = yf.download(self.vix_symbol, start=start_date, end=end_date, progress=False)
            if vix_data.empty or 'Close' not in vix_data.columns:
                logger.error(f"MarketSentiment: VIX data is empty or 'Close' column missing. Columns: {vix_data.columns if not vix_data.empty else 'N/A (empty df)'}")
                return None
            vix = vix_data['Close'].dropna()
            
            sp500_data = yf.download(self.sp500_symbol, start=start_date, end=end_date, progress=False)
            if sp500_data.empty or 'Close' not in sp500_data.columns:
                logger.error(f"MarketSentiment: S&P 500 data is empty or 'Close' column missing. Columns: {sp500_data.columns if not sp500_data.empty else 'N/A (empty df)'}")
                return None
            sp500 = sp500_data['Close'].dropna()

            if vix.empty:
                logger.error("MarketSentiment: VIX data is empty after selecting 'Close' column and dropping NaNs.")
                return None
            if sp500.empty:
                logger.error("MarketSentiment: S&P 500 data is empty after selecting 'Close' column and dropping NaNs.")
                return None
            
            if len(vix) < 1:
                logger.error(f"MarketSentiment: Not enough data points for VIX after download and dropna. Points: {len(vix)}")
                return None
            if len(sp500) < 1:
                logger.error(f"MarketSentiment: Not enough data points for S&P500 after download and dropna. Points: {len(sp500)}")

                return None

            vix_current_scalar = vix.iloc[-1]
            vix_mean_val = vix.mean()

            vix_std_val = vix.std(ddof=0)


            sp500_current_scalar = sp500.iloc[-1]
            sp500_initial_scalar = sp500.iloc[0]
            

            if pd.isna(vix_current_scalar) or pd.isna(vix_mean_val) or pd.isna(vix_std_val) or \
               pd.isna(sp500_current_scalar) or pd.isna(sp500_initial_scalar) or \
               (not pd.isna(sp500_initial_scalar) and sp500_initial_scalar == 0):
                logger.error(f"MarketSentiment: Critical VIX/SP500 scalar values are NaN or S&P initial is zero. "

                             f"VIX current: {vix_current_scalar}, VIX mean: {vix_mean_val}, VIX std: {vix_std_val}, "
                             f"SP500 current: {sp500_current_scalar}, SP500 initial: {sp500_initial_scalar}")
                return None

            sp500_returns = sp500.pct_change().dropna()
            sp500_volatility_annualized = sp500_returns.std(ddof=0) * np.sqrt(252) if not sp500_returns.empty else np.nan

            market_data_dict = {
                'vix_current': vix_current_scalar,
                'vix_30d_avg': vix_mean_val,
                'vix_30d_std': vix_std_val,
                'sp500_30d_return': (sp500_current_scalar / sp500_initial_scalar - 1) * 100,
                'sp500_volatility': sp500_volatility_annualized,

                'market_sentiment': 0.0 # Default, to be overwritten if calculation succeeds
            }

            if len(vix) >= 2 and len(sp500) >= 20:
                 market_data_dict['market_sentiment'] = self._calculate_market_sentiment(vix, sp500)
            else:
                logger.warning(f"MarketSentiment: Not enough data for full market sentiment calculation (VIX len: {len(vix)}, S&P500 len: {len(sp500)}). Setting market_sentiment to neutral (0.0).")
                # market_data_dict['market_sentiment'] is already 0.0

            logger.info(f"MarketSentiment: Successfully fetched and processed market indicators: {market_data_dict}")
            return market_data_dict
            
        except Exception as e:
            logger.error(f"MarketSentiment: Error processing market data in get_market_indicators: {e}", exc_info=True)


            logger.info(f"Successfully fetched and processed market indicators: {market_data_dict}")
            return market_data_dict
            
        except Exception as e:

            logger.error(f"Error processing market data in get_market_indicators: {e}", exc_info=True)

            return None
    
    def _get_fred_data(self, series_id):
        """Helper to fetch economic data from FRED."""
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'limit': 1,
                'sort_order': 'desc'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()['observations'][0]['value']
        except Exception as e:

            logger.warning(f"MarketSentiment: Could not fetch FRED data for series {series_id}: {e}")

            return None
    
    def _calculate_market_sentiment(self, vix, sp500):
        """Calculate a composite market sentiment score (-1 to 1)."""

        if len(vix) < 2:
            logger.warning(f"MarketSentiment._calculate_market_sentiment: VIX series has {len(vix)} points, less than 2 required for std dev. Returning 0.0.")
            return 0.0
        if len(sp500) < 20:
            logger.warning(f"MarketSentiment._calculate_market_sentiment: S&P500 series has {len(sp500)} points, less than 20 required for medium term trend. Returning 0.0.")

            return 0.0

        try:
            vix_mean_val = vix.mean()
            vix_std_val = vix.std(ddof=0)

            if pd.isna(vix_std_val) or vix_std_val == 0:

                logger.warning("MarketSentiment._calculate_market_sentiment: VIX std is NaN or 0. Defaulting VIX component to neutral.")

                vix_component = 0.0
            else:
                vix_norm = (vix.iloc[-1] - vix_mean_val) / vix_std_val
                if pd.isna(vix_norm):

                     logger.warning("MarketSentiment._calculate_market_sentiment: vix_norm is NaN after calculation. Defaulting VIX component.")

                     vix_component = 0.0
                else:
                    vix_component = -np.tanh(vix_norm)

            sp500_short_series = sp500.iloc[-5:]
            sp500_medium_series = sp500.iloc[-20:]

            sp500_short_return = sp500_short_series.pct_change().mean() * 100 if len(sp500_short_series) >= 2 else 0.0
            sp500_medium_return = sp500_medium_series.pct_change().mean() * 100 if len(sp500_medium_series) >= 2 else 0.0

            if pd.isna(sp500_short_return): sp500_short_return = 0.0
            if pd.isna(sp500_medium_return): sp500_medium_return = 0.0

            sentiment = (
                0.4 * vix_component +
                0.4 * np.tanh(sp500_short_return / 5) +
                0.2 * np.tanh(sp500_medium_return / 10)
            )

            final_sentiment = max(-1.0, min(1.0, sentiment))

            logger.debug(f"MarketSentiment._calculate_market_sentiment components: vix_comp={vix_component:.2f}, sp500_short_ret={sp500_short_return:.2f}, sp500_med_ret={sp500_medium_return:.2f}, final_sent={final_sentiment:.2f}")
            return final_sentiment
        except Exception as e:
            logger.error(f"MarketSentiment: Error in _calculate_market_sentiment calculation: {e}", exc_info=True)
            return 0.0

