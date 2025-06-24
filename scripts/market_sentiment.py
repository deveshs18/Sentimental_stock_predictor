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
        
        try:
            # Get VIX (Volatility Index)
            # Changed 'Adj Close' to 'Close' as it's more reliable for indices
            vix_data = yf.download(self.vix_symbol, start=start_date, end=end_date)
            if 'Close' not in vix_data.columns:
                logger.error(f"Could not find 'Close' column in VIX data. Columns: {vix_data.columns}")
                return None
            vix = vix_data['Close']

            # Get S&P 500
            # Changed 'Adj Close' to 'Close'
            sp500_data = yf.download(self.sp500_symbol, start=start_date, end=end_date)
            if 'Close' not in sp500_data.columns:
                logger.error(f"Could not find 'Close' column in S&P 500 data. Columns: {sp500_data.columns}")
                return None
            sp500 = sp500_data['Close']

            if vix.empty or sp500.empty:
                logger.error("VIX or S&P 500 data is empty after download and selecting 'Close' column.")
                return None

            # Ensure there's enough data for calculations
            if len(vix) < 1 or len(sp500) < 1:
                logger.error(f"Not enough data points after download. VIX points: {len(vix)}, S&P500 points: {len(sp500)}")
                return None

            vix_current = vix.iloc[-1] if not vix.empty else np.nan
            vix_mean = vix.mean()
            vix_std = vix.std()

            sp500_current = sp500.iloc[-1] if not sp500.empty else np.nan
            sp500_initial = sp500.iloc[0] if not sp500.empty else np.nan

            # Ensure scalar values for the conditional check to avoid "ValueError: The truth value of a Series is ambiguous"
            # by attempting to extract scalar if it's a single-item array/Series via .item()
            # This is a defensive measure in case iloc[-1] or iloc[0] didn't fully scalarize.
            
            vix_current_scalar = vix_current.item() if hasattr(vix_current, 'item') and callable(getattr(vix_current, 'item', None)) and np.size(vix_current) == 1 else vix_current
            sp500_current_scalar = sp500_current.item() if hasattr(sp500_current, 'item') and callable(getattr(sp500_current, 'item', None)) and np.size(sp500_current) == 1 else sp500_current
            sp500_initial_scalar = sp500_initial.item() if hasattr(sp500_initial, 'item') and callable(getattr(sp500_initial, 'item', None)) and np.size(sp500_initial) == 1 else sp500_initial

            is_vix_na = pd.isna(vix_current_scalar)
            is_sp500_current_na = pd.isna(sp500_current_scalar)
            is_sp500_initial_na = pd.isna(sp500_initial_scalar)
            # Ensure sp500_initial_scalar is not NaN before comparing to 0
            is_sp500_initial_zero = False
            if not is_sp500_initial_na:
                is_sp500_initial_zero = (sp500_initial_scalar == 0)

            if is_vix_na or is_sp500_current_na or is_sp500_initial_na or is_sp500_initial_zero:
                logger.error(f"Critical VIX/SP500 values are NaN or S&P initial is zero after download. VIX current: {vix_current_scalar}, SP500 current: {sp500_current_scalar}, SP500 initial: {sp500_initial_scalar}")
                return None

            # Calculate daily returns and moving averages
            sp500_returns = sp500.pct_change().dropna()
            # vix_returns = vix.pct_change().dropna() # vix_returns not used in current return dict

            sp500_volatility_annualized = sp500_returns.std() * np.sqrt(252) if not sp500_returns.empty else np.nan
            
            # Get economic indicators (example using FRED API)
            # Commented out as it requires API key
            # gdp = self._get_fred_data('GDP')
            # unemployment = self._get_fred_data('UNRATE')
            
            market_data_dict = {
                'vix_current': vix_current,
                'vix_30d_avg': vix_mean,
                'vix_30d_std': vix_std,
                'sp500_30d_return': (sp500_current / sp500_initial - 1) * 100,
                'sp500_volatility': sp500_volatility_annualized,
            }
            # Calculate market sentiment only if essential data is present
            # _calculate_market_sentiment itself needs at least 20 data points for sp500 for sp500_medium
            if len(vix) >= 1 and len(sp500) >= 20: # Adjusted minimum length for _calculate_market_sentiment
                 market_data_dict['market_sentiment'] = self._calculate_market_sentiment(vix, sp500)
            else:
                logger.warning(f"Not enough data for full market sentiment calculation (VIX len: {len(vix)}, S&P500 len: {len(sp500)}). Setting market_sentiment to neutral (0).")
                market_data_dict['market_sentiment'] = 0.0 # Default to neutral if not enough data

            logger.info(f"Successfully fetched and processed market indicators: {market_data_dict}")
            return market_data_dict
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}", exc_info=True)
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
            return response.json()['observations'][0]['value']
        except Exception as e:
            logger.warning(f"Could not fetch FRED data: {e}")
            return None
    
    def _calculate_market_sentiment(self, vix, sp500):
        """Calculate a composite market sentiment score (-1 to 1)."""
        # Normalize VIX (higher VIX = more fear)
        vix_norm = (vix[-1] - vix.mean()) / vix.std()
        vix_score = np.exp(-vix_norm)  # Invert so higher VIX = lower score
        
        # Calculate trend (recent returns)
        sp500_short = sp500[-5:].pct_change().mean() * 100  # 5-day return
        sp500_medium = sp500[-20:].pct_change().mean() * 100  # 20-day return
        
        # Combine factors (you can adjust weights)
        sentiment = (
            0.4 * vix_score + 
            0.4 * np.tanh(sp500_short/5) +  # Scale returns
            0.2 * np.tanh(sp500_medium/10)
        )
        
        # Ensure score is between -1 and 1
        return max(-1, min(1, sentiment))