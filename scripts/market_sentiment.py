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
        logger.debug(f"Fetching market data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
<<<<<<< feature/enhanced-stock-prediction-v2
            vix_data = yf.download(self.vix_symbol, start=start_date, end=end_date, progress=False)
            if vix_data.empty or 'Close' not in vix_data.columns:
                logger.error(f"VIX data is empty or 'Close' column missing. Columns: {vix_data.columns if not vix_data.empty else 'N/A (empty df)'}")
                return None
            vix = vix_data['Close'].dropna()
            
            sp500_data = yf.download(self.sp500_symbol, start=start_date, end=end_date, progress=False)
            if sp500_data.empty or 'Close' not in sp500_data.columns:
                logger.error(f"S&P 500 data is empty or 'Close' column missing. Columns: {sp500_data.columns if not sp500_data.empty else 'N/A (empty df)'}")
                return None
            sp500 = sp500_data['Close'].dropna()

            if vix.empty:
                logger.error("VIX data is empty after selecting 'Close' column and dropping NaNs.")
                return None
            if sp500.empty:
                logger.error("S&P 500 data is empty after selecting 'Close' column and dropping NaNs.")
                return None
=======
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
>>>>>>> main
            
            # These checks are critical before iloc access
            if len(vix) < 1:
                logger.error(f"Not enough data points for VIX after download and dropna. Points: {len(vix)}")
                return None
            if len(sp500) < 1:
                logger.error(f"Not enough data points for S&P500 after download and dropna. Points: {len(sp500)}")
                return None

            vix_current_scalar = vix.iloc[-1]
            vix_mean_val = vix.mean()
            vix_std_val = vix.std(ddof=0) # Use population standard deviation for consistency if sample is small

            sp500_current_scalar = sp500.iloc[-1]
            sp500_initial_scalar = sp500.iloc[0]
            
<<<<<<< feature/enhanced-stock-prediction-v2
            # Check for NaNs after iloc, which can happen if series became all NaN and then iloc was on empty after dropna
            if pd.isna(vix_current_scalar) or pd.isna(vix_mean_val) or pd.isna(vix_std_val) or \
               pd.isna(sp500_current_scalar) or pd.isna(sp500_initial_scalar) or \
               (not pd.isna(sp500_initial_scalar) and sp500_initial_scalar == 0): # Check for zero only if not NaN
                logger.error(f"Critical VIX/SP500 scalar values are NaN or S&P initial is zero. "
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
            }

            if len(vix) >= 2 and len(sp500) >= 20: # Min lengths for meaningful calculation in _calculate_market_sentiment
                 market_data_dict['market_sentiment'] = self._calculate_market_sentiment(vix, sp500)
            else:
                logger.warning(f"Not enough data for full market sentiment calculation (VIX len: {len(vix)}, S&P500 len: {len(sp500)}). Setting market_sentiment to neutral (0.0).")
                market_data_dict['market_sentiment'] = 0.0
=======
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
>>>>>>> main

            logger.info(f"Successfully fetched and processed market indicators: {market_data_dict}")
            return market_data_dict
            
        except Exception as e:
<<<<<<< feature/enhanced-stock-prediction-v2
            logger.error(f"Error processing market data in get_market_indicators: {e}", exc_info=True)
=======
            logger.error(f"Error processing market data: {e}", exc_info=True)
>>>>>>> main
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
            logger.warning(f"Could not fetch FRED data for series {series_id}: {e}")
            return None
    
    def _calculate_market_sentiment(self, vix, sp500):
        """Calculate a composite market sentiment score (-1 to 1)."""
<<<<<<< feature/enhanced-stock-prediction-v2
        if len(vix) < 2:
            logger.warning(f"_calculate_market_sentiment: VIX series has {len(vix)} points, less than 2 required for std dev. Returning 0.0.")
            return 0.0
        if len(sp500) < 20:
            logger.warning(f"_calculate_market_sentiment: S&P500 series has {len(sp500)} points, less than 20 required for medium term trend. Returning 0.0.")
            return 0.0

        try:
            vix_mean_val = vix.mean()
            vix_std_val = vix.std(ddof=0)

            if pd.isna(vix_std_val) or vix_std_val == 0:
                logger.warning("_calculate_market_sentiment: VIX std is NaN or 0. Defaulting VIX component to neutral.")
                vix_component = 0.0
            else:
                vix_norm = (vix.iloc[-1] - vix_mean_val) / vix_std_val
                if pd.isna(vix_norm):
                     logger.warning("_calculate_market_sentiment: vix_norm is NaN after calculation. Defaulting VIX component.")
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
            logger.debug(f"_calculate_market_sentiment components: vix_comp={vix_component:.2f}, sp500_short_ret={sp500_short_return:.2f}, sp500_med_ret={sp500_medium_return:.2f}, final_sent={final_sentiment:.2f}")
            return final_sentiment
        except Exception as e:
            logger.error(f"Error in _calculate_market_sentiment calculation: {e}", exc_info=True)
            return 0.0
=======
        # Ensure we have enough data points for these operations
        if len(vix) < 1 or len(sp500) < 20: # Need at least 20 for sp500_medium, 1 for vix_norm
            logger.warning(f"_calculate_market_sentiment: Not enough data. VIX len: {len(vix)}, S&P500 len: {len(sp500)}. Returning 0.")
            return 0.0

        # Normalize VIX (higher VIX = more fear)
        # Use .iloc for robust positional access
        vix_norm = (vix.iloc[-1] - vix.mean()) / vix.std()
        vix_score = np.exp(-vix_norm)  # Invert so higher VIX = lower score
        
        # Calculate trend (recent returns)
        # Use .iloc for robust positional access for slicing
        sp500_short = sp500.iloc[-5:].pct_change().mean() * 100 if len(sp500) >= 5 else 0 # 5-day return
        sp500_medium = sp500.iloc[-20:].pct_change().mean() * 100 if len(sp500) >= 20 else 0 # 20-day return
        
        # Combine factors (you can adjust weights)
        sentiment = (
            0.4 * vix_score + 
            0.4 * np.tanh(sp500_short/5) +  # Scale returns
            0.2 * np.tanh(sp500_medium/10)
        )
        
        # Ensure score is between -1 and 1
        return max(-1, min(1, sentiment))
>>>>>>> main
