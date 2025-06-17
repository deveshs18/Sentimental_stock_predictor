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
            vix = yf.download(self.vix_symbol, start=start_date, end=end_date)['Adj Close']
            # Get S&P 500
            sp500 = yf.download(self.sp500_symbol, start=start_date, end=end_date)['Adj Close']
            
            # Calculate daily returns and moving averages
            sp500_returns = sp500.pct_change().dropna()
            vix_returns = vix.pct_change().dropna()
            
            # Get economic indicators (example using FRED API)
            # Commented out as it requires API key
            # gdp = self._get_fred_data('GDP')
            # unemployment = self._get_fred_data('UNRATE')
            
            return {
                'vix_current': vix[-1],
                'vix_30d_avg': vix.mean(),
                'vix_30d_std': vix.std(),
                'sp500_30d_return': (sp500[-1] / sp500[0] - 1) * 100,
                'sp500_volatility': sp500_returns.std() * np.sqrt(252),  # Annualized
                'market_sentiment': self._calculate_market_sentiment(vix, sp500)
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
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