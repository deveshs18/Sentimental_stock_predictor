import os
import pandas as pd
import logging
from thefuzz import fuzz
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
import re
from datetime import datetime, timezone
import math
from market_sentiment import MarketSentiment

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/predict_growth.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def time_decay_weight(timestamp, reference_time=None, decay_rate=0.00001):
    """Applies exponential decay to older timestamps."""
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    delta_seconds = (reference_time - timestamp).total_seconds()
    return math.exp(-decay_rate * delta_seconds)

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob."""
    if not text or pd.isna(text):
        return 0
    
    # Clean text
    text = str(text).strip()
    if not text:
        return 0
    
    # Analyze sentiment
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception as e:
        logger.warning(f"Error analyzing sentiment: {e}")
        return 0

def load_data():
    """Load and prepare data from CSV files."""
    try:
        # Load sentiment data
        sentiment_file = 'data/merged_sentiment_output_with_companies_normalized.csv'
        sentiment_df = pd.read_csv(sentiment_file)
        logger.info(f"Loaded sentiment data with {len(sentiment_df)} rows")
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in sentiment_df.columns:
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], errors='coerce')
        else:
            logger.warning("No timestamp column found in sentiment data. Using current time as reference.")
            sentiment_df['timestamp'] = datetime.now(timezone.utc)
        
        # Check for required columns
        if 'company' not in sentiment_df.columns:
            raise ValueError("'company' column not found in sentiment data")
            
        # Log available columns
        logger.info(f"Available columns in sentiment data: {list(sentiment_df.columns)}")
        
        # Check for text columns
        text_columns = [col for col in ['headline', 'text', 'content'] if col in sentiment_df.columns]
        if not text_columns:
            raise ValueError("No text columns found in sentiment data")
        logger.info(f"Using text columns: {text_columns}")
        
        # Add sentiment analysis if not present
        if 'sentiment' not in sentiment_df.columns or sentiment_df['sentiment'].isna().all():
            logger.info("'sentiment' column not found or empty, analyzing sentiment from text")
            # Combine all text columns
            sentiment_df['combined_text'] = sentiment_df[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
            # Analyze sentiment
            sentiment_df['sentiment_score'] = sentiment_df['combined_text'].apply(analyze_sentiment)
            # Convert to sentiment labels
            sentiment_df['sentiment'] = sentiment_df['sentiment_score'].apply(
                lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
            )
            logger.info(f"Sentiment distribution: {sentiment_df['sentiment'].value_counts().to_dict()}")
        
        # Load EDW keyword weights
        edw_weights = pd.read_csv('data/edw_keywords.csv')
        logger.info(f"Loaded {len(edw_weights)} EDW keywords")
        
        # Load NASDAQ companies
        nasdaq_companies = pd.read_csv('data/nasdaq_top_companies.csv')
        logger.info(f"Loaded {len(nasdaq_companies)} NASDAQ companies")
        
        # Find the company name column (case insensitive)
        company_cols = [col for col in nasdaq_companies.columns 
                       if 'company' in col.lower() or 'name' in col.lower()]
        
        if not company_cols:
            raise ValueError("Could not find company name column in NASDAQ companies file")
            
        company_col = company_cols[0]  # Use the first matching column
        logger.info(f"Using column '{company_col}' as company name in NASDAQ data")
        
        return sentiment_df, edw_weights, nasdaq_companies, company_col
        
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise

def calculate_growth_scores(sentiment_df, edw_weights, decay_rate=0.00001, market_sentiment=0):
    """Calculate growth scores based on sentiment, keyword weights, and time decay."""
    growth_scores = {}
    
    # Get the latest timestamp in the data for reference
    if 'timestamp' in sentiment_df.columns:
        reference_time = sentiment_df['timestamp'].max()
        if pd.isna(reference_time):
            reference_time = datetime.now(timezone.utc)
    else:
        reference_time = datetime.now(timezone.utc)
    
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    
    logger.info(f"Using reference time for time decay: {reference_time}")
    logger.info(f"Market sentiment adjustment factor: {market_sentiment:.2f}")
    
    # Preprocess keywords for matching
    keywords = {row['keyword'].lower(): row['weight'] 
               for _, row in edw_weights.iterrows()}
    
    for _, row in sentiment_df.iterrows():
        company = str(row.get('company', '')).strip()
        if not company:
            continue
            
        # Get sentiment score
        sentiment = str(row.get('sentiment', 'neutral')).lower()
        sentiment_score = 1 if sentiment == 'positive' else (-1 if sentiment == 'negative' else 0)
        
        # Skip if no sentiment
        if sentiment_score == 0:
            continue
            
        # Get text for keyword matching
        text = ' '.join(str(row[col]) for col in row.index 
                       if col in ['headline', 'text', 'content', 'combined_text'] 
                       and pd.notna(row[col]))
        
        # Calculate keyword weight
        keyword_weight = 0
        for keyword, weight in keywords.items():
            if keyword in text.lower():
                keyword_weight += weight
        
        # Calculate time decay
        timestamp = row.get('timestamp', reference_time)
        if pd.isna(timestamp):
            timestamp = reference_time
            
        time_weight = time_decay_weight(timestamp, reference_time, decay_rate)
        
        # Apply market sentiment adjustment (0.5 = neutral, >0.5 = positive, <0.5 = negative)
        market_factor = 0.5 + (market_sentiment / 2)  # Convert -1:1 to 0:1
        
        # Calculate final score component
        base_score = sentiment_score * (1 + keyword_weight) * time_weight
        score_component = base_score * market_factor
        
        # Update growth score
        if company not in growth_scores:
            growth_scores[company] = {
                'total_score': 0,
                'count': 0,
                'latest_article': timestamp,
                'earliest_article': timestamp
            }
        else:
            growth_scores[company]['latest_article'] = max(
                growth_scores[company].get('latest_article', timestamp),
                timestamp
            )
            growth_scores[company]['earliest_article'] = min(
                growth_scores[company].get('earliest_article', timestamp),
                timestamp
            )
        
        growth_scores[company]['total_score'] += score_component
        growth_scores[company]['count'] += 1
    
    # Calculate average growth score per company
    result = []
    for company, scores in growth_scores.items():
        result.append({
            'company': company,
            'growth_score': scores['total_score'] / scores['count'] if scores['count'] > 0 else 0,
            'mentions': scores['count'],
            'latest_article': scores.get('latest_article'),
            'earliest_article': scores.get('earliest_article')
        })
    
    return pd.DataFrame(result)

def main():
    try:
        # Initialize market sentiment analyzer
        market = MarketSentiment()
        market_data = market.get_market_indicators()
        
        if market_data:
            market_sentiment = market_data['market_sentiment']
            logger.info(f"\nMarket Sentiment: {market_sentiment:.2f}")
            logger.info(f"VIX: {market_data['vix_current']:.2f} (30d avg: {market_data['vix_30d_avg']:.2f})")
            logger.info(f"S&P 500 30d return: {market_data['sp500_30d_return']:.2f}%")
        else:
            logger.warning("Could not fetch market data. Using neutral sentiment.")
            market_sentiment = 0  # Neutral sentiment if no market data
        
        # Load data
        sentiment_df, edw_weights, nasdaq_companies, company_col = load_data()
        
        # Get list of valid company names
        valid_companies = set(nasdaq_companies[company_col].str.lower().dropna())
        
        # Match companies
        matched_companies = set()
        unmatched_companies = set()
        
        # First pass: exact case-insensitive match
        for company in sentiment_df['company'].dropna().unique():
            company_lower = str(company).lower()
            if company_lower in valid_companies:
                matched_companies.add(company)
            else:
                unmatched_companies.add(company)
        
        # Second pass: fuzzy match for unmatched companies
        fuzzy_matches = {}
        for company in unmatched_companies.copy():
            best_match = None
            best_score = 60  # Minimum threshold
            
            for valid_company in valid_companies:
                score = fuzz.ratio(str(company).lower(), valid_company.lower())
                if score > best_score:
                    best_score = score
                    best_match = valid_company
            
            if best_match:
                fuzzy_matches[company] = best_match
                matched_companies.add(company)
                unmatched_companies.remove(company)
        
        # Log matching statistics
        logger.info(f"Total companies in sentiment data: {len(sentiment_df['company'].unique())}")
        logger.info(f"Exact matches: {len(matched_companies) - len(fuzzy_matches)}")
        logger.info(f"Fuzzy matches: {len(fuzzy_matches)}")
        logger.info(f"Unmatched companies: {len(unmatched_companies)}")
        
        if fuzzy_matches:
            logger.info("Sample fuzzy matches:")
            for orig, match in list(fuzzy_matches.items())[:5]:  # Show first 5 matches
                logger.info(f"  '{orig}' -> '{match}'")
        
        if len(unmatched_companies) > 0:
            logger.warning(f"Could not match {len(unmatched_companies)} companies")
            logger.debug("Unmatched companies: " + ", ".join(sorted(list(unmatched_companies)[:20])))
        
        # Calculate growth scores with time decay and market sentiment
        logger.info("Calculating growth scores with time decay and market sentiment...")
        growth_scores = calculate_growth_scores(
            sentiment_df[sentiment_df['company'].isin(matched_companies)],
            edw_weights,
            decay_rate=0.00001,
            market_sentiment=market_sentiment
        )
        
        if growth_scores.empty:
            logger.warning("No growth scores calculated - check your data")
            return
        
        # Sort by growth score
        growth_scores = growth_scores.sort_values('growth_score', ascending=False)
        
        # Format dates for display
        if 'latest_article' in growth_scores.columns:
            growth_scores['latest_article'] = pd.to_datetime(growth_scores['latest_article']).dt.strftime('%Y-%m-%d')
        if 'earliest_article' in growth_scores.columns:
            growth_scores['earliest_article'] = pd.to_datetime(growth_scores['earliest_article']).dt.strftime('%Y-%m-%d')
        
        # Add market sentiment metrics to output
        if market_data:
            growth_scores['market_sentiment'] = market_sentiment
            growth_scores['vix'] = market_data['vix_current']
            growth_scores['sp500_30d_return'] = market_data['sp500_30d_return']
        
        # Save results
        output_file = 'data/predict_growth.csv'
        os.makedirs('data', exist_ok=True)
        growth_scores.to_csv(output_file, index=False)
        logger.info(f"Saved growth predictions to {output_file}")
        
        # Log top companies
        logger.info("\nTop 10 companies by growth score:")
        logger.info(growth_scores.head(10).to_string(index=False))
        
        # Log companies with most mentions
        if 'mentions' in growth_scores.columns:
            logger.info("\nTop 5 companies by number of mentions:")
            logger.info(growth_scores.nlargest(5, 'mentions')[['company', 'mentions']].to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()