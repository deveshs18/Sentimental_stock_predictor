"""
Script to predict company growth scores based on sentiment analysis, keyword weights, and market indicators.

Usage:
    python predict_growth.py [--output OUTPUT_FILE]

By default, results are saved to 'data/predict_growth.csv'.
"""

import os
import numpy as np
import pandas as pd
import logging
from thefuzz import fuzz
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
import re
from datetime import datetime, timezone
import math
# Import removed as we're not using MarketSentiment anymore

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
# Create logs directory relative to the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'predict_growth.log')

# Clear previous log file if it exists
if os.path.exists(log_file):
    os.remove(log_file)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Write to file
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_file}")

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

def load_data():
    """Load and prepare data from CSV files."""
    try:
        logger.info("Loading data...")
        # Get the project root directory (three levels up from this script)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logger.info(f"Project root: {project_root}")
        
        # Check if data directory exists
        data_dir = os.path.join(project_root, 'data')
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        # Check if processed directory exists
        processed_dir = os.path.join(data_dir, 'processed')
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
            
        # List files in processed directory
        logger.info(f"Files in processed directory: {os.listdir(processed_dir)}")
        
        # Try loading sentiment data with error handling for malformed CSV
        sentiment_file = os.path.join(project_root, 'data', 'processed', 'sentiment_analysis.csv')
        try:
            # First try with error handling for malformed CSV
            try:
                sentiment_df = pd.read_csv(sentiment_file, on_bad_lines='skip')
                logger.info(f"Initially loaded {len(sentiment_df)} rows from {sentiment_file}")
                
                # If we have data but missing required columns, try with different encodings
                required_cols = ['company', 'sentiment', 'sentiment_score', 'timestamp']
                if not all(col in sentiment_df.columns for col in required_cols):
                    logger.warning("Required columns missing, trying different CSV parsing options")
                    
                    # Try with different encodings and error handling
                    for encoding in ['utf-8', 'latin1', 'cp1252']:
                        try:
                            sentiment_df = pd.read_csv(
                                sentiment_file,
                                encoding=encoding,
                                on_bad_lines='skip',
                                engine='python',
                                quoting=3  # QUOTE_NONE
                            )
                            if not sentiment_df.empty:
                                logger.info(f"Successfully loaded with {encoding} encoding")
                                break
                        except Exception as e:
                            logger.warning(f"Failed with {encoding} encoding: {str(e)}")
                            
                    # If we still don't have the right columns, try to rename them
                    if not all(col in sentiment_df.columns for col in required_cols):
                        logger.warning("Renaming columns to match expected format")
                        column_map = {
                            'Company': 'company',
                            'Sentiment': 'sentiment',
                            'Sentiment_Score': 'sentiment_score',
                            'Timestamp': 'timestamp',
                            'Date': 'timestamp',
                            'Text': 'text',
                            'Source': 'source'
                        }
                        sentiment_df = sentiment_df.rename(columns=column_map)
                        
            except Exception as e:
                logger.error(f"Error loading {sentiment_file}: {str(e)}")
                raise
                
            logger.info(f"Successfully loaded {len(sentiment_df)} rows from {sentiment_file}")
            
        except Exception as e:
            logger.error(f"Failed to load sentiment data: {str(e)}")
            # Fall back to company_sentiment.csv if available
            fallback_file = os.path.join(project_root, 'data', 'processed', 'company_sentiment.csv')
            try:
                sentiment_df = pd.read_csv(fallback_file)
                logger.info(f"Loaded fallback sentiment data with {len(sentiment_df)} rows from {fallback_file}")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback data: {str(fallback_error)}")
                raise FileNotFoundError("Could not load any sentiment data files")
        
        # Ensure required columns exist
        required_columns = ['company', 'sentiment_score', 'timestamp']
        missing_columns = [col for col in required_columns if col not in sentiment_df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in sentiment data: {', '.join(missing_columns)}")
            
            # If we have sentiment but not sentiment_score, create it
            if 'sentiment' in sentiment_df.columns and 'sentiment_score' in missing_columns:
                logger.info("Generating sentiment_score from sentiment column")
                sentiment_map = {
                    'positive': 1.0, 
                    'neutral': 0.0, 
                    'negative': -1.0,
                    'very_positive': 1.0,
                    'very_negative': -1.0,
                    'slightly_positive': 0.5,
                    'slightly_negative': -0.5
                }
                sentiment_df['sentiment_score'] = (
                    sentiment_df['sentiment']
                    .str.lower()
                    .map(sentiment_map)
                    .fillna(0.0)  # Default to neutral for unknown labels
                )
                missing_columns.remove('sentiment_score')
                
            # If still missing required columns, raise error
            if missing_columns:
                raise ValueError(f"Missing required columns in sentiment data: {', '.join(missing_columns)}")
        
        # Ensure sentiment_score is float and fill any NaNs with 0
        sentiment_df['sentiment_score'] = pd.to_numeric(sentiment_df['sentiment_score'], errors='coerce').fillna(0.0)
        
        # Log sentiment score statistics
        logger.info(f"Sentiment score statistics:\n{sentiment_df['sentiment_score'].describe()}")
        if 'sentiment' in sentiment_df.columns:
            logger.info(f"Sentiment value counts (top 10):\n{sentiment_df['sentiment'].value_counts().head(10)}")
            
        logger.info(f"Found required columns: {required_columns}")
        logger.info(f"Available columns in sentiment data: {list(sentiment_df.columns)}")
        
        # Load EDW keyword weights
        edw_weights_path = os.path.join(project_root, 'data', 'edw_keywords.csv')
        edw_weights = pd.read_csv(edw_weights_path)
        logger.info(f"Loaded {len(edw_weights)} EDW keywords from {edw_weights_path}")
        
        # Load NASDAQ companies with error handling for malformed CSV
        nasdaq_path = os.path.join(project_root, 'data', 'nasdaq_top_companies.csv')
        try:
            # First try with error handling for malformed CSV
            nasdaq_companies = pd.read_csv(nasdaq_path, on_bad_lines='skip')
            logger.info(f"Initially loaded {len(nasdaq_companies)} NASDAQ companies from {nasdaq_path}")
            
            # If we don't have enough columns, try with different parsing options
            if len(nasdaq_companies.columns) < 2:
                logger.warning("Not enough columns in NASDAQ data, trying with different parsing options")
                for sep in [',', ';', '\t']:
                    try:
                        nasdaq_companies = pd.read_csv(nasdaq_path, sep=sep, on_bad_lines='skip')
                        if len(nasdaq_companies.columns) >= 2:
                            logger.info(f"Successfully loaded with separator: {repr(sep)}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed with separator {repr(sep)}: {str(e)}")
            
            # Ensure we have at least 2 columns (company name and symbol)
            if len(nasdaq_companies.columns) < 2:
                raise ValueError(f"Not enough columns in NASDAQ data. Found: {nasdaq_companies.columns.tolist()}")
                
            # Use the first two columns as company name and symbol
            nasdaq_companies = nasdaq_companies.iloc[:, :2]
            nasdaq_companies.columns = ['company', 'symbol']
            
            logger.info(f"Successfully loaded {len(nasdaq_companies)} NASDAQ companies")
            
        except Exception as e:
            logger.error(f"Error loading NASDAQ companies: {str(e)}")
            logger.warning("Creating a minimal NASDAQ companies DataFrame with common stocks")
            # Create a minimal DataFrame with common stocks as fallback
            nasdaq_companies = pd.DataFrame({
                'company': [
                    'Apple Inc.', 'Microsoft Corporation', 'Amazon.com Inc.', 
                    'Alphabet Inc.', 'Meta Platforms Inc.', 'Tesla Inc.'
                ],
                'symbol': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA']
            })
        
        # Use 'company' as the company column (we set this in the previous step)
        company_col = 'company'
        if company_col not in nasdaq_companies.columns:
            # Try to find a suitable column
            possible_cols = ['company', 'name', 'company_name', 'Company', 'Name']
            for col in possible_cols:
                if col in nasdaq_companies.columns:
                    company_col = col
                    break
            else:
                # If no suitable column found, use the first column
                company_col = nasdaq_companies.columns[0]
                logger.warning(f"Using first column '{company_col}' as company name")
            
        logger.info(f"Using column '{company_col}' as company name in NASDAQ data")
        
        return sentiment_df, edw_weights, nasdaq_companies, company_col
        
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise

def calculate_growth_scores(sentiment_df, edw_weights, decay_rate=0.00001, market_sentiment=0):
    """Calculate growth scores based on sentiment, keyword weights, and time decay.
    
    Args:
        sentiment_df: DataFrame containing sentiment data with columns:
            - company: Company name or ticker
            - sentiment_score: Numeric sentiment score (-1 to 1)
            - timestamp: When the sentiment was recorded
            - (optional) text/headline: Text content for keyword analysis
        edw_weights: DataFrame with columns 'keyword' and 'weight'
        decay_rate: Rate at which older sentiment scores decay (default: 0.00001)
        market_sentiment: Overall market sentiment adjustment (-1 to 1)
        
    Returns:
        DataFrame with growth scores for each company
    """
    logger.info("="*80)
    logger.info("STARTING GROWTH SCORE CALCULATION")
    logger.info("="*80)
    
    # Log input data summary
    logger.info(f"Input data shape: {sentiment_df.shape}")
    logger.info(f"Sample data (first 2 rows):\n{sentiment_df[['company', 'sentiment_score', 'timestamp']].head(2).to_string()}")
    
    # Validate input data
    if sentiment_df.empty:
        logger.error("ERROR: No sentiment data provided")
        return pd.DataFrame()
    
    # Check required columns
    required_cols = ['company', 'sentiment_score', 'timestamp']
    missing_cols = [col for col in required_cols if col not in sentiment_df.columns]
    if missing_cols:
        logger.error(f"ERROR: Missing required columns: {missing_cols}")
        logger.error(f"Available columns: {list(sentiment_df.columns)}")
        return pd.DataFrame()
    
    # Log data types and missing values
    logger.info("\nData types and non-null counts:")
    logger.info(sentiment_df[required_cols].info())
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(sentiment_df['timestamp']):
        try:
            logger.info("Converting timestamp column to datetime...")
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
            logger.info("Timestamp conversion successful")
        except Exception as e:
            logger.error(f"ERROR converting timestamp: {str(e)}")
            logger.error(f"Sample timestamps: {sentiment_df['timestamp'].head().tolist()}")
            return pd.DataFrame()
    
    # Set reference time for decay calculation
    reference_time = sentiment_df['timestamp'].max()
    if pd.isna(reference_time):
        reference_time = datetime.now(timezone.utc)
        logger.warning(f"No valid timestamps found, using current time: {reference_time}")
    
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    
    logger.info(f"\n=== REFERENCE TIME ===\n{reference_time}")
    logger.info(f"=== MARKET SENTIMENT ADJUSTMENT ===\n{market_sentiment:.2f}")
    
    # Log sentiment score distribution
    logger.info("\n=== SENTIMENT SCORE DISTRIBUTION ===")
    logger.info(sentiment_df['sentiment_score'].describe().to_string())
    
    # Initialize results
    growth_scores = {}
    
    # Process each company's sentiment data
    logger.info("\n=== PROCESSING COMPANIES ===")
    companies_processed = 0
    
    for company, group in sentiment_df.groupby('company'):
        try:
            if pd.isna(company):
                logger.warning("Skipping record with missing company name")
                continue
                
            if len(group) == 0:
                logger.warning(f"No data for company: {company}")
                continue
                
            logger.info(f"\nProcessing company: {company}")
            logger.info(f"Number of records: {len(group)}")
            
            # Calculate time decay weights
            time_deltas = (reference_time - group['timestamp']).dt.total_seconds()
            weights = np.exp(-decay_rate * time_deltas)
            
            # Log some sample weights
            logger.info(f"Sample time deltas (hours): {time_deltas.head().values / 3600}")
            logger.info(f"Sample weights: {weights.head().values}")
            
            # Ensure we have valid sentiment scores
            valid_mask = (
                group['sentiment_score'].notna() & 
                ~np.isinf(group['sentiment_score']) &
                ~group['sentiment_score'].isna()
            )
            valid_scores = group.loc[valid_mask, 'sentiment_score']
            
            if len(valid_scores) == 0:
                logger.warning(f"No valid sentiment scores for {company}")
                logger.warning(f"Sample invalid scores: {group['sentiment_score'].head().tolist()}")
                continue
                
            logger.info(f"Valid sentiment scores: {len(valid_scores)}/{len(group)}")
            logger.info(f"Score stats - Min: {valid_scores.min():.4f}, "
                       f"Max: {valid_scores.max():.4f}, "
                       f"Mean: {valid_scores.mean():.4f}")
            
            # Calculate weighted sentiment scores
            weighted_scores = valid_scores * weights[valid_scores.index]
            
            # Apply market sentiment adjustment
            adjusted_scores = weighted_scores * (1 + market_sentiment)
            
            # Calculate final growth score (scaled to 0-100 range)
            avg_score = np.mean(adjusted_scores) if len(adjusted_scores) > 0 else 0
            growth_score = 50 + (avg_score * 50)  # Scale from [-1,1] to [0,100]
            
            # Store results
            growth_scores[company] = {
                'company': company,
                'growth_score': growth_score,
                'avg_sentiment': avg_score,
                'sentiment_count': len(adjusted_scores),
                'last_updated': reference_time.strftime('%Y-%m-%d %H:%M:%S %Z')
            }
            
            logger.info(f"Calculated scores for {company}: {growth_scores[company]}")
            companies_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing company {company}: {str(e)}", exc_info=True)
    
    # Convert results to DataFrame
    if not growth_scores:
        logger.error("No growth scores were calculated for any company")
        return pd.DataFrame()
        
    result_df = pd.DataFrame(growth_scores.values())
    
    # Log final results
    logger.info("\n" + "="*80)
    logger.info("GROWTH SCORE CALCULATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Successfully processed {companies_processed} companies")
    logger.info(f"Final growth scores (top 10):\n{result_df.sort_values('growth_score', ascending=False).head(10).to_string()}")
    
    return result_df

import argparse

def main():
    try:
        logger.info("Starting prediction pipeline...")
        
        # Get the project root directory (three levels up from this script)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logger.info(f"Project root: {project_root}")
        
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Predict company growth scores based on sentiment and market data.")
        parser.add_argument('--output', type=str, default='data/processed/predict_growth.csv',
                          help='Output CSV file path (default: data/processed/predict_growth.csv)')
        args = parser.parse_args()
        
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Check if processed directory exists
        processed_dir = os.path.join(project_root, 'data', 'processed')
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
            
        # List files in processed directory
        logger.info(f"Files in processed directory: {os.listdir(processed_dir)}")
        
        # Use neutral market sentiment since we don't have the MarketSentiment class
        market_sentiment = 0  # Neutral sentiment
        
        # Load and prepare data
        logger.info("Loading data...")
        sentiment_df, edw_weights, nasdaq_companies, company_col = load_data()
        
        if sentiment_df is None or sentiment_df.empty:
            logger.error("No sentiment data available for processing")
            return
            
        logger.info(f"Data loaded successfully. Sentiment data shape: {sentiment_df.shape}")
        logger.info(f"Sample sentiment data: {sentiment_df[['company', 'sentiment_score', 'timestamp']].head(2).to_dict()}")
        
        # Get unique companies
        companies = sentiment_df['company'].unique()
        logger.info(f"Total companies in sentiment data: {len(companies)}")
        
        # Match with NASDAQ companies
        matched_companies = []
        unmatched_companies = []
        
        for company in companies:
            if pd.isna(company):
                continue
                
            # Try exact match first
            matches = nasdaq_companies[nasdaq_companies[company_col].str.lower() == str(company).lower()]
            if not matches.empty:
                matched_companies.append((company, matches.iloc[0][company_col], 'exact'))
                continue
                
            # Try fuzzy match if no exact match
            max_ratio = 0
            best_match = None
            for _, row in nasdaq_companies.iterrows():
                ratio = fuzz.ratio(str(company).lower(), str(row[company_col]).lower())
                if ratio > max_ratio and ratio > 80:  # Threshold for fuzzy matching
                    max_ratio = ratio
                    best_match = row[company_col]
                    
            if best_match is not None:
                matched_companies.append((company, best_match, 'fuzzy'))
            else:
                unmatched_companies.append(company)
                
        logger.info(f"Exact matches: {len([m for m in matched_companies if m[2] == 'exact'])}")
        logger.info(f"Fuzzy matches: {len([m for m in matched_companies if m[2] == 'fuzzy'])}")
        logger.info(f"Unmatched companies: {len(unmatched_companies)}")
        
        if not matched_companies:
            logger.error("No companies matched with NASDAQ data. Cannot calculate growth scores.")
            return
            
        # Calculate growth scores
        logger.info("Calculating growth scores with time decay and market sentiment...")
        growth_scores = calculate_growth_scores(sentiment_df, edw_weights, market_sentiment=market_sentiment)
        
        if growth_scores is None or growth_scores.empty:
            logger.warning("No growth scores calculated - check your data")
            return
            
        # Save results
        output_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output
        growth_scores.to_csv(output_path, index=False)
        logger.info(f"Growth scores saved to {output_path}")
        
        # Log top 10 companies by growth score
        if not growth_scores.empty:
            logger.info("\nTop 10 companies by growth score:")
            logger.info(growth_scores.head(10).to_string())
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise
if __name__ == "__main__":
    main()