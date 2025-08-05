
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

def time_decay_weight(timestamp, decay_rate=0.01):
    """Applies exponential decay to older timestamps. Higher decay_rate means faster decay."""
    if pd.isna(timestamp):
        return 0.0
    now = datetime.now(timezone.utc)
    time_diff_hours = (now - timestamp).total_seconds() / 3600
    return np.exp(-decay_rate * time_diff_hours)

def analyze_stocks():
    """
    Performs a comprehensive analysis by combining sentiment, ML predictions, and time decay.
    Outputs a single, clean predict_growth.csv file.
    """
    # Define paths
    processed_path = "data/processed"
    sentiment_file = "data/company_sentiment_normalized.csv"
    ml_predictions_file = "data/ml_predictions.csv"
    news_file = "data/processed/sentiment_analysis.csv" # Corrected input file
    output_file = os.path.join(processed_path, "predict_growth.csv")

    # Create processed directory if it doesn't exist
    os.makedirs(processed_path, exist_ok=True)

    # Load data
    try:
        sentiment_df = pd.read_csv(sentiment_file)
        ml_df = pd.read_csv(ml_predictions_file)
        news_df = pd.read_csv(news_file) # This now loads the correct file with sentiment and company
    except FileNotFoundError as e:
        print(f"Error: Missing required data file: {e}. Please ensure all previous steps have run.")
        return

    # 1. Process sentiment data
    # Ensure required columns exist
    if 'timestamp' not in news_df.columns or 'company' not in news_df.columns:
        print("Error: Missing required columns in sentiment data. Expected 'timestamp' and 'company' columns.")
        return

    # Convert timestamp and handle missing values
    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'], utc=True, errors='coerce')

    # Calculate sentiment score if not present
    if 'sentiment_score' not in news_df.columns:
        if 'sentiment' in news_df.columns and 'confidence' in news_df.columns:
            # Calculate sentiment score from sentiment and confidence
            news_df['sentiment_score'] = news_df.apply(
                lambda x: (1 if x['sentiment'] == 'positive' else -1 if x['sentiment'] == 'negative' else 0) * x.get('confidence', 1),
                axis=1
            )
        else:
            print("Warning: 'sentiment_score' column not found and cannot be calculated. Using default score of 0.")
            news_df['sentiment_score'] = 0.0
    
    # Drop rows with missing required data
    news_df = news_df.dropna(subset=['timestamp', 'company'])
    
    # Calculate time-decayed sentiment
    news_df['time_decay_weight'] = news_df['timestamp'].apply(time_decay_weight) 
    news_df['time_decayed_sentiment'] = news_df['sentiment_score'] * news_df['time_decay_weight']
    
    # Group by company and calculate metrics
    if not news_df.empty:
        time_decay_sentiment = news_df.groupby('company')['time_decayed_sentiment'].mean().reset_index()
        total_mentions = news_df.groupby('company').size().reset_index(name='total_mentions')
    else:
        print("Warning: No valid data available after filtering. Using empty DataFrames.")
        time_decay_sentiment = pd.DataFrame(columns=['company', 'time_decayed_sentiment'])
        total_mentions = pd.DataFrame(columns=['company', 'total_mentions'])

    # 2. Calculate Overall Market Sentiment
    # A simple average of all time-decayed news sentiment
    market_sentiment = time_decay_sentiment['time_decayed_sentiment'].mean()

    # 3. Merge DataFrames
    # Start with the base sentiment dataframe
    final_df = pd.merge(sentiment_df, time_decay_sentiment, on='company', how='left')
    final_df = pd.merge(final_df, total_mentions, on='company', how='left')
    final_df = pd.merge(final_df, ml_df, on='company', how='left')

    # Fill NaNs for companies that might be in one file but not another
    final_df.fillna({
        'time_decayed_sentiment': 0,
        'total_mentions': 0,
        'ml_prediction': 'neutral' # Default ML prediction
    }, inplace=True)

    # 4. Calculate Final Growth Score
    # Normalize scores to be on a similar scale (-1 to 1)
    scaler = lambda x: (x - x.min()) / (x.max() - x.min()) * 2 - 1 if x.max() - x.min() != 0 else 0
    final_df['sentiment_score_norm'] = scaler(final_df['normalized_sentiment'])
    final_df['time_decay_sentiment_norm'] = scaler(final_df['time_decayed_sentiment'])
    final_df['ml_prediction_score'] = final_df['ml_prediction'].apply(lambda x: 1 if x == 'up' else -1 if x == 'down' else 0)

    # Weighted combination
    final_df['growth_score'] = (
        0.40 * final_df['sentiment_score_norm'] +
        0.35 * final_df['time_decay_sentiment_norm'] +
        0.25 * final_df['ml_prediction_score']
    )

    # 5. Add market sentiment and clean up
    final_df['market_sentiment'] = market_sentiment
    
    # Select and rename columns to the final desired structure
    final_df = final_df[[
        'company',
        'normalized_sentiment',
        'time_decayed_sentiment',
        'ml_prediction',
        'growth_score',
        'market_sentiment',
        'total_mentions'
    ]].rename(columns={'normalized_sentiment': 'sentiment_score'})

    # Save the final, clean file
    final_df.to_csv(output_file, index=False)
    print(f"✅ Comprehensive analysis complete. Output saved to {output_file}")
    print("\n--- Top 5 Stocks ---")
    print(final_df.sort_values(by='growth_score', ascending=False).head(5))

if __name__ == "__main__":
    analyze_stocks()
