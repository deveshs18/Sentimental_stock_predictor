import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from dotenv import load_dotenv
import logging
import math
from datetime import datetime, timezone

# Load environment variables from the scripts/.env file
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv_path = os.path.join(script_dir, 'scripts', '.env')
load_dotenv(dotenv_path)

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('logs/sentiment_analysis.log', mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

# FinBERT model configuration
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = ["positive", "neutral", "negative"]

def classify_sentiment(text):
    """
    Classifies the sentiment of a given text using FinBERT.
    """
    if pd.isna(text) or not text.strip():
        return "neutral", 0.0
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=-1).squeeze().numpy()
        pred_idx = probs.argmax()
        return labels[pred_idx], float(probs[pred_idx])
    except Exception as e:
        logger.error(f"Error classifying sentiment for text: '{text}'. Error: {e}")
        return "neutral", 0.0

def time_decay_weight(timestamp, reference_time=None, decay_rate=0.00001):
    """
    Applies exponential decay to older timestamps.
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)
    if pd.isna(timestamp):
        return 0.0
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    delta_seconds = (reference_time - timestamp).total_seconds()
    return math.exp(-decay_rate * delta_seconds)

def main():
    """
    Main function to run sentiment analysis on the ingested data.
    """
    # Load ingested data
    try:
        news_df = pd.read_csv("data/news_combined.csv")
    except FileNotFoundError:
        logger.error("news_combined.csv not found. Please run the data ingestion pipeline first.")
        news_df = pd.DataFrame()

    try:
        reddit_df = pd.read_csv("data/reddit_posts.csv")
    except FileNotFoundError:
        logger.error("reddit_posts.csv not found. Please run the data ingestion pipeline first.")
        reddit_df = pd.DataFrame()

    if news_df.empty and reddit_df.empty:
        logger.error("No data to process. Exiting.")
        return

    combined_df = pd.concat([news_df, reddit_df], ignore_index=True)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce', utc=True)

    # Run sentiment analysis
    sentiments = []
    for index, row in combined_df.iterrows():
        combined_text = f"{row['headline']} {row['text']}"
        sentiment, confidence = classify_sentiment(combined_text)
        sentiments.append({
            "timestamp": row["timestamp"],
            "source": row["source"],
            "headline": row["headline"],
            "text": row["text"],
            "sentiment": sentiment,
            "confidence": confidence,
            "url": row["url"],
            "company": row["company"]
        })

    sentiment_df = pd.DataFrame(sentiments)

    # Calculate sentiment score with time decay
    sentiment_df['sentiment_score'] = sentiment_df.apply(
        lambda row: (1 if row['sentiment'] == 'positive' else -1 if row['sentiment'] == 'negative' else 0) * row['confidence'],
        axis=1
    )
    sentiment_df['time_decay_weight'] = sentiment_df['timestamp'].apply(time_decay_weight)
    sentiment_df['weighted_sentiment'] = sentiment_df['sentiment_score'] * sentiment_df['time_decay_weight']

        # Save results
        os.makedirs('data/processed', exist_ok=True)
        combined_df.to_csv('data/processed/sentiment_analysis.csv', index=False)
        
        # If we have company data, aggregate by company
        if 'company' in combined_df.columns and not combined_df['company'].isna().all():
            company_sentiment = combined_df.groupby('company').agg(
                weighted_sentiment_sum=('weighted_sentiment', 'sum'),
                article_count=('headline', 'count')
            ).reset_index()
            
            company_sentiment['normalized_sentiment'] = company_sentiment['weighted_sentiment_sum'] / company_sentiment['article_count']
            company_sentiment.to_csv('data/processed/company_sentiment.csv', index=False)
            # Also save to data/company_sentiment_normalized.csv for downstream compatibility
            os.makedirs('data', exist_ok=True)
            company_sentiment.to_csv('data/company_sentiment_normalized.csv', index=False)
            
            company_mentions = combined_df['company'].value_counts().reset_index()
            company_mentions.columns = ['company', 'mention_count']
            company_mentions.to_csv('data/processed/company_mentions.csv', index=False)
        
        logger.info(f"Successfully processed {len(combined_df)} items for sentiment analysis.")
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise

if __name__ == "__main__":
    main()
