import pandas as pd
import os
import logging

# Configure logging
os.makedirs('../logs', exist_ok=True)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('../logs/merge_sources.log', mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

# Load both datasets
news_path = "data/news_combined.csv"
reddit_path = "data/reddit_posts.csv"

logger.info(f"Loading news data from {news_path}")
try:
    news_df = pd.read_csv(news_path)
    logger.info(f"Loaded {len(news_df)} news articles.")
except FileNotFoundError:
    logger.error(f"News data file not found: {news_path}. Please run fetch_news.py first.")
    raise
except Exception as e:
    logger.error(f"Error loading news data: {e}", exc_info=True)
    raise

logger.info(f"Loading Reddit data from {reddit_path}")
try:
    reddit_df = pd.read_csv(reddit_path)
    logger.info(f"Loaded {len(reddit_df)} Reddit posts.")
except FileNotFoundError:
    logger.error(f"Reddit data file not found: {reddit_path}. Please run fetch_reddit.py first.")
    raise
except Exception as e:
    logger.error(f"Error loading Reddit data: {e}", exc_info=True)
    raise

# Normalize column names
logger.info("Normalizing column names...")
news_df = news_df.rename(columns={
    "title": "headline",
    "description": "text",
    "publishedAt": "timestamp"
})
news_df["source"] = "news"

reddit_df = reddit_df.rename(columns={
    "title": "headline",
    "text": "text",
    "created_utc": "timestamp"
})
reddit_df["source"] = "reddit"

# Standardize column order
columns = ["timestamp", "headline", "text", "source", "url"]
news_df = news_df[columns]
reddit_df = reddit_df[columns]

# Convert and normalize timestamps (remove timezone to avoid TypeError)
reddit_df["timestamp"] = pd.to_datetime(reddit_df["timestamp"], unit="s", errors="coerce").dt.tz_localize(None)
news_df["timestamp"] = pd.to_datetime(news_df["timestamp"], errors="coerce").dt.tz_localize(None)


# Merge and sort
combined_df = pd.concat([news_df, reddit_df], ignore_index=True)
combined_df = combined_df.dropna(subset=["headline", "text"])
combined_df = combined_df.sort_values(by="timestamp", ascending=False)
logger.info(f"Combined data has {len(combined_df)} rows after merging and cleaning.")

# Save merged file
os.makedirs("data", exist_ok=True)
output_path = "data/merged_sentiment_input.csv"
try:
    combined_df.to_csv(output_path, index=False)
    logger.info(f"✅ Merged News + Reddit → {output_path}")
except Exception as e:
    logger.error(f"Error saving merged data: {e}", exc_info=True)
    raise
