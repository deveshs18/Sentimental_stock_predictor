import os
import pandas as pd

# Paths
news_path = "data/news_combined.csv"
reddit_path = "data/reddit_posts.csv"
sentiment_path = "data/processed/sentiment_analysis.csv"
output_path = "data/merged_sentiment_input.csv"

# Load news and reddit data
news_df = pd.read_csv(news_path) if os.path.exists(news_path) else pd.DataFrame()
reddit_df = pd.read_csv(reddit_path) if os.path.exists(reddit_path) else pd.DataFrame()

# Standardize columns
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
news_df = news_df[[col for col in columns if col in news_df.columns]]
reddit_df = reddit_df[[col for col in columns if col in reddit_df.columns]]

# Convert timestamps
if "timestamp" in reddit_df.columns:
    reddit_df["timestamp"] = pd.to_datetime(reddit_df["timestamp"], unit="s", errors="coerce").dt.tz_localize(None)
if "timestamp" in news_df.columns:
    news_df["timestamp"] = pd.to_datetime(news_df["timestamp"], errors="coerce").dt.tz_localize(None)

# Merge news and reddit
combined_df = pd.concat([news_df, reddit_df], ignore_index=True)
combined_df = combined_df.dropna(subset=["headline", "text"])

# Load sentiment analysis results (with company and sentiment columns)
sentiment_df = pd.read_csv(sentiment_path) if os.path.exists(sentiment_path) else pd.DataFrame()

# Merge on timestamp, headline, and text (best effort)
if not sentiment_df.empty:
    # Only keep relevant columns from sentiment_df
    sentiment_cols = [col for col in ["timestamp", "headline", "text", "company", "sentiment", "sentiment_score"] if col in sentiment_df.columns]
    sentiment_df = sentiment_df[sentiment_cols]
    # Convert timestamp for join
    sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"], errors="coerce").dt.tz_localize(None)
    # Merge (best effort, may duplicate if not unique)
    combined_df = pd.merge(
        combined_df,
        sentiment_df,
        on=["timestamp", "headline", "text"],
        how="left"
    )

# Sort and save
combined_df = combined_df.sort_values(by="timestamp", ascending=False)
os.makedirs("data", exist_ok=True)
combined_df.to_csv(output_path, index=False)
print(f"✅ Merged news, reddit, and sentiment → {output_path}")
