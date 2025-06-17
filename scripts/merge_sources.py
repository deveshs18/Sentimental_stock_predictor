import pandas as pd
import os

# Load both datasets
news_path = "data/news_combined.csv"
reddit_path = "data/reddit_posts.csv"

news_df = pd.read_csv(news_path)
reddit_df = pd.read_csv(reddit_path)

# Normalize column names
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

# Save merged file
os.makedirs("data", exist_ok=True)
combined_df.to_csv("data/merged_sentiment_input.csv", index=False)
print("✅ Merged News + Reddit → data/merged_sentiment_input.csv")
