import os
import sys
import pandas as pd
import requests
import praw
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('logs/data_ingestion.log', mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

# NewsAPI configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Reddit API configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

def fetch_news(keywords):
    """
    Fetches news articles from NewsAPI based on a list of keywords.
    """
    if not NEWS_API_KEY:
        logger.error("Missing NewsAPI key. Please check your .env file.")
        return pd.DataFrame()

    articles = []
    for keyword in keywords:
        url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={NEWS_API_KEY}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for article in data.get("articles", []):
                    articles.append({
                        "timestamp": article.get("publishedAt"),
                        "source": "news",
                        "headline": article.get("title"),
                        "text": article.get("description"),
                        "url": article.get("url"),
                        "company": keyword
                    })
            else:
                logger.error(f"Error fetching news for '{keyword}': Status code {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching news for '{keyword}': {e}")

    return pd.DataFrame(articles)

def fetch_reddit_posts(subreddits):
    """
    Fetches posts from Reddit based on a list of subreddits.
    """
    if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
        logger.error("Missing Reddit API credentials. Please check your .env file.")
        return pd.DataFrame()

    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        logger.info("Reddit API initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Reddit API: {e}", exc_info=True)
        return pd.DataFrame()

    posts = []
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            for post in subreddit.hot(limit=50):
                posts.append({
                    "timestamp": post.created_utc,
                    "source": "reddit",
                    "headline": post.title,
                    "text": post.selftext,
                    "url": post.url,
                    "company": "" # Company name will be extracted later
                })
        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit_name}: {e}")

    return pd.DataFrame(posts)

def main(keywords, subreddits):
    """
    Main function to fetch news and Reddit posts and save them to CSV files.
    """
    news_df = fetch_news(keywords)
    if not news_df.empty:
        news_df.to_csv("data/news_combined.csv", index=False)
        logger.info("Successfully saved news data to data/news_combined.csv")

    reddit_df = fetch_reddit_posts(subreddits)
    if not reddit_df.empty:
        reddit_df.to_csv("data/reddit_posts.csv", index=False)
        logger.info("Successfully saved Reddit data to data/reddit_posts.csv")

if __name__ == "__main__":
    # Example usage
    keywords = ["AAPL", "GOOGL", "MSFT"]
    subreddits = ["stocks", "investing", "wallstreetbets"]
    main(keywords, subreddits)
