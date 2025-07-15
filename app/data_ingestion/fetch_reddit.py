import praw
import os
import sys
import pandas as pd
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs('logs', exist_ok=True)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('logs/fetch_reddit.log', mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

# Check for API keys
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
    logger.error("Missing Reddit API credentials. Please check your .env file.")
    sys.exit(1)

# Initialize Reddit API with PRAW
try:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    logger.info("Reddit API initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Reddit API: {e}", exc_info=True)
    sys.exit(1)


# 🔍 Fetch top N posts from each subreddit (like NewsAPI: general trends)
def fetch_latest_posts(subreddit_name="stocks", limit=50):
    posts = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=limit):
            posts.append({
                "title": post.title,
                "text": post.selftext,
                "created_utc": post.created_utc,
                "url": post.url,
                "subreddit": subreddit_name
            })
    except Exception as e:
        logger.error(f"Error fetching posts from r/{subreddit_name}: {e}", exc_info=True)
    return posts

if __name__ == "__main__":
    subreddits = ["stocks", "investing", "wallstreetbets"]
    all_posts = []

    for sub in subreddits:
        logger.info(f"🔍 Scraping top posts from r/{sub}...")
        data = fetch_latest_posts(subreddit_name=sub, limit=50)
        if data:
            all_posts.extend(data)
            logger.info(f"✅ Retrieved {len(data)} posts from r/{sub}")
        else:
            logger.warning(f"⚠️ No posts retrieved from r/{sub}")

    if not all_posts:
        logger.warning("No posts retrieved from any subreddit. Exiting.")
        sys.exit(0) # Not an error, but no data to process

    # Create a DataFrame and save
    df = pd.DataFrame(all_posts)
    os.makedirs("data", exist_ok=True)
    file_path = "data/reddit_posts.csv"
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"📄 Saved all Reddit posts → {file_path}")
    except Exception as e:
        logger.error(f"Error saving Reddit posts to CSV: {e}", exc_info=True)
        sys.exit(1)
