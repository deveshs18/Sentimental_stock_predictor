import praw
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Reddit API with PRAW
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# 🔍 Fetch top N posts from each subreddit (like NewsAPI: general trends)
def fetch_latest_posts(subreddit_name="stocks", limit=50):
    posts = []
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.hot(limit=limit):
        posts.append({
            "title": post.title,
            "text": post.selftext,
            "created_utc": post.created_utc,
            "url": post.url,
            "subreddit": subreddit_name
        })
    return posts

if __name__ == "__main__":
    subreddits = ["stocks", "investing", "wallstreetbets"]
    all_posts = []

    for sub in subreddits:
        print(f"🔍 Scraping top posts from r/{sub}...")
        data = fetch_latest_posts(subreddit_name=sub, limit=50)
        all_posts.extend(data)
        print(f"✅ Retrieved {len(data)} posts from r/{sub}")

    # Create a DataFrame and save
    df = pd.DataFrame(all_posts)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/reddit_posts.csv", index=False)
    print("📄 Saved all Reddit posts → data/reddit_posts.csv")
