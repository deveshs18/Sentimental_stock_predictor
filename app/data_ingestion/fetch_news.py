import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests.exceptions

# Load NewsAPI Key
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

if not API_KEY:
    print("Error: NEWS_API_KEY not found in .env file or environment variables.", file=sys.stderr)
    sys.exit(1)

# Setup
base_url = "https://newsapi.org/v2/everything"
from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
to_date = datetime.now().strftime('%Y-%m-%d')

# General financial query (catch all trending companies)
query = "stock OR earnings OR merger OR acquisition OR ipo OR partnership OR launch OR chip OR ai OR company"

def fetch_trending_news():
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": API_KEY
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if response.status_code != 200 or data.get("status") != "ok":
        error_message = data.get('message', 'Unknown error')
        print(f"❌ Error fetching news: Status Code {response.status_code}, Message: {error_message}", file=sys.stderr)
        raise requests.exceptions.RequestException(f"API request failed: Status Code {response.status_code}, Message: {error_message}")

    articles = data.get("articles", [])

    if not articles:
        print("⚠️ No trending news found. Saving empty CSV with headers.")

    df = pd.DataFrame(articles)
    if not df.empty:
        df = df[["publishedAt", "title", "description", "url", "source"]]
    else:
        # Create empty DataFrame with headers if no articles
        df = pd.DataFrame(columns=["publishedAt", "title", "description", "url", "source"])

    os.makedirs("data", exist_ok=True)
    file_path = "data/news_combined.csv"
    df.to_csv(file_path, index=False)

    if not articles:
        print(f"✅ Saved empty news file with headers → {file_path}")
    else:
        print(f"✅ Saved {len(df)} trending news articles → {file_path}")

if __name__ == "__main__":
    fetch_trending_news()
