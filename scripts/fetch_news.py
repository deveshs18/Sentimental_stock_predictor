import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load NewsAPI Key
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

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

    if response.status_code == 200 and data.get("status") == "ok":
        articles = data["articles"]
        if not articles:
            print("⚠️ No trending news found.")
            return

        df = pd.DataFrame(articles)
        df = df[["publishedAt", "title", "description", "url", "source"]]
        os.makedirs("data", exist_ok=True)
        file_path = "data/news_combined.csv"
        df.to_csv(file_path, index=False)
        print(f"✅ Saved {len(df)} trending news articles → {file_path}")
    else:
        print(f"❌ Error fetching news: {data.get('message')}")

if __name__ == "__main__":
    fetch_trending_news()
