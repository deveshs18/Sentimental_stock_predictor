import json
import pandas as pd
from collections import defaultdict
import os

# Load data
df = pd.read_csv("data/merged_sentiment_output_with_companies.csv")
with open("data/macro_themes.json", "r") as f:
    themes = json.load(f)

# Invert macro_theme -> keyword mapping to keyword -> theme(s)
keyword_to_themes = defaultdict(list)
for theme, keywords in themes.items():
    for kw in keywords:
        keyword_to_themes[kw.lower()].append(theme)

# Initialize theme sentiment counters
theme_sentiment = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})

# Analyze each row
for _, row in df.iterrows():
    sentiment = row.get("sentiment", "")
    headline = str(row.get("headline", "")).lower()
    
    for word in headline.split():
        themes_matched = keyword_to_themes.get(word, [])
        for theme in themes_matched:
            if sentiment in theme_sentiment[theme]:
                theme_sentiment[theme][sentiment] += 1

# Build output
output = []
for theme, counts in theme_sentiment.items():
    total = sum(counts.values())
    score = (counts["positive"] - counts["negative"]) / total if total > 0 else 0
    output.append({
        "theme": theme,
        "positive": counts["positive"],
        "neutral": counts["neutral"],
        "negative": counts["negative"],
        "macro_sentiment_score": round(score, 3)
    })

# Save output
df_out = pd.DataFrame(output)
os.makedirs("data", exist_ok=True)
df_out.to_csv("data/macro_sentiment.csv", index=False)
print("✅ Saved macro sentiment to → data/macro_sentiment.csv")
