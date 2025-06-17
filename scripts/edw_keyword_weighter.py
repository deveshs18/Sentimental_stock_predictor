import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone
import math
from financial_keywords import extract_financial_keywords


def time_decay_weight(timestamp, reference_time):
    """Applies exponential decay to older timestamps."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    delta_seconds = (reference_time - timestamp).total_seconds()
    decay_rate = 0.00001  # Can tune for sensitivity
    return math.exp(-decay_rate * delta_seconds)

# Load merged file
df = pd.read_csv("data/merged_sentiment_output.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Extract important keywords
keywords_to_use = extract_financial_keywords(df["headline"], top_n=100)

weights = defaultdict(float)
now = datetime.now(timezone.utc)

for _, row in df.iterrows():
    text = str(row["headline"]) + " " + str(row.get("source", ""))
    keywords = text.lower().split()

    timestamp = row["timestamp"]
    decay_weight = time_decay_weight(timestamp, now)

    for keyword in keywords:
        if keyword in keywords_to_use:
            weights[keyword] += decay_weight

# Sort and print top 15 weighted keywords
top_weighted = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:15]

print("⏳ Top Keywords by Exponential Decay Weighting:")
for word, weight in top_weighted:
    print(f"🔹 {word}: {weight:.3f}")
# Save EDW results
edw_df = pd.DataFrame(top_weighted, columns=["keyword", "weight"])
edw_df.to_csv("data/edw_keywords.csv", index=False)
print("✅ EDW keywords saved → data/edw_keywords.csv")
