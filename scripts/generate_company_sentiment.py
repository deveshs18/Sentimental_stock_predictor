import pandas as pd

# Load file with company column
df = pd.read_csv("data/merged_sentiment_output_with_companies_and_sentiment.csv")

print("Columns in file:", df.columns.tolist())
print(df.head())

# Filter only rows with company
df = df[df["company"].notnull() & (df["company"] != "")]

# Count sentiment categories per company
summary = df.groupby("company")["sentiment"].value_counts().unstack(fill_value=0).reset_index()

# Rename columns for clarity
summary.columns.name = None
summary = summary.rename(columns={"positive": "positive", "neutral": "neutral", "negative": "negative"})
summary = summary.fillna(0)

# Ensure all sentiment columns exist
for col in ["positive", "neutral", "negative"]:
    if col not in summary.columns:
        summary[col] = 0

# Save the sentiment summary
summary.to_csv("data/company_sentiment_normalized.csv", index=False)
print("✅ Saved sentiment summary to → data/company_sentiment_normalized.csv")
