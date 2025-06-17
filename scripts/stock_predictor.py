import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv   # <-- ADD THIS
load_dotenv()                   # <-- AND THIS

# === Load Data ===
company_df = pd.read_csv("data/company_sentiment_normalized.csv")
growth_df = pd.read_csv("data/predict_growth.csv")
macro_df = pd.read_csv("data/macro_sentiment.csv")
mapping_df = pd.read_csv("data/nasdaq_top_companies.csv")

# === Clean column names ===
mapping_df.columns = mapping_df.columns.str.replace("\xa0", " ")

# === Macro mapping ===
theme_map = dict(zip(mapping_df["Company"], mapping_df["GICS Sector"]))
macro_map = dict(zip(macro_df["theme"], macro_df["macro_sentiment_score"]))

# === Merge sentiment and growth ===
df = pd.merge(company_df, growth_df, on="company", how="inner")
df["theme"] = df["company"].map(theme_map)
df["macro_sentiment_score"] = df["theme"].map(macro_map).fillna(0)

# === Adjust growth score ===
df["adjusted_growth_score"] = df["growth_score"].fillna(0) + df["macro_sentiment_score"] * 5

# === Filter only top NASDAQ companies & remove NaN/zero scores ===
nasdaq_companies = set(mapping_df["Company"].str.strip())
df = df[df["company"].isin(nasdaq_companies)]
df = df[df["growth_score"].notnull() & (df["growth_score"] != 0)]

# === Select top companies ===
top_companies = df.sort_values(by="adjusted_growth_score", ascending=False).head(25)
if top_companies.empty:
    print("❌ No companies to analyze. Check if growth_score values are all zero or missing!")
    exit()



# === Build prompt ===
prompt = "You are a stock analyst assistant. Below is a list of NASDAQ companies with sentiment, growth scores, and macro sector sentiment from news and Reddit in the past day:\n\n"
for _, row in top_companies.iterrows():
    line = f"- {row['company']}: Positive={row['positive']}, Neutral={row['neutral']}, Negative={row['negative']}, GrowthScore={row['growth_score']:.2f}, MacroSentiment={row['macro_sentiment_score']:.2f}, Sector={row['theme']}\n"
    prompt += line

prompt += """
Instructions:
1. Identify the top 3 companies likely to grow tomorrow.
2. Estimate the percentage increase in stock price for each.
3. Justify using GrowthScore, Sentiment, and MacroSector values.
Output as a ranked list with estimated gains and reasoning.
"""

# === Call OpenAI ===
print("🧠 Prompt Sent to GPT:\n", prompt)
print("\n📡 Querying OpenAI...\n")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
    messages=[
        {"role": "system", "content": "You are a stock analyst assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=600
)

result = response.choices[0].message.content
print("🔮 Prediction:\n", result)

# === Save result ===
os.makedirs("output", exist_ok=True)
with open("output/gpt_prediction.txt", "w", encoding="utf-8") as f:
    f.write(result)

print("💾 GPT Prediction saved to → output/gpt_prediction.txt")
print("✅ Done!")
