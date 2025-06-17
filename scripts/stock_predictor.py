import pandas as pd
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs('../logs', exist_ok=True)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('../logs/stock_predictor.log', mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

# Check for OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in .env file or environment variables.")
    sys.exit(1)

# === Load Data ===
logger.info("Loading data files...")
try:
    company_df = pd.read_csv("data/company_sentiment_normalized.csv")
    growth_df = pd.read_csv("data/predict_growth.csv")
    macro_df = pd.read_csv("data/macro_sentiment.csv")
    mapping_df = pd.read_csv("data/nasdaq_top_companies.csv")
    logger.info("All data files loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Error loading data: {e}. Please ensure all prerequisite scripts have run.")
    sys.exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred while loading data: {e}", exc_info=True)
    sys.exit(1)

# === Clean column names ===
logger.debug("Cleaning column names in mapping_df.")
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
logger.info("Filtering and selecting top companies...")
nasdaq_companies = set(mapping_df["Company"].str.strip())
df = df[df["company"].isin(nasdaq_companies)]
df = df[df["growth_score"].notnull() & (df["growth_score"] != 0)]

# === Select top companies ===
top_companies = df.sort_values(by="adjusted_growth_score", ascending=False).head(25)
if top_companies.empty:
    logger.warning("❌ No companies to analyze. Check if growth_score values are all zero or missing!")
    sys.exit(0) # Not an error, but no data to process

logger.info(f"Selected {len(top_companies)} top companies for GPT analysis.")
logger.debug(f"Top companies details:\n{top_companies.to_string()}")

# === Build prompt ===
logger.info("Building prompt for OpenAI...")
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
logger.debug(f"🧠 Prompt Sent to GPT:\n{prompt}")

# === Call OpenAI ===
logger.info("\n📡 Querying OpenAI...\n")
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
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
    logger.info(f"🔮 Prediction:\n{result}")
except Exception as e:
    logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
    sys.exit(1)

# === Save result ===
output_dir = "output"
output_file = os.path.join(output_dir, "gpt_prediction.txt")
logger.info(f"Saving GPT prediction to {output_file}...")
try:
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    logger.info(f"💾 GPT Prediction saved to → {output_file}")
except IOError as e:
    logger.error(f"Error saving prediction to file: {e}", exc_info=True)
    sys.exit(1)

logger.info("✅ Done!")
