import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = ["positive", "neutral", "negative"]

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1).squeeze().numpy()
    pred_idx = probs.argmax()
    return labels[pred_idx], float(probs[pred_idx])

def run_sentiment_analysis(input_csv, output_csv):
    try:
        df = pd.read_csv(input_csv)
        if df.empty:
            logging.warning(f"Input file '{input_csv}' is empty.")
            return
        logging.info("Running sentiment analysis...")
        df = df.dropna(subset=["headline", "text"])
        
        # Normalize company names (OPTIONAL)
        if "company" in df.columns:
            df["company"] = df["company"].str.replace("Inc.", "", regex=False).str.strip()

        results = []
        for _, row in df.iterrows():
            combined_text = f"{row['headline']} {row['text']}"
            sentiment, confidence = classify_sentiment(combined_text)
            results.append({
                "timestamp": row.get("timestamp", ""),
                "source": row.get("source", ""),
                "headline": row.get("headline", ""),
                "text": row.get("text", ""),
                "sentiment": sentiment,
                "confidence": confidence,
                "url": row.get("url", ""),
                "company": row.get("company", "")
            })

        result_df = pd.DataFrame(results)
        os.makedirs("data", exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        logging.info(f"✅ Saved sentiment results → {output_csv}")
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        raise

if __name__ == "__main__":
    try:
        input_df = pd.read_csv("data/merged_sentiment_output_with_companies.csv")
        if input_df.empty:
            logging.warning("Input file 'merged_sentiment_output_with_companies.csv' is empty.")
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        raise

    run_sentiment_analysis(
        "data/merged_sentiment_output_with_companies.csv",
        "data/merged_sentiment_output_with_companies_and_sentiment.csv"
    )
