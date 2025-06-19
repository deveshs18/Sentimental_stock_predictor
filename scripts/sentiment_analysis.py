import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from dotenv import load_dotenv
import logging
import sys

# Set up logging with Windows compatibility
os.makedirs('logs', exist_ok=True)  # Local logs directory
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Add file handler with UTF-8 encoding
file_handler = logging.FileHandler('logs/sentiment_analysis.log', mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(file_handler)

# Add console handler that can handle Unicode
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

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
            logger.warning(f"Input file '{input_csv}' is empty.")
            return
        logger.info("Running sentiment analysis...")
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
        logger.info(f"[SUCCESS] Saved sentiment results → {output_csv}")
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Set console output to UTF-8 on Windows
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    input_file = "data/merged_sentiment_output_with_companies.csv"
    output_file = "data/merged_sentiment_output_with_companies_and_sentiment.csv"

    logger.info(f"Starting sentiment analysis for input: {input_file}, output: {output_file}")
    try:
        run_sentiment_analysis(input_file, output_file)
        logger.info("Sentiment analysis script finished successfully.")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}. Please ensure previous steps ran successfully.")
    except Exception as e:
        logger.error(f"An error occurred in the main execution block: {e}", exc_info=True)