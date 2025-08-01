import os
import pandas as pd
import logging
from datetime import datetime, timezone
import math

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('logs/context_fusion.log', mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

def time_decay_weight(timestamp, reference_time=None, decay_rate=0.00001):
    """
    Applies exponential decay to older timestamps.
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)
    if pd.isna(timestamp):
        return 0.0
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    delta_seconds = (reference_time - timestamp).total_seconds()
    return math.exp(-decay_rate * delta_seconds)

def main():
    """
    Main function to fuse the sentiment and ML signals.
    """
    # Load data
    try:
        # Go up two levels from fuse.py to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        company_sentiment = pd.read_csv(os.path.join(project_root, "data", "processed", "company_sentiment.csv"))
    except FileNotFoundError:
        logger.error("company_sentiment.csv not found. Please run the sentiment analysis pipeline first.")
        company_sentiment = pd.DataFrame()

    try:
        ml_predictions = pd.read_csv(os.path.join(project_root, "data", "ml_predictions.csv"))
    except FileNotFoundError:
        logger.error("ml_predictions.csv not found. Please run the ML prediction pipeline first.")
        ml_predictions = pd.DataFrame()

    if company_sentiment.empty or ml_predictions.empty:
        logger.error("No data to process. Exiting.")
        return

    # Merge data
    fused_df = pd.merge(company_sentiment, ml_predictions, on="company", how="left")

    # Calculate final score
    fused_df['ml_prediction_score'] = fused_df['ml_prediction'].apply(lambda x: 1 if x == 'up' else -1)
    fused_df['final_score'] = fused_df['normalized_sentiment'] + fused_df['ml_prediction_score']

    # Sort by final score
    fused_df = fused_df.sort_values(by="final_score", ascending=False)

    # Save results
    fused_df.to_csv(os.path.join(project_root, "data", "fused_scores.csv"), index=False)
    logger.info("Successfully saved fused scores.")

if __name__ == "__main__":
    main()
