import pandas as pd
import logging

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('logs/sector_analysis.log', mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

def main():
    """
    Performs sector-level sentiment analysis.
    """
    try:
        sentiment_df = pd.read_csv("data/company_sentiment.csv")
    except FileNotFoundError:
        logger.error("company_sentiment.csv not found. Please run the sentiment analysis first.")
        return

    try:
        sector_df = pd.read_csv("data/company_sectors.csv")
    except FileNotFoundError:
        logger.error("company_sectors.csv not found. Please create this file with company-to-sector mappings.")
        return

    # Merge sentiment data with sector data
    merged_df = pd.merge(sentiment_df, sector_df, on="company")

    # Calculate average sentiment per sector
    sector_sentiment = merged_df.groupby("sector")["normalized_sentiment"].mean().reset_index()

    # Save results
    sector_sentiment.to_csv("data/sector_sentiment.csv", index=False)
    logger.info("Successfully saved sector sentiment data.")

if __name__ == "__main__":
    main()
