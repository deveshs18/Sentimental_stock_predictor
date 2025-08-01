import argparse
import logging
import sys
import os
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Load environment variables from the scripts/.env file
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path)

from app.data_ingestion import fetch
from app.sentiment_analysis import sentiment
from app.ml_prediction import predict
from app.context_fusion import fuse
from app.llm_integration import integration

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the entire pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the stock prediction pipeline.")
    parser.add_argument("--keywords", nargs="+", default=["AAPL", "GOOGL", "MSFT"], help="List of keywords to search for news.")
    parser.add_argument("--subreddits", nargs="+", default=["stocks", "investing", "wallstreetbets"], help="List of subreddits to fetch posts from.")
    args = parser.parse_args()

    logger.info("Starting the stock prediction pipeline.")

    # 1. Data Ingestion
    logger.info("--- Starting Data Ingestion ---")
    fetch.main(args.keywords, args.subreddits)
    logger.info("--- Data Ingestion Finished ---")

    # 1b. Merge Sources
    logger.info("--- Merging Data Sources ---")
    os.system(f"python scripts/merge_sources.py")
    logger.info("--- Merging Data Sources Finished ---")

    # 1c. Entity Extraction (extract company names and normalize)
    logger.info("--- Starting Entity Extraction ---")
    os.system(f"python scripts/extract_entities.py")
    logger.info("--- Entity Extraction Finished ---")

    # 2. Sentiment Analysis
    logger.info("--- Starting Sentiment Analysis ---")
    sentiment.main()
    logger.info("--- Sentiment Analysis Finished ---")

    # 3. Comprehensive Analysis (Replaces ML Prediction, Merge, and Context Fusion)
    logger.info("--- Starting Comprehensive Stock Analysis ---")
    os.system(f"python scripts/comprehensive_stock_analysis.py")
    logger.info("--- Comprehensive Stock Analysis Finished ---")

    logger.info("Stock prediction pipeline finished successfully.")

if __name__ == "__main__":
    main()
