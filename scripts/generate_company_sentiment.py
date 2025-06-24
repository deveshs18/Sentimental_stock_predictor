import pandas as pd
import os
import sys
import logging

# Add parent directory to sys.path to allow imports from 'utils'
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

from utils.normalization_utils import normalize_company_name, _load_normalization_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define input and output file paths using os.path.join for compatibility
input_file = os.path.join(parent_dir, "data", "merged_sentiment_output_with_companies_and_sentiment.csv")
output_file = os.path.join(parent_dir, "data", "company_sentiment_normalized.csv")

try:
    # Load file with company column
    df = pd.read_csv(input_file)
    logger.info(f"Successfully loaded input file: {input_file}")
    logger.info(f"Columns in file: {df.columns.tolist()}")
    logger.info(f"Sample of data:\n{df.head()}")

    # Ensure normalization data is loaded
    _load_normalization_data()
    logger.info("Normalization data loaded from utils.")

    # Handle cases where 'company' column might be missing or all NaN
    if 'company' not in df.columns:
        logger.error("'company' column not found in input CSV. Cannot proceed.")
        sys.exit(1)

    # Normalize the 'company' column
    # Convert to string first to handle potential float NaNs or other types
    df['normalized_company'] = df['company'].astype(str).apply(normalize_company_name)
    logger.info("Applied company name normalization.")
    logger.info(f"Sample of normalized names:\n{df[['company', 'normalized_company']].head(10)}")


    # Filter only rows with successfully normalized company names
    original_row_count = len(df)
    df = df[df["normalized_company"].notnull() & (df["normalized_company"] != "")]
    logger.info(f"Filtered out {original_row_count - len(df)} rows with no valid normalized company name.")

    if df.empty:
        logger.warning("DataFrame is empty after filtering for normalized company names. No sentiment summary will be generated.")
        # Create an empty summary df with expected columns to prevent downstream errors if the file is expected
        summary = pd.DataFrame(columns=['company', 'negative', 'neutral', 'positive'])
    elif 'sentiment' not in df.columns:
        logger.error("'sentiment' column not found in input CSV after filtering. Cannot generate sentiment summary.")
        summary = pd.DataFrame(columns=['company', 'negative', 'neutral', 'positive']) # Output empty structure
    else:
        # Count sentiment categories per normalized company
        logger.info("Aggregating sentiment counts per normalized company...")
        summary = df.groupby("normalized_company")["sentiment"].value_counts().unstack(fill_value=0).reset_index()

        # Rename columns for clarity
        summary.columns.name = None
        # Ensure standard column names, even if some sentiments don't appear for any company
        for sentiment_type in ["positive", "neutral", "negative"]:
            if sentiment_type not in summary.columns:
                summary[sentiment_type] = 0

        # Rename the 'normalized_company' column to 'company' for the output file
        summary = summary.rename(columns={"normalized_company": "company",
                                          "positive": "positive",
                                          "neutral": "neutral",
                                          "negative": "negative"})
        summary = summary.fillna(0) # Fill any remaining NaNs just in case

    # Ensure all required sentiment columns exist in the final summary, even if empty
    for col in ["positive", "neutral", "negative"]:
        if col not in summary.columns:
            summary[col] = 0

    # Select and order columns for the output CSV
    final_columns = ['company', 'negative', 'neutral', 'positive']
    summary = summary[final_columns]


    # Save the sentiment summary
    summary.to_csv(output_file, index=False)
    logger.info(f"✅ Saved sentiment summary to → {output_file}")
    logger.info(f"Final summary sample:\n{summary.head()}")

except FileNotFoundError:
    logger.error(f"Error: Input file not found at {input_file}. Please ensure prerequisite scripts have run.")
except Exception as e:
    logger.error(f"An unexpected error occurred: {e}", exc_info=True)

print("✅ generate_company_sentiment.py processing complete.") # Keep simple print for run_all.py
