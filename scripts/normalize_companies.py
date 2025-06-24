import pandas as pd
import logging
import os
import sys

# Add parent directory to sys.path to allow imports from 'utils'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.normalization_utils import normalize_company_name, _load_normalization_data

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/normalize_companies.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Input: raw NER output
input_file = os.path.join(parent_dir, "data", "merged_sentiment_output_with_companies.csv")
# nasdaq_file no longer needed here directly, it's used by normalization_utils
output_file = os.path.join(parent_dir, "data", "merged_sentiment_output_with_companies_normalized.csv")
mapping_file = os.path.join(parent_dir, "data", "company_name_mapping.csv") # This is an output review file

# Load files
try:
    df = pd.read_csv(input_file)
    logger.info(f"Loaded input file: {input_file}. Found {len(df)} records.")
    # Ensure normalization data (like NASDAQ list) is loaded in the utility module
    _load_normalization_data()
except FileNotFoundError as e:
    logger.error(f"Error: Input file not found: {e.filename}. Please check the path: {input_file}")
    sys.exit(1) # Exit if critical input file is missing
except Exception as e:
    logger.error(f"Error loading input file or normalization data: {str(e)}")
    raise


# Apply normalization using the imported function
logger.info("Starting company name normalization using normalization_utils...")
df["normalized_company"] = df["company"].astype(str).apply(normalize_company_name)

# Save mapping for review
mapping_df = df[["company", "normalized_company"]].drop_duplicates()
mapping_df.to_csv(mapping_file, index=False)
logging.info(f"Saved name mapping to {mapping_file}")

# Replace 'company' with the normalized version
df = df.drop(columns=["company"]).rename(columns={"normalized_company": "company"})

# Only keep rows with valid company
initial_count = len(df)
df = df[df["company"] != ""]
filtered_count = len(df)
logging.info(f"Filtered out {initial_count - filtered_count} records without valid company matches")

# Save results
df.to_csv(output_file, index=False)
logging.info(f"✅ Normalized data saved to: {output_file}")
print(f"\n✅ Normalization complete. Kept {filtered_count} of {initial_count} records.")
print(f"   - Output file: {output_file}")
print(f"   - Mapping file: {mapping_file}")