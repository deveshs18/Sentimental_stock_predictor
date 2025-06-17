import pandas as pd
from thefuzz import process
import re
import logging
import os

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

# Input: raw NER output
input_file = "data/merged_sentiment_output_with_companies.csv"
nasdaq_file = "data/nasdaq_top_companies.csv"
output_file = "data/merged_sentiment_output_with_companies_normalized.csv"
mapping_file = "data/company_name_mapping.csv"

# Load files
try:
    df = pd.read_csv(input_file)
    nasdaq_df = pd.read_csv(nasdaq_file)
    logging.info(f"Loaded input files. Found {len(df)} records and {len(nasdaq_df)} NASDAQ companies.")
except Exception as e:
    logging.error(f"Error loading input files: {str(e)}")
    raise

# Preprocess company names
def clean_company_name(name):
    if not isinstance(name, str) or not name.strip():
        return ""
    # Remove common suffixes, special chars, and extra spaces
    name = re.sub(r'[^\w\s]', ' ', name.lower())
    name = re.sub(r'\b(inc|llc|ltd|corp|corporation|company|co|plc|holdings|technologies|international|group)\b', '', name)
    return ' '.join(name.split()).strip()

# Create cleaned versions for matching
nasdaq_df['clean_name'] = nasdaq_df['Company'].fillna('').apply(clean_company_name)
name_mapping = dict(zip(nasdaq_df['clean_name'], nasdaq_df['Company']))
clean_names = list(name_mapping.keys())

def normalize_name(name):
    if not isinstance(name, str) or not name.strip():
        return ""
    
    cleaned = clean_company_name(name)
    if not cleaned:
        return ""
    
    # Try exact match first
    if cleaned in name_mapping:
        return name_mapping[cleaned]
    
    # Try fuzzy match
    try:
        match, score = process.extractOne(cleaned, clean_names)
        if score >= 60:  # Lowered threshold from 65
            logging.info(f"Matched '{name}' to '{name_mapping[match]}' with score {score}")
            return name_mapping[match]
    except Exception as e:
        logging.warning(f"Error matching '{name}': {str(e)}")
    
    logging.debug(f"No match found for: {name}")
    return ""

# Apply normalization
logging.info("Starting company name normalization...")
df["normalized_company"] = df["company"].astype(str).apply(normalize_name)

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