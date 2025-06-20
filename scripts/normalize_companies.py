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
clean_names = list(name_mapping.keys()) # list of cleaned official Nasdaq names

# Create ticker_map
ticker_map = {}
for _, row in nasdaq_df.iterrows():
    ticker = row['Ticker']
    company_name = row['Company']
    if pd.notna(ticker) and pd.notna(company_name):
        ticker_map[ticker.upper()] = company_name
logging.info(f"Created ticker_map with {len(ticker_map)} entries.")

# Create common_name_map
common_name_map = {
    "google": "Alphabet Inc. (Class A)",  # Or Class C, decide on a standard, typically A for primary
    "alphabet": "Alphabet Inc. (Class A)",
    "apple": "Apple Inc.",
    "microsoft": "Microsoft",
    "amazon": "Amazon",
    "nvidia": "Nvidia",
    "meta": "Meta Platforms",
    "tesla": "Tesla, Inc."
    # Add more common names as needed, ensuring they map to the *exact* names from nasdaq_top_companies.csv
}
# Ensure common_name_map values are in official names, if not, find the closest official name
for common, official_guess in common_name_map.items():
    if official_guess not in name_mapping.values():
        logging.warning(f"Common name '{common}' maps to '{official_guess}' which is not an official Nasdaq name. Attempting to find a match.")
        # Attempt to find a match in official names
        cleaned_guess = clean_company_name(official_guess)
        if cleaned_guess in name_mapping:
            common_name_map[common] = name_mapping[cleaned_guess]
            logging.info(f"Updated common_name_map: '{common}' now maps to '{name_mapping[cleaned_guess]}'")
        else:
            # Fallback to fuzzy match for the guess against official names
            match, score = process.extractOne(cleaned_guess, clean_names)
            if score >= 90: # High threshold for this auto-correction
                common_name_map[common] = name_mapping[match]
                logging.info(f"Fuzzy matched common_name_map: '{common}' now maps to '{name_mapping[match]}' (score: {score})")
            else:
                logging.error(f"Could not confidently map common name '{common}' (guessed '{official_guess}') to an official Nasdaq name. Please check mapping.")


logging.info(f"Created common_name_map with {len(common_name_map)} entries.")


def normalize_name(name):
    if not isinstance(name, str) or not name.strip():
        return ""

    # 1. Direct Ticker Match
    # Attempt to extract potential tickers (e.g., all-caps words of 1-5 length)
    potential_tickers = re.findall(r'\b([A-Z]{1,5})\b', name)
    for ticker in potential_tickers:
        if ticker in ticker_map:
            official_name = ticker_map[ticker]
            logging.info(f"Normalized '{name}' to '{official_name}' via direct ticker map ('{ticker}')")
            return official_name

    cleaned_name_lower = clean_company_name(name) # Also converts to lowercase
    if not cleaned_name_lower:
        logging.debug(f"Could not normalize '{name}' after cleaning, result is empty.")
        return ""

    # 2. Common Name Match
    if cleaned_name_lower in common_name_map:
        official_name = common_name_map[cleaned_name_lower]
        logging.info(f"Normalized '{name}' (cleaned: '{cleaned_name_lower}') to '{official_name}' via common name map")
        return official_name

    # 3. Exact Cleaned Match (Existing Logic)
    if cleaned_name_lower in name_mapping:
        official_name = name_mapping[cleaned_name_lower]
        logging.info(f"Normalized '{name}' (cleaned: '{cleaned_name_lower}') to '{official_name}' via exact cleaned match")
        return official_name
    
    # 4. Fuzzy Match (Existing Logic as Fallback)
    try:
        # Ensure clean_names is not empty and cleaned_name_lower is not empty
        if not clean_names:
            logging.warning("clean_names list is empty, skipping fuzzy match.")
            return ""
        if not cleaned_name_lower: # Should be caught earlier, but as a safeguard
            logging.warning(f"Cleaned name for '{name}' is empty, skipping fuzzy match.")
            return ""

        match_result = process.extractOne(cleaned_name_lower, clean_names)
        if match_result:
            match, score = match_result
            # Consider adjusting the score threshold if necessary (e.g., to 70 or 75)
            if score >= 70: # Adjusted score from 60 to 70 for potentially better accuracy
                official_name = name_mapping[match]
                logging.info(f"Normalized '{name}' (cleaned: '{cleaned_name_lower}') to '{official_name}' via fuzzy match (score: {score})")
                return official_name
            else:
                logging.info(f"Fuzzy match score for '{name}' (cleaned: '{cleaned_name_lower}') was {score}, below threshold 70. No match.")
        else:
            logging.info(f"No fuzzy match found for '{name}' (cleaned: '{cleaned_name_lower}'). process.extractOne returned None.")

    except Exception as e:
        logging.warning(f"Error during fuzzy matching for '{name}' (cleaned: '{cleaned_name_lower}'): {str(e)}")
    
    logging.info(f"No normalization match found for: '{name}' (cleaned: '{cleaned_name_lower}')")
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