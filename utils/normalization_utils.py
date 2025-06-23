import pandas as pd
from thefuzz import process
import re
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

# Constants
NASDAQS_COMPANIES_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "nasdaq_top_companies.csv")

# Global cache for loaded data to avoid reloading CSVs multiple times
_loaded_data = {
    "nasdaq_df": None,
    "name_mapping": None,
    "ticker_map": None,
    "common_name_map": None,
    "clean_official_names": None,
}

def clean_company_name(name):
    """
    Cleans a company name by removing common suffixes, special characters,
    and converting to lowercase.
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    # Remove common suffixes, special chars, and extra spaces
    name = re.sub(r'[^\w\s]', ' ', name.lower()) # Keep spaces
    name = re.sub(r'\b(inc|llc|ltd|corp|corporation|company|co|plc|holdings|technologies|international|group)\b', '', name, flags=re.IGNORECASE)
    return ' '.join(name.split()).strip()

def _load_normalization_data():
    """
    Loads data from nasdaq_top_companies.csv and prepares mappings.
    This function is intended for internal use and populates the _loaded_data cache.
    """
    if _loaded_data["nasdaq_df"] is not None:
        logger.debug("Normalization data already loaded. Skipping reload.")
        return

    try:
        nasdaq_df = pd.read_csv(NASDAQS_COMPANIES_CSV_PATH)
        logger.info(f"Loaded NASDAQ companies from {NASDAQS_COMPANIES_CSV_PATH}. Found {len(nasdaq_df)} companies.")
        _loaded_data["nasdaq_df"] = nasdaq_df
    except FileNotFoundError:
        logger.error(f"NASDAQ companies file not found at: {NASDAQS_COMPANIES_CSV_PATH}")
        # Create empty structures if file not found to prevent downstream errors on first load
        _loaded_data["nasdaq_df"] = pd.DataFrame(columns=['Company', 'Ticker'])
        _loaded_data["name_mapping"] = {}
        _loaded_data["ticker_map"] = {}
        _loaded_data["common_name_map"] = {}
        _loaded_data["clean_official_names"] = []
        return # Exit if the crucial file is missing
    except Exception as e:
        logger.error(f"Error loading NASDAQ companies file: {str(e)}")
        _loaded_data["nasdaq_df"] = pd.DataFrame(columns=['Company', 'Ticker']) # Fallback
        _loaded_data["name_mapping"] = {}
        _loaded_data["ticker_map"] = {}
        _loaded_data["common_name_map"] = {}
        _loaded_data["clean_official_names"] = []
        return

    # Create cleaned versions for matching
    nasdaq_df['clean_name'] = nasdaq_df['Company'].fillna('').apply(clean_company_name)
    _loaded_data["name_mapping"] = dict(zip(nasdaq_df['clean_name'], nasdaq_df['Company']))
    _loaded_data["clean_official_names"] = list(_loaded_data["name_mapping"].keys())

    # Create ticker_map
    ticker_map = {}
    for _, row in nasdaq_df.iterrows():
        ticker = row['Ticker']
        company_name = row['Company']
        if pd.notna(ticker) and pd.notna(company_name):
            ticker_map[ticker.upper()] = company_name
    _loaded_data["ticker_map"] = ticker_map
    logger.info(f"Created ticker_map with {len(ticker_map)} entries.")

    # Create common_name_map
    # Ensure these map to the *exact* names from nasdaq_top_companies.csv
    common_name_map = {
        "google": "Alphabet Inc. (Class A)",
        "alphabet": "Alphabet Inc. (Class A)",
        "apple": "Apple Inc.",
        "microsoft": "Microsoft", # Assuming 'Microsoft' is the exact name in nasdaq_top_companies.csv
        "amazon": "Amazon",       # Assuming 'Amazon' is the exact name
        "nvidia": "Nvidia",
        "meta": "Meta Platforms",
        "tesla": "Tesla, Inc."
        # Add more as needed
    }

    # Validate and correct common_name_map values
    # This ensures that the mapped official names are actually present in nasdaq_top_companies
    corrected_common_name_map = {}
    for common, official_guess in common_name_map.items():
        # Check if the guessed official name is directly in the list of official company names
        if official_guess in _loaded_data["name_mapping"].values():
            corrected_common_name_map[common] = official_guess
        else:
            # If not, try to find it by its cleaned version
            cleaned_official_guess = clean_company_name(official_guess)
            if cleaned_official_guess in _loaded_data["name_mapping"]:
                corrected_common_name_map[common] = _loaded_data["name_mapping"][cleaned_official_guess]
                logger.info(f"Common name '{common}' mapped to '{official_guess}' which was corrected to '{corrected_common_name_map[common]}' via clean name matching.")
            else:
                # Fallback: try fuzzy matching the GUESS against CLEANED official names
                match, score = process.extractOne(cleaned_official_guess, _loaded_data["clean_official_names"])
                if score >= 90: # High threshold for auto-correction
                    corrected_official_name = _loaded_data["name_mapping"][match]
                    corrected_common_name_map[common] = corrected_official_name
                    logger.warning(f"Common name '{common}' mapped to '{official_guess}', "
                                   f"fuzzy matched and corrected to '{corrected_official_name}' (score: {score}).")
                else:
                    logger.error(f"Could not confidently map common name '{common}' (guessed '{official_guess}') to an official NASDAQ name. "
                                 f"Best fuzzy match '{match}' (score: {score}) was below threshold 90. This mapping will be skipped.")

    _loaded_data["common_name_map"] = corrected_common_name_map
    logger.info(f"Created common_name_map with {len(_loaded_data['common_name_map'])} entries after validation.")


def normalize_company_name(name_to_normalize):
    """
    Normalizes a given company name or ticker to its official company name
    using various matching strategies.
    """
    _load_normalization_data() # Ensure data is loaded

    if not isinstance(name_to_normalize, str) or not name_to_normalize.strip():
        return ""

    # 1. Direct Ticker Match (if name_to_normalize is a potential ticker)
    # Check if the input itself is a ticker (common for user queries)
    potential_ticker_input = name_to_normalize.upper()
    if potential_ticker_input in _loaded_data["ticker_map"]:
        official_name = _loaded_data["ticker_map"][potential_ticker_input]
        logger.debug(f"Normalized '{name_to_normalize}' to '{official_name}' via direct ticker input map.")
        return official_name

    # Also check for tickers within a longer string (original logic)
    potential_tickers_in_string = re.findall(r'\b([A-Z]{1,5})\b', name_to_normalize)
    for ticker in potential_tickers_in_string:
        if ticker in _loaded_data["ticker_map"]:
            official_name = _loaded_data["ticker_map"][ticker]
            logger.debug(f"Normalized '{name_to_normalize}' to '{official_name}' via ticker '{ticker}' found in string.")
            return official_name

    cleaned_name_lower = clean_company_name(name_to_normalize)
    if not cleaned_name_lower:
        logger.debug(f"Could not normalize '{name_to_normalize}' after cleaning, result is empty.")
        return ""

    # 2. Common Name Match
    if cleaned_name_lower in _loaded_data["common_name_map"]:
        official_name = _loaded_data["common_name_map"][cleaned_name_lower]
        logger.debug(f"Normalized '{name_to_normalize}' (cleaned: '{cleaned_name_lower}') to '{official_name}' via common name map.")
        return official_name

    # 3. Exact Cleaned Match
    if cleaned_name_lower in _loaded_data["name_mapping"]:
        official_name = _loaded_data["name_mapping"][cleaned_name_lower]
        logger.debug(f"Normalized '{name_to_normalize}' (cleaned: '{cleaned_name_lower}') to '{official_name}' via exact cleaned match.")
        return official_name

    # 4. Fuzzy Match
    try:
        if not _loaded_data["clean_official_names"]:
            logger.warning("Clean official names list is empty, skipping fuzzy match.")
            return ""

        match_result = process.extractOne(cleaned_name_lower, _loaded_data["clean_official_names"])
        if match_result:
            match, score = match_result
            if score >= 75: # Adjusted threshold, can be tuned
                official_name = _loaded_data["name_mapping"][match]
                logger.debug(f"Normalized '{name_to_normalize}' (cleaned: '{cleaned_name_lower}') to '{official_name}' via fuzzy match (score: {score}).")
                return official_name
            else:
                logger.debug(f"Fuzzy match score for '{name_to_normalize}' (cleaned: '{cleaned_name_lower}') was {score}, below threshold 75.")
        else:
            logger.debug(f"No fuzzy match found for '{name_to_normalize}' (cleaned: '{cleaned_name_lower}').")

    except Exception as e:
        logger.warning(f"Error during fuzzy matching for '{name_to_normalize}' (cleaned: '{cleaned_name_lower}'): {str(e)}")

    logger.info(f"No normalization match found for: '{name_to_normalize}' (cleaned: '{cleaned_name_lower}'). Returning original as a last resort if it's a proper noun, else empty.")
    # Fallback: if it looks like a proper name (e.g., starts with capital or is all caps) return it, else empty
    if name_to_normalize.isupper() or (name_to_normalize[0].isupper() if name_to_normalize else False):
        # Check if the original name_to_normalize is an official name already
        if name_to_normalize in _loaded_data["name_mapping"].values():
             return name_to_normalize

    # If user query was "google stco ktowmorrow", this function would be called with "google stco ktowmorrow".
    # It would fail to find a direct match.
    # A more advanced query parser in stock_predictor.py might split this into "google", "stco", "ktowmorrow"
    # and call normalize_company_name on each part. "google" would then be normalized.
    logger.info(f"Final fallback: No normalization found for '{name_to_normalize}'. Returning empty string.")
    return ""


def get_ticker_for_company(official_company_name):
    """
    Returns the ticker for a given official company name.
    """
    _load_normalization_data()
    for ticker, company_name in _loaded_data["ticker_map"].items():
        if company_name == official_company_name:
            return ticker
    return None

def get_company_for_ticker(ticker_symbol):
    """
    Returns the official company name for a given ticker symbol.
    """
    _load_normalization_data()
    return _loaded_data["ticker_map"].get(ticker_symbol.upper())


if __name__ == '__main__':
    # Basic test cases
    logging.basicConfig(level=logging.DEBUG) # Enable debug for testing this module
    _load_normalization_data() # Explicitly load for testing

    test_names = [
        "Google", "google", "Alphabet Inc. (Class A)", "GOOGL", "GOOG",
        "Apple", "AAPL", "Apple Inc.",
        "Microsoft Corp", "MSFT",
        "Amazon.com Inc.", "AMZN",
        "NVIDIA Corporation", "NVDA",
        "Tesla", "TSLA",
        "NonExistentCompany", "XYZTICKER",
        "Adobe", "ADBE",
        "Netflix inc", "NFLX",
        "Meta Platforms",
        "google stco ktowmorrow" # Test the problematic query
    ]
    for name in test_names:
        normalized = normalize_company_name(name)
        ticker = get_ticker_for_company(normalized) if normalized else "N/A"
        print(f"Original: '{name}' -> Normalized: '{normalized}' (Ticker: {ticker})")

    print("\nTesting direct ticker lookup:")
    print(f"Ticker 'AAPL': Company is '{get_company_for_ticker('AAPL')}'")
    print(f"Ticker 'GOOGL': Company is '{get_company_for_ticker('GOOGL')}'")
    print(f"Ticker 'XYZ': Company is '{get_company_for_ticker('XYZ')}'")

    print("\nTesting common name map directly (for debugging):")
    if _loaded_data["common_name_map"]:
        for common, official in _loaded_data["common_name_map"].items():
            print(f"Common: '{common}' -> Official in map: '{official}'")
    else:
        print("Common name map is empty or not loaded.")

    print(f"\nIs 'Microsoft' an official name in mapping? {'Microsoft' in _loaded_data['name_mapping'].values()}")
    print(f"Is 'Amazon' an official name in mapping? {'Amazon' in _loaded_data['name_mapping'].values()}")

    # Test the problematic query again after ensuring data is loaded
    problem_query = "google stco ktowmorrow"
    # The current normalize_company_name expects a single company name.
    # A query parser would be needed to split this.
    # For now, let's test its components:
    print(f"\nTesting components of problematic query:")
    print(f"Original: 'google' -> Normalized: '{normalize_company_name('google')}'")
    print(f"Original: 'stco' -> Normalized: '{normalize_company_name('stco')}'") # Likely no match
    print(f"Original: 'ktowmorrow' -> Normalized: '{normalize_company_name('ktowmorrow')}'") # Likely no match

```

