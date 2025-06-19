import pandas as pd
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
from lstm_stock_predictor import get_or_train_lstm_for_stock
import logging

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs('output/logs', exist_ok=True) # Changed path

# Define log format
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers
logger.handlers = []

# File handler with UTF-8 encoding
file_handler = logging.FileHandler(
    'output/logs/stock_predictor.log', 
    mode='w', 
    encoding='utf-8',
    errors='replace'
)
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

# Stream handler with UTF-8 support
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

import re # Import re for simple_clean_name

# Check for OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY: # Temporarily allowing to proceed without key for prompt verification
#     logger.error("OPENAI_API_KEY not found in .env file or environment variables.")
#     sys.exit(1)

def get_qualitative_market_sentiment():
    """
    Reads market_sentiment from ../data/predict_growth.csv and returns a qualitative description.
    This function is intended to be called from scripts within the 'scripts' directory.
    """
    # Construct path relative to this script's location (scripts/)
    current_dir = os.path.dirname(__file__)
    predict_growth_file = os.path.join(current_dir, '..', 'data', 'predict_growth.csv')

    try:
        df = pd.read_csv(predict_growth_file)
        if 'market_sentiment' not in df.columns:
            logger.warning(f"'market_sentiment' column not found in {predict_growth_file}.")
            return "Data not available"
        if df.empty:
            logger.warning(f"{predict_growth_file} is empty.")
            return "Data not available"

        # Assuming market_sentiment is the same for all rows, take from the first row.
        market_sentiment_value = df['market_sentiment'].iloc[0]

        if market_sentiment_value > 0.1:
            qualitative_sentiment = "Positive"
        elif market_sentiment_value < -0.1:
            qualitative_sentiment = "Negative"
        else:
            qualitative_sentiment = "Neutral"
        logger.info(f"Qualitative market sentiment from stock_predictor: {qualitative_sentiment} (Raw: {market_sentiment_value})")
        return qualitative_sentiment
    except FileNotFoundError:
        logger.error(f"Market sentiment data file not found at {predict_growth_file}.")
        return "Data not available"
    except Exception as e:
        logger.error(f"Error reading or processing market sentiment data in stock_predictor: {e}", exc_info=True)
        return "Data not available"

# Helper function for formatting values in the prompt
def format_value_for_prompt(value):
    if isinstance(value, str):
        return value  # Return string indicators like "N/A (No Batch Data)" directly
    if pd.notna(value):
        try:
            return f"{float(value):.2f}" # Ensure it's a float before formatting
        except ValueError:
            return str(value) # If it can't be float (e.g. already a different string), return as is
    return "N/A" # For pd.NA or None

def generate_dynamic_prompt(user_query, top_companies_df, overall_market_sentiment_string):
    """
    Generates a dynamic prompt for the LLM based on user query, company data, and overall market sentiment.
    """
    prompt_lines = [
        "You are a stock analyst assistant. Analyze the provided company data in the context of the user's query and the stated overall market sentiment."
    ]
    prompt_lines.append(f"\nUser Query: \"{user_query}\"\n")
    prompt_lines.append(f"Overall Market Sentiment: {overall_market_sentiment_string}\n")

    prompt_lines.append("Company Data Context:")
    if top_companies_df.empty:
        prompt_lines.append("No specific company data is available from the latest pipeline run.")
    else:
        for _, row in top_companies_df.iterrows():
            macro_sentiment_val = row.get('macro_sentiment_score')
            macro_sentiment_str = f"{macro_sentiment_val:.2f}" if pd.notna(macro_sentiment_val) else "N/A"


            line = (
                f"- {row['company']}: Positive Sentiment={format_value_for_prompt(row.get('positive'))}, Neutral Sentiment={format_value_for_prompt(row.get('neutral'))}, "
                f"Negative Sentiment={format_value_for_prompt(row.get('negative'))}, GrowthScore={format_value_for_prompt(row.get('growth_score'))}, "
                f"MacroSentiment={macro_sentiment_str}, Sector={row['theme']}" # macro_sentiment_str and theme are usually fine

            )
            # Check if 'predicted_close_price' column exists and the value is not NaN or a placeholder string
            # The format_value_for_prompt could be used here too if predicted_close_price could also be "N/A (No Batch Data)"
            # However, current logic sets it to pd.NA for minimal entries if LSTM fails, which is handled by the existing pd.notna check.
            # If LSTM on-demand consistently provides a float or pd.NA, existing check is fine.
            # Let's assume predicted_close_price is either float or pd.NA from previous steps.
            if 'predicted_close_price' in row and pd.notna(row['predicted_close_price']):
                company_info += f"\n  - 🚀 LSTM Next Day Close Prediction: ${row['predicted_close_price']:.2f}"
                # Add some interpretation of the prediction
                if 'current_price' in row and pd.notna(row['current_price']):
                    current_price = row['current_price']
                    predicted_price = row['predicted_close_price']
                    pct_change = ((predicted_price - current_price) / current_price) * 100
                    direction = "↑" if pct_change >= 0 else "↓"
                    company_info += f" ({direction}{abs(pct_change):.2f}%)"
            else:
                company_info += "\n  - LSTM Prediction: Not available"
            prompt_lines.append(company_info)

    prompt_lines.append("\n---") # Separator before instructions


instructions = (
    "Your primary role is to answer the user's query by providing a detailed stock and sector analysis for 'tomorrow'. "
    "To do this, you MUST synthesize insights from four key pillars of information: the 'Overall Market Sentiment', specific 'MacroSentiment' (sector sentiment), individual company news sentiment scores, and 'LSTM Next Day Close' price predictions, all found within the 'Company Data Context' and 'Overall Market Sentiment' sections provided below. "
    "Use ONLY this provided information. Adhere strictly to these guidelines:\n\n"
    "**1. Sector Analysis (Tomorrow's Outlook):**\n"
    "   - Begin by identifying 2-3 sectors from the 'Company Data Context' that exhibit the most significant positive or "
    "     negative 'MacroSentiment score'.\n"
    "   - For each identified sector, explain what its 'MacroSentiment score' might imply for its potential performance "
    "     tomorrow. Relate this to the 'Overall Market Sentiment'.\n\n"
    "**2. Specific Stock Analysis (Tomorrow's Outlook):**\n"
    "   - **Crucially, if the 'User Query' mentions specific stocks by name or ticker, YOU MUST provide a direct and individual analysis for EACH of those stocks, provided their data is available in the 'Company Data Context'.**\n"
    "   - After addressing any directly queried stocks, you can then discuss other companies, prioritizing those within the sectors you've just "
    "     analyzed or others that seem particularly relevant based on their data.\n"
    "   - For EACH stock you analyze (whether directly queried or chosen by you), provide a brief outlook for tomorrow. Your justification MUST holistically synthesize all relevant data points provided for that stock:\n"
    "       a. The stock's individual sentiment scores (Positive, Neutral, Negative news counts).\n"
    "       b. Its 'GrowthScore'.\n"
    "       c. Its 'LSTM Next Day Close' price prediction (if available).\n"
    "       d. The 'MacroSentiment score' of its sector.\n"
    "       e. The context of the 'Overall Market Sentiment'.\n"
    "   - Explicitly discuss how these factors (a-e) interact. For instance, note if they are aligned (e.g., positive stock sentiment, strong sector sentiment, bullish LSTM prediction, within a positive overall market) or if they present a mixed picture (e.g., good individual stock sentiment but a bearish LSTM prediction or weak sector sentiment).\n"
    "   - **If some data points for a queried stock are neutral, minimal, absent (e.g., news sentiment counts are all zero, GrowthScore is close to zero, or key metrics like 'News Sentiment' or 'GrowthScore' are explicitly stated as 'N/A (No Batch Data)' or similar 'N/A' indicators), you MUST explicitly acknowledge this lack of specific batch-processed data. Then, proceed to make your best assessment for that stock based on the remaining available data (e.g., its LSTM prediction, its sector's MacroSentiment, and the Overall Market Sentiment). Do NOT avoid analyzing a queried stock simply because some of its individual metrics were not available from the batch news pipeline or are otherwise weak/neutral.**\n"
    "   - Explain *why* the combination of available factors leads to your outlook for that stock.\n\n"
    "**3. Addressing the User Query:**\n"
    "   - Ensure your entire analysis is framed to comprehensively answer the 'User Query'.\n"
    "   - **Reiterate: If the query asks about specific stocks, your primary goal is to provide a detailed analysis for THOSE stocks using the methodology outlined in section 2.**\n"
    "   - If the query is general (e.g., 'market outlook'), the structured sector and stock analysis (focusing on high-signal companies) serves as your response.\n\n"
    "**4. General Guidelines:**\n"
    "   - Base all predictions and analyses exclusively on the provided 'Company Data Context' and 'Overall Market Sentiment'.\n"
    "   - Do NOT use any external knowledge or real-time market data.\n"
    "   - When making an inference, clearly state it's derived from the provided dataset.\n"
    "   - **For any stock mentioned in the user query but NOT found in the 'Company Data Context', you MUST explicitly state that its specific data is not available in the current analysis dataset.** Do not attempt to infer its performance indirectly through other stocks unless clearly stating it's a broad sector observation.\n"
    "   - If the 'Company Data Context' is empty, inform the user that no specific company data is available to perform the requested analysis."
)

    prompt_lines.append("\nInstructions:\n" + instructions)
    return "\n".join(prompt_lines)

def get_openai_response(prompt_text):
    """
    Calls the OpenAI API with the given prompt and returns the response.
    """
    logger.info("\n📡 Querying OpenAI...\n")
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a stock analyst assistant."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            max_tokens=600
        )
        result_text = response.choices[0].message.content
        logger.info(f"🔮 OpenAI Response:\n{result_text}")
        return result_text
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
        # Depending on desired behavior, could re-raise, return None, or a specific error message
        raise  # Re-raise for the main block to handle for now

def prepare_llm_context_data(user_query, top_n=25): # user_query parameter added
    """
    Loads, processes, and filters data to prepare the context for the LLM.
    Ensures companies mentioned in user_query are included.
    Returns a tuple: (DataFrame of top N companies + queried companies, qualitative market sentiment string).
    """
    logger.info(f"Preparing LLM context for query: '{user_query}' with top_n={top_n}") # Updated log
    # Construct paths relative to this script's location (scripts/)
    # Construct paths relative to this script's location (scripts/)
    # Data files are expected to be in ../data/
    current_dir = os.path.dirname(__file__)
    data_files = {
        "company_df": os.path.join(current_dir, "..", "data", "company_sentiment_normalized.csv"),
        "growth_df": os.path.join(current_dir, "..", "data", "predict_growth.csv"),
        "macro_df": os.path.join(current_dir, "..", "data", "macro_sentiment.csv"),
        "mapping_df": os.path.join(current_dir, "..", "data", "nasdaq_top_companies.csv"),
    }
    lstm_predictions_df_path = os.path.join(current_dir, '..', 'data', 'lstm_daily_predictions.csv')
    dfs = {}

    try:
        for df_name, path in data_files.items():
            logger.debug(f"Loading {path}...")
            dfs[df_name] = pd.read_csv(path)

        try:
            logger.debug(f"Loading {lstm_predictions_df_path}...")
            dfs["lstm_df"] = pd.read_csv(lstm_predictions_df_path)
        except FileNotFoundError:
            logger.warning(f"LSTM predictions file not found at {lstm_predictions_df_path}. Creating empty DataFrame.")
            dfs["lstm_df"] = pd.DataFrame(columns=['stock_ticker', 'predicted_close_price'])

        logger.info("All data files loaded successfully (or handled if missing).")
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}. Please ensure all prerequisite scripts have run.")
        raise  # Re-raise to be caught by the caller or main script
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}", exc_info=True)
        raise

    company_df = dfs["company_df"]
    growth_df = dfs["growth_df"]
    macro_df = dfs["macro_df"]
    mapping_df = dfs["mapping_df"]

    # Standardize non-breaking spaces in 'Company' names in mapping_df
    if 'Company' in mapping_df.columns:
        mapping_df['Company'] = mapping_df['Company'].str.replace(' ', ' ', regex=False)
        logger.info("Replaced non-breaking spaces in mapping_df['Company'] names.")
    else:
        logger.warning("'Company' column not found in mapping_df. Skipping non-breaking space replacement.")


    # === Clean column names ===
    logger.debug("Cleaning column names in mapping_df.")
    mapping_df.columns = [col.strip().replace("\xa0", " ") for col in mapping_df.columns] # More robust cleaning

    # === Macro mapping ===
    logger.debug("Creating theme and macro maps.")
    theme_map = dict(zip(mapping_df["Company"], mapping_df["GICS Sector"]))
    macro_map = dict(zip(macro_df["theme"], macro_df["macro_sentiment_score"]))

    # === Merge sentiment and growth ===
    logger.debug("Merging company sentiment and growth data.")
    df = pd.merge(company_df, growth_df, on="company", how="inner")

    # Add ticker symbol to df from mapping_df to be used as a key for LSTM predictions
    # Assuming the ticker column in nasdaq_top_companies.csv is 'Ticker'
    # Clean 'Company' names in mapping_df and 'company' in df for robust merging
    mapping_df['Company_clean'] = mapping_df['Company'].str.strip() # Assumes 'Company' exists
    df['company_clean'] = df['company'].str.strip()

    # Attempt to merge using 'Ticker' as the symbol column name from mapping_df
    # This assumes 'Ticker' is the correct column name in the CSV after cleaning.
    if 'Ticker' in mapping_df.columns and 'Company_clean' in mapping_df.columns:
        df = pd.merge(df, mapping_df[['Company_clean', 'Ticker']], left_on='company_clean', right_on='Company_clean', how='left')
    else:
        logger.warning("Could not find 'Ticker' or 'Company_clean' in mapping_df. Skipping addition of ticker symbols.")
        df['Ticker'] = pd.NA # Ensure 'Ticker' column exists for consistency, even if empty

    # Clean up columns added for merging
    df = df.drop(columns=['company_clean'])
    if 'Company_clean' in df.columns:
        df = df.drop(columns=['Company_clean'])

    # Merge LSTM predictions using the 'Ticker' column
    lstm_df = dfs["lstm_df"]
    if not lstm_df.empty and 'Ticker' in df.columns:
        # Clean and standardize ticker symbols
        lstm_df['ticker_upper'] = lstm_df['stock_ticker'].str.strip().str.upper()
        df['ticker_upper'] = df['Ticker'].str.strip().str.upper()
        
        # Merge on the cleaned ticker symbols
        df = pd.merge(
            df, 
            lstm_df[['ticker_upper', 'predicted_close_price']], 
            left_on='ticker_upper', 
            right_on='ticker_upper', 
            how='left'
        )
        
        # Clean up temporary columns
        if 'ticker_upper' in df.columns:
            df = df.drop(columns=['ticker_upper'])
    else:
        # Ensure the column exists even if no LSTM data or no Ticker to merge on
        df['predicted_close_price'] = pd.NA

    df["theme"] = df["company"].map(theme_map)
    df["macro_sentiment_score"] = df["theme"].map(macro_map).fillna(pd.NA) # Changed from .fillna(0)

    # === Adjust growth score ===
    logger.debug("Calculating adjusted growth score.")
    # Ensure relevant columns are numeric before arithmetic operations
    df["macro_sentiment_score"] = pd.to_numeric(df["macro_sentiment_score"], errors='coerce').fillna(0)
    df["growth_score"] = pd.to_numeric(df["growth_score"], errors='coerce').fillna(0)
    df["adjusted_growth_score"] = df["growth_score"] + df["macro_sentiment_score"] * 5

    # === Filter main df to include only companies present in the NASDAQ list ===
    # This ensures all companies in 'df' are known entities from our primary reference (mapping_df).
    logger.info("Filtering main DataFrame to include only companies from NASDAQ list.")
    nasdaq_official_companies_set = set(mapping_df["Company"].str.strip())
    df = df[df['company'].isin(nasdaq_official_companies_set)].copy() # Use .copy() to avoid SettingWithCopyWarning
    logger.info(f"DataFrame reduced to {len(df)} rows after filtering by NASDAQ list.")

    def identify_companies_from_query(query, mapping_df):
        """
        Identifies company names from the user query by matching against the NASDAQ mapping.
        Returns a list of normalized company names found in the query.
        """
        logger.info(f"Identifying companies from user query: '{query}'")
        query_upper = query.upper()
        found_companies = set()
        
        # Create a clean copy of the mapping dataframe
        mapping_df = mapping_df.copy()
        mapping_df['Company'] = mapping_df['Company'].str.strip()
        mapping_df['Ticker'] = mapping_df['Ticker'].fillna('').astype(str).str.strip()
        
        # Enhanced mapping of variations to official names with more Google/Alphabet variations
        company_variations = {
            'GOOGLE': 'Alphabet Inc. (Class A)',
            'GOOG': 'Alphabet Inc. (Class A)',
            'ALPHABET': 'Alphabet Inc. (Class A)',
            'ALPHABET INC': 'Alphabet Inc. (Class A)',
            'ALPHABET INC.': 'Alphabet Inc. (Class A)',
            'GOOGLE PARENT': 'Alphabet Inc. (Class A)',
            'GOOGLE STOCK': 'Alphabet Inc. (Class A)',
            'MSFT': 'Microsoft',
            'MICROSOFT': 'Microsoft',
            'AMZN': 'Amazon',
            'AMAZON': 'Amazon',
            'AAPL': 'Apple',
            'APPLE': 'Apple',
            'TSLA': 'Tesla',
            'TESLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'NVIDIA': 'NVIDIA',
            'META': 'Meta Platforms',
            'FACEBOOK': 'Meta Platforms',
            'META PLATFORMS': 'Meta Platforms'
        }
        
        # Special handling for Google/Alphabet variations
        google_related_terms = ['GOOGLE', 'ALPHABET', 'GOOG']
        if any(term in query_upper for term in google_related_terms):
            # Check if we have Alphabet in our mapping
            alphabet_matches = mapping_df[mapping_df['Company'].str.contains('Alphabet', case=False, na=False)]
            if not alphabet_matches.empty:
                official_name = alphabet_matches['Company'].iloc[0]
                found_companies.add(official_name)
        
        # Check for common variations first
        for variation, official_name in company_variations.items():
            if variation in query_upper and official_name in mapping_df['Company'].values:
                found_companies.add(official_name)
        
        # Check for exact matches in the query (case-insensitive)
        for company in mapping_df['Company'].unique():
            company_clean = company.strip()
            if company_clean.upper() in query_upper:
                found_companies.add(company_clean)
        
        # Check for ticker symbols
        for _, row in mapping_df.iterrows():
            ticker = row['Ticker']
            if ticker and ticker.upper() in query_upper:
                found_companies.add(row['Company'].strip())
        
        # Also check for partial matches in company names
        for company in mapping_df['Company'].unique():
            # Split company name into words and check if any word is in the query
            words = [w.upper() for w in company.split() if len(w) > 2]  # Ignore short words
            if words and any(word in query_upper for word in words):
                found_companies.add(company)
        
        # If still no matches, try to find by partial match in the query
        if not found_companies:
            for company in mapping_df['Company'].unique():
                company_words = set(word.upper() for word in company.split() if len(word) > 2)
                query_words = set(query_upper.split())
                if company_words.intersection(query_words):
                    found_companies.add(company)
        
        logger.info(f"Normalized query terms to official company names: {found_companies}")
        return list(found_companies)

    def robust_clean_name_for_matching(name):
        """
        Robustly clean and standardize a company name for matching purposes.
        Handles None values, non-string inputs, and performs comprehensive cleaning.
        """
        if not name or not isinstance(name, str):
            return ""
            
        # Basic cleaning
        cleaned = name.strip().lower()
        
        # Remove common suffixes and words that might cause mismatches
        remove_phrases = [
            'inc', 'llc', 'ltd', 'corp', 'corporation', 'holdings', 'technologies', 
            'incorporated', 'company', 'plc', 'group', 'co', '& co', 'holding', 'holdings',
            'class a', 'class b', 'class c', 'class d', 'class 1', 'class 2',
            '(', ')', '[', ']', '{', '}', '&', ',', '.', ';', ':', "'", '\"',
            'the ', ' and ', ' or '
        ]
        
        for phrase in remove_phrases:
            cleaned = cleaned.replace(phrase, ' ')
        
        # Replace multiple spaces with single space
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def get_official_name(ticker, default_name):
        """
        Helper function to get the official company name from ticker.
        If the ticker is found in the mapping, returns the official name; otherwise returns the default.
        """
        try:
            if not isinstance(mapping_df, pd.DataFrame) or 'Ticker' not in mapping_df.columns:
                return default_name
                
            # Try to find the ticker in the mapping dataframe
            ticker_upper = str(ticker).strip().upper()
            ticker_match = mapping_df[mapping_df['Ticker'].apply(
                lambda x: str(x).strip().upper() == ticker_upper if pd.notna(x) else False
            )]
            
            if not ticker_match.empty and 'Company' in ticker_match.columns:
                return ticker_match['Company'].iloc[0]
            return default_name
        except Exception as e:
            logger.warning(f"Error in get_official_name for ticker {ticker}: {str(e)}")
            return default_name

    # Populate query_to_official_map with common names -> official names from mapping_df
    common_name_mappings_tuples = [
        ("google", "GOOGL", "Alphabet Inc. (Class A)"), # common, ticker, default official
        ("alphabet", "GOOGL", "Alphabet Inc. (Class A)"),
        ("apple", "AAPL", "Apple Inc."),
        ("microsoft", "MSFT", "Microsoft Corporation"), # Default name if MSFT ticker not found
        ("amazon", "AMZN", "Amazon.com, Inc."),
        ("nvidia", "NVDA", "NVIDIA Corporation"),
        ("meta", "META", "Meta Platforms, Inc."),
        ("facebook", "META", "Meta Platforms, Inc."),
        ("tesla", "TSLA", "Tesla, Inc.")
    ]

    
    # Initialize the query_to_official_map with common mappings
    query_to_official_map = {}

    logger.info(f"DEBUG: Initial common_name_mappings_tuples for Microsoft: {next(item for item in common_name_mappings_tuples if item[0] == 'microsoft')}")

    for common, ticker_val, default_name in common_name_mappings_tuples:
        # get_official_name now uses mapping_df where 'Company' is already cleaned of non-breaking spaces
        official_map_value = get_official_name(ticker_val, default_name)
        query_to_official_map[common] = official_map_value
        if common == "microsoft":
            logger.info(f"DEBUG: Mapping for 'microsoft' (common name): Key='{common}', Value='{official_map_value}' (from get_official_name using Ticker: {ticker_val}, Default: {default_name})")

    # Add all tickers (lowercase) and cleaned official names (lowercase) from NASDAQ list
    # mapping_df['Company'] has already been cleaned of non-breaking spaces at this point
    for _, row in mapping_df.iterrows():
        official_name = row['Company'] # This is the cleaned name (regular spaces)
        ticker = row['Ticker']

        # Debug logging for Microsoft specifically
        is_microsoft_debug = False
        if pd.notna(ticker) and ticker == 'MSFT':
            is_microsoft_debug = True
            logger.info(f"DEBUG: Processing MSFT row: Official Name='{official_name}', Ticker='{ticker}'")

        if pd.notna(official_name):
            cleaned_official = robust_clean_name_for_matching(official_name) # robust_clean also lowercases
            if cleaned_official and cleaned_official not in query_to_official_map:
                query_to_official_map[cleaned_official] = official_name
                if is_microsoft_debug:
                    logger.info(f"DEBUG: MSFT mapping: Key (cleaned_official)='{cleaned_official}', Value='{official_name}'")

            # Also add the lowercase version of the (already space-cleaned) official name as a key
            # if it's not already there (e.g. covered by common_name or cleaned_official)
            if official_name.lower() not in query_to_official_map:
                 query_to_official_map[official_name.lower()] = official_name
                 if is_microsoft_debug:
                    logger.info(f"DEBUG: MSFT mapping: Key (official_name.lower())='{official_name.lower()}', Value='{official_name}'")

        if pd.notna(ticker):
            if ticker.lower() not in query_to_official_map:
                query_to_official_map[ticker.lower()] = official_name
                if is_microsoft_debug:
                    logger.info(f"DEBUG: MSFT mapping: Key (ticker.lower())='{ticker.lower()}', Value='{official_name}'")

    logger.info(f"DEBUG: Query map contains MSFT ticker key ('msft'): {'msft' in query_to_official_map}, maps to: {query_to_official_map.get('msft')}")
    logger.info(f"DEBUG: Query map contains 'microsoft' common key: {'microsoft' in query_to_official_map}, maps to: {query_to_official_map.get('microsoft')}")
    # Log a few more cleaned keys for Microsoft if they exist
    if 'microsoft corporation' in query_to_official_map: # Example of a cleaned name
        logger.info(f"DEBUG: Query map contains 'microsoft corporation': maps to {query_to_official_map['microsoft corporation']}")
    if 'microsoft corp' in query_to_official_map:
        logger.info(f"DEBUG: Query map contains 'microsoft corp': maps to {query_to_official_map['microsoft corp']}")


    potential_queried_normalized_names = set()
    cleaned_user_query = robust_clean_name_for_matching(user_query) # e.g., "any news on google or microsoft"
    query_parts = cleaned_user_query.split()
    logger.info(f"DEBUG: Cleaned user query: '{cleaned_user_query}', Query parts: {query_parts}")
    max_n = 3
    for n in range(max_n, 0, -1):
        for i in range(len(query_parts) - n + 1):
            ngram = " ".join(query_parts[i:i+n])
            logger.debug(f"DEBUG: Checking ngram: '{ngram}'")
            if ngram in query_to_official_map:
                mapped_name = query_to_official_map[ngram]
                potential_queried_normalized_names.add(mapped_name)
                logger.info(f"DEBUG: Ngram match! Ngram='{ngram}', Mapped Official Name='{mapped_name}'. Added to potential_queried_normalized_names.")
            # else: # Optional: log non-matches for very verbose debugging
                # logger.debug(f"DEBUG: Ngram '{ngram}' not found in query_to_official_map.")


    logger.info(f"Normalized query terms to official company names: {potential_queried_normalized_names if potential_queried_normalized_names else 'None found'}")

    data_for_queried_companies = []
    # Define expected columns based on the fully processed 'df' structure later,
    # or a predefined list if 'df' could be empty at this stage.
    # For now, 'df.columns' will be used once 'df' is fully processed.

    if potential_queried_normalized_names:
        for official_name in potential_queried_normalized_names:
            current_stock_data_series = None
            ticker = None # Initialize ticker for this scope

            # Try to get ticker first, as it's needed for LSTM and potentially for minimal data historical check
            ticker_series = mapping_df.loc[mapping_df['Company'] == official_name, 'Ticker']
            if not ticker_series.empty and pd.notna(ticker_series.iloc[0]):
                ticker = ticker_series.iloc[0]
            else:
                logger.warning(f"No ticker found for '{official_name}' in NASDAQ list. Cannot perform on-demand LSTM or ensure historical data presence for minimal entry.")
                # Decide if we should still attempt to create a minimal entry without a ticker, or skip.
                # For now, we will skip if no ticker, as LSTM is a key part of this integration.
                # If a minimal entry without LSTM was desired, logic could be different.

            lstm_pred_on_demand = None
            if ticker: # Only proceed if ticker is valid
                # Check for historical data for the ticker (and fetch if missing)
                # This re-uses logic from previous subtask (Phase 1 historical data fetch)
                historical_data_path = os.path.join("data/historical_prices/", f"{ticker.upper()}.csv")
                if not os.path.exists(historical_data_path):
                    logger.info(f"Historical data file not found for {ticker} at {historical_data_path}. Attempting to fetch on-demand (Phase 1 type fetch).")
                    try:
                        # This is the historical prices fetch, not LSTM model training
                        # from scripts.fetch_historical_prices import fetch_single_stock_data <- ensure this is imported
                        fetched_hist_path = fetch_single_stock_data(ticker)
                        if fetched_hist_path:
                            logger.info(f"Successfully fetched historical price data for {ticker} to {fetched_hist_path}.")
                        else:
                            logger.warning(f"Failed to fetch historical price data for {ticker} on-demand.")
                    except Exception as e:
                        logger.error(f"Error during on-demand historical price data fetch for {ticker}: {e}", exc_info=True)

                # Now, attempt on-demand LSTM prediction/training
                try:
                    logger.info(f"Attempting on-demand LSTM prediction/training for {ticker}...")
                    # from scripts.lstm_stock_predictor import get_or_train_lstm_for_stock <- ensure this is imported
                    lstm_pred_on_demand = get_or_train_lstm_for_stock(ticker, training_epochs=10) # Using 10 epochs for on-demand
                    if lstm_pred_on_demand is not None:
                        logger.info(f"On-demand LSTM prediction for {ticker}: {lstm_pred_on_demand:.2f}")
                    else:
                        logger.warning(f"On-demand LSTM prediction/training for {ticker} returned None.")
                except Exception as e:
                    logger.error(f"Error calling get_or_train_lstm_for_stock for {ticker}: {e}", exc_info=True)

            # Check if company data is in the main batch-processed df
            company_data_from_main_df_row = df[df['company'] == official_name]

            if not company_data_from_main_df_row.empty:
                current_stock_data_series = company_data_from_main_df_row.iloc[0].copy()
                if lstm_pred_on_demand is not None:
                    current_stock_data_series['predicted_close_price'] = lstm_pred_on_demand
                    logger.info(f"Updated LSTM prediction for batch-processed '{official_name}' with on-demand value: {lstm_pred_on_demand:.2f}")
                else:
                    logger.info(f"Using batch LSTM prediction (if any) for '{official_name}'. On-demand LSTM did not yield a prediction.")
            elif ticker and os.path.exists(os.path.join("data/historical_prices/", f"{ticker.upper()}.csv")):
                # Not in batch df, but historical data exists (either pre-existing or fetched in this process)
                # Create a minimal entry
                logger.info(f"Creating minimal data entry for '{official_name}' (Ticker: {ticker}) as it was not in batch data.")
                sector_series = mapping_df.loc[mapping_df['Company'] == official_name, 'GICS Sector']
                sector = sector_series.iloc[0] if not sector_series.empty and pd.notna(sector_series.iloc[0]) else "N/A"
                current_macro_score = macro_map.get(sector, 0.0)
                if pd.isna(current_macro_score): current_macro_score = 0.0

                na_batch_str = "N/A (No Batch Data)" # Define the placeholder string

                minimal_data_dict = {col: pd.NA for col in df.columns} # Initialize with NA for all df columns

                # Update with specific values for the minimal entry
                minimal_data_dict.update({
                    'company': official_name,
                    'Ticker': ticker,
                    # Columns to be set to na_batch_str
                    'positive': na_batch_str,
                    'neutral': na_batch_str,
                    'negative': na_batch_str,
                    'sum_sentiment': na_batch_str, # Assuming this comes from batch news processing
                    'news_count': na_batch_str,    # Assuming this comes from batch news processing
                    'growth_score': na_batch_str,
                    'adjusted_growth_score': na_batch_str, # Since growth_score is string

                    # Columns with actual or derived values
                    'theme': sector, # Derived from mapping_df
                    'macro_sentiment_score': current_macro_score, # Derived from macro_map

                    # Predicted close price will be from on-demand LSTM or pd.NA
                    'predicted_close_price': lstm_pred_on_demand if lstm_pred_on_demand is not None else pd.NA,

                    # Other columns from df structure will remain pd.NA unless explicitly set
                    # e.g., 'date' from company_sentiment_normalized.csv would be pd.NA
                })

                # Ensure only columns that exist in the main 'df' are included, maintaining original NA for unspecified ones
                current_stock_data_series = pd.Series(minimal_data_dict).reindex(df.columns)
            else:
                logger.warning(f"Skipping '{official_name}': Not found in batch data and no historical data file present (even after attempting fetch) or no ticker.")

            if current_stock_data_series is not None:
                data_for_queried_companies.append(current_stock_data_series)

    if data_for_queried_companies:
        queried_companies_data_df = pd.DataFrame(data_for_queried_companies).reindex(columns=df.columns)
        # Ensure 'predicted_close_price' is float after potential pd.NA or numeric values
        if 'predicted_close_price' in queried_companies_data_df.columns:
             queried_companies_data_df['predicted_close_price'] = pd.to_numeric(queried_companies_data_df['predicted_close_price'], errors='coerce')
        # Ensure column order and presence matches 'df' - crucial for concat
        # If df might be empty here (e.g. all data files failed to load), this needs a fallback.
        # However, df is formed much earlier, so it should have its columns defined.
        queried_companies_data_df = queried_companies_data_df.reindex(columns=df.columns, fill_value=pd.NA)
    else:
        # Create an empty DataFrame with columns from 'df' if no queried companies processed
        queried_companies_data_df = pd.DataFrame(columns=df.columns)

    logger.info(f"Processed {len(data_for_queried_companies)} queried companies. DataFrame shape: {queried_companies_data_df.shape}")

    # === Select top N companies by score ===
    # These are companies not necessarily in the query, selected by performance.
    # Filter out companies with null or zero growth scores for this selection.
    df_for_top_n_selection = df[df["growth_score"].notnull() & (df["growth_score"] != 0)].copy()
    top_n_by_score_df = df_for_top_n_selection.sort_values(by="adjusted_growth_score", ascending=False).head(top_n)
    logger.info(f"Selected {len(top_n_by_score_df)} additional top companies by adjusted_growth_score.")

    # Ensure queried companies are included even if they weren't in the top N
    if not queried_companies_data_df.empty:
        # Get the official names from the mapping for queried companies
        queried_official_names = set(queried_companies_data_df['company'])
        
        # Find any queried companies that might be missing from the top N
        missing_queried = []
        for company in queried_official_names:
            if company not in top_n_by_score_df['company'].values:
                missing_queried.append(company)
        
        # If we have missing queried companies, add them to the context
        if missing_queried:
            logger.info(f"Adding missing queried companies to final context: {missing_queried}")
            # Get the company data for missing queried companies
            missing_data = company_df[company_df['company'].isin(missing_queried)].copy()
            
            # Add LSTM predictions if available
            if 'Ticker' in missing_data.columns and not lstm_df.empty:
                missing_data = pd.merge(
                    missing_data,
                    lstm_df[['stock_ticker', 'predicted_close_price']],
                    left_on='Ticker',
                    right_on='stock_ticker',
                    how='left'
                ).drop(columns=['stock_ticker'])
            
            # Add sector information
            missing_data['theme'] = missing_data['company'].map(theme_map)
            missing_data['macro_sentiment_score'] = missing_data['theme'].map(macro_map).fillna(0)
            
            # Add to queried companies data
            queried_companies_data_df = pd.concat([queried_companies_data_df, missing_data])
    
    # Combine batch top companies and queried companies data
    final_context_df = pd.concat([top_n_by_score_df, queried_companies_data_df])
    
    # Remove duplicates, keeping the first occurrence (prioritize queried companies data)
    final_context_df = final_context_df.drop_duplicates(subset=['company'], keep='first').reset_index(drop=True)
    
    logger.info(f"Final LLM context will include {len(final_context_df)} unique companies (Top N by score + Queried).")
    
    # Log the final list of companies being returned
    logger.info(f"Companies in final context: {final_context_df['company'].tolist()}")
    
    if final_context_df.empty:
        logger.warning("LLM context is empty: No top companies met criteria and no queried companies identified/found in data.")
    else:
        # Sort the final list by adjusted_growth_score for consistent presentation in the prompt, if desired
        final_context_df = final_context_df.sort_values(by="adjusted_growth_score", ascending=False).reset_index(drop=True)
        logger.debug(f"Final companies for LLM context (sorted by score):\n{final_context_df.to_string()}")

    market_sentiment_str = get_qualitative_market_sentiment()
    logger.info(f"Overall market sentiment for LLM context: {market_sentiment_str}")

    return final_context_df, market_sentiment_str

# === Main script execution starts here ===
if __name__ == "__main__":
    try:
        # Sample user query for testing
        sample_user_query = "What's the market outlook for tomorrow? Highlight key sectors and specific stocks to watch, considering all available data. Any news on Google or Microsoft?"
        # sample_user_query = "Any news on semiconductor stocks like Nvidia?"
        # sample_user_query = "Tell me about TSLA and AAPL."

        logger.info(f"Using sample user query for main execution: \"{sample_user_query}\"")

        # Pass user_query to prepare_llm_context_data
        context_df, market_sentiment_str = prepare_llm_context_data(user_query=sample_user_query, top_n=25)

        if context_df.empty:
            logger.warning("Pipeline did not identify any top companies or queried companies based on current data. LLM will have limited context.")
            # Continue with an empty DataFrame, generate_dynamic_prompt will handle it.

        logger.info(f"Main test: Overall market sentiment: {market_sentiment_str}")

        logger.info(f"Main test: Overall market sentiment: {market_sentiment_str}")

        # Generate the dynamic prompt using the combined context_df
        dynamic_prompt = generate_dynamic_prompt(sample_user_query, context_df, market_sentiment_str)
        logger.info("Generated Dynamic Prompt:")
        logger.info(dynamic_prompt)

        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not found. Skipping OpenAI API call and output file generation.")
            sys.exit(0) # Exit gracefully after printing prompt

        # Get OpenAI response
        llm_response = get_openai_response(dynamic_prompt)

        # Log and print the response (already logged in get_openai_response)
        print("\nLLM Full Response:")
        print(llm_response)

        # Save result (optional for this testing phase, but kept for consistency)
        output_dir = "output"
        output_file = os.path.join(output_dir, "gpt_prediction_dynamic.txt") # New file for dynamic
        logger.info(f"Saving LLM response to {output_file}...")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(llm_response)
        logger.info(f"💾 LLM response saved to → {output_file}")

    except FileNotFoundError as e:
        logger.error(f"A required data file was not found during context preparation: {e}. Cannot proceed with LLM query.")
        sys.exit(1)
    except Exception as e: # Catch exceptions from get_openai_response or other issues
        logger.error(f"An unexpected error occurred in the main script: {e}", exc_info=True)
        sys.exit(1)

    logger.info("✅ Done!")
