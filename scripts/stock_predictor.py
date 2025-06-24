import logging
import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import re # For parsing user query

# Add parent directory to sys.path to allow imports from 'utils'
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.append(parent_dir)

from utils.normalization_utils import normalize_company_name, get_ticker_for_company, _load_normalization_data as load_norm_data
from utils.yfinance_utils import get_current_stock_data

print("--- Running stock_predictor.py - VERSION: DEBUG_LOGGING_V2 ---") # Unmissable print statement

# Load environment variables from .env file
# Assuming .env is in the parent directory (project root) or this script's directory
env_path_script_dir = Path(__file__).parent / '.env'
env_path_parent_dir = Path(__file__).parent.parent / '.env'

if os.path.exists(env_path_script_dir):
    load_dotenv(dotenv_path=env_path_script_dir)
    env_path = env_path_script_dir
elif os.path.exists(env_path_parent_dir):
    load_dotenv(dotenv_path=env_path_parent_dir)
    env_path = env_path_parent_dir
else:
    env_path = None


# Configure logging
# Change level to DEBUG to see more detailed logs for diagnosing this issue
logging.basicConfig(
    level=logging.DEBUG, # << TEMP CHANGE TO DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Ensure OPENAI_API_KEY is loaded from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not found.")
    if env_path:
        logger.warning(f"Looked for .env file at: {env_path.absolute()} and {env_path_parent_dir.absolute()}")
    else:
        logger.warning(f"No .env file found at {env_path_script_dir.absolute()} or {env_path_parent_dir.absolute()}")
    logger.warning(f"Current working directory: {os.getcwd()}")


# Helper function for formatting values in the prompt
def format_value_for_prompt(value):
    if isinstance(value, str):
        return value
    if pd.notna(value):
        try:
            return f"{float(value):.2f}"
        except ValueError:
            return str(value)
    return "N/A"

def parse_user_query_for_companies(user_query):
    normalized_full_query = normalize_company_name(user_query)
    if normalized_full_query:
        logger.debug(f"Normalized full user query '{user_query}' to '{normalized_full_query}'")
        return [normalized_full_query]

    potential_terms = re.split(r'[\s,]+(?:and|or)?[\s,]*', user_query, flags=re.IGNORECASE)
    normalized_companies = set()
    for term in potential_terms:
        term = term.strip()
        if not term or term.lower() in ["stock", "stocks", "company", "companies", "sector", "market", "tomorrow", "ktomorrow", "stco"]:
            continue
        normalized_name = normalize_company_name(term)
        if normalized_name:
            logger.debug(f"Normalized term '{term}' from user query to '{normalized_name}'")
            normalized_companies.add(normalized_name)
        else:
            logger.debug(f"Term '{term}' from user query could not be normalized to a known company: '{term}'")
    return list(normalized_companies)


def prepare_llm_context_data(user_query, top_n=25):
    logger.info(f"Starting LLM context preparation for query: '{user_query}'")

    load_norm_data()
    
    queried_companies_normalized = parse_user_query_for_companies(user_query)
    logger.info(f"Normalized companies from query '{user_query}': {queried_companies_normalized}")

    data_dir = os.path.join(parent_dir, "data")
    company_sentiment_path = os.path.join(data_dir, "company_sentiment_normalized.csv")
    predict_growth_path = os.path.join(data_dir, "predict_growth.csv")
    macro_sentiment_path = os.path.join(data_dir, "macro_sentiment.csv")

    final_df = pd.DataFrame()

    try:
        logger.debug(f"Loading company sentiment from: {company_sentiment_path}")
        company_df_raw = pd.read_csv(company_sentiment_path)
        logger.debug(f"Loaded company_df_raw. Shape: {company_df_raw.shape}. Columns: {company_df_raw.columns.tolist()}")
        logger.debug(f"Sample company_df_raw 'company' before upper: \n{company_df_raw['company'].head().apply(type).value_counts()}\n{company_df_raw['company'].head()}")


        logger.debug(f"Loading growth data from: {predict_growth_path}")
        growth_df_raw = pd.read_csv(predict_growth_path)
        logger.debug(f"Loaded growth_df_raw. Shape: {growth_df_raw.shape}. Columns: {growth_df_raw.columns.tolist()}")
        logger.debug(f"Sample growth_df_raw 'company' before upper: \n{growth_df_raw['company'].head().apply(type).value_counts()}\n{growth_df_raw['company'].head()}")


        logger.debug(f"Loading macro data from: {macro_sentiment_path}")
        macro_df = pd.read_csv(macro_sentiment_path)
        logger.debug(f"Loaded macro_df. Shape: {macro_df.shape}. Columns: {macro_df.columns.tolist()}")

        # Store original company names from growth_df for accurate ticker lookup later
        # as get_ticker_for_company expects original casing from nasdaq_top_companies.csv
        growth_df = growth_df_raw.copy() # Work with a copy
        growth_df['original_company_for_ticker_lookup'] = growth_df_raw['company']

        # For merging, use uppercased company names
        company_df = company_df_raw.copy()
        company_df['company_upper_merge_key'] = company_df_raw['company'].astype(str).str.upper()
        growth_df['company_upper_merge_key'] = growth_df_raw['company'].astype(str).str.upper()
        
        logger.debug(f"Unique company keys in company_df for merge (first 5): {company_df['company_upper_merge_key'].unique()[:5]}")
        logger.debug(f"Unique company keys in growth_df for merge (first 5): {growth_df['company_upper_merge_key'].unique()[:5]}")

        # Merge 1: growth_df with company_df (sentiment)
        logger.info("Merging growth data with company sentiment data...")
        merged_df = pd.merge(growth_df, company_df, on="company_upper_merge_key", how="left", suffixes=('_growth', '_sentiment'))
        logger.info(f"Shape after merging with company sentiment: {merged_df.shape}")
        # Use the company name from growth_df as the primary one, as it's likely more official
        merged_df['company'] = merged_df['company_growth']
        merged_df.drop(columns=['company_growth', 'company_sentiment'], errors='ignore', inplace=True)


        # Log some results of the first merge for problematic companies
        example_companies_check = ['ALPHABET INC. (CLASS A)', 'MICROSOFT CORP.', 'AMAZON.COM, INC.', 'ASTRAZENECA PLC', 'ASML HOLDING N.V.'] # Check with official names
        if not merged_df.empty:
            logger.debug(f"Post-Merge 1 (Sentiment) check for example companies (matched on 'company_upper_merge_key'):")
            for ex_co in example_companies_check:
                ex_data = merged_df[merged_df['company_upper_merge_key'] == ex_co.upper()]
                if not ex_data.empty:
                    logger.debug(f"Data for {ex_co.upper()}: Positive={ex_data['positive'].values}, Neutral={ex_data['neutral'].values}, Negative={ex_data['negative'].values}")
                else:
                    logger.debug(f"No data found for {ex_co.upper()} after sentiment merge.")

        # Merge 2: merged_df with macro_df (macro sentiment)
        if 'theme' in merged_df.columns and not macro_df.empty:
            logger.info("Merging with macro sentiment data...")
            logger.debug(f"Unique themes in merged_df (first 5): {merged_df['theme'].unique()[:5]}")
            logger.debug(f"Unique themes in macro_df (first 5): {macro_df['theme'].unique()[:5]}")
            merged_df = pd.merge(merged_df, macro_df[['theme', 'macro_sentiment_score']], on="theme", how="left")
            logger.info(f"Shape after merging with macro sentiment: {merged_df.shape}")
        else:
            logger.warning("'theme' column not in merged_df or macro_df is empty. Skipping macro sentiment merge.")
            merged_df['macro_sentiment_score'] = pd.NA

        if not merged_df.empty:
            logger.debug(f"Post-Merge 2 (Macro) check for example companies:")
            for ex_co in example_companies_check:
                ex_data = merged_df[merged_df['company_upper_merge_key'] == ex_co.upper()]
                if not ex_data.empty:
                    logger.debug(f"Data for {ex_co.upper()}: MacroSectorSentiment={ex_data['macro_sentiment_score'].values}")
                else:
                    logger.debug(f"No data for {ex_co.upper()} after macro merge (or it was filtered out).")


        overall_market_sentiment_value = growth_df_raw['market_sentiment'].iloc[0] if 'market_sentiment' in growth_df_raw.columns and not growth_df_raw.empty else 0
        overall_market_sentiment_string = "Positive" if overall_market_sentiment_value > 0.1 else "Negative" if overall_market_sentiment_value < -0.1 else "Neutral"

        if merged_df.empty or 'growth_score' not in merged_df.columns:
            logger.error("merged_df is empty or 'growth_score' is missing before sorting. Cannot proceed with company selection.")
            return pd.DataFrame(), overall_market_sentiment_string, queried_companies_normalized

        top_companies_df = merged_df.sort_values(by="growth_score", ascending=False)

        # Prepare final_df (selection of companies)
        queried_companies_uppercase_keys = [name.upper() for name in queried_companies_normalized]
        final_df_list = []

        if not queried_companies_uppercase_keys:
            final_df = top_companies_df.head(top_n).copy() # Use .copy() to avoid SettingWithCopyWarning
            logger.debug(f"No specific companies queried. Selected top {top_n}. Shape: {final_df.shape}")
        else:
            logger.debug(f"Processing queried companies: {queried_companies_uppercase_keys}")
            queried_company_data_list = []
            for company_key_upper in queried_companies_uppercase_keys:
                # Match against the merge key
                company_data = top_companies_df[top_companies_df['company_upper_merge_key'] == company_key_upper]
                if not company_data.empty:
                    queried_company_data_list.append(company_data)
                    logger.debug(f"Found data for queried company key {company_key_upper}. Shape: {company_data.shape}")
                else:
                    logger.warning(f"Data for queried company key '{company_key_upper}' not found in top_companies_df.")

            if queried_company_data_list:
                queried_companies_selected_df = pd.concat(queried_company_data_list).drop_duplicates(subset=['company_upper_merge_key'])
                final_df_list.append(queried_companies_selected_df)

            if not final_df_list: # if no queried companies were found in data
                 final_df = top_companies_df.head(top_n).copy()
                 logger.debug(f"No queried companies found in data. Selected top {top_n}. Shape: {final_df.shape}")
            else:
                companies_to_exclude_keys = final_df_list[0]['company_upper_merge_key'].tolist()
                num_to_fetch = top_n - len(companies_to_exclude_keys)
                if num_to_fetch > 0:
                    remaining_top_df = top_companies_df[~top_companies_df['company_upper_merge_key'].isin(companies_to_exclude_keys)].head(num_to_fetch)
                    if not remaining_top_df.empty:
                        final_df_list.append(remaining_top_df)
                final_df = pd.concat(final_df_list).drop_duplicates(subset=['company_upper_merge_key']).reset_index(drop=True).copy()
                logger.debug(f"Combined queried and top companies. Shape: {final_df.shape}")
        
        # --- Integrate yfinance data ---
        if not final_df.empty:
            logger.info(f"Enhancing {len(final_df)} companies with live yfinance data...")
            yfinance_updates = []
            # Ensure original_company_for_ticker_lookup is present for all rows in final_df
            # This might require merging it back if it was lost, or ensuring it's selected carefully
            if 'original_company_for_ticker_lookup' not in final_df.columns:
                 logger.error("'original_company_for_ticker_lookup' is missing from final_df. Ticker lookup will fail.")
                 # Attempt to recover it if 'company' (which was company_growth) is the original name
                 if 'company' in final_df.columns: # This 'company' should be from growth_df
                     logger.warning("Attempting to use 'company' column from final_df for ticker lookup. Assumed to be original casing.")
                     final_df['original_company_for_ticker_lookup'] = final_df['company']
                 else: # Cannot proceed with yfinance if no suitable name for ticker lookup
                      final_df['original_company_for_ticker_lookup'] = pd.NA


            for index, row in final_df.iterrows():
                # Use the original casing company name for ticker lookup
                company_name_for_ticker = row.get('original_company_for_ticker_lookup')

                update_values = {'current_price': None, 'sma_20': None, 'sma_50': None, 'previous_close': None, 'day_high': None, 'day_low': None, 'volume': None, 'market_cap': None}

                if pd.isna(company_name_for_ticker):
                    logger.warning(f"Missing original company name for row index {index}, company_upper_merge_key: {row.get('company_upper_merge_key')}. Cannot fetch yfinance data.")
                    yfinance_updates.append(update_values)
                    continue

                logger.debug(f"Attempting ticker lookup for: '{company_name_for_ticker}' (original name)")
                ticker_symbol = get_ticker_for_company(company_name_for_ticker)

                if ticker_symbol:
                    logger.debug(f"Fetching yfinance data for {company_name_for_ticker} ({ticker_symbol})")
                    live_data = get_current_stock_data(ticker_symbol)
                    if not live_data.get('error'):
                        update_values['current_price'] = live_data.get('current_price')
                        update_values['sma_20'] = live_data.get('sma_20')
                        update_values['sma_50'] = live_data.get('sma_50')
                        update_values['previous_close'] = live_data.get('previous_close')
                        update_values['day_high'] = live_data.get('day_high')
                        update_values['day_low'] = live_data.get('day_low')
                        update_values['volume'] = live_data.get('volume')
                        update_values['market_cap'] = live_data.get('market_cap')
                        logger.debug(f"yfinance data for {ticker_symbol}: Price={update_values['current_price']}, SMA20={update_values['sma_20']}")
                    else:
                        logger.warning(f"Could not fetch yfinance data for {company_name_for_ticker} ({ticker_symbol}): {live_data.get('error')}")
                else:
                    logger.warning(f"No ticker found for company: '{company_name_for_ticker}'. Cannot fetch yfinance data.")
                yfinance_updates.append(update_values)

            # Update DataFrame with yfinance data
            if yfinance_updates: # Check if list is not empty
                for col_name in ['current_price', 'sma_20', 'sma_50', 'previous_close', 'day_high', 'day_low', 'volume', 'market_cap']:
                    final_df[col_name] = [d[col_name] for d in yfinance_updates]
            else:
                logger.warning("yfinance_updates list is empty. No yfinance data was processed to update final_df.")

        else:
            logger.info("No companies selected for yfinance data enhancement (final_df is empty).")

        logger.info(f"Successfully prepared LLM context data. Final DF Shape: {final_df.shape if not final_df.empty else '0 rows'}")
        if not final_df.empty:
            logger.debug(f"Final DF columns: {final_df.columns.tolist()}")
            logger.debug(f"Sample of final_df with yfinance data (first 3 rows): \n{final_df.head(3)}")

        if not final_df.empty and 'predicted_close_price' in final_df.columns:
            final_df.drop(columns=['predicted_close_price'], inplace=True, errors='ignore')

        return final_df, overall_market_sentiment_string, queried_companies_normalized

    except FileNotFoundError as e:
        logger.error(f"Error loading data for LLM context: {e}. Please ensure all prerequisite scripts have run ({e.filename}).")
        return pd.DataFrame(), "Neutral (Data Missing)", queried_companies_normalized # return normalized names even if data is missing
    except Exception as e:
        logger.error(f"An unexpected error occurred while preparing LLM context data: {e}", exc_info=True)
        return pd.DataFrame(), "Neutral (Error)", queried_companies_normalized


def generate_dynamic_prompt(user_query, top_companies_df, overall_market_sentiment_string, normalized_queried_companies):
    """
    Generates a dynamic prompt for the LLM based on user query, company data, and overall market sentiment.
    """
    prompt_lines = [
        "You are a stock analyst assistant. Analyze the provided company data in the context of the user's query and the stated overall market sentiment."
    ]
    prompt_lines.append(f"\nUser Query: \"{user_query}\"")
    if normalized_queried_companies:
        prompt_lines.append(f"Normalized Companies from Query: {', '.join(normalized_queried_companies)}")
    else:
        prompt_lines.append("No specific companies were clearly identified from the query for focused analysis, but consider the query's general intent.")

    prompt_lines.append(f"Overall Market Sentiment: {overall_market_sentiment_string}\n")
    
    instructions = """Your primary role is to answer the user's query by providing a detailed stock and sector analysis for 'tomorrow'. 
    Focus on the following key aspects:
    1. Overall Market Sentiment: Summarize the general market mood based on the provided sentiment data.
    2. Sector Analysis: Highlight which sectors are showing strength or weakness (e.g. based on 'theme' and 'macro_sentiment_score').
    3. Company Analysis: For companies mentioned in the query or showing significant movement (present in the data), provide:
       - Sentiment analysis (positive/negative/neutral from 'positive', 'neutral', 'negative' columns)
       - Growth score and its implications (from 'growth_score' column)
       - Current price (from 'current_price' column)
       - Macro sentiment impact on the sector (from 'macro_sentiment_score' for the company's 'theme')
    4. Actionable Insights: Provide clear, data-driven recommendations based on the analysis.
    
    Do NOT mention LSTM predictions as they are not part of this analysis.
    Always consider the context of the user's query and provide specific, relevant information.
    If data is missing or unclear, acknowledge it and explain any limitations in the analysis."""

    prompt_lines.append("\nCompany Data Context:")
    if top_companies_df.empty:
        prompt_lines.append("No specific company data is available from the latest pipeline run or an error occurred loading it.")
    else:
        for _, row in top_companies_df.iterrows():
            company_name = row.get('company', 'N/A')
            macro_sentiment_val = row.get('macro_sentiment_score')
            macro_sentiment_str = f"{macro_sentiment_val:.2f}" if pd.notna(macro_sentiment_val) else "N/A"
            theme_str = row.get('theme', 'N/A')
            current_price_str = format_value_for_prompt(row.get('current_price'))

            line = (
                f"- {company_name}: GrowthScore={format_value_for_prompt(row.get('growth_score'))}, "
                f"CurrentPrice=${current_price_str}, " # Added CurrentPrice
                f"PositiveSentiment={format_value_for_prompt(row.get('positive'))}, NeutralSentiment={format_value_for_prompt(row.get('neutral'))}, "
                f"NegativeSentiment={format_value_for_prompt(row.get('negative'))}, "
                f"Sector={theme_str}, MacroSectorSentiment={macro_sentiment_str}"
            )
            
            # LSTM Prediction part COMPLETELY REMOVED
            prompt_lines.append(line)

    prompt_lines.append("\n---\nInstructions:\n" + instructions)
    return "\n".join(prompt_lines)


def get_openai_response(prompt_text):
    """
    Calls the OpenAI API with the given prompt and returns the response.
    """
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key is not configured. Cannot query OpenAI.")
        return "Error: OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable."

    logger.info("\n📡 Querying OpenAI (LSTM removed from prompt)...\n")
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a stock analyst assistant. You do not have access to LSTM model predictions."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            max_tokens=700 # Adjusted tokens, as prompt might be slightly shorter
        )
        result_text = response.choices[0].message.content
        logger.info(f"🔮 OpenAI Response received.")
        logger.debug(f"OpenAI Response content:\n{result_text}")
        return result_text
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
        return f"Error communicating with OpenAI: {e}"

# Test execution block removed as it's not essential for the primary functionality.
# The main focus is on the functions used by Streamlit.
# To test, one would typically run the Streamlit app itself after these changes.
# And ensure data files (excluding LSTM related ones) are present.
