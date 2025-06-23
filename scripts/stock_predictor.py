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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
    """
    Parses the user query to extract potential company names or tickers.
    Simple approach: split by common delimiters and try to normalize.
    More complex parsing can be added later if needed.
    """
    normalized_full_query = normalize_company_name(user_query)
    if normalized_full_query:
        logger.info(f"Normalized full user query '{user_query}' to '{normalized_full_query}'")
        return [normalized_full_query]

    potential_terms = re.split(r'[\s,]+(?:and|or)?[\s,]*', user_query, flags=re.IGNORECASE)
    normalized_companies = set()
    for term in potential_terms:
        term = term.strip()
        if not term or term.lower() in ["stock", "stocks", "company", "companies", "sector", "market", "tomorrow", "ktomorrow", "stco"]:
            continue
        normalized_name = normalize_company_name(term)
        if normalized_name:
            logger.info(f"Normalized term '{term}' from user query to '{normalized_name}'")
            normalized_companies.add(normalized_name)
        else:
            logger.info(f"Term '{term}' from user query could not be normalized to a known company.")
    return list(normalized_companies)


def prepare_llm_context_data(user_query, top_n=25):
    """
    Loads, processes, and filters data from various CSVs to prepare the context for the LLM.
    Also fetches live stock data using yfinance for included companies.
    Returns a tuple: (DataFrame of top N companies + queried companies with live yfinance data,
                     qualitative market sentiment string, list of normalized queried companies).
    """
    logger.info(f"Preparing LLM context for query: '{user_query}' with top_n={top_n}")

    load_norm_data() # Ensures normalization maps (including ticker maps) are ready
    
    queried_companies_normalized = parse_user_query_for_companies(user_query)
    if queried_companies_normalized:
        logger.info(f"User query '{user_query}' normalized to: {queried_companies_normalized}")
    else:
        logger.info(f"No specific companies identified or normalized from user query: '{user_query}'")

    data_dir = os.path.join(parent_dir, "data")
    company_sentiment_path = os.path.join(data_dir, "company_sentiment_normalized.csv")
    predict_growth_path = os.path.join(data_dir, "predict_growth.csv")
    macro_sentiment_path = os.path.join(data_dir, "macro_sentiment.csv")

    final_df = pd.DataFrame() # Initialize final_df

    try:
        company_df = pd.read_csv(company_sentiment_path)
        growth_df = pd.read_csv(predict_growth_path)
        macro_df = pd.read_csv(macro_sentiment_path)

        company_df['company'] = company_df['company'].str.upper()
        growth_df['company'] = growth_df['company'].str.upper()
        queried_companies_uppercase = [name.upper() for name in queried_companies_normalized]

        merged_df = pd.merge(growth_df, company_df, on="company", how="left")

        if 'theme' in merged_df.columns and not macro_df.empty:
             merged_df = pd.merge(merged_df, macro_df[['theme', 'macro_sentiment_score']], on="theme", how="left")
        else:
            merged_df['macro_sentiment_score'] = pd.NA

        overall_market_sentiment_value = growth_df['market_sentiment'].iloc[0] if 'market_sentiment' in growth_df.columns and not growth_df.empty else 0
        overall_market_sentiment_string = "Positive" if overall_market_sentiment_value > 0.1 else "Negative" if overall_market_sentiment_value < -0.1 else "Neutral"

        top_companies_df = merged_df.sort_values(by="growth_score", ascending=False)

        final_df_list = []
        if not queried_companies_uppercase:
            final_df = top_companies_df.head(top_n)
        else:
            queried_company_data_list = []
            for company_name_upper in queried_companies_uppercase:
                company_data = merged_df[merged_df['company'] == company_name_upper]
                if not company_data.empty:
                    queried_company_data_list.append(company_data)
                else:
                    logger.warning(f"Data for queried company '{company_name_upper}' not found in merged CSV data.")

            if queried_company_data_list:
                queried_companies_selected_df = pd.concat(queried_company_data_list).drop_duplicates(subset=['company'])
                final_df_list.append(queried_companies_selected_df)

            if not final_df_list:
                 final_df = top_companies_df.head(top_n)
            else:
                companies_to_exclude = final_df_list[0]['company'].tolist()
                remaining_top_df = top_companies_df[~top_companies_df['company'].isin(companies_to_exclude)].head(top_n - len(companies_to_exclude) if top_n > len(companies_to_exclude) else 0)
                if not remaining_top_df.empty:
                    final_df_list.append(remaining_top_df)
                final_df = pd.concat(final_df_list).drop_duplicates(subset=['company']).reset_index(drop=True)
        
        # --- Integrate yfinance data ---
        if not final_df.empty:
            logger.info(f"Enhancing {len(final_df)} companies with live yfinance data...")
            yfinance_updates = []
            for index, row in final_df.iterrows():
                official_company_name = row['company'] # Assuming 'company' column has official name (uppercase)
                ticker_symbol = get_ticker_for_company(official_company_name) # Get ticker from official name

                update_values = {'current_price': None, 'sma_20': None, 'sma_50': None, 'previous_close': None, 'day_high': None, 'day_low': None, 'volume': None, 'market_cap': None}

                if ticker_symbol:
                    logger.debug(f"Fetching yfinance data for {official_company_name} ({ticker_symbol})")
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
                    else:
                        logger.warning(f"Could not fetch yfinance data for {official_company_name} ({ticker_symbol}): {live_data.get('error')}")
                else:
                    logger.warning(f"No ticker found for company: {official_company_name}. Cannot fetch yfinance data.")
                yfinance_updates.append(update_values)

            # Update DataFrame with yfinance data
            for col_name in ['current_price', 'sma_20', 'sma_50', 'previous_close', 'day_high', 'day_low', 'volume', 'market_cap']:
                final_df[col_name] = [d[col_name] for d in yfinance_updates]
        else:
            logger.info("No companies selected for yfinance data enhancement (final_df is empty).")

        logger.info(f"Successfully prepared LLM context data. Shape: {final_df.shape if not final_df.empty else '0 rows'}")

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
