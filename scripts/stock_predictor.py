import pandas as pd
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs('output/logs', exist_ok=True) # Changed path
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('output/logs/stock_predictor.log', mode='w') # Changed path
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
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
                f"- {row['company']}: Positive Sentiment={row['positive']:.2f}, Neutral Sentiment={row['neutral']:.2f}, "
                f"Negative Sentiment={row['negative']:.2f}, GrowthScore={row['growth_score']:.2f}, "

                f"MacroSentiment={macro_sentiment_str}, Sector={row['theme']}"
            )
            # Check if 'predicted_close_price' column exists and the value is not NaN
            if 'predicted_close_price' in row and pd.notna(row['predicted_close_price']):
                line += f", LSTM Next Day Close: ${row['predicted_close_price']:.2f}"
            prompt_lines.append(line)

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
    "   - **If some data points for a queried stock are neutral, minimal, or absent (e.g., news sentiment counts are all zero, or GrowthScore is close to zero), explicitly state this. Then, proceed to make your best assessment for that stock based on the remaining available data (e.g., its LSTM prediction, its sector's MacroSentiment, and the Overall Market Sentiment). Do NOT avoid analyzing a queried stock simply because some of its individual metrics are weak or neutral.**\n"
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
        lstm_df['stock_ticker_clean'] = lstm_df['stock_ticker'].str.strip()
        df = pd.merge(df, lstm_df[['stock_ticker_clean', 'predicted_close_price']], left_on='Ticker', right_on='stock_ticker_clean', how='left')
        if 'stock_ticker_clean' in df.columns:
            df = df.drop(columns=['stock_ticker_clean'])
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

    # === Identify companies from user_query using the NASDAQ list for normalization ===
    logger.info(f"Identifying companies from user query: '{user_query}'")

    def robust_clean_name_for_matching(name):
        if not isinstance(name, str): return ""
        cleaned_name = name.lower()
        cleaned_name = re.sub(r'[^\w\s.-]', '', cleaned_name)
        suffixes = ['inc', 'corp', 'corporation', 'ltd', 'llc', 'co', 'company', 'plc', 'group',
                    'holding', 'holdings', 'technologies', 'solutions', 'services', 'international']
        for suffix in suffixes:
            cleaned_name = re.sub(r'\b' + suffix + r'[.]?\b', '', cleaned_name, flags=re.IGNORECASE)
        return ' '.join(cleaned_name.split()).strip()

    query_to_official_map = {}
    # Helper to get official name from mapping_df, ensures we use the exact name from the CSV
    def get_official_name(ticker_or_common_name, default_if_not_found):
        # Check if it's a ticker
        ticker_match = mapping_df[mapping_df['Ticker'].str.fullmatch(ticker_or_common_name, case=False, na=False)]
        if not ticker_match.empty:
            return ticker_match['Company'].iloc[0]
        # Check if it's a common name that needs to map to an official name (already handled by structure below)
        # This is more for ensuring the default values are correct if a ticker isn't found for some reason.
        return default_if_not_found

    # Populate query_to_official_map with common names -> official names from mapping_df
    common_name_mappings_tuples = [
        ("google", "GOOGL", "Alphabet Inc. (Class A)"), # common, ticker, default official
        ("alphabet", "GOOGL", "Alphabet Inc. (Class A)"),
        ("apple", "AAPL", "Apple Inc."),
        ("microsoft", "MSFT", "Microsoft Corporation"),
        ("amazon", "AMZN", "Amazon.com, Inc."),
        ("nvidia", "NVDA", "NVIDIA Corporation"),
        ("meta", "META", "Meta Platforms, Inc."),
        ("facebook", "META", "Meta Platforms, Inc."),
        ("tesla", "TSLA", "Tesla, Inc.")
    ]
    for common, ticker_val, default_name in common_name_mappings_tuples:
        query_to_official_map[common] = get_official_name(ticker_val, default_name)

    # Add all tickers (lowercase) and cleaned official names (lowercase) from NASDAQ list
    for _, row in mapping_df.iterrows():
        official_name = row['Company']
        ticker = row['Ticker']
        if pd.notna(official_name):
            cleaned_official = robust_clean_name_for_matching(official_name)
            if cleaned_official and cleaned_official not in query_to_official_map:
                query_to_official_map[cleaned_official] = official_name
            if official_name.lower() not in query_to_official_map:
                 query_to_official_map[official_name.lower()] = official_name
        if pd.notna(ticker):
            if ticker.lower() not in query_to_official_map:
                query_to_official_map[ticker.lower()] = official_name # official_name here is from the current row

    potential_queried_normalized_names = set()
    cleaned_user_query = robust_clean_name_for_matching(user_query)
    query_parts = cleaned_user_query.split()
    max_n = 3
    for n in range(max_n, 0, -1):
        for i in range(len(query_parts) - n + 1):
            ngram = " ".join(query_parts[i:i+n])
            if ngram in query_to_official_map:
                potential_queried_normalized_names.add(query_to_official_map[ngram])

    logger.info(f"Normalized query terms to official company names: {potential_queried_normalized_names if potential_queried_normalized_names else 'None found'}")

    queried_companies_data_df = pd.DataFrame()
    if potential_queried_normalized_names:
        # Filter the main 'df' (which is already filtered by NASDAQ list)
        queried_companies_data_df = df[df['company'].isin(list(potential_queried_normalized_names))].copy()
        logger.info(f"Found {len(queried_companies_data_df)} queried companies in the dataset.")
        if not queried_companies_data_df.empty:
            logger.debug(f"Data for queried companies:\n{queried_companies_data_df.to_string()}")

    # === Select top N companies by score ===
    # These are companies not necessarily in the query, selected by performance.
    # Filter out companies with null or zero growth scores for this selection.
    df_for_top_n_selection = df[df["growth_score"].notnull() & (df["growth_score"] != 0)].copy() # Use .copy()
    top_n_by_score_df = df_for_top_n_selection.sort_values(by="adjusted_growth_score", ascending=False).head(top_n)
    logger.info(f"Selected {len(top_n_by_score_df)} additional top companies by adjusted_growth_score.")

    # === Combine top N with queried companies, ensuring no duplicates ===
    if not queried_companies_data_df.empty:
        final_context_df = pd.concat([top_n_by_score_df, queried_companies_data_df]).drop_duplicates(subset=['company']).reset_index(drop=True)
    else:
        final_context_df = top_n_by_score_df.reset_index(drop=True) # If no queried companies, top_n is final

    logger.info(f"Final LLM context will include {len(final_context_df)} unique companies (Top N by score + Queried).")

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
