import pandas as pd
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs('../logs', exist_ok=True)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('../logs/stock_predictor.log', mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

# Check for OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in .env file or environment variables.")
    sys.exit(1)

def generate_dynamic_prompt(user_query, top_companies_df):
    """
    Generates a dynamic prompt for the LLM based on user query and company data.
    """
    prompt_lines = [
        "You are a stock analyst assistant. Analyze the provided company data in the context of the user's query."
    ]
    prompt_lines.append(f"\nUser Query: \"{user_query}\"\n")

    if top_companies_df.empty:
        prompt_lines.append("No specific company data is available from the latest pipeline run. Please answer the query based on general knowledge if appropriate.")
    else:
        prompt_lines.append("Here is the relevant company data from the last day:")
        for _, row in top_companies_df.iterrows():
            line = (
                f"- {row['company']}: Positive Sentiment={row['positive']}, Neutral Sentiment={row['neutral']}, "
                f"Negative Sentiment={row['negative']}, GrowthScore={row['growth_score']:.2f}, "
                f"MacroSentiment={row['macro_sentiment_score']:.2f}, Sector={row['theme']}"
            )
            prompt_lines.append(line)

    prompt_lines.append(
        "\nInstructions: Based on the user's query and the provided company data, "
        "provide a concise analysis and answer the query. "
        "If the query is general, identify top companies and justify your choices using the data. "
        "If the query is about specific companies, focus on them if they are in the provided data, or state if they are not."
    )
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

def prepare_llm_context_data(top_n=25):
    """
    Loads, processes, and filters data to prepare the context for the LLM.
    Returns a DataFrame of top N companies.
    """
    logger.info("Loading data files for LLM context preparation...")
    data_files = {
        "company_df": "data/company_sentiment_normalized.csv",
        "growth_df": "data/predict_growth.csv",
        "macro_df": "data/macro_sentiment.csv",
        "mapping_df": "data/nasdaq_top_companies.csv"
    }
    dfs = {}

    try:
        for df_name, path in data_files.items():
            logger.debug(f"Loading {path}...")
            dfs[df_name] = pd.read_csv(path)
        logger.info("All data files loaded successfully.")
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
    mapping_df.columns = mapping_df.columns.str.replace("\xa0", " ")

    # === Macro mapping ===
    logger.debug("Creating theme and macro maps.")
    theme_map = dict(zip(mapping_df["Company"], mapping_df["GICS Sector"]))
    macro_map = dict(zip(macro_df["theme"], macro_df["macro_sentiment_score"]))

    # === Merge sentiment and growth ===
    logger.debug("Merging company sentiment and growth data.")
    df = pd.merge(company_df, growth_df, on="company", how="inner")
    df["theme"] = df["company"].map(theme_map)
    df["macro_sentiment_score"] = df["theme"].map(macro_map).fillna(0)

    # === Adjust growth score ===
    logger.debug("Calculating adjusted growth score.")
    df["adjusted_growth_score"] = df["growth_score"].fillna(0) + df["macro_sentiment_score"] * 5

    # === Filter only top NASDAQ companies & remove NaN/zero scores ===
    logger.info("Filtering and selecting top companies...")
    nasdaq_companies = set(mapping_df["Company"].str.strip())
    df = df[df["company"].isin(nasdaq_companies)]
    df = df[df["growth_score"].notnull() & (df["growth_score"] != 0)]

    # === Select top companies ===
    top_companies_df = df.sort_values(by="adjusted_growth_score", ascending=False).head(top_n)

    if top_companies_df.empty:
        logger.warning("No companies found after filtering and ranking. LLM context will be empty.")
    else:
        logger.info(f"Selected {len(top_companies_df)} top companies for LLM context.")
        logger.debug(f"Top companies details for LLM:\n{top_companies_df.to_string()}")

    return top_companies_df

# === Main script execution starts here ===
if __name__ == "__main__":
    try:
        top_companies_df = prepare_llm_context_data(top_n=25)

        if top_companies_df.empty:
            logger.warning("Pipeline did not identify any top companies based on current data. LLM will have limited context.")
            # Continue with an empty DataFrame, generate_dynamic_prompt will handle it.

        # Sample user query for testing
        sample_user_query = "Which stocks are looking good today based on sentiment and growth potential?"
        # sample_user_query = "What's the outlook for Apple and Microsoft?"
        # sample_user_query = "Any news on semiconductor stocks?"
        logger.info(f"Using sample user query: \"{sample_user_query}\"")

        # Generate the dynamic prompt
        dynamic_prompt = generate_dynamic_prompt(sample_user_query, top_companies_df)
        logger.info("Generated Dynamic Prompt:")
        logger.info(dynamic_prompt) # Using logger.info for multiline, could also just print

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
